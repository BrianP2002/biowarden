"""Rules salmon.duplicate_transcripts (family F45, call) and salmon.quant_duplicates (fact).

The trap (family F45). salmon 2.x collapses transcripts with identical sequences when it builds an
index; the dropped transcript has no row in `quant.sf` or `quant.genes.sf` (not even a zero) and
its reads are credited to the retained twin. `--keepDuplicates` keeps both.

Verified here (salmon 2.8.0, 6 genes of 590-930 bases, gene06 an exact copy of gene02, 377 read
pairs, 82 fragments from each of the two): `salmon index -t transcripts.fa -i idx -p 1` exits 0 and
logs at INFO level `removed 1 transcripts that were exact sequence duplicates (use
--keepDuplicates to retain them)`; `idx/duplicate_clusters.tsv` holds `gene02.t1 gene05.t1`;
`salmon quant -i idx -l A -1 .. -2 .. -g genes.gtf` writes quant.genes.sf with five genes, gene02
at 164 reads and no gene06 row. `salmon index ... --keepDuplicates` then gives gene02 and gene06
82.000 each. `quant.sf` records the index in `cmd_info.json` (`"index": "sidx2"`, as given on the
command line).

salmon.duplicate_transcripts (slot completeness, CALL rule on `salmon index`, tier T1,
conditional declaration `duplicate_transcripts`: keep | drop). It reads the FASTA files named by
`-t/--transcripts` (plain or gzip; several files are allowed) and compares the sequences
(upper-cased, whitespace removed): with identical sequences and no `--keepDuplicates` it fails
unless
`duplicate_transcripts=drop` says the collapse is intended; `keep` without the flag fails, `drop`
with the flag fails (the options keep them). No duplicates: pass, whatever the declaration.
Limits: salmon compares the sequences after its own clean-up (non-ACGT bases replaced, poly-A tails
of 10 or more clipped), so two sequences that become equal only after that are not seen, and a FASTA
that is not on disk yet is not judged.

salmon.quant_duplicates (slot completeness, FACT rule, tier T1, conditional declaration
`duplicate_transcripts`). It applies to `quant.sf` and `quant.genes.sf` named in a command, finds
the index from `cmd_info.json` next to the file (relative paths are tried against the file's
folder, its parent and the working directory), reads `duplicate_clusters.tsv` (which also lists
the duplicates that `--keepDuplicates` retained) and keeps those with no row in the folder's
quant.sf. With one or more removed duplicates the quant tables lack those transcripts. It fires
only for a command that computes with the numbers (awk, python, perl, R, datamash, a `bash -c`
holding one) and that does not name any of the removed transcripts or read
`duplicate_clusters.tsv`. `drop` passes, `keep` fails (the file shows they were removed).
"""

from __future__ import annotations

import re
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.quant_featurecounts import command_reads_numbers


def fasta_arguments(argv: tuple[str, ...]) -> list[str]:
    """The files after -t / --transcripts (several allowed) of `salmon index`."""
    out: list[str] = []
    taking = False
    for token in argv:
        if token in ("-t", "--transcripts"):
            taking = True
        elif token.startswith("--transcripts="):
            out.append(token.split("=", 1)[1])
        elif token.startswith("-t") and not token.startswith("--") and len(token) > 2:
            out.append(token[2:])
        elif token.startswith("-"):
            taking = False
        elif taking:
            out.append(token)
    return out


def read_sequences(ctx: Ctx, path: Path, limit: int = 200_000) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line in ctx.lines(path, limit=limit):
        if line.startswith(">"):
            records.append((line[1:].split()[0] if line[1:].split() else "", ""))
        elif records:
            name, seq = records[-1]
            records[-1] = (name, seq + line.strip().upper())
    return records


def duplicate_groups(records: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """[(kept name, [dropped names])] for sequences identical to an earlier one."""
    first: dict[str, str] = {}
    dropped: dict[str, list[str]] = {}
    for name, seq in records:
        if not seq:
            continue
        if seq in first:
            dropped.setdefault(first[seq], []).append(name)
        else:
            first[seq] = name
    return list(dropped.items())


@checker("salmon_duplicate_transcripts")
def salmon_duplicate_transcripts(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    keeps = "--keepDuplicates" in call.argv
    files = [ctx.resolve(f) for f in fasta_arguments(call.argv)]
    names, records = [], []
    for path in files:
        if path.is_file():
            names.append(path.name)
            records += read_sequences(ctx, path)
    groups = duplicate_groups(records)  # salmon indexes the files as one transcriptome
    if not groups:
        return passed()
    n_dropped = sum(len(d) for _, d in groups)
    example = ", ".join(f"{d[0]} = {k}" for k, d in groups[:3])
    where = ", ".join(names)
    if keeps:
        if declared == "drop":
            return CheckResult(
                "fail",
                "declared_drop_but_kept",
                f"you declared {key}=drop but --keepDuplicates keeps the {n_dropped} duplicate "
                f"transcripts of {where}.",
                expected="drop",
                got="--keepDuplicates",
                fix=("remove --keepDuplicates, or declare duplicate_transcripts=keep",),
            )
        return passed()
    if declared == "drop":
        return passed()
    message = (
        f"{where} holds {n_dropped} transcript(s) with a sequence identical to an earlier one "
        f"({example}): salmon index keeps only the first, so the others get no row in quant.sf "
        "and quant.genes.sf and their reads are credited to the retained copy."
    )
    if declared == "keep":
        message = f"you declared {key}=keep but without --keepDuplicates: " + message
    return CheckResult(
        "fail",
        "duplicates_collapsed",
        message,
        expected="--keepDuplicates",
        got="collapsed",
        fix=(
            "add --keepDuplicates to salmon index to keep every transcript",
            f"or --declare {key}=drop if collapsing identical transcripts is intended",
        ),
    )


def removed_duplicates(ctx: Ctx, table: Path, cwd: Path) -> list[str]:
    """Duplicate transcripts missing from the transcript table of this quant run (empty: none or
    unknown). `duplicate_clusters.tsv` lists the duplicates even when --keepDuplicates retained
    them, so a transcript counts as removed only when `quant.sf` has no row for it."""
    info = table.parent / "cmd_info.json"
    if not info.is_file():
        return []
    import json

    try:
        index = json.loads(info.read_text()).get("index")
    except (OSError, ValueError):
        return []
    if not isinstance(index, str):
        return []
    for base in (cwd, table.parent.parent, table.parent):
        clusters = (base / index) / "duplicate_clusters.tsv"
        if clusters.is_file():
            rows = [line.split("\t") for line in ctx.lines(clusters)]
            listed = [r[1] for r in rows[1:] if len(r) > 1]
            break
    else:
        return []
    transcripts = table.parent / "quant.sf"
    if not transcripts.is_file():
        return listed
    present = {line.split("\t", 1)[0] for line in ctx.lines(transcripts)}
    return [name for name in listed if name not in present]


@checker("salmon_quant_duplicates")
def salmon_quant_duplicates(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = "\n".join([call.tool, *call.argv])
    if not command_reads_numbers(call.tool, text) or "duplicate_clusters" in text:
        return passed()  # shows the table, or reads the list of removed transcripts itself
    declared = call.declared.get(key)
    hits: list[tuple[str, list[str]]] = []
    for path in match.files:
        gone = removed_duplicates(ctx, path, call.cwd)
        if gone and not any(re.search(rf"(?<![\w.]){re.escape(g)}(?![\w.])", text) for g in gone):
            hits.append((path.name, gone))
    if not hits or declared == "drop":
        return passed()
    name, gone = hits[0]
    listing = ", ".join(gone[:3])
    message = (
        f"{name} comes from a salmon index that removed {len(gone)} exact-duplicate "
        f"transcript(s) ({listing}): they have no row in it, and their reads are counted under "
        "the retained copy."
    )
    if declared == "keep":
        message = f"you declared {key}=keep but " + message
    return CheckResult(
        "fail",
        "quant_lacks_duplicates",
        message,
        expected="drop declared",
        got=declared,
        fix=(
            "re-build the index with salmon index --keepDuplicates and re-run salmon quant",
            f"or --declare {key}=drop and treat the retained copy as the pooled count",
        ),
    )
