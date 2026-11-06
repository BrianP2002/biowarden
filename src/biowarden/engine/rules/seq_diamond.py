"""DIAMOND rules of the sequence pack: the hit cap of 25 and the default mode.

diamond.hit_cap (family F79, slot record_count; T1, call, declaration optional)
    `diamond blastp` / `blastx` report at most 25 target sequences per query by default
    (`--max-target-seqs`, `-k`; 0 means all). The tabular file ends at 25 subjects per query, best
    first, and stderr reads "Reported 25 pairwise alignments, 25 HSPs." as for any run. A query
    with 40 homologs looks like a query with 25 and the weakest homologs are the ones missing.
    `--top P` (a score window) replaces the cap.

    Verified here (DIAMOND 2.2.8; one 200-residue query, 40 homologs at 55-95% identity plus 30
    unrelated proteins): default 25 rows, `-k 100` and `-k 0` 39 rows. On the F79 sets (seeds
    0-59) the default call returns 25 subjects exactly when the family has more than 25 members.

    What the rule does. It applies to `diamond blastp` and `blastx`. It passes when `-k` is 0 or
    `--top` is given (`--top` with the declaration all_hits fails: a score window is not all
    hits). Otherwise, when `-q` and `-d` name small existing inputs (query up to 5 MB, database
    up to 20 MB), it reruns the call with `-k 0` (outputs to a temporary file, `-f 6 qseqid
    sseqid`, one thread) and fails when some query has more hit subjects than the cap in force
    (25 by default, or the `-k` value). Declaration `hit_limit`: all_hits (every hit is needed:
    fail when the cap binds) or top_hits (a capped list is intended: pass).
    Not seen: queries on stdin, `--daa` output, databases given as FASTA.

diamond.sensitivity_mode (family F80, slot default_param; T1, call, declaration optional)
    Without a sensitivity flag `diamond blastp` runs its default mode, tuned for speed on close
    homologs: pairs below about 50% identity fall through the seeding stage and the query has no
    row. The stderr summary reads "N queries aligned". `--ultra-sensitive` finds them.

    Verified here (DIAMOND 2.2.8, BLAST+ 2.17.0; twelve 180-residue queries with one mutated
    partner each, in a database of 42 proteins): partner identity 30 35 40 45 50 55 60 70 80 | 30
    35 40: default found 4 of 12 queries (50, 60, 70 and 80%), `--sensitive` 10, `--ultra-sensitive`
    12, blastp (E 0.001) all 12. On the F80 sets the default call also finds some 35-41% pairs by
    chance (never all).

    What the rule does. It applies to `diamond blastp` and `blastx`. A call with
    `--ultra-sensitive` passes. Otherwise, on small existing inputs, it runs the call and the same
    call with `--ultra-sensitive` (outputs to temporary files, `-f 6 qseqid evalue`) and fails when
    a query has a hit with E <= min(1e-4, the call's -e) only in the ultra-sensitive run.
    Declaration `homology_range`: close (hits above about 60% identity are all that is wanted:
    pass) or remote (distant homologs matter: fail unless the mode finds them all).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from biowarden.engine.argv import first_positional, option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.seq_common import (
    RUN_ERRORS,
    existing,
    memo,
    read_fasta,
    small_file,
    stamp,
)

DEFAULT_CAP = 25
DB_LIMIT = 20_000_000
EXTRA_EVALUE = 1e-4
SUBCOMMANDS = {"blastp", "blastx"}
SENSITIVITY = (
    "--faster",
    "--fast",
    "--mid-sensitive",
    "--sensitive",
    "--more-sensitive",
    "--very-sensitive",
)
# options replaced or removed in the dry runs (value counts: 1, except -f which takes several)
DROP_WITH_VALUE = {
    "-o", "--out", "-p", "--threads", "-k", "--max-target-seqs", "--top", "-t", "--tmpdir",
    "-a", "--daa", "--log", "--parallel-tmpdir",
}  # fmt: skip
DROP_FLAGS = {"--quiet", "-v", "--verbose", "--log"}


def _strip(argv: tuple[str, ...]) -> list[str]:
    """The call's arguments without output, thread, cap and verbosity options."""
    out: list[str] = []
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if name in ("-f", "--outfmt"):
            i += 1
            while i < len(items) and not items[i].startswith("-") and "=" not in token:
                i += 1
            continue
        if name in DROP_WITH_VALUE:
            i += 1 if "=" in token and token.startswith("--") else 2
            continue
        if name in DROP_FLAGS:
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _inputs(ctx: Ctx, argv: tuple[str, ...]) -> tuple[Path, Path] | None:
    """The query file and the database file of a call that is small enough to rerun."""
    queries = option_values(argv, "-q", "--query")
    dbs = option_values(argv, "-d", "--db")
    if not queries or not dbs or queries[-1] == "-":
        return None
    query = existing(ctx, queries[-1])
    name = dbs[-1]
    db = existing(ctx, name if name.endswith(".dmnd") else name + ".dmnd")
    if query is None or db is None or not small_file(query) or not small_file(db, DB_LIMIT):
        return None
    return query, db


def _dry_run(ctx: Ctx, argv: tuple[str, ...], files: tuple[Path, ...], fields: str, *extra: str):
    """Run `diamond <call>` with a temporary output; returns the rows (lists) or None."""
    base = _strip(argv)
    key = ("diamond", str(ctx.cwd), tuple(base), fields, extra, stamp(files))

    def compute():
        with tempfile.TemporaryDirectory(prefix="biowarden-diamond-") as tmp:
            out = Path(tmp) / "hits.tsv"
            cmd = [*base, "-f", "6", *fields.split(), *extra, "-o", str(out), "--threads", "1"]
            cmd += ["--quiet", "--tmpdir", tmp]
            try:
                proc = ctx.run_real("diamond", cmd)
            except RUN_ERRORS:  # a failing dry run is not evidence
                return None
            if proc.returncode != 0 or not out.is_file():
                return None
            return [line.split("\t") for line in out.read_text().splitlines() if line.strip()]

    return memo(key, compute)


def _query_count(ctx: Ctx, query: Path) -> int | str:
    records = read_fasta(ctx, query)
    return len(records) if records is not None else "?"


@checker("diamond_hit_cap")
def diamond_hit_cap(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    if first_positional(argv) not in SUBCOMMANDS:
        return passed()
    declared = call.declared.get(key)
    if option_values(argv, "--top"):
        if declared == "all_hits":
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}=all_hits but --top reports only the hits within a score "
                "window of the best one.",
                expected="all_hits",
                got="top_hits",
                fix=("replace --top by --max-target-seqs 0 (all hits)",),
            )
        return passed()
    if declared == "top_hits":
        return passed()
    caps = option_values(argv, "-k", "--max-target-seqs")
    try:
        cap = int(caps[-1]) if caps else DEFAULT_CAP
    except ValueError:
        return passed()
    if cap == 0:
        return passed()
    found = _inputs(ctx, argv)
    if found is None:
        return passed()
    query, db = found
    rows = _dry_run(ctx, argv, (query, db), "qseqid sseqid", "-k", "0")
    if rows is None:
        return passed()
    subjects: dict[str, set[str]] = {}
    for row in rows:
        if len(row) >= 2:
            subjects.setdefault(row[0], set()).add(row[1])
    capped = {q: len(s) for q, s in subjects.items() if len(s) > cap}
    if not capped:
        return passed()
    worst, most = max(capped.items(), key=lambda item: item[1])
    used = "the default" if not caps else f"-k {cap}"
    return CheckResult(
        "fail",
        "hit_cap_binds",
        f"diamond reports at most {cap} target sequences per query ({used}), and "
        f"{len(capped)} of {_query_count(ctx, query)} queries in {query.name} have more hit "
        f"subjects ({worst} has {most}): the weakest homologs are missing from the output with no "
        "message.",
        expected="all hits",
        got=cap,
        fix=(
            "add --max-target-seqs 0 (all hits)",
            f"or, if only the best {cap} hits are wanted: --declare {key}=top_hits",
        ),
    )


@checker("diamond_sensitivity_mode")
def diamond_sensitivity_mode(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    if first_positional(argv) not in SUBCOMMANDS:
        return passed()
    if "--ultra-sensitive" in argv:
        return passed()
    declared = call.declared.get(key)
    if declared == "close":
        return passed()
    mode = next((m for m in SENSITIVITY if m in argv), "default")
    found = _inputs(ctx, argv)
    if found is None:
        return passed()
    query, db = found
    current = _dry_run(ctx, argv, (query, db), "qseqid evalue")
    plain = tuple(a for a in argv if a not in SENSITIVITY)
    ultra = _dry_run(ctx, (*plain, "--ultra-sensitive"), (query, db), "qseqid evalue")
    if current is None or ultra is None:
        return passed()
    evalues = option_values(argv, "-e", "--evalue")
    try:
        limit = min(EXTRA_EVALUE, float(evalues[-1])) if evalues else EXTRA_EVALUE
    except ValueError:
        limit = EXTRA_EVALUE
    have = {row[0] for row in current}
    best: dict[str, float] = {}
    for row in ultra:
        if len(row) >= 2:
            try:
                best[row[0]] = min(float(row[1]), best.get(row[0], float(row[1])))
            except ValueError:
                continue
    gained = sorted(q for q, e in best.items() if e <= limit and q not in have)
    if not gained:
        return passed()
    shown = ", ".join(gained[:3]) + (", ..." if len(gained) > 3 else "")
    return CheckResult(
        "fail",
        "mode_misses_hits",
        f"{len(gained)} of {_query_count(ctx, query)} queries in {query.name} ({shown}) have no "
        f"hit in the {mode} mode but a hit with E <= {limit:g} with --ultra-sensitive: the "
        "default mode misses homologs below about 50% identity, with no message.",
        expected="--ultra-sensitive",
        got=mode,
        fix=(
            "add --ultra-sensitive (or --very-sensitive / --more-sensitive; check that it finds "
            "the same queries)",
            f"or, if only close homologs are wanted: --declare {key}=close",
        ),
    )
