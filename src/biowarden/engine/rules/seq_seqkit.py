"""seqkit rule of the sequence pack: `rmdup -s` merges a sequence with its reverse complement.

seqkit.rmdup_strands (family F82, slot strand; T1, call, declaration optional)
    `seqkit rmdup --by-seq` removes records with identical sequences and, for nucleotides,
    compares both strands unless `-P, --only-positive-strand` is given: a guide, primer or barcode
    listed as the reverse complement of another entry is dropped as a duplicate. Exit 0; stderr
    carries an INFO line with the number of removed records.

    Verified here (seqkit v2.14.0): six records (a1, its reverse complement a3, an exact copy a4, a
    lower-case copy of a1, two unrelated): `rmdup -s` kept 4 (a3 and a4 removed), `rmdup -s -P` kept
    5 (only a4 removed), `rmdup -s -i` also removed the lower-case copy. On the F82 sets (seeds
    0-39) the plain call leaves one record fewer than there are distinct sequences per
    reverse-complement pair, and `-P` leaves exactly the distinct sequences.

    What the rule does. It applies to `seqkit rmdup` calls that compare by sequence (`-s`,
    `--by-seq`, also inside a combined flag such as `-si`) without `-P`/`--only-positive-strand`.
    It reads the FASTA/FASTQ files named in the call (up to 5 MB each), and when the sequences are
    nucleotides and two of them are reverse complements of each other (case ignored only with `-i`)
    it fails. Declaration `duplicate_strand_policy`: both_strands (a reverse-complement pair is
    one sequence: pass) or same_strand_only (they are two: fail, the fix is -P). Silent when no
    such pair exists, for protein files, by-name runs and standard input.
"""

from __future__ import annotations

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.seq_common import (
    existing,
    is_nucleotide,
    positionals,
    read_fasta,
    read_fastq,
    revcomp,
    small_file,
)

VALUE_OPTIONS = frozenset(
    "-o --out-file -d --dup-seqs-file -D --dup-num-file -j --threads -w --line-width --id-regexp "
    "-I --infile-list --alphabet-guess-seq-length -t --seq-type --compress-level".split()
)


def _flags(args: list[str]) -> set[str]:
    """Long flags and the letters of short flags (values of value options skipped)."""
    found: set[str] = set()
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token.startswith("--"):
            name = token.split("=", 1)[0]
            found.add(name)
            skip = name in VALUE_OPTIONS and "=" not in token
        elif token.startswith("-") and len(token) > 1:
            if token in VALUE_OPTIONS:
                skip = True
            else:
                found.update(token[1:])
    return found


def _sequences(ctx: Ctx, path) -> list[str] | None:
    fasta = read_fasta(ctx, path)
    if fasta:
        return [seq for _, seq in fasta]
    fastq = read_fastq(ctx, path)
    return [seq for _, seq, _ in fastq] if fastq else None


@checker("seqkit_rmdup_strands")
def seqkit_rmdup_strands(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = list(call.argv)
    if "rmdup" not in argv:
        return passed()
    args = argv[argv.index("rmdup") + 1 :]
    flags = _flags(args)
    if not ({"s", "--by-seq"} & flags) or {"P", "--only-positive-strand"} & flags:
        return passed()
    declared = call.declared.get(key)
    if declared == "both_strands":
        return passed()
    ignore_case = bool({"i", "--ignore-case"} & flags)
    for name in positionals(args, VALUE_OPTIONS):
        path = existing(ctx, name) if name != "-" else None
        if path is None or not small_file(path):
            continue
        seqs = _sequences(ctx, path)
        if not seqs or not is_nucleotide(seqs):
            continue
        norm = [s.upper() if ignore_case else s for s in seqs]
        present = set(norm)
        pairs = sorted(
            {min(s, revcomp(s)) for s in present if revcomp(s) != s and revcomp(s) in present}
        )
        if not pairs:
            continue
        shown = pairs[0]
        return CheckResult(
            "fail",
            "reverse_complement_duplicates",
            f"seqkit rmdup -s also treats a sequence and its reverse complement as duplicates: "
            f"{path.name} holds {len(pairs)} such pair(s) (for example {shown} and "
            f"{revcomp(shown)}), and one of each pair is dropped.",
            expected="same_strand_only",
            got="both_strands",
            fix=(
                "add -P (--only-positive-strand) to compare sequences as written",
                f"or, if a reverse-complement pair is one sequence: --declare {key}=both_strands",
            ),
        )
    return passed()
