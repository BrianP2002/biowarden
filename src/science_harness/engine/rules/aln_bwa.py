"""bwa rule of the alignment pack: reads shorter than the minimum output score are never aligned.

bwa.mem_min_score (T1, call, declaration optional)
    `bwa mem -T INT` is the minimum score to output an alignment, default 30; with the default
    match score `-A 1` a read of L bases scores at most L, so every read shorter than 30 bases comes
    out as an unmapped record (flag 4, no tags) however well it matches. Exit status 0, valid SAM,
    only "[M::mem_process_seqs] Processed N reads" on stderr.

Verified here (bwa 0.7.19, samtools 1.24), 40 exact reads per length at unique places of a 4 kb
genome, mapped records (`samtools view -c -F 4`) by read length and `-T`:
    length 20: default 0, -T 15 40;  25: 0, 40;  30: 40, 40;  35: 40, 40;  50: 40, 40
bowtie2 and hisat2 align the same 20- and 25-base reads (40 of 40); minimap2 -x sr aligns none of
them (0 of 40 at 20, 25 and 30 bases, 25 at 35, 40 at 50), so short reads need an aligner that
supports them: bwa aln, bowtie2, or bwa mem with a lowered -T (seed length -k is 19, so reads under
19 bases cannot be seeded at all). In the family F30 data (reads 20-75 bases) the default gives 46
of 93 reads on one seed, the exact count of reads >= 30 bases.

What the rule does. It applies to `bwa mem` calls. It reads the FASTQ (or FASTA) files named among
the positional arguments (index prefix first, then 1 or 2 read files; plain or gzip), the first 5000
reads, and computes the longest possible score of each read, length x `-A` (default 1). Reads with
that score below `-T` (default 30) can never be output. It is silent when none of the sampled reads
is that short. Otherwise, without a declaration it fails; the fix is a lowered -T. With the
declaration `short_read_policy`:
    drop_short  the caller knows reads under -T are dropped and wants it: pass
    keep_all    the caller wants every read: pass only when -T is at most the shortest sampled
                read's best score, otherwise fail (the call contradicts the declaration)
Not seen: reads piped in (`-` or process substitution), read files that do not exist yet, an
interleaved file with `-p` is read like any other, reads shorter than the seed length -k.
"""

from __future__ import annotations

import zlib
from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.aln_samtools import flag_value, parse_options

KEY = "short_read_policy"
SAMPLE_READS = 5000
DEFAULT_MIN_SCORE = 30
DEFAULT_MATCH = 1
BWA_MEM_ARG_SHORT = "tkwdrycDWmABOELUxRHvThKIo"
SEED_LENGTH = 19


def read_lengths(ctx: Ctx, path: Path, limit: int = SAMPLE_READS) -> list[int]:
    """Lengths of the first `limit` reads of a FASTQ or FASTA file (plain or gzip)."""
    lengths: list[int] = []
    lines = ctx.lines(path)
    try:
        first = next(lines, "")
        if first.startswith("@"):
            for i, line in enumerate(lines, start=1):
                if i % 4 == 1:
                    lengths.append(len(line.strip()))
                    if len(lengths) >= limit:
                        break
        elif first.startswith(">"):
            current, started = 0, True
            for line in lines:
                if line.startswith(">"):
                    lengths.append(current)
                    current = 0
                    if len(lengths) >= limit:
                        started = False
                        break
                else:
                    current += len(line.strip())
            if started:
                lengths.append(current)
    finally:
        lines.close()  # type: ignore[attr-defined]
    return lengths


@checker("bwa_mem_min_score")
def bwa_mem_min_score(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "mem":
        return passed()
    opts, positionals = parse_options(call.argv[1:], BWA_MEM_ARG_SHORT, {})
    threshold = _int_option(opts, "T", DEFAULT_MIN_SCORE)
    match_score = _int_option(opts, "A", DEFAULT_MATCH)
    reads = [ctx.resolve(p) for p in positionals[1:] if ctx.resolve(p).is_file()]
    if not reads:
        return passed()
    short: list[int] = []
    total = 0
    unreadable: list[str] = []
    for path in reads[:2]:
        try:
            lengths = read_lengths(ctx, path)
        except (OSError, EOFError, zlib.error, UnicodeError) as exc:
            unreadable.append(f"{path.name} ({type(exc).__name__})")
            continue
        total += len(lengths)
        short += [n for n in lengths if n * match_score < threshold]
    declared = call.declared.get(KEY)
    if not short:
        if unreadable and declared is None:
            return CheckResult(
                "unknown",
                "reads_unreadable",
                f"could not read {', '.join(unreadable)}; reads shorter than -T {threshold} are "
                "not output by bwa mem and the lengths could not be checked.",
            )
        return passed()
    shortest = min(short)
    best = shortest * match_score
    if declared == "drop_short":
        return passed()
    if declared == "keep_all" and threshold <= best:
        return passed()
    fix_t = max(1, best)
    note = ""
    if shortest < SEED_LENGTH:
        note = (
            f" ({shortest}-base reads are shorter than the seed length -k {SEED_LENGTH}: use "
            "bwa aln for them)"
        )
    message = (
        f"bwa mem outputs no alignment scoring below -T {threshold}"
        f"{'' if 'T' in opts else ' (default)'}: {len(short)} of {total} sampled reads "
        f"are shorter than {threshold} bases (the shortest has {shortest}, best score {best}), so "
        f"they come out as unmapped records however well they match{note}."
    )
    if declared == "keep_all":
        message = f"you declared {KEY}=keep_all but -T {threshold} drops reads: " + message
    return CheckResult(
        "fail",
        "reads_below_min_score",
        message,
        expected=f"-T at most {best}",
        got=threshold,
        fix=(
            f"add -T {fix_t} (the lowest best score among the sampled reads) to bwa mem",
            f"or, if dropping reads under {threshold} bases is intended, declare it: "
            f"--declare {KEY}=drop_short",
        ),
    )


def _int_option(opts: dict[str, list[str]], letter: str, default: int) -> int:
    values = [flag_value(v) for v in opts.get(letter, [])]
    for value in reversed(values):
        if value is not None:
            return value
    return default
