"""minimap2 rule of the alignment pack: the default parameters drop reads under about 60 bases.

minimap2.short_reads_preset (T1, call, declaration optional)
    minimap2's defaults (and the long-read presets map-ont, map-pb, map-hifi, splice, asm*) need a
    chain of at least 3 minimizers (k=15, w=10) and a DP alignment score of 80 (`-s 80`, match
    score 2). `-x sr` lowers the thresholds for short reads. Without it, reads of 50 bases that
    match
    the genome exactly are often left unmapped. Exit status 0, valid SAM or PAF, no message that the
    reads are too short for the preset.

Verified here (minimap2 2.31, samtools 1.24; 100 exact reads per length at unique places of a 4 kb
genome; reads mapped by default / with `-x sr` / with `-x map-ont`):
    38: 0 / 76 / 0    40: 2 / 84    42: 6 / 92    44: 14 / 100    46: 29 / 100    48: 46 / 100
    50: 62 / 100 / (28 of 40 on another set)    52: 72 / 100    55: 92 / 100    60: 100 / 100
    70: 100 / 100
and with a few substitutions (60 reads per length): 36 bases 0 / 24, 50 bases 22 / 44, 75 bases 59 /
59, 100 and 150 bases 60 / 60 / 60. `-x map-hifi` mapped none of the 36-150 base reads.

What the rule does. It applies to `minimap2` calls. It reads the query files (the positional
arguments after the target; plain or gzip FASTQ/FASTA, first 5000 reads) and counts the reads
shorter than 60 bases. It is silent when fewer than 5% are that short, or when the call sets
`-x sr` (or `-ax sr`), or when the call sets -k, -w, -s, -m or -n itself (`-k 12 -w 5`
mapped all of the 47-52 base reads; the rule cannot tell which values suffice). Otherwise it fails;
the fix is `-x sr`. Declaration `short_read_policy`:
    drop_short  losing reads under about 60 bases is intended: pass
    keep_all    every read must be mapped: fail unless the preset is sr
Not seen: reads piped in, a preset set by an environment variable, the mapped fraction itself.
"""

from __future__ import annotations

import zlib

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.aln_bwa import read_lengths
from science_harness.engine.rules.aln_samtools import parse_options

KEY = "short_read_policy"
MM2_ARG_SHORT = "kwIdfgGFrnmpNABOEzsuJjoRtKx"
MM2_LONG = {"--preset": "x", "--threads": "t"}
SHORT_LIMIT = 60
MIN_SHARE = 0.05
SHORT_PRESETS = {"sr"}
TUNED = {"k", "w", "s", "m", "n"}  # -k -w -s -m -n: the caller tunes seeds and thresholds


@checker("minimap2_short_reads_preset")
def minimap2_short_reads_preset(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    opts, positionals = parse_options(call.argv, MM2_ARG_SHORT, MM2_LONG)
    preset = (opts.get("x") or [""])[-1]
    if preset in SHORT_PRESETS:
        return passed()
    if TUNED.intersection(opts):
        return passed("the call sets the seed or chaining thresholds itself")
    queries = [ctx.resolve(p) for p in positionals[1:] if ctx.resolve(p).is_file()]
    if not queries:
        return passed()
    lengths: list[int] = []
    unreadable: list[str] = []
    for path in queries[:2]:
        try:
            lengths += read_lengths(ctx, path)
        except (OSError, EOFError, zlib.error, UnicodeError) as exc:
            unreadable.append(f"{path.name} ({type(exc).__name__})")
    declared = call.declared.get(KEY)
    short = [n for n in lengths if n < SHORT_LIMIT]
    if not lengths or len(short) / len(lengths) < MIN_SHARE:
        if unreadable and declared is None:
            return CheckResult(
                "unknown",
                "reads_unreadable",
                f"could not read {', '.join(unreadable)}; minimap2 without -x sr loses reads under "
                f"{SHORT_LIMIT} bases and the lengths could not be checked.",
            )
        return passed()
    if declared == "drop_short":
        return passed()
    used = f"-x {preset}" if preset else "the default parameters"
    return CheckResult(
        "fail",
        "short_reads_without_sr_preset",
        f"{len(short)} of {len(lengths)} sampled reads are shorter than {SHORT_LIMIT} bases "
        f"(the shortest has {min(lengths)}), and minimap2 with {used} leaves many reads that "
        "short unmapped even when they match the genome exactly (checked: 50-base exact reads, "
        "62 of 100 mapped by default, 100 of 100 with -x sr).",
        expected="-x sr",
        got=preset or "default",
        fix=(
            "add -x sr (short-read preset) to the minimap2 call",
            f"or, if losing reads under {SHORT_LIMIT} bases is intended: "
            f"--declare {KEY}=drop_short",
        ),
    )
