"""bowtie2 rule of the alignment pack: the default maximum fragment length of paired-end runs.

bowtie2.maxins_default (T0, call, declaration required for paired-end runs)
    `bowtie2 -X/--maxins` is "the maximum fragment length for valid paired-end alignments"
    (default 500). A pair whose mates map facing each other but with a fragment of 600 bases is
    still aligned, but as a discordant pair: SAM flag 0x2 (proper pair) is clear and the stderr
    summary counts it under "aligned discordantly". Nothing fails. Libraries with longer inserts
    (size-selected RNA-seq, ChIP, long-insert libraries) lose their concordant counts and every
    downstream `-f 2` filter.

Verified here (bowtie2 2.5.5, samtools 1.24, family F33 data, seed 1: 92 pairs, 28 with fragments of
600-800 bases): the default run gives 64 first-mate records with flags 0x1 + 0x2 + 0x40 (the
28 long pairs come back "aligned discordantly 1 time"); `-X 1000` gives 92, `-X 800` too; on the
first test set (140 pairs, 20 of them 600-700 bases) 18 of those 20 were discordant by default. The
mates of a discordant pair are both reported, MAPQ and TLEN as for any pair. hisat2 2.2.3 prints the
same default but reported all 92 pairs concordant, so this rule does not apply to it.

What the rule does. It applies to `bowtie2` calls that read pairs (`-1` or `-2`). A T0 rule: it sees
the call only. The agent declares the longest fragment the library can have:
    up_to_500   no fragment is longer than 500 bases: pass (a larger -X is harmless)
    over_500    some fragments are longer: pass only when the call sets -X/--maxins above 500,
                otherwise fail (the default 500 would make those pairs discordant)
No declaration is refused by the engine (declaration_missing) with the fix below. Single-end runs
(`-U`) are not matched. Not seen: the fragment lengths themselves (a T0 rule cannot read the files),
`--no-discordant`, `--local` (fragment length then includes soft-clipped bases) and `-I`.
"""

from __future__ import annotations

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed, suggester
from science_harness.engine.rules.aln_samtools import flag_value, parse_options
from science_harness.engine.spec import Rule

KEY = "fragment_length_max"
DEFAULT_MAXINS = 500
BT2_ARG_SHORT = "x12Ub35pNLiIXkMDRsuSFo"
BT2_LONG = {"--maxins": "X", "--minins": "I", "--threads": "p", "--upto": "u", "--skip": "s"}


def maxins(call: Call) -> int:
    opts, _ = parse_options(call.argv, BT2_ARG_SHORT, BT2_LONG)
    values = [flag_value(v) for v in opts.get("X", [])]
    for value in reversed(values):
        if value is not None:
            return value
    return DEFAULT_MAXINS


@checker("bowtie2_maxins")
def bowtie2_maxins(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    declared = call.declared.get(KEY)
    longest = maxins(call)
    if declared == "over_500" and longest <= DEFAULT_MAXINS:
        source = f"-X {longest}" if longest != DEFAULT_MAXINS else "the default -X 500"
        return CheckResult(
            "fail",
            "maxins_too_small",
            f"you declared fragments over {DEFAULT_MAXINS} bases, but this call uses {source}: "
            "mates of a longer fragment align as a discordant pair (flag 0x2 clear, 'aligned "
            "discordantly' in the summary), so a count of proper or concordant pairs drops them.",
            expected="-X above 500",
            got=longest,
            fix=("add -X <the longest fragment of the library>, for example -X 1000",),
        )
    return passed()


@suggester("bowtie2_maxins")
def maxins_missing(call: Call, rule: Rule) -> tuple[str, ...]:
    longest = maxins(call)
    if longest > DEFAULT_MAXINS:
        return (
            f"this call sets -X {longest}; declare the longest fragment of the library: "
            f"--declare {KEY}=over_500",
        )
    return (
        f"bowtie2 calls a pair with a fragment over {DEFAULT_MAXINS} bases discordant (default "
        f"-X {DEFAULT_MAXINS}). If every fragment is at most {DEFAULT_MAXINS} bases, re-run with "
        f"--declare {KEY}=up_to_500; if some are longer, add -X <longest fragment> (for example "
        f"-X 1000) and --declare {KEY}=over_500",
    )
