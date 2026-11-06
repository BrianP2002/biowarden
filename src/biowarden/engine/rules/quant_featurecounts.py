"""Rules featurecounts.read_pairs (F42), .strandedness (F43) and .counts_unit (a fact rule).

featurecounts.read_pairs (slot record_count, CALL rule, tier T0, declaration `count_unit`
required). GEN-24. Since Subread 2.0.2 `-p` only says the input is paired-end; fragments are
counted only with `--countReadPairs`. Verified on featureCounts 2.1.1 on a paired BAM of 640
fragments (1280 records): `featureCounts -p -a genes.gtf -o c.txt reads.bam` exits 0, the counts
sum to 1280 and the summary says `Assigned 1280`; with `-p --countReadPairs` they sum to 640;
without `-p` the run stops (`ERROR: Paired-end reads were detected in single-end read library`,
exit 255). The banner on stderr says `Count read pairs : no`, nothing else. The rule applies to
every call with `-p`; the options decide what the call counts (`--countReadPairs` present:
fragment, absent: read), and the declaration must say the same. It cannot know which one the task
wants; it makes the agent state it. Calls without `-p` are not matched.

featurecounts.strandedness (slot strand, CALL rule, tier T1, conditional declaration
`library_strandedness`). GEN-20. `-s` defaults to 0 (unstranded). Verified on the F43 files (dUTP
BAM, two genes overlapping on opposite strands, 64 of 299 reads in the overlap): `-s 0` and `-s 1`
lose reads (`Unassigned_Ambiguity` 64 and 130 for -s 0 resp. `Unassigned_NoFeatures`), `-s 2`
assigns every read exactly once. Without an antisense overlap `-s 0` on a stranded library gives
the same counts as `-s 2`, so the rule stays silent there. The sniff is the one of quant_htseq
(strand of read 1 against the exons of the GTF named by `-a`, `-t`/`-g` as given): call `-s` is
0 unstranded / 1 forward / 2 reverse; a comma list of different values, `-F SAF` and a missing file
are not judged. Fails: a stranded `-s` that is not the library's; `-s 1|2` on an unstranded
library; `-s 0` on a stranded library when at least 3 sniffed reads touch exons of genes on both
strands. Limits as in quant_htseq (first 5000 alignments, annotated reads only).

featurecounts.counts_unit (slot units, FACT rule, tier T1, conditional declaration `count_unit`).
A counts table written by `featureCounts -p` without `--countReadPairs` holds numbers of reads
(mates): twice the number of fragments. The first line of every featureCounts table records the
command (`# Program:featureCounts v2.1.1; Command:"featureCounts" "-p" "-a" ...`), which the rule
reads, so it knows what the numbers are whoever made the file. It applies to files whose name
holds `count` and ends in .txt/.tsv, and fires only when (a) line 1 shows `-p` and no
`--countReadPairs` and (b) the command reads the numbers (awk, python, perl, R, datamash, a
`bash -c` holding one of them; head, cat, wc and the like are silent) and (c) the command shows no
division by two. `count_unit=read` passes (the agent knows), `fragment` fails.
"""

from __future__ import annotations

import re

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed, suggester
from biowarden.engine.rules import quant_common as qc
from biowarden.engine.spec import Rule

FC_VALUES = frozenset(
    "-a -o -F -t -g -A -G -R -s -Q -T -d -D --extraAttributes --minOverlap --fracOverlap "
    "--fracOverlapFeature --readExtension5 --readExtension3 --read2pos --Rpath --tmpDir "
    "--nonOverlap --nonOverlapFeature".split()
)
STRAND_OF = {"0": qc.UNSTRANDED, "1": qc.FORWARD, "2": qc.REVERSE}
FIX = {qc.FORWARD: "-s 1", qc.REVERSE: "-s 2", qc.UNSTRANDED: "-s 0"}


@checker("featurecounts_read_pairs")
def featurecounts_read_pairs(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    implied = "fragment" if "--countReadPairs" in call.argv else "read"
    declared = call.declared[key]
    if declared == implied:
        return passed()
    if implied == "read":
        message = (
            f"you declared {key}=fragment but featureCounts -p without --countReadPairs "
            "counts every mate as a read: each fragment (read pair) adds 2 to its gene."
        )
        fix = ("add --countReadPairs to count read pairs",)
    else:
        message = (
            f"you declared {key}=read but --countReadPairs makes featureCounts count read "
            "pairs (fragments), one per pair."
        )
        fix = ("drop --countReadPairs to count every mate as a read",)
    return CheckResult(
        "fail", "declared_mismatch", message, expected=declared, got=implied, fix=fix
    )


@suggester("featurecounts_read_pairs")
def _read_pairs_fix(call: Call, rule: Rule) -> tuple[str, ...]:
    key = rule.declare.key if rule.declare else "count_unit"
    if "--countReadPairs" in call.argv:
        return (f"re-run with --declare {key}=fragment (--countReadPairs counts read pairs)",)
    return (
        f"for fragments add --countReadPairs and --declare {key}=fragment: -p alone counts "
        "every mate as a read",
        f"or declare --declare {key}=read if reads (mates) are what you want",
    )


@checker("featurecounts_strandedness")
def featurecounts_strandedness(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, positionals, _ = qc.split_args(call.argv, FC_VALUES)
    annotation = qc.last_value(opts, "-a")
    if annotation is None or (qc.last_value(opts, "-F") or "GTF").upper() != "GTF":
        return passed()
    gtf = ctx.resolve(annotation)
    bams = [ctx.resolve(p) for p in positionals if ctx.resolve(p).is_file()]
    if not gtf.is_file() or not bams:
        return passed()
    given = qc.last_value(opts, "-s")
    values = set((given or "0").split(","))
    if len(values) != 1 or not values <= set(STRAND_OF):
        return passed()
    setting = values.pop()
    option = f"-s {setting}" if given is not None else "-s 0 (the default)"
    feature = (qc.last_value(opts, "-t") or "exon").split(",")[0]
    exons = qc.read_exons(ctx, gtf, feature, qc.last_value(opts, "-g") or "gene_id")
    profile = qc.strand_profile(ctx, bams[0], exons) if exons else None
    return qc.strand_verdict(
        tool="featureCounts",
        key=rule.declare.key,
        call_lib=STRAND_OF[setting],
        option=option,
        profile=profile,
        declared=call.declared.get(rule.declare.key),
        unstranded_hurts=True,
        fix_for=FIX,
        data=bams[0].name,
    )


# ---- the counts table (fact rule) --------------------------------------------------------------

READS_NUMBERS = re.compile(
    r"\b(?:g?awk|mawk|python[\d.]*|python-\w+|perl|Rscript|R|datamash|julia)\b"
)
DIRECT = frozenset(
    "awk gawk mawk python python3 perl Rscript R datamash julia python-bio python-sc "
    "python-struct".split()
)
SHELLS = frozenset("bash sh zsh dash env xargs parallel timeout time nohup srun".split())
HALVES = re.compile(r"/\s*2(?![\d.])|//\s*2\b|\*\s*0?\.5\b|/\s*2\.0\b|\bhalf\b", re.IGNORECASE)


def command_reads_numbers(tool: str, text: str) -> bool:
    """Does this command compute with the numbers of a table, not just show it?"""
    if tool in SHELLS:
        return READS_NUMBERS.search(text) is not None
    return tool in DIRECT or re.fullmatch(r"python[\d.]+", tool) is not None


@checker("featurecounts_counts_unit")
def featurecounts_counts_unit(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = "\n".join([call.tool, *call.argv])
    if not command_reads_numbers(call.tool, text) or HALVES.search(text):
        return passed()
    declared = call.declared.get(key)
    bad: list[str] = []
    for path in match.files:
        try:
            head = next(iter(ctx.lines(path, limit=1)), "")
        except (OSError, EOFError):
            continue
        if not head.startswith("# Program:featureCounts"):
            continue
        words = re.findall(r'"([^"]*)"', head)
        if "-p" in words and "--countReadPairs" not in words:
            bad.append(path.name)
    if not bad:
        return passed()
    names = ", ".join(bad)
    if declared == "read":
        return passed()
    if declared == "fragment":
        return CheckResult(
            "fail",
            "declared_fragments_but_reads",
            f"you declared {key}=fragment but {names} was made by featureCounts -p without "
            "--countReadPairs: its counts are reads (mates), 2 per fragment.",
            expected="read",
            got="fragment",
            fix=("re-run featureCounts with -p --countReadPairs, or divide the counts by 2",),
        )
    return CheckResult(
        "fail",
        "counts_are_reads",
        f"{names} was made by featureCounts -p without --countReadPairs: every count is a number "
        "of reads (mates), twice the number of fragments (read pairs).",
        expected="fragment or read declared",
        got="read",
        fix=(
            "re-run featureCounts with -p --countReadPairs for fragment counts",
            f"or --declare {key}=read if the counts are meant as reads",
        ),
    )
