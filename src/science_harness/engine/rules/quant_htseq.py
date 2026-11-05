"""Rules htseq.strandedness (slot strand, family F41) and htseq.min_mapq (default_param, F44).

htseq.strandedness. GEN-22 (HTSeq documentation: "If your RNA-Seq data has not been made with a
strand-specific protocol, this causes half of the reads to be lost"). `htseq-count --help` of
HTSeq 2.1.2 says `-s {yes,no,reverse}` defaults to `yes`: only reads on the same strand as the
gene are counted. Verified here on the F41 files (5 genes, 262 single-end reads, dUTP library with
4 % leaked sense reads): `htseq-count -q -s yes reads.bam genes.gtf` exits 0 with empty stderr and
prints 1-3 per gene, `-s reverse` prints 30-79, `-s no` the same as reverse here; on an
unstranded BAM `-s yes` reports about half of the reads. The only trace is `__no_feature`.

What the rule does (a CALL rule, tier T1, conditional declaration). It reads the BAM and the GTF
the call names (`-t` / `-i` as given, defaults exon / gene_id), looks at the first 5000 primary
alignments (read 1 of pairs) that overlap exons of genes on a single strand and measures the share
on the gene's strand: >= 85 % forward, <= 15 % reverse, 35-65 % unstranded, otherwise unknown; fewer
than 20 informative reads is unknown too. The call's strandedness is `-s`/`--stranded` (default
yes = forward, no = unstranded, reverse = reverse). Then:
  call == data                        pass, no declaration needed (zero friction)
  call stranded, data differs         fail (the reads that count are the minority or half)
  call `no` on a stranded library     fail only if >= 3 reads touch exons of two genes on opposite
                                      strands (htseq-count reports them `__ambiguous`), else pass
  declaration `library_strandedness`  must equal what the options mean and what the reads show
  data not provable (unknown)         unknown (a warning) when the call is stranded and undeclared
  files missing, `-` for stdin        pass
Limits: the first 5000 alignments of a sorted BAM only; reads outside the annotation are not
informative; the declaration is checked against options and data, not against what a script does.

htseq.min_mapq. `htseq-count -a/--minaqual` defaults to 10 (featureCounts `-Q` to 0). Verified on
the F44 files (unstranded BAM, 15-30 % of the reads at MAPQ 0-3): `htseq-count -q -s no` reports
counts 15-30 % short and `__too_low_aQual 128`; `-a 0` gives every read, `featureCounts -s 0`
counts all of them. A CALL rule (T1, conditional): it counts the primary alignments with NH <= 1 and
MAPQ below the call's floor (`-a`, default 10). Fewer than 3 such reads, or under 1 % of the
alignments looked at, pass silently (a STAR BAM keeps its multi-mappers at MAPQ 0-3 with NH > 1,
which htseq-count drops on purpose). Otherwise it fails unless `low_mapq_reads=drop` says the
filter is intended; `keep` with a floor above 0 fails. Limit: MAPQ semantics differ between
aligners; the rule only reports the number of unique alignments the floor removes.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules import quant_common as qc

HTSEQ_VALUES = frozenset(
    "-f --format -r --order --max-reads-in-buffer -s --stranded -a --minaqual -t --type "
    "-i --idattr "
    "--additional-attr -m --mode --nonunique --secondary-alignments --supplementary-alignments "
    "-o --samout -p --samout-format -d --delimiter -c --counts_output -n --nprocesses "
    "--feature-query".split()
)
STRAND_OF = {"yes": qc.FORWARD, "no": qc.UNSTRANDED, "reverse": qc.REVERSE}
FIX = {qc.FORWARD: "-s yes", qc.REVERSE: "-s reverse", qc.UNSTRANDED: "-s no"}
DEFAULT_MINAQUAL = 10
MIN_LOW_READS = 3


def _inputs(call: Call, ctx: Ctx):
    """(options, existing alignment files, existing GTF) or None when the call is not readable."""
    opts, positionals, _ = qc.split_args(call.argv, HTSEQ_VALUES)
    if len(positionals) < 2 or "-" in positionals:
        return None
    gtf = ctx.resolve(positionals[-1])
    bams = [ctx.resolve(p) for p in positionals[:-1]]
    if not gtf.is_file() or not any(b.is_file() for b in bams):
        return None
    return opts, [b for b in bams if b.is_file()], gtf


@checker("htseq_strandedness")
def htseq_strandedness(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    parsed = _inputs(call, ctx)
    if parsed is None:
        return passed()
    opts, bams, gtf = parsed
    setting = qc.last_value(opts, "-s", "--stranded") or "yes"
    if setting not in STRAND_OF:
        return passed()  # htseq-count refuses it
    option = f"-s {setting}" if qc.last_value(opts, "-s", "--stranded") else "-s yes (the default)"
    exons = qc.read_exons(
        ctx,
        gtf,
        qc.last_value(opts, "-t", "--type") or "exon",
        qc.last_value(opts, "-i", "--idattr") or "gene_id",
    )
    profile = qc.strand_profile(ctx, bams[0], exons) if exons else None
    return qc.strand_verdict(
        tool="htseq-count",
        key=rule.declare.key,
        call_lib=STRAND_OF[setting],
        option=option,
        profile=profile,
        declared=call.declared.get(rule.declare.key),
        unstranded_hurts=True,
        fix_for=FIX,
        data=bams[0].name,
    )


def low_mapq(ctx: Ctx, bam: Path, floor: int, limit: int = qc.DEFAULT_LIMIT) -> tuple[int, int]:
    """(unique alignments below `floor`, alignments looked at)."""
    low = total = 0
    for aln in qc.iter_alignments(ctx, bam, limit):
        total += 1
        if aln.nh <= 1 and aln.mapq < floor:
            low += 1
    return low, total


@checker("htseq_min_mapq")
def htseq_min_mapq(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    parsed = _inputs(call, ctx)
    if parsed is None:
        return passed()
    opts, bams, _ = parsed
    given = qc.last_value(opts, "-a", "--minaqual")
    try:
        floor = int(given) if given is not None else DEFAULT_MINAQUAL
    except ValueError:
        return passed()
    declared = call.declared.get(rule.declare.key)
    low, total = low_mapq(ctx, bams[0], floor)
    if low < MIN_LOW_READS or low * 100 < total:
        if declared == "keep" and low:
            return CheckResult(
                "fail",
                "keep_but_floor_drops",
                f"you declared {rule.declare.key}=keep but -a {floor} still skips {low} unique "
                f"alignments of {bams[0].name}.",
                expected="keep",
                got=f"-a {floor}",
                fix=("add -a 0 so that no alignment is skipped",),
            )
        return passed()
    where = "the default -a 10" if given is None else f"-a {floor}"
    if declared == "drop":
        return passed()
    if declared == "keep":
        return CheckResult(
            "fail",
            "keep_but_floor_drops",
            f"you declared {rule.declare.key}=keep but {where} skips {low} of the {total} "
            f"alignments of {bams[0].name} (unique, MAPQ below {floor}).",
            expected="keep",
            got=where,
            fix=("add -a 0 so that no alignment is skipped",),
        )
    return CheckResult(
        "fail",
        "low_mapq_dropped",
        f"{where} makes htseq-count skip {low} of the {total} alignments of {bams[0].name} "
        f"(unique, MAPQ below {floor}); they are counted only in __too_low_aQual.",
        expected="keep or drop declared",
        got=where,
        fix=(
            f"add -a 0 and --declare {rule.declare.key}=keep to count every alignment",
            f"or --declare {rule.declare.key}=drop if skipping MAPQ < {floor} is intended",
        ),
    )
