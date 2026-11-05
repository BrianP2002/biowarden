"""Rule stringtie.strandedness (slot strand, CALL rule, tier T1, conditional declaration).

The trap. `stringtie --rf` (fr-firststrand, dUTP) and `--fr` (fr-secondstrand) restrict the reads
that count towards a transcript to one orientation; without either, StringTie is unstranded
(nf-core/rnaseq 3.27.0 restored the flags its v3 module update had dropped, GEN-23). The names are
easy to confuse: dUTP / TruSeq Stranded is `--rf`.

Verified here (StringTie 3.0.3, `stringtie reads.bam -G genes.gtf -e -A tab.tsv -o out.gtf -p 1`
on single-end and paired BAMs of 5 one-isoform genes; per-gene reads recovered as
coverage * length / 50, halved for pairs): a dUTP library gives the exact read counts with `--rf`
and with no flag, and loses up to 89 reads of a gene with `--fr` (coverage 0.0 for most genes); an
unstranded library is exact with no flag and loses 50 resp. 39 reads of a gene with `--rf` resp.
`--fr`; a forward library mirrors the dUTP case. Exit 0, empty stderr in every case. With a gene
pair overlapping on opposite strands the unstranded default changed the coverage of the pair by
under 1 % in the one case tried, so the rule never fails a call without a strand flag.

What the rule does. It reads the BAM (first positional ending .bam/.sam/.cram) and the GTF named by
`-G`, measures the share of read 1 on the gene's strand exactly as quant_htseq does, and compares
the library with the flag of the call: `--rf` is reverse, `--fr` forward, none unstranded. A
flag that disagrees with the reads fails; no flag passes; `--merge` calls and calls without `-G`
or with unreadable files pass. The declaration `library_strandedness` must equal the flag and
the reads. Limits as in quant_htseq (first 5000 alignments, annotated reads only, 20 informative
reads).
"""

from __future__ import annotations

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules import quant_common as qc

STRINGTIE_VALUES = frozenset(
    "-G -l -o -p -a -m -j -f -c -g -s -M -A -b -x -E -F -T --ptf --ref --cram-ref".split()
)
FIX = {qc.FORWARD: "--fr", qc.REVERSE: "--rf", qc.UNSTRANDED: "no --rf/--fr flag"}


@checker("stringtie_strandedness")
def stringtie_strandedness(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, positionals, flags = qc.split_args(call.argv, STRINGTIE_VALUES)
    if "--merge" in flags or ("--rf" in flags and "--fr" in flags):
        return passed()
    annotation = qc.last_value(opts, "-G")
    bams = [
        ctx.resolve(p)
        for p in positionals
        if p.lower().endswith((".bam", ".sam", ".cram")) and ctx.resolve(p).is_file()
    ]
    if annotation is None or not bams or not ctx.resolve(annotation).is_file():
        return passed()
    library, option = (
        (qc.REVERSE, "--rf")
        if "--rf" in flags
        else (qc.FORWARD, "--fr")
        if "--fr" in flags
        else (qc.UNSTRANDED, "no strand flag")
    )
    exons = qc.read_exons(ctx, ctx.resolve(annotation))
    profile = qc.strand_profile(ctx, bams[0], exons) if exons else None
    return qc.strand_verdict(
        tool="stringtie",
        key=rule.declare.key,
        call_lib=library,
        option=option,
        profile=profile,
        declared=call.declared.get(rule.declare.key),
        unstranded_hurts=False,
        fix_for=FIX,
        data=bams[0].name,
    )
