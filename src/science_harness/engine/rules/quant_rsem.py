"""Rule rsem.strandedness (slot strand, family F46, CALL rule, tier T1, conditional declaration).

The trap. `rsem-calculate-expression --strandedness` defaults to `none`; `forward` or `reverse`
makes RSEM keep only alignments in that orientation. On an unstranded library that silently
drops about half of the reads, and the output looks normal.

Verified here (RSEM 1.3.1, `--alignments --paired-end --no-bam-output` on a paired transcriptome BAM
of 640 simulated fragments, 8 transcripts; rsem-sam-validator: valid): a dUTP library with
`--strandedness reverse` or `none` gives `sum(expected_count) = 640`; an unstranded library with
`--strandedness none` gives 640 and with `--strandedness forward` 303, with `reverse` 337
(the .cnt file still says `0 640 0 640`, exit 0, empty stderr). A stranded option that does not
match a stranded library (forward on dUTP) stops with `No valid read to estimate the length
distribution!` (exit 255), which is loud. That `none` on a stranded library loses reads where
genes overlap on opposite strands was not verified for RSEM, so the rule never fails a `none` call.

What the rule does. For `--alignments` calls it reads the BAM named as input (the first positional
ending .bam/.sam/.cram): the reference of an RSEM input is the transcriptome, so the strand of read
1 against the reference is the library (flag 0x10 clear = same strand as the transcript). >= 85 %
clear = forward, <= 15 % = reverse, 35-65 % = unstranded (at least 20 reads, the first 5000
alignments only). The call's type is `--strandedness` (none | forward | reverse) or `--forward-prob`
(1 forward, 0 reverse, otherwise unstranded). A stranded call that disagrees with the reads fails;
a `none` call and a call on reads (no `--alignments`: the library cannot be seen) pass. The
declaration `library_strandedness` must equal the options and the reads.
"""

from __future__ import annotations

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules import quant_common as qc

RSEM_VALUES = frozenset(
    "--strandedness --forward-prob -p --num-threads --seed --fai --fragment-length-min "
    "--fragment-length-max --fragment-length-mean --fragment-length-sd --bowtie-path "
    "--bowtie2-path --star-path --hisat2-path --temporary-folder --sampling-for-bam "
    "--ci-credibility-level --ci-memory --ci-number-samples-per-count-vector "
    "--bowtie-n --bowtie-e --bowtie-m --bowtie-chunkmbs --bowtie2-k --bowtie2-sensitivity-level "
    "--bowtie2-mismatch-rate --star-gzipped-read-file --star-bzipped-read-file "
    "--star-output-genome-bam "
    "--gibbs-burnin --gibbs-number-of-samples --gibbs-sampling-gap --pme-sample-number".split()
)
STRAND_OF = {"none": qc.UNSTRANDED, "forward": qc.FORWARD, "reverse": qc.REVERSE}
FIX = {
    qc.FORWARD: "--strandedness forward",
    qc.REVERSE: "--strandedness reverse",
    qc.UNSTRANDED: "--strandedness none",
}


def call_library(opts: dict[str, list[str]]) -> tuple[str, str] | None:
    """(library the options mean, how the call says it) or None for a value RSEM refuses."""
    setting = qc.last_value(opts, "--strandedness")
    if setting is not None:
        return (STRAND_OF[setting], f"--strandedness {setting}") if setting in STRAND_OF else None
    prob = qc.last_value(opts, "--forward-prob")
    if prob is not None:
        try:
            value = float(prob)
        except ValueError:
            return None
        lib = qc.FORWARD if value == 1 else qc.REVERSE if value == 0 else qc.UNSTRANDED
        return lib, f"--forward-prob {prob}"
    return qc.UNSTRANDED, "--strandedness none (the default)"


@checker("rsem_strandedness")
def rsem_strandedness(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, positionals, flags = qc.split_args(call.argv, RSEM_VALUES)
    if "--alignments" not in flags:
        return passed()
    stated = call_library(opts)
    bams = [
        ctx.resolve(p)
        for p in positionals
        if p.lower().endswith((".bam", ".sam", ".cram")) and ctx.resolve(p).is_file()
    ]
    if stated is None or not bams:
        return passed()
    library, option = stated
    profile = qc.transcriptome_profile(ctx, bams[0])
    return qc.strand_verdict(
        tool="rsem-calculate-expression",
        key=rule.declare.key,
        call_lib=library,
        option=option,
        profile=profile,
        declared=call.declared.get(rule.declare.key),
        unstranded_hurts=False,
        fix_for=FIX,
        data=bams[0].name,
    )
