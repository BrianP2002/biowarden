"""hisat2 rule of the alignment pack: --rna-strandness sets the XS strand of every alignment.

hisat2.rna_strandness (T0, call, declaration required when the option is used)
    `hisat2 --rna-strandness <string>` ("specify strand-specific information", default unstranded
    in `hisat2 --help`): one letter for single reads (F, R), two for pairs (FR, RF); dUTP libraries
    (TruSeq Stranded) are RF (R for single reads), ligation-based forward libraries FR (F). Once the
    option is given hisat2 writes an XS:A strand tag on EVERY alignment (verified: 400 of 400
    records; without it only the spliced records carry XS, 130 of 130). The value is never checked
    against the reads: a wrong one gives the same alignment rate and flips XS.

Verified here (hisat2 2.2.3, StringTie, samtools 1.24) on 200 dUTP read pairs from four genes
(+, -, +, -; family F34 uses five): no option and `RF`: transcript strands + - + -; `FR`: - + - +
(every strand flipped, 100% overall alignment either way, XS:A on all 400 records); `R` or `F` on
the pairs: + - + + - . and + + - + (XS:A:+ on 300 of 400 records, a transcript with strand '.');
single-end read 1 of the same library: `R` + - + -, `F` - + - +. `stringtie --rf` on the FR
alignments still gave - + - +. Without --rna-strandness the strands are right (XS comes from the
splice motifs), so the trap needs the option set wrongly.

What the rule does. It applies to `hisat2` calls with `--rna-strandness`. A T0 rule. The agent
declares the library: `forward` (the read, or read 1, is on the transcript's strand: F, FR) or
`reverse` (opposite strand, dUTP: R, RF). It fails when the option value implies the other
library than the declaration (FR with reverse,
RF with forward). It only warns (unknown) when the reads are paired (-1/-2) and the value has one
letter (F or R): the outcome then depends on the data (it was right on one seed of 6 by accident).
The engine refuses a call without the declaration. Not seen: the library of the reads themselves
(a T0 rule cannot read them): a declaration is checked against the option, not against the data.
"""

from __future__ import annotations

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed, suggester
from science_harness.engine.rules.aln_samtools import parse_options
from science_harness.engine.spec import Rule

KEY = "library_strandedness"
HISAT2_ARG_SHORT = "x12USpkIXNLi53suR"
HISAT2_LONG = {"--rna-strandness": "R", "--threads": "p", "--maxins": "X", "--minins": "I"}
IMPLIED = {"F": "forward", "FR": "forward", "R": "reverse", "RF": "reverse"}
RIGHT_VALUE = {
    ("forward", True): "FR",
    ("forward", False): "F",
    ("reverse", True): "RF",
    ("reverse", False): "R",
}


def strandness(call: Call) -> tuple[str | None, bool]:
    """(the last --rna-strandness value, whether the call reads pairs)."""
    opts, _ = parse_options(call.argv, HISAT2_ARG_SHORT, HISAT2_LONG)
    values = opts.get("R", [])
    paired = "1" in opts or "2" in opts
    return (values[-1].strip().upper() if values else None), paired


@checker("hisat2_rna_strandness")
def hisat2_rna_strandness(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    value, paired = strandness(call)
    declared = call.declared.get(KEY)
    if value is None or value not in IMPLIED:
        return passed()
    implied = IMPLIED[value]
    if declared is not None and declared != implied:
        right = RIGHT_VALUE[(declared, paired)]
        meaning = (
            "read 1 on the transcript strand"
            if implied == "forward"
            else "read 1 opposite to the transcript strand, dUTP"
        )
        return CheckResult(
            "fail",
            "strandness_contradicts_declaration",
            f"--rna-strandness {value} means a {implied}-stranded library ({meaning}), but you "
            f"declared {KEY}={declared}: hisat2 does not check the reads, it writes the XS strand "
            "of every alignment from this value, so assemblers such as StringTie put every "
            "transcript on the wrong strand.",
            expected=right,
            got=value,
            fix=(f"use --rna-strandness {right} for a {declared}-stranded library",),
        )
    if paired and len(value) == 1:
        right = RIGHT_VALUE[(declared or implied, True)]
        return CheckResult(
            "unknown",
            "single_letter_on_pairs",
            f"--rna-strandness {value} is the single-end form; for paired reads hisat2 needs FR or "
            "RF, and with one letter the XS strand comes out inconsistent between the mates "
            "(checked: transcripts with strand '.' or wrong strands after StringTie).",
            expected=right,
            got=value,
            fix=(f"use --rna-strandness {right}",),
        )
    return passed()


@suggester("hisat2_rna_strandness")
def strandness_missing(call: Call, rule: Rule) -> tuple[str, ...]:
    value, paired = strandness(call)
    implied = IMPLIED.get(value or "")
    if implied:
        return (
            f"--rna-strandness {value} says the library is {implied}-stranded; state it: "
            f"--declare {KEY}={implied} (dUTP / TruSeq Stranded is reverse: "
            f"{'RF' if paired else 'R'}; ligation-based forward is {'FR' if paired else 'F'})",
        )
    return (f"re-run with --declare {KEY}=<forward|reverse>",)
