"""Where the error in a task lives: the layer an agent would have to inspect to find it.

data            the files themselves carry the convention (encoding, ID form, coordinate base)
tool_default    the tool's documented-but-obscure default or option semantics decide the result
task_statement  the user's wording disagrees with the data or a convention
documentation   the README or --help text no longer matches the installed behaviour
wrapper         a local wrapper script changes what the tool does or hides what it says
service         a remote or local service returns plausible but incomplete or retired results
"""

from __future__ import annotations

LOCI = ("data", "tool_default", "task_statement", "documentation", "wrapper", "service")

# The sixteen original families (extra modules may also set FamilySpec.locus, which wins).
FAMILY_LOCUS = {
    "F1": "tool_default",  # bcftools reads a regions file by its extension
    "F2": "data",  # region strings carry a BED-style 0-based start, samtools reads them 1-based
    "F3": "data",  # chr1 against 1
    "F4": "data",  # genome build differs between files
    "F5": "tool_default",  # samtools depth positions against BED coordinates
    "F6": "tool_default",  # default duplicate filter
    "F7": "tool_default",  # -R against -T
    "F8": "tool_default",  # record counts include secondary/supplementary
    "F9": "data",  # multi-allelic rows
    "F10": "data",  # indel anchor base in VCF
    "F11": "data",  # GFF3 against BED coordinates
    "F12": "tool_default",  # getfasta ignores strand unless -s
    "F13": "data",  # Phred+33 against Phred+64
    "F14": "data",  # versioned against unversioned IDs
    "F15": "data",  # log-transformed matrix
    "F16": "tool_default",  # flank ignores strand unless -s
}


def locus_of(family: str) -> str:
    """The family's locus: the table above for F1-F16, otherwise the extra module's FamilySpec."""
    locus = FAMILY_LOCUS.get(family)
    if locus is None:
        from science_harness.bench.tasks.extra import load

        spec = load().get(family)
        locus = spec.locus if spec else ""
    if locus not in LOCI:
        raise KeyError(f"family {family} has no valid locus ({locus!r}); use one of {LOCI}")
    return locus
