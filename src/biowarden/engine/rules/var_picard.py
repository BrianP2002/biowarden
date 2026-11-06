"""Picard rule of the variant pack (MarkDuplicates).

picard.markduplicates_remove (T0, call, declaration required `duplicate_output`)
    `picard MarkDuplicates I=in.bam O=out.bam M=m.txt` writes EVERY read of the input to the
    output: the duplicates get flag 0x400 and stay in the file. They are removed only with
    REMOVE_DUPLICATES=true (REMOVE_SEQUENCING_DUPLICATES removes the optical and other
    sequencing-process duplicates only). A read count of the output, an idxstats, a depth or
    a coverage that ignores the flag counts the reads before de-duplication; the metrics file
    counts PAIRS (READ_PAIR_DUPLICATES), not reads. Exit 0; stderr has the JVM's native-access
    warnings and an INFO line "Marking 40 records as duplicates", nothing else about it.

Verified here (Picard 3.5.0 through the conda wrapper, `-Xmx512m`; samtools 1.24), 120 FR pairs of
which 20 duplicate another pair, coordinate-sorted, one library:
    default                                 240 records, 40 with flag 0x400
    REMOVE_DUPLICATES=true                  200 records (legacy syntax)
    --REMOVE_DUPLICATES true                200 records
    --REMOVE_SEQUENCING_DUPLICATES true     240 records (the PCR duplicates are kept)
    metrics: READ_PAIRS_EXAMINED 120, READ_PAIR_DUPLICATES 20, PERCENT_DUPLICATION 0.166667
`--REMOVE_DUPLICATES=true` is rejected with the usage text (loud).

What the rule does. It is decided from the call alone (T0). For `picard MarkDuplicates` and
`gatk MarkDuplicates` calls it reads the REMOVE_DUPLICATES setting (`REMOVE_DUPLICATES=true`,
`--REMOVE_DUPLICATES true`, `-REMOVE_DUPLICATES true`; default false) and checks it against the
optional declaration:
    flagged   the output keeps the duplicates, flagged 0x400: the call must not remove them
    removed   the output holds no duplicate: the call must have REMOVE_DUPLICATES=true
The engine fails a MarkDuplicates call without the declaration (declaration_missing). Other
picard tools are not matched.
Not seen: MarkDuplicatesWithMateCigar, the barcode-aware variants, REMOVE_DUPLICATES given twice.
"""

from __future__ import annotations

import re

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.var_common import fail

REMOVE = re.compile(r"^-{0,2}REMOVE_DUPLICATES=(\w+)$")
SEPARATE = "REMOVE_DUPLICATES"


def removes_duplicates(argv: tuple[str, ...]) -> bool:
    value = "false"
    items = list(argv)
    for i, token in enumerate(items):
        hit = REMOVE.match(token)
        if hit:
            value = hit.group(1)
        elif token.lstrip("-") == SEPARATE and token.startswith("-"):
            value = items[i + 1] if i + 1 < len(items) else "true"
    return value.lower() == "true"


@checker("picard_markduplicates_remove")
def picard_markduplicates_remove(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.subcommand not in ("MarkDuplicates", "MarkDuplicatesSpark"):
        return passed()
    declared = call.declared.get(rule.declare.key)
    removed = removes_duplicates(call.argv)
    if declared == "removed" and not removed:
        return fail(
            "duplicates_flagged_not_removed",
            f"you declared {rule.declare.key}=removed but MarkDuplicates only flags the "
            "duplicates (flag 0x400) and writes every read unless REMOVE_DUPLICATES=true: a "
            "count of the output records still includes them.",
            fix=(
                "add REMOVE_DUPLICATES=true (or --REMOVE_DUPLICATES true)",
                "or keep the default and count with samtools view -c -F 0x400",
            ),
            expected="removed",
            got="flagged",
        )
    if declared == "flagged" and removed:
        return fail(
            "duplicates_removed_not_flagged",
            f"you declared {rule.declare.key}=flagged but REMOVE_DUPLICATES=true drops the "
            "duplicates from the output.",
            fix=("drop REMOVE_DUPLICATES=true to keep the duplicates, flagged 0x400",),
            expected="flagged",
            got="removed",
        )
    return passed()
