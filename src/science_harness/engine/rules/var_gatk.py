"""GATK rule of the variant pack (VariantFiltration).

gatk.missing_annotation (T1, call, declaration optional `missing_annotation`)
    `gatk VariantFiltration --filter-expression "QD < 2.0"` marks a record as filtered when the
    expression is TRUE. A record without the annotation makes the expression unevaluable and
    PASSES the filter: it is not filtered, exactly like `bcftools filter -e 'INFO/QD<2'`.
    `--missing-values-evaluate-as-failing` filters such records. Exit status 0; the log has no
    line about the missing values.

Verified here (GATK 4.6.2.0, `--java-options -Xmx512m`), records a (DP 40), b (DP 5), c (no DP),
d (DP 30), filter expression "DP < 10" named lowDP:
    default                                      a PASS, b lowDP, c PASS, d PASS
    --missing-values-evaluate-as-failing         a PASS, b lowDP, c lowDP, d PASS
    "DP >= 10" named okDP                        a okDP, b PASS, c PASS, d okDP
VariantFiltration never removes a record: it sets FILTER. (SelectVariants -select "DP > 10" does
not select record c, so it behaves like bcftools -i; only VariantFiltration is covered.)

What the rule does. It applies to VariantFiltration calls with --filter-expression / -filter. It
takes the attribute names the expressions compare (QD, FS, DP, ...), reads the VCF named by -V /
--variant and looks for records whose INFO lacks one of them while other records have it (an
attribute missing everywhere is not this trap). If there is none it is silent.
Otherwise the call implies `record_kept` (the record is not filtered) without
--missing-values-evaluate-as-failing and `record_removed` with it. Undeclared, the default form
fails (the surprising one); declared, the declaration must agree with the call.
Not seen: --genotype-filter-expression (per-genotype attributes), JEXL calls such as
vc.hasAttribute('QD') that test the missing value themselves (they pass: no attribute is compared),
input given through a stream.

Not shipped as a rule: `gatk SelectVariants -sn s1 -sn s2` keeps the sites at which the selected
samples carry no alternate allele (AC=0 in the output, 4 of 4 records on the test file) and
`--exclude-non-variants` drops them (3); the plink/vcftools/bcftools rules of family F57 cover
the same convention.
"""

from __future__ import annotations

import re

from science_harness.engine.argv import has_option, option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.var_common import fail, read_vcf

JEXL_COMPARE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<=|>=|==|!=|<|>)")
JEXL_NOT_ATTRIBUTES = {"QUAL", "vc", "g", "true", "false", "null"}
MISSING_FLAG = "--missing-values-evaluate-as-failing"


def expression_attributes(expressions: list[str]) -> list[str]:
    names: list[str] = []
    for expression in expressions:
        for name in JEXL_COMPARE.findall(expression):
            if name not in JEXL_NOT_ATTRIBUTES and name not in names:
                names.append(name)
    return names


@checker("gatk_missing_annotation")
def gatk_missing_annotation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    expressions = option_values(call.argv, "--filter-expression", "-filter")
    if call.subcommand != "VariantFiltration" or not expressions:
        return passed()
    names = option_values(call.argv, "-V", "--variant")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    if vcf is None:
        return passed()
    attributes = expression_attributes(expressions)
    lacking: dict[str, int] = {}
    for tag in attributes:
        count = sum(1 for r in vcf.records if r.info.get(tag, ".") in (".", ""))
        if 0 < count < len(vcf.records):  # an attribute the file never has is another problem
            lacking[tag] = count
    if not lacking:
        return passed()
    tag, count = next(iter(lacking.items()))
    implied = "record_removed" if has_option(call.argv, MISSING_FLAG) else "record_kept"
    declared = call.declared.get(rule.declare.key)
    facts = f"{count} of {len(vcf.records)} records of {names[0]} have no {tag} annotation"
    if implied == "record_kept":
        reading = (
            f"a record without {tag} makes the filter expression unevaluable and it PASSES "
            "(it is not filtered)"
        )
        fix = (
            f"add {MISSING_FLAG} to filter the records without {tag}",
            f"or select with bcftools view -i 'INFO/{tag}>=<cutoff>', which drops them",
        )
    else:
        reading = f"{MISSING_FLAG} filters every record that has no {tag} annotation"
        fix = (f"drop {MISSING_FLAG} to leave the records without {tag} unfiltered",)
    if declared is None:
        if implied == "record_removed":
            return passed()
        return fail(
            "missing_values_kept",
            f"{reading}: {facts}.",
            fix=(*fix, f"or --declare {rule.declare.key}=record_kept if that is intended"),
            expected=["record_kept", "record_removed"],
        )
    if declared == implied:
        return passed()
    return fail(
        "declared_missing_mismatch",
        f"you declared {rule.declare.key}={declared} but {reading}: {facts}.",
        fix=fix,
        expected=declared,
        got=implied,
    )
