"""PLINK rules of the variant pack (plink 1.9 and plink2 on VCF input).

plink.subset_monomorphic (T1, call, declaration optional `subset_sites`)
    `plink --vcf F --keep K --make-bed` writes every variant for the kept samples, also those at
    which they are all homozygous for one allele; `--mac 1` (or `--maf X`) drops them.
    Verified here (plink 1.9, plink2 2.0.0-a.7), six variants and four people, two kept: 6
    variants written (the log says "6 variants and 2 people pass filters and QC"), with `--mac 1`
    5 ("1 variant removed due to minor allele threshold(s)"). A sample absent from the VCF in
    --keep is ignored.

plink.allele_order (T1, call, declaration optional `plink_a1_allele`)
    plink 1.9 gives A1 to the MINOR allele of the data when it imports a VCF (A2 is the major
    one); `--keep-allele-order` makes A1 the ALT allele. `--freq` reports the frequency of A1 in
    the MAF column, `--assoc/--linear/--logistic` use A1 as the tested allele, `--score` and
    `--recode A` count A1. Where ALT is the major allele, A1 is REF. No message.
    Verified here (plink 1.9), ten samples, twelve sites of ALT frequency 0.15-0.85 (v02 0.85, v03
    0.55, v04 0.65, v05 0.8, v07 0.8): default `--freq` prints A1/A2 G/A for the sites with ALT
    frequency < 0.5 and A/G with MAF 0.15, 0.45, 0.35, 0.2, 0.2 for the five above 0.5;
    `--keep-allele-order` prints G/A everywhere with MAF equal to the ALT frequency
    (0.85, 0.55, 0.65, 0.8, 0.8). A tie (ALT frequency 0.5) keeps A1 = ALT. plink2 keeps REF/ALT
    (`--freq` writes REF, ALT and ALT_FREQS).
    The rule looks at the options that report or count A1 (REPORT_OPTIONS). `--make-bed` and
    `--make-just-bim` write a fileset and print no statistic of A1, so a call that only filters and
    writes one is not judged.

plink.vcf_multiallelic (T1, call, declaration optional `vcf_multiallelic`)
    plink 1.9 `--vcf` reads a site with two ALT alleles as a biallelic one: the first ALT is the
    alternate allele and every genotype call that uses the second ALT becomes missing. The log has
    no warning ("6 variants loaded"); `--biallelic-only` skips such sites ("(1 variant skipped.)").
    Verified here (plink 1.9): site c (G -> A,T) with genotypes 0/1, 0/2, 1/1, 0/1 is written with
    the 0/2 call missing (total genotyping rate 0.917 instead of 0.958 with the real ./. only);
    `--geno 0.1` then removes c (25% "missing") although the VCF has no missing call there, and
    `plink2 --geno 0.1` keeps it. plink2 `--make-bed` stops on multiallelic variants (loud); plink2
    `--freq` writes ALT_FREQS "0.5,0.125" for c.
"""

from __future__ import annotations

from biowarden.engine.argv import has_option, option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.var_common import (
    Vcf,
    fail,
    read_list,
    read_vcf,
    subset_verdict,
)

OUTPUT_OPTIONS = (
    "--make-bed",
    "--make-just-bim",
    "--make-pgen",
    "--make-just-pvar",
    "--recode",
    "--write-snplist",
    "--freq",
    "--freqx",
)
# the options whose output reports or counts A1; --make-bed and --make-just-bim write a fileset and
# print no statistic of A1, so a call that only filters and writes one is left alone (E-12: they
# made the rule fire on every correct F59 call)
REPORT_OPTIONS = (
    "--freq",
    "--freqx",
    "--assoc",
    "--linear",
    "--logistic",
    "--score",
    "--recode",
)
ALLELE_CONTROL = ("--keep-allele-order", "--a1-allele", "--a2-allele", "--real-ref-alleles")


# ---- plink.subset_monomorphic --------------------------------------------------------------------


def plink_drops_monomorphic(call: Call) -> bool:
    for value in option_values(call.argv, "--mac"):
        if value.isdigit() and int(value) >= 1:
            return True
    for value in option_values(call.argv, "--maf"):
        try:
            if float(value) > 0:
                return True
        except ValueError:
            continue
    return False


def iid_of(line: str) -> str:
    fields = line.split()
    return fields[1] if len(fields) > 1 else fields[0]


def plink_selected(call: Call, ctx: Ctx, vcf: Vcf) -> list[int]:
    keep: list[str] = []
    for name in option_values(call.argv, "--keep"):
        keep += [iid_of(x) for x in read_list(ctx, name, first_field_only=False) or []]
    drop: list[str] = []
    for name in option_values(call.argv, "--remove"):
        drop += [iid_of(x) for x in read_list(ctx, name, first_field_only=False) or []]
    return [
        i
        for i, sample in enumerate(vcf.samples)
        if (not keep_set_given(call) or sample in keep) and sample not in drop
    ]


def keep_set_given(call: Call) -> bool:
    return bool(option_values(call.argv, "--keep"))


@checker("plink_subset_monomorphic")
def plink_subset_monomorphic(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if not any(has_option(call.argv, option) for option in OUTPUT_OPTIONS):
        return passed()
    names = option_values(call.argv, "--vcf")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    if vcf is None:
        return passed()
    which = plink_selected(call, ctx, vcf)
    if not which or len(which) == len(vcf.samples):
        return passed()
    return subset_verdict(
        call,
        rule.declare.key,
        call.tool,
        names[0],
        vcf,
        which,
        plink_drops_monomorphic(call),
        fix=("add --mac 1 to drop the variants at which the kept samples carry one allele only",),
        fixed_counts=True,
    )


# ---- plink.allele_order --------------------------------------------------------------------------


def alt_major_sites(vcf: Vcf) -> list[tuple[str, float]]:
    """(label, ALT frequency) of the sites where the first ALT allele is more frequent than REF."""
    found: list[tuple[str, float]] = []
    for rec in vcf.records:
        ref = alt = 0
        for i in range(len(vcf.samples)):
            for a in rec.genotype(i):
                if a == 0:
                    ref += 1
                elif a == 1:
                    alt += 1
        if alt > ref:
            found.append((rec.id or f"{rec.chrom}:{rec.pos}", alt / (alt + ref)))
    return found


@checker("plink_allele_order")
def plink_allele_order(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.tool != "plink" or not any(has_option(call.argv, o) for o in REPORT_OPTIONS):
        return passed()
    if any(has_option(call.argv, o) for o in ALLELE_CONTROL[1:]):
        return passed()
    names = option_values(call.argv, "--vcf")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    if vcf is None:
        return passed()
    major = alt_major_sites(vcf)
    if not major:
        return passed()
    implied = "alt_allele" if has_option(call.argv, "--keep-allele-order") else "minor_allele"
    declared = call.declared.get(rule.declare.key)
    label, freq = major[0]
    facts = (
        f"at {len(major)} of {len(vcf.records)} sites of {names[0]} the ALT allele is the major "
        f"one (for example {label}, ALT frequency {freq:.2f}), so there A1 is the REF allele"
    )
    fix = ("add --keep-allele-order (A1 = ALT), or read ALT_FREQS from plink2 --freq",)
    if declared is None:
        if implied == "alt_allele":
            return passed()
        return fail(
            "a1_is_minor_allele",
            f"plink 1.9 sets A1 to the minor allele of the data, not to ALT: {facts}; the MAF "
            "column and the tested or counted allele are those of A1.",
            fix=(
                *fix,
                f"or --declare {rule.declare.key}=minor_allele if A1 = minor allele is intended",
            ),
            expected=["minor_allele", "alt_allele"],
        )
    if declared == implied:
        return passed()
    if declared == "alt_allele":
        return fail(
            "declared_alt_but_minor",
            f"you declared {rule.declare.key}=alt_allele but without --keep-allele-order plink 1.9 "
            f"sets A1 to the minor allele: {facts}.",
            fix=fix,
            expected="alt_allele",
            got="minor_allele",
        )
    return fail(
        "declared_minor_but_alt",
        f"you declared {rule.declare.key}=minor_allele but --keep-allele-order makes A1 the ALT "
        f"allele: {facts}.",
        fix=("drop --keep-allele-order to let plink put the minor allele in A1",),
        expected="minor_allele",
        got="alt_allele",
    )


# ---- plink.vcf_multiallelic ----------------------------------------------------------------------


def second_alt_calls(vcf: Vcf) -> tuple[int, int, str]:
    """(multiallelic sites with a call of the second or later ALT, such calls, first such site)."""
    sites = calls = 0
    first = ""
    for rec in vcf.records:
        if len(rec.alts) < 2:
            continue
        n = sum(1 for i in range(len(vcf.samples)) for a in rec.genotype(i) if a and a >= 2)
        if n:
            sites += 1
            calls += n
            first = first or (rec.id or f"{rec.chrom}:{rec.pos}")
    return sites, calls, first


@checker("plink_vcf_multiallelic")
def plink_vcf_multiallelic(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.tool != "plink":
        return passed()
    names = option_values(call.argv, "--vcf")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    if vcf is None:
        return passed()
    sites, calls, first = second_alt_calls(vcf)
    if not sites:
        return passed()
    implied = "biallelic_only" if has_option(call.argv, "--biallelic-only") else "first_alt_only"
    declared = call.declared.get(rule.declare.key)
    facts = (
        f"{names[0]} has {sites} multiallelic sites with {calls} genotype calls of a second "
        f"alternate allele (for example at {first})"
    )
    fix = (
        "run plink2 for the same options (it keeps every allele: plink2 --vcf ... --geno 0.1)",
        "or skip the multiallelic sites on purpose with --biallelic-only",
    )
    if declared is None:
        if implied == "biallelic_only":
            return passed()
        return fail(
            "second_alt_set_missing",
            f"plink 1.9 --vcf reads a multiallelic site as biallelic (first ALT only) and sets "
            f"every call of another ALT allele to missing, without a warning: {facts}.",
            fix=(
                *fix,
                f"or --declare {rule.declare.key}=first_alt_only if that loss is acceptable",
            ),
            expected=["first_alt_only", "biallelic_only"],
        )
    if declared == implied:
        return passed()
    if declared == "biallelic_only":
        return fail(
            "declared_biallelic_only_without_option",
            f"you declared {rule.declare.key}=biallelic_only but the call has no --biallelic-only, "
            f"so the multiallelic sites are read as first-ALT-only: {facts}.",
            fix=("add --biallelic-only",),
            expected="biallelic_only",
            got="first_alt_only",
        )
    return fail(
        "declared_first_alt_only_with_option",
        f"you declared {rule.declare.key}=first_alt_only but --biallelic-only skips the "
        f"multiallelic sites: {facts}.",
        fix=("drop --biallelic-only to keep the sites as first-ALT-only",),
        expected="first_alt_only",
        got="biallelic_only",
    )
