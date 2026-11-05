"""vcftools rules of the variant pack (the evidence of each rule is quoted in its own section).

vcftools.bed_header, vcftools.keep_filtered_dot, vcftools.subset_monomorphic and the
var.missing_threshold spec entry (a T0 built-in, see engine/specs/var_vcftools.yaml).

vcftools.bed_header (T1, call, no declaration)
    `vcftools --bed FILE` and `--exclude-bed FILE` treat the first line of the file as a header
    and skip it (the manual: "The BED file is expected to have a header line"). A BED file written
    the usual way, without a header, loses its first interval. Exit status 0; stderr says only
    "Read N BED file entries." (the skipped line is counted in N) and the usual "After filtering,
    kept K out of a possible M Sites".

Verified here (vcftools 0.1.17, bcftools 1.24), six-site VCF (chr1:100, chr1:101, chr1:200,
chr1:300, chr2:100, chr2:150):
    BED "chr1 99 100 / chr2 99 100 / chr2 149 150" (no header): kept 2 sites (chr2:100, chr2:150),
        "Read 3 BED file entries."; the chr1:100 interval was dropped
    the same lines after "chrom start end": kept 3 sites; after "track name=x": kept 3 sites
    a one-line BED "chr1 99 100": kept 0 sites ("Read 1 BED file entries." and "No data left")
    `--exclude-bed` with "chr1 99 100 / chr2 99 100" (no header) removed only chr2:100: the first
        interval excluded nothing
`bcftools view -R` and `bedtools intersect` read the same headerless file completely.

What the rule does. It looks at the first line of every file given to --bed / --exclude-bed. If
it is an interval (tab- or space-separated, at least three fields, integer start and end with
start <= end) it fails: that line is not a header, so vcftools drops that interval. A first line
that is not an interval (a '#' comment, "track name=...", "chrom start end") passes, an empty file
passes, a file that cannot be read gives `unknown`. No declaration: nothing the agent can state
makes a headerless file safe; the fix is to give the file a header line.
Not seen: the BED given as a process substitution or from stdin.
"""

from __future__ import annotations

from science_harness.engine.argv import has_option, option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.var_common import (
    Vcf,
    fail,
    first_line,
    is_interval_line,
    read_list,
    read_vcf,
    subset_verdict,
    unknown,
)


@checker("vcftools_bed_header")
def vcftools_bed_header(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    names = option_values(call.argv, "--bed", "--exclude-bed")
    unreadable: list[str] = []
    for name in names:
        line = first_line(ctx, name)
        if line is None:
            if not ctx.resolve(name).is_file():
                unreadable.append(name)
            continue  # an empty file has no first line to lose
        if is_interval_line(line):
            shown = line.replace("\t", " ")
            return fail(
                "bed_first_line_skipped",
                f"vcftools reads the first line of {name} as a header and skips it, but that line "
                f"is the interval '{shown}', so this interval is silently ignored "
                "(vcftools counts the skipped line in 'Read N BED file entries').",
                fix=(
                    f"put a header line first: (printf '#chrom\\tstart\\tend\\n'; cat {name}) "
                    f"> {name}.hdr, then use --bed {name}.hdr",
                    f"or read every line with bcftools view -R {name} / bedtools intersect "
                    f"-b {name}",
                ),
                expected="a header line first",
                got=shown,
            )
    if unreadable:
        return unknown(
            "bed_not_found",
            f"{', '.join(unreadable)} cannot be read, so its first line cannot be checked "
            "(vcftools skips it as a header).",
            fix=("make sure the first line of the BED file is a header line",),
        )
    return passed()


# ---- vcftools.keep_filtered_dot ------------------------------------------------------------------
#
# vcftools.keep_filtered_dot (T1, call, declaration optional `unfiltered_records`)
#     vcftools ignores the FILTER column unless asked. `--keep-filtered PASS` keeps only the records
#     whose FILTER is PASS and drops those whose FILTER is "." (never filtered); there is no value
#     that keeps "." with --keep-filtered. `--remove-filtered-all` (and `--remove-filtered NAME`)
#     keep PASS and "." records. Exit 0; stderr says only "kept N out of a possible M Sites".
#
# Verified here (vcftools 0.1.17), records a PASS, b LowQual, c PASS, d PASS, e ".", f PASS:
#     (no filter option)                          a b c d e f    (LowQual b is kept)
#     --keep-filtered PASS                        a c d f        (e dropped)
#     --keep-filtered PASS --keep-filtered .      a c d f        --keep-filtered .   (nothing)
#     --remove-filtered-all                       a c d e f
#     --remove-filtered LowQual                   a c d e f
# What the rule does. A call with --keep-filtered implies `drop` for the "." records, one with
# --remove-filtered-all or --remove-filtered implies `keep`. It reads the VCF of --vcf/--gzvcf; if
# no record has FILTER "." it is silent; otherwise it applies the same declaration logic as
# bcftools.filter_pass_dot. Not seen: --bcf input, files that do not exist yet.

FILTER_KEY = "unfiltered_records"


def vcftools_dot_selection(call: Call) -> str | None:
    keeps = option_values(call.argv, "--keep-filtered")
    removes = option_values(call.argv, "--remove-filtered") or has_option(
        call.argv, "--remove-filtered-all"
    )
    if keeps and not removes:
        return "drop"
    if removes and not keeps:
        return "keep"
    return None


@checker("vcftools_keep_filtered_dot")
def vcftools_keep_filtered_dot(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    selection = vcftools_dot_selection(call)
    if selection is None:
        return passed()
    names = option_values(call.argv, "--vcf", "--gzvcf")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    dots = [r for r in vcf.records if r.filt == "."] if vcf else []
    if not dots:
        return passed()
    declared = call.declared.get(rule.declare.key)
    facts = (
        f"{len(dots)} of {len(vcf.records)} records of {names[0]} have FILTER '.' "  # type: ignore[union-attr]
        f"(for example {dots[0].id or dots[0].chrom + ':' + str(dots[0].pos)})"
    )
    if selection == "drop" and declared is None:
        return fail(
            "dot_filter_dropped",
            f"--keep-filtered keeps only the records named, so it drops the records whose FILTER "
            f"is '.' (no filter was applied to them): {facts}.",
            fix=(
                "use --remove-filtered-all (keeps PASS and '.' records, drops every named filter)",
                "or --declare unfiltered_records=drop if dropping them is intended",
            ),
            expected=["keep", "drop"],
        )
    if declared is None or declared == selection:
        return passed()
    if declared == "keep":
        return fail(
            "declared_keep_but_dropped",
            f"you declared {rule.declare.key}=keep but --keep-filtered drops the records whose "
            f"FILTER is '.': {facts}.",
            fix=("use --remove-filtered-all instead of --keep-filtered",),
            expected="keep",
            got="drop",
        )
    return fail(
        "declared_drop_but_kept",
        f"you declared {rule.declare.key}=drop but this call keeps the records whose FILTER is "
        f"'.': {facts}.",
        fix=("use --keep-filtered PASS to drop the records with FILTER '.'",),
        expected="drop",
        got="keep",
    )


# ---- vcftools.subset_monomorphic -----------------------------------------------------------------
#
# vcftools.subset_monomorphic (T1, call, declaration optional `subset_sites`)
#     `vcftools --keep/--indv/--remove ... --recode` writes every site of the input for the kept
#     individuals, also those where nobody carries an alternate allele; `--mac 1`, `--maf X`
#     (X > 0), `--non-ref-ac 1` or `--non-ref-ac-any 1` drop them. The log line "kept N out of a
#     possible M Sites" counts them.
#
# Verified here (vcftools 0.1.17), six sites a..f, `--keep` with s1 and s2: 6 sites kept; with
# `--mac 1`, `--maf 0.0001`, `--non-ref-ac 1` or `--non-ref-ac-any 1` 5 (site e, hom-reference in
# every sample, is gone). `--keep` reads the first token of each line, a file name that has an
# unknown sample is accepted without a message ("kept 1 out of 4 Individuals").
# What the rule does. For calls with --keep/--indv/--remove/--remove-indv it selects the
# individuals, reads the VCF of --vcf/--gzvcf and counts the sites without an alternate allele in
# them; silent when there is none or the call has one of the filters above. Same declaration logic
# as bcftools.subset_monomorphic.
# Not seen: --bcf input, individuals given by --keep-indv-style options that vcftools does not have.

SUBSET_ONLY_FILTERS = ("--mac", "--non-ref-ac", "--non-ref-ac-any")


def vcftools_selected(call: Call, ctx: Ctx, vcf: Vcf) -> list[int]:
    keep_names: list[str] = list(option_values(call.argv, "--indv"))
    for name in option_values(call.argv, "--keep"):
        keep_names += read_list(ctx, name) or []
    drop_names: list[str] = list(option_values(call.argv, "--remove-indv"))
    for name in option_values(call.argv, "--remove"):
        drop_names += read_list(ctx, name) or []
    chosen = [
        i
        for i, sample in enumerate(vcf.samples)
        if (not keep_names or sample in keep_names) and sample not in drop_names
    ]
    return chosen


def vcftools_drops_monomorphic(call: Call) -> bool:
    for option in SUBSET_ONLY_FILTERS:
        for value in option_values(call.argv, option):
            if value.isdigit() and int(value) >= 1:
                return True
    for option in ("--maf", "--non-ref-af", "--non-ref-af-any"):
        for value in option_values(call.argv, option):
            try:
                if float(value) > 0:
                    return True
            except ValueError:
                continue
    return False


@checker("vcftools_subset_monomorphic")
def vcftools_subset_monomorphic(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    names = option_values(call.argv, "--vcf", "--gzvcf")
    if len(names) != 1:
        return passed()
    vcf = read_vcf(ctx, names[0])
    if vcf is None:
        return passed()
    which = vcftools_selected(call, ctx, vcf)
    if not which or len(which) == len(vcf.samples):
        return passed()
    return subset_verdict(
        call,
        rule.declare.key,
        "vcftools",
        names[0],
        vcf,
        which,
        vcftools_drops_monomorphic(call),
        fix=("add --mac 1 (or --non-ref-ac-any 1) to drop the sites without an alternate allele",),
    )
