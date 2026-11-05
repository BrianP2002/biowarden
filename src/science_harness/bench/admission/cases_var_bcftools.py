"""Admission cases for the bcftools rules of the variant pack.

Every case runs the call for real in a benchmark workspace of its family (both twins, implicit
tier), counts or compares the records it prints with `task.truth` (bench.score.answer_matches) and
labels the call from that. Labels never come from a rule.

bcftools.format_filter_scope (family F54): cohort.vcf.gz has six samples; the task counts the
sites at which every sample has GQ >= 20. On the clean twin every site has all GQ on one side of
20, so "any sample" and "every sample" agree; on the faulted twin 8-12 sites are mixed. A call that
tests the FORMAT field with -i (any sample) is right on the clean twin and wrong on the faulted one;
-e with the negated condition and the MIN/F_PASS/N_PASS forms are right on both. Each call is
emitted undeclared, with the declaration that matches the intent of the task (all_samples), and,
for the wrong calls, with the declaration their own options imply. `view -s` evaluates its test on
all samples before subsetting, `query -s` after.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.admission.cases_aln_samtools import Template, emit, run_all
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task

SCOPE_KEY = "format_filter_scope"
VCF = "cohort.vcf.gz"


def _view(name: str, meaning: str, *options: str) -> Template:
    return Template(name, "bcftools", ("view", "-H", *options, VCF), meaning)


SCOPE_TEMPLATES: tuple[Template, ...] = (
    # --- -i with a FORMAT comparison: true when ANY sample satisfies it
    _view("include_fmt", "any_sample", "-i", "FMT/GQ>=20"),
    _view("include_bare", "any_sample", "-i", "GQ>=20"),
    _view("include_format_prefix", "any_sample", "-i", "FORMAT/GQ>=20"),
    _view("include_gt_20", "any_sample", "-i", "FMT/GQ>20"),
    _view("include_and_dp", "any_sample", "-i", "FMT/GQ>=20 && FMT/DP>0"),
    _view("include_long_option", "any_sample", "--include", "FMT/GQ>=20"),
    _view("include_attached", "any_sample", "-iFMT/GQ>=20"),
    _view("include_threads", "any_sample", "--threads", "2", "-i", "FMT/GQ>=20"),
    Template(
        "query_include",
        "bcftools",
        ("query", "-f", "%POS\\n", "-i", "FMT/GQ>=20", VCF),
        "any_sample",
    ),
    Template("filter_include", "bcftools", ("filter", "-i", "GQ>=20", VCF), "any_sample"),
    # --- -e with the negated condition, and aggregates: every sample
    _view("exclude_lt", "all_samples", "-e", "FMT/GQ<20"),
    _view("exclude_bare", "all_samples", "-e", "GQ<20"),
    _view("exclude_long_option", "all_samples", "--exclude", "FMT/GQ<20"),
    _view("min_aggregate", "all_samples", "-i", "MIN(FMT/GQ)>=20"),
    _view("f_pass", "all_samples", "-i", "F_PASS(FMT/GQ>=20)==1"),
    _view("n_pass", "all_samples", "-i", "N_PASS(FMT/GQ>=20)==N_SAMPLES"),
    Template(
        "query_exclude",
        "bcftools",
        ("query", "-f", "%POS\\n", "-e", "FMT/GQ<20", VCF),
        "all_samples",
    ),
    Template("filter_exclude", "bcftools", ("filter", "-e", "FMT/GQ<20", VCF), "all_samples"),
    # --- query subsets before it tests: one sample left, nothing to disagree about
    Template(
        "query_one_sample",
        "bcftools",
        ("query", "-f", "%POS\\n", "-s", "s1", "-i", "FMT/GQ>=20", VCF),
        "any_sample",
    ),
    # --- view tests all samples and subsets afterwards: the test is still any-sample
    _view("view_one_sample", "any_sample", "-s", "s1", "-i", "FMT/GQ>=20"),
)


def count_label(task, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    n = sum(1 for line in proc.stdout.splitlines() if line and not line.startswith("#"))
    return "right" if answer_matches(task.truth, n) else "wrong"


@case_source("bcftools.format_filter_scope")
def scope_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F54", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, SCOPE_TEMPLATES)
        for template, proc in zip(SCOPE_TEMPLATES, procs, strict=True):
            label = count_label(task, proc)
            decls = [{SCOPE_KEY: "all_samples"}]
            if label == "wrong":
                decls.append({SCOPE_KEY: template.meaning})
            emit(out, template, label, work, "F54", seed, twin, decls)
    return out


# ---- bcftools.filter_pass_dot --------------------------------------------------------------------
# Family F55 (implicit tier): variants.vcf.gz has records with FILTER PASS, the names of failed
# filters and, on the faulted twin only, "." (no filter applied). The task lists the IDs of the
# variants that no filter flagged, i.e. PASS and ".". Calls that keep only PASS are right on the
# clean twin and drop the "." records on the faulted one; calls that also accept "." (or exclude
# by filter name) are right on both. Declarations: `keep` is the intent of the task; a call that
# drops also carries the declaration its own options imply (`drop`).

DOT_KEY = "unfiltered_records"


def _ids(template: Template, stdout: str) -> list[str]:
    ids = []
    for line in stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        ids.append(line.split("\t")[2] if template.argv[0] == "view" else line.split("\t")[0])
    return sorted(ids)


def _vq(name: str, meaning: str, *options: str) -> Template:
    return Template(name, "bcftools", ("view", "-H", *options, "variants.vcf.gz"), meaning)


DOT_TEMPLATES: tuple[Template, ...] = (
    # --- only PASS survives
    _vq("apply_pass", "drop", "-f", "PASS"),
    _vq("apply_pass_long", "drop", "--apply-filters", "PASS"),
    _vq("apply_pass_attached", "drop", "-fPASS"),
    _vq("apply_pass_threads", "drop", "--threads", "2", "-f", "PASS"),
    _vq("include_pass", "drop", "-i", 'FILTER="PASS"'),
    _vq("include_pass_double_equals", "drop", "-i", 'FILTER=="PASS"'),
    _vq("exclude_not_pass", "drop", "-e", 'FILTER!="PASS"'),
    _vq("include_pass_and_maf", "drop", "-i", 'FILTER="PASS" && POS>0'),
    Template(
        "query_include_pass",
        "bcftools",
        ("query", "-f", "%ID\\n", "-i", 'FILTER="PASS"', "variants.vcf.gz"),
        "drop",
    ),
    # --- "." is accepted too
    _vq("apply_dot_pass", "keep", "-f", ".,PASS"),
    _vq("apply_pass_dot", "keep", "-f", "PASS,."),
    _vq("apply_long_dot_pass", "keep", "--apply-filters", ".,PASS"),
    _vq("include_pass_or_dot", "keep", "-i", 'FILTER="PASS" || FILTER="."'),
    _vq("exclude_not_pass_not_dot", "keep", "-e", 'FILTER!="PASS" && FILTER!="."'),
    _vq(
        "exclude_by_name", "keep", "-e", 'FILTER="LowQual" || FILTER="LowDP" || FILTER="StrandBias"'
    ),
    Template(
        "query_apply_dot_pass",
        "bcftools",
        ("query", "-f", "%ID\\n", "-i", 'FILTER="PASS" || FILTER="."', "variants.vcf.gz"),
        "keep",
    ),
)


@case_source("bcftools.filter_pass_dot")
def dot_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F55", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, DOT_TEMPLATES)
        for template, proc in zip(DOT_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = (
                    "right" if answer_matches(task.truth, _ids(template, proc.stdout)) else "wrong"
                )
            decls = [{DOT_KEY: "keep"}]
            if label == "wrong":
                decls.append({DOT_KEY: template.meaning})
            emit(out, template, label, work, "F55", seed, twin, decls)
    return out


# ---- bcftools.missing_value_filter ---------------------------------------------------------------
# Family F56 (implicit tier): variants.vcf.gz has an INFO/QD value at every record (clean) or lacks
# it at 8-12 poor records (faulted). The task lists the variants that have QD >= 2. -e with the
# low-QD condition keeps the records without QD (right on the clean twin, wrong on the faulted one);
# -i with the QD >= 2 condition drops them (right on both). `record_removed` is the intent of the
# task; a call that keeps also carries the declaration its own options imply (`record_kept`).

MISSING_KEY = "missing_annotation"


def _qd(name: str, meaning: str, *options: str) -> Template:
    return Template(name, "bcftools", ("view", "-H", *options, "variants.vcf.gz"), meaning)


MISSING_TEMPLATES: tuple[Template, ...] = (
    # --- exclusion of the low values: records without the value stay
    _qd("exclude_info", "record_kept", "-e", "INFO/QD<2"),
    _qd("exclude_bare", "record_kept", "-e", "QD<2"),
    _qd("exclude_long", "record_kept", "--exclude", "INFO/QD<2"),
    _qd("exclude_float", "record_kept", "-e", "INFO/QD<2.0"),
    _qd("exclude_le", "record_kept", "-e", "INFO/QD<=1.99"),
    _qd("exclude_threads", "record_kept", "--threads", "2", "-e", "INFO/QD<2"),
    Template(
        "query_exclude",
        "bcftools",
        ("query", "-f", "%ID\\n", "-e", "INFO/QD<2", "variants.vcf.gz"),
        "record_kept",
    ),
    # --- selection of the good values: records without the value go
    _qd("include_info", "record_removed", "-i", "INFO/QD>=2"),
    _qd("include_bare", "record_removed", "-i", "QD>=2"),
    _qd("include_long", "record_removed", "--include", "INFO/QD>=2"),
    _qd("include_gt", "record_removed", "-i", "INFO/QD>1.99"),
    _qd("include_two_terms", "record_removed", "-i", "INFO/QD>=2 && INFO/DP>10"),
    _qd("exclude_or_missing", "record_removed", "-e", 'INFO/QD<2 || INFO/QD="."'),
    Template(
        "query_include",
        "bcftools",
        ("query", "-f", "%ID\\n", "-i", "INFO/QD>=2", "variants.vcf.gz"),
        "record_removed",
    ),
)


@case_source("bcftools.missing_value_filter")
def missing_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F56", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, MISSING_TEMPLATES)
        for template, proc in zip(MISSING_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = (
                    "right" if answer_matches(task.truth, _ids(template, proc.stdout)) else "wrong"
                )
            decls = [{MISSING_KEY: "record_removed"}]
            if label == "wrong":
                decls.append({MISSING_KEY: template.meaning})
            emit(out, template, label, work, "F56", seed, twin, decls)
    return out


# ---- bcftools.subset_monomorphic -----------------------------------------------------------------
# Family F57 (implicit tier): cohort.vcf.gz has eight samples; group_b.txt lists four. The task
# counts the variant sites at which at least one group-B sample carries an alternate allele. The
# faulted twin adds 6-10 sites that only the other samples carry. `bcftools view -S/-s` alone
# writes every site (right on the clean twin, wrong on the faulted one); with -c 1, --min-ac 1 or
# an AC test the monomorphic sites go (right on both). The declaration of the task's intent is
# `variant_sites_only`; a call that keeps all sites also carries `keep_all_sites`.

SUBSET_KEY = "subset_sites"


def _sv(name: str, meaning: str, *options: str) -> Template:
    return Template(name, "bcftools", ("view", "-H", *options, "cohort.vcf.gz"), meaning)


def subset_templates(group_b: list[str]) -> list[Template]:
    names = ",".join(group_b)
    return [
        _sv("samples_file", "keep_all_sites", "-S", "group_b.txt"),
        _sv("samples_list", "keep_all_sites", "-s", names),
        _sv("samples_file_long", "keep_all_sites", "--samples-file", "group_b.txt"),
        _sv("samples_list_long", "keep_all_sites", "--samples", names),
        _sv("samples_force", "keep_all_sites", "--force-samples", "-S", "group_b.txt"),
        _sv("samples_threads", "keep_all_sites", "--threads", "2", "-S", "group_b.txt"),
        _sv("samples_snps", "keep_all_sites", "-v", "snps", "-S", "group_b.txt"),
        _sv("samples_no_update", "keep_all_sites", "-I", "-S", "group_b.txt"),
        _sv("min_ac", "variant_sites_only", "-S", "group_b.txt", "-c", "1"),
        _sv("min_ac_attached", "variant_sites_only", "-S", "group_b.txt", "-c1"),
        _sv("min_ac_long", "variant_sites_only", "--min-ac", "1", "-S", "group_b.txt"),
        _sv("min_ac_list", "variant_sites_only", "-s", names, "-c", "1"),
        _sv("exclude_ac0", "variant_sites_only", "-S", "group_b.txt", "-e", "AC==0"),
        _sv("include_ac", "variant_sites_only", "-S", "group_b.txt", "-i", "AC>0"),
        _sv("min_ac_nref", "variant_sites_only", "-S", "group_b.txt", "-c", "1:nref"),
    ]


@case_source("bcftools.subset_monomorphic")
def subset_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F57", seed, twin, work, tools, "implicit", "free")
        group_b = (work / "group_b.txt").read_text().split()
        templates = subset_templates(group_b)
        procs = run_all(tools, work, templates)
        for template, proc in zip(templates, procs, strict=True):
            label = count_label(task, proc)
            decls = [{SUBSET_KEY: "variant_sites_only"}]
            if label == "wrong":
                decls.append({SUBSET_KEY: template.meaning})
            emit(out, template, label, work, "F57", seed, twin, decls)
    return out


# ---- bcftools.isec_representation ----------------------------------------------------------------
# Family F62 (implicit tier): caller_a.vcf.gz and caller_b.vcf.gz share every variant; caller B
# writes the indels right-aligned on the faulted twin. The task counts the variants both callers
# called. `bcftools isec -n=2` on the files as they are finds all of them on the clean twin and the
# SNVs only on the faulted one; after `bcftools norm -f ref.fa` on both (done here with the calls
# below as the preparatory step) it finds all. The declaration of the task's intent is
# `normalized`; a call on the raw files also carries the one its options imply (`as_written`).

ISEC_KEY = "isec_matching"


def _isec(name: str, meaning: str, a: str, b: str, *options: str) -> Template:
    return Template(name, "bcftools", ("isec", "-n=2", "-w1", *options, a, b), meaning)


ISEC_TEMPLATES: tuple[Template, ...] = (
    _isec("raw", "as_written", "caller_a.vcf.gz", "caller_b.vcf.gz"),
    _isec("raw_some", "as_written", "caller_a.vcf.gz", "caller_b.vcf.gz", "-c", "some"),
    _isec("raw_indels", "as_written", "caller_a.vcf.gz", "caller_b.vcf.gz", "-c", "indels"),
    _isec("raw_all", "as_written", "caller_a.vcf.gz", "caller_b.vcf.gz", "-c", "all"),
    _isec("raw_swapped", "as_written", "caller_b.vcf.gz", "caller_a.vcf.gz"),
    _isec("raw_threads", "as_written", "caller_a.vcf.gz", "caller_b.vcf.gz", "--threads", "2"),
    _isec("norm", "normalized", "caller_a.norm.vcf.gz", "caller_b.norm.vcf.gz"),
    _isec("norm_some", "normalized", "caller_a.norm.vcf.gz", "caller_b.norm.vcf.gz", "-c", "some"),
    _isec("norm_swapped", "normalized", "caller_b.norm.vcf.gz", "caller_a.norm.vcf.gz"),
)


@case_source("bcftools.isec_representation")
def isec_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    from science_harness.bench.tasks.extra.f62_isec_repr import normalised

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F62", seed, twin, work, tools, "implicit", "free")
        normalised(work, tools, "caller_a.vcf.gz")
        normalised(work, tools, "caller_b.vcf.gz")
        procs = run_all(tools, work, ISEC_TEMPLATES)
        for template, proc in zip(ISEC_TEMPLATES, procs, strict=True):
            label = count_label(task, proc)
            decls = [{ISEC_KEY: "normalized"}]
            if label == "wrong":
                decls.append({ISEC_KEY: template.meaning})
            emit(out, template, label, work, "F62", seed, twin, decls)
    return out


# ---- bcftools.merge_absent_sites -----------------------------------------------------------------
# Family F63 (implicit tier): six single-sample VCFs; the clean twin's files list every site (0/0
# where the sample has no ALT), the faulted twin's are variants-only. The task reports the ALT
# allele frequency over all six samples. `bcftools merge` then `+fill-tags -- -t AF` is right on the
# clean twin and divides by the called alleles on the faulted one; `merge -0` is right on both.
# The declaration of the task's intent is `reference`; a call without -0 also carries `missing`.

ABSENT_KEY = "absent_sites"
SAMPLE_FILES = tuple(f"sample{i}.vcf.gz" for i in range(1, 7))


def _merge(name: str, meaning: str, *options: str) -> Template:
    return Template(
        name, "bcftools", ("merge", *options, "-o", f"m_{name}.vcf", *SAMPLE_FILES), meaning
    )


MERGE_TEMPLATES: tuple[Template, ...] = (
    _merge("plain", "missing"),
    _merge("threads", "missing", "--threads", "2"),
    _merge("force", "missing", "--force-samples"),
    _merge("ov", "missing", "-O", "v"),
    _merge("info_rules", "missing", "-i", "-"),
    _merge("zero", "reference", "-0"),
    _merge("zero_long", "reference", "--missing-to-ref"),
    _merge("zero_threads", "reference", "-0", "--threads", "2"),
    _merge("zero_ov", "reference", "-0", "-O", "v"),
)


def _merged_label(task, tools: ToolEnv, work: Path, template: Template, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    name = template.argv[template.argv.index("-o") + 1]
    tagged = f"t_{template.name}.vcf"
    step = tools.run(["bcftools", "+fill-tags", name, "-o", tagged, "--", "-t", "AF"], work)
    if step.returncode != 0:
        return "wrong"
    query = tools.run(["bcftools", "query", "-f", "%ID\t%AF\n", tagged], work)
    freqs = {}
    for line in query.stdout.splitlines():
        ident, af = line.split("\t")
        freqs[ident] = float(af) if af != "." else 0.0
    return "right" if answer_matches(task.truth, freqs) else "wrong"


@case_source("bcftools.merge_absent_sites")
def merge_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F63", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, MERGE_TEMPLATES)
        for template, proc in zip(MERGE_TEMPLATES, procs, strict=True):
            label = _merged_label(task, tools, work, template, proc)
            decls = [{ABSENT_KEY: "reference"}]
            if label == "wrong":
                decls.append({ABSENT_KEY: template.meaning})
            emit(out, template, label, work, "F63", seed, twin, decls)
    return out
