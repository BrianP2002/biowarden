"""Admission cases for the vcftools rules of the variant pack.

Every case runs the call for real in a benchmark workspace of its family (both twins, implicit
tier), turns the output into the task's answer and compares it with `task.truth`
(bench.score.answer_matches). Labels never come from a rule.

vcftools.bed_header (family F53): targets.bed starts with a header line on the clean twin and with
an interval on the faulted twin; the task is the IDs of the variants inside the intervals. A
`vcftools --bed targets.bed` call is right on the clean twin and loses the first interval on the
faulted one; a call that gives vcftools a copy of the file with a header (targets_hdr.bed, made
here with the printf/cat idiom the rule suggests) is right on both. `--exclude-bed` answers another
question (the variants outside), so it is wrong on both twins. The rule has no declaration: the same
call is emitted in the declared and the undeclared cell.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.admission.cases_aln_samtools import Template, emit, run_all
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call


def recode_ids(text: str) -> list[str]:
    return sorted(
        line.split("\t")[2] for line in text.splitlines() if line and not line.startswith("#")
    )


def _bed(name: str, *args: str, bed: str = "targets.bed", out: str = "") -> Template:
    argv = ["--gzvcf", "variants.vcf.gz", *args, "--recode"]
    argv += ["--out", out] if out else ["--stdout"]
    return Template(name, "vcftools", tuple(argv), "", out)


BED_TEMPLATES: tuple[Template, ...] = (
    # --- the file is used as it is: right only when its first line is a header
    _bed("bed", "--bed", "targets.bed"),
    _bed("bed_equals", "--bed=targets.bed"),
    _bed("bed_out", "--bed", "targets.bed", out="o_bed"),
    _bed("bed_with_maf", "--bed", "targets.bed", "--maf", "0"),
    _bed("bed_min_q", "--bed", "targets.bed", "--minQ", "10"),
    _bed("bed_recode_info", "--bed", "targets.bed", "--recode-INFO-all"),
    _bed("bed_keep_pass", "--bed", "targets.bed", "--remove-filtered-all"),
    # --- a header line in front of the intervals: right on both twins
    _bed("bed_header_copy", "--bed", "targets_hdr.bed"),
    _bed("bed_header_copy_out", "--bed", "targets_hdr.bed", out="o_hdr"),
    _bed("bed_header_copy_equals", "--bed=targets_hdr.bed"),
    # --- another question (the variants outside the intervals): wrong on both twins
    _bed("exclude_bed", "--exclude-bed", "targets.bed"),
    _bed("exclude_bed_header_copy", "--exclude-bed", "targets_hdr.bed"),
)


def _label(task, template: Template, work: Path, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    text = (work / f"{template.note}.recode.vcf").read_text() if template.note else proc.stdout
    return "right" if answer_matches(task.truth, recode_ids(text)) else "wrong"


@case_source("vcftools.bed_header")
def bed_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F53", seed, twin, work, tools, "implicit", "free")
        bed = (work / "targets.bed").read_text()
        header = "#chrom\tstart\tend\n"
        (work / "targets_hdr.bed").write_text(bed if bed.startswith("#") else header + bed)
        procs = run_all(tools, work, BED_TEMPLATES)
        for template, proc in zip(BED_TEMPLATES, procs, strict=True):
            label = _label(task, template, work, proc)
            for declared in (True, False):  # no declaration exists: both cells hold the same call
                call = Call(template.tool, template.argv, work, {})
                out.append(Case(call, label, declared, "F53", seed, twin, template.name))
    return out


# ---- vcftools.keep_filtered_dot ------------------------------------------------------------------
# Family F55 (implicit tier): the IDs of the variants no filter flagged (PASS and "."). vcftools
# runs with --recode --stdout; `--keep-filtered PASS` is right on the clean twin (no "." records)
# and drops them on the faulted one, `--remove-filtered-all` and `--remove-filtered NAME` keep them.

FILTER_KEY = "unfiltered_records"


def _vt(name: str, meaning: str, *options: str) -> Template:
    argv = ("--gzvcf", "variants.vcf.gz", *options, "--recode", "--stdout")
    return Template(name, "vcftools", argv, meaning)


FILTER_TEMPLATES: tuple[Template, ...] = (
    _vt("keep_pass", "drop", "--keep-filtered", "PASS"),
    _vt("keep_pass_maf", "drop", "--keep-filtered", "PASS", "--maf", "0"),
    _vt("keep_pass_dot", "drop", "--keep-filtered", "PASS", "--keep-filtered", "."),
    _vt("keep_pass_minq", "drop", "--keep-filtered", "PASS", "--minQ", "0"),
    _vt("remove_all", "keep", "--remove-filtered-all"),
    _vt("remove_all_maf", "keep", "--remove-filtered-all", "--maf", "0"),
    _vt(
        "remove_named",
        "keep",
        "--remove-filtered",
        "LowQual",
        "--remove-filtered",
        "LowDP",
        "--remove-filtered",
        "StrandBias",
    ),
)


@case_source("vcftools.keep_filtered_dot")
def filter_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F55", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, FILTER_TEMPLATES)
        for template, proc in zip(FILTER_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = "right" if answer_matches(task.truth, recode_ids(proc.stdout)) else "wrong"
            decls = [{FILTER_KEY: "keep"}]
            if label == "wrong":
                decls.append({FILTER_KEY: template.meaning})
            emit(out, template, label, work, "F55", seed, twin, decls)
    return out


# ---- var.missing_threshold -----------------------------------------------------------------------
# Family F60 (implicit tier): the task keeps the sites with at most 10% missing genotypes and
# counts them. vcftools --max-missing is a completeness (0.9 is right; 0.1 also keeps the graded
# sites of the faulted twin); plink --geno is a missing rate (0.1 is right). Each call is run for
# real and the number of sites written is compared with the truth. The declaration is the meaning
# of the number the call passes (completeness or missing_rate), required by the rule: every call is
# emitted undeclared (it fails by design), with the meaning its option has, and, when it is wrong,
# with the other meaning (the author thought of the number as the opposite quantity).

MISSING_KEY = "missing_threshold_meaning"
PLINK_MEM = ("--memory", "1000", "--silent")


def _vt_missing(name: str, value: str, *extra: str) -> Template:
    argv = ("--gzvcf", "cohort.vcf.gz", "--max-missing", value, *extra, "--recode", "--stdout")
    return Template(name, "vcftools", argv, "completeness")


def _plink_geno(name: str, tool: str, value: str, *extra: str) -> Template:
    make = "--make-just-bim" if tool == "plink" else "--make-just-pvar"
    argv = (*PLINK_MEM, "--vcf", "cohort.vcf.gz", "--geno", value, *extra, make)
    argv += ("--out", f"o_{name}")
    return Template(name, tool, argv, "missing_rate", f"o_{name}")


MISSING_TEMPLATES: tuple[Template, ...] = (
    _vt_missing("vcftools_0_9", "0.9"),
    _vt_missing("vcftools_0_95", "0.95"),
    _vt_missing("vcftools_0_1", "0.1"),
    _vt_missing("vcftools_0_2", "0.2"),
    _vt_missing("vcftools_0_5", "0.5"),
    _vt_missing("vcftools_0_9_maf", "0.9", "--maf", "0"),
    _plink_geno("plink_0_1", "plink", "0.1"),
    _plink_geno("plink_0_9", "plink", "0.9"),
    _plink_geno("plink_0_05", "plink", "0.05"),
    _plink_geno("plink2_0_1", "plink2", "0.1"),
    _plink_geno("plink2_0_9", "plink2", "0.9"),
)


def _count_label(task, template: Template, work: Path, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    if template.tool == "vcftools":
        n = len(recode_ids(proc.stdout))
    elif template.tool == "plink":
        n = len((work / f"{template.note}.bim").read_text().splitlines())
    else:
        pvar = (work / f"{template.note}.pvar").read_text().splitlines()
        n = sum(1 for line in pvar if line and not line.startswith("#"))
    return "right" if answer_matches(task.truth, n) else "wrong"


@case_source("var.missing_threshold")
def missing_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F60", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, MISSING_TEMPLATES)
        for template, proc in zip(MISSING_TEMPLATES, procs, strict=True):
            label = _count_label(task, template, work, proc)
            own = "completeness" if template.tool == "vcftools" else "missing_rate"
            other = "missing_rate" if own == "completeness" else "completeness"
            emit_missing(
                out, template, label, work, seed, twin, [own, other][: 1 + (label == "wrong")]
            )
    return out


def emit_missing(out, template, label, work, seed, twin, meanings) -> None:
    """The call with each declaration, then undeclared (a required declaration: fails by design)."""
    for meaning in meanings:
        call = Call(template.tool, template.argv, work, {MISSING_KEY: meaning})
        out.append(Case(call, label, True, "F60", seed, twin, template.name))
    out.append(
        Case(
            Call(template.tool, template.argv, work, {}),
            label,
            False,
            "F60",
            seed,
            twin,
            template.name,
        )
    )


# ---- vcftools.subset_monomorphic -----------------------------------------------------------------
# Family F57 (implicit tier), see cases_var_bcftools.py. vcftools --keep/--indv ... --recode writes
# every site for the kept individuals (right on the clean twin, wrong on the faulted one); --mac 1,
# --non-ref-ac-any 1, --non-ref-ac 1 or --maf 0.0001 drop the monomorphic ones (right on both).

SUBSET_KEY = "subset_sites"


def _vk(name: str, meaning: str, *options: str) -> Template:
    argv = ("--gzvcf", "cohort.vcf.gz", *options, "--recode", "--stdout")
    return Template(name, "vcftools", argv, meaning)


def vcftools_subset_templates(group_b: list[str]) -> list[Template]:
    indv = [x for name in group_b for x in ("--indv", name)]
    return [
        _vk("keep_file", "keep_all_sites", "--keep", "group_b.txt"),
        _vk("indv_list", "keep_all_sites", *indv),
        _vk("keep_file_maf0", "keep_all_sites", "--keep", "group_b.txt", "--maf", "0"),
        _vk("keep_file_minq", "keep_all_sites", "--keep", "group_b.txt", "--minQ", "10"),
        _vk("keep_mac0", "keep_all_sites", "--keep", "group_b.txt", "--mac", "0"),
        _vk("keep_mac1", "variant_sites_only", "--keep", "group_b.txt", "--mac", "1"),
        _vk("indv_mac1", "variant_sites_only", *indv, "--mac", "1"),
        _vk(
            "keep_non_ref_any",
            "variant_sites_only",
            "--keep",
            "group_b.txt",
            "--non-ref-ac-any",
            "1",
        ),
        _vk("keep_non_ref", "variant_sites_only", "--keep", "group_b.txt", "--non-ref-ac", "1"),
        _vk("keep_maf", "variant_sites_only", "--keep", "group_b.txt", "--maf", "0.0001"),
    ]


@case_source("vcftools.subset_monomorphic")
def vcftools_subset_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F57", seed, twin, work, tools, "implicit", "free")
        templates = vcftools_subset_templates((work / "group_b.txt").read_text().split())
        procs = run_all(tools, work, templates)
        for template, proc in zip(templates, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = (
                    "right" if answer_matches(task.truth, len(recode_ids(proc.stdout))) else "wrong"
                )
            decls = [{SUBSET_KEY: "variant_sites_only"}]
            if label == "wrong":
                decls.append({SUBSET_KEY: template.meaning})
            emit(out, template, label, work, "F57", seed, twin, decls)
    return out
