"""Admission cases for the plink rules of the variant pack.

Every case runs the call for real in a benchmark workspace of its family (both twins, implicit
tier), turns the output into the task's answer and compares it with `task.truth`
(bench.score.answer_matches). Labels never come from a rule.

plink.subset_monomorphic (family F57): cohort.vcf.gz has eight samples, group_b.txt lists four
(keep_fid.txt, made here, is the FID IID file plink wants). `plink --vcf ... --keep ...` with
--make-just-bim writes every variant (right on the clean twin, wrong on the faulted one),
`--mac 1` or `--maf X` drops the monomorphic ones (right on both).

plink.allele_order (family F58, documentation tier): the same VCF in both twins, ALT the major
allele at 9-13 variants; the task reports the ALT frequency of every variant. The MAF column of
`plink --vcf V --freq` is the ALT frequency only with --keep-allele-order (A1 = ALT); plink2
--freq writes ALT_FREQS. The label of a call comes from the frequencies it prints.

plink.vcf_multiallelic (family F59): 20 samples, the sites to keep have at most one missing call;
on the faulted twin 6-8 of them are multiallelic with calls of the second ALT. The task counts the
variants with at most 10% missing calls; `plink --vcf --geno 0.1` removes the multiallelic ones.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.admission.cases_aln_samtools import Template, emit, run_all
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task

MEM = ("--memory", "1000", "--silent")

# ---- plink.subset_monomorphic --------------------------------------------------------------------

SUBSET_KEY = "subset_sites"


def _pk(name: str, tool: str, meaning: str, *options: str) -> Template:
    make = "--make-just-bim" if tool == "plink" else "--make-just-pvar"
    keep = "keep_fid.txt" if tool == "plink" else "group_b.txt"  # plink2 wants IIDs only
    argv = (*MEM, "--vcf", "cohort.vcf.gz", "--keep", keep, *options, make)
    argv += ("--out", f"o_{name}")
    return Template(name, tool, argv, meaning, f"o_{name}")


SUBSET_TEMPLATES: tuple[Template, ...] = (
    _pk("plain", "plink", "keep_all_sites"),
    _pk("freq_filter_unrelated", "plink", "keep_all_sites", "--geno", "0.5"),
    _pk("hwe_unrelated", "plink", "keep_all_sites", "--hwe", "1e-8"),
    _pk("mac0", "plink", "keep_all_sites", "--mac", "0"),
    _pk("maf0", "plink", "keep_all_sites", "--maf", "0"),
    _pk("mac1", "plink", "variant_sites_only", "--mac", "1"),
    _pk("maf_small", "plink", "variant_sites_only", "--maf", "0.0001"),
    _pk("mac1_geno", "plink", "variant_sites_only", "--mac", "1", "--geno", "0.5"),
    _pk("plink2_plain", "plink2", "keep_all_sites"),
    _pk("plink2_mac1", "plink2", "variant_sites_only", "--mac", "1"),
)


def _variant_count(template: Template, work: Path) -> int:
    if template.tool == "plink":
        return len((work / f"{template.note}.bim").read_text().splitlines())
    pvar = (work / f"{template.note}.pvar").read_text().splitlines()
    return sum(1 for line in pvar if line and not line.startswith("#"))


@case_source("plink.subset_monomorphic")
def subset_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F57", seed, twin, work, tools, "implicit", "free")
        names = (work / "group_b.txt").read_text().split()
        (work / "keep_fid.txt").write_text("".join(f"{n} {n}\n" for n in names))
        procs = run_all(tools, work, SUBSET_TEMPLATES)
        for template, proc in zip(SUBSET_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                n = _variant_count(template, work)
                label = "right" if answer_matches(task.truth, n) else "wrong"
            decls = [{SUBSET_KEY: "variant_sites_only"}]
            if label == "wrong":
                decls.append({SUBSET_KEY: template.meaning})
            emit(out, template, label, work, "F57", seed, twin, decls)
    return out


# ---- plink.allele_order --------------------------------------------------------------------------

A1_KEY = "plink_a1_allele"


def _pf(name: str, meaning: str, *options: str) -> Template:
    argv = (*MEM, "--vcf", "cohort.vcf.gz", *options, "--freq", "--out", f"o_{name}")
    return Template(name, "plink", argv, meaning, f"o_{name}")


A1_TEMPLATES: tuple[Template, ...] = (
    # --- the default import: A1 is the minor allele
    _pf("default", "minor_allele"),
    _pf("readme_command", "minor_allele", "--allow-no-sex"),
    _pf("nonfounders", "minor_allele", "--nonfounders"),
    _pf("maf_filter", "minor_allele", "--maf", "0.01"),
    _pf("chr_filter", "minor_allele", "--chr", "1-22"),
    _pf("freq_after", "minor_allele", "--geno", "0.5"),
    # --- A1 = ALT
    _pf("keep_allele_order", "alt_allele", "--keep-allele-order"),
    _pf("keep_allele_order_threads", "alt_allele", "--keep-allele-order", "--threads", "1"),
    _pf("keep_allele_order_maf", "alt_allele", "--keep-allele-order", "--maf", "0.01"),
)


def _freqs(work: Path, template: Template) -> dict[str, float]:
    rows = [line.split() for line in (work / f"{template.note}.frq").read_text().splitlines()[1:]]
    return {r[1]: float(r[4]) for r in rows}


@case_source("plink.allele_order")
def allele_order_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F58", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, A1_TEMPLATES)
        for template, proc in zip(A1_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = "right" if answer_matches(task.truth, _freqs(work, template)) else "wrong"
            decls = [{A1_KEY: "alt_allele"}]
            if label == "wrong":
                decls.append({A1_KEY: template.meaning})
            emit(out, template, label, work, "F58", seed, twin, decls)
    return out


# ---- plink.vcf_multiallelic ----------------------------------------------------------------------

MULTI_KEY = "vcf_multiallelic"


def _pg(name: str, tool: str, meaning: str, *options: str) -> Template:
    make = "--make-just-bim" if tool == "plink" else "--make-just-pvar"
    argv = (*MEM, "--vcf", "cohort.vcf.gz", *options, "--geno", "0.1", make, "--out", f"o_{name}")
    return Template(name, tool, argv, meaning, f"o_{name}")


MULTI_TEMPLATES: tuple[Template, ...] = (
    _pg("default", "plink", "first_alt_only"),
    _pg("maf0", "plink", "first_alt_only", "--maf", "0"),
    _pg("allow_extra_chr", "plink", "first_alt_only", "--allow-extra-chr"),
    _pg("double_id", "plink", "first_alt_only", "--double-id"),
    _pg("biallelic_only", "plink", "biallelic_only", "--biallelic-only"),
    _pg("biallelic_only_strict", "plink", "biallelic_only", "--biallelic-only", "strict"),
    _pg("plink2", "plink2", "first_alt_only"),
)


@case_source("plink.vcf_multiallelic")
def multiallelic_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F59", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, MULTI_TEMPLATES)
        for template, proc in zip(MULTI_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = (
                    "right"
                    if answer_matches(task.truth, _variant_count(template, work))
                    else "wrong"
                )
            decls = [{MULTI_KEY: "first_alt_only"}]
            if label == "wrong" and template.meaning != "first_alt_only":
                decls.append({MULTI_KEY: template.meaning})
            emit(out, template, label, work, "F59", seed, twin, decls)
    return out
