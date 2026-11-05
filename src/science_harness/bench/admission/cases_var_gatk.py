"""Admission cases for the GATK and Picard rules of the variant pack.

GATK is a JVM program (about 2-8 s per call at -Xmx512m), so these case sources run each call once
per twin with two workers; the admission tests use few seeds.

gatk.missing_annotation (family F56, implicit tier): variants.vcf.gz has an INFO/QD value at every
record (clean) or lacks it at 8-12 poor records (faulted). The task lists the variants that have
QD >= 2. A VariantFiltration call is run for real and the records whose FILTER is PASS in its
output are compared with `task.truth`. The default expression "QD < 2.0" leaves the records
without QD PASS (right on the clean twin, wrong on the faulted one); with
--missing-values-evaluate-as-failing they are filtered (right on both). `record_removed` is the
intent of the task; a call that keeps also carries the declaration its options imply.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.admission.cases_aln_samtools import Template, emit
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

MISSING_KEY = "missing_annotation"
JAVA = ("--java-options", "-Xmx512m")


def _vf(name: str, meaning: str, *options: str) -> Template:
    argv = (*JAVA, "VariantFiltration", "-V", "variants.vcf.gz", *options, "-O", f"o_{name}.vcf.gz")
    return Template(name, "gatk", argv, meaning)


GATK_TEMPLATES: tuple[Template, ...] = (
    _vf("default", "record_kept", "--filter-expression", "QD < 2.0", "--filter-name", "lowQD"),
    _vf("short_option", "record_kept", "-filter", "QD < 2", "--filter-name", "lowQD"),
    _vf(
        "missing_fails",
        "record_removed",
        "--filter-expression",
        "QD < 2.0",
        "--filter-name",
        "lowQD",
        "--missing-values-evaluate-as-failing",
    ),
    _vf(
        "missing_fails_short",
        "record_removed",
        "-filter",
        "QD < 2",
        "--filter-name",
        "lowQD",
        "--missing-values-evaluate-as-failing",
    ),
)


def _run(tools: ToolEnv, work: Path, template: Template):
    return tools.run([template.tool, *template.argv], work)


def pass_ids(tools: ToolEnv, work: Path, template: Template) -> list[str]:
    out = template.argv[-1]
    proc = tools.run(["bcftools", "view", "-H", "-f", "PASS", out], work)
    return sorted(line.split("\t")[2] for line in proc.stdout.splitlines() if line)


@case_source("gatk.missing_annotation")
def missing_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F56", seed, twin, work, tools, "implicit", "free")
        with ThreadPoolExecutor(max_workers=2) as pool:
            procs = list(pool.map(partial(_run, tools, work), GATK_TEMPLATES))
        for template, proc in zip(GATK_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                label = (
                    "right"
                    if answer_matches(task.truth, pass_ids(tools, work, template))
                    else "wrong"
                )
            decls = [{MISSING_KEY: "record_removed"}]
            if label == "wrong":
                decls.append({MISSING_KEY: template.meaning})
            emit(out, template, label, work, "F56", seed, twin, decls)
    return out


# ---- picard.markduplicates_remove ----------------------------------------------------------------
# Family F64 (implicit tier): reads.bam has distinct read pairs (clean twin) plus exact duplicates
# of some of them (faulted twin). The task counts the pairs that remain after removing PCR
# duplicates. MarkDuplicates is run for real (the JVM, about 8 s per call: few seeds and few
# calls) and the first mates of its output are counted with samtools. A default run writes every
# read (right on the clean twin, wrong on the faulted one); REMOVE_DUPLICATES=true leaves the
# distinct pairs (right on both). The rule requires a declaration (it looks at the call alone):
# a call carries the declaration its options imply; a wrong call that flags only also carries the
# intent of the task (`removed`).

DUP_KEY = "duplicate_output"


def _md(name: str, meaning: str, *options: str) -> Template:
    out, metrics = f"o_{name}.bam", f"o_{name}.txt"
    argv = ("-Xmx512m", "MarkDuplicates", *options, f"O={out}", f"M={metrics}")
    return Template(name, "picard", argv, meaning, out)


PICARD_TEMPLATES: tuple[Template, ...] = (
    _md("flag_legacy", "flagged", "I=reads.bam"),
    _md(
        "flag_remove_sequencing", "flagged", "I=reads.bam", "--REMOVE_SEQUENCING_DUPLICATES", "true"
    ),
    _md("remove_legacy", "removed", "I=reads.bam", "REMOVE_DUPLICATES=true"),
    _md("remove_new", "removed", "I=reads.bam", "--REMOVE_DUPLICATES", "true"),
)


@case_source("picard.markduplicates_remove")
def picard_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F64", seed, twin, work, tools, "implicit", "free")
        with ThreadPoolExecutor(max_workers=2) as pool:
            procs = list(pool.map(partial(_run, tools, work), PICARD_TEMPLATES))
        for template, proc in zip(PICARD_TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                count = tools.run(["samtools", "view", "-c", "-f", "0x40", template.note], work)
                label = "right" if answer_matches(task.truth, int(count.stdout)) else "wrong"
            # a right call carries the declaration its own options imply (this rule looks at the
            # call alone, so a flag-only call that is right on the clean twin cannot say
            # "removed"); a wrong call also carries the intent of the task (removed)
            decls = [{DUP_KEY: template.meaning}]
            if label == "wrong" and template.meaning != "removed":
                decls.append({DUP_KEY: "removed"})
            for declared, decl in [(True, d) for d in decls] + [(False, {})]:
                call = Call(template.tool, template.argv, work, decl)
                out.append(Case(call, label, declared, "F64", seed, twin, template.name))
    return out
