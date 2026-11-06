"""F20 wrap_drop (record_count, locus wrapper): the lab's record-count wrapper only looks at a
hard-coded list of standard chromosomes, so records on chrM and on unplaced contigs are dropped.

Source SCD-01 (a wrapper layer that returns a valid-looking but partial result) and GEN-08 (records
of a requested set silently missing). Verified on bcftools 1.24: `bcftools view -H variants.vcf.gz`
prints every record, including those on contigs declared in the header such as chrM and
chrUn_KI270742v1; `bcftools view -H -r chr1,...,chr22,chrX,chrY variants.vcf.gz` prints only the
records on the listed contigs, exits 0 and writes nothing to stderr, also for listed contigs that
the file does not have.

Twin axis: the VCF is the same in both twins (SNVs on chr1, chr2, chrM and chrUn_KI270742v1; 12-18
of the 56-72 records are on the last two). The clean wrapper counts `bcftools view -H`; the faulted
one adds `-r "$STD"` with the lab's list of standard chromosomes. The wrapper is plain text, so
`cat` shows the list; the VCF header declares all four contigs. The README names the list only in
the explicit tier and never lists the contigs. Plain sh; bcftools by name.

Exposure: the shared rule of F17. This wrapper has no override, so a command is safe only when it
also runs bcftools on the records itself or rewrites the wrapper.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f17_wrap_filter import (
    result,
    run_wrapper,
    wrapper_exposed,
    write_script,
)
from biowarden.bench.tasks.tools import ToolEnv

WRAPPER = "lab-nvar.sh"
CONTIGS = {
    "chr1": (1200, 1800),
    "chr2": (1000, 1500),
    "chrM": (500, 700),
    "chrUn_KI270742v1": (250, 400),
}
PER_CONTIG = {"chr1": (22, 30), "chr2": (18, 26), "chrM": (8, 12), "chrUn_KI270742v1": (4, 6)}
STANDARD = frozenset({*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY"})
TWIN_FILES = frozenset({WRAPPER, "README.md"})

HEAD = """#!/bin/sh
# lab-nvar.sh: number of variant records in a VCF/BCF.
# usage: ./lab-nvar.sh VCF
set -eu
[ $# -ge 1 ] || { echo "usage: $0 VCF" >&2; exit 2; }
"""
WRAPPER_CLEAN = HEAD + 'bcftools view -H "$1" | wc -l\n'
WRAPPER_FAULTED = (
    HEAD
    + "# standard chromosomes of the lab's reference\n"
    + "STD=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12\n"
    + "STD=$STD,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY\n"
    + 'bcftools view -H -r "$STD" "$1" | wc -l\n'
)


@dataclass(frozen=True)
class Variant:
    id: str
    chrom: str
    pos: int
    ref: str
    alt: str


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    variants: list[Variant]


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f20-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    variants: list[Variant] = []
    for chrom, n in contigs.items():
        lo, hi = PER_CONTIG[chrom]
        for pos in sorted(rnd.sample(range(10, n - 10), rnd.randint(lo, hi))):
            ref = rnd.choice("ACGT")
            alt = rnd.choice([b for b in "ACGT" if b != ref])
            variants.append(Variant(f"v{len(variants) + 1:03d}", chrom, pos, ref, alt))
    return Plan(contigs, variants)


def truth_counts(plan: Plan) -> tuple[int, int]:
    """(every record, the records on the wrapper's standard chromosomes)."""
    return len(plan.variants), sum(v.chrom in STANDARD for v in plan.variants)


def vcf_text(plan: Plan) -> tuple[str, str]:
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items()
    )
    body = "".join(
        f"{v.chrom}\t{v.pos}\t{v.id}\t{v.ref}\t{v.alt}\t.\t.\t.\n" for v in plan.variants
    )
    return header, body


def build_f20(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """record_count: records on non-standard contigs dropped by the lab's counting wrapper."""
    plan = make_plan(seed)
    common.write_vcf_text(work, *vcf_text(plan), tools)
    faulted = twin == "faulted"
    write_script(work, WRAPPER, WRAPPER_FAULTED if faulted else WRAPPER_CLEAN)
    line = (
        f"{WRAPPER}: the lab's record-count wrapper around bcftools view -H "
        "(usage: ./lab-nvar.sh VCF)."
    )
    if tier == "explicit":
        line += (
            " It only counts records on chr1-chr22, chrX and chrY."
            if faulted
            else " It counts every record in the file."
        )
    readme = [
        "variants.vcf.gz (with variants.vcf.gz.csi): SNVs called against the lab's reference, "
        "VCF 4.2.",
        line,
    ]
    (work / "README.md").write_text(common.readme(readme))
    source = (
        f'{WRAPPER} source: bcftools view -H -r "$STD" with a hard-coded list of standard '
        "chromosomes"
    )
    header = "the VCF header declares chrM and chrUn_KI270742v1, which the wrapper's list lacks"
    if faulted:
        evidence = (
            f"README: {WRAPPER} only counts records on chr1-chr22, chrX and chrY",
            source,
            header,
        )
        evidence = evidence if tier == "explicit" else evidence[1:]
    else:
        evidence = (
            (f"README: {WRAPPER} counts every record in the file",) if tier == "explicit" else ()
        )
    prompt = common.prompt_for(
        route,
        "report how many variant records variants.vcf.gz contains "
        f"(the lab usually counts them with ./{WRAPPER})",
        f"use the lab's ./{WRAPPER} wrapper to report how many variant records variants.vcf.gz "
        "contains",
        "a JSON integer",
    )
    total, standard = truth_counts(plan)
    truth = {"kind": "int", "expected": total, "naive": standard if faulted else total}
    task_id = common.task_id("F20", seed, twin, tier, route)
    return Task(task_id, "F20", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def reference_f20(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work)
    return result(
        len(proc.stdout.splitlines()), "bcftools view -H on the whole file, no contig list"
    )


def naive_f20(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = run_wrapper(tools, work, WRAPPER, "variants.vcf.gz")
    return result(int(proc.stdout), f"./{WRAPPER} variants.vcf.gz")


def exposed_f20(commands: list[str], work: Path) -> bool:
    return wrapper_exposed(commands, WRAPPER, "bcftools")


SPEC = FamilySpec(
    id="F20",
    slot="record_count",
    source="SCD-01 (wrapper-layer partial results); GEN-08 (requested records silently missing)",
    build=build_f20,
    reference=reference_f20,
    naive=naive_f20,
    exposed=exposed_f20,
    twin_files=TWIN_FILES,
    locus="wrapper",
)
