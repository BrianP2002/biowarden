"""F60 max_missing (metric_scale, tool default): vcftools --max-missing is a completeness.

`vcftools --max-missing X` keeps the sites at which at least a fraction X of the genotypes are
called: 0 keeps every site, 1 keeps only complete sites. Written for "at most 10% missing", a copy
of the rate from plink (`--geno 0.1`, `--mind 0.1`: the maximum missing rate) as `--max-missing 0.1`
keeps every site that has at least 10% of its genotypes called. Exit status 0, "kept N out of a
possible M Sites" with a plausible N.

Task: keep the sites with at most 10% missing genotypes and report how many sites remain. Both twins
carry the same 20 samples and the same sites. 18-26 sites are complete (0 or 1 missing call) and
the others are almost empty (19 or 20 missing) in the clean twin, where 0.1 and 0.9 select the same
sites. In the faulted twin 8-12 of the almost empty sites hold 6-14 missing calls (graded
missingness), which `--max-missing 0.1` keeps and the task excludes.

Verified here (vcftools 0.1.17, plink 1.9, plink2 2.0.0-a.7, bcftools 1.24): on a six-site VCF with
one missing call at site b, `--max-missing 1` kept 5 sites (b dropped), `--max-missing 0.75` kept
all 6, `--max-missing 0.1` kept all 6; `plink --geno 0.1` removed 2 sites and `plink2 --geno 0.1` 1.
`bcftools view -i 'F_MISSING<=0.1'` and `vcftools --max-missing 0.9` agree on the family data.

  explicit  the README says what --max-missing means ("0 allows completely missing sites, 1 none")
  implicit  the README only describes the VCF; the graded missingness is the evidence
  forced    the sentence asks for `vcftools --max-missing`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import (
    FORMAT_GT,
    result,
    segments,
    vcf_text,
    write_vcf_gz,
)
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 2400, "chr2": 1800}
N_SAMPLES = 20


def make_plan(seed: int):
    """Sites (contig, pos, ref, alt, genotypes with ./. in the clean twin) and the graded sites'
    missing counts for the faulted twin."""
    rng = random.Random(f"f60-{seed}")
    n_complete, n_empty = rng.randint(18, 26), rng.randint(14, 20)
    kinds = ["complete"] * n_complete + ["empty"] * n_empty
    rng.shuffle(kinds)
    places: set[tuple[str, int]] = set()
    while len(places) < len(kinds):
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    for (chrom, pos), kind in zip(
        sorted(places, key=lambda cp: (order[cp[0]], cp[1])), kinds, strict=True
    ):
        ref = rng.choice("ACGT")
        alt = rng.choice([b for b in "ACGT" if b != ref])
        gts = [rng.choice(["0/0", "0/1", "1/1"]) for _ in range(N_SAMPLES)]
        n_missing = rng.choice([0, 0, 1]) if kind == "complete" else rng.choice([19, 20])
        for i in rng.sample(range(N_SAMPLES), n_missing):
            gts[i] = "./."
        sites.append([chrom, pos, ref, alt, gts, kind])
    empty = [i for i, s in enumerate(sites) if s[5] == "empty"]
    graded = {i: rng.randint(6, 14) for i in rng.sample(empty, rng.randint(8, min(12, len(empty))))}
    return sites, graded


def rows(sites, graded: dict[int, int], twin: str, seed: int) -> list[str]:
    out = []
    rng = random.Random(f"f60-grade-{seed}")
    for i, (chrom, pos, ref, alt, gts, _kind) in enumerate(sites):
        gts = list(gts)
        if twin == "faulted" and i in graded:
            gts = [rng.choice(["0/0", "0/1", "1/1"]) for _ in range(N_SAMPLES)]
            for j in rng.sample(range(N_SAMPLES), graded[i]):
                gts[j] = "./."
        out.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\tPASS\t.\tGT\t" + "\t".join(gts))
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, graded = make_plan(seed)
    samples = [f"s{i + 1}" for i in range(N_SAMPLES)]
    write_vcf_gz(
        work,
        vcf_text(CONTIGS, samples, rows(sites, graded, twin, seed), FORMAT_GT),
        tools,
        "cohort",
    )
    lines = [
        "cohort.vcf.gz (with cohort.vcf.gz.tbi): SNVs of twenty samples, contigs chr1 and chr2."
    ]
    if tier == "explicit":
        lines.append(
            "vcftools --max-missing takes a fraction between 0 and 1: 0 allows sites that are "
            "completely missing, 1 allows no missing data."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("every site is complete or has at most one called genotype",)
    else:
        evidence = (
            f"{len(graded)} sites have 30-70% missing genotypes",
            "vcftools --max-missing X keeps the sites with a fraction of at least X called",
        )
    prompt = common.prompt_for(
        route,
        "keep the sites of cohort.vcf.gz with at most 10% missing genotypes and report how many "
        "sites remain",
        "use `vcftools --max-missing` to keep the sites of cohort.vcf.gz with at most 10% missing "
        "genotypes and report how many sites remain",
        "a JSON integer",
    )
    expected = sum(1 for s in sites if s[5] == "complete")
    naive = expected + (len(graded) if twin == "faulted" else 0)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F60", seed, twin, tier, route)
    return Task(task_id, "F60", "metric_scale", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def kept_sites(work: Path, tools: ToolEnv, value: str) -> int:
    argv = ["vcftools", "--gzvcf", "cohort.vcf.gz", "--max-missing", value, "--recode", "--stdout"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sum(1 for line in proc.stdout.splitlines() if line and line[0] != "#")


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(kept_sites(work, tools, "0.9"), "vcftools --max-missing 0.9 (completeness)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(kept_sites(work, tools, "0.1"), "vcftools --max-missing 0.1")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: vcftools --max-missing with a value of 0.5 or less (a rate copied as a completeness).
# Safe: 0.9 or more, other tools (bcftools F_MISSING, plink --geno 0.1). Applied per command.
MAX_MISSING = re.compile(r"--max-missing[\s=]+(\d*\.?\d+)")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if "vcftools" not in segment:
                continue
            for value in MAX_MISSING.findall(segment):
                if float(value) <= 0.5:
                    return True
    return False


SPEC = FamilySpec(
    id="F60",
    slot="metric_scale",
    source="GEN-31 (vcftools --max-missing against PLINK --geno)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cohort.vcf.gz", "cohort.vcf.gz.tbi"}),
    locus="tool_default",
)
