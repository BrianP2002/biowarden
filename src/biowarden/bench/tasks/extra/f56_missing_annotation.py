"""F56 missing_annotation (missing_value, tool default): a missing annotation passes an exclusion.

A comparison on a value the record does not have is false for every operator. `bcftools view -e
'INFO/QD<2'` ("drop the low-QD records") therefore keeps the records without a QD annotation, while
`-i 'INFO/QD>=2'` drops them. GATK VariantFiltration behaves the same way: `--filter-expression
"QD < 2.0"` filters a record when the expression is true, and a record without QD is not filtered
(it is marked PASS) unless `--missing-values-evaluate-as-failing` is given. Nothing is printed.

Task: list the IDs of the variants that have a QD annotation of at least 2. Both twins carry the
same sites. The clean twin has a QD value at every record (good ones 2.5..30, poor ones 0.2..1.9).
In the faulted twin 8-12 of the poor records have no QD annotation (as for calls made by another
tool), so they are not "QD >= 2" but also not "QD < 2" for the tool: the exclusion
`-e 'INFO/QD<2'` keeps them.

Verified here (bcftools 1.24, GATK 4.6.2.0): on the family data `bcftools view -H -e 'INFO/QD<2'`
returns the good records plus the ones without QD, `-i 'INFO/QD>=2'` returns the good ones;
`gatk VariantFiltration --filter-expression "QD < 2.0"` marks the records without QD PASS and
`--missing-values-evaluate-as-failing` filters them (tests/test_family_var.py).

  explicit  the README says which records carry no QD annotation
  implicit  the README only describes the INFO fields; the records without QD are the evidence
  forced    the sentence asks for `bcftools view -e` with the QD cutoff
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import (
    FORMAT_GT,
    pipelines,
    result,
    vcf_text,
    write_vcf_gz,
)
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 2400, "chr2": 1800}
CUTOFF = 2
HEADER = (
    '##INFO=<ID=QD,Number=1,Type=Float,Description="Variant confidence divided by depth">\n'
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n' + FORMAT_GT
)


def make_plan(seed: int) -> tuple[list[tuple[str, int, str, str, float, int]], list[int]]:
    """Sites (contig, pos, ref, alt, QD, DP) and the indices of the poor ones that lose QD in the
    faulted twin."""
    rng = random.Random(f"f56-{seed}")
    n_good, n_poor = rng.randint(14, 20), rng.randint(14, 20)
    kinds = [True] * n_good + [False] * n_poor
    rng.shuffle(kinds)
    places: set[tuple[str, int]] = set()
    while len(places) < len(kinds):
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    for (chrom, pos), good in zip(
        sorted(places, key=lambda cp: (order[cp[0]], cp[1])), kinds, strict=True
    ):
        ref = rng.choice("ACGT")
        qd = round(rng.uniform(2.5, 30), 2) if good else round(rng.uniform(0.2, 1.9), 2)
        sites.append(
            (chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref]), qd, rng.randint(20, 90))
        )
    poor = [i for i, s in enumerate(sites) if s[4] < CUTOFF]
    return sites, sorted(rng.sample(poor, rng.randint(8, min(12, len(poor)))))


def rows(sites, lost: set[int], rng: random.Random) -> list[str]:
    out = []
    for i, (chrom, pos, ref, alt, qd, dp) in enumerate(sites):
        gts = "\t".join(rng.choice(["0/0", "0/1", "1/1"]) for _ in range(3))
        info = f"DP={dp}" if i in lost else f"QD={qd};DP={dp}"
        out.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\tPASS\t{info}\tGT\t{gts}")
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, lost = make_plan(seed)
    rng = random.Random(f"f56-gt-{seed}")
    text = vcf_text(
        CONTIGS,
        ["s1", "s2", "s3"],
        rows(sites, set(lost) if twin == "faulted" else set(), rng),
        HEADER,
    )
    write_vcf_gz(work, text, tools)
    lines = [
        "variants.vcf.gz (with variants.vcf.gz.tbi): SNVs of three samples, contigs chr1 and chr2; "
        "INFO fields QD (quality by depth) and DP (depth)."
    ]
    if tier == "explicit":
        lines.append(
            "Some records carry no QD annotation (QD is absent from their INFO)."
            if twin == "faulted"
            else "Every record carries a QD annotation."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("every record has an INFO/QD value",)
    else:
        evidence = (
            f"{len(lost)} records have no INFO/QD annotation",
            "a comparison on a missing value is false, so -e 'INFO/QD<2' keeps those records",
        )
    prompt = common.prompt_for(
        route,
        f"list the IDs of the variants in variants.vcf.gz that have a QD annotation of at least "
        f"{CUTOFF}",
        f"use `bcftools view -e` to drop the variants with a low QD and list the IDs of the "
        f"variants in variants.vcf.gz that have a QD annotation of at least {CUTOFF}",
        "a JSON list of variant IDs",
    )
    expected = [f"v{i:02d}" for i, s in enumerate(sites) if s[4] >= CUTOFF]
    naive = sorted(expected + [f"v{i:02d}" for i in lost]) if twin == "faulted" else expected
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F56", seed, twin, tier, route)
    return Task(task_id, "F56", "missing_value", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def bcftools_ids(work: Path, tools: ToolEnv, *options: str) -> list[str]:
    argv = ["bcftools", "view", "-H", *options, "variants.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sorted(line.split("\t")[2] for line in proc.stdout.splitlines() if line)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(bcftools_ids(work, tools, "-i", f"INFO/QD>={CUTOFF}"), "bcftools view -i QD>=2")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(bcftools_ids(work, tools, "-e", f"INFO/QD<{CUTOFF}"), "bcftools view -e QD<2")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: an exclusion of low QD (bcftools -e 'QD<c' / 'INFO/QD<c', alone or inside a longer
# expression; a GATK VariantFiltration expression "QD < c" without
# --missing-values-evaluate-as-failing). Safe: -i 'QD>=c', an exclusion combined with an explicit
# test of the missing value or with an -i 'QD>=c' stage of the same pipeline, the GATK flag.
# Judged per pipeline: a `;` starts another one.
EXCLUDE_QD = re.compile(r"(?:\s-e\s*|--exclude[\s=])(['\"])[^'\"]*?\bQD\s*<=?\s*[\d.]+[^'\"]*\1")
GATK_QD = re.compile(
    r"\bVariantFiltration\b[^|;&\n]*(?:--filter-expression|-filter)\s+(['\"]?)[^'\"]*?\bQD\s*<"
)
MISSING_SAFE = re.compile(
    r"QD\s*=\s*[\"']\.[\"']|--missing-values-evaluate-as-failing|\bexists\(|\bhasAttribute\b"
    r"|(?:\s-i\s*|--include[\s=])(['\"])[^'\"]*?\bQD\s*>=?\s*[\d.]+[^'\"]*\1"
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for pipeline in pipelines(command):
            if MISSING_SAFE.search(pipeline):
                continue
            if EXCLUDE_QD.search(pipeline) or GATK_QD.search(pipeline):
                return True
    return False


SPEC = FamilySpec(
    id="F56",
    slot="missing_value",
    source="bcftools expressions manual (missing values are false); GATK VariantFiltration docs",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"variants.vcf.gz", "variants.vcf.gz.tbi", "README.md"}),
    locus="tool_default",
)
