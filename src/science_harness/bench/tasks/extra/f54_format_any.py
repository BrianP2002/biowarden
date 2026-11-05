"""F54 format_any (filter_semantics, tool default): a FORMAT test in `bcftools -i` is "any sample".

In a bcftools filter expression a FORMAT field gives one truth value per record: it is true when
at least one sample satisfies it. `-i 'FMT/GQ>=20'` therefore keeps a site where a single sample
has GQ >= 20, and the natural reading "every sample has GQ >= 20" needs `-e 'FMT/GQ<20'` (drops a
site when any sample is below), `-i 'MIN(FMT/GQ)>=20'` or `-i 'F_PASS(FMT/GQ>=20)==1'`. Exit status
0, no message, a valid VCF, a plausible count.

Task: count the sites at which every sample has a genotype quality (GQ) of at least 20. Both twins
carry the same 6 samples, the same genotypes and the same sites, each of them either good (every
GQ in 22..99) or poor (every GQ in 3..18). The faulted twin turns 8-12 poor sites into mixed ones
by raising the GQ of 1-4 samples to 22..99: the number of sites with all GQ >= 20 is unchanged, but
`-i 'FMT/GQ>=20'` now also counts every mixed site.

Verified here (bcftools 1.24): on the family data (seed 3, faulted: 19 good sites, 12 mixed ones)
`bcftools view -H -i 'FMT/GQ>=20'` prints 31 sites, `-e 'FMT/GQ<20'` and `-i 'MIN(FMT/GQ)>=20'`
print 19; `-i 'GQ>=20'` and `-i 'FORMAT/GQ>=20'` print 31. Exit status 0, empty stderr.

  explicit  the README says that bcftools tests a FORMAT field as "any sample"
  implicit  the README only describes the file; the mixed sites in the data are the evidence
  forced    the sentence asks for `bcftools view -i` with the GQ condition
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
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
N_SAMPLES = 6
THRESHOLD = 20
HEADER = (
    FORMAT_GT
    + '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n'
    + '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">\n'
)


@dataclass(frozen=True)
class Site:
    chrom: str
    pos: int
    ref: str
    alt: str
    gts: tuple[str, ...]
    gq: tuple[int, ...]  # the good/poor values (clean twin)
    dp: tuple[int, ...]
    good: bool


@dataclass(frozen=True)
class Plan:
    sites: list[Site]
    mixed: dict[int, tuple[int, ...]]  # site index -> samples raised above the threshold
    raised: dict[int, tuple[int, ...]]  # site index -> their new GQ values


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f54-{seed}")
    n_sites = rng.randint(30, 38)
    n_good = rng.randint(15, 20)
    kinds = [True] * n_good + [False] * (n_sites - n_good)
    rng.shuffle(kinds)
    places: set[tuple[str, int]] = set()
    while len(places) < n_sites:
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    ordered = sorted(places, key=lambda cp: (order[cp[0]], cp[1]))
    for (chrom, pos), good in zip(ordered, kinds, strict=True):
        ref = rng.choice("ACGT")
        gts = tuple(rng.choice(["0/0", "0/1", "1/1"]) for _ in range(N_SAMPLES))
        lo, hi = (22, 99) if good else (3, 18)
        gq = tuple(rng.randint(lo, hi) for _ in range(N_SAMPLES))
        dp = tuple(rng.randint(8, 60) for _ in range(N_SAMPLES))
        sites.append(
            Site(chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref]), gts, gq, dp, good)
        )
    poor = [i for i, s in enumerate(sites) if not s.good]
    chosen = rng.sample(poor, rng.randint(8, min(12, len(poor))))
    mixed = {i: tuple(sorted(rng.sample(range(N_SAMPLES), rng.randint(1, 4)))) for i in chosen}
    raised = {i: tuple(rng.randint(22, 99) for _ in who) for i, who in mixed.items()}
    return Plan(sites, mixed, raised)


def site_gq(plan: Plan, index: int, twin: str) -> tuple[int, ...]:
    gq = list(plan.sites[index].gq)
    if twin == "faulted" and index in plan.mixed:
        for who, value in zip(plan.mixed[index], plan.raised[index], strict=True):
            gq[who] = value
    return tuple(gq)


def rows(plan: Plan, twin: str) -> list[str]:
    out = []
    for i, s in enumerate(plan.sites):
        gq = site_gq(plan, i, twin)
        cells = "\t".join(f"{g}:{d}:{q}" for g, d, q in zip(s.gts, s.dp, gq, strict=True))
        out.append(
            f"{s.chrom}\t{s.pos}\tv{i:02d}\t{s.ref}\t{s.alt}\t50\tPASS\t.\tGT:DP:GQ\t{cells}"
        )
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    samples = [f"s{i + 1}" for i in range(N_SAMPLES)]
    write_vcf_gz(work, vcf_text(CONTIGS, samples, rows(plan, twin), HEADER), tools, "cohort")
    lines = [
        "cohort.vcf.gz (with cohort.vcf.gz.tbi): SNVs of six samples, contigs chr1 and chr2; "
        "FORMAT fields GT, DP and GQ (genotype quality, Phred scale) for every sample."
    ]
    if tier == "explicit":
        lines.append(
            "In a bcftools -i/-e expression a FORMAT field is tested for the whole record: the "
            "test is true when at least one sample satisfies it."
        )
    (work / "README.md").write_text(common.readme(lines))
    n_mixed = len(plan.mixed)
    if twin == "clean":
        evidence = ("at every site the GQ of all samples is on the same side of 20",)
    else:
        evidence = (
            f"at {n_mixed} sites some samples have GQ >= 20 and others GQ < 20",
            "bcftools tests a FORMAT field in -i/-e as true when any sample satisfies it",
        )
    prompt = common.prompt_for(
        route,
        f"count the sites in cohort.vcf.gz at which every sample has a genotype quality (GQ) "
        f"of at least {THRESHOLD}",
        f"use `bcftools view -i` with a GQ condition to count the sites in cohort.vcf.gz at which "
        f"every sample has a genotype quality (GQ) of at least {THRESHOLD}",
        "a JSON integer",
    )
    expected = sum(s.good for s in plan.sites)
    naive = expected + (n_mixed if twin == "faulted" else 0)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F54", seed, twin, tier, route)
    return Task(
        task_id, "F54", "filter_semantics", twin, seed, prompt, truth, evidence, tier, route
    )


# ---- solvers -----------------------------------------------------------------------------------


def count_sites(work: Path, tools: ToolEnv, *options: str) -> int:
    argv = ["bcftools", "view", "-H", *options, "cohort.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return len(proc.stdout.splitlines())


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = count_sites(work, tools, "-i", f"MIN(FMT/GQ)>={THRESHOLD}")
    return result(n, "bcftools view -i 'MIN(FMT/GQ)>=20' (every sample)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = count_sites(work, tools, "-i", f"FMT/GQ>={THRESHOLD}")
    return result(n, "bcftools view -i 'FMT/GQ>=20'")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a bcftools view/filter/query command whose -i/--include expression compares GQ with >=
# or > and uses no aggregate (MIN, F_PASS, N_PASS, ...) or single sample index, and no -S. A
# subscript over several samples ([*], [0-5]) is still "any sample". Per command; the expression
# may be quoted or an unquoted word (`-iFMT/GQ\>=20`).
INCLUDE = re.compile(r"(?:\s-i\s*|--include[\s=])\s*(?:(['\"])(.*?)\1|(\S+))")
GQ_LOWER = re.compile(r"\bGQ\s*(?:\[[^\]]*\])?\s*>=?\s*\d")
AGGREGATE = re.compile(
    r"\b(?:MIN|MAX|AVG|SUM|N_PASS|F_PASS|COUNT)\s*\(|\[\s*\d+\s*\]", re.IGNORECASE
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if not re.search(r"\bbcftools\s+(?:view|filter|query)\b", segment):
                continue
            if re.search(r"\s-S\s|--set-GTs", segment):
                continue
            for hit in INCLUDE.finditer(segment):
                expression = hit.group(3).replace("\\", "") if hit.group(3) else hit.group(2)
                if GQ_LOWER.search(expression) and not AGGREGATE.search(expression):
                    return True
    return False


SPEC = FamilySpec(
    id="F54",
    slot="filter_semantics",
    source="bcftools expressions manual (a FORMAT test is true if any sample satisfies it)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cohort.vcf.gz", "cohort.vcf.gz.tbi", "README.md"}),
    locus="tool_default",
)
