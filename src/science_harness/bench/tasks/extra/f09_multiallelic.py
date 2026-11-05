"""F9 multiallelic: a multi-allelic record is one VCF row but several alternate alleles, so counting
rows (`bcftools view -H | wc -l`) undercounts alleles.

Verified on bcftools 1.24 (exit 0, empty stderr): `bcftools view -H` prints one row per site with
the alleles joined by commas in ALT; `bcftools index -n` and the "number of records" / "number of
SNPs" lines of `bcftools stats` also count rows. `bcftools norm -m -any` splits the alleles into one
row each (it prints a "Lines total/split/..." summary on stderr, which is information, not a
warning).

Twin axis: both twins carry the same number of alternate alleles. The clean twin writes each on its
own bi-allelic row; the faulted twin folds 6-9 groups of them into multi-allelic rows (2 or 3 ALT
alleles), so the row count falls short of the allele count by at least 6.
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
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": (900, 1200), "chr2": (700, 1000)}
TWIN_FILES = frozenset({"variants.vcf.gz", "variants.vcf.gz.csi", "README.md"})


@dataclass(frozen=True)
class Site:
    chrom: str
    pos: int
    ref: str
    alts: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    clean: list[Site]  # one ALT allele per site
    faulted: list[Site]  # some sites carry two or three ALT alleles

    def sites(self, twin: str) -> list[Site]:
        return self.clean if twin == "clean" else self.faulted


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f09-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    n_single, n_multi = rnd.randint(14, 20), rnd.randint(6, 9)
    sizes = [rnd.choice([2, 2, 3]) for _ in range(n_multi)]
    sizes[0] = 3  # at least one triple, so row count, multi-allelic rows and alleles all differ
    total = n_single + sum(sizes)
    order = {c: i for i, c in enumerate(contigs)}
    taken: dict[tuple[str, int], None] = {}
    while len(taken) < total:
        chrom = rnd.choice(list(contigs))
        pos = rnd.randint(10, contigs[chrom] - 10)
        if all(c != chrom or abs(p - pos) > 2 for c, p in taken):
            taken[(chrom, pos)] = None
    places = sorted(taken, key=lambda cp: (order[cp[0]], cp[1]))
    refs = {cp: rnd.choice("ACGT") for cp in places}
    clean = [
        Site(c, p, refs[c, p], (rnd.choice([b for b in "ACGT" if b != refs[c, p]]),))
        for c, p in places
    ]
    # fold n_multi hosts with sizes-1 absorbed rows each into multi-allelic rows
    picks = rnd.sample(range(total), n_multi + sum(s - 1 for s in sizes))
    hosts, absorbed = picks[:n_multi], set(picks[n_multi:])
    faulted = []
    for i, site in enumerate(clean):
        if i in absorbed:
            continue
        if i in hosts:
            others = [b for b in "ACGT" if b != site.ref]
            alts = tuple(rnd.sample(others, sizes[hosts.index(i)]))
            site = Site(site.chrom, site.pos, site.ref, alts)
        faulted.append(site)
    return Plan(contigs, clean, faulted)


def n_alleles(sites: list[Site]) -> int:
    return sum(len(s.alts) for s in sites)


def truth_counts(plan: Plan, twin: str) -> tuple[int, int]:
    """(expected, naive): ALT alleles in the file, and VCF rows."""
    sites = plan.sites(twin)
    return n_alleles(sites), len(sites)


def _vcf_text(plan: Plan, twin: str) -> tuple[str, str]:
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in plan.contigs.items()
    )
    body = "".join(
        f"{s.chrom}\t{s.pos}\tv{i:02d}\t{s.ref}\t{','.join(s.alts)}\t.\t.\t.\n"
        for i, s in enumerate(plan.sites(twin))
    )
    return header, body


def build_f9(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """record_count: rows against alternate alleles in a VCF with multi-allelic sites."""
    plan = make_plan(seed)
    common.write_vcf_text(work, *_vcf_text(plan, twin), tools)
    head = "variants.vcf.gz (with variants.vcf.gz.csi): "
    tail = ", VCF 4.2, contigs chr1 and chr2."
    lines = [head + ("SNVs" if tier == "explicit" else "joint-called SNVs") + tail]
    if tier == "explicit":
        lines.append(
            "Some sites are multi-allelic: ALT lists several comma-separated alleles."
            if twin == "faulted"
            else "Every site is bi-allelic: ALT holds a single allele."
        )
        evidence = (
            ("README: some sites are multi-allelic (comma-separated alleles in ALT)",)
            if twin == "faulted"
            else ("README: every site is bi-allelic",)
        )
    else:
        evidence = (
            ("ALT holds comma-separated alleles at several sites", "README: joint-called")
            if twin == "faulted"
            else ("every ALT holds a single allele",)
        )
    (work / "README.md").write_text(common.readme(lines))
    prompt = common.prompt_for(
        route,
        "count the alternate alleles in variants.vcf.gz, counting each allele listed in the ALT "
        "column separately",
        "use `bcftools view -H` to list the records of variants.vcf.gz and count the alternate "
        "alleles in them, counting each allele listed in the ALT column separately",
        "a JSON integer",
    )
    expected, naive = truth_counts(plan, twin)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F9", seed, twin, tier, route)
    return Task(task_id, "F9", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference_f9(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["bcftools", "query", "-f", "%ALT\n", "variants.vcf.gz"], work)
    count = sum(len(line.split(",")) for line in proc.stdout.splitlines() if line)
    return _ok(count, "bcftools query %ALT, split on commas")


def naive_f9(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work)
    return _ok(len(proc.stdout.splitlines()), "bcftools view -H | wc -l")


# Exposure: a command counted the rows of the VCF (wc -l, grep -c, bcftools index -n, NR) without
# splitting ALT. Any quoted comma, split(), awk -F, or bcftools norm -m marks an allele-aware
# command; a count of other files (ls | wc -l) does not name variants.vcf.
ROW_COUNT = re.compile(
    r"\bwc\s+-l\b|\bgrep\s+-\w*c|\bindex\s+(?:\S+\s+)*(?:-n|--nrecords)\b|\bEND\s*\{\s*print\s+NR"
)
ALLELE_SPLIT = re.compile(
    r"""['"/],['"/]|-F\s*['"]?,|\bsplit\(|\.split\(|\bnorm\b.*(?:-m|--multiallelics)|\bNF\b"""
)


def exposed_f9(commands: list[str], work: Path) -> bool:
    return any(
        "variants.vcf" in c and ROW_COUNT.search(c) and not ALLELE_SPLIT.search(c) for c in commands
    )


SPEC = FamilySpec(
    id="F9",
    slot="record_count",
    source="VCF multi-allelic records (one row, several ALT alleles)",
    build=build_f9,
    reference=reference_f9,
    naive=naive_f9,
    exposed=exposed_f9,
    twin_files=TWIN_FILES,
)
