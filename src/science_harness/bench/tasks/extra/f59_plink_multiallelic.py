"""F59 plink_multiallelic (record_count, tool default): plink 1.9 --vcf drops a second ALT.

plink 1.9 reads a VCF site with two ALT alleles as a biallelic variant: the first ALT is the
alternate allele and every genotype call that uses the second ALT (0/2, 2/2, 1/2) becomes a missing
call. The log does not mention it ("N variants loaded", the genotyping rate just comes out lower);
only `--biallelic-only` skips such sites, with a note. Quality filters then act on the invented
missing calls: `--geno 0.1` removes a multiallelic site at which a few samples carry the second ALT
although the VCF has no missing call there. plink2 keeps every allele (`plink2 --geno 0.1` keeps
the site, `plink2 --make-bed` refuses multiallelic variants loudly).

Task: keep the variants with at most 10% missing genotype calls and report how many remain. Both
twins carry the same 20 samples and the same sites. Every site has 0 or 1 missing call (to keep)
or 4-7 (to drop). In the faulted twin 6-8 of the sites to keep are multiallelic (ALT A,T) with
3-5 samples whose call is 0/2, 1/2 or 2/2; in the clean twin the same cells hold 0/1 (a biallelic
site). The VCF has no more missing calls at those sites than before, so the answer is the same,
but plink 1.9 sees 15-25% missing there and drops them.

Verified here (plink 1.9, plink2 2.0.0-a.7, bcftools 1.24): on a six-variant VCF with site c
(G -> A,T; calls 0/1, 0/2, 1/1, 0/1) `plink --vcf --geno 0.1 --make-just-bim` removes c and b (b has
a real ./.) while `plink2 --vcf --geno 0.1` removes b only; the plink log shows "Total genotyping
rate is 0.916667" against 0.958333 for the real missing calls, with no warning; `--biallelic-only`
prints "(1 variant skipped.)". `bcftools view -i 'F_MISSING<=0.1'` gives the truth.

  explicit  the README says that plink 1.9 keeps only the first ALT allele of a site
  implicit  the README only describes the VCF; the ALT column with two alleles is the evidence
  forced    the sentence asks for plink 1.9 with `--geno`
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
PLINK = ["plink", "--memory", "1000", "--silent"]


def make_plan(seed: int):
    """Sites (contig, pos, ref, alt1, alt2, genotypes, keep) with the genotype cells of the
    multiallelic sites that hold the second ALT (index set), and the multiallelic site indices."""
    rng = random.Random(f"f59-{seed}")
    n = rng.randint(30, 36)
    n_keep = rng.randint(18, 24)
    keep = [True] * n_keep + [False] * (n - n_keep)
    rng.shuffle(keep)
    places: set[tuple[str, int]] = set()
    while len(places) < n:
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    for (chrom, pos), kept in zip(
        sorted(places, key=lambda cp: (order[cp[0]], cp[1])), keep, strict=True
    ):
        ref = rng.choice("ACGT")
        alts = rng.sample([b for b in "ACGT" if b != ref], 2)
        n_missing = rng.choice([0, 0, 1]) if kept else rng.randint(4, 7)
        gts = [rng.choice(["0/0", "0/1", "1/1"]) for _ in range(N_SAMPLES)]
        for i in rng.sample(range(N_SAMPLES), n_missing):
            gts[i] = "./."
        sites.append([chrom, pos, ref, alts[0], alts[1], gts, kept])
    kept_idx = [i for i, s in enumerate(sites) if s[6]]
    multi = sorted(rng.sample(kept_idx, rng.randint(6, 8)))
    second = {}
    for i in multi:
        ok = [j for j, g in enumerate(sites[i][5]) if g != "./."]
        second[i] = sorted(rng.sample(ok, rng.randint(3, 5)))
    return sites, multi, second


def rows(sites, multi: list[int], second: dict[int, list[int]], twin: str) -> list[str]:
    out = []
    for i, (chrom, pos, ref, alt1, alt2, gts, _keep) in enumerate(sites):
        gts = list(gts)
        alt = alt1
        if i in multi and twin == "faulted":
            alt = f"{alt1},{alt2}"
            for j in second[i]:
                gts[j] = rng_second(i, j)
        elif i in multi:
            for j in second[i]:
                gts[j] = "0/1"
        out.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\tPASS\t.\tGT\t" + "\t".join(gts))
    return out


def rng_second(i: int, j: int) -> str:
    """A call of the second ALT allele, fixed by the site and the sample (no random state)."""
    return ("0/2", "1/2", "2/2")[(i + j) % 3]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, multi, second = make_plan(seed)
    samples = [f"s{i + 1}" for i in range(N_SAMPLES)]
    text = vcf_text(CONTIGS, samples, rows(sites, multi, second, twin), FORMAT_GT)
    write_vcf_gz(work, text, tools, "cohort")
    lines = [
        "cohort.vcf.gz (with cohort.vcf.gz.tbi): SNVs of twenty samples, contigs chr1 and chr2."
    ]
    if tier == "explicit":
        lines.append(
            "Some sites are multiallelic (two ALT alleles); plink 1.9 --vcf keeps only the first "
            "ALT allele and makes the calls of the second one missing."
            if twin == "faulted"
            else "Every site is biallelic (one ALT allele)."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("every site has a single ALT allele",)
    else:
        evidence = (
            f"{len(multi)} sites have two ALT alleles, with calls of the second one (0/2, 1/2)",
            "plink 1.9 --vcf turns the calls of the second ALT into missing genotypes",
        )
    prompt = common.prompt_for(
        route,
        "keep the variants of cohort.vcf.gz with at most 10% missing genotype calls and report "
        "how many variants remain",
        "use plink 1.9 with `--geno` to keep the variants of cohort.vcf.gz with at most 10% "
        "missing genotype calls and report how many variants remain",
        "a JSON integer",
    )
    expected = sum(1 for s in sites if s[6])
    naive = expected - (len(multi) if twin == "faulted" else 0)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F59", seed, twin, tier, route)
    return Task(task_id, "F59", "record_count", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def plink_geno_count(work: Path, tools: ToolEnv, *options: str) -> int:
    argv = [*PLINK, "--vcf", "cohort.vcf.gz", *options, "--geno", "0.1", "--make-just-bim"]
    argv += ["--out", "geno"]
    common.check(tools.run(argv, work), argv)
    return len((work / "geno.bim").read_text().splitlines())


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bcftools", "view", "-H", "-i", "F_MISSING<=0.1", "cohort.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return result(len(proc.stdout.splitlines()), "bcftools view -i 'F_MISSING<=0.1'")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(plink_geno_count(work, tools), "plink --vcf --geno 0.1, variants in the .bim")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a plink 1.9 command that imports the VCF (--vcf) and filters or writes variants
# (--geno, --mind, --maf, --missing, --make-bed, --make-just-bim, --freq, --recode, --write-snplist)
# without --biallelic-only. plink2 and bcftools are safe. Applied per command.
PLINK_VCF = re.compile(r"(?:^|[\s(])plink(?:\s|$)(?=[^|;&\n]*--vcf\b)")
ACTS = re.compile(r"--(?:geno|mind|maf|missing|make-bed|make-just-bim|freq|recode|write-snplist)\b")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if (
                PLINK_VCF.search(segment)
                and ACTS.search(segment)
                and "--biallelic-only" not in segment
            ):
                return True
    return False


SPEC = FamilySpec(
    id="F59",
    slot="record_count",
    source="plink 1.9 --vcf manual (multiallelic sites); GEN-31 (missingness thresholds)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cohort.vcf.gz", "cohort.vcf.gz.tbi", "README.md"}),
    locus="tool_default",
)
