"""F58 plink_a1 (allele_coding, documentation): the lab's plink command leaves A1 = minor allele.

plink 1.9 gives A1 to the MINOR allele of the data whenever it loads genotypes (a VCF, and also a
.bed fileset: the order in the .bim is re-derived at load), unless `--keep-allele-order` is given;
then A1 is ALT for a VCF import. `--freq` writes the frequency of A1 in the MAF column, the
association tests test A1, `--score` and `--recode A` count A1. At a variant where ALT is the major
allele A1 is REF and the MAF column is 1 - ALT frequency. Exit status 0, nothing on stderr.

Task: report the frequency of the ALT allele of every variant, using the lab's plink command given
in the README. Both twins hold the same VCF: ten samples, 28-34 variants with ALT frequencies
0.05-0.95 (ALT is the major allele at 9-13 of them, no ties) and no missing genotype. The twins
differ only in the README command: the clean twin's is
`plink --vcf cohort.vcf.gz --keep-allele-order --freq`, the faulted twin's is
`plink --vcf cohort.vcf.gz --freq` (the option was never added), so its MAF column is the ALT
frequency only where ALT is the minor allele.

Verified here (plink 1.9, plink2 2.0.0-a.7): on ten samples and twelve variants of ALT frequency
0.15-0.85, `plink --vcf V --freq` prints A1/A2 = ALT/REF and MAF = ALT frequency for the sites with
frequency below 0.5, and A1/A2 = REF/ALT with MAF 0.15, 0.45, 0.35, 0.2, 0.2 for the five sites at
0.85, 0.55, 0.65, 0.8, 0.8; `--keep-allele-order` prints ALT/REF everywhere with MAF = ALT
frequency; reloading a fileset written with ALT as A1 gives the minor-first order again (the .bim
does not decide). plink2 `--freq` writes REF, ALT and ALT_FREQS; `bcftools +fill-tags -- -t AF`
agrees with them.

  explicit  the README adds that plink 1.9 puts the minor allele in A1
  implicit  the README only gives the command; ALT-major sites in the VCF are the evidence
  forced    the sentence asks for the README's `plink --freq` command
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
    result,
    segments,
    vcf_text,
    write_vcf_gz,
)
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 2400, "chr2": 1800}
N_SAMPLES = 10
N_ALLELES = 2 * N_SAMPLES
PLINK = ["plink", "--memory", "1000", "--silent"]


def make_plan(seed: int):
    """(sites, mirrored indices): sites are (contig, pos, ref, alt, ALT allele count of 20)."""
    rng = random.Random(f"f58-{seed}")
    n = rng.randint(28, 34)
    places: set[tuple[str, int]] = set()
    while len(places) < n:
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites = []
    for chrom, pos in sorted(places, key=lambda cp: (order[cp[0]], cp[1])):
        ref = rng.choice("ACGT")
        alt = rng.choice([b for b in "ACGT" if b != ref])
        sites.append([chrom, pos, ref, alt, rng.randint(1, 9)])
    mirrored = sorted(rng.sample(range(n), rng.randint(9, 13)))
    for i in mirrored:
        sites[i][4] = N_ALLELES - sites[i][4]  # ALT is the major allele
    return [tuple(s) for s in sites], mirrored


def genotypes(count: int, rng: random.Random) -> list[str]:
    """Ten diploid genotypes carrying `count` ALT alleles, shuffled."""
    alleles = [1] * count + [0] * (N_ALLELES - count)
    rng.shuffle(alleles)
    return [f"{alleles[2 * i]}/{alleles[2 * i + 1]}" for i in range(N_SAMPLES)]


def vcf_for(sites, seed: int) -> str:
    rng = random.Random(f"f58-gt-{seed}")
    rows = [
        f"{c}\t{p}\tv{i:02d}\t{r}\t{a}\t50\tPASS\t.\tGT\t" + "\t".join(genotypes(k, rng))
        for i, (c, p, r, a, k) in enumerate(sites)
    ]
    samples = [f"s{i + 1}" for i in range(N_SAMPLES)]
    return vcf_text(CONTIGS, samples, rows, FORMAT_GT)


LAB_COMMAND = {
    "clean": "plink --vcf cohort.vcf.gz --keep-allele-order --freq --out lab",
    "faulted": "plink --vcf cohort.vcf.gz --freq --out lab",
}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, mirrored = make_plan(seed)
    write_vcf_gz(work, vcf_for(sites, seed), tools, "cohort")
    lines = [
        "cohort.vcf.gz (with cohort.vcf.gz.tbi): SNVs of ten samples, contigs chr1 and chr2; no "
        "genotype is missing.",
        f"Our command for allele frequencies from a VCF: `{LAB_COMMAND[twin]}`; the MAF column "
        "of lab.frq is the frequency we report.",
    ]
    if tier == "explicit":
        lines.append(
            "plink 1.9 puts the minor allele in A1 (and the major allele in A2) unless "
            "--keep-allele-order is given; MAF is the frequency of A1."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = (
            "the README command has --keep-allele-order, so A1 is ALT and MAF is its frequency",
        )
    else:
        evidence = (
            f"the README command has no --keep-allele-order and ALT is the major allele at "
            f"{len(mirrored)} variants, where A1 is REF",
            "plink 1.9 puts the minor allele in A1 and MAF is the frequency of A1",
        )
    prompt = common.prompt_for(
        route,
        "report the frequency of the ALT allele of every variant in cohort.vcf.gz, as a number "
        "between 0 and 1 per variant ID, with the plink command of the README",
        "use the plink command of the README to report the frequency of the ALT allele of every "
        "variant in cohort.vcf.gz, as a number between 0 and 1 per variant ID",
        "a JSON object mapping each variant ID to its ALT allele frequency",
    )
    expected = {f"v{i:02d}": round(s[4] / N_ALLELES, 4) for i, s in enumerate(sites)}
    wrong = {f"v{i:02d}": round(1 - expected[f"v{i:02d}"], 4) for i in mirrored}
    naive = {**expected, **wrong} if twin == "faulted" else expected
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.005}
    task_id = common.task_id("F58", seed, twin, tier, route)
    return Task(task_id, "F58", "allele_coding", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def run_frq(work: Path, tools: ToolEnv, *options: str) -> dict[str, float]:
    """MAF column of `plink --vcf cohort.vcf.gz --freq` by variant ID."""
    argv = [*PLINK, "--vcf", "cohort.vcf.gz", *options, "--freq", "--out", "frq"]
    common.check(tools.run(argv, work), argv)
    rows = [line.split() for line in (work / "frq.frq").read_text().splitlines()[1:]]
    return {r[1]: float(r[4]) for r in rows}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    freq = run_frq(work, tools, "--keep-allele-order")
    return result(freq, "plink --keep-allele-order --freq (A1 = ALT), MAF column")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    readme = (work / "README.md").read_text()
    keep = ["--keep-allele-order"] if "--keep-allele-order --freq" in readme.split("`")[1] else []
    return result(run_frq(work, tools, *keep), "the plink command of the README, MAF column")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a plink 1.9 command that imports the VCF (--vcf) and reports A1-based results (--freq,
# --freqx, --recode, --make-bed, --make-just-bim, --assoc, --linear, --logistic, --score) without
# --keep-allele-order, --a1-allele, --a2-allele or --real-ref-alleles. plink2 and bcftools are
# safe. Applied per command.
PLINK_VCF = re.compile(r"(?:^|[\s(])plink(?:\s|$)(?=[^|;&\n]*--vcf\b)")
A1_OUTPUT = re.compile(
    r"--(?:freq|freqx|recode|make-bed|make-just-bim|assoc|linear|logistic|score)\b"
)
ALLELES_FIXED = re.compile(r"--keep-allele-order|--a1-allele|--a2-allele|--real-ref-alleles")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if (
                PLINK_VCF.search(segment)
                and A1_OUTPUT.search(segment)
                and not ALLELES_FIXED.search(segment)
            ):
                return True
    return False


SPEC = FamilySpec(
    id="F58",
    slot="allele_coding",
    source="GEN-30, R2-OMX-25 (A1/A2 against REF/ALT); plink 1.9 data management manual",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"README.md"}),
    locus="documentation",
)
