"""F57 subset_monomorphic (record_count, tool default): a sample subset keeps monomorphic sites.

Selecting samples does not select variants. `bcftools view -s/-S`, `vcftools --keep ... --recode`,
`plink --keep ... --make-bed` and `gatk SelectVariants -sn` all write every site of the input for
the chosen samples, including the sites at which none of them carries an alternate allele any more.
`bcftools view -s` recomputes INFO/AC and AN (AC=0 there) but leaves AF alone; the monomorphic sites
stay unless `-c 1` (bcftools), `--mac 1` (vcftools, plink), `--exclude-non-variants` (GATK) says
otherwise. Exit status 0, nothing on stderr about the sites.

Task: count the variant sites of cohort.vcf.gz at which at least one of the samples listed in
group_b.txt carries an alternate allele. Both twins carry the same 8 samples and the same group B
(4 of them) with 22-30 sites variable in group B. The faulted twin adds 6-10 sites at which only
samples outside group B carry the alternate allele, which the subset keeps as all-reference sites.

Verified here (bcftools 1.24, vcftools 0.1.17, plink 1.9, GATK 4.6.2.0; four samples s1..s4, six
sites of which b has no alternate allele in s1 and s2): `bcftools view -s s1,s2` writes 6 sites,
site b with AC=0 and the old AF; `-c1` writes 5. `vcftools --keep s1+s2 --recode` writes 6, with
`--mac 1`, `--maf 0.0001`, `--non-ref-ac 1` or `--non-ref-ac-any 1` 5. `plink --keep --make-bed`
writes 6 variants, with `--mac 1` 5 ("1 variant removed due to minor allele threshold(s)").
`gatk SelectVariants -sn s1 -sn s2` writes 4 of 4 records (AC=0 at the monomorphic one),
`--exclude-non-variants` 3. `bcftools view -s s1,s2 -e 'AC==0'` (also -i 'AC>0' and
-i 'COUNT(GT="alt")>0') keeps b as well: bcftools evaluates -i/-e on the full cohort before it
selects the samples, so only `-c 1` drops it. On the family data the counts differ by the number
of added sites.

  explicit  the README says that subsetting keeps the sites that are monomorphic in the subset
  implicit  the README only describes the files; group B's genotypes at those sites are the evidence
  forced    the sentence asks for `bcftools view -S`
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
    pipelines,
    result,
    vcf_text,
    write_vcf_gz,
)
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 2400, "chr2": 1800}
SAMPLES = [f"s{i}" for i in range(1, 9)]
N_B = 4
DOSAGE = {0: "0/0", 1: "0/1", 2: "1/1"}


def make_plan(seed: int):
    """(group B sample names, sites): a site is (contig, pos, ref, alt, genotypes, kind), kind
    'b' (variable in group B) or 'a' (alternate allele only outside group B)."""
    rng = random.Random(f"f57-{seed}")
    b_idx = sorted(rng.sample(range(len(SAMPLES)), N_B))
    n_b, n_a = rng.randint(22, 30), rng.randint(6, 10)
    kinds = ["b"] * n_b + ["a"] * n_a
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
        while True:
            dose = [rng.choice([0, 0, 1, 2]) for _ in SAMPLES]
            in_b = [dose[i] for i in b_idx]
            outside = [dose[i] for i in range(len(SAMPLES)) if i not in b_idx]
            if kind == "b" and any(in_b) and not all(d == 2 for d in in_b):
                break
            if kind == "a" and not any(in_b) and any(outside):
                break
        sites.append((chrom, pos, ref, alt, tuple(DOSAGE[d] for d in dose), kind))
    return [SAMPLES[i] for i in b_idx], sites


def rows(sites, twin: str) -> list[str]:
    out = []
    n = 0
    for chrom, pos, ref, alt, gts, kind in sites:
        if kind == "a" and twin == "clean":
            continue
        out.append(f"{chrom}\t{pos}\tv{n:02d}\t{ref}\t{alt}\t50\tPASS\t.\tGT\t" + "\t".join(gts))
        n += 1
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    group_b, sites = make_plan(seed)
    write_vcf_gz(work, vcf_text(CONTIGS, SAMPLES, rows(sites, twin), FORMAT_GT), tools, "cohort")
    (work / "group_b.txt").write_text("".join(f"{s}\n" for s in group_b))
    lines = [
        "cohort.vcf.gz (with cohort.vcf.gz.tbi): SNVs of eight samples (s1..s8), contigs chr1 and "
        "chr2.",
        "group_b.txt: the samples of group B, one name per line.",
    ]
    if tier == "explicit":
        lines.append(
            "Selecting samples (bcftools view -s/-S, vcftools --keep, plink --keep, GATK -sn) "
            "writes every site of the input, also the sites at which none of the selected "
            "samples carries an alternate allele."
        )
    (work / "README.md").write_text(common.readme(lines))
    n_a = sum(1 for s in sites if s[5] == "a")
    if twin == "clean":
        evidence = ("every site of cohort.vcf.gz is variable in group B",)
    else:
        evidence = (
            f"{n_a} sites carry the alternate allele only in samples outside group B",
            "selecting group B keeps those sites with all-reference genotypes (AC=0)",
        )
    prompt = common.prompt_for(
        route,
        "count the variant sites in cohort.vcf.gz at which at least one of the samples listed in "
        "group_b.txt carries an alternate allele",
        "use `bcftools view -S group_b.txt` to select the samples listed in group_b.txt of "
        "cohort.vcf.gz and count the variant sites at which at least one of them carries an "
        "alternate allele",
        "a JSON integer",
    )
    expected = sum(1 for s in sites if s[5] == "b")
    naive = expected + (n_a if twin == "faulted" else 0)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F57", seed, twin, tier, route)
    return Task(task_id, "F57", "record_count", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def subset_sites(work: Path, tools: ToolEnv, *options: str) -> int:
    argv = ["bcftools", "view", "-H", "-S", "group_b.txt", *options, "cohort.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return len(proc.stdout.splitlines())


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(subset_sites(work, tools, "-c", "1"), "bcftools view -S group_b.txt -c 1")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(subset_sites(work, tools), "bcftools view -S group_b.txt")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a pipeline (commands joined by pipes) that selects samples (bcftools view/query
# -s/-S, vcftools --keep/--indv, plink --keep, gatk SelectVariants -sn/-sf) and has no allele-count
# filter anywhere in it (-c, --min-ac, --mac, --maf, --non-ref-ac, --exclude-non-variants, or a
# second bcftools stage with -c / an AC or GT="alt" test: after a pipe the test sees the subset;
# inside bcftools view -s the same test runs before the subset and does not count). Per pipeline;
# a short option may sit in a bundle (`-Hs s1,s2`, `-Hc1`).
SUBSET = re.compile(
    r"\bbcftools\s+(?:view|query)\b[^;&\n]*?\s(?:-[A-Za-z]*[sS]|--samples(?:-file)?)[\s=]\S"
    r"|\bvcftools\b[^;&\n]*--(?:keep|indv|remove|remove-indv)\b"
    r"|\bplink2?\b[^;&\n]*--(?:keep|remove)\b"
    r"|\bSelectVariants\b[^;&\n]*(?:-sn|-sf|--sample-name|--sample-file)\b"
)
FILTERED = re.compile(
    r"\s-[A-Za-z]*c\s*[1-9]|--min-ac[\s=][1-9]"
    r"|--mac\s+[1-9]|--maf\s+0*\.0*[1-9]|--non-ref-a[cf](?:-any)?\b|--exclude-non-variants"
)
PIPED_TEST = re.compile(
    r"\|\s*bcftools\s+(?:view|filter)\b[^|;&\n]*(?:\s-[ie]\s|--(?:include|exclude))"
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for pipeline in pipelines(command):
            if SUBSET.search(pipeline) and not FILTERED.search(pipeline):
                if not PIPED_TEST.search(pipeline):  # a later stage tests the subset's records
                    return True
    return False


SPEC = FamilySpec(
    id="F57",
    slot="record_count",
    source="bcftools view / vcftools / plink / GATK SelectVariants manuals (sample subsetting)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cohort.vcf.gz", "cohort.vcf.gz.tbi"}),
    locus="tool_default",
)
