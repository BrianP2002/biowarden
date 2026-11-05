"""F63 merge_absent (missing_value, tool default): `bcftools merge` turns an absent site into ./.

Merging single-sample VCFs gives a multi-sample VCF whose genotype at a site one input does not
list is missing (`./.`), not homozygous reference. For VCFs that list every site (joint calls,
gVCF-style output) that is the right reading. For variants-only calls from complete coverage, a
site that a sample does not list is homozygous reference: `bcftools merge -0` (`--missing-to-ref`)
says so. Without -0 an allele frequency computed from the merged file (INFO/AF from `bcftools
+fill-tags`, or AC/AN) divides by the called alleles only, so a variant carried by one sample of
six gets frequency 0.5 instead of 1/12. Exit 0, no message.

Task: report the ALT allele frequency among all six samples of every variant of the panel. Both
twins carry six single-sample VCFs (sample1..sample6) over the same 14-18 panel sites. The clean
twin's VCFs list every site (genotype 0/0 where the sample carries no ALT allele). The faulted
twin's VCFs are variants-only: a site with genotype 0/0 is not written, and the README says the
coverage was complete. The truth (alternate alleles over 12) is the same in both twins; merging
the faulted files without -0 divides by the called alleles of the samples that list the site.

Verified here (bcftools 1.24): three single-sample VCFs, site a in all, b in smp2 and smp3, c in
smp3 only: `merge` gives ./. for the absent samples (b: F_MISSING 0.333333, AN 4, AF 0.75; c: AN
2, AF 0.5) and `merge -0` gives 0/0 (AN 6, AF 0.5 and 0.166667).

  explicit  the README says that a site a sample does not list is homozygous reference
  implicit  the README only says that the VCFs are variants-only calls
  forced    the sentence asks for `bcftools merge` of the six files
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
N_SAMPLES = 6


def make_plan(seed: int):
    """Sites (contig, pos, ref, alt) and the genotype of every sample at every site."""
    rng = random.Random(f"f63-{seed}")
    n = rng.randint(14, 18)
    places: set[tuple[str, int]] = set()
    while len(places) < n:
        chrom = rng.choice(list(CONTIGS))
        places.add((chrom, rng.randint(20, CONTIGS[chrom] - 20)))
    order = {c: i for i, c in enumerate(CONTIGS)}
    sites, genotypes = [], []
    for chrom, pos in sorted(places, key=lambda cp: (order[cp[0]], cp[1])):
        ref = rng.choice("ACGT")
        sites.append((chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref])))
        carriers = rng.randint(1, 3)
        who = set(rng.sample(range(N_SAMPLES), carriers))
        genotypes.append(
            [rng.choice(["0/1", "1/1"]) if s in who else "0/0" for s in range(N_SAMPLES)]
        )
    return sites, genotypes


def sample_rows(sites, genotypes, sample: int, twin: str) -> list[str]:
    rows = []
    for i, (chrom, pos, ref, alt) in enumerate(sites):
        gt = genotypes[i][sample]
        if twin == "faulted" and gt == "0/0":
            continue  # variants-only calls
        rows.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\tPASS\t.\tGT\t{gt}")
    return rows


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    sites, genotypes = make_plan(seed)
    for s in range(N_SAMPLES):
        text = vcf_text(
            CONTIGS, [f"sample{s + 1}"], sample_rows(sites, genotypes, s, twin), FORMAT_GT
        )
        write_vcf_gz(work, text, tools, f"sample{s + 1}")
    lines = [
        "sample1.vcf.gz .. sample6.vcf.gz (each with .tbi): one single-sample VCF per sample, "
        "SNVs of a panel of sites on chr1 and chr2."
    ]
    if twin == "faulted":
        lines.append(
            "These are variants-only calls from complete 30x coverage"
            + (
                ": a site a sample does not list is homozygous reference in that sample."
                if tier == "explicit"
                else "."
            )
        )
    else:
        lines.append(
            "Every file lists every site of the panel, with genotype 0/0 where the sample carries "
            "no ALT allele."
        )
    (work / "README.md").write_text(common.readme(lines))
    absent = sum(1 for g in genotypes for x in g if x == "0/0")
    if twin == "clean":
        evidence = ("every sample file lists every site of the panel",)
    else:
        evidence = (
            f"the sample files are variants-only: {absent} sample-site pairs are not listed",
            "bcftools merge writes ./. for a site an input does not list unless -0 is given",
        )
    prompt = common.prompt_for(
        route,
        "report the frequency of the ALT allele among all six samples for every variant ID of "
        "the panel (an allele count over the 12 alleles of the six samples)",
        "merge the six sample VCFs with `bcftools merge` and report the frequency of the ALT "
        "allele among all six samples for every variant ID of the panel (an allele count over "
        "the 12 alleles of the six samples)",
        "a JSON object mapping each variant ID to its ALT allele frequency",
    )
    expected = {}
    naive = {}
    for i, g in enumerate(genotypes):
        alt = sum({"0/0": 0, "0/1": 1, "1/1": 2}[x] for x in g)
        expected[f"v{i:02d}"] = round(alt / (2 * N_SAMPLES), 4)
        listed = [x for x in g if x != "0/0"] if twin == "faulted" else g
        naive[f"v{i:02d}"] = round(alt / (2 * len(listed)), 4)
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.005}
    task_id = common.task_id("F63", seed, twin, tier, route)
    return Task(task_id, "F63", "missing_value", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def merged_frequencies(work: Path, tools: ToolEnv, *options: str) -> dict[str, float]:
    files = [f"sample{s + 1}.vcf.gz" for s in range(N_SAMPLES)]
    steps = (
        ["bcftools", "merge", *options, "-Oz", "-o", "merged.vcf.gz", *files],
        ["bcftools", "+fill-tags", "merged.vcf.gz", "-Oz", "-o", "tagged.vcf.gz", "--", "-t", "AF"],
        ["bcftools", "query", "-f", "%ID\t%AF\n", "tagged.vcf.gz"],
    )
    proc = None
    for argv in steps:
        proc = tools.run(argv, work)
        common.check(proc, argv)
    assert proc is not None
    return {line.split("\t")[0]: float(line.split("\t")[1]) for line in proc.stdout.splitlines()}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(merged_frequencies(work, tools, "-0"), "bcftools merge -0, +fill-tags AF")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(merged_frequencies(work, tools), "bcftools merge, +fill-tags AF")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped, per command: a bcftools merge of the sample files without -0 / --missing-to-ref.
# Safe: -0, --missing-to-ref, or no merge (python counting the files, bcftools query per file).
MERGE = re.compile(r"\bbcftools\s+merge\b")
MISSING_TO_REF = re.compile(r"\s-[A-Za-z]*0|--missing-to-ref")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if MERGE.search(segment) and not MISSING_TO_REF.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F63",
    slot="missing_value",
    source="bcftools merge manual (-0 / --missing-to-ref: genotypes at missing sites are 0/0)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset(
        {f"sample{s}.vcf.gz{ext}" for s in range(1, 7) for ext in ("", ".tbi")} | {"README.md"}
    ),
    locus="tool_default",
)
