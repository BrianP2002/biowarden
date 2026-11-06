"""F62 isec_repr (callset_representation, data): isec misses an indel written right-aligned.

`bcftools isec` matches records by position and alleles (`-c none` is the default; `-c some`,
`indels`, `all` relax the allele test but still compare POS). The same deletion or insertion in a
repeat can be written at different positions: left-aligned (the VCF convention: the leftmost
position, anchor base before the run) or right-aligned. Two callers that disagree on the alignment
share none of their indels as far as `isec` can see, and nothing says so (exit 0, "Note: -w
option not given" at most). `bcftools norm -f ref.fa` left-aligns both files and the indels match.

Task: how many variants did both callers call (the same variant, however a caller writes it)?
Both twins carry the same reference, the same SNVs and the same indels in homopolymer runs of
3-6 bases, called by both callers. The clean twin has both callers writing the indels left-aligned.
In the faulted twin caller B writes them right-aligned: the shared count by position and alleles
drops by the number of indels (8-12).

Verified here (bcftools 1.24): a one-base deletion in a run of six T at 8-13 is `GT>G` at POS 7
left-aligned and `TT>T` at POS 12 right-aligned. `isec -n=2 -c none|some|indels|all` on the
two files finds only the SNV; after `bcftools norm -f ref.fa` on both files (`Lines
total/split/joined/realigned/...: 2/0/0/1/0/0/0` for the right-aligned one) it finds both.

  explicit  the README says that caller B writes indels right-aligned
  implicit  the README only names the files and the reference
  forced    the sentence asks for `bcftools isec -n=2`
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
    pipelines,
    result,
    vcf_text,
    write_vcf_gz,
)
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 1800, "chr2": 1400}


def make_plan(seed: int):
    """(genome, snvs, indels): snvs (contig, pos, ref, alt); indels (contig, run_start0, run_len,
    kind) with kind 'del' or 'ins' of one base of the run."""
    rng = random.Random(f"f62-{seed}")
    genome = {c: [rng.choice("ACGT") for _ in range(n)] for c, n in CONTIGS.items()}
    n_ind = rng.randint(8, 12)
    slots = []
    for chrom, size in CONTIGS.items():
        slots += [(chrom, s) for s in range(60, size - 100, 90)]
    chosen = sorted(
        rng.sample(slots, n_ind + 16), key=lambda cs: (list(CONTIGS).index(cs[0]), cs[1])
    )
    ind_slots, snv_slots = chosen[:n_ind], chosen[n_ind:]
    indels = []
    for chrom, start in ind_slots:
        run = rng.randint(3, 6)
        base = rng.choice("ACGT")
        for k in range(run):
            genome[chrom][start + k] = base
        other = [b for b in "ACGT" if b != base]
        genome[chrom][start - 1] = rng.choice(other)  # the anchor differs from the run
        genome[chrom][start + run] = rng.choice(other)
        indels.append((chrom, start, run, rng.choice(["del", "ins"])))
    snvs = []
    for chrom, start in snv_slots:
        pos = start + rng.randint(30, 50)
        ref = genome[chrom][pos - 1]
        snvs.append((chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref])))
    return {c: "".join(s) for c, s in genome.items()}, snvs, indels


def indel_record(genome, indel, right: bool):
    chrom, start0, run, kind = indel
    seq = genome[chrom]
    base = seq[start0]
    if kind == "del":
        if right:  # anchor = the second-to-last base of the run
            pos0 = start0 + run - 2
            return chrom, pos0 + 1, base + base, base
        return chrom, start0, seq[start0 - 1] + base, seq[start0 - 1]  # POS = anchor (1-based)
    if right:
        return chrom, start0 + run, base, base + base
    return chrom, start0, seq[start0 - 1], seq[start0 - 1] + base


def vcf_rows(genome, snvs, indels, right: bool) -> list[str]:
    records = [(c, p, r, a) for c, p, r, a in snvs]
    records += [indel_record(genome, i, right) for i in indels]
    order = {c: i for i, c in enumerate(CONTIGS)}
    records.sort(key=lambda r: (order[r[0]], r[1]))
    return [f"{c}\t{p}\t.\t{r}\t{a}\t50\tPASS\t." for c, p, r, a in records]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    genome, snvs, indels = make_plan(seed)
    common.write_fasta(work, genome, tools)
    write_vcf_gz(
        work, vcf_text(CONTIGS, [], vcf_rows(genome, snvs, indels, False)), tools, "caller_a"
    )
    right = twin == "faulted"
    write_vcf_gz(
        work, vcf_text(CONTIGS, [], vcf_rows(genome, snvs, indels, right)), tools, "caller_b"
    )
    lines = [
        "caller_a.vcf.gz, caller_b.vcf.gz (each with .tbi): the variants two callers called on "
        "the same sample, sites only, contigs chr1 and chr2.",
        "ref.fa (with ref.fa.fai): the reference both callers used.",
    ]
    if tier == "explicit":
        lines.append(
            "Caller A writes indels left-aligned; caller B writes them right-aligned."
            if twin == "faulted"
            else "Both callers write indels left-aligned."
        )
    (work / "README.md").write_text(common.readme(lines))
    n_ind = len(indels)
    if twin == "clean":
        evidence = ("both callers write every indel at the same position",)
    else:
        evidence = (
            f"caller B writes {n_ind} indels in homopolymer runs at the right end of the run, "
            "caller A at the left end",
            "bcftools isec compares position and alleles as written: shifted indels do not match",
        )
    prompt = common.prompt_for(
        route,
        "count the variants that both callers called (the same variant, however a caller "
        "writes it)",
        "use `bcftools isec -n=2` to count the variants that both callers called (the same "
        "variant, however a caller writes it)",
        "a JSON integer",
    )
    expected = len(snvs) + n_ind
    naive = len(snvs) if twin == "faulted" else expected
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F62", seed, twin, tier, route)
    return Task(
        task_id, "F62", "callset_representation", twin, seed, prompt, truth, evidence, tier, route
    )


# ---- solvers -----------------------------------------------------------------------------------


def shared_count(work: Path, tools: ToolEnv, a: str, b: str, *options: str) -> int:
    argv = ["bcftools", "isec", "-n=2", "-w1", *options, a, b]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sum(1 for line in proc.stdout.splitlines() if line and not line.startswith("#"))


def normalised(work: Path, tools: ToolEnv, name: str) -> str:
    out = name.replace(".vcf.gz", ".norm.vcf.gz")
    for argv in (
        ["bcftools", "norm", "-f", "ref.fa", "-Oz", "-o", out, name],
        ["bcftools", "index", "-f", "-t", out],
    ):
        common.check(tools.run(argv, work), argv)
    return out


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    a, b = normalised(work, tools, "caller_a.vcf.gz"), normalised(work, tools, "caller_b.vcf.gz")
    return result(
        shared_count(work, tools, a, b), "bcftools norm -f ref.fa on both, then isec -n=2"
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = shared_count(work, tools, "caller_a.vcf.gz", "caller_b.vcf.gz")
    return result(n, "bcftools isec -n=2 on the two files as they are")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped, per pipeline: the two caller files compared as given, that is a bcftools isec or stats
# that names caller_a.vcf.gz and caller_b.vcf.gz (or a glob of both), or `comm` over listings of
# the two, without a norm before the comparison (`bcftools stats A B` reports the shared records
# by position and alleles, like isec). Safe: a comparison of normalised copies (norm in the
# pipeline, or names with norm/left/leftalign), or no comparison (bcftools norm alone, view, stats
# of one file).
COMPARE = re.compile(
    r"\b(?:bcftools\s+(?:isec|stats)|comm)\b.*?"
    r"(?:caller_a\S*.*?caller_b|caller_b\S*.*?caller_a|caller_[*?\[]\S*)"
)
NORMALISED = re.compile(r"norm|left|align|\bnormalised\b|\bnormalized\b", re.IGNORECASE)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for pipeline in pipelines(command):
            hit = COMPARE.search(pipeline)
            if hit and not NORMALISED.search(pipeline[: hit.end()]):
                return True
    return False


SPEC = FamilySpec(
    id="F62",
    slot="callset_representation",
    source="GEN-28 (indel representation across callers); bcftools isec / norm manuals",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"caller_b.vcf.gz", "caller_b.vcf.gz.tbi", "README.md"}),
    locus="data",
)
