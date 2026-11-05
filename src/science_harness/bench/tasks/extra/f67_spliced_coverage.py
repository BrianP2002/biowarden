"""F67 spliced_coverage (split, tool default): bedtools counts the intron of a spliced read.

`bedtools genomecov -ibam`, `coverage`, `multicov` and `intersect` take an alignment as the
interval from its first to its last aligned base. Without `-split` a spliced read (CIGAR
`40M300N20M`) therefore covers the 300 bases it skipped. Nothing is printed: exit 0, empty
stderr, a table that looks like coverage. (`samtools depth` does not count the N bases; this is
bedtools-specific.)

Verified here (bedtools 2.31.1, samtools 1.24): two reads on chr1, `101 20M300N20M` and
`111 40M`, region chr1 100-500: `genomecov -ibam -bg` gives 100-110:1, 110-150:2, 150-440:1
(the intron counted), with `-split` 100-110:1, 110-120:2, 120-150:1, 420-440:1; `genomecov -d`
positive in 340 positions without `-split`, 70 with it; `coverage -a reg.bed -b reads.bam` gives
count 2, covered bases 340, fraction 0.85, with `-split` count 3, covered 70, fraction 0.175
(`-mean` 0.95 and 0.2).

The task asks for the number of intron bases covered by at least one read, from introns.bed. The
twins hold the same intronic reads (a few retained-intron reads per intron) and the same exonic
reads; the faulted twin adds junction reads (CIGAR `kM<intron>N(L-k)M`) that cross every intron
and cover no intronic base. The expected answer is therefore the same in both twins. Every read is
named r0000, r0001, ... in a random order: a name does not mark a junction read.

  explicit  the README says that spliced reads carry N in the CIGAR
  implicit  the README only says "RNA-seq reads aligned to the genome"
  forced    the sentence asks for `bedtools coverage`
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import pipelines
from science_harness.bench.tasks.extra.f64_picard_dups import read_names
from science_harness.bench.tasks.tools import ToolEnv

CHROMS = {"chr1": 12000, "chr2": 8000}
GENES = {"chr1": 3, "chr2": 2}
SPACING = 4000


@dataclass(frozen=True)
class Read:
    name: str
    chrom: str
    pos0: int
    blocks: tuple[tuple[int, int], ...]  # aligned blocks, 0-based half-open
    minus: bool

    def cigar(self) -> str:
        out = ""
        for i, (s, e) in enumerate(self.blocks):
            if i:
                out += f"{s - self.blocks[i - 1][1]}N"
            out += f"{e - s}M"
        return out

    def sam(self) -> str:
        length = sum(e - s for s, e in self.blocks)
        flag = 16 if self.minus else 0
        return (
            f"{self.name}\t{flag}\t{self.chrom}\t{self.pos0 + 1}\t60\t{self.cigar()}\t*\t0\t0\t"
            f"{'A' * length}\t{'I' * length}\n"
        )


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f67-{seed}")
    introns: list[tuple[str, int, int, str]] = []
    exonic: list[Read] = []
    retained: list[Read] = []
    junction: list[Read] = []
    n_read = 0

    def name(prefix: str) -> str:
        nonlocal n_read
        n_read += 1
        return f"{prefix}{n_read:04d}"

    gene_no = 0
    for chrom, count in GENES.items():
        for k in range(count):
            gene_no += 1
            cursor = 400 + k * SPACING + rng.randint(0, 200)
            exons = []
            for _ in range(3):
                size = rng.randint(220, 340)
                exons.append((cursor, cursor + size))
                cursor += size + rng.randint(420, 700)
            for s, e in exons:
                for _ in range(rng.randint(14, 24)):
                    length = rng.randint(50, 76)
                    start = rng.randint(s, e - length)
                    exonic.append(
                        Read(
                            name("e"), chrom, start, ((start, start + length),), rng.random() < 0.5
                        )
                    )
            for i in range(2):
                a, b = exons[i][1], exons[i + 1][0]
                introns.append((chrom, a, b, f"g{gene_no:02d}_intron{i + 1}"))
                for _ in range(rng.choice([0, 0, 3, 4, 6])):
                    length = rng.randint(50, 76)
                    start = rng.randint(a + 5, b - length - 5)
                    retained.append(
                        Read(
                            name("i"), chrom, start, ((start, start + length),), rng.random() < 0.5
                        )
                    )
                for _ in range(rng.randint(6, 10)):
                    length = rng.randint(60, 76)
                    left = rng.randint(18, length - 18)
                    start = a - left
                    blocks = ((start, a), (b, b + length - left))
                    junction.append(Read(name("j"), chrom, start, blocks, rng.random() < 0.5))
    return {"introns": introns, "exonic": exonic, "retained": retained, "junction": junction}


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def covered_intron_bases(plan: dict[str, Any]) -> int:
    """Intron bases touched by an aligned block (N and unaligned bases do not count)."""
    total = 0
    for chrom, a, b, _ in plan["introns"]:
        blocks = [
            blk
            for r in (*plan["exonic"], *plan["retained"], *plan["junction"])
            if r.chrom == chrom
            for blk in r.blocks
        ]
        for s, e in merged(blocks):
            total += max(0, min(e, b) - max(s, a))
    return total


def sam_text(plan: dict[str, Any], twin: str, seed: int) -> str:
    reads = [*plan["exonic"], *plan["retained"], *(plan["junction"] if twin == "faulted" else [])]
    reads.sort(key=lambda r: (list(CHROMS).index(r.chrom), r.pos0, r.name))
    # one name pattern for every read (the plan's e/i/j prefixes only order the draw)
    names = read_names(f"f67-names-{seed}", len(reads))
    header = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in CHROMS.items()
    )
    return header + "".join(replace(r, name=n).sam() for r, n in zip(reads, names, strict=True))


def covered_sum(tools: ToolEnv, work: Path, *extra: str) -> int:
    """Sum of the 'bases covered' column of `bedtools coverage -a introns.bed -b reads.bam`."""
    argv = ["bedtools", "coverage", "-a", "introns.bed", "-b", "reads.bam", *extra]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sum(int(line.split("\t")[5]) for line in proc.stdout.splitlines() if line.strip())


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    common.write_bam(work, sam_text(plan, twin, seed), tools, "reads")
    (work / "introns.bed").write_text(
        "".join(f"{c}\t{a}\t{b}\t{n}\n" for c, a, b, n in plan["introns"])
    )
    lines = [
        "reads.bam (with reads.bam.bai): RNA-seq reads aligned to the genome.",
        "introns.bed: the introns of five genes, BED4 (chrom, start, end, name).",
    ]
    if tier == "explicit":
        lines.append("Spliced reads carry N in their CIGAR string.")
    (work / "README.md").write_text(common.readme(lines))
    expected = covered_intron_bases(plan)
    naive = expected if twin == "clean" else covered_sum(tools, work)
    if twin == "clean":
        evidence: tuple[str, ...] = ()
    else:
        evidence = (
            f"{len(plan['junction'])} reads in reads.bam are spliced (CIGAR N) across introns: "
            "bedtools without -split counts the skipped bases as covered",
        )
    what = (
        "the number of intron bases (bases inside the intervals of introns.bed) covered by at "
        "least one read"
    )
    prompt = common.prompt_for(
        route,
        f"report {what}",
        f"use `bedtools coverage` to report {what}",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F67", seed, twin, tier, route)
    return Task(task_id, "F67", "split", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(covered_sum(tools, work, "-split"), "bedtools coverage -split, covered bases")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(covered_sum(tools, work), "bedtools coverage, covered bases")


SUBCOMMAND = re.compile(r"\bbedtools\s+(?:coverage|genomecov|multicov|intersect|window)\b")
BAMTOBED = re.compile(r"\bbedtools\s+bamtobed\b")
SPLIT = re.compile(r"(?<![\w-])-split\b")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command runs a bedtools coverage-type subcommand over a BAM without -split, or
    over a BED that `bedtools bamtobed` made from the BAM without -split earlier in the same
    command (the BED interval of a spliced read spans its intron). Judged per pipeline."""
    for command in commands:
        converted = False
        for part in pipelines(command):
            if BAMTOBED.search(part) and not SPLIT.search(part):
                converted = True
            if SUBCOMMAND.search(part) and not SPLIT.search(part):
                if converted or re.search(r"\.bam\b|-ibam|-abam", part):
                    return True
    return False


SPEC = FamilySpec(
    id="F67",
    slot="split",
    source="bedtools -split semantics (genomecov/coverage count the N gap of a spliced read)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai"}),
    locus="tool_default",
)
