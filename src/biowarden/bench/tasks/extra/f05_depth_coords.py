"""F5 depth_coords (index_base, GEN-03): samtools depth prints 1-based positions, always.

The task asks for the first position where the read depth reaches 3, "in the same coordinates as
the regions file". Clean twin: the regions are samtools strings (chrom:start-end, 1-based, both ends
included), so depth's position is the answer as printed. Faulted twin: the regions are a BED file
(0-based start), so the answer is depth's position minus one; the naive agent reports it raw, one
base too far, with exit 0 and no warning (also with -b, which reads the BED file correctly).

The grader requires the same expected answer in both twins, so the faulted BAM is the clean one
shifted one base to the right: the same answer numbers, expressed in the other convention. Depth
first reaches 3 strictly inside every region, region by region, so it never reaches 3 earlier.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

READ_LEN = 20
THRESHOLD = 3
SLOTS = 3  # regions per contig
CONTIGS = {"chr1": (1000, 1500), "chr2": (800, 1200)}  # total <= 2.7 kb
# samtools depth and samtools mpileup both print 1-based positions
DEPTH_CALL = re.compile(r"\bsamtools\s+(?:depth|mpileup)\b")


@dataclass(frozen=True)
class Cluster:
    """One region plus the answer inside it and the offsets of the reads that build the depth."""

    chrom: str
    start: int  # the region numbers: samtools start (1-based) or BED start (0-based)
    end: int  # samtools end (inclusive) or BED end (exclusive): the same number
    answer: int  # first position at depth >= 3, in the regions file's own coordinates
    back: tuple[int, int]  # the two reads that already cover the answer start this far before it
    ahead: int  # a fourth read starts this far after it


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    clusters: list[Cluster]
    background: list[tuple[str, int]]  # lone reads between the clusters (1-based start)


def plan(seed: int) -> Plan:
    rnd = random.Random(f"f05-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    clusters: list[Cluster] = []
    background: list[tuple[str, int]] = []
    for chrom, n in contigs.items():
        width = n // SLOTS
        for k in range(SLOTS):
            base = k * width
            start = base + rnd.randint(30, 60)
            end = start + rnd.randint(30, 80) - 1
            answer = rnd.randint(start + 3, end - 3)  # strictly inside the region
            back = (rnd.randint(10, 16), rnd.randint(2, 8))
            clusters.append(Cluster(chrom, start, end, answer, back, rnd.randint(1, 5)))
            background.append((chrom, base + rnd.randint(200, 230)))
    return Plan(contigs, clusters, background)


def read_starts(p: Plan, twin: str) -> list[tuple[str, int]]:
    """(contig, 1-based start) of every read; the faulted BAM sits one base to the right."""
    shift = 1 if twin == "faulted" else 0
    starts = list(p.background)
    for c in p.clusters:
        pos = c.answer + shift
        starts += [(c.chrom, pos - c.back[0]), (c.chrom, pos - c.back[1])]
        starts += [(c.chrom, pos), (c.chrom, pos + c.ahead)]
    return sorted(starts, key=lambda cs: (list(p.contigs).index(cs[0]), cs[1]))


def first_depth(starts: list[tuple[str, int]], chrom: str, lo: int, hi: int) -> int:
    """First 1-based position in [lo, hi] covered by at least THRESHOLD reads."""
    for pos in range(lo, hi + 1):
        if sum(c == chrom and s <= pos < s + READ_LEN for c, s in starts) >= THRESHOLD:
            return pos
    raise ValueError(f"depth never reaches {THRESHOLD} in {chrom}:{lo}-{hi}")


def answers(p: Plan, twin: str) -> tuple[list[int], list[int]]:
    """(expected, raw): depth's 1-based first positions, and those in the regions file's base."""
    starts = read_starts(p, twin)
    shift = 1 if twin == "faulted" else 0  # a BED start is one below the region's first base
    raw = [first_depth(starts, c.chrom, c.start + shift, c.end) for c in p.clusters]
    return [r - shift for r in raw], raw


def sam_text(seed: int, p: Plan, twin: str) -> str:
    rnd = random.Random(f"f05-{seed}-seq")
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in p.contigs.items()
    )
    rows = []
    for i, (chrom, start) in enumerate(read_starts(p, twin)):
        seq = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
        rows.append(
            f"r{i:03d}\t0\t{chrom}\t{start}\t60\t{READ_LEN}M\t*\t0\t0\t{seq}\t{'I' * READ_LEN}"
        )
    return header + "\n".join(rows) + "\n"


def regions_text(p: Plan, twin: str) -> str:
    if twin == "clean":
        return "".join(f"{c.chrom}:{c.start}-{c.end}\n" for c in p.clusters)
    return "".join(f"{c.chrom}\t{c.start}\t{c.end}\n" for c in p.clusters)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    p = plan(seed)
    common.write_bam(work, sam_text(seed, p, twin), tools)
    explicit = tier == "explicit"
    if twin == "clean":
        name = "regions.txt"
        layout = "one per line as chrom:start-end"
        convention = "samtools region syntax" + (
            ": positions count from 1 (1-based, both ends included)" if explicit else ""
        )
        evidence = (
            ("README states samtools region syntax, 1-based, both ends included",)
            if explicit
            else ("README names samtools region syntax",)
        )
    else:
        name = "regions.bed"
        layout = "tab-separated: chrom, start, end"
        convention = "BED format" + (
            ": positions count from 0 (start 0-based, end exclusive)" if explicit else ""
        )
        evidence = (
            ("README states BED coordinates: 0-based start, end exclusive",)
            if explicit
            else ("README names BED format",)
        ) + ("regions.bed has a .bed extension",)
    (work / name).write_text(regions_text(p, twin))
    readme = [
        "reads.bam (with reads.bam.bai): single-end reads, all forward strand, MAPQ 60, "
        "contigs chr1 and chr2",
        f"{name}: target regions, {layout}. Coordinates are in {convention}.",
    ]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "for each target region, report the first position at which the read depth in reads.bam "
        "is at least 3, in the same coordinates as the regions file",
        "use `samtools depth` to find, for each target region, the first position at which the "
        "read depth in reads.bam is at least 3, reported in the same coordinates as the regions "
        "file",
        "a JSON list of integers, one per region, in the order of the regions file",
    )
    expected, raw = answers(p, twin)
    truth = {"kind": "int_list", "expected": expected, "naive": raw}
    task_id = common.task_id("F5", seed, twin, tier, route)
    return Task(task_id, "F5", "index_base", twin, seed, prompt, truth, evidence, tier, route)


def regions_of(work: Path) -> tuple[list[tuple[str, int, int]], bool]:
    """(chrom, first, last) as 1-based inclusive for each region in file order, and is-BED."""
    bed = work / "regions.bed"
    if bed.exists():
        rows = [line.split("\t") for line in bed.read_text().splitlines() if line.strip()]
        return [(c, int(s) + 1, int(e)) for c, s, e in rows], True
    regions = []
    for line in (work / "regions.txt").read_text().split():
        chrom, span = line.split(":")
        first, last = span.split("-")
        regions.append((chrom, int(first), int(last)))
    return regions, False


def depth_rows(stdout: str) -> list[tuple[str, int, int]]:
    rows = (line.split("\t") for line in stdout.splitlines() if line.strip())
    return [(r[0], int(r[1]), int(r[2])) for r in rows]


def first_at_least(rows: list[tuple[str, int, int]], chrom: str, lo: int, hi: int) -> int:
    return next(pos for c, pos, d in rows if c == chrom and lo <= pos <= hi and d >= THRESHOLD)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    regions, is_bed = regions_of(work)
    answer = []
    for chrom, lo, hi in regions:
        proc = tools.run(["samtools", "depth", "-r", f"{chrom}:{lo}-{hi}", "reads.bam"], work)
        answer.append(first_at_least(depth_rows(proc.stdout), chrom, lo, hi) - (1 if is_bed else 0))
    return _result(answer, "samtools depth -r; 1-based position converted to the file's base")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    regions, is_bed = regions_of(work)
    if is_bed:
        rows = depth_rows(
            tools.run(["samtools", "depth", "-b", "regions.bed", "reads.bam"], work).stdout
        )
        answer = [first_at_least(rows, chrom, lo, hi) for chrom, lo, hi in regions]
        return _result(answer, "samtools depth -b, positions as printed")
    answer = []
    for chrom, lo, hi in regions:
        proc = tools.run(["samtools", "depth", "-r", f"{chrom}:{lo}-{hi}", "reads.bam"], work)
        answer.append(first_at_least(depth_rows(proc.stdout), chrom, lo, hi))
    return _result(answer, "samtools depth -r, positions as printed")


def exposed(commands: list[str], work: Path) -> bool:
    """True iff a command ran samtools depth or mpileup, whose printed position is always 1-based.

    Whether the agent then subtracted one is the grader's business; running the call is exposure.
    Reading the BAM with samtools view and counting in Python or awk is the route around it.
    """
    return any(DEPTH_CALL.search(command) for command in commands)


SPEC = FamilySpec(
    id="F5",
    slot="index_base",
    source="GEN-03",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "regions.txt", "regions.bed", "README.md"}),
)
