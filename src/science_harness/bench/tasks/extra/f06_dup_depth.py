"""F6 dup_depth (default_param, GEN-32): samtools depth and mpileup silently skip duplicate reads.

The task asks how many alignment records overlap each listed position, counting every record
whatever its flags. Clean twin: no record carries a flag, so depth equals the record count.
Faulted twin: the same records, but each position is covered by one or two PCR copies flagged 0x400,
which depth, samtools mpileup and bcftools mpileup leave out by default. Their output is valid and
exits 0; depth is lower than the count at every listed position. samtools view -c with a region
counts every record.

The evidence the agent can see: the README, `samtools flagstat` (duplicates), and the help text of
samtools depth, which lists DUP among the flags it filters out by default.
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

READ_LEN = 30
DUP = 0x400
SLOTS = 4  # listed positions per contig
CONTIGS = {"chr1": (1000, 1500), "chr2": (800, 1200)}  # total <= 2.7 kb
# the calls that apply the default read filter (samtools depth -g / mpileup --ff change it)
FILTERED_CALL = re.compile(r"\b(?:samtools\s+(?:depth|mpileup)|bcftools\s+mpileup)\b")
FILTER_OVERRIDE = re.compile(r"\s-g\s*\S|\s--ff[\s=]")


@dataclass(frozen=True)
class Rec:
    name: str
    chrom: str
    start: int  # 1-based
    reverse: bool
    duplicate: bool  # flagged 0x400 in the faulted twin only


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    positions: list[tuple[str, int]]  # 1-based
    records: list[Rec]


def plan(seed: int) -> Plan:
    rnd = random.Random(f"f06-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    positions: list[tuple[str, int]] = []
    records: list[Rec] = []

    def add(chrom: str, start: int, reverse: bool, duplicate: bool = False) -> None:
        records.append(Rec(f"q{len(records):03d}", chrom, start, reverse, duplicate))

    for chrom, n in contigs.items():
        width = n // SLOTS
        for k in range(SLOTS):
            base = k * width
            pos = base + rnd.randint(60, 100)
            positions.append((chrom, pos))
            starts = rnd.sample(range(pos - READ_LEN + 1, pos + 1), rnd.randint(3, 5))
            fragments = [(s, rnd.random() < 0.5) for s in starts]
            for start, reverse in fragments:
                add(chrom, start, reverse)
            for _ in range(rnd.randint(1, 2)):  # PCR copies of a fragment: same start and strand
                start, reverse = rnd.choice(fragments)
                add(chrom, start, reverse, duplicate=True)
            for _ in range(2):  # reads clear of the listed position
                add(chrom, base + rnd.randint(130, 150), rnd.random() < 0.5)
    return Plan(contigs, positions, records)


def counts(p: Plan, twin: str) -> tuple[list[int], list[int]]:
    """(records overlapping each position, those depth keeps: the unflagged ones)."""
    covering = [
        [r for r in p.records if r.chrom == c and r.start <= pos < r.start + READ_LEN]
        for c, pos in p.positions
    ]
    everything = [len(rs) for rs in covering]
    kept = [sum(not (r.duplicate and twin == "faulted") for r in rs) for rs in covering]
    return everything, kept


def sam_text(seed: int, p: Plan, twin: str) -> str:
    rnd = random.Random(f"f06-{seed}-seq")
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in p.contigs.items()
    )
    rows = []
    for r in p.records:
        flag = (16 if r.reverse else 0) | (DUP if r.duplicate and twin == "faulted" else 0)
        seq = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
        rows.append(
            f"{r.name}\t{flag}\t{r.chrom}\t{r.start}\t60\t{READ_LEN}M\t*\t0\t0\t{seq}\t"
            + "I" * READ_LEN
        )
    return header + "\n".join(rows) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    p = plan(seed)
    common.write_bam(work, sam_text(seed, p, twin), tools)
    (work / "positions.txt").write_text("".join(f"{c}:{pos}\n" for c, pos in p.positions))
    explicit = tier == "explicit"
    if twin == "clean":
        status = (
            "No read is flagged as a duplicate (0x400)."
            if explicit
            else "Duplicates were not marked."
        )
        evidence = (
            ("README states that no read is flagged as a duplicate",)
            if explicit
            else ("README says duplicates were not marked",)
        )
    else:
        status = (
            "Some reads are flagged as duplicates (0x400)."
            if explicit
            else "Duplicates are marked."
        )
        evidence = (
            ("README states that some reads are flagged as duplicates (0x400)",)
            if explicit
            else ("README says duplicates are marked",)
        ) + (
            "`samtools flagstat reads.bam` reports the duplicates",
            "`samtools depth` help lists DUP among the flags it filters out by default",
        )
    readme = [
        "reads.bam (with reads.bam.bai): single-end reads on chr1 and chr2, coordinate-sorted. "
        + status,
        "positions.txt: positions of interest, one per line as chrom:pos (1-based).",
    ]
    (work / "README.md").write_text(common.readme(readme))
    prompt = common.prompt_for(
        route,
        "for each position listed in positions.txt, report how many alignment records in "
        "reads.bam overlap it, counting every record whatever its flags",
        "use `samtools depth` to report, for each position listed in positions.txt, how many "
        "alignment records in reads.bam overlap it, counting every record whatever its flags",
        "a JSON list of integers, one per position, in the order of positions.txt",
    )
    everything, kept = counts(p, twin)
    truth = {"kind": "int_list", "expected": everything, "naive": kept}
    task_id = common.task_id("F6", seed, twin, tier, route)
    return Task(task_id, "F6", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def positions_of(work: Path) -> list[tuple[str, int]]:
    rows = (line.split(":") for line in (work / "positions.txt").read_text().split())
    return [(chrom, int(pos)) for chrom, pos in rows]


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    answer = []
    for chrom, pos in positions_of(work):
        proc = tools.run(["samtools", "view", "-c", "reads.bam", f"{chrom}:{pos}-{pos}"], work)
        answer.append(int(proc.stdout.strip()))
    return _result(answer, "samtools view -c on a one-base region: counts every record")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    answer = []
    for chrom, pos in positions_of(work):
        proc = tools.run(
            ["samtools", "depth", "-a", "-r", f"{chrom}:{pos}-{pos}", "reads.bam"], work
        )
        answer.append(int(proc.stdout.split("\t")[2]))
    return _result(answer, "samtools depth at each position, default filters")


def exposed(commands: list[str], work: Path) -> bool:
    """True iff a command ran a depth or pileup call whose default filter drops duplicates.

    A call that changes the filter (samtools depth -g, mpileup --ff) took the safe route, as did
    samtools view -c or reading the records in Python.
    """
    return any(FILTERED_CALL.search(c) and not FILTER_OVERRIDE.search(c) for c in commands)


SPEC = FamilySpec(
    id="F6",
    slot="default_param",
    source="GEN-32",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "README.md"}),
)
