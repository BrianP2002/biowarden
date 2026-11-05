"""F81 seqtk_floor: `seqtk trimfq` never trims a read below 30 bases unless told to.

`seqtk trimfq` (Phred algorithm, `-q` error-rate threshold, default 0.05) has a second default,
`-l INT  maximally trim down to INT bp (disabled by -b/-e) [30]`. A read whose good part is shorter
than 30 bases is therefore cut at 30 bases and keeps a low-quality tail of up to 30 - g bases
(g = the good part). Exit status 0, empty stderr, no count of reads that hit the floor.

Verified here (seqtk 1.5-r133; 60-base reads with a high-quality head (Q32-40) of g bases and a Q2-8
tail): `trimfq -q 0.01` returns max(g, 30) bases for the tailed reads, `trimfq -q 0.01 -l 1` (and
`-l 0`) returns exactly g (g = 25 33 40 23 17 18 19 29 57 44 38 20: lengths 30 33 40 30 30 30 30
30 57 44 38 30 by default, 25 33 40 23 17 18 19 29 57 44 38 20 with `-l 1`). On the generated
sets of this family (seeds 0-29) the same holds for every read.

Twins: the same good heads. Clean: the reads are the heads only (delivered already trimmed, so
the floor never binds; 10-90 bases). Faulted: 60% of the reads also carry a Q2-8 tail, and the
heads of 40% of those are 10-26 bases long (60-90 bases before trimming). The answer is the total
number of bases in the heads.

  explicit  the README says that a read is trimmed to its good part however short that is
  implicit  the README says that only the good part of a read counts (it does not name the floor)
  forced    the task sentence dictates seqtk trimfq
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f77_blastn_task import (
    TOOLS_NOTE,
    fastq_text,
    phred,
    rand_dna,
    result,
    run_ok,
    segments,
)
from science_harness.bench.tasks.tools import ToolEnv

FLOOR = 30
THRESHOLD = "0.01"
TWIN_FILES = frozenset({"reads.fastq", "README.md"})


def make_plan(seed: int, twin: str) -> list[tuple[str, str, str, int]]:
    """(name, sequence, quality string, good bases) per read."""
    rng = random.Random(f"f81-{seed}")
    reads = []
    for i in range(rng.randint(40, 60)):
        length = rng.randint(60, 90)
        tailed = rng.random() < 0.6
        short = rng.random() < 0.4
        head = length
        if tailed:
            head = rng.randint(10, 26) if short else rng.randint(32, length - 10)
        tail = length - head
        seq = rand_dna(rng, length)
        quals = [rng.randint(32, 40) for _ in range(head)]
        quals += [rng.randint(2, 8) for _ in range(tail)]
        if twin == "clean":  # already trimmed: the reads are the good heads
            seq, quals = seq[:head], quals[:head]
        reads.append((f"read{i + 1:03d}", seq, phred(quals), head))
    return reads


def trimmed_bases(tools: ToolEnv, work: Path, *options: str) -> int:
    argv = ["seqtk", "trimfq", "-q", THRESHOLD, *options, "reads.fastq"]
    out = run_ok(tools, argv, work).stdout.splitlines()[1::4]
    return sum(len(line) for line in out)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    reads = make_plan(seed, twin)
    (work / "reads.fastq").write_text(fastq_text([(n, s, q) for n, s, q, _ in reads]))
    shortest = 60 if twin == "faulted" else 10  # the clean reads are the good heads
    lines = [
        f"reads.fastq: single-end Illumina reads, {shortest}-90 bases, Phred+33 qualities.",
        "The answer counts only the good part of each read: its low-quality 3' tail is left out.",
    ]
    if tier == "explicit":
        lines.append(
            "Some reads carry a low-quality 3' tail and must be cut to their good part however "
            "short that part is; no read is kept longer than the trimming allows."
            if twin == "faulted"
            else "The reads were trimmed before delivery; no read has a low-quality tail."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sum(head for *_, head in reads)
    naive = sum(max(head, FLOOR) if head < len(seq) else len(seq) for _, seq, _, head in reads)
    shorter = [h for _, seq, _, h in reads if h < FLOOR and h < len(seq)]
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{len(shorter)} reads have a good head shorter than {FLOOR} bases (the shortest is "
            f"{min(shorter)})",
            f"seqtk trimfq keeps at least {FLOOR} bases per read by default (-l {FLOOR})",
        )
    prompt = common.prompt_for(
        route,
        f"quality-trim the 3' ends of the reads in reads.fastq with the error-rate threshold "
        f"{THRESHOLD} per base (the Phred algorithm of seqtk trimfq) and report the total number "
        "of bases left across all reads",
        f"use seqtk trimfq with the error-rate threshold {THRESHOLD} to quality-trim the reads in "
        "reads.fastq and report the total number of bases left across all reads",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F81", seed, twin, tier, route)
    return Task(task_id, "F81", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = trimmed_bases(tools, work, "-l", "1")
    return result(n, "seqtk trimfq -q 0.01 -l 1 (no minimum length kept)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(trimmed_bases(tools, work), "seqtk trimfq -q 0.01 with the default -l 30")


# Exposure: a seqtk trimfq command that trims by quality (no -b/-e/-L, which switch quality
# trimming off) and keeps the default floor, or sets -l to 2 or more. Judged per command.
TRIMFQ = re.compile(r"(?<![\w-])seqtk\s+trimfq\b")
FLOOR_OPTION = re.compile(r"(?<![\w-])-l[ =]*(\d+)")
FIXED = re.compile(r"(?<![\w-])-[beL][ =]*[1-9]")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if not TRIMFQ.search(segment) or FIXED.search(segment):
                continue
            floor = FLOOR_OPTION.findall(segment)
            if floor and int(floor[-1]) <= 1:
                continue
            return True
    return False


SPEC = FamilySpec(
    id="F81",
    slot="default_param",
    source="seqtk trimfq usage: -l INT maximally trim down to INT bp [30]",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
