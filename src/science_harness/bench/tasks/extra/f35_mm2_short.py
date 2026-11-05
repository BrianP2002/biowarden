"""F35 mm2_short (default_param): minimap2 without a short-read preset loses reads under ~60 bases.

minimap2 is tuned for long reads: its defaults need a chain of at least 3 minimizers (k=15, w=10)
and a DP alignment score of 80 (match score 2 gives 40 matching bases at the very least, more once
the ends and the chaining threshold are counted). `-x sr` lowers the thresholds (-s 40, -m 20) for
short reads. Without it, a 50-base read that matches the genome exactly may come out unmapped;
`-x map-ont` and the other long-read presets behave like the default. Exit status 0, valid SAM, a
mapped fraction that merely looks lower. It does not warn that reads are too short for the preset.

Task: how many reads of reads.fastq map to each contig. All reads are exact pieces of the genome at
a unique place. Clean twin: reads of 100 bases (the default maps all of them). Faulted twin: the
same reads cut to 47-52 bases (adapter-trimmed ChIP-seq style), which `-x sr` still maps in full.

Verified here (minimap2 2.31, samtools 1.24, 100 exact reads per length at unique places of a 4 kb
genome), reads mapped by default / with -x sr:
    38: 0/76   40: 2/84   42: 6/92   44: 14/100   46: 29/100   48: 46/100   50: 62/100
    52: 72/100  55: 92/100  60: 100/100  70: 100/100
and with -x map-ont the same as the default (28 of 40 at 50 bases). bwa mem loses nothing above 30
bases and bowtie2 nothing at all on the same reads.

  explicit  the README says the reads are 47-52 bases (faulted) or 100 (clean) after trimming
  implicit  the README only says the reads were adapter-trimmed
  forced    the sentence asks for minimap2
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
from science_harness.bench.tasks.extra.f29_mapq_unique import (
    fastq_text,
    rand_seq,
    revcomp,
    sam_records,
    write_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

SHORT = (47, 52)
LONG = 100
CONTIGS = {"chr1": (1800, 2200), "chr2": (1400, 1800)}


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    reads: list[tuple[str, str, str, int]]  # (name, contig, sequence of 100 bases, short length)


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f35-{seed}")
    genome = {c: rand_seq(rng, rng.randint(lo, hi)) for c, (lo, hi) in CONTIGS.items()}
    reads = []
    for i in range(rng.randint(80, 100)):
        contig = rng.choice(list(genome))
        start = rng.randint(0, len(genome[contig]) - LONG)
        seq = genome[contig][start : start + LONG]
        seq = revcomp(seq) if rng.random() < 0.5 else seq
        reads.append((f"read{i:03d}", contig, seq, rng.randint(*SHORT)))
    return Plan(genome, reads)


def read_list(plan: Plan, twin: str) -> list[tuple[str, str]]:
    return [(n, s[:k] if twin == "faulted" else s) for n, _, s, k in plan.reads]


def mapped_per_contig(sam_text: str, contigs: list[str]) -> dict[str, int]:
    names: dict[str, set[str]] = {c: set() for c in contigs}
    for f in sam_records(sam_text):
        if not int(f[1]) & 0x904 and f[2] in names:
            names[f[2]].add(f[0])
    return {c: len(s) for c, s in names.items()}


def _minimap2(work: Path, tools: ToolEnv, *options: str) -> dict[str, int]:
    argv = ["minimap2", "-a", "-t", "1", *options, "genome.fa", "reads.fastq"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    contigs = [line.split("\t")[0] for line in (work / "genome.fa.fai").read_text().splitlines()]
    return mapped_per_contig(proc.stdout, contigs)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_genome(work, plan.genome, tools)
    reads = read_list(plan, twin)
    (work / "reads.fastq").write_text(fastq_text(reads))
    explicit = tier == "explicit"
    line = "reads.fastq: single-end reads (FASTQ), adapter-trimmed"
    if explicit:
        line += "; 47-52 bases long." if twin == "faulted" else "; 100 bases long."
    else:
        line += "."
    lines = [
        line,
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2.",
        "Every read was cut from the genome, so every read has a place on chr1 or chr2.",
    ]
    (work / "README.md").write_text(common.readme(lines))
    expected: dict[str, int] = {c: 0 for c in plan.genome}
    for _, contig, _, _ in plan.reads:
        expected[contig] += 1
    naive = _minimap2(
        work, tools
    )  # the default run is the trap answer; the same call on both twins
    lengths = [len(s) for _, s in reads]
    if twin == "clean":
        evidence = (f"all reads are {min(lengths)} bases or longer, as minimap2's defaults expect",)
    else:
        evidence = (
            f"reads.fastq holds reads of {min(lengths)}-{max(lengths)} bases (awk length)",
            "minimap2's defaults are tuned for long reads; -x sr is the short-read preset",
        )
    prompt = common.prompt_for(
        route,
        "report, for each contig, how many reads of reads.fastq map to it (every read is a piece "
        "of the genome)",
        "use minimap2 to align reads.fastq to genome.fa and report, for each contig, how many "
        "reads map to it (every read is a piece of the genome)",
        "a JSON object mapping each contig name to its read count",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0}
    task_id = common.task_id("F35", seed, twin, tier, route)
    return Task(task_id, "F35", "default_param", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_minimap2(work, tools, "-x", "sr"), "minimap2 -a -x sr (short reads)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_minimap2(work, tools), "minimap2 -a with default options")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a minimap2 command without the short-read preset (-x sr, or the ax-sr spelling `-ax sr`).
# Any other preset (map-ont, map-pb, splice, asm5, ...) or none is trapped. Per command.
MINIMAP2 = re.compile(r"\bminimap2\b")
SHORT_PRESET = re.compile(r"(?:(?<=\s)-[A-Za-z]*x\s*|--preset[\s=]\s*)['\"]?sr\b")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if MINIMAP2.search(segment) and not SHORT_PRESET.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F35",
    slot="default_param",
    source="minimap2 manual (presets; -s minimal peak DP alignment score 80, -x sr)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.fastq", "README.md"}),
    locus="tool_default",
)
