"""F30 bwa_short (default_param): `bwa mem` does not output alignments that score below 30.

`bwa mem -T INT` ("minimum score to output", default 30) with the default match score 1 means a
read shorter than 30 bases can never be output as aligned, however perfect the match: a 25-base
read that matches the genome exactly scores 25 and comes out as an unmapped record (flag 4). After
adapter trimming (ChIP-seq, ATAC-seq, CLIP, small RNA) many reads are that short. bwa exits 0, the
SAM is valid and the mapped fraction merely looks lower, so nothing points at the cause.

Task: how many reads of reads.fastq map to each contig. All reads are exact (or nearly exact)
substrings of the genome at a unique place. Clean twin: reads are 40-75 bases. Faulted twin: the
same reads, 40% of them trimmed from the 3' end to 20-28 bases. Same positions, same strands, so
the true count per contig is the same in both twins.

Verified here (bwa 0.7.19, samtools 1.24, 40 exact reads per length, unique loci in a 4 kb genome):
    length   mapped (default)   mapped (-T 15)   bowtie2   minimap2 -x sr   hisat2
    20       0                  40               40        0                40
    25       0                  40               40        0                40
    30       40                 40               40        0                40
    35       40                 40               40        25               40
    50       40                 40               40        40               40
Default `bwa mem` output for a 20-base read: flag 4, no tags. `bwa mem -T 15` aligns it (seed length
19, so reads under 19 bases cannot be seeded at all and need `bwa aln`).

  explicit  the README says the reads are adapter-trimmed with lengths 20-75 (faulted) / 40-75
  implicit  the README only says the reads are adapter-trimmed
  forced    the sentence asks for bwa mem
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
    mutate,
    rand_seq,
    revcomp,
    sam_records,
    write_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

MIN_SCORE = 30  # bwa mem -T default
SHORT = (20, 28)
LONG = (40, 75)
CONTIGS = {"chr1": (1800, 2200), "chr2": (1400, 1800)}


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    reads: list[tuple[str, str, str, int]]  # (name, contig, long sequence, short length or 0)


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f30-{seed}")
    genome = {c: rand_seq(rng, rng.randint(lo, hi)) for c, (lo, hi) in CONTIGS.items()}
    reads = []
    for i in range(rng.randint(80, 100)):
        contig = rng.choice(list(genome))
        length = rng.randint(*LONG)
        start = rng.randint(0, len(genome[contig]) - length)
        seq = genome[contig][start : start + length]
        seq = revcomp(seq) if rng.random() < 0.5 else seq
        # a mismatch only after base 30 of the read as sequenced, so every 3' trimmed prefix
        # (at most 28 bases) is exact
        if rng.random() < 0.3:
            seq = seq[:30] + mutate(rng, seq[30:], 1)
        reads.append(
            (f"read{i:03d}", contig, seq, rng.randint(*SHORT) if rng.random() < 0.4 else 0)
        )
    return Plan(genome, reads)


def read_list(plan: Plan, twin: str) -> list[tuple[str, str]]:
    return [
        (name, seq[:short] if twin == "faulted" and short else seq)
        for name, _, seq, short in plan.reads
    ]


def counts(plan: Plan, twin: str, minimum: int = 0) -> dict[str, int]:
    """Reads per contig whose length is at least `minimum`."""
    out = {c: 0 for c in plan.genome}
    for (_, contig, _, _), (_, seq) in zip(plan.reads, read_list(plan, twin), strict=True):
        out[contig] += len(seq) >= minimum
    return out


def bwa_index(work: Path, tools: ToolEnv) -> None:
    argv = ["bwa", "index", "genome.fa"]
    common.check(tools.run(argv, work), argv)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_genome(work, plan.genome, tools)
    bwa_index(work, tools)
    reads = read_list(plan, twin)
    (work / "reads.fastq").write_text(fastq_text(reads))
    explicit = tier == "explicit"
    low = min(len(seq) for _, seq in reads)
    lengths = "between 20 and 75 bases" if twin == "faulted" else "between 40 and 75 bases"
    line = "reads.fastq: single-end reads (FASTQ), adapter-trimmed"
    line += f"; lengths {lengths}." if explicit else "."
    lines = [
        line,
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2; its bwa index "
        "(genome.fa.amb, .ann, .bwt, .pac, .sa) is in this directory.",
        "Every read was cut from the genome, so every read has a place on chr1 or chr2.",
    ]
    (work / "README.md").write_text(common.readme(lines))
    n_short = sum(len(seq) < MIN_SCORE for _, seq in reads)
    if twin == "clean":
        evidence = (f"the shortest read of reads.fastq has {low} bases, above bwa mem's -T 30",)
    else:
        evidence = (
            f"{n_short} reads of reads.fastq are shorter than 30 bases (the shortest has {low})",
            "bwa mem -T defaults to 30 and the match score to 1, so a read under 30 bases cannot "
            "score 30",
        )
    prompt = common.prompt_for(
        route,
        "report, for each contig, how many reads of reads.fastq map to it (every read is a piece "
        "of the genome)",
        "use bwa mem to align reads.fastq to genome.fa and report, for each contig, how many "
        "reads map to it (every read is a piece of the genome)",
        "a JSON object mapping each contig name to its read count",
    )
    expected = counts(plan, twin)
    naive_counts = counts(plan, twin, MIN_SCORE)
    truth = {"kind": "dict_num", "expected": expected, "naive": naive_counts, "tol": 0}
    task_id = common.task_id("F30", seed, twin, tier, route)
    return Task(task_id, "F30", "default_param", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def mapped_per_contig(sam_text: str, contigs: list[str]) -> dict[str, int]:
    names: dict[str, set[str]] = {c: set() for c in contigs}
    for f in sam_records(sam_text):
        if not int(f[1]) & 0x904 and f[2] in names:
            names[f[2]].add(f[0])
    return {c: len(s) for c, s in names.items()}


def _bwa(work: Path, tools: ToolEnv, *options: str) -> dict[str, int]:
    argv = ["bwa", "mem", "-t", "1", *options, "genome.fa", "reads.fastq"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    contigs = [line.split("\t")[0] for line in (work / "genome.fa.fai").read_text().splitlines()]
    return mapped_per_contig(proc.stdout, contigs)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_bwa(work, tools, "-T", "15"), "bwa mem -T 15 (reads under 30 bases need it)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_bwa(work, tools), "bwa mem with default options, mapped records per contig")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a command that runs `bwa mem` with a -T above 20 (or none): the shortest read of the
# family has 20 bases, so a smaller -T keeps every read. `bwa aln`, other aligners and a lowered -T
# are the safe routes. Per command.
BWA_MEM = re.compile(r"\bbwa\s+mem\b")
T_OPTION = re.compile(r"(?<=\s)-T\s*(\d+)")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if not BWA_MEM.search(segment):
                continue
            values = [int(v) for v in T_OPTION.findall(segment)]
            if not values or values[-1] > SHORT[0]:
                return True
    return False


SPEC = FamilySpec(
    id="F30",
    slot="default_param",
    source="bwa mem manual (-T INT: minimum score to output, default 30)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.fastq", "README.md"}),
    locus="tool_default",
)
