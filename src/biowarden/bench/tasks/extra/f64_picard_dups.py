"""F64 picard_dups (default_param, tool default): MarkDuplicates marks, it does not remove.

`picard MarkDuplicates I=in.bam O=out.bam M=metrics.txt` writes every read of the input: the
duplicate pairs get flag 0x400 and stay in the file (REMOVE_DUPLICATES defaults to false). A count
of the records of out.bam (`samtools view -c`, `idxstats`, `flagstat` "total", a depth, a coverage
or a read count script that ignores the flag) is a count of the reads BEFORE de-duplication. Exit
0, a valid BAM, the metrics file lists the duplicates and nothing else. (The JVM prints
native-access warnings on stderr on every run, and the INFO log says "Marking 40 records as
duplicates": the only hint that they are marked, not removed.)

Task: report the number of read pairs that remain after removing PCR duplicates with Picard
MarkDuplicates. Both twins carry the same 100-130 distinct pairs (coordinate-sorted BAM, one
library, FR pairs of 250-350 bases). The faulted twin adds 20-30% more pairs that are exact
duplicates of existing ones (same positions, own names); the clean twin has none. All pairs are
named p000, p001, ... in a random order, so a name does not tell a copy from its original. The
answer is the number of distinct pairs in both twins.

Verified here (Picard 3.5.0, samtools 1.24; 120 pairs of which 20 are duplicates): default run ->
240 records written, 40 with flag 0x400, metrics READ_PAIR_DUPLICATES 20 (pairs, not reads) and
READ_PAIRS_EXAMINED 120 (a METRICS line counts pairs); `REMOVE_DUPLICATES=true` -> 200 records.
`samtools view -c -f 0x40` on the default output counts the duplicate first mates too.

  explicit  the README says that Picard flags duplicates unless REMOVE_DUPLICATES=true
  implicit  the README only names the BAM
  forced    the sentence asks for MarkDuplicates and a samtools count of its output
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f29_mapq_unique import rand_seq, revcomp
from biowarden.bench.tasks.extra.f53_vcftools_bed import result, segments
from biowarden.bench.tasks.tools import ToolEnv

CONTIG = ("chr1", 3000)
READ_LEN = 50
JAVA = "-Xmx512m"


def read_names(key: str, count: int, prefix: str = "r", width: int = 4) -> list[str]:
    """`count` names of one pattern (prefix and a zero-padded number) in a seeded random order, so
    that the name of a read says nothing about the read: a copy or an added read is not labelled.
    F67 and F75 name their reads with it too."""
    order = list(range(count))
    random.Random(key).shuffle(order)
    return [f"{prefix}{n:0{width}d}" for n in order]


def make_plan(seed: int):
    """(genome, distinct pairs (start0, fragment), duplicate indices into the pairs)."""
    rng = random.Random(f"f64-{seed}")
    genome = rand_seq(rng, CONTIG[1])
    n = rng.randint(100, 130)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < n:
        frag = rng.randint(250, 350)
        start = rng.randint(0, CONTIG[1] - frag)
        if (start, frag) not in seen and all(abs(start - s) > 3 for s, _ in pairs):
            seen.add((start, frag))
            pairs.append((start, frag))
    dups = sorted(rng.sample(range(n), round(n * rng.uniform(0.2, 0.3))))
    return genome, pairs, dups


def sam_text(genome: str, pairs, dups, twin: str, seed: int) -> str:
    header = (
        "@HD\tVN:1.6\tSO:unsorted\n"
        f"@SQ\tSN:{CONTIG[0]}\tLN:{CONTIG[1]}\n"
        "@RG\tID:rg1\tSM:sample\tLB:lib1\tPL:ILLUMINA\n"
    )
    fragments = list(pairs) + ([pairs[i] for i in dups] if twin == "faulted" else [])
    # one name pattern for every pair, numbered in a random order: a copy is not labelled
    entries = zip(read_names(f"f64-names-{seed}", len(fragments), "p", 3), fragments, strict=True)
    rows = []
    for name, (start, frag) in entries:
        left = genome[start : start + READ_LEN]
        right = revcomp(genome[start + frag - READ_LEN : start + frag])
        p1, p2 = start + 1, start + frag - READ_LEN + 1
        qual = "I" * READ_LEN
        rows.append(
            f"{name}\t99\t{CONTIG[0]}\t{p1}\t60\t{READ_LEN}M\t=\t{p2}\t{frag}\t{left}\t{qual}\tRG:Z:rg1"
        )
        rows.append(
            f"{name}\t147\t{CONTIG[0]}\t{p2}\t60\t{READ_LEN}M\t=\t{p1}\t{-frag}\t{right}\t{qual}"
            "\tRG:Z:rg1"
        )
    return header + "\n".join(rows) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    genome, pairs, dups = make_plan(seed)
    common.write_bam(work, sam_text(genome, pairs, dups, twin, seed), tools, "reads")
    lines = [
        "reads.bam (with reads.bam.bai): paired-end reads of 50 bases from one library (lib1), "
        "aligned to chr1, coordinate-sorted."
    ]
    if tier == "explicit":
        lines.append(
            "Picard MarkDuplicates sets flag 0x400 on duplicate reads and keeps them in the "
            "output unless REMOVE_DUPLICATES=true."
        )
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("no read pair shares its start and fragment length with another pair",)
    else:
        evidence = (
            f"{len(dups)} read pairs repeat the positions of another pair (PCR duplicates)",
            "Picard MarkDuplicates keeps the duplicates in its output, flagged 0x400, by default",
        )
    prompt = common.prompt_for(
        route,
        "report the number of read pairs of reads.bam that remain after removing PCR duplicates "
        "with Picard MarkDuplicates",
        "run Picard MarkDuplicates on reads.bam and use samtools to count the read pairs in its "
        "output after removing PCR duplicates",
        "a JSON integer",
    )
    expected = len(pairs)
    naive = expected + (len(dups) if twin == "faulted" else 0)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F64", seed, twin, tier, route)
    return Task(task_id, "F64", "default_param", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------
# Picard is a JVM program (5-9 s per call), so the scripted solvers count with samtools on the
# records MarkDuplicates would write: a default run writes every read, REMOVE_DUPLICATES=true
# writes the distinct pairs (the real runs are in tests/test_family_var.py and the admission cases).


def first_mates(work: Path, tools: ToolEnv, *options: str) -> int:
    argv = ["samtools", "view", "-c", "-f", "0x40", *options, "reads.bam"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return int(proc.stdout)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = distinct_pairs(work, tools)
    return result(n, "MarkDuplicates REMOVE_DUPLICATES=true: pairs left after the removal")


def distinct_pairs(work: Path, tools: ToolEnv) -> int:
    proc = tools.run(["samtools", "view", "-f", "0x40", "reads.bam"], work)
    common.check(proc, ["samtools", "view"])
    keys = {(f[3], f[8]) for f in (line.split("\t") for line in proc.stdout.splitlines())}
    return len(keys)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(first_mates(work, tools), "MarkDuplicates (default), first mates of its output")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped, per command: a MarkDuplicates run (picard or gatk) without REMOVE_DUPLICATES=true, or
# a count of its output records without a flag test (-F 1024 / -F 0x400 / -f 0x400 / -e flag).
# Safe: REMOVE_DUPLICATES=true / --REMOVE_DUPLICATES true, a samtools count with -F 0x400/1024.
MARK = re.compile(r"\b(?:picard|gatk)\b[^|;&\n]*\bMarkDuplicates(?:Spark)?\b")
REMOVE = re.compile(r"REMOVE_DUPLICATES(?:=|\s+)true", re.IGNORECASE)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if MARK.search(segment) and not REMOVE.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F64",
    slot="default_param",
    source="Picard MarkDuplicates documentation (REMOVE_DUPLICATES defaults to false)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "README.md"}),
    locus="tool_default",
)
