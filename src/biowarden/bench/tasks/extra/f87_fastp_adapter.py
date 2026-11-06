"""F87 fastp_adapter: fastp's automatic adapter detection for single-end reads needs 10,000 reads.

fastp's README: "For SE data, the adapter sequence auto-detection is enabled by default". The
detection only runs on a large sample: with fewer than 10,000 reads it never reports an adapter,
so `fastp -i reads.fastq -o out.fastq` trims nothing. The exit status is 0, the report says
"reads with adapter trimmed: 0", and the stderr line "No adapter detected for read1" is the only
trace. An explicit `-a AGATCGGAAGAGC` trims.

Verified here (fastp 1.4.0; 100-base reads, 75% with a 25-85 base insert followed by the TruSeq
adapter AGATCGGAAGAGCACACGTC...): detection found the adapter at 10,000 reads and 10,001 reads and
did not at 9,000, 9,500, 9,900 or 9,999 (also with 30% adapter reads: 10,000 detected, 8,000 not);
on the generated sets of this family (250-600 reads, seeds 0-39) it never detects. Paired-end data
are different: the overlap analysis trims adapters without any detection (400 pairs: mean length
55.6 from 100).

Twins: the same inserts. Clean: the reads are the inserts only (delivered already trimmed).
Faulted: 75% of the reads are insert + the whole TruSeq Read 1 adapter (33 bases, the sequence an
expert types) + the C that follows it in the library + sequencing noise, cut at 100 bases. Every
adapter length people type then trims exactly: the first 12, 13, 20, 33 and 34 bases of that
sequence each return the inserts of every read (`draw_reads` redraws inserts until they do, on both
twins), so a careful `-a` call passes whatever adapter it names. The first version of this family
put only the 13-base stem and noise behind each insert: the 12- and 13-base calls were exact, the
20- and 33-base ones left 9-20 thousand adapter bases in (54174 and 54464 against 35752 on seed 0),
so the textbook adapter scored wrong. The answer is the total number of insert bases.

  explicit  the README gives the adapter sequence and says the reads still carry it (faulted)
  implicit  the README only says the library is a TruSeq library
  forced    the task sentence dictates fastp
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f77_blastn_task import (
    TOOLS_NOTE,
    fastq_text,
    phred,
    rand_dna,
    result,
    run_ok,
    segments,
)
from biowarden.bench.tasks.tools import ToolEnv

ADAPTER = "AGATCGGAAGAGC"  # the 13-base stem: reference(), the explicit README and the rule
TRUSEQ_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"  # TruSeq Read 1 adapter, 33 bases
CARRIED = TRUSEQ_R1 + "C"  # what a read runs into: the adapter and the base after it
PREFIXES = (12, 13, 20, 33, 34)  # adapter lengths that must each trim exactly (see draw_reads)
READ_LEN = 100
TWIN_FILES = frozenset({"reads.fastq", "README.md"})


def read_text(inserts: list[str], noise: list[str], quals: list[str], twin: str) -> str:
    reads = []
    for i, (insert, tail, q) in enumerate(zip(inserts, noise, quals, strict=True)):
        sequence = insert
        if twin == "faulted" and len(insert) < READ_LEN:
            sequence = (insert + CARRIED + tail)[:READ_LEN]
        if twin == "clean":
            q = q[: len(sequence)]
        reads.append((f"read{i + 1:04d}", sequence, q[: len(sequence)]))
    return fastq_text(reads)


def fastp_reads(tools: ToolEnv, fastq: str, *options: str) -> dict[str, str]:
    """Name -> sequence of the reads `fastp -i reads.fastq <options>` keeps (own temp directory)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "reads.fastq").write_text(fastq)
        argv = ["fastp", "-i", "reads.fastq", "-o", "o.fastq", "-j", "o.json", "-h", "o.html"]
        run_ok(tools, [*argv, "--thread", "1", *options], work)
        lines = (work / "o.fastq").read_text().splitlines()
    return {lines[i][1:]: lines[i + 1] for i in range(0, len(lines), 4)}


@functools.lru_cache(maxsize=256)
def draw_reads(
    tools: ToolEnv, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Inserts, noise and qualities for which `fastp -a <adapter>` returns exactly the inserts
    for the 12, 13, 20, 33 and 34 first bases of the carried adapter, on both twins, while the
    call without `-a` leaves the faulted total wrong."""
    rng = random.Random(f"f87-{seed}")
    n = rng.randint(250, 600)

    def insert() -> str:
        if rng.random() < 0.25:
            return rand_dna(rng, READ_LEN)
        return rand_dna(rng, rng.randint(25, 85))

    inserts = [insert() for _ in range(n)]
    noise = [rand_dna(rng, READ_LEN) for _ in range(n)]
    quals = [phred([rng.randint(36, 40) for _ in range(READ_LEN)]) for _ in range(n)]
    for _ in range(40):
        bad: set[int] = set()
        for twin in ("clean", "faulted"):
            text = read_text(inserts, noise, quals, twin)
            for length in PREFIXES:
                kept = fastp_reads(tools, text, "-a", CARRIED[:length])
                for k, ins in enumerate(inserts):
                    if kept.get(f"read{k + 1:04d}") != ins:
                        bad.add(k)
        if not bad:
            plain = fastp_reads(tools, read_text(inserts, noise, quals, "faulted"))
            if sum(map(len, plain.values())) == sum(map(len, inserts)):
                raise RuntimeError(f"the untrimmed total is right for seed {seed}")
            return tuple(inserts), tuple(noise), tuple(quals)
        for k in bad:
            inserts[k] = rand_dna(rng, rng.randint(25, 85))
    raise RuntimeError(f"no clean read set for seed {seed}")


def trimmed_bases(tools: ToolEnv, work: Path, *options: str) -> int:
    argv = ["fastp", "-i", "reads.fastq", "-o", "trimmed.fastq", "-j", "fastp.json"]
    run_ok(tools, [*argv, "-h", "fastp.html", "--thread", "1", *options], work)
    return sum(len(s) for s in (work / "trimmed.fastq").read_text().splitlines()[1::4])


@functools.lru_cache(maxsize=256)
def plain_bases(tools: ToolEnv, fastq: str) -> int:
    """What `fastp -i -o` with no adapter option returns for this file (the recorded naive)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "reads.fastq").write_text(fastq)
        return trimmed_bases(tools, work)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    inserts, noise, quals = draw_reads(tools, seed)
    text = read_text(list(inserts), list(noise), list(quals), twin)
    (work / "reads.fastq").write_text(text)
    lines = ["reads.fastq: single-end Illumina reads (FASTQ, Phred+33), TruSeq library."]
    if tier == "explicit":
        lines.append(
            f"TruSeq adapter sequence: {ADAPTER}. "
            + (
                "Reads from short fragments still run into the adapter."
                if twin == "faulted"
                else "The adapter was already removed by the vendor."
            )
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sum(len(s) for s in inserts)
    naive = plain_bases(tools, text)
    with_adapter = sum(len(s) < READ_LEN for s in inserts)
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{with_adapter} of {len(inserts)} reads end in the TruSeq adapter AGATCGGAAGAGC...",
            f"only {len(inserts)} reads: fastp detects adapters automatically from 10,000 reads",
        )
    prompt = common.prompt_for(
        route,
        "remove the 3' adapter from the reads in reads.fastq and report the total number of "
        "bases left across all reads",
        "use fastp to remove the 3' adapter from the reads in reads.fastq and report the total "
        "number of bases left across all reads",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F87", seed, twin, tier, route)
    return Task(task_id, "F87", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = trimmed_bases(tools, work, "-a", ADAPTER)
    return result(n, f"fastp -a {ADAPTER} (explicit adapter sequence)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(trimmed_bases(tools, work), "fastp with automatic adapter detection")


# Exposure: a fastp command that gives no adapter sequence (-a or --adapter_sequence with a
# sequence, --adapter_fasta). `-a auto` asks for the detection and is not a given adapter. Judged
# per command.
FASTP = re.compile(r"(?<![\w-])fastp\b")
GIVEN = re.compile(
    r"(?<![\w-])(?:-a|--adapter_sequence)(?:\s+|=)(?!['\"]?auto\b)\S|(?<![\w-])--adapter_fasta\b"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        FASTP.search(segment) and not GIVEN.search(segment)
        for command in commands
        for segment in segments(command)
    )


SPEC = FamilySpec(
    id="F87",
    slot="default_param",
    source="fastp README: SE adapter auto-detection (needs 10,000 reads); -a/--adapter_sequence",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
