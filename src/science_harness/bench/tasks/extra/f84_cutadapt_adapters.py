"""F84 cutadapt_adapters: cutadapt removes ONE adapter per read unless told otherwise (`-n`).

`cutadapt -g PRIMER -a ADAPTER` looks like "remove the 5' primer and the 3' adapter", but
`-n/--times` defaults to 1: each read is trimmed once, with the best-scoring of ALL listed
adapters, so a read that carries both keeps one of them. The report only says "Reads with adapters:
N (100%)". `-n 2` (or a linked adapter `-g PRIMER...ADAPTER`) removes both.

Verified here (cutadapt 5.2): 30 amplicon reads (19-base primer, 20-35 base insert, 13-base
adapter, random tail; `-g GTGCCAGCAGCCGCGGTAA -a AGATCGGAAGAGC`): 0 of 30 reads trimmed to the
insert; with `-n 2` 30 of 30; with the linked adapter `-g PRIMER...ADAPTER` 30 of 30; with `-a
ADAPTER` or `-g PRIMER` alone 0 of 30. Exit 0 and an empty stderr in every case (the report goes to
stdout). On the generated sets of this family (seeds 0-99) the reference equals the sum of the
insert lengths and the plain call leaves one adapter on every read that has two.

Twins: the same inserts. Clean: the reads are primer + insert only (the vendor already removed
the 3' adapter). Faulted: the reads hold primer + insert + 3' adapter + sequencing through, so every
read has both. The answer is the total number of insert bases.

  explicit  the README says the reads carry the primer at the 5' end and the adapter at the 3' end
  implicit  the README only gives the two sequences
  forced    the task sentence dictates cutadapt
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
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

PRIMER = "GTGCCAGCAGCCGCGGTAA"
ADAPTER = "AGATCGGAAGAGCACACGTC"
READ_LEN = 100
TWIN_FILES = frozenset({"reads.fastq", "README.md"})


def read_text(inserts: list[str], filler: list[str], twin: str) -> str:
    reads = []
    for i, (insert, tail) in enumerate(zip(inserts, filler, strict=True)):
        sequence = PRIMER + insert
        if twin == "faulted":
            sequence = (sequence + ADAPTER + tail)[:READ_LEN]
        reads.append((f"read{i + 1:03d}", sequence, phred([36] * len(sequence))))
    return fastq_text(reads)


@functools.lru_cache(maxsize=256)
def draw_inserts(tools: ToolEnv, seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Inserts and sequencing noise for which `cutadapt -n 2` returns exactly the inserts.

    A partial primer or adapter (with an indel) can occur by chance at the ends of an insert;
    reads that cutadapt would trim too far are redrawn, on both twins' layouts.
    """
    rng = random.Random(f"f84-{seed}")
    n = rng.randint(30, 50)
    inserts = [rand_dna(rng, rng.randint(20, 40)) for _ in range(n)]
    filler = [rand_dna(rng, READ_LEN) for _ in range(n)]
    for _ in range(40):
        bad: set[int] = set()
        for twin in ("clean", "faulted"):
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                (work / "reads.fastq").write_text(read_text(inserts, filler, twin))
                argv = ["cutadapt", "-g", PRIMER, "-a", ADAPTER, "-n", "2", "-o", "o.fastq"]
                run_ok(tools, [*argv, "reads.fastq"], work)
                out = (work / "o.fastq").read_text().splitlines()[1::4]
            bad |= {i for i, (o, ins) in enumerate(zip(out, inserts, strict=True)) if o != ins}
            bad |= {i for i in range(n) if inserts[i][-1] == ADAPTER[0]}
        if not bad:
            return tuple(inserts), tuple(filler)
        for i in bad:
            inserts[i] = rand_dna(rng, rng.randint(20, 40))
    raise RuntimeError(f"no clean insert set for seed {seed}")


def make_plan(tools: ToolEnv, seed: int, twin: str) -> list[tuple[str, str, int]]:
    """(name, sequence, insert length) per read."""
    inserts, filler = draw_inserts(tools, seed)
    text = read_text(list(inserts), list(filler), twin).splitlines()[1::4]
    return [
        (f"read{i + 1:03d}", s, len(ins))
        for i, (s, ins) in enumerate(zip(text, inserts, strict=True))
    ]


def trimmed_bases(tools: ToolEnv, work: Path, *options: str) -> int:
    argv = ["cutadapt", "-g", PRIMER, "-a", ADAPTER, *options, "-o", "trimmed.fastq", "reads.fastq"]
    run_ok(tools, argv, work)
    seqs = (work / "trimmed.fastq").read_text().splitlines()[1::4]
    return sum(len(s) for s in seqs)


@functools.lru_cache(maxsize=256)
def plain_bases(tools: ToolEnv, fastq: str) -> int:
    """What `cutadapt -g PRIMER -a ADAPTER` returns for this file (the recorded naive answer)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "reads.fastq").write_text(fastq)
        return trimmed_bases(tools, work)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    reads = make_plan(tools, seed, twin)
    text = fastq_text([(n, s, phred([36] * len(s))) for n, s, _ in reads])
    (work / "reads.fastq").write_text(text)
    lines = [
        "reads.fastq: amplicon reads (FASTQ, Phred+33).",
        f"5' primer: {PRIMER}. 3' adapter: {ADAPTER}.",
    ]
    if tier == "explicit":
        lines.append(
            "A read starts with the primer; when the amplicon is short the read runs on into "
            "the 3' adapter and then into sequencing noise."
            if twin == "faulted"
            else "A read starts with the primer and ends with the amplicon; the adapter was "
            "already removed by the vendor."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sum(n for *_, n in reads)
    naive = plain_bases(tools, text)
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"every one of the {len(reads)} reads holds both the 5' primer and the 3' adapter",
            "cutadapt removes only one adapter per read by default (-n 1)",
        )
    prompt = common.prompt_for(
        route,
        "remove the 5' primer and the 3' adapter from every read in reads.fastq and report the "
        "total number of amplicon (insert) bases left across all reads",
        "use cutadapt to remove the 5' primer and the 3' adapter from every read in reads.fastq "
        "and report the total number of amplicon (insert) bases left across all reads",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F84", seed, twin, tier, route)
    return Task(task_id, "F84", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = trimmed_bases(tools, work, "-n", "2")
    return result(n, "cutadapt -g PRIMER -a ADAPTER -n 2 (both removed from each read)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(trimmed_bases(tools, work), "cutadapt -g PRIMER -a ADAPTER (-n 1)")


# Exposure: a cutadapt command that lists two adapters for the same read (-g/-a/-b together, or
# two of the same) and neither raises -n/--times to 2 or more nor uses a linked adapter
# (`FRONT...BACK`). Judged per command.
CUTADAPT = re.compile(r"(?<![\w-])cutadapt\b")
ADAPTER_OPTION = re.compile(
    r"(?<![\w-])(?:-[agb]|--adapter|--front|--anywhere)"
    r"(?:[ =]+|(?=[ACGTUNacgtun^]))['\"]?([^\s'\"]+)"
)
TIMES = re.compile(r"(?<![\w-])(?:-n|--times)(?:[ =]+|(?=\d))['\"]?(\d+)")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if not CUTADAPT.search(segment):
                continue
            adapters = [a for a in ADAPTER_OPTION.findall(segment) if "..." not in a]
            times = TIMES.findall(segment)
            if len(adapters) >= 2 and (not times or int(times[-1]) < 2):
                return True
    return False


SPEC = FamilySpec(
    id="F84",
    slot="default_param",
    source="cutadapt -n/--times (default 1): remove up to COUNT adapters from each read",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
