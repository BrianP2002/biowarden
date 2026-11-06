"""F85 trim_minlen: Trimmomatic runs its steps in the order given, so a MINLEN written before the
trimming steps filters the untrimmed reads.

Trimmomatic's manual: "the trimming steps are performed in the order in which they are specified
on the command line". `MINLEN:36` is a filter on the length of the read AT THAT POINT. Listed
before `SLIDINGWINDOW` and `TRAILING` (for example because a protocol sheet lists the parameters
alphabetically: LEADING, MINLEN, SLIDINGWINDOW, TRAILING) it checks the untrimmed length, and the
reads that the later steps shorten below 36 are kept. Exit 0; the summary line on stderr reads
"Surviving: 40 (100.00%)".

Verified here (Trimmomatic 0.41, `SE -phred33`; 40 reads of 60 bases with a Q32-40 head of 10-60
bases and a Q2-8 tail): `LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36` kept 19 reads (the
shortest 40 bases); `MINLEN:36 SLIDINGWINDOW:4:15` kept all 40 (the shortest 10 bases);
`SLIDINGWINDOW:4:15 MINLEN:36` kept 19; `MINLEN:36 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15` kept 40.
On the generated sets of this family (seeds 0-39) the alphabetical order keeps every read with a
tail that the trimming cuts under 36 bases.

Twins: the same good heads. Clean: the reads are the heads only (delivered already trimmed), so
the order makes no difference. Faulted: 60% of the reads also carry a Q2-8 tail, and the heads of
40% of those are 10-28 bases long (under 36 after trimming); the others are 44 bases or longer.
The answer is the number of reads of 36 or more bases after trimming.

  explicit  the README says that the length filter is meant for the trimmed reads
  implicit  the README only lists the parameters
  forced    the task sentence dictates Trimmomatic SE
"""

from __future__ import annotations

import random
import re
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

MINLEN = 36
CORRECT_ORDER = ("LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36")
LISTED_ORDER = ("LEADING:3", "MINLEN:36", "SLIDINGWINDOW:4:15", "TRAILING:3")
TWIN_FILES = frozenset({"reads.fastq", "README.md"})


def make_plan(seed: int, twin: str) -> list[tuple[str, str, str, int]]:
    """(name, sequence, quality string, good bases) per read."""
    rng = random.Random(f"f85-{seed}")
    reads = []
    for i in range(rng.randint(45, 60)):
        length = rng.randint(60, 80)
        tailed = rng.random() < 0.6
        short = rng.random() < 0.4
        head = length
        if tailed:
            head = rng.randint(10, 28) if short else rng.randint(44, length - 8)
        seq = rand_dna(rng, length)
        quals = [rng.randint(32, 40) for _ in range(head)]
        quals += [rng.randint(2, 8) for _ in range(length - head)]
        if twin == "clean":
            seq, quals = seq[:head], quals[:head]
        reads.append((f"read{i + 1:03d}", seq, phred(quals), head))
    return reads


def survivors(tools: ToolEnv, work: Path, steps: tuple[str, ...]) -> int:
    argv = ["trimmomatic", "SE", "-Xmx512m", "-threads", "1", "-phred33"]
    run_ok(tools, [*argv, "reads.fastq", "trimmed.fastq", *steps], work)
    return len((work / "trimmed.fastq").read_text().splitlines()) // 4


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    reads = make_plan(seed, twin)
    (work / "reads.fastq").write_text(fastq_text([(n, s, q) for n, s, q, _ in reads]))
    lines = [
        "reads.fastq: single-end Illumina reads, 60-80 bases, Phred+33 qualities.",
        "Lab trimming protocol (Trimmomatic step names): LEADING:3, MINLEN:36, "
        "SLIDINGWINDOW:4:15, TRAILING:3.",
    ]
    if tier == "explicit":
        lines.append(
            "The protocol lists its parameters alphabetically, not in the order they run; the "
            "minimum length applies to the trimmed reads."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sum(head >= MINLEN for *_, head in reads)
    naive = sum(len(seq) >= MINLEN for _, seq, _, _ in reads)
    shortened = [1 for _, seq, _, head in reads if head < len(seq) and head < MINLEN]
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{len(shortened)} reads fall under {MINLEN} bases once their low-quality tail is cut",
            "Trimmomatic runs its steps in the order given: a MINLEN before SLIDINGWINDOW "
            "filters the untrimmed reads",
        )
    prompt = common.prompt_for(
        route,
        "apply the lab's trimming steps LEADING:3, MINLEN:36, SLIDINGWINDOW:4:15 and TRAILING:3 "
        "(Trimmomatic names) to reads.fastq and report how many reads remain; a read shorter "
        "than 36 bases after trimming is discarded",
        "use Trimmomatic SE with the steps LEADING:3, MINLEN:36, SLIDINGWINDOW:4:15 and "
        "TRAILING:3 on reads.fastq and report how many reads remain; a read shorter than 36 "
        "bases after trimming is discarded",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F85", seed, twin, tier, route)
    return Task(task_id, "F85", "step_order", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = survivors(tools, work, CORRECT_ORDER)
    return result(n, "Trimmomatic SE LEADING TRAILING SLIDINGWINDOW MINLEN (trimming first)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(survivors(tools, work, LISTED_ORDER), "Trimmomatic SE, steps in listed order")


# Exposure: a trimmomatic command whose MINLEN comes before a step that shortens reads.
# Judged per command; `java -jar trimmomatic.jar` counts as trimmomatic.
TRIMMOMATIC = re.compile(r"(?<![\w-])(?:trimmomatic|Trimmomatic[\w.-]*\.jar)\b")
STEP = re.compile(
    r"(?<![\w:.-])(ILLUMINACLIP|SLIDINGWINDOW|LEADING|TRAILING|CROP|HEADCROP|MAXINFO|MINLEN)(?=[:\s'\"]|$)"
)
SHORTENING = {"ILLUMINACLIP", "SLIDINGWINDOW", "LEADING", "TRAILING", "CROP", "HEADCROP", "MAXINFO"}


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if not TRIMMOMATIC.search(segment):
                continue
            steps = STEP.findall(segment)
            if "MINLEN" in steps and SHORTENING & set(steps[steps.index("MINLEN") + 1 :]):
                return True
    return False


SPEC = FamilySpec(
    id="F85",
    slot="step_order",
    source="Trimmomatic manual: steps are performed in the order specified; MINLEN",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
