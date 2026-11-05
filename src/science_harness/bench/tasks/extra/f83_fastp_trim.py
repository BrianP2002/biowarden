"""F83 fastp_trim: fastp's `-q` is a read filter, not a quality trimmer.

`fastp -q 20` sets the "qualified base" cut-off (`--qualified_quality_phred`); together with
`-u` (`--unqualified_percent_limit`, default 40) it DROPS whole reads that have more than 40% of
their bases below Q20. It trims nothing: the 3' tails of the reads that survive keep their
low-quality bases. Tail trimming is `--cut_right` / `--cut_tail` / `--cut_front` (window size `-W`,
mean quality `-M`). Exit 0, a "Filtering result" summary on stderr, no warning that nothing was
trimmed.

Verified here (fastp 1.4.0; 50 reads of 80-100 bases, half with a Q12-18 tail after a Q30-40 head,
20 seeds): `--cut_right -W 4 -M 20` returns exactly the head of every read (all 50 reads kept);
`-q 20` returned 35-40 of the 50 reads with the tails of the rest still attached (seed 0: 3210
bases and 35 reads against 3581 bases and 50 reads), and `-q 20` alone is the same as the default
run when no read has more than 40% of its bases below Q15. `--cut_tail -W 4 -M 20` (the 3' end
scan) cuts one base further on some reads and is not the 5' window scan the task describes.

Twins: the same reads. Clean: the reads are given as the good heads only (already trimmed).
Faulted: each read of a random half also carries a 6-40 base low-quality tail (Q14-18) after a
head of Q38-40. The answer is the total number of bases in the heads.

  explicit  the README says some reads carry a low-quality 3' tail (faulted) or none (clean)
  implicit  the README only says "Illumina reads"
  forced    the task sentence dictates fastp
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

TWIN_FILES = frozenset({"reads.fastq", "README.md"})


MIN_TRAP = 60  # bases between the `-q 20` answer and the truth on the faulted twin


def make_plan(seed: int, twin: str, salt: int = 0) -> list[tuple[str, str, str, int]]:
    """(name, sequence, quality string, good bases) per read."""
    rng = random.Random(f"f83-{seed}-{salt}")
    reads = []
    for i in range(rng.randint(40, 60)):
        length = rng.choice([80, 90, 100])
        head = length
        if rng.random() < 0.55:
            head = rng.randint(22, length - 12)
        tail = rng.randint(6, length - head) if head < length else 0
        seq = rand_dna(rng, head + tail)
        quals = [rng.randint(38, 40) for _ in range(head)]
        quals += [rng.randint(14, 18) for _ in range(tail)]
        plain = [rng.randint(30, 40) for _ in range(head)]  # untailed reads: Q30-40 (same draws)
        if twin == "clean":
            seq, quals = seq[:head], quals[:head]
        elif tail == 0:
            quals = plain
        reads.append((f"read{i + 1:03d}", seq, phred(quals), head))
    return reads


def fastp_bases(tools: ToolEnv, work: Path, *options: str) -> tuple[int, int]:
    """(bases, reads) in the output of `fastp -i reads.fastq <options>`."""
    argv = ["fastp", "-i", "reads.fastq", "-o", "trimmed.fastq", "-j", "fastp.json"]
    argv += ["-h", "fastp.html", "--thread", "1", *options]
    run_ok(tools, argv, work)
    seqs = (work / "trimmed.fastq").read_text().splitlines()[1::4]
    return sum(len(s) for s in seqs), len(seqs)


@functools.lru_cache(maxsize=256)
def q20_bases(tools: ToolEnv, fastq: str) -> int:
    """What `fastp -q 20` returns for this file (the recorded naive answer)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "reads.fastq").write_text(fastq)
        return fastp_bases(tools, work, "-q", "20")[0]


@functools.lru_cache(maxsize=256)
def resolve_salt(tools: ToolEnv, seed: int) -> int:
    """The first salt whose faulted twin makes `-q 20` miss the truth by at least MIN_TRAP bases."""
    for salt in range(50):
        reads = make_plan(seed, "faulted", salt)
        text = fastq_text([(n, s, q) for n, s, q, _ in reads])
        if abs(q20_bases(tools, text) - sum(head for *_, head in reads)) >= MIN_TRAP:
            return salt
    raise RuntimeError(f"no salt found for seed {seed}")


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    reads = make_plan(seed, twin, resolve_salt(tools, seed))
    text = fastq_text([(n, s, q) for n, s, q, _ in reads])
    (work / "reads.fastq").write_text(text)
    lines = ["reads.fastq: single-end Illumina reads (FASTQ), Phred+33 qualities."]
    if tier == "explicit":
        lines.append(
            "Some reads carry a low-quality tail at the 3' end."
            if twin == "faulted"
            else "The reads were trimmed before delivery; none has a low-quality tail."
        )
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    expected = sum(head for *_, head in reads)
    tailed = [1 for _, s, _, head in reads if len(s) > head]
    if twin == "clean":
        evidence = ()
    else:
        evidence = (
            f"{len(tailed)} of {len(reads)} reads end in bases of quality 14-18",
            "fastp -q is the qualified-base cut-off of the read filter, not a trimming option",
        )
    prompt = common.prompt_for(
        route,
        "remove the low-quality 3' tail of every read in reads.fastq (scan from the 5' end and "
        "cut at the first 4-base window whose mean Phred quality is below 20), keep every read, "
        "and report the total number of bases left across all reads",
        "use fastp to remove the low-quality 3' tail of every read in reads.fastq (scan from the "
        "5' end and cut at the first 4-base window whose mean Phred quality is below 20), keep "
        "every read, and report the total number of bases left across all reads",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": q20_bases(tools, text)}
    task_id = common.task_id("F83", seed, twin, tier, route)
    return Task(task_id, "F83", "flag_semantics", twin, seed, prompt, truth, evidence, tier, route)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n, _ = fastp_bases(tools, work, "--cut_right", "-W", "4", "-M", "20")
    return result(n, "fastp --cut_right -W 4 -M 20 (sliding-window tail trimming)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n, _ = fastp_bases(tools, work, "-q", "20")
    return result(n, "fastp -q 20")


# Exposure: a fastp command with no trimming option. Tail trimming is --cut_right/--cut_tail/
# --cut_front (-r/-3/-5 and the --cut_*_window_size/--cut_*_mean_quality family), or a fixed
# trim with -f/-t/--trim_front1/--trim_tail1/-b. Judged per command.
FASTP = re.compile(r"(?<![\w-])fastp\b")
TRIMMING = re.compile(
    r"(?<![\w-])(?:--cut_(?:right|tail|front|window_size|mean_quality)|-[r35]|--trim_tail1|-t|-b)"
    r"(?![\w-])|--cut_(?:right|tail|front)_"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        FASTP.search(segment) and not TRIMMING.search(segment)
        for command in commands
        for segment in segments(command)
    )


SPEC = FamilySpec(
    id="F83",
    slot="flag_semantics",
    source="fastp usage: -q qualified_quality_phred / -u unqualified_percent_limit versus --cut_*",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
