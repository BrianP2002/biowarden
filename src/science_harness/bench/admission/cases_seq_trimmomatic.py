"""Admission cases for trimmomatic.minlen_order (family F85, implicit tier).

`trimmomatic SE -Xmx512m -threads 1 -phred33 reads.fastq out <steps>`; the answer is the number of
reads kept. A call is right when it keeps exactly the reads of 36 or more bases after trimming:
MINLEN last (any order of the other steps) is right on both twins; MINLEN before a shortening step
is right on the clean twin (reads delivered already trimmed) and wrong on the faulted one.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv

KEY = "minlen_stage"


def kept(name: str):
    def read(work: Path, proc) -> int:
        return len((work / name).read_text().splitlines()) // 4

    return read


def _trim(name: str, *steps: str, minlen_last: bool) -> Tpl:
    out = f"out_{name}.fastq"
    argv = ("SE", "-Xmx512m", "-threads", "1", "-phred33", "reads.fastq", out, *steps)
    right, wrong = (("after_trimming",), ("after_trimming",))
    if not minlen_last:
        right, wrong = (("before_trimming",), ("before_trimming", "after_trimming"))
    return Tpl(name, "trimmomatic", argv, kept(out), right, wrong)


CORRECT = ("LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36")
TEMPLATES: tuple[Tpl, ...] = (
    # --- MINLEN last
    _trim("standard", *CORRECT, minlen_last=True),
    _trim(
        "window_first",
        "SLIDINGWINDOW:4:15",
        "LEADING:3",
        "TRAILING:3",
        "MINLEN:36",
        minlen_last=True,
    ),
    _trim("window_only", "SLIDINGWINDOW:4:15", "MINLEN:36", minlen_last=True),
    _trim("trailing_window", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36", minlen_last=True),
    # --- MINLEN before a shortening step
    _trim(
        "alphabetical",
        "LEADING:3",
        "MINLEN:36",
        "SLIDINGWINDOW:4:15",
        "TRAILING:3",
        minlen_last=False,
    ),
    _trim(
        "minlen_first",
        "MINLEN:36",
        "LEADING:3",
        "TRAILING:3",
        "SLIDINGWINDOW:4:15",
        minlen_last=False,
    ),
    _trim("minlen_before_window", "MINLEN:36", "SLIDINGWINDOW:4:15", minlen_last=False),
    _trim(
        "minlen_middle",
        "LEADING:3",
        "TRAILING:3",
        "MINLEN:36",
        "SLIDINGWINDOW:4:15",
        minlen_last=False,
    ),
)


@case_source("trimmomatic.minlen_order")
def minlen_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F85", KEY, seed, scratch, tools, TEMPLATES)
