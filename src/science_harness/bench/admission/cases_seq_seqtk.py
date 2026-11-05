"""Admission cases for seqtk.trimfq_floor (family F81, implicit tier).

`seqtk trimfq <options> reads.fastq`; the answer is the total number of bases of the trimmed reads.
A call is right when it equals the total of the good heads: `-l 1` (or 0) with any error-rate
threshold between 0.001 and 0.05, and any floor that never binds. The default floor (30), `-l 30`
and `-l 20` are right on the clean twin (reads delivered already trimmed) and wrong on the faulted
twin. `-q 20` (an error rate of 20: nothing is trimmed) and `-b 5` (a fixed trim) are wrong on the
faulted twin for another reason: the rule does not see them (recall counts them).
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv

KEY = "trim_floor"


def total_bases(work: Path, proc) -> int:
    return sum(len(line) for line in proc.stdout.splitlines()[1::4])


def _trim(name: str, *options: str, floor: bool) -> Tpl:
    right = ("no_minimum",)
    wrong = ("keep_minimum", "no_minimum") if floor else ("no_minimum",)
    return Tpl(name, "seqtk", ("trimfq", *options, "reads.fastq"), total_bases, right, wrong)


TEMPLATES: tuple[Tpl, ...] = (
    # --- a floor of 2 or more: reads are not cut below it
    _trim("default_q001", "-q", "0.01", floor=True),
    _trim("default_q005", "-q", "0.05", floor=True),
    _trim("default_q0001", "-q", "0.001", floor=True),
    _trim("floor30", "-q", "0.01", "-l", "30", floor=True),
    _trim("floor20", "-q", "0.01", "-l", "20", floor=True),
    _trim("floor10", "-q", "0.01", "-l", "10", floor=True),
    _trim("floor5", "-q", "0.01", "-l", "5", floor=True),
    _trim("force_fastq", "-Q", "-q", "0.01", floor=True),
    _trim("defaults_only", floor=True),
    # --- no floor
    _trim("floor1", "-q", "0.01", "-l", "1", floor=False),
    _trim("floor0", "-q", "0.01", "-l", "0", floor=False),
    _trim("floor1_attached", "-q", "0.01", "-l1", floor=False),
    _trim("floor1_q005", "-q", "0.05", "-l", "1", floor=False),
    _trim("floor1_q0001", "-q", "0.001", "-l", "1", floor=False),
    # --- calls that are wrong for another reason
    _trim("phred_style_q", "-q", "20", floor=True),
    _trim("fixed_left", "-b", "5", floor=False),
    _trim("fixed_right", "-e", "5", floor=False),
)


@case_source("seqtk.trimfq_floor")
def trimfq_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F81", KEY, seed, scratch, tools, TEMPLATES)
