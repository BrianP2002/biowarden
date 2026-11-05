"""Admission cases for cutadapt.adapters_per_read (family F84, implicit tier).

`cutadapt -g PRIMER -a ADAPTER <options> -o out reads.fastq`; the answer is the total number of
bases left (the inserts). Calls that keep the default `-n 1` are right on the clean twin (the reads
carry the primer only) and wrong on the faulted one (every read carries both); `-n 2`, `--times 2`,
`-n 3` and linked adapters are right on both. Options such as `-m`, `-e`, `-O`, `-q` do not change
the verdict.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.extra.f84_cutadapt_adapters import ADAPTER, PRIMER

KEY = "adapters_per_read"


def total_bases(name: str):
    def read(work: Path, proc) -> int:
        lines = (work / name).read_text().splitlines()
        return sum(len(line) for line in lines[1::4])

    return read


def _cut(name: str, adapters: tuple[str, ...], *options: str, removes_all: bool) -> Tpl:
    out = f"out_{name}.fastq"
    argv = (*adapters, *options, "-o", out, "reads.fastq")
    right, wrong = (("all",), ("all", "one")) if removes_all else (("one", "all"), ("one", "all"))
    return Tpl(name, "cutadapt", argv, total_bases(out), right, wrong)


BOTH = ("-g", PRIMER, "-a", ADAPTER)
LINKED = ("-g", f"{PRIMER}...{ADAPTER}")

TEMPLATES: tuple[Tpl, ...] = (
    # --- one adapter removed per read
    _cut("plain", BOTH, removes_all=False),
    _cut("times_one", BOTH, "-n", "1", removes_all=False),
    _cut("minlen", BOTH, "-m", "10", removes_all=False),
    _cut("swapped", ("-a", ADAPTER, "-g", PRIMER), removes_all=False),
    _cut("error_rate", BOTH, "-e", "0.15", removes_all=False),
    _cut("overlap", BOTH, "-O", "5", removes_all=False),
    _cut("quality", BOTH, "-q", "20", removes_all=False),
    _cut("long_options", ("--front", PRIMER, "--adapter", ADAPTER), removes_all=False),
    _cut("cores", BOTH, "-j", "1", removes_all=False),
    # --- every listed adapter removed
    _cut("times_two", BOTH, "-n", "2", removes_all=True),
    _cut("times_long", BOTH, "--times", "2", removes_all=True),
    _cut("times_three", BOTH, "-n", "3", removes_all=True),
    _cut("times_minlen", BOTH, "-n", "2", "-m", "10", removes_all=True),
    _cut("linked_front", LINKED, removes_all=True),
    _cut("linked_adapter", ("-a", f"{PRIMER}...{ADAPTER}"), removes_all=True),
    _cut("linked_minlen", LINKED, "-m", "10", removes_all=True),
)


@case_source("cutadapt.adapters_per_read")
def adapters_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F84", KEY, seed, scratch, tools, TEMPLATES)
