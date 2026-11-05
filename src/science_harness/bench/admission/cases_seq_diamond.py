"""Admission cases for the DIAMOND rules (families F79 and F80, implicit tier).

diamond.hit_cap (F79): `diamond blastp -q query.fasta -d db -e 1e-5` with many cap settings. The
answer is the family member with the lowest bit score among the hits; a call is right when it
returns the weakest member. The default cap (25), `-k 10`, `-k 30` and `--top` lose it on the
faulted twin (33-45 members); `-k 0`, `--max-target-seqs 0`, `-k 100` and `-k 500` keep it.

diamond.sensitivity_mode (F80): `diamond blastp -q queries.fasta -d db` with every sensitivity
flag. The answer is the list of queries with a hit (E < 0.001, the default); the default and
faster modes miss the 35-41% identical partners of the faulted twin, the sensitive modes are
labelled by running them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task

QUIET = ("--threads", "1", "--quiet")

# ---- diamond.hit_cap (F79) --------------------------------------------------------------------

KEY_CAP = "hit_limit"


def weakest(work: Path, name: str) -> list[str]:
    best: dict[str, float] = {}
    for line in (work / name).read_text().splitlines():
        subject, score = line.split("\t")
        best[subject] = max(best.get(subject, 0.0), float(score))
    return [min(best, key=lambda s: (best[s], s))]


def _cap(name: str, *options: str, intent: str = "all_hits") -> Tpl:
    out = f"out_{name}.tsv"
    argv = ("blastp", "-q", "query.fasta", "-d", "db", "-o", out, "-e", "1e-5")
    argv += ("-f", "6", "sseqid", "bitscore", *QUIET, *options)
    reader: Callable[[Path, Any], list[str]] = lambda work, proc, n=out: weakest(work, n)  # noqa: E731
    other = "top_hits" if intent == "all_hits" else "all_hits"
    return Tpl(name, "diamond", argv, reader, (intent,), (intent, other))


CAP_TEMPLATES: tuple[Tpl, ...] = (
    # --- the cap in force: 25 by default
    _cap("default"),
    _cap("k10", "-k", "10", intent="top_hits"),
    _cap("k25", "--max-target-seqs", "25"),
    _cap("k30", "-k", "30"),
    _cap("k40", "-k", "40"),
    _cap("top5", "--top", "5", intent="top_hits"),
    _cap("sensitive", "--sensitive"),
    _cap("block", "-b", "0.5"),
    # --- every hit
    _cap("k0", "-k", "0"),
    _cap("max0", "--max-target-seqs", "0"),
    _cap("k100", "-k", "100"),
    _cap("k500", "-k", "500"),
    _cap("sensitive_k0", "--sensitive", "-k", "0"),
    _cap("k0_attached", "-k0"),
)


@case_source("diamond.hit_cap")
def hit_cap_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def prepare(work: Path, task: Task) -> None:
        argv = ["diamond", "makedb", "--in", "db.fasta", "-d", "db", *QUIET]
        assert tools.run(argv, work).returncode == 0

    return cases_for("F79", KEY_CAP, seed, scratch, tools, CAP_TEMPLATES, prepare)


# ---- diamond.sensitivity_mode (F80) -----------------------------------------------------------

KEY_MODE = "homology_range"


def queries_with_hits(work: Path, name: str) -> list[str]:
    rows = (work / name).read_text().splitlines()
    return sorted({row.split("\t")[0] for row in rows if row.strip()})


def _mode(name: str, *options: str, sensitive: bool) -> Tpl:
    out = f"out_{name}.tsv"
    argv = ("blastp", "-q", "queries.fasta", "-d", "db", "-o", out, "-f", "6", "qseqid", "sseqid")
    reader = lambda work, proc, n=out: queries_with_hits(work, n)  # noqa: E731
    right, wrong = (
        (("remote",), ("remote", "close")) if sensitive else (("close",), ("close", "remote"))
    )
    return Tpl(name, "diamond", (*argv, *QUIET, *options), reader, right, wrong)


MODE_TEMPLATES: tuple[Tpl, ...] = (
    _mode("default", sensitive=False),
    _mode("fast", "--fast", sensitive=False),
    _mode("mid", "--mid-sensitive", sensitive=False),
    _mode("evalue", "-e", "0.001", sensitive=False),
    _mode("k0", "-k", "0", sensitive=False),
    _mode("block", "-b", "0.5", sensitive=False),
    _mode("sensitive", "--sensitive", sensitive=True),
    _mode("more", "--more-sensitive", sensitive=True),
    _mode("very", "--very-sensitive", sensitive=True),
    _mode("ultra", "--ultra-sensitive", sensitive=True),
    _mode("ultra_k0", "--ultra-sensitive", "-k", "0", sensitive=True),
)


@case_source("diamond.sensitivity_mode")
def sensitivity_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def prepare(work: Path, task: Task) -> None:
        argv = ["diamond", "makedb", "--in", "db.fasta", "-d", "db", *QUIET]
        assert tools.run(argv, work).returncode == 0

    return cases_for("F80", KEY_MODE, seed, scratch, tools, MODE_TEMPLATES, prepare)
