"""Shared machinery of the `seq` admission case sources (no rule is registered here).

A case source builds, for one seed, the benchmark workspaces of a family (clean and faulted twin,
implicit tier), runs a list of realistic calls with the REAL tools in each workspace, turns every
output into the task's answer and labels the call right or wrong against `task.truth`
(bench.score.answer_matches). The label never comes from a rule. Each call is then emitted without
a declaration and with the declarations its author could give:

  right call   declared with the convention its own options implement (and nothing else: a right
               call that states the opposite convention would be a contradiction, not a right call)
  wrong call   declared with the convention its own options implement (the call is botched, the
               rule cannot see it) and with the opposite convention (contradicts the call or the
               data)

Calls of one workspace run concurrently (the tools are single-threaded; outputs go to files named
after the call), so a seed costs about as much as its slowest tool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, Label
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task
from science_harness.engine import Call

WORKERS = 4


@dataclass(frozen=True)
class Tpl:
    """One realistic call and how to read its answer.

    right_decl / wrong_decl: declaration values emitted when the call is labelled right / wrong
    (empty tuple: only the undeclared call is emitted). `answer(work, proc)` returns the task answer
    from the finished run; None marks an inspection call (labelled right, it cannot be wrong).
    """

    name: str
    tool: str
    argv: tuple[str, ...]
    answer: Callable[[Path, Any], Any] | None
    right_decl: tuple[str, ...] = ()
    wrong_decl: tuple[str, ...] = ()
    subdir: str = ""  # run (and emit the call) in this subdirectory of the workspace


def run_all(tools: ToolEnv, work: Path, templates: Sequence[Tpl]) -> list[Any]:
    def one(tpl: Tpl) -> Any:
        return tools.run([tpl.tool, *tpl.argv], work / tpl.subdir)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, templates))


def label_of(task: Task, work: Path, tpl: Tpl, proc: Any) -> Label:
    work = work / tpl.subdir
    if tpl.answer is None:
        return "right"
    if proc.returncode != 0:
        return "wrong"
    try:
        answer = tpl.answer(work, proc)
    except Exception:  # an output the task cannot use is a wrong call
        return "wrong"
    return "right" if answer_matches(task.truth, answer) else "wrong"


def emit(
    out: list[Case],
    tpl: Tpl,
    label: Label,
    key: str,
    work: Path,
    family: str,
    seed: int,
    twin: str,
) -> None:
    decls = tpl.right_decl if label == "right" else tpl.wrong_decl
    cwd = work / tpl.subdir
    out.append(Case(Call(tpl.tool, tpl.argv, cwd), label, False, family, seed, twin, tpl.name))
    for value in decls:
        call = Call(tpl.tool, tpl.argv, cwd, {key: value})
        out.append(Case(call, label, True, family, seed, twin, f"{tpl.name}:{value}"))


def cases_for(
    family: str,
    key: str,
    seed: int,
    scratch: Path,
    tools: ToolEnv,
    templates: Sequence[Tpl],
    prepare: Callable[[Path, Task], None] | None = None,
    task_tier: str = "implicit",
) -> list[Case]:
    """Both twins of `family` at one seed: run every template, label it, emit the cases."""
    from science_harness.bench.tasks import build_task

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task(family, seed, twin, work, tools, task_tier, "free")
        if prepare is not None:
            prepare(work, task)
        procs = run_all(tools, work, templates)
        for tpl, proc in zip(templates, procs, strict=True):
            emit(out, tpl, label_of(task, work, tpl, proc), key, work, family, seed, twin)
    return out
