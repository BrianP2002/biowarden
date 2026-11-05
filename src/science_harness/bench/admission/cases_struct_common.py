"""Shared machinery of the `struct` admission case sources (no rule is registered here).

A case source builds, for one seed, the benchmark workspaces of a family (clean and faulted twin),
runs a list of realistic calls with the REAL tools in each workspace, turns every output into the
task's answer and labels the call right or wrong against `task.truth` (bench.score.answer_matches).
The label never comes from a rule. Each call is emitted without a declaration and with the
declarations a careful or a careless caller would write:

  right call   declared with the convention the task needs (and nothing else: a right call that
               states the wrong convention would be a contradiction, not a right call)
  wrong call   declared with the needed convention (the code is botched, the rule cannot see it)
               and declared with the convention its own code uses (contradicts the data)

Library calls (python-struct snippets) run together in ONE interpreter per workspace, each snippet
in its own namespace with its own stdout, so that a seed costs one library import instead of one per
call; a snippet is exactly the code of a `python-struct -c` call. Calls of command-line tools run as
processes. An inspection call (head, a printed summary) produces no answer: it is labelled right
(it cannot be wrong) and is the zero-friction probe.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, Label
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.bench.tasks.base import Task
from science_harness.engine import Call

DRIVER = r"""
import contextlib, io, json, sys
snippets = json.loads(sys.stdin.read())
out = {}
for name, code in snippets.items():
    buf, err, rc = io.StringIO(), io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            exec(compile(code, name, "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        rc = 1 if exc.code else 0
    except BaseException as exc:
        rc, _ = 1, err.write(repr(exc))
    out[name] = [rc, buf.getvalue(), err.getvalue()[-300:]]
print(json.dumps(out))
"""


@dataclass(frozen=True)
class Tpl:
    """One realistic call. `uses` is the convention its code applies (None: not applicable)."""

    name: str
    tool: str
    argv: tuple[str, ...]
    uses: str | None = None
    parse: Callable[[str], Any] | None = None  # stdout -> answer; None for an inspection call
    note: str = ""


def py(name: str, code: str, uses: str | None, parse: Callable[[str], Any] | None) -> Tpl:
    """A python-struct snippet call (`python-struct -c code`)."""
    return Tpl(name, "python-struct", ("-c", code), uses, parse)


def last_json(out: str) -> Any:
    return json.loads(out.strip().splitlines()[-1])


def last_float(out: str) -> float:
    return float(out.strip().splitlines()[-1])


def last_int(out: str) -> int:
    return int(float(out.strip().splitlines()[-1]))


def run_all(tools: ToolEnv, work: Path, templates: list[Tpl]) -> dict[str, tuple[int, str, str]]:
    """name -> (returncode, stdout, stderr) for every template."""
    results: dict[str, tuple[int, str, str]] = {}
    snippets = {t.name: t.argv[1] for t in templates if t.tool == "python-struct"}
    if snippets:
        proc = tools.run(["python-struct", "-s", "-c", DRIVER], work, stdin=json.dumps(snippets))
        if proc.returncode != 0:
            raise RuntimeError(f"python-struct driver failed: {proc.stderr[-500:]}")
        for name, (rc, out, err) in json.loads(proc.stdout.strip().splitlines()[-1]).items():
            results[name] = (rc, out, err)
    for t in templates:
        if t.tool != "python-struct":
            proc = tools.run([t.tool, *t.argv], work)
            results[t.name] = (proc.returncode, proc.stdout, proc.stderr)
    return results


def label_of(t: Tpl, task: Task, result: tuple[int, str, str]) -> Label:
    rc, out, _ = result
    if t.parse is None:
        return "right"
    if rc != 0:
        return "wrong"
    try:
        answer = t.parse(out)
    except (ValueError, IndexError, KeyError, json.JSONDecodeError):
        return "wrong"
    return "right" if answer_matches(task.truth, answer) else "wrong"


def declarations(
    t: Tpl, label: Label, key: str, needed: str, values: tuple[str, ...]
) -> list[dict[str, str]]:
    """The declarations to emit for a call: the needed convention, plus its own when it differs."""
    if label == "right" or t.uses is None:
        return [{key: needed}]
    options = [{key: needed}]
    if t.uses in values and t.uses != needed:
        options.append({key: t.uses})
    return options


def build_cases(
    *,
    family: str,
    seed: int,
    scratch: Path,
    tools: ToolEnv,
    templates: Callable[[Task], list[Tpl]],
    key: str,
    needed: Callable[[str, Task], str],
    values: tuple[str, ...],
    tier: str = "implicit",
    twins: tuple[str, ...] = ("clean", "faulted"),
) -> list[Case]:
    out: list[Case] = []
    for twin in twins:
        work = scratch / twin
        task = build_task(family, seed, twin, work, tools, tier, "free")
        tpls = templates(task)
        results = run_all(tools, work, tpls)
        for t in tpls:
            label = label_of(t, task, results[t.name])
            out.append(Case(Call(t.tool, t.argv, work), label, False, family, seed, twin, t.name))
            for decl in declarations(t, label, key, needed(twin, task), values):
                call = Call(t.tool, t.argv, work, decl)
                out.append(Case(call, label, True, family, seed, twin, t.name))
    return out
