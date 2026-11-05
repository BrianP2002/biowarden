"""Shared machinery of the sc pack's admission cases (registers no rule).

A case source builds the workspaces of one seed (both twins of a family, implicit tier), then
turns a list of `Template`s (python-sc code an agent could write) into labelled engine cases.

LABELS come from running the code with the real libraries in the workspace, never from a rule:
the last line of stdout is parsed as JSON and compared with `task.truth` by `answer_matches`;
a crash, empty output or a wrong value is "wrong". To keep 40 seeds affordable the code of one
seed is executed in ONE long-lived python-sc process (`ScWorker`): scanpy is imported once, each
snippet runs under `exec` with its own globals, stdout / stderr captured, the working directory set
to the workspace. This is the same interpreter and the same libraries as `python-sc -c`, only
without a process start per snippet; tests/test_rules_sc_*.py cross-check a sample against a real
`tools.run(["python-sc", "-c", code])`.

A template is emitted in one of three forms (the call the engine sees):
  c        python-sc -c CODE                      (the code is an argument)
  script   python-sc NAME.py                      (the code is a file in the workspace)
  heredoc  python-sc -   with the code as context (what the shell pre-check passes)
Each form of one snippet shares the label of its single execution.

Cells (the four of bench.admission): for a right call the true declaration of the twin gives a
right_declared case; the undeclared call is a right_undeclared case on a twin whose data are not
in the risky state (the clean one: zero friction is promised) and on a risky twin only when the
template's code visibly handles the state (`evident`; `risky_twins` says which twins are risky,
a loom file is genes x cells in both). For a wrong call the undeclared call and the declarations in
`wrong_decl` (default: the true one) give wrong_undeclared and wrong_declared cases. Inspection
templates (print, shape, head) produce no answer and are right by construction.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, Label
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra.f101_de_use_raw import BUILD_HOOKS
from science_harness.engine import Call

_WORKER = r"""
import contextlib, io, json, os, sys, traceback
for line in sys.stdin:
    req = json.loads(line)
    os.chdir(req["cwd"])
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    sys.argv = ["-c", *req.get("argv", [])]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(req["code"], "<snippet>", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        rc = 1
        err.write(traceback.format_exc())
    sys.stdout.write(json.dumps({"rc": rc, "out": out.getvalue(), "err": err.getvalue()}) + "\n")
    sys.stdout.flush()
"""


class ScWorker:
    """One long-lived python-sc process that executes snippets (see the module docstring)."""

    def __init__(self, tools: ToolEnv) -> None:
        exe = tools.real_bins(["python-sc"]).get("python-sc")
        if exe is None:
            raise FileNotFoundError("python-sc is not available")
        self._proc = subprocess.Popen(
            [str(exe), "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=tools.process_env(),
        )
        self._lock = threading.Lock()
        atexit.register(self.close)

    def run(self, code: str, cwd: Path, argv: list[str] | None = None) -> tuple[int, str, str]:
        with self._lock:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            request = {"code": code, "cwd": str(cwd), "argv": argv or []}
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            reply = self._proc.stdout.readline()
        if not reply:
            raise RuntimeError("the python-sc worker died")
        data = json.loads(reply)
        return data["rc"], data["out"], data["err"]

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


_WORKERS: dict[str, ScWorker] = {}


def worker_for(tools: ToolEnv) -> ScWorker:
    key = str(tools.real_bins(["python-sc"]).get("python-sc"))
    worker = _WORKERS.get(key)
    if worker is None or worker._proc.poll() is not None:
        worker = _WORKERS[key] = ScWorker(tools)
    return worker


@dataclass(frozen=True)
class Template:
    name: str
    code: str
    form: str = "c"  # c | script | heredoc
    inspection: bool = False
    evident: bool = (
        False  # the code visibly handles the state (right undeclared on the faulted twin)
    )
    decl_true: dict[str, str] | None = None  # declaration of a right call (default: the twin's)
    wrong_decl: tuple[dict[str, str], ...] = ()  # declarations tried on a wrong call
    run_as: str = ""  # run this code instead (for forms that need a different snippet)
    tool: str = "python-sc"  # another program (awk): run with tools.run, argv in `argv`
    argv: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def parse_answer(stdout: str) -> Any:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError:
        return None


def label_of(task: Task, template: Template, rc: int, out: str) -> Label:
    if rc != 0:
        return "wrong"
    if template.inspection:
        return "right"
    answer = parse_answer(out)
    try:
        return "right" if answer is not None and answer_matches(task.truth, answer) else "wrong"
    except (TypeError, ValueError, KeyError):
        return "wrong"


def make_call(template: Template, work: Path, declared: dict[str, str] | None) -> Call:
    decl = declared or {}
    if template.tool != "python-sc":
        return Call(template.tool, template.argv, work, decl)
    if template.form == "script":
        path = work / f"{template.name}.py"
        path.write_text(template.code)
        return Call("python-sc", (path.name,), work, decl)
    if template.form == "heredoc":
        return Call("python-sc", ("-",), work, decl, (template.code,))
    return Call("python-sc", ("-c", template.code), work, decl)


def make_cases(
    family: str,
    seed: int,
    scratch: Path,
    tools: ToolEnv,
    templates: Callable[[Task, str, Path], list[Template]],
    true_decl: Callable[[str], dict[str, str]],
    tier: str = "implicit",
    after_build: Callable[[Path, Task, str], None] | None = None,
    risky_twins: tuple[str, ...] = ("faulted",),
) -> list[Case]:
    """Cases of one seed: both twins, every template, labelled by running the code."""
    out: list[Case] = []
    worker = worker_for(tools)
    BUILD_HOOKS.append(partial(_build_run, worker))
    try:
        _fill(
            out,
            family,
            seed,
            scratch,
            tools,
            templates,
            true_decl,
            tier,
            after_build,
            worker,
            risky_twins,
        )
    finally:
        BUILD_HOOKS.pop()
    return out


def _build_run(worker: ScWorker, code: str, argv: list[str], cwd: Path) -> tuple[int, str]:
    rc, _, err = worker.run(code, cwd, argv)
    return rc, err


def _fill(
    out: list[Case],
    family: str,
    seed: int,
    scratch: Path,
    tools: ToolEnv,
    templates: Callable[[Task, str, Path], list[Template]],
    true_decl: Callable[[str], dict[str, str]],
    tier: str,
    after_build: Callable[[Path, Task, str], None] | None,
    worker: ScWorker,
    risky_twins: tuple[str, ...],
) -> None:
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task(family, seed, twin, work, tools, tier, "free")
        if after_build is not None:
            after_build(work, task, twin)
        done: dict[str, tuple[int, str]] = {}
        for template in templates(task, twin, work):
            code = template.run_as or template.code
            if template.tool != "python-sc":
                code = json.dumps([template.tool, *template.argv])
            if code not in done:
                if template.tool != "python-sc":
                    proc = tools.run([template.tool, *template.argv], work)
                    done[code] = (proc.returncode, proc.stdout)
                else:
                    rc, stdout, _ = worker.run(code, work)
                    done[code] = (rc, stdout)
            rc, stdout = done[code]
            label = label_of(task, template, rc, stdout)
            note = template.name
            decl_true = template.decl_true if template.decl_true is not None else true_decl(twin)
            risky_twin = twin in risky_twins
            if label == "right":
                if not risky_twin or template.evident or template.inspection:
                    out.append(
                        Case(
                            make_call(template, work, None), label, False, family, seed, twin, note
                        )
                    )
                out.append(
                    Case(
                        make_call(template, work, decl_true), label, True, family, seed, twin, note
                    )
                )
            else:
                out.append(
                    Case(make_call(template, work, None), label, False, family, seed, twin, note)
                )
                options = template.wrong_decl or (decl_true,)
                for decl in options:
                    out.append(
                        Case(make_call(template, work, decl), label, True, family, seed, twin, note)
                    )
