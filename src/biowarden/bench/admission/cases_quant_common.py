"""Shared helpers of the quant admission case sources (no rule of its own).

Every case source of the pack builds one workspace per twin, runs the REAL quantifier for each
distinct way of calling it and compares the result with the task's truth; the rule is never asked.
`Template` is one call spelling; `key` names its effective semantics (the options that change the
numbers), and one real run is made per key and twin. A spelling variant that does not change the
numbers (`-s yes`, `--stranded=yes`, `-q`) shares the key and therefore the label; the family tests
(tests/test_family_quant.py, tests/test_rules_quant_*.py) run every spelling for real on a few
seeds and assert that the output of each one equals that of its key.

The declarations of a case follow one scheme (see `declarations`):
  right call   undeclared; plus the true value when the call's own value equals it
  wrong call   undeclared; plus the call's own value (declared but botched) and the true value
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.admission import Case, Label
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv
from biowarden.engine import Call

WORKERS = 2  # the login node is shared: at most two heavy processes at a time


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    key: str  # effective semantics: one real run per key and twin
    own: str | None = None  # the declaration value this call's options imply (None: no scheme)


def parallel(fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, items))


def label_of(truth: dict[str, Any], answer: Any) -> Label:
    """right iff a numeric answer is within the task's tolerance (None: the call failed)."""
    return "right" if answer is not None and answer_matches(truth, answer) else "wrong"


def declarations(label: Label, own: str | None, true: str) -> list[str]:
    if own is None:
        return [true]
    if label == "right":
        return [true] if own == true else []
    return list(dict.fromkeys([own, true]))


def cases_for(
    templates: Sequence[Template],
    labels: dict[str, Label],
    work: Path,
    *,
    key: str,
    true: str,
    family: str,
    seed: int,
    twin: str,
    cwd: Path | None = None,
) -> list[Case]:
    """Undeclared and declared cases for every template (the labels are keyed by Template.key)."""
    out: list[Case] = []
    for t in templates:
        label = labels[t.key]
        call = Call(t.tool, t.argv, cwd or work)
        out.append(Case(call, label, False, family, seed, twin, t.name))
        for value in declarations(label, t.own, true):
            declared = Call(t.tool, t.argv, cwd or work, {key: value})
            out.append(Case(declared, label, True, family, seed, twin, f"{t.name}+{value}"))
    return out


def run_ok(tools: ToolEnv, argv: Sequence[str], cwd: Path):
    """Run a tool; the process, or None when it could not start."""
    try:
        return tools.run(list(argv), cwd)
    except OSError:
        return None
