"""Admission: measure a rule against labelled calls before it may block anything.

Each rule has a case source. For one seed it builds benchmark workspaces (clean and faulted twins)
and returns calls with a ground-truth label obtained by running the real tool: would this call's
result be right for this workspace? Running the engine over many seeds gives, without any model:

  right_declared    a correct call that carries the declaration  -> the rule must NOT fail it
  right_undeclared  a correct call without the declaration       -> friction (reported, not gated)
  wrong_declared    a wrong call with a declaration              -> recall
  wrong_undeclared  a wrong call without one                      -> recall

A rule is promoted to blocking only when its false-block rate on right_declared has an exact
upper bound <= 2% over >= 200 calls. A rule is admitted at all only when at least one wrong call
is caught (a killed mutant) and no right_declared call fails. A conditional rule (a fact rule, or a
call rule whose declaration is optional) promises zero friction on correct data, so right_undeclared
calls are gated the same way.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from biowarden.bench.score import clopper_pearson
from biowarden.bench.tasks.tools import ToolEnv
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.tools import call_rule_tools

Label = Literal["right", "wrong"]
UPPER_LIMIT = 0.02
MIN_CALLS = 200


@dataclass(frozen=True)
class Case:
    call: Call
    label: Label  # ground truth from running the real tool, never from the rule under test
    declared: bool  # the call carries the declaration the rule asks for
    family: str
    seed: int
    twin: str
    note: str = ""


CaseSource = Callable[[int, Path, ToolEnv], list[Case]]
_SOURCES: dict[str, CaseSource] = {}


def case_source(rule_id: str) -> Callable[[CaseSource], CaseSource]:
    def register(fn: CaseSource) -> CaseSource:
        if rule_id in _SOURCES:
            raise ValueError(f"case source for {rule_id!r} is already registered")
        _SOURCES[rule_id] = fn
        return fn

    return register


def known_sources() -> dict[str, CaseSource]:
    """Every registered case source; modules named cases_*.py in this package are imported first."""
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("cases_"):
            importlib.import_module(f"{__name__}.{module.name}")
    return dict(_SOURCES)


@dataclass
class Cell:
    n: int = 0
    failed: int = 0  # the rule's verdict was fail
    unknown: int = 0
    crashed: int = 0

    def upper(self) -> float:
        return clopper_pearson(self.failed, self.n)[1]


@dataclass
class Stats:
    rule_id: str
    cells: dict[tuple[Label, bool], Cell] = field(
        default_factory=lambda: {
            (lab, dec): Cell() for lab in ("right", "wrong") for dec in (True, False)
        }
    )
    seeds: int = 0
    conditional: bool = False  # right_undeclared calls must not fail either

    @property
    def right_declared(self) -> Cell:
        return self.cells[("right", True)]

    def recall(self) -> float | None:
        wrong = [self.cells[("wrong", True)], self.cells[("wrong", False)]]
        n = sum(c.n for c in wrong)
        return sum(c.failed for c in wrong) / n if n else None

    def killed(self) -> int:
        return sum(self.cells[("wrong", d)].failed for d in (True, False))

    def admitted(self) -> bool:
        friction_free = not self.conditional or self.cells[("right", False)].failed == 0
        return self.killed() >= 1 and self.right_declared.failed == 0 and friction_free

    def promotion_entry(self) -> dict[str, Any]:
        rd = self.right_declared
        ru = self.cells[("right", False)]
        promoted = self.admitted() and rd.n >= MIN_CALLS and rd.upper() <= UPPER_LIMIT
        if self.conditional:
            promoted = promoted and ru.n >= MIN_CALLS and ru.upper() <= UPPER_LIMIT
        return {
            "mode": "block" if promoted else "warn",
            "seeds": self.seeds,
            "right_declared": [rd.failed, rd.n],
            "false_block_upper": round(rd.upper(), 4) if rd.n else None,
            "right_undeclared": [
                self.cells[("right", False)].failed,
                self.cells[("right", False)].n,
            ],
            "recall": None if self.recall() is None else round(self.recall(), 4),
        }


def measure(
    rule_id: str,
    specs: Sequence[Spec],
    seeds: Sequence[int],
    tools: ToolEnv,
    scratch: Path,
    source: CaseSource | None = None,
) -> Stats:
    """Run the engine (mode block, no retry cap) over the rule's cases for every seed."""
    source = source or known_sources()[rule_id]
    rule = next(r for spec in specs for r in spec.rules if r.id == rule_id)
    stats = Stats(
        rule_id, conditional=rule.kind == "fact" or not (rule.declare and rule.declare.required)
    )
    real_env: Mapping[str, str] = tools.process_env()
    engine = Engine(
        specs,
        mode="block",
        real_bins=tools.real_bins(call_rule_tools(specs)),
        real_env=real_env,
        retry_cap=10**9,
    )
    for seed in seeds:
        stats.seeds += 1
        for case in source(seed, scratch / f"{rule_id}-s{seed:04d}", tools):
            decision = engine.evaluate(case.call)
            own = [o for o in decision.outcomes if o.rule.id == rule_id]
            cell = stats.cells[(case.label, case.declared)]
            cell.n += 1
            cell.failed += any(o.result.verdict == "fail" for o in own)
            cell.unknown += any(o.result.verdict == "unknown" for o in own)
            cell.crashed += any(o.result.code == "checker_crashed" for o in own)
    return stats
