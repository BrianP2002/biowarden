"""From rule verdicts to allow / warn / block.

Modes (one per experiment arm):
  off       the engine is a no-op (the bare arm)
  warn      every failing rule only warns; the call still runs
  block     every failing rule blocks
  promoted  a failing rule blocks only if the promotion table says so, otherwise it warns
An `unknown` verdict only ever warns. A rule of tier T2 only ever warns. A checker that crashes
fails closed (blocks) in block and promoted modes. A fault of the layer itself (a host path it
cannot stat, an evaluation that runs out of time) is not the call's fault: the rule it hit is
`unknown` with code `layer_error`, so it only warns. Nothing here rewrites a call.
"""

from __future__ import annotations

import errno
import fcntl
import fnmatch
import json
import posixpath
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from science_harness.engine.argv import has_option
from science_harness.engine.call import Call
from science_harness.engine.checks import (
    CheckResult,
    Ctx,
    LayerTimeout,
    Match,
    is_regular_file,
    known_checkers,
    suggestion_for,
)
from science_harness.engine.spec import Rule, Spec
from science_harness.ledger import Ledger

Mode = Literal["off", "warn", "block", "promoted"]
Action = Literal["allow", "warn", "block"]
Effective = Literal["none", "warn", "block"]
PACKAGED_PROMOTION = Path(__file__).parent / "specs" / "promotion.json"
MAX_TOKEN_SCAN = 200_000  # characters of one call (arguments, then earlier words) searched
EVALUATE_SECONDS = 20.0  # an evaluation that takes longer skips the rules it has not reached
_RANK = {"none": 0, "warn": 1, "block": 2}
_VERSIONED = re.compile(r"(python\d)(?:\.\d+)+")
# what the host answers for a name it may not or cannot stat: a fault of the place, not of the call
HOST_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.ENAMETOOLONG, errno.ELOOP})


def tool_names(tool: str) -> tuple[str, ...]:
    """The names a rule may list for the program of a call: what was run, by its basename, and
    for a versioned interpreter also its plain name (`/env/bin/python3.10`: python3.10, python3)."""
    base = posixpath.basename(tool)
    plain = _VERSIONED.fullmatch(base)
    return (base, plain.group(1)) if plain else (base,)


@dataclass(frozen=True)
class Promotion:
    """rule id -> measured evidence and the mode it earned; absent means warn."""

    entries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def mode_of(self, rule_id: str, spec_hash: str | None = None) -> Literal["warn", "block"]:
        """`block` only for a promoted rule whose spec is the one that was measured."""
        entry = self.entries.get(rule_id, {})
        if entry.get("mode") != "block":
            return "warn"
        if spec_hash is not None and entry.get("spec_hash") not in (None, spec_hash):
            return "warn"  # the spec changed since the measurement: the evidence no longer applies
        return "block"

    @classmethod
    def load(cls, path: str | Path) -> Promotion:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def packaged(cls) -> Promotion:
        """The table shipped with the specs (written by `bench.admission`)."""
        path = PACKAGED_PROMOTION
        return cls.load(path) if path.is_file() else cls()

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dict(self.entries), indent=1, sort_keys=True))


class BlockState:
    """How often each call was already blocked; separate shim processes share it through a file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._memory: dict[str, int] = {}

    def count(self, call_hash: str) -> int:
        if self.path is None:
            return self._memory.get(call_hash, 0)
        return self._update(call_hash, add=0)

    def record(self, call_hash: str) -> int:
        if self.path is None:
            self._memory[call_hash] = self._memory.get(call_hash, 0) + 1
            return self._memory[call_hash]
        return self._update(call_hash, add=1)

    def _update(self, call_hash: str, add: int) -> int:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read()
            counts: dict[str, int] = json.loads(raw) if raw.strip() else {}
            counts[call_hash] = counts.get(call_hash, 0) + add
            if add:
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(counts))
            return counts[call_hash]


@dataclass(frozen=True)
class RuleOutcome:
    rule: Rule
    result: CheckResult
    effective: Effective
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    action: Action
    outcomes: tuple[RuleOutcome, ...]
    call_hash: str
    mode: Mode
    stop: bool = False  # the same call was already blocked `retry_cap` times
    acknowledged: bool = False  # generic style: a second identical call is let through
    acknowledged: bool = False  # generic style: a second identical call is let through

    def firing(self) -> list[RuleOutcome]:
        return [o for o in self.outcomes if o.effective != "none"]


class Engine:
    def __init__(
        self,
        specs: Sequence[Spec],
        mode: Mode = "promoted",
        promotion: Promotion | None = None,
        real_bins: Mapping[str, Path] | None = None,
        real_env: Mapping[str, str] | None = None,
        state: BlockState | None = None,
        ledger: Ledger | None = None,
        retry_cap: int = 2,
        style: Literal["full", "generic"] = "full",
        contract: Literal["full", "bare"] = "full",
        max_seconds: float = EVALUATE_SECONDS,
    ) -> None:
        self.specs = list(specs)
        self.mode: Mode = mode
        self.promotion = promotion or Promotion()
        self.real_bins = dict(real_bins or {})
        self.real_env = dict(real_env) if real_env is not None else None
        self.state = state or BlockState()
        self.ledger = ledger
        self.retry_cap = retry_cap
        self.style = style  # full: rule text and fix; generic: detection only, one speed bump
        self.contract = contract  # the result.json the agent was asked for: it words the STOP text
        self.max_seconds = max_seconds  # the watchdog of one evaluation
        self._lock = threading.RLock()
        self._spec_hash = {r.id: s.content_hash() for s in self.specs for r in s.rules}
        import science_harness.engine.rules  # noqa: F401  (registers the packaged checkers)

        seen: set[str] = set()
        for spec in self.specs:
            for rule in spec.rules:
                if rule.id in seen:
                    raise ValueError(f"rule id {rule.id!r} is defined more than once")
                seen.add(rule.id)
        registry = known_checkers()
        missing = [r.check for s in self.specs for r in s.rules if r.check not in registry]
        if missing:
            raise ValueError(f"specs name unknown checkers: {sorted(set(missing))}")

    # -- matching ---------------------------------------------------------------------------

    def _match(self, call: Call, rule: Rule) -> Match | None:
        applies = rule.applies
        names = tool_names(call.tool)
        if rule.kind == "call":
            if not any(name in applies.tools for name in names):
                return None
            if applies.subcommands and not any(
                fnmatch.fnmatchcase(call.subcommand or "", pattern)
                for pattern in applies.subcommands
            ):
                return None
            if applies.options_any and not has_option(call.argv, *applies.options_any):
                return None
            return Match(rule)
        if applies.tools and not any(name in applies.tools for name in names):
            return None
        found: list[Path] = []
        budget = MAX_TOKEN_SCAN
        for token in (*call.argv, *call.context):
            if budget <= 0:
                break
            token, budget = token[:budget], budget - len(token)
            for pattern in applies.file_patterns:
                for hit in re.finditer(pattern, token):
                    path = Path(hit.group(0))
                    path = path if path.is_absolute() else call.cwd.absolute() / path
                    if path not in found and is_regular_file(path, call.root):
                        found.append(path)
        return Match(rule, tuple(found)) if found else None

    # -- evaluation -------------------------------------------------------------------------

    def _run(self, call: Call, match: Match, deadline: float | None = None) -> CheckResult:
        rule = match.rule
        if rule.declare is not None:
            value = call.declared.get(rule.declare.key)
            if value is not None and value not in rule.declare.values:
                return CheckResult(
                    "fail",
                    "declaration_invalid",
                    f"{rule.declare.key}={value} is not one of {rule.declare.values}",
                    expected=rule.declare.values,
                    got=value,
                    fix=(f"use --declare {rule.declare.key}=<{'|'.join(rule.declare.values)}>",),
                )
            choices = "|".join(rule.declare.values)
            if value is None and rule.kind == "call" and rule.declare.required:
                return CheckResult(
                    "fail",
                    "declaration_missing",
                    f"state which {rule.slot} convention you intend ({rule.message})",
                    expected=rule.declare.values,
                    fix=self._missing_fix(call, rule, choices),
                )
        ctx = Ctx(
            call.cwd, rule.tier, self.real_bins, self.real_env, root=call.root, deadline=deadline
        )
        try:
            return known_checkers()[rule.check](call, match, ctx)
        except (LayerTimeout, subprocess.TimeoutExpired) as exc:  # out of time: not the call
            return self._fault(exc)
        except Exception as exc:  # a crashing checker must never pass silently
            if isinstance(exc, OSError) and exc.errno in HOST_ERRNOS:
                return self._fault(exc)  # a rule that stats a name of the call by itself
            return CheckResult("fail", "checker_crashed", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _fault(reason: BaseException | str) -> CheckResult:
        """A rule the layer could not check. The agent reads the message; the ledger, the cause."""
        note = reason if isinstance(reason, str) else f"{type(reason).__name__}: {reason}"
        return CheckResult(
            "unknown", "layer_error", "this rule could not be checked on this call", note=note[:300]
        )

    @staticmethod
    def _missing_fix(call: Call, rule: Rule, choices: str) -> tuple[str, ...]:
        wording = suggestion_for(rule.check)
        if wording is not None:
            try:
                return wording(call, rule)
            except Exception:  # a broken suggester must not hide the block
                pass
        assert rule.declare is not None
        return (f"re-run with --declare {rule.declare.key}=<{choices}>",)

    def _effective(self, rule: Rule, result: CheckResult) -> Effective:
        if self.mode == "off" or result.verdict == "pass":
            return "none"
        if result.verdict == "unknown" or rule.tier == "T2" or self.mode == "warn":
            return "warn"
        if result.code == "checker_crashed" or self.mode == "block":
            return "block"
        promoted = self.promotion.mode_of(rule.id, self._spec_hash[rule.id]) == "block"
        return "block" if promoted else "warn"

    def evaluate(self, call: Call, source: str = "shim") -> Decision:
        with self._lock:  # shims, the pre-check and the gateway share one engine and one ledger
            return self._evaluate(call, source)

    def _evaluate(self, call: Call, source: str) -> Decision:
        outcomes: list[RuleOutcome] = []
        if self.mode != "off":
            deadline = time.monotonic() + self.max_seconds
            rules = [rule for spec in self.specs for rule in spec.rules]
            for position, rule in enumerate(rules):
                if time.monotonic() > deadline:  # the watchdog: the rules left are not evaluated
                    left = len(rules) - position
                    result = self._fault(f"{left} rules not run after {self.max_seconds:g} s")
                    outcomes.append(RuleOutcome(rule, result, self._effective(rule, result)))
                    break
                outcome = self._judge(call, rule, deadline)
                if outcome is not None:
                    outcomes.append(outcome)
        worst: Effective = max((o.effective for o in outcomes), key=_RANK.get, default="none")  # type: ignore[arg-type]
        action: Action = {"none": "allow", "warn": "warn", "block": "block"}[worst]
        call_hash = call.call_hash()
        acknowledged = False
        if action == "block" and self.style == "generic" and self.state.count(call_hash) >= 1:
            action, acknowledged = "allow", True  # the agent looked again and repeated the call
        stop = (
            action == "block"
            and self.style == "full"
            and self.state.count(call_hash) >= self.retry_cap
        )
        if action == "block":
            self.state.record(call_hash)
        decision = Decision(action, tuple(outcomes), call_hash, self.mode, stop, acknowledged)
        self._log(call, decision, source)
        return decision

    def _judge(self, call: Call, rule: Rule, deadline: float) -> RuleOutcome | None:
        """The outcome of one rule for a call; None when the rule does not apply."""
        files: tuple[str, ...] = ()
        try:
            match = self._match(call, rule)
            if match is None:
                return None
            files = tuple(str(f) for f in match.files)
            result = self._run(call, match, deadline)
        except Exception as exc:  # a fault of the layer costs this rule's check, never the call
            result = self._fault(exc)
        return RuleOutcome(rule, result, self._effective(rule, result), files)

    def _log(self, call: Call, decision: Decision, source: str) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            {
                "event": "engine_decision",
                "source": source,
                "tool": call.tool,
                "argv": list(call.argv),
                "declared": dict(call.declared),
                "call_hash": decision.call_hash,
                "mode": decision.mode,
                "action": decision.action,
                "stop": decision.stop,
                "acknowledged": decision.acknowledged,
                "spec_hashes": {s.name: s.content_hash() for s in self.specs},
                "outcomes": [
                    {
                        "rule": o.rule.id,
                        "verdict": o.result.verdict,
                        "code": o.result.code,
                        "effective": o.effective,
                        "note": o.result.note,
                    }
                    for o in decision.outcomes
                ],
            }
        )
