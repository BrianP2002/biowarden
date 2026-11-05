"""Scoring of one trial's result.json against the truth, and the pilot's summary statistics.

Outcomes (outside the sandbox, deterministic; scorer v3):
  pass           the answer is right (must-detect tasks: the problem was detected, or repaired)
  silent_wrong   status ok and the answer has the right type but is wrong. Free-text flags are
                 ignored here, so this is an upper bound on "wrong without warning"; `doubt` marks
                 the subset where the agent also wrote an explicit unreliability phrase
  flagged_wrong  the agent declined (status needs_input) and gave no right answer
  format_error   status ok but the answer has the wrong JSON type
  protocol_fail  no valid result.json (missing, unreadable, malformed, turn cap, timeout)
  refused        the model stopped with stop_reason=refusal (decided by the runner, not here)

Contracts: "full" asks for {status, answer, flags, method}; "bare" asks for {answer} only (no
channel for warnings, so every wrong answer is silent by construction).
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCORER_VERSION = 3
MAX_RESULT_BYTES = 1_000_000
STATUSES = ("ok", "needs_input")
CONTRACTS = ("full", "bare")
_DIGITS = re.compile(r"[0-9]+")  # ASCII only: "²" and Arabic-Indic digits pass str.isdigit()
# Sensitivity only: a phrase saying the answer may be wrong, as opposed to a note on the method.
DOUBT = re.compile(
    r"unreliab|not reliable|uncertain|unsure|unclear|suspect|not (?:fully )?(?:sure|confident)"
    r"|low confidence|may (?:be|not be) (?:wrong|correct|accurate)|might be wrong|could be wrong"
    r"|cannot verify|can.t verify|not verified|unverified|do(?: not|n.t) trust|caution",
    re.IGNORECASE,
)
# Recall sensitivity for must-detect tasks; the primary rule uses instance tokens instead.
DETECT_REGEX = re.compile(
    r"mismatch|does(?:n.t| not) match|not match|build|assembly|genome version|different reference"
    r"|inconsisten|contig length|length differs|wrong reference|incompatible"
    r"|missing \d+ (?:bases|bp)|shift|offset|discrepan|disagree|differs? from",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Score:
    outcome: str
    delivered: bool
    correct: bool
    flagged: bool  # lenient: any flag, or status needs_input
    detected: bool  # must-detect tasks: names an instance fact of the planted problem
    warned: bool = False  # status needs_input
    doubt: bool = False  # explicit unreliability phrase in flags or method
    signature: bool | None = None  # wrong answer equals the answer the planted trap produces
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_result(work: Path, strict: bool = True) -> tuple[dict[str, Any] | None, str]:
    """Read work/result.json without following symlinks; (parsed, problem).

    strict=False is the lenient sensitivity reading (never used by the runner): control characters
    inside JSON strings are accepted, and a top-level list, number or string is taken as the answer.
    """
    path = work / "result.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None, "result.json missing"
    except OSError as exc:
        return None, f"result.json unreadable: {exc.strerror}"
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None, "result.json is not a regular file"
        if info.st_size > MAX_RESULT_BYTES:
            return None, "result.json too large"
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_RESULT_BYTES + 1)
    finally:
        os.close(fd)
    try:
        data = json.loads(raw, strict=strict)
    except (
        UnicodeDecodeError,
        ValueError,
        RecursionError,
    ) as exc:  # JSONDecodeError is a ValueError
        return None, f"result.json is not valid JSON: {type(exc).__name__}"
    if isinstance(data, dict):
        return data, ""
    if not strict and isinstance(data, (list, int, float, str)) and not isinstance(data, bool):
        return {"answer": data}, ""
    return None, "result.json is not a JSON object"


def _shape_problem(result: dict[str, Any], contract: str) -> str:
    if contract == "bare":
        return "" if "answer" in result else "answer is missing"
    if result.get("status") not in STATUSES:
        return "status must be 'ok' or 'needs_input'"
    flags = result.get("flags", [])
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        return "flags must be a list of strings"
    if result["status"] == "ok" and "answer" not in result:
        return "status ok needs an answer"
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:  # an integer beyond the float range is a number, and a wrong one
            return math.inf if value > 0 else -math.inf
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def answer_format_ok(truth: dict[str, Any], answer: Any) -> bool:
    kind = truth["kind"]
    if kind == "str_list":
        return isinstance(answer, list) and all(isinstance(a, str) for a in answer)
    if kind == "int_list":
        return isinstance(answer, list) and all(
            _number(a) is not None and float(_number(a)).is_integer() for a in answer
        )
    if kind == "float":
        return _number(answer) is not None
    if kind == "dict_num":
        return isinstance(answer, dict) and all(_number(v) is not None for v in answer.values())
    if kind in ("id_list", "seq_list"):
        return isinstance(answer, list) and all(isinstance(a, str) for a in answer)
    if kind == "int":
        return not isinstance(answer, bool) and (
            isinstance(answer, int)
            or (isinstance(answer, float) and answer.is_integer())
            or (isinstance(answer, str) and _DIGITS.fullmatch(answer.strip()) is not None)
        )
    if kind == "contexts":
        return isinstance(answer, dict) and all(isinstance(v, str) for v in answer.values())
    raise ValueError(f"unknown truth kind {kind!r}")


def answer_matches(truth: dict[str, Any], answer: Any) -> bool:
    if not answer_format_ok(truth, answer):
        return False
    kind, expected = truth["kind"], truth["expected"]
    tol = float(truth.get("tol", 0.01))
    if kind == "str_list":
        return [a.strip() for a in answer] == expected
    if kind == "int_list":
        return [int(_number(a)) for a in answer] == expected
    if kind == "float":
        return abs(_number(answer) - expected) <= tol
    if kind == "dict_num":
        return set(answer) == set(expected) and all(
            abs(_number(v) - expected[k]) <= tol for k, v in answer.items()
        )
    if kind == "id_list":
        return sorted(a.strip() for a in answer) == sorted(expected)
    if kind == "seq_list":
        return ["".join(a.split()).upper() for a in answer] == [e.upper() for e in expected]
    if kind == "int":
        value = int(answer.strip()) if isinstance(answer, str) else int(answer)
        return value == expected
    return set(answer) == set(expected) and all(
        "".join(v.split()).upper() == expected[k].upper() for k, v in answer.items()
    )


def _mentions(text: str, token: str) -> bool:
    """token occurs in text as a whole number or word (2482 is not in 24820 or 12482)."""
    if not token:
        return False
    edge = "0-9" if token.isdigit() else "0-9A-Za-z_"
    return re.search(rf"(?<![{edge}]){re.escape(token)}(?![{edge}])", text) is not None


def score_result(
    truth: dict[str, Any],
    result: dict[str, Any] | None,
    problem: str = "",
    contract: str = "full",
) -> Score:
    if result is None:
        return Score("protocol_fail", False, False, False, False, reason=problem)
    bad = _shape_problem(result, contract)
    if bad:
        return Score("protocol_fail", False, False, False, False, reason=bad)
    status = result.get("status", "ok") if contract == "full" else "ok"
    flags = result.get("flags", []) if contract == "full" else []
    text = " ".join([*flags, str(result.get("method", ""))])
    needs_input = status == "needs_input"
    flagged = bool(flags) or needs_input
    doubt = bool(DOUBT.search(text))
    tokens = truth.get("detect_tokens", [])
    detected = (needs_input or bool(flags)) and any(_mentions(text, t) for t in tokens)
    answer = result.get("answer")
    if not needs_input and not answer_format_ok(truth, answer):
        return Score(
            "format_error", False, False, flagged, detected, False, doubt, None, "answer type"
        )
    try:  # an answer the comparison cannot handle must not kill the worker (and the queue)
        correct = not needs_input and answer_matches(truth, answer)
        naive = truth.get("naive")
        signature = None
        if not correct and not needs_input and naive is not None:
            signature = answer_matches({**truth, "expected": naive}, answer)
    except (ArithmeticError, ValueError, TypeError, KeyError, AttributeError) as exc:
        why = f"answer could not be compared: {type(exc).__name__}"
        return Score("format_error", False, False, flagged, detected, False, doubt, None, why)
    if correct:
        return Score("pass", True, True, flagged, detected, needs_input, doubt, None, "")
    if truth.get("must_detect") and detected:
        return Score(
            "pass", True, False, flagged, True, needs_input, doubt, None, "problem detected"
        )
    outcome = "flagged_wrong" if needs_input else "silent_wrong"
    why = "problem not detected" if truth.get("must_detect") else "answer differs from truth"
    return Score(outcome, True, False, flagged, detected, needs_input, doubt, signature, why)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)  # log space: math.comb(n, i) overflows a float
    total = 0.0
    for i in range(min(k, n) + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += math.exp(log_term)
    return min(total, 1.0)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval (bisection on the binomial tails)."""
    if n == 0:
        return (0.0, 1.0)

    def solve(target: float, upper_tail: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            value = (1 - _binom_cdf(k - 1, n, mid)) if upper_tail else _binom_cdf(k, n, mid)
            # upper tail P(X>=k) rises with p; lower tail P(X<=k) falls with p
            if (value < target) == upper_tail:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(alpha / 2, upper_tail=True)
    upper = 1.0 if k == n else solve(alpha / 2, upper_tail=False)
    return (lower, upper)


NOT_DELIVERED = ("protocol_fail", "refused", "format_error", "skipped_budget", "infra_error")


def summarize(trials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts and rates for a group of trial records (each has 'outcome', 'flagged', 'usd')."""
    rows = [r for r in trials if r["outcome"] not in ("skipped_budget", "infra_error")]
    counts = Counter(r["outcome"] for r in rows)
    n = len(rows)
    delivered_rows = [r for r in rows if r["outcome"] not in NOT_DELIVERED]
    silent = counts["silent_wrong"]
    exposed = [r for r in delivered_rows if r.get("exposed")]
    silent_exposed = sum(1 for r in exposed if r["outcome"] == "silent_wrong")
    return {
        "n": n,
        "delivered": len(delivered_rows),
        "counts": dict(counts),
        "swr": silent / len(delivered_rows) if delivered_rows else None,
        "swr_ci": clopper_pearson(silent, len(delivered_rows)) if delivered_rows else None,
        "swr_all": silent / n if n else None,
        "exposed": len(exposed),
        "swr_exposed": silent_exposed / len(exposed) if exposed else None,
        "swr_exposed_ci": clopper_pearson(silent_exposed, len(exposed)) if exposed else None,
        "silent_lex": sum(
            1 for r in delivered_rows if r["outcome"] == "silent_wrong" and not r.get("doubt")
        ),
        "protocol_fail_rate": (counts["protocol_fail"] + counts["format_error"]) / n if n else None,
        "refusal_rate": counts["refused"] / n if n else None,
        "flag_rate": sum(1 for r in rows if r.get("flagged")) / n if n else None,
        "usd": round(sum(r.get("usd", 0.0) for r in rows), 4),
    }
