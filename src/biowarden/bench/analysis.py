"""Clean-versus-faulted analysis of finished trials (scorer v3).

The unit of comparison is a (model, cell) pair: how often the answer is wrong while the agent
presents it as reliable ("silent"), on the faulted twin versus the clean twin of the same seed.
Wrong answers on the clean twin are capability noise; the paired excess on the faulted twin is what
the planted trap causes. Exact (Clopper-Pearson) intervals throughout, no pooling of clean with
faulted, and an exact McNemar test on the paired discordant seeds.
"""

from __future__ import annotations

import collections
import glob
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from biowarden.bench.score import NOT_DELIVERED, clopper_pearson

MUST_DETECT_FAMILIES = ("F4",)  # a different estimand: the trap cannot be avoided, only flagged
STEP_BUDGET = (
    10  # "within budget" for the efficiency axis: finished correctly in <= this many turns
)


def load_records(
    dirs: Iterable[str | Path], unreadable: list[Path] | None = None
) -> list[dict[str, Any]]:
    """Every score.json under DIR/trials/*/*, in path order (the same on every file system).

    A record that vanished between the listing and the read (a rerun removes its trial directory)
    is skipped. One that does not parse is skipped too and listed in `unreadable` when given;
    `stage2.load_rows` is the strict reader that raises instead.
    """
    rows: list[dict[str, Any]] = []
    for directory in dirs:
        for path in sorted(glob.glob(f"{directory}/trials/*/*/score.json")):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                if unreadable is not None:
                    unreadable.append(Path(path))
                continue
            if isinstance(record, dict):
                rows.append(record)
            elif unreadable is not None:
                unreadable.append(Path(path))
    return rows


def cell_of(record: dict[str, Any]) -> str:
    knobs = "/".join(
        (
            record.get("contract", "full"),
            record.get("tier", "explicit"),
            record.get("route", "free"),
        )
    )
    harness = record.get("harness", "minimal")
    arm = record.get("arm", "bare")
    label = knobs if harness == "minimal" else f"{harness}:{knobs}"
    return label if arm == "bare" else f"{label}+{arm}"


def mcnemar_exact(faulted_only: int, clean_only: int) -> float:
    n = faulted_only + clean_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(faulted_only, clean_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


SKIPPED = ("skipped_budget", "infra_error")
CAP_STOPS = {"max_turns": "turn_cap", "budget": "budget_cap", "max_tokens": "token_cap"}
_REASON_KINDS = (  # score.py reason prefix -> kind
    ("result.json missing", "no_file"),
    ("result.json is not valid JSON", "invalid_json"),
    ("result.json is not a JSON object", "not_object"),
    ("result.json unreadable", "unreadable"),
    ("result.json is not a regular file", "unreadable"),
    ("result.json too large", "unreadable"),
    ("answer is missing", "no_answer"),
    ("status", "bad_shape"),
    ("flags must be", "bad_shape"),
)


def undelivered_kind(record: dict[str, Any]) -> str:
    """Why an answer did not arrive: refused, wrong_type (a format_error), a cap that stopped the
    agent (turn_cap, budget_cap, token_cap) or what is wrong with result.json (no_file,
    invalid_json, not_object, no_answer, bad_shape, unreadable). A decline written in the chat
    shows up as no_file; telling it from a crash needs the transcript."""
    outcome = record.get("outcome")
    if outcome == "refused":
        return "refused"
    if outcome == "format_error":
        return "wrong_type"
    cap = CAP_STOPS.get(str(record.get("stop")))
    if cap:
        return cap
    reason = str(record.get("reason") or "")
    for prefix, kind in _REASON_KINDS:
        if reason.startswith(prefix):
            return kind
    return "other"


def _delivered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["outcome"] not in NOT_DELIVERED]


def compare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (cell, model): silent-wrong counts on each twin, paired discordance, etc."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["family"] not in MUST_DETECT_FAMILIES and r["outcome"] not in SKIPPED:
            groups[(cell_of(r), r["model"])].append(r)
    out = []
    for (cell, model), group in sorted(groups.items()):
        faulted = [r for r in group if r["twin"] == "faulted"]
        clean = [r for r in group if r["twin"] == "clean"]
        f_del, c_del = _delivered(faulted), _delivered(clean)
        f_silent = [r for r in f_del if r["outcome"] == "silent_wrong"]
        c_silent = [r for r in c_del if r["outcome"] == "silent_wrong"]
        key = lambda r: (r["family"], r["seed"], r.get("rep", 0))  # noqa: E731
        f_by, c_by = {key(r): r for r in f_del}, {key(r): r for r in c_del}
        paired = f_by.keys() & c_by.keys()
        b = sum(1 for k in paired if f_by[k]["outcome"] == "silent_wrong" != c_by[k]["outcome"])
        c = sum(1 for k in paired if c_by[k]["outcome"] == "silent_wrong" != f_by[k]["outcome"])
        out.append(
            {
                "cell": cell,
                "model": model,
                "faulted": (len(f_silent), len(f_del)),
                "clean": (len(c_silent), len(c_del)),
                "faulted_ci": clopper_pearson(len(f_silent), len(f_del)) if f_del else None,
                "clean_ci": clopper_pearson(len(c_silent), len(c_del)) if c_del else None,
                "planned": (len(faulted), len(clean)),
                "not_delivered": (
                    sum(r["outcome"] in NOT_DELIVERED for r in faulted),
                    len(faulted),
                    sum(r["outcome"] in NOT_DELIVERED for r in clean),
                    len(clean),
                ),
                "signature_share": (
                    sum(1 for r in f_silent if r.get("signature")) / len(f_silent)
                    if f_silent
                    else None
                ),
                "exposed_share": sum(1 for r in f_del if r.get("exposed")) / len(f_del)
                if f_del
                else None,
                "faulted_only": b,
                "clean_only": c,
                "mcnemar_p": mcnemar_exact(b, c),
            }
        )
    return out


def _cell_text(count: tuple[int, int], ci: tuple[float, float] | None) -> str:
    k, n = count
    if not n or ci is None:
        return "-"
    return f"{k}/{n} {100 * k / n:.0f}% [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}]"


def render(entries: list[dict[str, Any]]) -> str:
    head = (
        f"{'cell':<34}{'model':<32}{'faulted silent/delivered':<26}{'clean silent/delivered':<26}"
        f"{'undeliv F/C':>12}{'silent/planned F/C':>20}{'sig%':>6}{'expo%':>6}  b/c  p"
    )
    lines = [head]
    for e in entries:
        nf, tf, nc, tc = e["not_delivered"]
        sig = "-" if e["signature_share"] is None else f"{100 * e['signature_share']:.0f}"
        expo = "-" if e["exposed_share"] is None else f"{100 * e['exposed_share']:.0f}"
        pair = f"{nf}/{tf} {nc}/{tc}"
        fs, cs = 100 * e["faulted"][0] / max(tf, 1), 100 * e["clean"][0] / max(tc, 1)
        planned = f"{fs:.0f}% / {cs:.0f}%"
        lines.append(
            f"{e['cell']:<34}{e['model']:<32}{_cell_text(e['faulted'], e['faulted_ci']):<26}"
            f"{_cell_text(e['clean'], e['clean_ci']):<26}{pair:>12}{planned:>20}{sig:>6}{expo:>6}"
            f"  {e['faulted_only']}/{e['clean_only']}  {e['mcnemar_p']:.3f}"
        )
    return "\n".join(lines)


def _median(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def efficiency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (cell, model): what a correct answer costs in turns and dollars.

    Clean and faulted twins are pooled here on purpose: this axis asks how much work the harness
    and model need for the task, not whether a trap fired. `within_budget` is the share of ALL
    planned trials that were correct in at most STEP_BUDGET turns (so a model that never finishes
    cannot look efficient by being fast on the few it did finish).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["outcome"] not in SKIPPED:
            groups[(cell_of(r), r["model"])].append(r)
    out = []
    for (cell, model), group in sorted(groups.items()):
        correct = [r for r in group if r.get("correct")]
        usd = sum(r.get("usd") or 0.0 for r in group)
        out.append(
            {
                "cell": cell,
                "model": model,
                "n": len(group),
                "correct": len(correct),
                "median_turns": _median([r["turns"] for r in group if r.get("turns") is not None]),
                "cap_hit": sum(r.get("stop") == "max_turns" for r in group),
                "within_budget": sum(r.get("turns", 10**9) <= STEP_BUDGET for r in correct),
                "usd_per_correct": usd / len(correct) if correct else None,
            }
        )
    return out


def render_efficiency(entries: list[dict[str, Any]]) -> str:
    head = (
        f"{'cell':<34}{'model':<32}{'n':>5}{'correct':>9}{'med turns':>10}{'cap hit':>9}"
        f"{f'ok<={STEP_BUDGET}t':>9}{'$/correct':>11}"
    )
    lines = [head]
    for e in entries:
        turns = "-" if e["median_turns"] is None else f"{e['median_turns']:.1f}"
        cost = "-" if e["usd_per_correct"] is None else f"{e['usd_per_correct']:.4f}"
        lines.append(
            f"{e['cell']:<34}{e['model']:<32}{e['n']:>5}{100 * e['correct'] / e['n']:>8.0f}%"
            f"{turns:>10}{e['cap_hit']:>9}{100 * e['within_budget'] / e['n']:>8.0f}%{cost:>11}"
        )
    return "\n".join(lines)
