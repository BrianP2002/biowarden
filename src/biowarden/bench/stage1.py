"""Stage-1 analysis (docs/preregistration/stage1.md): layer arms against the bare arm.

    python -I -m biowarden.bench.stage1 --out runs/stage1

Unit of comparison: the faulted-twin trial of one (family, seed) for one model. `silent` is 1 when
the trial's outcome is silent_wrong, over ALL planned trials (an undelivered trial counts as not
silent, so an arm cannot look good by never answering; the delivered, flagged and friction columns
show that). Exact McNemar per model and arm against `bare`, Holm over a model's non-bare arms.
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
import sys
from collections.abc import Iterable
from typing import Any

from biowarden.bench.analysis import load_records, mcnemar_exact
from biowarden.bench.score import NOT_DELIVERED, clopper_pearson

FAMILIES = ("F1", "F7", "F10", "F13", "F15")
ARMS = ("bare", "warn", "block", "block-bypass", "block-gateway")
PRIMARY = ("bare", "implicit", "forced")  # contract, tier, route
SAFETY = ("full", "explicit", "free")
SKIPPED = ("skipped_budget", "infra_error")
MIN_DROP = 0.10  # preregistered: silent/planned must fall by at least this much ...
MIN_MODELS = 3  # ... for at least this many open-weight models ...
MAX_PASS_LOSS = 0.05  # ... while pooled clean pass rate and delivered share lose at most this much
ALPHA = 0.05


def _cell(r: dict[str, Any]) -> tuple[str, str, str]:
    return (r.get("contract", "full"), r.get("tier", "explicit"), r.get("route", "free"))


def usable(rows: Iterable[dict[str, Any]], cell: tuple[str, str, str]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r["family"] in FAMILIES
        and r["outcome"] not in SKIPPED
        and _cell(r) == cell
        and r.get("harness", "minimal") == "minimal"
    ]


def key_of(r: dict[str, Any]) -> tuple[str, int, int]:
    return (r["family"], r["seed"], r.get("rep", 0))


def is_open(model: str) -> bool:
    return "/" in model


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def _by_key(
    rows: list[dict[str, Any]], model: str, arm: str, twin: str
) -> dict[Any, dict[str, Any]]:
    return {
        key_of(r): r
        for r in rows
        if r["model"] == model and r.get("arm", "bare") == arm and r["twin"] == twin
    }


def silent_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (model, non-bare arm): paired faulted silent/planned against bare."""
    models = sorted({r["model"] for r in rows})
    out: list[dict[str, Any]] = []
    for model in models:
        base = _by_key(rows, model, "bare", "faulted")
        entries: dict[str, dict[str, Any]] = {}
        for arm in ARMS[1:]:
            other = _by_key(rows, model, arm, "faulted")
            pairs = base.keys() & other.keys()
            if not pairs:
                continue
            b = sum(
                base[k]["outcome"] == "silent_wrong" != (other[k]["outcome"] == "silent_wrong")
                for k in pairs
            )
            only_bare = sum(
                base[k]["outcome"] == "silent_wrong" and other[k]["outcome"] != "silent_wrong"
                for k in pairs
            )
            only_arm = sum(
                other[k]["outcome"] == "silent_wrong" and base[k]["outcome"] != "silent_wrong"
                for k in pairs
            )
            del b
            n = len(pairs)
            k_bare = sum(base[k]["outcome"] == "silent_wrong" for k in pairs)
            k_arm = sum(other[k]["outcome"] == "silent_wrong" for k in pairs)
            entries[arm] = {
                "model": model,
                "arm": arm,
                "pairs": n,
                "bare": (k_bare, n),
                "arm_rate": (k_arm, n),
                "drop": (k_bare - k_arm) / n,
                "bare_only": only_bare,
                "arm_only": only_arm,
                "p": mcnemar_exact(only_arm, only_bare),
                "arm_ci": clopper_pearson(k_arm, n),
            }
        adjusted = holm({a: e["p"] for a, e in entries.items()})
        for arm, entry in entries.items():
            entry["p_holm"] = adjusted[arm]
            out.append(entry)
    return out


def side_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (model, arm): clean pass and friction, delivered and flagged shares, turns, cost."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r.get("arm", "bare"))].append(r)
    out = []
    for (model, arm), group in sorted(groups.items()):
        clean = [r for r in group if r["twin"] == "clean"]
        correct = [r for r in group if r.get("correct")]
        blocked_clean = sum(
            ((r.get("layer") or {}).get("actions") or {}).get("block", 0) > 0 for r in clean
        )
        out.append(
            {
                "model": model,
                "arm": arm,
                "n": len(group),
                "clean_pass": (sum(r["outcome"] == "pass" for r in clean), len(clean)),
                "clean_blocked": (blocked_clean, len(clean)),
                "delivered": sum(r["outcome"] not in NOT_DELIVERED for r in group) / len(group),
                "flagged_wrong": sum(r["outcome"] == "flagged_wrong" for r in group) / len(group),
                "median_turns": statistics.median(
                    r["turns"] for r in group if r.get("turns") is not None
                )
                if any(r.get("turns") is not None for r in group)
                else None,
                "usd_per_correct": sum(r.get("usd") or 0 for r in group) / len(correct)
                if correct
                else None,
            }
        )
    return out


def pooled_drop(
    rows: list[dict[str, Any]], arm: str, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Mean paired drop in faulted silent/planned pooled over open-weight models (bootstrap)."""
    clusters: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for model in sorted({r["model"] for r in rows if is_open(r["model"])}):
        base = _by_key(rows, model, "bare", "faulted")
        other = _by_key(rows, model, arm, "faulted")
        for k in base.keys() & other.keys():
            diff = (base[k]["outcome"] == "silent_wrong") - (other[k]["outcome"] == "silent_wrong")
            clusters[(k[0], k[1])].append(int(diff))
    keys = sorted(clusters)
    if not keys:
        return {"drop": None, "ci": None, "clusters": 0}

    def mean_of(chosen: list[tuple[str, int]]) -> float:
        values = [v for c in chosen for v in clusters[c]]
        return sum(values) / len(values)

    rng = random.Random(seed)
    boots = sorted(mean_of([rng.choice(keys) for _ in keys]) for _ in range(n_boot))
    return {
        "drop": mean_of(keys),
        "ci": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
        "clusters": len(keys),
    }


def pooled_loss(rows: list[dict[str, Any]], arm: str, what: str) -> float | None:
    """Pooled (bare minus arm) over open-weight models for clean pass rate or delivered share."""

    def rate(arm_name: str) -> float | None:
        if what == "clean_pass":
            pool = [
                r
                for r in rows
                if is_open(r["model"]) and r.get("arm", "bare") == arm_name and r["twin"] == "clean"
            ]
            return sum(r["outcome"] == "pass" for r in pool) / len(pool) if pool else None
        pool = [r for r in rows if is_open(r["model"]) and r.get("arm", "bare") == arm_name]
        return sum(r["outcome"] not in NOT_DELIVERED for r in pool) / len(pool) if pool else None

    base, other = rate("bare"), rate(arm)
    return None if base is None or other is None else base - other


def verdict(rows: list[dict[str, Any]], arm: str = "block") -> dict[str, Any]:
    table = [e for e in silent_table(rows) if e["arm"] == arm and is_open(e["model"])]
    winners = [e["model"] for e in table if e["drop"] >= MIN_DROP and e["p_holm"] < ALPHA]
    clean_loss = pooled_loss(rows, arm, "clean_pass")
    delivered_loss = pooled_loss(rows, arm, "delivered")
    effective = (
        len(winners) >= MIN_MODELS
        and clean_loss is not None
        and clean_loss <= MAX_PASS_LOSS
        and delivered_loss is not None
        and delivered_loss <= MAX_PASS_LOSS
    )
    return {
        "arm": arm,
        "models_meeting_drop": winners,
        "clean_pass_loss": clean_loss,
        "delivered_loss": delivered_loss,
        "effective": effective,
    }


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} {100 * k / n:.0f}%" if n else "-"


def _silent_lines(primary: list[dict[str, Any]]) -> list[str]:
    head = (
        f"{'model':<32}{'arm':<14}{'bare':>12}{'arm':>12}{'drop':>8}{'b/c':>8}{'p':>8}{'p_holm':>8}"
    )
    lines = [
        "PRIMARY CELL (bare reporting / implicit / forced): faulted silent_wrong / planned",
        head,
    ]
    for e in silent_table(primary):
        counts = f"{e['bare_only']}/{e['arm_only']}"
        lines.append(
            f"{e['model']:<32}{e['arm']:<14}{_pct(*e['bare']):>12}{_pct(*e['arm_rate']):>12}"
            f"{100 * e['drop']:>7.0f}%{counts:>8}{e['p']:>8.3f}{e['p_holm']:>8.3f}"
        )
    return lines


def _side_lines(cell_rows: list[dict[str, Any]]) -> list[str]:
    head = (
        f"{'model':<32}{'arm':<14}{'clean pass':>14}{'clean blocked':>16}"
        f"{'deliv':>7}{'flag':>6}{'turns':>7}{'$/ok':>8}"
    )
    lines = ["SIDE EFFECTS (both twins)", head]
    for e in side_effects(cell_rows):
        turns = "-" if e["median_turns"] is None else f"{e['median_turns']:.0f}"
        cost = "-" if e["usd_per_correct"] is None else f"{e['usd_per_correct']:.4f}"
        lines.append(
            f"{e['model']:<32}{e['arm']:<14}{_pct(*e['clean_pass']):>14}"
            f"{_pct(*e['clean_blocked']):>16}{100 * e['delivered']:>6.0f}%"
            f"{100 * e['flagged_wrong']:>5.0f}%{turns:>7}{cost:>8}"
        )
    return lines


def render(rows: list[dict[str, Any]]) -> str:
    primary = usable(rows, PRIMARY)
    out = [*_silent_lines(primary), "", *_side_lines(primary), ""]
    out.append("POOLED over open-weight models (cluster bootstrap over family x seed; exploratory)")
    for arm in ARMS[1:]:
        pooled = pooled_drop(primary, arm)
        if pooled["drop"] is not None:
            low, high = pooled["ci"]
            out.append(
                f"  {arm:<14} drop {100 * pooled['drop']:+.1f} pts  "
                f"95% CI [{100 * low:+.1f}, {100 * high:+.1f}]  ({pooled['clusters']} clusters)"
            )
    result = verdict(primary)
    out += ["", f"PREREGISTERED CRITERION (block vs bare): effective = {result['effective']}"]
    out.append(
        f"  models with drop >= {MIN_DROP:.0%} and Holm p < {ALPHA}: "
        f"{result['models_meeting_drop']} (need {MIN_MODELS})"
    )
    out.append(
        f"  pooled clean pass loss {result['clean_pass_loss']}, "
        f"delivered loss {result['delivered_loss']} (each must be <= {MAX_PASS_LOSS})"
    )
    safety = usable(rows, SAFETY)
    if safety:
        out += ["", "SAFETY CELL (full reporting / explicit / free)", *_side_lines(safety)[1:]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biowarden.bench.stage1", description=__doc__)
    parser.add_argument("--out", nargs="+", required=True)
    args = parser.parse_args(argv)
    rows = load_records(args.out)
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
