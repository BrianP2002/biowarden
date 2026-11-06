"""Family admission pilot: does a new family bite, or is it a control?

    python -I -m biowarden.bench.pilot --out runs/pilot-new [--seeds 0-4]

Rule (decided with the owner), applied to ONE model at a time: run a family for a few seeds in the
hardest cell. Among the delivered faulted-twin trials in which the agent took the trapped route
(`took_route`: the family's `exposed` check, or an answer equal to the trap's answer, which cannot
be reached without the route),

  * fewer than 3 such trials: control (routed around), reported with its exposure share;
  * at least 2 silent_wrong, and at least 10% of them: bites (a strong bite);
  * exactly 1 silent_wrong: a weak bite, reported apart and never part of the primary set;
  * none: control (handled).

With at most five seeds "10%" is "one trial", so the registered cut is a count: at least 2 silent
among at least 3 exposed. The clean twin is screened as well: 2 or more silent_wrong clean trials
mark a family on which this model is wrong without any trap (an ill-posed task, or an inverted
contrast such as F34, clean 4 of 5 silent). Models are never pooled.
"""

from __future__ import annotations

import argparse
import collections
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from biowarden.bench.analysis import load_records
from biowarden.bench.score import NOT_DELIVERED, clopper_pearson
from biowarden.bench.tasks.locus import locus_of

BITE_THRESHOLD = 0.10  # share of exposed trials; with 5 seeds the count below is the binding cut
BITE_MIN_SILENT = 2  # silent_wrong among the exposed trials for a strong bite
MIN_EXPOSED = 3  # fewer exposed trials than this (of 5 seeds) cannot show a bite
CLEAN_SCREEN = 2  # silent_wrong clean trials that flag a family for this model
PILOT_SEEDS = (0, 1, 2, 3, 4)
HARD_CELL = ("bare", "implicit", "forced")
LABEL_BITES, LABEL_WEAK = "bites", "weak bite"
LABEL_HANDLED, LABEL_ROUTED = "control (handled)", "control (routed around)"
UNCOUNTED = ("infra_error", "skipped_budget")


def _number(family: str) -> int:
    return int(family[1:])


def took_route(record: dict[str, Any]) -> bool:
    """A faulted trial took the trapped route: the family's `exposed` check says so, or the answer
    is the trap's answer (the `exposed` heuristics miss routes, 5 of 32 such pilot trials)."""
    return bool(record.get("exposed")) or record.get("signature") is True


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        seeds += range(int(low), int(high or low) + 1)
    return tuple(seeds)


def label_of(exposed: int, silent: int) -> str:
    if exposed < MIN_EXPOSED:
        return LABEL_ROUTED
    if silent >= BITE_MIN_SILENT and silent / exposed >= BITE_THRESHOLD:
        return LABEL_BITES
    if silent >= 1:
        return LABEL_WEAK
    return LABEL_HANDLED


def classify(
    rows: Iterable[dict[str, Any]], seeds: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """One entry per (model, family) for the bare arm of the hard cell; `seeds` limits the rows."""
    wanted = None if seeds is None else set(seeds)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cell = (r.get("contract"), r.get("tier"), r.get("route"))
        if (
            cell == HARD_CELL
            and r.get("arm", "bare") == "bare"
            and r["outcome"] not in UNCOUNTED
            and (wanted is None or r.get("seed") in wanted)
        ):
            groups[(r.get("model", ""), r["family"])].append(r)
    out = []
    for (model, family), group in sorted(
        groups.items(), key=lambda kv: (kv[0][0], _number(kv[0][1]))
    ):
        faulted = [r for r in group if r["twin"] == "faulted" and r["outcome"] not in NOT_DELIVERED]
        clean_all = [r for r in group if r["twin"] == "clean"]
        clean = [r for r in clean_all if r["outcome"] not in NOT_DELIVERED]
        exposed = [r for r in faulted if took_route(r)]
        silent = sum(r["outcome"] == "silent_wrong" for r in exposed)
        clean_silent = sum(r["outcome"] == "silent_wrong" for r in clean)
        out.append(
            {
                "model": model,
                "family": family,
                "locus": locus_of(family),
                "trials": len(group),
                "undelivered": len(group) - len(faulted) - len(clean),
                "faulted_delivered": len(faulted),
                "exposed": len(exposed),
                "exposed_by_signature": sum(1 for r in exposed if not r.get("exposed")),
                "silent_given_exposed": (silent, len(exposed)),
                "silent_unexposed": sum(
                    r["outcome"] == "silent_wrong" for r in faulted if not took_route(r)
                ),
                "ci": clopper_pearson(silent, len(exposed)) if exposed else None,
                "clean_silent": (clean_silent, len(clean)),
                "clean_pass": (sum(r["outcome"] == "pass" for r in clean_all), len(clean_all)),
                "clean_flag": clean_silent >= CLEAN_SCREEN,
                "label": label_of(len(exposed), silent),
            }
        )
    return out


def render(entries: list[dict[str, Any]]) -> str:
    head = (
        f"{'family':<7}{'locus':<15}{'undeliv':>8}{'exposed':>9}{'silent|exposed':>16}"
        f"{'clean silent':>14}  label"
    )
    lines: list[str] = []
    models = list(dict.fromkeys(e.get("model", "") for e in entries))
    for model in models:
        if model:
            lines.append(f"model {model}")
        lines.append(head)
        for e in (e for e in entries if e.get("model", "") == model):
            k, n = e["silent_given_exposed"]
            cs, cn = e["clean_silent"]
            shown = f"{k}/{n}" + (
                f" [{100 * e['ci'][0]:.0f}-{100 * e['ci'][1]:.0f}]" if e["ci"] else ""
            )
            flag = " (clean twin wrong)" if e.get("clean_flag") else ""
            lines.append(
                f"{e['family']:<7}{e['locus']:<15}{e['undelivered']:>8}"
                f"{e['exposed']:>5}/{e['faulted_delivered']:<3}"
                f"{shown:>16}{f'{cs}/{cn}':>14}  {e['label']}{flag}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biowarden.bench.pilot", description=__doc__)
    parser.add_argument("--out", nargs="+", required=True)
    parser.add_argument(
        "--seeds", default=None, help="only these seeds (e.g. 0-4; the stage-2 bite set uses 0-4)"
    )
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds) if args.seeds else None
    print(render(classify(load_records(args.out), seeds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
