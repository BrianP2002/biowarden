"""Stage 2 analysis (docs/preregistration/stage2-draft.md): the unit of inference is the family.

    python -I -m science_harness.bench.stage2 sets --pilot runs/pilot2/* --replay replay.json \\
        --out docs/preregistration/stage2-sets.json
    python -I -m science_harness.bench.stage2 scan --sets docs/preregistration/stage2-sets.json \\
        --runs runs/* --own runs/stage2/*
    python -I -m science_harness.bench.stage2 health --out runs/stage2/* \\
        --sets docs/preregistration/stage2-sets.json
    python -I -m science_harness.bench.stage2 analyze --out runs/stage2/* \\
        --sets docs/preregistration/stage2-sets.json

Design. For every model each (family, seed, twin) runs in the primary cell under `bare` and the
layer arm `block-nonote` (and `warn-nonote` on the bite families). A family's difference d_f is the
mean over its valid pairs of metric(bare) - metric(layer); the estimate is the unweighted mean of
d_f over the families of a set. The test statistic is S = sum of d_f; under the sharp null (the
layer changes no trial) every d_f is symmetric about zero, so its sign is exchangeable and
p = P(S* >= S_obs) over the 2^n sign patterns. The d_f are multiples of 1/60 (at most five pairs per
family), so the count is exact: integer convolution of the numerators, no Monte Carlo and no
dependence on record order. (Worked example: d = [1/5, 2/5, -1/5] gives S_obs = 2/5, the sign
patterns give 4/5, 2/5, 2/5, 0, 0, -2/5, -2/5, -4/5 and p = 3/8.) Only if the exact count would
be too large does a Monte Carlo count with a fixed generator take over. The interval is a family
bootstrap. The standard library only: the numbers do not depend on a numpy version.

The registered decision is per model; the layer is called effective when at least two of the three
models pass all of: silent drop on P x faulted >= 10 points, pass gain on P x faulted >= 10 points
(both with the same sign-flip test, per-model p = the larger of the two, Holm over the three fixed
model names), delivered-share loss on P x faulted <= 5 points, clean-pass loss <= 5 points on P and
on every gated family. A layer that makes the agent stop answering therefore cannot win: the silent
drop must show up as passes, not as undelivered answers. A model without data has p = 1 (the Holm
family stays three) and a missing or NaN number fails its condition. The analysis refuses to run
unless every planned trial is on disk, and says so (`--allow-missing`).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from science_harness.bench.analysis import (
    SKIPPED,
    load_records,
    mcnemar_exact,
    undelivered_kind,
)
from science_harness.bench.pilot import (
    HARD_CELL,
    LABEL_BITES,
    LABEL_WEAK,
    PILOT_SEEDS,
    classify,
    took_route,
)
from science_harness.bench.score import (
    DOUBT,
    NOT_DELIVERED,
    SCORER_VERSION,
    clopper_pearson,
    read_result,
    score_result,
)

# --- the design, frozen with this file ---------------------------------------------------------

MODELS = ("claude-haiku-5-5", "deepseek/deepseek-v4.1-flash", "openai/gpt-oss-120b")
PIN_TAGS = {"deepseek/deepseek-v4.1-flash": "wafer/us", "openai/gpt-oss-120b": "near-ai/fp4"}
PIN_NAMES = {
    "deepseek/deepseek-v4.1-flash": frozenset({"Wafer"}),
    "openai/gpt-oss-120b": frozenset({"Near AI"}),
}


def _squash(name: str) -> str:
    """A provider's display name without case or punctuation, as the runner's pin check reads it."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


PRIMARY_CELL = HARD_CELL  # reporting bare, tier implicit, route forced
DEFAULT_CELL = ("full", "explicit", "free")
BASE_ARM, PRIMARY_ARM, WARN_ARM = "bare", "block-nonote", "warn-nonote"
ARMS = (BASE_ARM, WARN_ARM, PRIMARY_ARM)
LAYER_ARMS = (WARN_ARM, PRIMARY_ARM)
CAMPAIGN = (  # (arm, cell) of every job; warn-nonote runs on the bite families only
    (BASE_ARM, PRIMARY_CELL),
    (PRIMARY_ARM, PRIMARY_CELL),
    (WARN_ARM, PRIMARY_CELL),
    (BASE_ARM, DEFAULT_CELL),
    (PRIMARY_ARM, DEFAULT_CELL),
)

# Seeds. The 17 families an agent has seen on seeds 1000-1009 (stage 1 and the stage-1b negative
# control) move to a block no agent has run; the other 88 stay on 1000-1004.
SEEN_FAMILIES = ("F1", "F7", "F10", "F13", "F15", *(f"F{n}" for n in range(17, 29)))
STAGE1_FAMILIES = ("F1", "F7", "F10", "F13", "F15")  # the replication row
FRESH_SEEDS = (1000, 1001, 1002, 1003, 1004)
SEEN_SEEDS = (1020, 1021, 1022, 1023, 1024)
DEFAULT_SEED_COUNT = 3  # the default cell takes the first three seeds of a family's block
DEFAULT_ONLY = ("F4",)  # must-detect: it needs the warning channel, so only the default cell
DEFAULT_EXCLUDED = ("F34",)  # the implicit tier does not say the library; no redesign (owner)
SETS_FORMAT = 2

# The pre-registered decision rule.
MIN_DROP = 0.10  # silent drop on P x faulted, as a fraction (10 percentage points)
MIN_GAIN = 0.10  # pass gain on P x faulted
MAX_LOSS = 0.05  # delivered-share loss on P x faulted; clean-pass loss on P and on all gated
ALPHA = 0.05  # Holm-adjusted, one-sided
MODELS_NEEDED = 2  # of the three
MISSING_GAP = 0.02  # between-arm difference in lost trials that makes the worst case binding
MIN_PRIMARY = 20  # stop rule: fewer families in P, or ...
MIN_PRIMARY_MODEL = 15  # ... fewer than MODELS_NEEDED models with this many bite families of
# their own, and the confirmatory test is declared underpowered and reported as descriptive
CLEAN_PASS_FLOOR = 0.60  # exclusion: clean-twin bare pass below this for two or more models
CLEAN_EXCLUDE_MODELS = 2
THIN_PAIRS = 3  # a family with fewer valid pairs is flagged

FLIPS = 200_000  # Monte Carlo flips, used only when the exact count is too large
BOOTSTRAPS = 10_000
SEED = 20261031
EXACT_WORK = 30_000_000  # numerators x span above which the sign-flip count is Monte Carlo
FRACTION_LIMIT = (
    100_000  # values given as floats are read as fractions with at most this denominator
)
EPS = 1e-12
MAX_ATTEMPTS = 3  # the runner's: an infra_error with this many tries is final (a test pins it)

Row = Mapping[str, Any]
Cell = tuple[str, str, str]
Metric = Callable[[Row], bool]
Key = tuple[str, int, str]  # family, seed, twin


def silent(r: Row) -> bool:
    return r["outcome"] == "silent_wrong"


def passed(r: Row) -> bool:
    return r["outcome"] == "pass"


def notpassed(r: Row) -> bool:
    return r["outcome"] != "pass"


def delivered(r: Row) -> bool:
    return r["outcome"] not in NOT_DELIVERED


METRICS: dict[str, Metric] = {
    "silent": silent,
    "notpass": notpassed,  # its drop is the pass GAIN: notpass(bare) - notpass(layer)
    "pass": passed,
    "delivered": delivered,
}
# What a lost trial counts as in the worst case, for the BARE trial (the layer trial gets the
# opposite): a lost bare trial is not silent, a lost layer trial is silent; for pass and delivered
# (losses) a lost bare trial passed and a lost layer trial did not.
WORST_BARE = {"silent": 0, "notpass": 0, "pass": 1, "delivered": 1}

PACKS = (
    ("original", 1, 16),
    ("wrapper", 17, 20),
    ("statement", 21, 24),
    ("documentation", 25, 28),
    ("alignment", 29, 39),
    ("quantification", 40, 52),
    ("variants", 53, 64),
    ("intervals", 65, 76),
    ("sequence", 77, 88),
    ("structure", 89, 100),
    ("single-cell", 101, 112),
)


def fam_no(family: str) -> int:
    return int(family[1:])


def pack_of(family: str) -> str:
    number = fam_no(family)
    for name, low, high in PACKS:
        if low <= number <= high:
            return name
    raise ValueError(f"no pack for {family}")


def cell_of(record: Row) -> Cell:
    return (record.get("contract", ""), record.get("tier", ""), record.get("route", ""))


def is_openrouter(model: str) -> bool:
    return "/" in model


def sorted_families(families: Iterable[str]) -> list[str]:
    return sorted(set(families), key=fam_no)


# --- the seed registry and the strata, parsed from the frozen sets file ------------------------


def registry_seeds(family: str) -> tuple[int, ...]:
    """The five primary-cell seeds of a family: the unseen block for the 17 seen families."""
    return SEEN_SEEDS if family in SEEN_FAMILIES else FRESH_SEEDS


@dataclass(frozen=True)
class Registry:
    """What the sets file freezes: the strata, the seeds and the rule-based lists.

    P (primary) = bites for at least one model and an own rule would block; Q1 = bites, no own rule
    would block; Q2 = no bite, an own rule would block; Q3 = neither. The complement of P is
    Q1 + Q2 + Q3. `bite` (P + Q1) is where warn-nonote runs.
    """

    primary: tuple[str, ...]
    q1: tuple[str, ...] = ()
    q2: tuple[str, ...] = ()
    q3: tuple[str, ...] = ()
    seeds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    default_seeds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    weak: tuple[str, ...] = ()
    weak_reachable: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    default_only: tuple[str, ...] = DEFAULT_ONLY
    per_model: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    underpowered: bool = False
    flags: tuple[str, ...] = ()  # "lenient" or "incomplete": the sets file is not confirmatory

    @classmethod
    def make(
        cls,
        primary: Iterable[str],
        q1: Iterable[str] = (),
        q2: Iterable[str] = (),
        q3: Iterable[str] = (),
        **more: Any,
    ) -> Registry:
        """A registry with the registered seeds for these families (tests and rehearsals)."""
        strata = [tuple(sorted_families(s)) for s in (primary, q1, q2, q3)]
        everyone = [f for s in strata for f in s]
        defaults = tuple(more.pop("default_only", DEFAULT_ONLY))
        seeds = {f: registry_seeds(f) for f in everyone}
        default_seeds = {f: seeds[f][:DEFAULT_SEED_COUNT] for f in everyone}
        for f in defaults:
            default_seeds[f] = registry_seeds(f)[:DEFAULT_SEED_COUNT]
        reg = cls(
            *strata,
            seeds=more.pop("seeds", seeds),
            default_seeds=more.pop("default_seeds", default_seeds),
            default_only=defaults,
            **more,
        )
        reg.validate()
        return reg

    @classmethod
    def from_sets(cls, sets: Mapping[str, Any]) -> Registry:
        if sets.get("format") != SETS_FORMAT or "seeds" not in sets:
            raise ValueError(
                f"the sets file has no seed registry (format {SETS_FORMAT}); rebuild it with "
                "`stage2 sets`"
            )
        power = sets.get("power", {})
        reg = cls(
            primary=tuple(sets["primary"]),
            q1=tuple(sets["q1"]),
            q2=tuple(sets["q2"]),
            q3=tuple(sets["q3"]),
            seeds={f: tuple(s) for f, s in sets["seeds"].items()},
            default_seeds={f: tuple(s) for f, s in sets["default_seeds"].items()},
            weak=tuple(sets.get("weak", ())),
            weak_reachable=tuple(sets.get("weak_reachable", ())),
            excluded=tuple(sets.get("excluded", ())),
            default_only=tuple(sets.get("default_only", DEFAULT_ONLY)),
            per_model={
                m: {k: tuple(v) for k, v in d.items()} for m, d in sets.get("per_model", {}).items()
            },
            underpowered=bool(power.get("underpowered")),
            flags=tuple(sets.get("flags", ())),
        )
        reg.validate()
        return reg

    def validate(self) -> None:
        strata = {"P": self.primary, "Q1": self.q1, "Q2": self.q2, "Q3": self.q3}
        seen: dict[str, str] = {}
        for name, members in strata.items():
            for f in members:
                if f in seen:
                    raise ValueError(f"{f} is in both {seen[f]} and {name}")
                seen[f] = name
        for f in self.excluded:
            if f in seen:
                raise ValueError(f"{f} is excluded and also in {seen[f]}")
        for f in (*self.gated, *self.default_only):
            if f in self.excluded:
                continue
            want = registry_seeds(f)
            if f in self.gated and tuple(self.seeds.get(f, ())) != want:
                raise ValueError(
                    f"{f} must run on seeds {want[0]}-{want[-1]} (the registry); "
                    f"the sets file has {tuple(self.seeds.get(f, ()))}"
                )
            if tuple(self.default_seeds.get(f, ())) != want[:DEFAULT_SEED_COUNT]:
                raise ValueError(f"{f}: default-cell seeds must be {want[:DEFAULT_SEED_COUNT]}")

    # derived lists
    @property
    def complement(self) -> tuple[str, ...]:
        return tuple(sorted_families((*self.q1, *self.q2, *self.q3)))

    @property
    def bite_unreachable(self) -> tuple[str, ...]:
        return self.q1

    @property
    def q_other(self) -> tuple[str, ...]:
        return tuple(sorted_families((*self.q2, *self.q3)))

    @property
    def bite(self) -> tuple[str, ...]:
        return tuple(sorted_families((*self.primary, *self.q1)))

    @property
    def gated(self) -> tuple[str, ...]:
        """Every family of the primary cell that is not excluded: P and its complement."""
        return tuple(sorted_families((*self.primary, *self.complement)))

    @property
    def default_gated(self) -> tuple[str, ...]:
        return tuple(sorted_families((*self.gated, *self.default_only)))

    def seeds_for(self, cell: Cell) -> Mapping[str, tuple[int, ...]]:
        return self.default_seeds if cell == DEFAULT_CELL else self.seeds

    def families_for(self, arm: str, cell: Cell) -> tuple[str, ...]:
        """The families a job of this arm runs in this cell."""
        if cell == PRIMARY_CELL:
            return self.bite if arm == WARN_ARM else self.gated
        if cell == DEFAULT_CELL:
            return () if arm == WARN_ARM else self.default_gated
        return ()

    def pair_keys(self, families: Iterable[str], cell: Cell, twins: Sequence[str]) -> list[Key]:
        seeds = self.seeds_for(cell)
        return [
            (f, s, t)
            for f in sorted_families(families)
            if f in seeds
            for s in seeds[f]
            for t in twins
        ]


# --- loading and provenance ---------------------------------------------------------------------


def row_key(r: Row) -> tuple[Any, ...]:
    return (
        r["model"],
        r.get("arm", BASE_ARM),
        cell_of(r),
        fam_no(r["family"]),
        r["seed"],
        r["twin"],
        r.get("rep", 0),
        r.get("trial", ""),
    )


def load_rows(dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Every score.json of the run directories, sorted by (model, arm, cell, family, seed, twin,
    rep). Raises on a truncated file, a trial that appears twice (inside or across directories)
    and a scorer other than SCORER_VERSION: a quiet skip would shrink the analysis."""
    unreadable: list[Path] = []
    rows = load_records([str(d) for d in dirs], unreadable)
    if unreadable:
        raise ValueError(f"{len(unreadable)} unreadable score.json, first {unreadable[0]}")
    seen: set[str] = set()
    for r in rows:
        if r.get("scorer") != SCORER_VERSION:
            raise ValueError(
                f"trial {r.get('trial')}: scorer {r.get('scorer')!r}, not {SCORER_VERSION}"
            )
        if r.get("trial") in seen:
            raise ValueError(f"two records for trial {r['trial']}")
        seen.add(r["trial"])
    try:
        rows.sort(key=row_key)
    except KeyError as exc:
        raise ValueError(f"a record lacks the field {exc}") from exc
    return rows


def load_plan(dirs: Sequence[str | Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """trial id -> spec from the plan.json of every directory, and the directories without one."""
    ids: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for d in dirs:
        try:
            plan = json.loads((Path(d) / "plan.json").read_text())
            specs = plan["specs"]
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(str(d))
            continue
        ids.update(specs)
    return ids, missing


def load_unrun(dirs: Sequence[str | Path]) -> dict[str, str]:
    """trial id -> reason, from unrun.jsonl (trials the budget skipped; a later run may have done
    them since, so it is a hint for the absent list, never a record)."""
    unrun: dict[str, str] = {}
    for d in dirs:
        try:
            lines = (Path(d) / "unrun.jsonl").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
                unrun[item["trial"]] = str(item.get("reason", ""))
            except (ValueError, KeyError, TypeError):
                continue
    return unrun


def registered(r: Row, reg: Registry) -> bool:
    """The record belongs to the registered design: the minimal harness, a registered arm, the
    primary or the default cell, a seed of the family's registry block, and repetition 0 (or the
    bare A/A replicate on P, repetition 1)."""
    cell, arm, rep = cell_of(r), r.get("arm", BASE_ARM), r.get("rep", 0)
    if r.get("harness", "minimal") != "minimal" or arm not in ARMS:
        return False
    if cell not in (PRIMARY_CELL, DEFAULT_CELL) or (arm == WARN_ARM and cell != PRIMARY_CELL):
        return False
    if r["seed"] not in reg.seeds_for(cell).get(r["family"], ()):
        return False
    replicate = rep == 1 and arm == BASE_ARM and cell == PRIMARY_CELL and r["family"] in reg.primary
    return rep == 0 or replicate


def usable(r: Row, reg: Registry | None = None) -> bool:
    """A trial the contrasts count: not lost to infrastructure or the budget, minimal harness,
    first repetition, and, given the registry, a registered arm, cell, family and seed."""
    if r["outcome"] in SKIPPED or r.get("harness", "minimal") != "minimal" or r.get("rep", 0) != 0:
        return False
    return reg is None or registered(r, reg)


def layer_flag(r: Row) -> bool:
    """A layer trial whose layer was not intact: it failed open (errors), crashed, or the agent
    touched the layer's files (the -nonote arms are discoverable)."""
    layer = r.get("layer")
    if not isinstance(layer, Mapping):
        return False
    return bool(
        layer.get("errors")
        or layer.get("ledger_errors")
        or layer.get("crashed")
        or layer.get("touched_layer")
    )


def provenance_problems(
    rows: Sequence[Row],
    reg: Registry | None = None,
    *,
    code_digest: str | None = None,
    pilot: bool = False,
) -> list[str]:
    """Reasons the records cannot be one campaign; [] = ok. Always fatal (no override).

    A model, arm, harness, cell, family or seed outside the registry; a second repetition other
    than the bare A/A replicate; an OpenRouter record that was not served by the pinned endpoint;
    a layer-arm record without a layer summary; records made by different code or configurations.
    """
    found: dict[str, list[str]] = collections.defaultdict(list)

    def note(kind: str, r: Row) -> None:
        found[kind].append(str(r.get("trial")))

    arms = (BASE_ARM,) if pilot else ARMS
    for r in rows:
        model, arm = r.get("model"), r.get("arm", BASE_ARM)
        counted = r["outcome"] not in SKIPPED
        cell = cell_of(r)
        if model not in MODELS:
            note(f"model {model!r} is not one of the three", r)
        if arm not in arms:
            note(f"arm {arm!r} is not registered", r)
        if r.get("harness", "minimal") != "minimal":
            note("harness is not minimal", r)
        if r.get("scorer") != SCORER_VERSION:
            note(f"scorer is not {SCORER_VERSION}", r)
        if pilot:
            if cell != PRIMARY_CELL:
                note("cell is not the primary cell", r)
        elif cell not in (PRIMARY_CELL, DEFAULT_CELL):
            note("cell is neither the primary nor the default cell", r)
        elif arm == WARN_ARM and cell != PRIMARY_CELL:
            note("warn-nonote outside the primary cell", r)
        elif reg is not None:
            seeds = reg.seeds_for(cell).get(r["family"])
            if seeds is None or r["seed"] not in seeds:
                note("(family, seed) is not in the seed registry", r)
            rep = r.get("rep", 0)
            if rep not in (0, 1) or (
                rep == 1
                and not (arm == BASE_ARM and cell == PRIMARY_CELL and r["family"] in reg.primary)
            ):
                note("repetition other than the bare A/A replicate on P", r)
        if counted and model in PIN_TAGS:
            if r.get("provider_pin") != [PIN_TAGS[model]]:
                note("OpenRouter record without the registered provider pin", r)
            names = {_squash(n) for n in r.get("providers") or [] if isinstance(n, str)}
            if names and not names <= {_squash(n) for n in PIN_NAMES[model]}:
                note("served by a provider other than the pinned one", r)
            if r.get("pin_match") is False:
                note("pin_match is false", r)
        if counted and arm in LAYER_ARMS and r.get("layer") is None:
            note("layer arm without a layer summary", r)
    problems = [
        f"{len(ids)} record(s): {kind} (e.g. {', '.join(ids[:2])})" for kind, ids in found.items()
    ]
    for key in ("code_digest", "run_config_sha", "tool_manifest_sha"):
        values = {r.get(key) for r in rows}
        if None in values:
            problems.append(f"records without {key}: not made by the frozen runner")
        elif len(values) > 1:
            problems.append(f"{len(values)} different {key} values: the jobs ran different code")
    if code_digest and rows and {r.get("code_digest") for r in rows} != {code_digest}:
        problems.append("code_digest differs from the frozen digest given with --code-digest")
    return problems


# --- planned trials, completeness, missingness ---------------------------------------------------


def runner_plan() -> Callable[..., Iterable[Any]]:
    """The runner's own plan(), imported late: the analysis needs the SDKs only for this."""
    from science_harness.bench.runner.cli import plan

    return plan


def _registry(sets: Registry | Mapping[str, Any]) -> Registry:
    return sets if isinstance(sets, Registry) else Registry.from_sets(sets)


def planned(
    sets: Registry | Mapping[str, Any],
    model: str,
    arm: str,
    cell: Cell,
    plan_fn: Callable[..., Iterable[Any]] | None = None,
) -> list[str]:
    """The trial ids a job of (model, arm, cell) plans, from the runner's own plan().

    The runner takes one seed list per call, the registry has two seed blocks, so the families
    are grouped by block. Sorted; the same ids are in the job's plan.json.
    """
    reg = _registry(sets)
    families = reg.families_for(arm, cell)
    if not families:
        return []
    plan_fn = plan_fn or runner_plan()
    seeds = reg.seeds_for(cell)
    blocks: dict[tuple[int, ...], list[str]] = collections.defaultdict(list)
    for f in families:
        blocks[tuple(seeds[f])].append(f)
    contract, tier, route = cell
    ids: list[str] = []
    for block, members in sorted(blocks.items()):
        specs = plan_fn([model], members, list(block), 1, tier, route, contract, "minimal", arm)
        ids += [spec.trial_id for spec in specs]
    return sorted(ids)


def planned_ids(
    reg: Registry,
    *,
    model: str,
    arm: str,
    cell: Cell,
    plan_fn: Callable[..., Iterable[Any]] | None = None,
) -> set[str]:
    return set(planned(reg, model, arm, cell, plan_fn))


@dataclass
class Completeness:
    """Planned versus recorded, over the whole campaign."""

    planned: int = 0
    recorded: int = 0
    counted: int = 0
    infra_final: list[str] = field(default_factory=list)  # listed exclusions: tried 3 times
    infra_open: list[str] = field(default_factory=list)  # attempts left: the job is not finished
    absent: list[str] = field(default_factory=list)  # planned, no record
    unplanned: list[str] = field(default_factory=list)  # recorded, not planned
    unrun: list[str] = field(default_factory=list)  # absent ids the budget skipped (unrun.jsonl)
    never_planned: list[str] = field(default_factory=list)  # in no job's plan.json

    def problems(self) -> list[str]:
        out = []
        if self.absent:
            out.append(
                f"{len(self.absent)} planned trial(s) have no record (e.g. {self.absent[0]})"
            )
        if self.infra_open:
            out.append(
                f"{len(self.infra_open)} infra_error(s) still have attempts left (e.g. "
                f"{self.infra_open[0]}): run the job again"
            )
        if self.never_planned:
            out.append(f"{len(self.never_planned)} registered trial(s) are in no plan.json")
        return out


def campaign_ids(
    reg: Registry,
    models: Sequence[str] = MODELS,
    plan_fn: Callable[..., Iterable[Any]] | None = None,
) -> dict[tuple[str, str, Cell], list[str]]:
    plan_fn = plan_fn or runner_plan()
    return {
        (model, arm, cell): planned(reg, model, arm, cell, plan_fn)
        for model in models
        for arm, cell in CAMPAIGN
    }


def check_complete(
    rows: Sequence[Row],
    reg: Registry,
    models: Sequence[str] = MODELS,
    plan_fn: Callable[..., Iterable[Any]] | None = None,
    unrun: Mapping[str, str] | None = None,
    plan_ids: Collection[str] | None = None,
) -> Completeness:
    """planned = recorded (+ listed exclusions) or the analysis must not run."""
    expected = {i for ids in campaign_ids(reg, models, plan_fn).values() for i in ids}
    by_id = {r["trial"]: r for r in rows}
    comp = Completeness(planned=len(expected))
    for trial in sorted(expected):
        r = by_id.get(trial)
        if r is None:
            comp.absent.append(trial)
            continue
        comp.recorded += 1
        if r["outcome"] not in SKIPPED:
            comp.counted += 1
        elif r["outcome"] == "infra_error":
            final = int(r.get("attempt") or 1) >= MAX_ATTEMPTS
            (comp.infra_final if final else comp.infra_open).append(trial)
    comp.unplanned = sorted(
        r["trial"] for r in rows if r.get("rep", 0) == 0 and r["trial"] not in expected
    )
    skipped = unrun or {}
    comp.unrun = [t for t in comp.absent if t in skipped]
    if plan_ids is not None:
        comp.never_planned = sorted(expected - set(plan_ids))
    return comp


def missingness(
    rows: Sequence[Row],
    reg: Registry,
    *,
    cell: Cell = PRIMARY_CELL,
    arms: Sequence[str] = (BASE_ARM, PRIMARY_ARM),
    models: Sequence[str] = MODELS,
    plan_fn: Callable[..., Iterable[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per (model, arm): planned, present, lost (absent or infra_error), refused and the other
    undelivered counts; `gap` is the model's largest between-arm difference of the lost share and
    `binding` says it exceeds two points (the worst case then decides the verdict)."""
    plan_fn = plan_fn or runner_plan()
    by_id = {r["trial"]: r for r in rows}
    table: list[dict[str, Any]] = []
    for model in models:
        mine = []
        for arm in arms:
            ids = planned(reg, model, arm, cell, plan_fn)
            present = [by_id[i] for i in ids if i in by_id]
            outcomes = collections.Counter(r["outcome"] for r in present)
            lost = len(ids) - len(present) + outcomes["infra_error"] + outcomes["skipped_budget"]
            mine.append(
                {
                    "model": model,
                    "arm": arm,
                    "planned": len(ids),
                    "present": len(present),
                    "infra_error": outcomes["infra_error"],
                    "absent": len(ids) - len(present),
                    "lost": lost,
                    "lost_share": lost / len(ids) if ids else float("nan"),
                    "refused": outcomes["refused"],
                    "protocol_fail": outcomes["protocol_fail"],
                    "format_error": outcomes["format_error"],
                }
            )
        shares = [m["lost_share"] for m in mine if m["lost_share"] == m["lost_share"]]
        gap = max(shares) - min(shares) if shares else float("nan")
        for m in mine:
            m["gap"] = gap
            m["binding"] = gap > MISSING_GAP + EPS
        table += mine
    return table


# --- pairs and family differences ---------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One (family, seed, twin): the baseline trial and the arm's trial; None = no counted trial."""

    family: str
    seed: int
    twin: str
    base: Row | None
    arm: Row | None


def pair_trials(
    rows: Iterable[Row],
    *,
    model: str,
    arm: str,
    cell: Cell,
    twins: Sequence[str],
    families: Iterable[str],
    baseline: str = BASE_ARM,
    base_rep: int = 0,
    arm_rep: int = 0,
    expected: Iterable[Key] | None = None,
) -> list[Pair]:
    """The pairs of (baseline, arm) trials. A trial lost to an infrastructure error or the
    budget is None, and with `expected` so is a trial that was never recorded. Raises on two
    records for the same trial. (baseline, base_rep) against (arm, arm_rep) with arm == baseline
    and arm_rep == 1 pairs the bare A/A replicate with the first bare run."""
    if (baseline, base_rep) == (arm, arm_rep):
        raise ValueError("the baseline and the arm are the same trials")
    wanted = set(families)
    slots: dict[Key, dict[tuple[str, int], Row]] = collections.defaultdict(dict)
    sides = {(baseline, base_rep), (arm, arm_rep)}
    for r in rows:
        if (
            r["model"] != model
            or r["family"] not in wanted
            or r["twin"] not in twins
            or cell_of(r) != cell
            or r.get("harness", "minimal") != "minimal"
            or r["outcome"] in SKIPPED
        ):
            continue
        side = (r.get("arm", BASE_ARM), r.get("rep", 0))
        if side not in sides:
            continue
        key = (r["family"], r["seed"], r["twin"])
        if side in slots[key]:
            raise ValueError(f"two records for trial {r['trial']}")
        slots[key][side] = r
    keys = set(slots)
    if expected is not None:
        keys |= {k for k in expected if k[0] in wanted and k[2] in twins}
    pairs = []
    for key in sorted(keys, key=lambda k: (fam_no(k[0]), k[1], k[2])):
        slot = slots.get(key, {})
        pairs.append(Pair(*key, slot.get((baseline, base_rep)), slot.get((arm, arm_rep))))
    return pairs


def _resolve(metric: Metric | str) -> tuple[str | None, Metric]:
    if isinstance(metric, str):
        return metric, METRICS[metric]
    for name, fn in METRICS.items():
        if fn is metric:
            return name, metric
    return None, metric


def _fractions(
    pairs: Iterable[Pair], metric: Metric | str, worst: bool = False
) -> dict[str, tuple[Fraction, int]]:
    """family -> (exact mean over its pairs of metric(base) - metric(arm), number of pairs).

    A pair with a lost trial is dropped; with worst=True it is kept and the lost trial counts as
    the outcome that is worst for the layer (WORST_BARE).
    """
    name, fn = _resolve(metric)
    if worst and name is None:
        raise ValueError("the worst case needs a registered metric name")
    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for p in pairs:
        if p.base is None or p.arm is None:
            if not worst:
                continue
            bare_lost = WORST_BARE[str(name)]
            b = bare_lost if p.base is None else int(fn(p.base))
            a = 1 - bare_lost if p.arm is None else int(fn(p.arm))
        else:
            b, a = int(fn(p.base)), int(fn(p.arm))
        totals[p.family][0] += b - a
        totals[p.family][1] += 1
    return {
        f: (Fraction(totals[f][0], totals[f][1]), totals[f][1]) for f in sorted(totals, key=fam_no)
    }


def family_differences(
    pairs: Iterable[Pair], metric: Metric | str, worst: bool = False
) -> dict[str, tuple[float, int]]:
    """family -> (mean over its pairs of metric(baseline) - metric(arm), n pairs); positive means
    the layer lowered the metric. Families without a pair are absent; sorted by family number."""
    return {f: (float(d), n) for f, (d, n) in _fractions(pairs, metric, worst).items()}


def paired_differences(
    rows: Iterable[Row],
    model: str,
    arm: str,
    metric: Metric | str,
    twins: Sequence[str],
    cell: Cell,
    families: Iterable[str],
    baseline: str = BASE_ARM,
) -> dict[str, tuple[float, int]]:
    pairs = pair_trials(
        rows, model=model, arm=arm, cell=cell, twins=twins, families=families, baseline=baseline
    )
    return family_differences(pairs, metric)


# --- inference ----------------------------------------------------------------------------------


def _numerators(values: Iterable[Any]) -> list[int]:
    """The non-zero values as integers over their common denominator (the d_f are multiples of
    1/60, so the numerators are small)."""
    fractions = []
    for v in values:
        if isinstance(v, float) and v != v:
            raise ValueError("NaN in the family differences")
        f = Fraction(v).limit_denominator(FRACTION_LIMIT)
        if f != 0:
            fractions.append(f)
    if not fractions:
        return []
    den = math.lcm(*(f.denominator for f in fractions))
    return [int(f * den) for f in fractions]


def _subset_sums(values: Sequence[int], width: int = 8) -> list[list[int]]:
    """Per chunk of `width` values, the sum of every subset (indexed by bit mask)."""
    tables = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        table = [0] * (1 << len(chunk))
        for mask in range(1, 1 << len(chunk)):
            low = mask & -mask
            table[mask] = table[mask ^ low] + chunk[low.bit_length() - 1]
        tables.append(table)
    return tables


def _hit(plus: int, total: int, two_sided: bool) -> bool:
    """Is the pattern whose chosen (sign +) numerators sum to `plus` at least as extreme as the
    observed one? S* = 2 plus - total and S_obs = total."""
    if two_sided:
        return 2 * plus >= total + abs(total) or 2 * plus <= total - abs(total)
    return plus >= total


def sign_flip_p(
    values: Sequence[Any],
    flips: int = FLIPS,
    seed: int = SEED,
    sided: str = "greater",
    *,
    alternative: str | None = None,
    exact_work: int = EXACT_WORK,
) -> float:
    """p of the sign-flip randomization test on the family differences d_f.

    Statistic S = sum d_f. `greater`: p = P(S* >= S_obs), H1 'the mean difference is above zero';
    `two-sided`: p = P(|S*| >= |S_obs|). S* flips every sign independently. Zero differences never
    change S and are dropped; if none is left p = 1. The count is exact (integer convolution of
    the numerators) unless it would take more than `exact_work` steps, then Monte Carlo with a
    fixed generator and the add-one correction (1 + hits) / (1 + flips). The caller sorts the
    values by family: only the Monte Carlo branch depends on their order.
    """
    mode = alternative or sided
    if mode not in ("greater", "two-sided"):
        raise ValueError(f"sided must be 'greater' or 'two-sided', not {mode!r}")
    two_sided = mode == "two-sided"
    a = _numerators(values)
    if not a:
        return 1.0
    n, total = len(a), sum(a)
    if n * (2 * sum(abs(x) for x in a) + 1) <= exact_work:
        ways: dict[int, int] = {0: 1}
        for x in a:
            nxt = dict(ways)
            for s, c in ways.items():
                nxt[s + x] = nxt.get(s + x, 0) + c
            ways = nxt
        return sum(c for s, c in ways.items() if _hit(s, total, two_sided)) / (1 << n)
    tables = _subset_sums(a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(flips):
        bits = rng.getrandbits(n)
        plus = sum(table[(bits >> (8 * i)) & 255] for i, table in enumerate(tables))
        hits += _hit(plus, total, two_sided)
    return (1 + hits) / (1 + flips)


def bootstrap_ci(
    values: Sequence[float], draws: int = BOOTSTRAPS, seed: int = SEED + 1, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the mean over families resampled with replacement (NaN, NaN if
    there is nothing to resample or draws is 0). Pass the values sorted by family."""
    n = len(values)
    if n == 0 or draws <= 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    low = means[int(draws * (1 - level) / 2)]
    high = means[min(draws - 1, int(draws * (1 + level) / 2))]
    return (low, high)


def bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    draws: int = BOOTSTRAPS,
    seed: int = SEED + 3,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a percentile interval; the two groups of families are resampled
    independently. (NaN, NaN, NaN) when a group is empty."""
    if not a or not b:
        return (float("nan"),) * 3  # type: ignore[return-value]
    estimate = sum(a) / len(a) - sum(b) / len(b)
    if draws <= 0:
        return (estimate, float("nan"), float("nan"))
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    diffs = sorted(
        sum(a[rng.randrange(na)] for _ in range(na)) / na
        - sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        for _ in range(draws)
    )
    return (
        estimate,
        diffs[int(draws * (1 - level) / 2)],
        diffs[min(draws - 1, int(draws * (1 + level) / 2))],
    )


def cluster_bootstrap_ci(
    per_family: Mapping[str, float],
    key: Callable[[str], str],
    draws: int = BOOTSTRAPS,
    seed: int = SEED + 4,
    level: float = 0.95,
) -> tuple[float, float, int]:
    """Percentile interval of the mean over families when whole clusters (packs of families that
    share rules and tools) are resampled with replacement: the inference that does not treat the
    families of a pack as independent. Returns (low, high, number of clusters); NaN under two."""
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for family in sorted(per_family, key=fam_no):
        groups[key(family)].append(per_family[family])
    names = sorted(groups)
    n = len(names)
    if n < 2 or draws <= 0:
        return (float("nan"), float("nan"), n)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        picked = [v for name in (names[rng.randrange(n)] for _ in range(n)) for v in groups[name]]
        means.append(sum(picked) / len(picked))
    means.sort()
    return (
        means[int(draws * (1 - level) / 2)],
        means[min(draws - 1, int(draws * (1 + level) / 2))],
        n,
    )


def holm(pvalues: Mapping[str, float | None], m: int | None = None) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1). A hypothesis without a p-value
    counts as 1.0 and stays in the family, whose size is `m` (default: the number of keys)."""
    p = {k: 1.0 if v is None else v for k, v in pvalues.items()}
    order = sorted(p, key=lambda k: (p[k], k))
    size = max(len(order), m or 0)
    running = 0.0
    adjusted: dict[str, float] = {}
    for i, key in enumerate(order):
        running = max(running, (size - i) * p[key])
        adjusted[key] = min(1.0, running)
    return adjusted


@dataclass
class Contrast:
    """A paired difference d = metric(baseline) - metric(arm) averaged over families (positive =
    the layer lowers the metric; for 'notpass' that is a pass GAIN)."""

    model: str
    arm: str
    baseline: str
    metric: str
    twins: tuple[str, ...]
    cell: Cell
    set_name: str
    expected_families: int  # the families asked for
    families: int  # of them with at least one valid pair
    dropped: tuple[str, ...]  # asked for, no valid pair: never set to 0
    pairs: int
    expected_pairs: int
    estimate: float
    low: float
    high: float
    sided: str | None = None
    p: float | None = None
    p_holm: float | None = None
    per_family: dict[str, float] = field(default_factory=dict)
    pair_weighted: float = float("nan")  # mean over pairs, a sensitivity to the equal weights
    thin: tuple[str, ...] = ()  # families with fewer than THIN_PAIRS valid pairs


def contrast(
    rows: Sequence[Row],
    model: str,
    arm: str,
    metric: Metric | str,
    twins: Sequence[str],
    cell: Cell,
    families: Iterable[str],
    set_name: str = "",
    sided: str | None = None,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
    *,
    baseline: str = BASE_ARM,
    base_rep: int = 0,
    arm_rep: int = 0,
    expected: Iterable[Key] | None = None,
    worst: bool = False,
    alternative: str | None = None,
) -> Contrast:
    """The equal-family-weight mean difference of one metric, with a family-bootstrap interval
    and, when `sided` is given, the sign-flip p."""
    wanted = sorted_families(families)
    keys = None if expected is None else list(expected)
    pairs = pair_trials(
        rows,
        model=model,
        arm=arm,
        cell=cell,
        twins=twins,
        families=wanted,
        baseline=baseline,
        base_rep=base_rep,
        arm_rep=arm_rep,
        expected=keys,
    )
    diffs = _fractions(pairs, metric, worst)
    values = list(diffs.values())
    exact = [d for d, _n in values]
    floats = [float(d) for d in exact]
    n_pairs = sum(n for _d, n in values)
    estimate = float(sum(exact) / len(exact)) if exact else float("nan")
    weighted = float(sum(d * n for d, n in values) / n_pairs) if n_pairs else float("nan")
    low, high = bootstrap_ci(floats, draws)
    mode = alternative or sided
    p = sign_flip_p(exact, flips, SEED, mode) if mode and exact else None
    name, _fn = _resolve(metric)
    return Contrast(
        model=model,
        arm=arm,
        baseline=baseline,
        metric=name or getattr(metric, "__name__", "metric"),
        twins=tuple(twins),
        cell=cell,
        set_name=set_name,
        expected_families=len(wanted),
        families=len(values),
        dropped=tuple(f for f in wanted if f not in diffs),
        pairs=n_pairs,
        expected_pairs=len(keys) if keys is not None else len(pairs),
        estimate=estimate,
        low=low,
        high=high,
        sided=mode,
        p=p,
        per_family=dict(zip(diffs, floats, strict=True)),
        pair_weighted=weighted,
        thin=tuple(f for f, (_d, n) in diffs.items() if n < THIN_PAIRS),
    )


# --- the registered decision --------------------------------------------------------------------


@dataclass
class ModelDecision:
    """One model against the five conditions of the registered rule."""

    model: str
    drop: Contrast  # silent, P x faulted: >= MIN_DROP and p
    gain: Contrast  # pass gain, P x faulted: >= MIN_GAIN and p
    delivered_loss: Contrast  # delivered, P x faulted: <= MAX_LOSS
    clean_loss_primary: Contrast  # pass, clean twin, P: <= MAX_LOSS
    clean_loss_gated: Contrast  # pass, clean twin, every gated family: <= MAX_LOSS
    p: float | None  # the larger of the two p-values (intersection-union); None = no data
    p_holm: float  # Holm over the three fixed model names; 1.0 without data
    checks: dict[str, bool]
    passes: bool


@dataclass
class Primary:
    arm: str
    set_name: str
    n_primary: int
    n_safety: int
    models: list[ModelDecision]
    models_passing: tuple[str, ...]
    effective: bool
    claim: bool  # the registered success claim belongs to block-nonote only
    worst_case: bool  # lost trials were imputed adversarially
    mean_clean_pass_loss: float = float("nan")
    mean_delivered_loss: float = float("nan")


def passes(c: Contrast, minimum: float = MIN_DROP) -> bool:
    """One contrast of the rule: the estimate reaches `minimum` and its Holm p is below ALPHA."""
    return c.estimate >= minimum - EPS and c.p_holm is not None and c.p_holm < ALPHA


def _mean(values: Sequence[float]) -> float:
    """Mean over the models; NaN if any is NaN (a missing model must not be averaged away)."""
    return sum(values) / len(values) if values and all(v == v for v in values) else float("nan")


def decide(
    rows: Sequence[Row],
    reg: Registry,
    *,
    arm: str = PRIMARY_ARM,
    models: Sequence[str] = MODELS,
    primary_set: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    safety_set: Iterable[str] | None = None,
    set_name: str = "P",
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
    worst: bool = False,
) -> Primary:
    """The registered decision for one layer arm against bare, in the primary cell.

    `primary_set` is a list of families, or a model -> list map (the per-model P_m sensitivity);
    `safety_set` is where the clean-pass guard looks beside P (default: every gated family, for the
    warn arm the bite families it ran on). worst=True counts every lost trial as the outcome worst
    for the layer.
    """
    cell, faulted, clean = PRIMARY_CELL, ("faulted",), ("clean",)
    safety = sorted_families(
        safety_set if safety_set is not None else (reg.bite if arm == WARN_ARM else reg.gated)
    )
    sets: dict[str, list[str]] = {}
    for m in models:
        chosen = primary_set[m] if isinstance(primary_set, Mapping) else primary_set
        sets[m] = sorted_families(reg.primary if chosen is None else chosen)

    def run(
        m: str,
        metric: str,
        twins: Sequence[str],
        families: list[str],
        name: str,
        sided: str | None = None,
    ) -> Contrast:
        return contrast(
            rows,
            m,
            arm,
            metric,
            twins,
            cell,
            families,
            name,
            sided,
            flips,
            draws,
            expected=reg.pair_keys(families, cell, twins),
            worst=worst,
        )

    drafts = {}
    for m in models:
        fam = sets[m]
        drafts[m] = (
            run(m, "silent", faulted, fam, set_name, "greater"),
            run(m, "notpass", faulted, fam, set_name, "greater"),
            run(m, "delivered", faulted, fam, set_name),
            run(m, "pass", clean, fam, set_name),
            run(m, "pass", clean, safety, "gated" if arm != WARN_ARM else "bite"),
        )
    pvalues: dict[str, float | None] = {}
    for m in MODELS:  # the Holm family is the three fixed model names, whoever has data
        if m in drafts and drafts[m][0].p is not None and drafts[m][1].p is not None:
            pvalues[m] = max(drafts[m][0].p, drafts[m][1].p)  # type: ignore[type-var]
        else:
            pvalues[m] = None
    adjusted = holm(pvalues, m=len(MODELS))
    decisions = []
    for m in models:
        drop, gain, deliv, clean_p, clean_g = drafts[m]
        drop.p_holm = gain.p_holm = adjusted[m]
        checks = {
            "silent_drop": drop.estimate >= MIN_DROP - EPS,
            "pass_gain": gain.estimate >= MIN_GAIN - EPS,
            "holm_p": adjusted[m] < ALPHA,
            "delivered_loss": deliv.estimate <= MAX_LOSS + EPS,
            "clean_loss_primary": clean_p.estimate <= MAX_LOSS + EPS,
            "clean_loss_gated": clean_g.estimate <= MAX_LOSS + EPS,
        }
        decisions.append(
            ModelDecision(
                m,
                drop,
                gain,
                deliv,
                clean_p,
                clean_g,
                pvalues[m],
                adjusted[m],
                checks,
                all(checks.values()),
            )
        )
    passing = tuple(d.model for d in decisions if d.passes)
    return Primary(
        arm=arm,
        set_name=set_name,
        n_primary=len({f for fam in sets.values() for f in fam}),
        n_safety=len(safety),
        models=decisions,
        models_passing=passing,
        effective=len(passing) >= MODELS_NEEDED,
        claim=arm == PRIMARY_ARM,
        worst_case=worst,
        mean_clean_pass_loss=_mean([d.clean_loss_gated.estimate for d in decisions]),
        mean_delivered_loss=_mean([d.delivered_loss.estimate for d in decisions]),
    )


def primary(
    rows: Sequence[Row],
    reg: Registry,
    *,
    arm: str = PRIMARY_ARM,
    models: Sequence[str] = MODELS,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
) -> Primary:
    return decide(rows, reg, arm=arm, models=models, flips=flips, draws=draws)


def warn_vs_bare(rows: Sequence[Row], reg: Registry, **kw: Any) -> Primary:
    """warn-nonote against bare on P (own Holm family); the safety guards look at the bite
    families, the only ones that arm ran on. No success claim."""
    return decide(rows, reg, arm=WARN_ARM, **kw)


def sensitivity_missing(rows: Sequence[Row], reg: Registry, **kw: Any) -> dict[str, Any]:
    """The decision with every lost trial counted as the outcome worst for the layer (a lost bare
    trial is not silent, a lost layer trial is silent): the verdict must survive it."""
    result = decide(rows, reg, worst=True, **kw)
    return {
        "effective": result.effective,
        "models_passing": result.models_passing,
        "primary": result,
    }


def final_verdict(
    result: Primary, worst: Primary | None, binding: bool, underpowered: bool, incomplete: bool
) -> str:
    """EFFECTIVE only if the rule is met, the worst case (binding when the arms lost trials
    unevenly) agrees, the design is powered and the data are complete."""
    if underpowered:
        return (
            f"DESCRIPTIVE ONLY (underpowered: P under {MIN_PRIMARY} families, or fewer than "
            f"{MODELS_NEEDED} models with {MIN_PRIMARY_MODEL} of their own)"
        )
    if not result.effective:
        return "NOT SHOWN"
    if binding and worst is not None and not worst.effective:
        return "NOT SHOWN (the verdict does not survive the worst case for lost trials)"
    return "EFFECTIVE (incomplete data, --allow-missing)" if incomplete else "EFFECTIVE"


# --- what the layer did to the silent errors ----------------------------------------------------


def outcome_split(
    rows: Sequence[Row],
    *,
    model: str,
    arm: str = PRIMARY_ARM,
    cell: Cell = PRIMARY_CELL,
    families: Iterable[str],
    twins: Sequence[str] = ("faulted",),
) -> dict[str, Any]:
    """Where the bare silent_wrong trials went under the layer: to a pass, to a decline
    (flagged_wrong), to no answer (undelivered, by kind), or stayed silent; and the trials that
    were not silent in bare and are silent with the layer. Overall and per family."""

    def empty() -> dict[str, Any]:
        return {
            "pairs": 0,
            "bare_silent": 0,
            "to_pass": 0,
            "to_silent": 0,
            "to_flagged": 0,
            "to_undelivered": 0,
            "undelivered_kinds": collections.Counter(),
            "new_silent": 0,
        }

    total, per_family = empty(), collections.defaultdict(empty)
    for p in pair_trials(rows, model=model, arm=arm, cell=cell, twins=twins, families=families):
        if p.base is None or p.arm is None:
            continue
        for bucket in (total, per_family[p.family]):
            bucket["pairs"] += 1
            if silent(p.arm) and not silent(p.base):
                bucket["new_silent"] += 1
            if not silent(p.base):
                continue
            bucket["bare_silent"] += 1
            outcome = p.arm["outcome"]
            if outcome == "pass":
                bucket["to_pass"] += 1
            elif outcome == "silent_wrong":
                bucket["to_silent"] += 1
            elif outcome == "flagged_wrong":
                bucket["to_flagged"] += 1
            else:
                bucket["to_undelivered"] += 1
                bucket["undelivered_kinds"][undelivered_kind(p.arm)] += 1

    def plain(b: dict[str, Any]) -> dict[str, Any]:
        return {**b, "undelivered_kinds": dict(sorted(b["undelivered_kinds"].items()))}

    return {
        "model": model,
        **plain(total),
        "per_family": {f: plain(per_family[f]) for f in sorted(per_family, key=fam_no)},
    }


def trial_summary(trials: Sequence[Row]) -> dict[str, Any]:
    """Counts of one group of counted trials (descriptive, trial-level, ignores clustering)."""
    n = len(trials)
    got = [r for r in trials if delivered(r)]
    sil = sum(silent(r) for r in trials)
    right = sum(bool(r.get("correct")) for r in trials)
    usd = sum(float(r.get("usd") or 0.0) for r in trials)
    layers = [r["layer"] for r in trials if isinstance(r.get("layer"), Mapping)]
    kinds = collections.Counter(undelivered_kind(r) for r in trials if not delivered(r))
    blocked = sum(1 for r in layers if (r.get("actions") or {}).get("block", 0) > 0)
    turns = [r["turns"] for r in trials if r.get("turns") is not None]
    return {
        "n": n,
        "pass": sum(passed(r) for r in trials),
        "silent": sil,
        "silent_planned": sil / n if n else float("nan"),
        "delivered": len(got),
        "delivered_share": len(got) / n if n else float("nan"),
        "silent_delivered": sil / len(got) if got else float("nan"),
        "silent_delivered_ci": clopper_pearson(sil, len(got)) if got else (float("nan"),) * 2,
        "flagged_wrong": sum(r["outcome"] == "flagged_wrong" for r in trials),
        "undelivered_kinds": dict(sorted(kinds.items())),
        "median_turns": statistics.median(turns) if turns else float("nan"),
        "usd": round(usd, 4),
        "usd_per_correct": usd / right if right else float("nan"),
        "layer_trials": len(layers),
        "layer_calls": sum(r.get("calls", 0) for r in layers),
        "layer_blocks": sum((r.get("actions") or {}).get("block", 0) for r in layers),
        "layer_warns": sum((r.get("actions") or {}).get("warn", 0) for r in layers),
        "layer_stops": sum(r.get("stops", 0) for r in layers),
        "friction": blocked / len(layers) if layers else float("nan"),  # trials with a block
    }


def _select(
    rows: Sequence[Row], model: str, arm: str, cell: Cell, twin: str, families: Iterable[str]
) -> list[Row]:
    wanted = set(families)
    return [
        r
        for r in rows
        if r["model"] == model
        and r.get("arm", BASE_ARM) == arm
        and r.get("rep", 0) == 0
        and cell_of(r) == cell
        and r["twin"] == twin
        and r["family"] in wanted
        and r["outcome"] not in SKIPPED
        and r.get("harness", "minimal") == "minimal"
    ]


def safety_table(
    rows: Sequence[Row],
    reg: Registry,
    *,
    cell: Cell = PRIMARY_CELL,
    arms: Sequence[str] = (BASE_ARM, WARN_ARM, PRIMARY_ARM),
    models: Sequence[str] = MODELS,
) -> list[dict[str, Any]]:
    """Per (model, arm, twin) over the gated families, and over P: clean pass, friction (clean
    trials with a block), delivered share, silent among delivered, flagged_wrong (the full
    contract only has it), median turns, USD per correct, the layer's own record, undelivered by
    kind. Trial-level counts: descriptive."""
    out = []
    for model in models:
        for arm in arms:
            families = reg.families_for(arm, cell)
            for scope, fams in (("gated", families), ("P", reg.primary)):
                for twin in ("clean", "faulted"):
                    trials = _select(rows, model, arm, cell, twin, fams)
                    if trials:
                        out.append(
                            {
                                "model": model,
                                "arm": arm,
                                "twin": twin,
                                "set": scope,
                                "cell": cell,
                                **trial_summary(trials),
                            }
                        )
    return out


def bypass(
    rows: Sequence[Row], reg: Registry, *, arm: str = PRIMARY_ARM, models: Sequence[str] = MODELS
) -> list[dict[str, Any]]:
    """Faulted layer-arm trials on P by whether the trapped route ran (`exposed` is judged on the
    executed commands): bypass = it ran although a rule fired in the trial, or because none fired;
    `blocked_only` = it was issued but never executed (exposed_issued without exposed)."""
    out = []
    for model in models:
        trials = _select(rows, model, arm, PRIMARY_CELL, "faulted", reg.primary)
        ran = [r for r in trials if r.get("exposed")]
        with_fire = [r for r in ran if (r.get("layer") or {}).get("fired")]
        out.append(
            {
                "model": model,
                "arm": arm,
                "faulted": len(trials),
                "route_ran": len(ran),
                "route_ran_after_a_fire": len(with_fire),
                "route_ran_no_fire": len(ran) - len(with_fire),
                "blocked_only": sum(
                    1 for r in trials if r.get("exposed_issued") and not r.get("exposed")
                ),
                "silent_given_route_ran": sum(silent(r) for r in ran),
                "layer_flagged": sum(layer_flag(r) for r in trials),
            }
        )
    return out


def mcnemar_seed_level(
    rows: Sequence[Row],
    reg: Registry,
    *,
    arm: str = PRIMARY_ARM,
    models: Sequence[str] = MODELS,
    cell: Cell = PRIMARY_CELL,
) -> list[dict[str, Any]]:
    """Exact McNemar on the discordant (family, seed) pairs of P x faulted; descriptive, it
    ignores the clustering by family."""
    out = []
    for model in models:
        pairs = [
            p
            for p in pair_trials(
                rows, model=model, arm=arm, cell=cell, twins=("faulted",), families=reg.primary
            )
            if p.base is not None and p.arm is not None
        ]
        b = sum(silent(p.base) and not silent(p.arm) for p in pairs)  # type: ignore[arg-type]
        c = sum(silent(p.arm) and not silent(p.base) for p in pairs)  # type: ignore[arg-type]
        out.append(
            {"model": model, "pairs": len(pairs), "fixed": b, "broken": c, "p": mcnemar_exact(b, c)}
        )
    return out


def regret(
    rows: Sequence[Row],
    model: str,
    arm: str,
    cell: Cell,
    families: Iterable[str],
    baseline: str = BASE_ARM,
    base_rep: int = 0,
    arm_rep: int = 0,
) -> dict[str, tuple[int, int, int]]:
    """twin -> (regretted, passes of the baseline, pairs): trials right without the layer and
    wrong with it. faulted: pass, then silent_wrong; clean: pass, then not pass. With
    arm == baseline and arm_rep == 1 this is the A/A noise floor of the same count."""
    counts = {"faulted": [0, 0, 0], "clean": [0, 0, 0]}
    pairs = pair_trials(
        rows,
        model=model,
        arm=arm,
        cell=cell,
        twins=("faulted", "clean"),
        families=families,
        baseline=baseline,
        base_rep=base_rep,
        arm_rep=arm_rep,
    )
    for p in pairs:
        if p.base is None or p.arm is None:
            continue
        counts[p.twin][2] += 1
        if passed(p.base):
            counts[p.twin][1] += 1
            counts[p.twin][0] += silent(p.arm) if p.twin == "faulted" else not passed(p.arm)
    return {twin: (c[0], c[1], c[2]) for twin, c in counts.items()}


def heterogeneity(
    rows: Sequence[Row],
    model: str,
    arm: str,
    families: Iterable[str],
    key: Callable[[str], str],
    cell: Cell = PRIMARY_CELL,
    draws: int = BOOTSTRAPS,
    min_group: int = 5,
) -> dict[str, Contrast]:
    """The silent contrast within groups of families (by pack or by locus); descriptive. A group
    below `min_group` families gets its estimate and no interval."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for family in families:
        groups[key(family)].append(family)
    return {
        name: contrast(
            rows,
            model,
            arm,
            "silent",
            ("faulted",),
            cell,
            members,
            name,
            None,
            0,
            draws if len(members) >= min_group else 0,
        )
        for name, members in sorted(groups.items())
    }


def pooled(
    contrasts: Sequence[Contrast], draws: int = BOOTSTRAPS, seed: int = SEED + 2
) -> dict[str, Any]:
    """Mean of the models' estimates over the families they all have, resampled jointly. The
    per-model numbers are the reported unit; this one is labelled pooled."""
    have = [c for c in contrasts if c.per_family]
    common = (
        sorted(set.intersection(*(set(c.per_family) for c in have)), key=fam_no) if have else []
    )
    out: dict[str, Any] = {
        "label": "pooled over models",
        "models": len(have),
        "families": len(common),
        "estimate": float("nan"),
        "low": float("nan"),
        "high": float("nan"),
    }
    if not common:
        return out
    table = [[c.per_family[f] for f in common] for c in have]

    def mean_over(idx: Sequence[int]) -> float:
        return sum(sum(row[i] for i in idx) / len(idx) for row in table) / len(table)

    out["estimate"] = mean_over(range(len(common)))
    if draws > 0:
        rng = random.Random(seed)
        n = len(common)
        means = sorted(mean_over([rng.randrange(n) for _ in range(n)]) for _ in range(draws))
        out["low"], out["high"] = (
            means[int(draws * 0.025)],
            means[min(draws - 1, int(draws * 0.975))],
        )
    return out


def baseline_rate(
    rows: Sequence[Row], model: str, families: Iterable[str], cell: Cell = PRIMARY_CELL
) -> float:
    """Mean over families of the bare silent/planned rate on the faulted twin."""
    rates = []
    for f in sorted_families(families):
        trials = _select(rows, model, BASE_ARM, cell, "faulted", [f])
        if trials:
            rates.append(sum(silent(r) for r in trials) / len(trials))
    return sum(rates) / len(rates) if rates else float("nan")


# --- strata, replication, the default cell -------------------------------------------------------


def specificity(
    rows: Sequence[Row],
    reg: Registry,
    *,
    arm: str = PRIMARY_ARM,
    models: Sequence[str] = MODELS,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """The silent contrast on Q1, Q2 and Q3 (two-sided sign-flip, family-bootstrap interval, the
    baseline silent rate and the clean-pass loss), on P for reference, and P minus Q1 per model.
    Report only: Q1 is a transparency check (expected 0, not proof of specificity), Q2 shows the
    specific effect plus friction, Q3 is the floor."""
    cell, faulted = PRIMARY_CELL, ("faulted",)
    out: dict[str, Any] = {"strata": {}, "P_minus_Q1": {}}
    for name, fams in (("P", reg.primary), ("Q1", reg.q1), ("Q2", reg.q2), ("Q3", reg.q3)):
        out["strata"][name] = {
            "families": len(fams),
            "models": {
                m: {
                    "silent": contrast(
                        rows,
                        m,
                        arm,
                        "silent",
                        faulted,
                        cell,
                        fams,
                        name,
                        "two-sided",
                        flips,
                        draws,
                        expected=reg.pair_keys(fams, cell, faulted),
                    ),
                    "clean_pass_loss": contrast(
                        rows,
                        m,
                        arm,
                        "pass",
                        ("clean",),
                        cell,
                        fams,
                        name,
                        None,
                        flips,
                        draws,
                        expected=reg.pair_keys(fams, cell, ("clean",)),
                    ),
                    "baseline_silent": baseline_rate(rows, m, fams, cell),
                }
                for m in models
            },
        }
    for m in models:
        a = [c for c in out["strata"]["P"]["models"][m]["silent"].per_family.values()]
        b = [c for c in out["strata"]["Q1"]["models"][m]["silent"].per_family.values()]
        est, low, high = bootstrap_difference(a, b, draws)
        out["P_minus_Q1"][m] = {
            "estimate": est,
            "low": low,
            "high": high,
            "P": len(a),
            "Q1": len(b),
        }
    return out


def coverage(reg: Registry) -> dict[str, Any]:
    n_bite, n_gated = len(reg.bite), len(reg.gated)
    return {
        "P": len(reg.primary),
        "bite": n_bite,
        "gated": n_gated,
        "q1": len(reg.q1),
        "q2": len(reg.q2),
        "q3": len(reg.q3),
        "weak": len(reg.weak),
        "P_over_bite": len(reg.primary) / n_bite if n_bite else float("nan"),
        "bite_over_gated": n_bite / n_gated if n_gated else float("nan"),
    }


def warn_bite(
    rows: Sequence[Row],
    reg: Registry,
    models: Sequence[str] = MODELS,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
) -> dict[str, Contrast] | None:
    """The silent contrast of warn-nonote against bare on the whole bite set it ran on (P and
    Q1; no test), or None if no warn-nonote record exists."""
    if not any(r.get("arm") == WARN_ARM for r in rows):
        return None
    cell, faulted = PRIMARY_CELL, ("faulted",)
    return {
        m: contrast(
            rows, m, WARN_ARM, "silent", faulted, cell, reg.bite, "bite", None, flips, draws,
            expected=reg.pair_keys(reg.bite, cell, faulted),
        )
        for m in models
    }  # fmt: skip


def family_sets(
    rows: Sequence[Row],
    reg: Registry,
    arm: str,
    models: Sequence[str] = MODELS,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """The silent contrast (estimate and interval, no test) on the broader sets the draft names:
    every bite family (reachable or not), every gated family, and the five stage-1 families as a
    replication row."""
    cell, faulted = PRIMARY_CELL, ("faulted",)
    sets = {
        "all_bite": reg.bite,
        "all_gated": reg.gated,
        "stage1_replication": [f for f in STAGE1_FAMILIES if f in reg.gated],
    }
    return {
        name: {
            "families": len(fams),
            "models": {
                m: contrast(
                    rows,
                    m,
                    arm,
                    "silent",
                    faulted,
                    cell,
                    fams,
                    name,
                    None,
                    flips,
                    draws,
                    expected=reg.pair_keys(fams, cell, faulted),
                )
                for m in models
            },
        }
        for name, fams in sets.items()
    }


def default_cell(
    rows: Sequence[Row],
    reg: Registry,
    *,
    arm: str = PRIMARY_ARM,
    models: Sequence[str] = MODELS,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """The full-contract default cell as a harm and friction check: faulted silent drop (no test),
    clean-pass loss, delivered loss over both twins, friction, flagged_wrong. F4 (must-detect,
    this cell only) is reported apart: its pass is a detection."""
    cell = DEFAULT_CELL
    out: dict[str, Any] = {"cell": cell, "arm": arm, "models": {}, "F4": {}}
    both, faulted, clean = ("clean", "faulted"), ("faulted",), ("clean",)
    fams = reg.gated
    for m in models:

        def run(metric: str, twins: Sequence[str], m: str = m) -> Contrast:
            return contrast(
                rows,
                m,
                arm,
                metric,
                twins,
                cell,
                fams,
                "default",
                None,
                flips,
                draws,
                expected=reg.pair_keys(fams, cell, twins),
            )

        clean_loss = run("pass", clean)
        out["models"][m] = {
            "silent_drop": run("silent", faulted),
            "clean_pass_loss": clean_loss,
            "delivered_loss": run("delivered", both),
            "harm_flag": not (clean_loss.estimate <= MAX_LOSS + EPS),
        }
        out["F4"][m] = {
            a: {
                twin: collections.Counter(
                    r["outcome"] for r in _select(rows, m, a, cell, twin, reg.default_only)
                )
                for twin in both
            }
            for a in (BASE_ARM, arm)
        }
    return out


# --- sensitivities --------------------------------------------------------------------------------


def _without_flagged(rows: Sequence[Row]) -> list[Row]:
    """The rows minus the layer trials whose layer was not intact (their pairs are dropped)."""
    return [r for r in rows if not (r.get("arm", BASE_ARM) in LAYER_ARMS and layer_flag(r))]


def sensitivities(
    rows: Sequence[Row], reg: Registry, *, flips: int = FLIPS, models: Sequence[str] = MODELS
) -> list[dict[str, Any]]:
    """The registered decision recomputed on other sets or data: without the five stage-1
    families, leaving one pack out, without the k most influential families, with the weak bites
    added, with each model's own bite families, with every bite family (no reachability), and
    without the layer trials whose layer was not intact. No intervals (draws 0)."""
    base = decide(rows, reg, models=models, flips=flips, draws=0)
    runs: list[tuple[str, Any, Sequence[Row]]] = [("registered P", None, rows)]
    p = list(reg.primary)
    runs.append(
        ("without the five stage-1 families", [f for f in p if f not in STAGE1_FAMILIES], rows)
    )
    for pack in dict.fromkeys(pack_of(f) for f in p):
        runs.append((f"leave out the {pack} pack", [f for f in p if pack_of(f) != pack], rows))
    influence = {
        f: sum(d.drop.per_family.get(f, 0.0) for d in base.models) / len(base.models) for f in p
    }
    ranked = sorted(p, key=lambda f: (-influence[f], fam_no(f)))
    for k in (1, 2, 3):
        runs.append(
            (
                f"without the {k} most influential families",
                [f for f in p if f not in ranked[:k]],
                rows,
            )
        )
    runs.append(("weak bites added (at least one silent)", [*p, *reg.weak_reachable], rows))
    if reg.per_model:
        runs.append(
            (
                "each model's own bite families (P_m)",
                {m: reg.per_model.get(m, {}).get("primary", ()) for m in models},
                rows,
            )
        )
    runs.append(("every bite family (no reachability)", list(reg.bite), rows))
    flagged = sum(1 for r in rows if r.get("arm", BASE_ARM) in LAYER_ARMS and layer_flag(r))
    if flagged:
        runs.append(
            (
                f"without {flagged} layer trial(s) with errors or a touched layer",
                None,
                _without_flagged(rows),
            )
        )
    out = []
    for name, chosen, data in runs:
        if chosen is not None and not any(
            chosen.values() if isinstance(chosen, Mapping) else chosen
        ):
            continue
        res = decide(data, reg, models=models, primary_set=chosen, flips=flips, draws=0)
        out.append(
            {
                "name": name,
                "n_primary": res.n_primary,
                "estimates": {d.model: d.drop.estimate for d in res.models},
                "models_passing": list(res.models_passing),
                "effective": res.effective,
            }
        )
    return out


def lenient_rows(rows: Sequence[Row], dirs: Sequence[str | Path]) -> tuple[list[Row], int]:
    """The rows rescored from the trial's work/result.json with the lenient reading (a top-level
    list or number is the answer, control characters in strings are accepted): the sensitivity to
    the strict format. Read only. Returns the rows and how many could not be re-read."""
    out: list[Row] = []
    unavailable = 0
    for r in rows:
        if r["outcome"] in SKIPPED or r["outcome"] == "refused":
            out.append(r)
            continue
        found = None
        for d in dirs:
            trial = Path(d) / "trials" / r["trial"]
            if (trial / "task.json").is_file():
                found = trial
                break
        if found is None:
            unavailable += 1
            out.append(r)
            continue
        if not (
            found / "work"
        ).is_dir():  # nothing to re-read: say so, as the runner's rescore does
            unavailable += 1
            out.append(r)
            continue
        try:
            truth = json.loads((found / "task.json").read_text())["truth"]
            result, problem = read_result(found / "work", strict=False)
            new = score_result(truth, result, problem, r.get("contract", "full")).to_json()
        except (OSError, ValueError, KeyError):
            unavailable += 1
            out.append(r)
            continue
        out.append({**r, **new})
    return out, unavailable


def sensitivity_lenient(
    rows: Sequence[Row], dirs: Sequence[str | Path], reg: Registry, **kw: Any
) -> dict[str, Any]:
    """The decision on the leniently rescored rows, next to the strict one."""
    relaxed, unavailable = lenient_rows(rows, dirs)
    result = decide(relaxed, reg, **kw)
    return {"unavailable": unavailable, "effective": result.effective, "primary": result}


# --- undisclosed silent errors (the chat reply is not scored; descriptive) -----------------------


def final_message(messages: Sequence[Any]) -> str:
    """The text of the last assistant message of a transcript (Anthropic or OpenAI style)."""
    for message in reversed(list(messages)):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, Mapping) and b.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
    return ""


def _atoms(truth: Mapping[str, Any]) -> list[str]:
    """The pieces of the expected answer that a message would have to contain."""
    expected = truth.get("expected")
    kind = truth.get("kind")
    if kind in ("int", "float"):
        return [str(expected)]
    if kind in ("id_list", "str_list", "seq_list", "int_list"):
        return [str(x) for x in expected]
    if kind in ("dict_num", "contexts"):
        return [f"{v}" for v in expected.values()]
    return []


def contains_expected(message: str, truth: Mapping[str, Any]) -> bool:
    """The message holds the exact expected value (every piece of it, as a whole token)."""
    atoms = _atoms(truth)
    if not atoms:
        return False
    before, after = r"[0-9A-Za-z_.]", r"(?:[0-9A-Za-z_]|\.[0-9A-Za-z_])"
    return all(re.search(rf"(?<!{before}){re.escape(a)}(?!{after})", message) for a in atoms)


def disclosure(
    rows: Sequence[Row],
    dirs: Sequence[str | Path],
    reg: Registry,
    models: Sequence[str] = MODELS,
    sample: int = 20,
) -> dict[str, Any]:
    """Of the bare silent_wrong trials on P x faulted: how many have a final chat message that
    holds the expected value or an unreliability phrase (disclosed) and how many do not
    (undisclosed silent). A regex, so a deterministic sample of trials is listed for a hand audit.
    The score ignores the chat reply: this only says how much of 'silent' was said in prose."""
    out: dict[str, Any] = {"models": {}, "audit": []}
    for model in models:
        trials = [
            r
            for r in _select(rows, model, BASE_ARM, PRIMARY_CELL, "faulted", reg.primary)
            if silent(r)
        ]
        counts = {
            "silent": len(trials),
            "expected_value": 0,
            "doubt": 0,
            "undisclosed": 0,
            "unavailable": 0,
        }
        for r in trials:
            found = next(
                (
                    Path(d) / "trials" / r["trial"]
                    for d in dirs
                    if (Path(d) / "trials" / r["trial"] / "task.json").is_file()
                ),
                None,
            )
            if found is None:
                counts["unavailable"] += 1
                continue
            try:
                truth = json.loads((found / "task.json").read_text())["truth"]
                text = final_message(json.loads((found / "transcript.json").read_text()))
            except (OSError, ValueError, KeyError):
                counts["unavailable"] += 1
                continue
            has_value, has_doubt = contains_expected(text, truth), bool(DOUBT.search(text))
            counts["expected_value"] += has_value
            counts["doubt"] += has_doubt
            counts["undisclosed"] += not (has_value or has_doubt)
            out["audit"].append(
                (
                    hashlib.sha256(r["trial"].encode()).hexdigest()[:12],
                    r["trial"],
                    has_value,
                    has_doubt,
                    text[-300:],
                )
            )
        out["models"][model] = counts
    out["audit"] = [list(a[1:]) for a in sorted(out["audit"])[:sample]]
    return out


# --- the bite set (development seeds only) -----------------------------------------------------


@dataclass(frozen=True)
class FamilyRule:
    """The rules written for one family's trap."""

    rules: tuple[str, ...] = ()
    route_rules: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    seed_parity: str | None = None  # "even" or "odd": the own rules cover only those seeds
    note: str = ""

    def own(self, seed: int | None = None, route: str = "forced") -> frozenset[str]:
        if (
            self.seed_parity
            and seed is not None
            and (seed % 2 == 0) != (self.seed_parity == "even")
        ):
            return frozenset()
        return frozenset(self.rules) | frozenset(self.route_rules.get(route, ()))


FAMILY_RULES_PATH = Path(__file__).parent / "tasks" / "family_rules.json"


def load_family_rules(path: str | Path | None = None) -> dict[str, FamilyRule]:
    doc = json.loads(Path(path or FAMILY_RULES_PATH).read_text())
    return {
        f: FamilyRule(
            tuple(e.get("rules", ())),
            {k: tuple(v) for k, v in e.get("route_rules", {}).items()},
            e.get("seed_parity"),
            e.get("note", ""),
        )
        for f, e in doc["families"].items()
    }


def blocking_rules(promotion_path: str | Path | None = None) -> frozenset[str]:
    """The rules that block in the block arm: promoted by calibration for the spec as it is now
    (a T2 rule only ever warns). The replay runs the engine in block mode, so it must not credit
    a rule the arm would only warn about."""
    from science_harness.engine.decide import Promotion
    from science_harness.engine.spec import packaged_specs

    promotion = Promotion.load(promotion_path) if promotion_path else Promotion.packaged()
    return frozenset(
        rule.id
        for spec in packaged_specs()
        for rule in spec.rules
        if rule.tier != "T2" and promotion.mode_of(rule.id, spec.content_hash()) == "block"
    )


def pilot_gaps(
    rows: Sequence[Row], model: str, families: Iterable[str], seeds: Sequence[int] = PILOT_SEEDS
) -> list[tuple[str, str, int]]:
    """The (family, twin, seed) cells of this model with no counted bare trial in the hard cell."""
    have = {
        (r["family"], r["twin"], r["seed"])
        for r in rows
        if r.get("model") == model
        and cell_of(r) == PRIMARY_CELL
        and r.get("arm", BASE_ARM) == BASE_ARM
        and r["outcome"] not in SKIPPED
    }
    return [
        (f, t, s)
        for f in sorted_families(families)
        for t in ("clean", "faulted")
        for s in seeds
        if (f, t, s) not in have
    ]


def classify_model(
    rows: Sequence[Row],
    model: str,
    families: Iterable[str],
    seeds: Sequence[int] = PILOT_SEEDS,
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """pilot.classify for one model on seeds exactly 0-4. With strict=True a missing (family,
    twin, seed) cell raises: an incomplete pilot must not shrink the bite set silently."""
    wanted = sorted_families(families)
    mine = [r for r in rows if r.get("model") == model]
    gaps = pilot_gaps(mine, model, wanted, seeds)
    if gaps and strict:
        raise ValueError(
            f"{model}: {len(gaps)} pilot cell(s) have no counted trial, first {gaps[:3]}"
        )
    entries = classify(mine, seeds)
    return {e["family"]: e for e in entries if e["family"] in set(wanted)}


def _exposed_pilot(pilot_rows: Iterable[Row], seeds: Sequence[int]) -> dict[str, Row]:
    """trial id -> the faulted pilot trials (bare, hard cell, these seeds) that took the route."""
    wanted = set(seeds)
    return {
        r["trial"]: r
        for r in pilot_rows
        if r["twin"] == "faulted"
        and cell_of(r) == PRIMARY_CELL
        and r.get("arm", BASE_ARM) == BASE_ARM
        and r["outcome"] not in SKIPPED
        and r.get("seed") in wanted
        and took_route(r)
    }


def _fired(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(rule, effective) of every rule the replay says did not pass. Older replays only have the
    blocking rules (and no warn-only ones)."""
    if "fired" in record:
        out = []
        for item in record["fired"]:
            rule, _code, effective = (
                str(item).rsplit(":", 2) if str(item).count(":") >= 2 else (str(item), "", "block")
            )
            out.append((rule, effective))
        return out
    return [(rule, "block") for rule in record.get("blocking_rules", ())]


def reachable(
    replays: Iterable[Mapping[str, Any]],
    pilot_rows: Iterable[Row],
    *,
    own_rules: Mapping[str, FamilyRule] | None = None,
    blocking: Collection[str] | None = None,
    seeds: Sequence[int] = PILOT_SEEDS,
) -> dict[str, dict[str, Any]]:
    """family -> what replaying its exposed faulted pilot transcripts through the engine showed.

    `block`: one of the family's OWN rules would block (and, if `blocking` is given, blocks in the
    arm: promoted by calibration); this is 'reachable'. `warn_only`: an own rule fired but never
    as a block; `foreign`: some other rule fired; `crashed`: replays that crashed or did not
    complete (an error: a family that is not reachable and has one is undecided); `n` replays used
    and `missing` exposed trials with no replay at all. Without `own_rules` any rule is own.
    """
    exposed = _exposed_pilot(pilot_rows, seeds)
    wanted: collections.Counter[str] = collections.Counter(r["family"] for r in exposed.values())
    out: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for rec in replays:
        trial = rec["trial"]
        if trial not in exposed or trial in seen:
            continue
        seen.add(trial)
        family = exposed[trial]["family"]
        e = out.setdefault(
            family,
            {
                "block": False,
                "warn_only": False,
                "foreign": False,
                "crashed": 0,
                "n": 0,
                "missing": 0,
            },
        )
        e["n"] += 1
        if rec.get("crashed") or rec.get("problem"):
            e["crashed"] += 1
        own = None
        if own_rules is not None:
            rule = own_rules.get(family, FamilyRule())
            own = rule.own(
                exposed[trial].get("seed"),
                str(rec.get("route") or exposed[trial].get("route") or "forced"),
            )
        for rule_id, effective in _fired(rec):
            if own is not None and rule_id not in own:
                e["foreign"] = True
            elif effective == "block" and (blocking is None or rule_id in blocking):
                e["block"] = True
            else:
                e["warn_only"] = True
    for family, count in wanted.items():
        e = out.setdefault(
            family,
            {
                "block": False,
                "warn_only": False,
                "foreign": False,
                "crashed": 0,
                "n": 0,
                "missing": 0,
            },
        )
        e["missing"] = count - e["n"]
        e["warn_only"] = e["warn_only"] and not e["block"]
    return dict(sorted(out.items(), key=lambda kv: fam_no(kv[0])))


def _blocks(entry: Any) -> bool:
    return entry if isinstance(entry, bool) else bool(entry and entry.get("block"))


def build_sets(
    pilot_rows: Sequence[Row],
    families: Iterable[str],
    reach: Mapping[str, Any],
    excluded: Iterable[str] = (),
    models: Sequence[str] = MODELS,
    *,
    seeds: Sequence[int] = PILOT_SEEDS,
    strict: bool = True,
) -> dict[str, Any]:
    """The frozen sets from the development-seed pilot (bare arm, primary cell, seeds 0-4).

    Per model a family bites if at least two silent_wrong among at least three exposed delivered
    faulted trials (`exposed`, or an answer equal to the trap's); exactly one is a weak bite,
    reported apart. P = bites for at least one model and an own rule would block (`reach`);
    Q1 = bite, no own rule would block; Q2 = own rule would block, no bite; Q3 = neither.
    A family is excluded if named, or if the clean-twin bare pass is below 60% for two or more
    models. With strict=True an incomplete pilot (a missing model x family x twin x seed cell) or
    an undecided reachability (a replay error, an exposed trial without a replay) raises.
    """
    explicit = sorted_families(excluded)
    every = sorted_families(families)
    default_only = [f for f in every if f in DEFAULT_ONLY]
    cellf = [f for f in every if f not in DEFAULT_ONLY]
    candidates = [f for f in cellf if f not in explicit]
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    gaps: dict[str, list[Any]] = {}
    for m in models:
        found = pilot_gaps([r for r in pilot_rows if r.get("model") == m], m, candidates, seeds)
        if found:
            gaps[m] = [list(g) for g in found]
        entries[m] = classify_model(pilot_rows, m, cellf, seeds, strict=False)
    if gaps and strict:
        first = next(iter(gaps))
        raise ValueError(
            "incomplete pilot: "
            + "; ".join(f"{m}: {len(g)} cell(s)" for m, g in gaps.items())
            + f", first {gaps[first][:3]}"
        )
    auto: dict[str, str] = {}
    for f in candidates:
        low = []
        for m in models:
            k, n = entries[m].get(f, {}).get("clean_pass", (0, 0))
            if n and k / n < CLEAN_PASS_FLOOR - EPS:
                low.append(f"{m} ({k}/{n})")
        if len(low) >= CLEAN_EXCLUDE_MODELS:
            auto[f] = "clean-twin bare pass below 60% in the pilot for " + ", ".join(low)
    out = set(explicit) | set(auto)
    gated = [f for f in cellf if f not in out]
    bite_m = {
        m: [f for f in gated if entries[m].get(f, {}).get("label") == LABEL_BITES] for m in models
    }
    weak_m = {
        m: [f for f in gated if entries[m].get(f, {}).get("label") == LABEL_WEAK] for m in models
    }
    bite = sorted_families(f for fams in bite_m.values() for f in fams)
    weak = [
        f for f in sorted_families(f for fams in weak_m.values() for f in fams) if f not in bite
    ]
    reach_all = {f: reach.get(f) for f in gated}
    undecided = [
        f
        for f in gated
        if reach_all[f]
        and not isinstance(reach_all[f], bool)
        and not reach_all[f].get("block")
        and (reach_all[f].get("crashed") or reach_all[f].get("missing"))
    ]
    if undecided and strict:
        raise ValueError(
            f"reachability is undecided for {undecided[:5]}: replay errors or exposed pilot trials "
            "with no replay (rerun the replay over the same pilot directories)"
        )
    primary_set = [f for f in bite if _blocks(reach_all[f])]
    q1 = [f for f in bite if f not in primary_set]
    rest = [f for f in gated if f not in bite]
    q2 = [f for f in rest if _blocks(reach_all[f])]
    q3 = [f for f in rest if not _blocks(reach_all[f])]
    p_m = {m: [f for f in bite_m[m] if _blocks(reach_all[f])] for m in models}
    enough = sum(len(p_m[m]) >= MIN_PRIMARY_MODEL for m in models)
    underpowered = len(primary_set) < MIN_PRIMARY or enough < MODELS_NEEDED
    to_run = [*gated, *default_only]
    wanted_seeds = set(seeds)
    used = [
        r
        for r in pilot_rows
        if cell_of(r) == PRIMARY_CELL
        and r.get("arm", BASE_ARM) == BASE_ARM
        and r.get("seed") in wanted_seeds
    ]
    digest = hashlib.sha256(
        "\n".join(
            sorted(
                f"{r['trial']}|{r['outcome']}|{bool(r.get('exposed'))}|{r.get('signature')}"
                for r in used
            )
        ).encode()
    ).hexdigest()
    return {
        "format": SETS_FORMAT,
        "rule": (
            "bite per model on pilot seeds 0-4: at least 2 silent_wrong among at least 3 exposed "
            "delivered faulted trials (exposed, or the trap's own answer); weak bite = exactly 1; "
            "reachable = one of the family's own rules (family_rules.json) would block in a replay "
            "of the exposed transcripts and is promoted to block"
        ),
        "pilot": {
            "rows": len(used),
            "digest": digest,
            "seeds": list(seeds),
            "models": list(models),
        },
        "labels": {m: {f: e["label"] for f, e in entries[m].items()} for m in models},
        "counts": {
            m: {
                f: {
                    "trials": e["trials"],
                    "undelivered": e["undelivered"],
                    "delivered_faulted": e["faulted_delivered"],
                    "exposed": e["exposed"],
                    "silent": e["silent_given_exposed"][0],
                    "silent_unexposed": e["silent_unexposed"],
                    "exposed_by_signature": e["exposed_by_signature"],
                    "clean_silent": e["clean_silent"][0],
                    "clean_delivered": e["clean_silent"][1],
                    "clean_pass": e["clean_pass"][0],
                    "clean_trials": e["clean_pass"][1],
                }
                for f, e in entries[m].items()
            }
            for m in models
        },
        "bite": bite,
        "weak": weak,
        "weak_reachable": [f for f in weak if _blocks(reach_all[f])],
        "primary": primary_set,
        "q1": q1,
        "bite_unreachable": q1,
        "q2": q2,
        "q3": q3,
        "complement": sorted_families([*q1, *q2, *q3]),
        "default_only": default_only,
        "excluded": sorted_families(out),
        "exclusion_reasons": {**{f: "named before launch" for f in explicit}, **auto},
        "per_model": {m: {"bite": bite_m[m], "weak": weak_m[m], "primary": p_m[m]} for m in models},
        "reach": {f: reach_all[f] for f in gated if reach_all[f] is not None},
        "clean_screen": {
            f: [m for m in models if entries[m].get(f, {}).get("clean_flag")]
            for f in cellf
            if any(entries[m].get(f, {}).get("clean_flag") for m in models)
        },
        "audit": {
            "silent_unexposed": [
                [m, f, e["silent_unexposed"]]
                for m in models
                for f, e in entries[m].items()
                if e["silent_unexposed"] and f not in out
            ],
        },
        "power": {
            "primary": len(primary_set),
            "primary_by_model": {m: len(p_m[m]) for m in models},
            "min_primary": MIN_PRIMARY,
            "min_primary_model": MIN_PRIMARY_MODEL,
            "models_needed": MODELS_NEEDED,
            "underpowered": underpowered,
        },
        "seeds": {f: list(registry_seeds(f)) for f in gated},
        "default_seeds": {f: list(registry_seeds(f)[:DEFAULT_SEED_COUNT]) for f in to_run},
        "flags": ["incomplete"] if gaps else [],
        "incomplete": gaps,
    }


# --- report -------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:+.1f}"


def _plain(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def _num(x: float | None, spec: str = ".1f") -> str:
    return "n/a" if x is None or x != x else format(x, spec)


def _p(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _ci(c: Mapping[str, Any]) -> str:
    return f"[{_pct(c['low'])}, {_pct(c['high'])}]"


def _table(header: Sequence[str], body: Sequence[Sequence[Any]], indent: str = "  ") -> list[str]:
    """Left-aligned text columns sized to their content."""
    cells = [[str(c) for c in header], *([str(c) for c in row] for row in body)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return [
        indent + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in cells
    ]


def render(result: Primary | Mapping[str, Any]) -> str:
    """The registered decision as text (a Primary, or its dict from the JSON)."""
    r = asdict(result) if isinstance(result, Primary) else result
    head = (
        f"arm {r['arm']} against bare, primary cell, set {r['set_name']} "
        f"({r['n_primary']} families), faulted twin; points; "
        "positive = the layer lowers silent / raises pass"
    )
    lines = [
        head
        + ("; WORST CASE: every lost trial counted against the layer" if r["worst_case"] else "")
    ]
    body = []
    for d in r["models"]:
        drop, gain = d["drop"], d["gain"]
        body.append(
            [
                d["model"],
                drop["families"],
                drop["pairs"],
                _pct(drop["estimate"]),
                _ci(drop),
                _pct(gain["estimate"]),
                _p(d["p"]),
                _p(d["p_holm"]),
                _pct(d["delivered_loss"]["estimate"]),
                _pct(d["clean_loss_primary"]["estimate"]),
                _pct(d["clean_loss_gated"]["estimate"]),
                "yes"
                if d["passes"]
                else "no: " + ", ".join(k for k, ok in d["checks"].items() if not ok),
            ]
        )
    lines += _table(
        [
            "model",
            "fam",
            "pairs",
            "silent drop",
            "95% CI",
            "pass gain",
            "p",
            "p Holm",
            "deliv. loss",
            "clean P",
            "clean all",
            "passes",
        ],
        body,
    )
    lines.append(
        f"  models passing {len(r['models_passing'])} of {len(r['models'])} "
        f"({', '.join(r['models_passing']) or 'none'}); rule: drop and gain >= "
        f"{100 * MIN_DROP:.0f}, Holm p < {ALPHA}, delivered loss and clean-pass loss <= "
        f"{100 * MAX_LOSS:.0f}"
    )
    return "\n".join(lines)


def _split_lines(splits: Sequence[Mapping[str, Any]]) -> list[str]:
    body = []
    for s in splits:
        kinds = ", ".join(f"{k} {v}" for k, v in s["undelivered_kinds"].items()) or "-"
        body.append(
            [
                s["model"],
                s["pairs"],
                s["bare_silent"],
                s["to_pass"],
                s["to_flagged"],
                s["to_undelivered"],
                s["to_silent"],
                s["new_silent"],
                kinds,
            ]
        )
    return _table(
        [
            "model",
            "pairs",
            "bare silent",
            "-> pass",
            "-> flagged",
            "-> undelivered",
            "still silent",
            "new silent",
            "undelivered by kind",
        ],
        body,
    )


def _strata_lines(spec: Mapping[str, Any]) -> list[str]:
    body = []
    for name, stratum in spec["strata"].items():
        for m, cells in stratum["models"].items():
            c = cells["silent"]
            body.append(
                [
                    name,
                    stratum["families"],
                    m,
                    _plain(cells["baseline_silent"]),
                    _pct(c["estimate"]),
                    _ci(c),
                    _p(c["p"]),
                    _pct(cells["clean_pass_loss"]["estimate"]),
                ]
            )
    lines = _table(
        [
            "stratum",
            "fam",
            "model",
            "bare silent",
            "silent drop",
            "95% CI",
            "p (2-sided)",
            "clean-pass loss",
        ],
        body,
    )
    for m, d in spec["P_minus_Q1"].items():
        lines.append(
            f"  P minus Q1, {m}: {_pct(d['estimate'])} {_ci(d)} (P {d['P']} families, Q1 {d['Q1']})"
        )
    return lines


def _safety_lines(safety: Sequence[Mapping[str, Any]]) -> list[str]:
    body = []
    for t in safety:
        if (t["set"], t["twin"]) in (("P", "faulted"), ("gated", "clean")):
            body.append(
                [
                    t["model"],
                    t["arm"],
                    "/".join(t["cell"]),
                    t["set"],
                    t["twin"],
                    t["n"],
                    _plain(t["silent_planned"]),
                    _plain(t["silent_delivered"]),
                    _plain(t["delivered_share"]),
                    _plain(t["friction"]),
                    t["flagged_wrong"],
                    _num(t["median_turns"]),
                    _num(t["usd_per_correct"], ".4f"),
                    f"{t['layer_calls']}/{t['layer_blocks']}/{t['layer_warns']}/{t['layer_stops']}",
                ]
            )
    return _table(
        [
            "model",
            "arm",
            "cell",
            "set",
            "twin",
            "n",
            "silent/planned",
            "silent/delivered",
            "delivered",
            "friction",
            "flagged_wrong",
            "turns",
            "$/correct",
            "calls/blocks/warns/stops",
        ],
        body,
    )


def render_report(result: Mapping[str, Any]) -> str:
    """The whole analysis as text. Percentages are in points; trial-level tables are descriptive."""
    r = _jsonable(result)
    lines = [f"WARNING: {note}" for note in r.get("warnings", [])]
    lines += [f"VERDICT: {r['verdict']}", "", render(r["primary"]), ""]
    lines += [
        "  P is a conditional estimand: the effect on traps that bite these agents and where "
        "one of the family's own rules would block;",
        "  it is larger than the all-bite and all-gated figures below and leaves out biting "
        "families without a rule.",
    ]
    cluster = r.get("cluster", {})
    for m, c in cluster.items():
        lines.append(
            f"  pack-clustered interval of the silent drop, {m}: {_ci(c)} ({c['packs']} packs)"
        )
    lines += ["", "where the bare silent_wrong trials of P x faulted went under block-nonote"]
    lines += _split_lines(r["split"])
    lines += [
        "",
        "attrition: lost = absent + infra_error; a between-arm gap above 2 points is binding",
    ]
    lines += _table(
        ["model", "arm", "planned", "present", "lost", "lost %", "refused", "gap"],
        [
            [
                m["model"],
                m["arm"],
                m["planned"],
                m["present"],
                m["lost"],
                _plain(m["lost_share"]),
                m["refused"],
                "BINDING" if m["binding"] else "ok",
            ]
            for m in r["missingness"]
        ],
    )
    worst = r["worst_case"]
    lines.append(
        f"  worst case for lost trials: {'effective' if worst['effective'] else 'NOT effective'} "
        f"({len(worst['models_passing'])} models pass)"
    )
    if r.get("warn"):
        lines += ["", render(r["warn"]), "  secondary arm: no success claim"]
        lines += ["  warn-nonote on the whole bite set (silent drop, no test)"]
        lines += _table(
            ["model", "fam", "silent drop", "95% CI"],
            [
                [m, c["families"], _pct(c["estimate"]), _ci(c)]
                for m, c in (r.get("warn_bite") or {}).items()
            ],
        )
    lines += ["", "strata of the complement (report only; two-sided sign-flip p, not adjusted)"]
    lines += _strata_lines(r["specificity"])
    cov = r["coverage"]
    lines.append(
        f"  coverage: P {cov['P']} of {cov['bite']} bite families ({_plain(cov['P_over_bite'])}%); "
        f"bite {cov['bite']} of {cov['gated']} gated ({_plain(cov['bite_over_gated'])}%); "
        f"Q1 {cov['q1']}, Q2 {cov['q2']}, Q3 {cov['q3']}, weak bites {cov['weak']}"
    )
    lines += ["", "silent drop on broader sets (estimate and interval, no test)"]
    lines += _table(
        ["set", "fam", "model", "silent drop", "95% CI"],
        [
            [name, b["families"], m, _pct(c["estimate"]), _ci(c)]
            for name, b in r["sets"].items()
            for m, c in b["models"].items()
        ],
    )
    lines += ["", "sensitivity of the registered decision (no intervals)"]
    rows_ = [
        [
            s["name"],
            s["n_primary"],
            len(s["models_passing"]),
            "effective" if s["effective"] else "not shown",
        ]
        for s in r["sensitivity"]
    ]
    if r.get("lenient"):
        ln = r["lenient"]
        rows_.append(
            [
                f"lenient reading of result.json ({ln['unavailable']} trials not re-read)",
                "",
                len(ln["primary"]["models_passing"]),
                "effective" if ln["effective"] else "not shown",
            ]
        )
    lines += _table(["variation", "P", "models passing", "verdict"], rows_)
    lines += ["", "default cell (full contract): a harm and friction check, no test"]
    lines += _table(
        ["model", "faulted silent drop", "clean-pass loss", "delivered loss (both twins)", "harm"],
        [
            [
                m,
                _pct(d["silent_drop"]["estimate"]),
                _pct(d["clean_pass_loss"]["estimate"]),
                _pct(d["delivered_loss"]["estimate"]),
                "HARM" if d["harm_flag"] else "-",
            ]
            for m, d in r["default_cell"]["models"].items()
        ],
    )
    lines += ["", "delivered basis, friction and the layer's own record (trial level, descriptive)"]
    lines += _safety_lines(r["safety"])
    lines += ["", "bypass: faulted layer trials on P in which the trapped route ran"]
    lines += _table(
        [
            "model",
            "faulted",
            "route ran",
            "after a fire",
            "no fire",
            "issued but blocked",
            "silent | route ran",
            "layer not intact",
        ],
        [
            [
                b["model"],
                b["faulted"],
                b["route_ran"],
                b["route_ran_after_a_fire"],
                b["route_ran_no_fire"],
                b["blocked_only"],
                b["silent_given_route_ran"],
                b["layer_flagged"],
            ]
            for b in r["bypass"]
        ],
    )
    lines += [
        "",
        "McNemar on discordant (family, seed) pairs; descriptive, ignores family clustering",
    ]
    lines += _table(
        ["model", "pairs", "fixed", "broken", "p"],
        [[d["model"], d["pairs"], d["fixed"], d["broken"], _p(d["p"])] for d in r["mcnemar"]],
    )
    lines += ["", "regret (right without the layer, wrong with it) and its A/A noise floor"]
    body = []
    for m, g in r["regret"].items():
        aa = g.get("aa_faulted")
        body.append(
            [
                m,
                "{}/{}".format(*g["faulted"][:2]),
                "{}/{}".format(*g["clean"][:2]),
                "{}/{}".format(*aa[:2]) if aa else "no replicate",
            ]
        )
    lines += _table(["model", "faulted", "clean", "A/A faulted"], body)
    for m, d in r.get("aa", {}).items():
        lines.append(f"  A/A silent difference, {m}: {_pct(d['estimate'])} {_ci(d)}")
    pool = r["pooled"]
    lines.append(
        f"  pooled over models (labelled pooled): silent drop {_pct(pool['estimate'])} "
        f"{_ci(pool)} on {pool['families']} families"
    )
    lines += [
        "",
        "heterogeneity of the silent drop (no tests; a group under 5 families has no interval)",
    ]
    for key, per_model in r["heterogeneity"].items():
        for m, groups in per_model.items():
            shown = ", ".join(
                f"{g} {_pct(c['estimate'])} ({c['families']})" for g, c in groups.items()
            )
            lines.append(f"  by {key}, {m}: {shown}")
    if r.get("disclosure"):
        lines += ["", "bare silent_wrong on P x faulted: what the final chat message said (regex)"]
        lines += _table(
            ["model", "silent", "expected value", "doubt phrase", "undisclosed", "not re-read"],
            [
                [
                    m,
                    c["silent"],
                    c["expected_value"],
                    c["doubt"],
                    c["undisclosed"],
                    c["unavailable"],
                ]
                for m, c in r["disclosure"]["models"].items()
            ],
        )
    return "\n".join(lines)


# --- the commands ---------------------------------------------------------------------------------


def _jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return x if x == x and x not in (math.inf, -math.inf) else None
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(x) if isinstance(x, (set, frozenset)) else x)]
    if hasattr(x, "__dataclass_fields__"):
        return _jsonable(asdict(x))
    return x


def analyze(
    rows: Sequence[Row],
    reg: Registry,
    *,
    dirs: Sequence[str | Path] = (),
    plan_fn: Callable[..., Iterable[Any]] | None = None,
    flips: int = FLIPS,
    draws: int = BOOTSTRAPS,
    incomplete: bool = False,
    files: bool = True,
) -> dict[str, Any]:
    """Every registered number and the descriptive ones, as one dict (the --json output)."""
    rows = [r for r in rows if registered(r, reg)]
    block = decide(rows, reg, flips=flips, draws=draws)
    table = missingness(rows, reg, plan_fn=plan_fn)
    binding = any(m["binding"] for m in table)
    worst = sensitivity_missing(rows, reg, flips=flips, draws=0)
    splits = [outcome_split(rows, model=m, families=reg.primary) for m in MODELS]
    cell = PRIMARY_CELL
    regrets, aa = {}, {}
    for m in MODELS:
        r = regret(rows, m, PRIMARY_ARM, cell, reg.primary)
        floor = regret(rows, m, BASE_ARM, cell, reg.primary, BASE_ARM, 0, 1)
        regrets[m] = {**r, "aa_faulted": floor["faulted"] if floor["faulted"][2] else None}
        if floor["faulted"][2]:
            aa[m] = contrast(
                rows,
                m,
                BASE_ARM,
                "silent",
                ("faulted",),
                cell,
                reg.primary,
                "A/A",
                None,
                flips,
                draws,
                arm_rep=1,
            )
    result: dict[str, Any] = {
        "verdict": final_verdict(block, worst["primary"], binding, reg.underpowered, incomplete),
        "primary": block,
        "worst_case": worst,
        "split": splits,
        "missingness": table,
        "warn": warn_vs_bare(rows, reg, flips=flips, draws=draws)
        if any(r.get("arm") == WARN_ARM for r in rows)
        else None,
        "warn_bite": warn_bite(rows, reg, flips=flips, draws=draws),
        "specificity": specificity(rows, reg, flips=flips, draws=draws),
        "coverage": coverage(reg),
        "sets": family_sets(rows, reg, PRIMARY_ARM, flips=flips, draws=draws),
        "sensitivity": sensitivities(rows, reg, flips=flips),
        "default_cell": default_cell(rows, reg, flips=flips, draws=draws),
        "safety": safety_table(rows, reg, cell=cell)
        + safety_table(rows, reg, cell=DEFAULT_CELL, arms=(BASE_ARM, PRIMARY_ARM)),
        "bypass": bypass(rows, reg),
        "mcnemar": mcnemar_seed_level(rows, reg),
        "regret": regrets,
        "aa": aa,
        "pooled": pooled([c.drop for c in block.models], draws),
        "cluster": {
            d.model: dict(
                zip(
                    ("low", "high", "packs"),
                    cluster_bootstrap_ci(d.drop.per_family, pack_of, draws),
                    strict=True,
                ),
                estimate=d.drop.estimate,
            )
            for d in block.models
        },
        "heterogeneity": {
            key: {
                m: heterogeneity(rows, m, PRIMARY_ARM, reg.primary, fn, cell, draws) for m in MODELS
            }
            for key, fn in (("pack", pack_of), ("locus", _locus))
        },
        "underpowered": reg.underpowered,
    }
    if files and dirs:
        result["lenient"] = sensitivity_lenient(rows, dirs, reg, flips=flips, draws=0)
        result["disclosure"] = disclosure(rows, dirs, reg)
    return result


def _locus(family: str) -> str:
    from science_harness.bench.tasks.locus import locus_of

    return locus_of(family)


def _abort(lines: Iterable[str]) -> int:
    for line in lines:
        print(f"ABORT: {line}", file=sys.stderr)
    return 2


def dropped_families(
    rows: Sequence[Row], reg: Registry, arm: str = PRIMARY_ARM
) -> dict[str, tuple[str, ...]]:
    """model -> the P families that have no valid (bare, layer) faulted pair: they would vanish
    from the mean, so the analysis refuses to run (A-7) unless told to."""
    return {
        m: contrast(
            rows, m, arm, "silent", ("faulted",), PRIMARY_CELL, reg.primary, "P", None, 0, 0
        ).dropped
        for m in MODELS
    }


_TRIAL_DIR = re.compile(r"^(F[0-9]+)-s([0-9]+)-")


def seen_pairs(
    runs: Sequence[str | Path], skip: Sequence[str | Path] = ()
) -> dict[tuple[str, int], dict[str, int]]:
    """(family, seed) -> {run directory: number of trial directories}, from the directory names
    under DIR/trials/*/ of every run except those in `skip`."""
    skipped = {str(Path(d).resolve()) for d in skip}
    found: dict[tuple[str, int], dict[str, int]] = collections.defaultdict(dict)
    for run in runs:
        root = Path(run)
        if str(root.resolve()) in skipped or not (root / "trials").is_dir():
            continue
        for model_dir in sorted((root / "trials").iterdir()):
            if not model_dir.is_dir():
                continue
            for entry in os.scandir(model_dir):
                m = _TRIAL_DIR.match(entry.name)
                if m:
                    key = (m.group(1), int(m.group(2)))
                    found[key][str(root)] = found[key].get(str(root), 0) + 1
    return dict(found)


def registered_pairs(reg: Registry) -> set[tuple[str, int]]:
    """Every (family, seed) the stage-2 jobs will run, both cells."""
    pairs = {(f, s) for f, seeds in reg.seeds.items() for s in seeds}
    return pairs | {(f, s) for f, seeds in reg.default_seeds.items() for s in seeds}


def cmd_scan(args: argparse.Namespace) -> int:
    """Pre-launch: no registered (family, seed) may exist in any other run directory, or an agent
    has already seen that instance and 'no agent has seen a held-out seed' is false."""
    try:
        with open(args.sets) as handle:
            reg = Registry.from_sets(json.load(handle))
    except (ValueError, OSError) as exc:
        return _abort([str(exc)])
    wanted = registered_pairs(reg)
    seen = seen_pairs(args.runs, args.own)
    clash = {key: where for key, where in seen.items() if key in wanted}
    for (family, seed), where in sorted(clash.items(), key=lambda kv: (fam_no(kv[0][0]), kv[0][1])):
        print(f"{family} seed {seed}: " + ", ".join(f"{d} ({n})" for d, n in sorted(where.items())))
    print(
        f"{len(wanted)} registered (family, seed) pairs, {len(clash)} already run in "
        f"{len(args.runs)} run directories"
    )
    return 2 if clash else 0


def _promotion_sha(path: str | Path | None) -> str | None:
    """sha256 of the promotion table the reachability was judged against (None: no table)."""
    from science_harness.engine.decide import PACKAGED_PROMOTION

    table = Path(path) if path else PACKAGED_PROMOTION
    return hashlib.sha256(table.read_bytes()).hexdigest() if table.is_file() else None


def cmd_sets(args: argparse.Namespace) -> int:
    from science_harness.bench.tasks.families import all_families

    try:
        rows = load_rows(args.pilot)
        with open(args.replay) as handle:
            replays = json.load(handle)
        problems = provenance_problems(rows, pilot=True)
        if problems and not args.lenient:
            return _abort(problems)
        families = sorted_families(args.families or all_families())
        rules = load_family_rules(args.family_rules)
        blocking = blocking_rules(args.promotion)
        excluded = DEFAULT_EXCLUDED if args.exclude is None else args.exclude
        pilot_rows = [r for r in rows if r.get("seed") in set(PILOT_SEEDS)]
        reach = reachable(replays, pilot_rows, own_rules=rules, blocking=blocking)
        sets = build_sets(rows, families, reach, excluded, strict=not args.allow_incomplete)
    except (ValueError, OSError) as exc:
        return _abort([str(exc)])
    if args.lenient and problems:
        sets["flags"] = sorted({*sets["flags"], "lenient"})
        sets["provenance_problems"] = problems
    sets["pilot"]["dirs"] = [str(d) for d in args.pilot]
    sets["family_rules_sha"] = hashlib.sha256(
        Path(args.family_rules or FAMILY_RULES_PATH).read_bytes()
    ).hexdigest()
    sets["promotion_sha"] = _promotion_sha(args.promotion)
    sets["blocking_rules"] = len(blocking)
    with open(args.out, "w") as handle:
        json.dump(_jsonable(sets), handle, indent=1)
        handle.write("\n")
    power = sets["power"]
    print(
        f"{len(families)} families: bite {len(sets['bite'])}, primary {len(sets['primary'])}, "
        f"bite but no own rule blocks {len(sets['q1'])}, complement {len(sets['complement'])}, "
        f"excluded {len(sets['excluded'])}"
    )
    print(
        f"Q1 {len(sets['q1'])} Q2 {len(sets['q2'])} Q3 {len(sets['q3'])}; "
        f"weak bites {len(sets['weak'])}; P per model {power['primary_by_model']}; "
        + (
            "UNDERPOWERED (stop rule): the confirmatory test is descriptive"
            if power["underpowered"]
            else "powered by the stop rule"
        )
    )
    for f, reason in sets["exclusion_reasons"].items():
        print(f"excluded {f}: {reason}")
    if sets["clean_screen"]:
        print(
            "clean twin wrong in 2 or more of 5 trials (flagged, still in): "
            + ", ".join(f"{f} {ms}" for f, ms in sets["clean_screen"].items())
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Counts only: no outcome rate is printed, so nobody sees bare against layer before the
    analysis (draft: no interim look)."""
    try:
        with open(args.sets) as handle:
            reg = Registry.from_sets(json.load(handle))
        unreadable: list[Path] = []
        rows = load_records([str(d) for d in args.out], unreadable)
        plan_fn = runner_plan()
        ids = campaign_ids(reg, plan_fn=plan_fn)
        unrun = load_unrun(args.out)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        return _abort([str(exc)])
    by_id = {r["trial"]: r for r in rows}
    table = []
    for (model, arm, cell), planned_list in ids.items():
        present = [by_id[i] for i in planned_list if i in by_id]
        counts = collections.Counter(r["outcome"] for r in present)
        providers: collections.Counter[str] = collections.Counter()
        for r in present:
            providers.update(r.get("providers") or [])
        layers = [r["layer"] for r in present if isinstance(r.get("layer"), Mapping)]
        table.append(
            {
                "model": model,
                "arm": arm,
                "cell": "/".join(cell),
                "planned": len(planned_list),
                "recorded": len(present),
                "absent": len(planned_list) - len(present),
                "unrun": sum(i in unrun and i not in by_id for i in planned_list),
                "infra_error": counts["infra_error"],
                "infra_open": sum(
                    r["outcome"] == "infra_error" and int(r.get("attempt") or 1) < MAX_ATTEMPTS
                    for r in present
                ),
                "refused": counts["refused"],
                "usd": round(sum(float(r.get("usd") or 0) for r in present), 3),
                "providers": dict(sorted(providers.items())),
                "pin_off": sum(r.get("pin_match") is False for r in present),
                "layer_errors": sum(
                    len(x.get("errors") or []) + (x.get("ledger_errors") or 0) for x in layers
                ),
                "layer_crashed": sum(bool(x.get("crashed")) for x in layers),
                "layer_touched": sum(bool(x.get("touched_layer")) for x in layers),
                "reruns": sum(int(r.get("attempt") or 1) > 1 for r in present),
            }
        )
    everything = {i for v in ids.values() for i in v}
    extra = [r["trial"] for r in rows if r.get("rep", 0) == 0 and r["trial"] not in everything]
    columns = {
        "model": "model", "arm": "arm", "cell": "cell", "planned": "planned",
        "recorded": "recorded", "absent": "absent", "unrun": "budget-skipped",
        "infra_error": "infra", "infra_open": "open", "reruns": "reruns", "refused": "refused",
        "usd": "usd", "providers": "providers", "layer_errors": "layer errors",
        "layer_crashed": "crashed", "layer_touched": "touched", "pin_off": "pin off",
    }  # fmt: skip
    header = list(columns.values())
    body = [[t[key] for key in columns] for t in table]
    print("\n".join(_table(header, body, indent="")))
    total = sum(t["usd"] for t in table)
    print(
        f"unreadable files {len(unreadable)}; records outside the plan {len(extra)}; "
        f"total spend ${total:.2f}"
    )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                _jsonable(
                    {"table": table, "unreadable": [str(p) for p in unreadable], "unplanned": extra}
                ),
                handle,
                indent=1,
            )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        with open(args.sets) as handle:
            reg = Registry.from_sets(json.load(handle))
        rows = load_rows(args.out)
    except (ValueError, OSError) as exc:
        return _abort([str(exc)])
    fatal = provenance_problems(rows, reg, code_digest=args.code_digest)
    if fatal:
        return _abort(fatal)
    plan_ids, no_plan = load_plan(args.out)
    comp = check_complete(
        rows, reg, unrun=load_unrun(args.out), plan_ids=None if no_plan else plan_ids
    )
    gaps = comp.problems() + ([f"no plan.json in {', '.join(no_plan)}"] if no_plan else [])
    if reg.flags:
        gaps.append(f"the sets file is not confirmatory (flags: {', '.join(reg.flags)})")
    if comp.unplanned:
        return _abort(
            [f"{len(comp.unplanned)} record(s) are not planned (e.g. {comp.unplanned[0]})"]
        )
    lost = {m: d for m, d in dropped_families(rows, reg).items() if d}
    gaps += [f"{m}: no valid pair for {', '.join(d)} (P x faulted)" for m, d in lost.items()]
    if gaps and not args.allow_missing:
        return _abort([*gaps, "rerun the jobs, or pass --allow-missing to analyse what is there"])
    result = analyze(
        rows, reg, dirs=args.out, flips=args.flips, draws=args.bootstraps, incomplete=bool(gaps)
    )
    result["warnings"] = [f"--allow-missing: {g}" for g in gaps]
    result["completeness"] = asdict(comp)
    print(render_report(result))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(_jsonable(result), handle, indent=1)
            handle.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="science_harness.bench.stage2", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sets = sub.add_parser("sets", help="bite set, strata and seed registry from the pilot")
    sets.add_argument("--pilot", nargs="+", required=True, help="pilot run directories")
    sets.add_argument("--replay", required=True, help="JSON of bench.admission.replay --json")
    sets.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help=f"families dropped before launch (default {' '.join(DEFAULT_EXCLUDED)})",
    )
    sets.add_argument(
        "--families", nargs="*", default=None, help="only these families (rehearsals)"
    )
    sets.add_argument(
        "--family-rules", default=None, help="family -> own rules (default: the packaged file)"
    )
    sets.add_argument(
        "--promotion", default=None, help="promotion table (default: the packaged one)"
    )
    sets.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a sets file from an incomplete pilot; it is flagged and analyze refuses it",
    )
    sets.add_argument(
        "--lenient",
        action="store_true",
        help="accept pilot records with provenance problems (a dry run); the sets file is flagged",
    )
    sets.add_argument("--out", required=True)
    sets.set_defaults(func=cmd_sets)
    health = sub.add_parser(
        "health", help="counts only: infra errors, absent trials, spend, providers"
    )
    health.add_argument("--out", nargs="+", required=True, help="run directories")
    health.add_argument("--sets", required=True)
    health.add_argument("--json", default=None)
    health.set_defaults(func=cmd_health)
    scan = sub.add_parser("scan", help="pre-launch: no registered seed may exist in another run")
    scan.add_argument("--sets", required=True)
    scan.add_argument("--runs", nargs="+", required=True, help="run directories to look through")
    scan.add_argument("--own", nargs="*", default=[], help="the stage-2 directories themselves")
    scan.set_defaults(func=cmd_scan)
    analyze_ = sub.add_parser("analyze", help="the registered contrasts on the stage-2 runs")
    analyze_.add_argument("--out", nargs="+", required=True, help="run directories")
    analyze_.add_argument("--sets", required=True, help="JSON written by `sets` (frozen)")
    analyze_.add_argument("--json", default=None, help="also write the numbers here")
    analyze_.add_argument(
        "--flips",
        type=int,
        default=FLIPS,
        help="Monte Carlo flips (only if the exact count is too large)",
    )
    analyze_.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    analyze_.add_argument(
        "--code-digest", default=None, help="the frozen code digest the records must carry"
    )
    analyze_.add_argument(
        "--allow-missing",
        action="store_true",
        help="analyse although planned trials are absent (printed)",
    )
    analyze_.set_defaults(func=cmd_analyze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
