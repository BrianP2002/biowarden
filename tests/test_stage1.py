"""The preregistered stage-1 analysis on synthetic trial records with a known effect."""

from __future__ import annotations

import random

from biowarden.bench import stage1

MODELS = ["a/m1", "b/m2", "c/m3", "d/m4", "claude-haiku-5-5"]


def record(model, arm, family, seed, twin, outcome, blocked=0, turns=6, correct=None):
    return {
        "model": model,
        "arm": arm,
        "family": family,
        "seed": seed,
        "rep": 0,
        "twin": twin,
        "outcome": outcome,
        "contract": "bare",
        "tier": "implicit",
        "route": "forced",
        "harness": "minimal",
        "turns": turns,
        "usd": 0.002,
        "correct": outcome == "pass" if correct is None else correct,
        "layer": {"actions": {"block": blocked}} if arm != "bare" else None,
    }


def world(bare_silent: float, block_silent: float, block_clean_fail: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for model in MODELS:
        for family in stage1.FAMILIES:
            for s in range(1000, 1020):
                bare_bad = rng.random() < bare_silent
                # paired: a trial that is silent under block is (mostly) silent under bare too
                block_bad = bare_bad and rng.random() < (block_silent / max(bare_silent, 1e-9))
                clean_ok = rng.random() >= block_clean_fail
                rows += [
                    record(model, "bare", family, s, "clean", "pass"),
                    record(
                        model, "bare", family, s, "faulted", "silent_wrong" if bare_bad else "pass"
                    ),
                    record(
                        model,
                        "block",
                        family,
                        s,
                        "clean",
                        "pass" if clean_ok else "protocol_fail",
                        blocked=1,
                    ),
                    record(
                        model,
                        "block",
                        family,
                        s,
                        "faulted",
                        "silent_wrong" if block_bad else "pass",
                        blocked=1,
                    ),
                ]
    return rows


def test_a_large_effect_meets_the_criterion_and_a_null_effect_does_not():
    strong = stage1.verdict(stage1.usable(world(0.4, 0.05), stage1.PRIMARY))
    null = stage1.verdict(stage1.usable(world(0.4, 0.4), stage1.PRIMARY))

    assert strong["effective"] is True and len(strong["models_meeting_drop"]) == 4
    assert null["effective"] is False and null["models_meeting_drop"] == []


def test_the_reference_model_never_counts_towards_the_criterion():
    verdict = stage1.verdict(stage1.usable(world(0.4, 0.05), stage1.PRIMARY))

    assert "claude-haiku-5-5" not in verdict["models_meeting_drop"]


def test_friction_that_costs_clean_passes_blocks_the_criterion():
    harmful = stage1.verdict(stage1.usable(world(0.4, 0.05, block_clean_fail=0.2), stage1.PRIMARY))

    assert harmful["effective"] is False and harmful["clean_pass_loss"] > stage1.MAX_PASS_LOSS


def test_holm_adjustment_is_monotone_and_capped():
    adjusted = stage1.holm({"a": 0.01, "b": 0.02, "c": 0.5})

    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}
    assert all(0 <= p <= 1 for p in adjusted.values())


def test_pairs_are_matched_by_family_and_seed_and_counted_both_ways():
    rows = [
        record("a/m", "bare", "F1", 1, "faulted", "silent_wrong"),
        record("a/m", "block", "F1", 1, "faulted", "pass"),
        record("a/m", "bare", "F1", 2, "faulted", "pass"),
        record("a/m", "block", "F1", 2, "faulted", "silent_wrong"),
        record("a/m", "bare", "F7", 1, "faulted", "silent_wrong"),
        record("a/m", "block", "F7", 1, "faulted", "protocol_fail"),  # undelivered: not silent
        record("a/m", "bare", "F7", 2, "faulted", "pass"),  # no partner in `block`
    ]

    (entry,) = stage1.silent_table(rows)

    assert (entry["pairs"], entry["bare_only"], entry["arm_only"]) == (3, 2, 1)
    assert entry["bare"] == (2, 3) and entry["arm_rate"] == (1, 3)


def test_side_effects_report_friction_and_delivery():
    rows = [
        record("a/m", "block", "F1", 1, "clean", "pass", blocked=2),
        record("a/m", "block", "F1", 2, "clean", "protocol_fail"),
        record("a/m", "block", "F1", 3, "faulted", "flagged_wrong"),
    ]

    (entry,) = stage1.side_effects(rows)

    assert entry["clean_pass"] == (1, 2) and entry["clean_blocked"] == (1, 2)
    assert abs(entry["delivered"] - 2 / 3) < 1e-9 and abs(entry["flagged_wrong"] - 1 / 3) < 1e-9


def test_the_pooled_bootstrap_is_reproducible_and_positive_for_a_real_effect():
    primary = stage1.usable(world(0.4, 0.05), stage1.PRIMARY)

    first, second = stage1.pooled_drop(primary, "block"), stage1.pooled_drop(primary, "block")

    assert first == second and first["ci"][0] > 0.2


def test_the_report_renders_every_section():
    text = stage1.render(world(0.4, 0.05))

    for part in ("PRIMARY CELL", "SIDE EFFECTS", "POOLED", "PREREGISTERED CRITERION"):
        assert part in text
