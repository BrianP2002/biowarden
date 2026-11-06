"""The stage-2 analysis on synthetic trial records: the exact sign-flip test, pairing, the seed
registry, planned versus recorded, the per-model decision rule, the worst case, the command line.
The bite set and the strata are in test_stage2_strata.py."""

from __future__ import annotations

import itertools
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from biowarden.bench import stage2
from biowarden.bench.runner import cli as runner_cli
from biowarden.bench.runner.trial import MAX_ATTEMPTS, TrialSpec

M1, M2, M3 = stage2.MODELS
PRIMARY = stage2.PRIMARY_CELL
DEFAULT = stage2.DEFAULT_CELL
FLIPS, DRAWS = 2000, 200  # small sizes: the tests check logic, not precision
P_SET = [
    f"F{n}" for n in (29, 30, 31, 32, 33, 35, 36, 37, 38, 39)
]  # ten families in P (F34 is out)
Q1, Q2, Q3 = ["F41", "F42"], ["F43", "F44"], ["F45", "F46"]
PROVENANCE = {
    "scorer": 3,
    "code_digest": "d" * 64,
    "run_config_sha": "c" * 64,
    "tool_manifest_sha": "t" * 64,
}
SERVED_BY = {M2: ["Wafer"], M3: ["Near AI"]}


def rec(model, family, seed, twin, arm, outcome, cell=PRIMARY, rep=0, harness="minimal", **more):
    """One score.json as the runner writes it (the id comes from the runner's own TrialSpec)."""
    contract, tier, route = cell
    spec = TrialSpec(model, family, seed, twin, rep, tier, route, contract, harness, arm)
    row = {
        "trial": spec.trial_id,
        "model": model,
        "family": family,
        "seed": seed,
        "twin": twin,
        "arm": arm,
        "outcome": outcome,
        "rep": rep,
        "harness": harness,
        "contract": contract,
        "tier": tier,
        "route": route,
        "delivered": outcome not in stage2.NOT_DELIVERED,
        "correct": outcome == "pass",
        "turns": 7,
        "usd": 0.01,
        **PROVENANCE,
    }
    if model in stage2.PIN_TAGS:
        row |= {
            "provider_pin": [stage2.PIN_TAGS[model]],
            "providers": SERVED_BY[model],
            "pin_match": True,
        }
    if arm in stage2.LAYER_ARMS:
        row["layer"] = {
            "calls": 3,
            "actions": {"allow": 3, "warn": 0, "block": 0},
            "stops": 0,
            "errors": [],
            "ledger_errors": 0,
            "crashed": 0,
            "touched_layer": 0,
            "fired": [],
        }
    return row | more


def pair(model, family, seed, twin, bare, layer, arm="block-nonote", **kw):
    return [
        rec(model, family, seed, twin, "bare", bare, **kw),
        rec(model, family, seed, twin, arm, layer, **kw),
    ]


def registry(**more):
    return stage2.Registry.make(P_SET, Q1, Q2, Q3, **more)


def behave(bare_silent=1.0, layer_silent=0.0, layer_undelivered=0.0, clean_layer="pass"):
    """Outcomes by seed index: bare is silent on the first `bare_silent` share of the five seeds;
    the layer leaves the first `layer_silent` of them silent, makes the next `layer_undelivered`
    share protocol_fail and passes the rest. The clean twin passes in bare and `clean_layer`."""

    def outcome(model, family, seed, twin, arm, cell):
        k = seed % 10  # 1000-1004 and 1020-1024 -> 0-4 (the default cell uses 0-2)
        if twin == "clean":
            return "pass" if arm == "bare" else clean_layer
        n_bare = round(5 * bare_silent)
        if arm == "bare":
            return "silent_wrong" if k < n_bare else "pass"
        n_silent, n_undel = round(5 * layer_silent), round(5 * layer_undelivered)
        if k < n_silent:
            return "silent_wrong"
        return "protocol_fail" if k < n_silent + n_undel else "pass"

    return outcome


def campaign(reg, outcome=None, models=stage2.MODELS):
    """Every planned trial of the registry (all five jobs of each model) with these outcomes."""
    outcome = outcome or behave()
    rows = []
    for model in models:
        for arm, cell in stage2.CAMPAIGN:
            for family in reg.families_for(arm, cell):
                for seed in reg.seeds_for(cell)[family]:
                    for twin in ("clean", "faulted"):
                        rows.append(
                            rec(
                                model,
                                family,
                                seed,
                                twin,
                                arm,
                                outcome(model, family, seed, twin, arm, cell),
                                cell,
                            )
                        )
    return rows


def decide(rows, reg=None, **kw):
    return stage2.decide(rows, reg or registry(), flips=FLIPS, draws=DRAWS, **kw)


# --- the test and the intervals ------------------------------------------------------------


def brute_force(values, two_sided=False):
    nonzero = [Fraction(v) for v in values if v]
    observed = sum(nonzero)
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(nonzero)):
        s = sum(sign * v for sign, v in zip(signs, nonzero, strict=True))
        hits += abs(s) >= abs(observed) if two_sided else s >= observed
    return hits / 2 ** len(nonzero) if nonzero else 1.0


def test_sign_flip_is_exact_for_few_families():
    assert stage2.sign_flip_p([1, 1, 1, 1]) == pytest.approx(1 / 16)
    assert stage2.sign_flip_p([1.0, -1.0]) == pytest.approx(3 / 4)
    assert stage2.sign_flip_p([0.0, 0.0]) == 1.0 and stage2.sign_flip_p([]) == 1.0
    assert stage2.sign_flip_p([0.0, 1.0, 1.0, 1.0]) == pytest.approx(1 / 8)  # zeros are dropped


def test_the_worked_example_of_the_docstring():
    d = [Fraction(1, 5), Fraction(2, 5), Fraction(-1, 5)]
    assert stage2.sign_flip_p(d) == pytest.approx(3 / 8)
    assert stage2.sign_flip_p(d, sided="two-sided") == pytest.approx(6 / 8)
    assert stage2.sign_flip_p(d, alternative="two-sided") == pytest.approx(6 / 8)
    with pytest.raises(ValueError):
        stage2.sign_flip_p(d, sided="less")


def test_the_exact_count_equals_brute_force_with_zeros_and_ties():
    rng = random.Random(11)
    pool = [Fraction(k, 60) for k in (-12, -6, -3, -1, 0, 0, 1, 3, 3, 6, 12, 20)]
    for _ in range(300):
        d = [rng.choice(pool) for _ in range(rng.randint(1, 11))]
        assert stage2.sign_flip_p(d) == pytest.approx(brute_force(d), abs=1e-12)
        assert stage2.sign_flip_p(d, sided="two-sided") == pytest.approx(
            brute_force(d, two_sided=True), abs=1e-12
        )


def test_the_p_value_does_not_depend_on_record_order_or_on_float_noise():
    d = [0.2, 0.4, -0.2, 0.6, 0.0, 0.2, 1.0 / 3, 2.0 / 3]
    shuffled = d[:]
    random.Random(1).shuffle(shuffled)
    assert stage2.sign_flip_p(d) == stage2.sign_flip_p(shuffled)
    noisy = [0.1 + 0.1 + 0.1 - 0.1, 0.30000000000000004, 0.6000000000000001]  # 0.2, 0.3, 0.6
    assert stage2.sign_flip_p(noisy) == stage2.sign_flip_p([0.2, 0.3, 0.6])


def test_monte_carlo_fallback_agrees_with_the_exact_count():
    rng = random.Random(7)
    d = [rng.choice([-0.4, -0.2, 0.0, 0.2, 0.2, 0.4, 0.6]) for _ in range(18)]
    exact = stage2.sign_flip_p(d)
    assert stage2.sign_flip_p(d, flips=20000, exact_work=0) == pytest.approx(exact, abs=0.02)
    again = stage2.sign_flip_p(d, flips=500, exact_work=0)
    assert again == stage2.sign_flip_p(d, flips=500, exact_work=0)  # fixed generator


def test_monte_carlo_p_never_reaches_zero():
    assert stage2.sign_flip_p([1.0] * 30, flips=999, exact_work=0) == pytest.approx(1 / 1000)
    assert 0 < stage2.sign_flip_p([1.0] * 30) < 1e-8  # the exact count has no floor but is not 0


def test_the_test_is_not_anticonservative_under_the_null():
    rng = random.Random(3)
    rejected = 0
    for _ in range(200):
        d = []
        for _family in range(30):
            q = rng.random()
            diffs = [(rng.random() < q) - (rng.random() < q) for _seed in range(5)]
            d.append(Fraction(sum(diffs), 5))
        rejected += stage2.sign_flip_p(d) < 0.05
    assert rejected / 200 < 0.10


def test_holm_is_monotone_capped_and_keeps_a_missing_hypothesis_in_the_family():
    adjusted = stage2.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert stage2.holm({"a": 0.6, "b": 0.7})["b"] == 1.0
    two = stage2.holm({"a": 0.00005, "b": 0.5, "c": None})  # c has no data: p = 1, still counted
    assert two["a"] == pytest.approx(1.5e-4) and two["c"] == 1.0
    assert stage2.holm({"a": 0.00005, "b": 0.5}, m=3)["a"] == pytest.approx(1.5e-4)  # fixed size 3


def test_bootstrap_interval_brackets_the_mean_and_is_nan_without_data():
    low, high = stage2.bootstrap_ci([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5, draws=500)
    assert low < 0.5 < high and high - low < 0.4
    assert all(x != x for x in stage2.bootstrap_ci([]))
    assert all(x != x for x in stage2.bootstrap_ci([1.0], draws=0))
    assert stage2.bootstrap_ci([0.0] * 6, draws=100) == (0.0, 0.0)  # all zero: [0, 0]


def test_bootstrap_difference_of_two_groups():
    est, low, high = stage2.bootstrap_difference([1.0] * 6, [0.0] * 6, draws=200)
    assert est == 1.0 and low == 1.0 and high == 1.0
    assert all(x != x for x in stage2.bootstrap_difference([], [0.0]))


# --- pairing ---------------------------------------------------------------------------------


def diffs(rows, **kw):
    args = dict(
        model=M1, arm="block-nonote", metric=stage2.silent, twins=("faulted",), cell=PRIMARY
    )
    args["families"] = kw.pop("families", ["F1"])
    return stage2.paired_differences(rows, **(args | kw))


def test_a_pair_needs_both_arms_and_infra_errors_drop_the_partner():
    rows = (
        pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass")
        + pair(M1, "F1", 1021, "faulted", "silent_wrong", "infra_error")
        + pair(M1, "F1", 1022, "faulted", "infra_error", "pass")
        + [rec(M1, "F1", 1023, "faulted", "bare", "silent_wrong")]  # no layer partner at all
    )
    assert diffs(rows) == {"F1": (1.0, 1)}


def test_cells_models_twins_harnesses_and_other_arms_are_not_mixed_in():
    rows = (
        pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass", cell=DEFAULT)
        + pair(M2, "F1", 1020, "faulted", "silent_wrong", "pass")
        + pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass", harness="opencode")
        + pair(M1, "F1", 1020, "clean", "silent_wrong", "pass")
        + pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass", arm="warn-nonote")
    )
    assert diffs(rows) == {}


def test_families_weigh_equally_whatever_their_number_of_seeds():
    rows = []
    for seed in range(1020, 1025):  # F1: five seeds, all lowered
        rows += pair(M1, "F1", seed, "faulted", "silent_wrong", "pass")
    rows += pair(M1, "F2", 1000, "faulted", "pass", "pass")  # F2: one seed, no change
    result = stage2.contrast(
        rows, M1, "block-nonote", "silent", ("faulted",), PRIMARY, ["F1", "F2"]
    )
    assert result.estimate == pytest.approx(0.5) and result.pairs == 6 and result.families == 2
    assert result.pair_weighted == pytest.approx(5 / 6) and result.thin == ("F2",)


def test_a_duplicate_trial_record_is_an_error():
    rows = pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass") + [
        rec(M1, "F1", 1020, "faulted", "bare", "pass")
    ]
    with pytest.raises(ValueError, match="two records"):
        diffs(rows)


def test_refusals_and_protocol_failures_count_as_not_silent_and_not_delivered():
    rows = pair(M1, "F1", 1020, "faulted", "silent_wrong", "refused") + pair(
        M1, "F1", 1021, "faulted", "silent_wrong", "protocol_fail"
    )
    assert diffs(rows) == {"F1": (1.0, 2)}
    assert diffs(rows, metric=stage2.delivered) == {"F1": (1.0, 2)}  # answers lost, not errors


def test_the_pass_gain_is_the_drop_of_not_passing():
    rows = pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass") + pair(
        M1, "F1", 1021, "faulted", "silent_wrong", "protocol_fail"
    )
    assert diffs(rows, metric="silent") == {"F1": (1.0, 2)}  # both silent errors are gone
    assert diffs(rows, metric="notpass") == {"F1": (0.5, 2)}  # but only one became a pass


def test_a_lost_trial_counts_against_the_layer_in_the_worst_case():
    rows = (
        pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass")
        + pair(M1, "F1", 1021, "faulted", "silent_wrong", "infra_error")  # layer trial lost
        + pair(M1, "F1", 1022, "faulted", "infra_error", "pass")  # bare trial lost
    )
    keys = [("F1", s, "faulted") for s in (1020, 1021, 1022, 1023)]  # 1023 never recorded
    pairs = stage2.pair_trials(
        rows,
        model=M1,
        arm="block-nonote",
        cell=PRIMARY,
        twins=("faulted",),
        families=["F1"],
        expected=keys,
    )
    assert len(pairs) == 4
    assert stage2.family_differences(pairs, "silent") == {"F1": (1.0, 1)}  # lost pairs dropped
    # worst: pair 1 d=+1; layer lost -> it counts silent: 1-1=0; bare lost, layer pass: 0-0=0;
    # both lost: 0-1=-1
    assert stage2.family_differences(pairs, "silent", worst=True) == {"F1": (0.0, 4)}
    # for a loss the lost bare trial passed and the lost layer trial did not
    lost = pair(M1, "F1", 1020, "clean", "pass", "infra_error") + pair(
        M1, "F1", 1021, "clean", "infra_error", "pass"
    )
    more = stage2.pair_trials(
        lost,
        model=M1,
        arm="block-nonote",
        cell=PRIMARY,
        twins=("clean",),
        families=["F1"],
        expected=[("F1", 1020, "clean"), ("F1", 1021, "clean")],
    )
    assert stage2.family_differences(more, "pass", worst=True) == {"F1": (0.5, 2)}  # +1 and 0


def test_the_bare_replicate_is_paired_with_the_first_bare_run():
    rows = [
        rec(M1, "F1", 1020, "faulted", "bare", "silent_wrong"),
        rec(M1, "F1", 1020, "faulted", "bare", "pass", rep=1),
    ]
    c = stage2.contrast(rows, M1, "bare", "silent", ("faulted",), PRIMARY, ["F1"], arm_rep=1)
    assert c.estimate == 1.0 and c.pairs == 1
    with pytest.raises(ValueError, match="same trials"):  # bare against itself
        stage2.contrast(rows, M1, "bare", "silent", ("faulted",), PRIMARY, ["F1"])


def test_a_family_without_a_valid_pair_is_dropped_and_listed_not_set_to_zero():
    rows = pair(M1, "F1", 1020, "faulted", "silent_wrong", "pass") + pair(
        M1, "F2", 1000, "faulted", "silent_wrong", "infra_error"
    )
    c = stage2.contrast(rows, M1, "block-nonote", "silent", ("faulted",), PRIMARY, ["F1", "F2"])
    assert (c.expected_families, c.families, c.dropped) == (2, 1, ("F2",)) and c.estimate == 1.0


def test_all_zero_differences_give_p_one_and_a_zero_interval():
    rows = []
    for family in P_SET[:6]:
        rows += pair(M1, family, 1000, "faulted", "pass", "pass")
    c = stage2.contrast(
        rows,
        M1,
        "block-nonote",
        "silent",
        ("faulted",),
        PRIMARY,
        P_SET[:6],
        "x",
        "greater",
        FLIPS,
        100,
    )
    assert (c.estimate, c.p, c.low, c.high) == (0.0, 1.0, 0.0, 0.0)


# --- loading and provenance -------------------------------------------------------------------


def write_run(root, rows):
    for r in rows:
        directory = root / "trials" / r["trial"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "score.json").write_text(json.dumps(r))


def test_load_rows_sorts_and_refuses_truncated_files_duplicates_and_other_scorers(tmp_path):
    rows = pair(M1, "F30", 1001, "faulted", "pass", "pass") + pair(
        M1, "F9", 1000, "clean", "pass", "pass"
    )
    write_run(tmp_path / "a", rows)
    loaded = stage2.load_rows([tmp_path / "a"])
    assert [(r["arm"], r["family"]) for r in loaded] == [
        ("bare", "F9"),
        ("bare", "F30"),
        ("block-nonote", "F9"),
        ("block-nonote", "F30"),
    ]  # by family number, not by file name or glob order

    write_run(tmp_path / "b", rows[:1])  # the same trial in a second directory
    with pytest.raises(ValueError, match="two records"):
        stage2.load_rows([tmp_path / "a", tmp_path / "b"])

    victim = next((tmp_path / "a" / "trials").glob("*/*/score.json"))
    victim.write_text(victim.read_text()[:30])
    with pytest.raises(ValueError, match="unreadable"):
        stage2.load_rows([tmp_path / "a"])

    write_run(tmp_path / "c", [rec(M1, "F30", 1000, "faulted", "bare", "pass", scorer=2)])
    with pytest.raises(ValueError, match="scorer"):
        stage2.load_rows([tmp_path / "c"])


def test_clean_records_have_no_provenance_problems():
    reg = registry()
    assert stage2.provenance_problems(campaign(reg), reg) == []


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"model": "some/other-model"}, "not one of the three"),
        ({"arm": "block-note"}, "not registered"),
        ({"harness": "opencode"}, "harness"),
        ({"seed": 5}, "seed registry"),
        ({"rep": 2}, "repetition"),
        ({"scorer": 2}, "scorer"),
        ({"contract": "full", "tier": "implicit"}, "neither the primary"),
        ({"provider_pin": None}, "provider pin"),
        ({"providers": ["Other"]}, "other than the pinned"),
        ({"pin_match": False}, "pin_match"),
        ({"layer": None}, "layer summary"),
        ({"code_digest": "e" * 64}, "different code_digest"),
        ({"run_config_sha": None}, "without run_config_sha"),
    ],
)
def test_each_provenance_problem_is_named(change, text):
    reg = registry()
    rows = campaign(reg, models=(M2,))
    target = next(r for r in rows if r["arm"] == "block-nonote" and r["family"] == "F30")
    target.update(change)
    if "model" in change:
        target["trial"] = "other/x"
    assert any(text in p for p in stage2.provenance_problems(rows, reg)), text


def test_a_seen_family_on_an_unregistered_seed_block_is_refused():
    reg = registry()
    bad = rec(M1, "F30", 1020, "faulted", "bare", "pass")  # F30 is a fresh family: 1000-1004
    assert any("seed registry" in p for p in stage2.provenance_problems([bad], reg))
    with pytest.raises(ValueError, match="1020"):
        stage2.Registry.make(["F1"], seeds={"F1": (1000, 1001, 1002, 1003, 1004)})


def test_the_replicate_is_allowed_only_for_bare_on_p():
    reg = registry()
    good = rec(M1, "F30", 1000, "faulted", "bare", "pass", rep=1)
    outside = rec(M1, "F41", 1000, "faulted", "bare", "pass", rep=1)  # a Q1 family
    layered = rec(M1, "F30", 1000, "faulted", "block-nonote", "pass", rep=1)
    assert not any("repetition" in p for p in stage2.provenance_problems([good], reg))
    for bad in (outside, layered):
        assert any("repetition" in p for p in stage2.provenance_problems([bad], reg))


# --- the seed registry and the planned trials -----------------------------------------------------


def test_the_seen_families_use_the_unseen_seed_block():
    seen = {"F1", "F7", "F10", "F13", "F15", *(f"F{n}" for n in range(17, 29))}
    assert set(stage2.SEEN_FAMILIES) == seen and len(seen) == 17
    assert stage2.registry_seeds("F22") == (1020, 1021, 1022, 1023, 1024)
    assert stage2.registry_seeds("F30") == (1000, 1001, 1002, 1003, 1004)
    reg = stage2.Registry.make(["F1", "F30"], default_only=("F4",))
    assert reg.default_seeds["F1"] == (1020, 1021, 1022) and reg.default_seeds["F30"] == (
        1000,
        1001,
        1002,
    )
    assert reg.default_seeds["F4"] == (1000, 1001, 1002) and "F4" not in reg.gated
    assert "F4" in reg.default_gated


def test_the_sets_file_must_carry_a_valid_registry():
    with pytest.raises(ValueError, match="seed registry"):
        stage2.Registry.from_sets({"primary": ["F30"], "complement": []})
    with pytest.raises(ValueError, match="both P and Q1"):
        stage2.Registry.make(["F30"], ["F30"])
    reg = stage2.Registry.make(["F30"], ["F41"], ["F43"], ["F45"], excluded=("F34",))
    assert reg.complement == ("F41", "F43", "F45") and reg.bite == ("F30", "F41")
    assert reg.bite_unreachable == ("F41",) and reg.q_other == ("F43", "F45")
    assert reg.families_for("warn-nonote", PRIMARY) == reg.bite
    assert reg.families_for("warn-nonote", DEFAULT) == ()


def expected_ids(reg, model, arm, cell):
    contract, tier, route = cell
    keys = reg.pair_keys(reg.families_for(arm, cell), cell, ("clean", "faulted"))
    return sorted(
        TrialSpec(model, f, s, t, 0, tier, route, contract, "minimal", arm).trial_id
        for f, s, t in keys
    )


def test_planned_calls_the_runners_plan_and_agrees_with_the_registry():
    reg = stage2.Registry.make(["F1", "F30"], ["F41"], ["F22"], ["F45"])
    for arm, cell in stage2.CAMPAIGN:
        ids = stage2.planned(reg, M2, arm, cell)
        assert ids == expected_ids(reg, M2, arm, cell) and len(set(ids)) == len(ids)
    primary_ids = stage2.planned(reg, M1, "bare", PRIMARY)
    assert any("F1-s1020-" in i for i in primary_ids) and any(
        "F30-s1000-" in i for i in primary_ids
    )
    assert not any("F4-" in i for i in primary_ids)  # F4 is a default-cell family
    assert any("F4-s1000-clean-r0" in i for i in stage2.planned(reg, M1, "bare", DEFAULT))
    assert stage2.planned(reg, M1, "warn-nonote", DEFAULT) == []
    assert stage2.planned_ids(reg, model=M1, arm="bare", cell=PRIMARY) == set(primary_ids)


def test_planned_reproduces_the_plan_json_the_runner_writes(tmp_path):
    reg = registry()
    specs = list(
        runner_cli.plan(
            [M1],
            reg.bite,
            [1000, 1001, 1002, 1003, 1004],
            1,
            "implicit",
            "forced",
            "bare",
            "minimal",
            "warn-nonote",
        )
    )
    runner_cli.update_plan(
        tmp_path, specs, {"job_id": "j", "started": "t", "args": {}, "code_digest": "x"}
    )
    ids, missing = stage2.load_plan([tmp_path])
    assert missing == [] and set(ids) == set(stage2.planned(reg, M1, "warn-nonote", PRIMARY))


def test_plan_ids_are_the_runners_ids_for_both_seed_blocks_in_one_call():
    reg = stage2.Registry.make(["F1"], ["F30"])
    ids = stage2.planned(reg, M3, "block-nonote", PRIMARY)
    assert len(ids) == 2 * 5 * 2
    assert ids[0].startswith(
        "openai__gpt-oss-120b/F1-s1020-clean-implicit-forced-bare-block-nonote-r0"
    )


# --- planned versus recorded ----------------------------------------------------------------------


def test_a_complete_campaign_has_nothing_absent():
    reg = registry()
    comp = stage2.check_complete(campaign(reg), reg)
    assert comp.absent == [] and comp.unplanned == [] and comp.problems() == []
    assert comp.planned == comp.recorded == comp.counted > 1000


def test_absent_trials_open_infra_errors_and_unplanned_records_are_listed():
    reg = registry()
    rows = campaign(reg)
    gone = rows.pop(5)  # a trial that has no record at all
    final = next(r for r in rows if r["arm"] == "block-nonote")
    final.update(outcome="infra_error", attempt=MAX_ATTEMPTS)
    open_ = next(r for r in rows if r["arm"] == "bare" and r["trial"] != final["trial"])
    open_.update(outcome="infra_error", attempt=1)
    rows.append(rec(M1, "F30", 1000, "faulted", "bare", "pass", cell=("bare", "explicit", "free")))
    comp = stage2.check_complete(rows, reg, unrun={gone["trial"]: "skipped_budget"})
    assert comp.absent == [gone["trial"]] and comp.unrun == [gone["trial"]]
    assert comp.infra_final == [final["trial"]] and comp.infra_open == [open_["trial"]]
    assert len(comp.unplanned) == 1
    assert len(comp.problems()) == 2  # absent, and the open infra_error
    assert comp.recorded == comp.planned - 1


def test_the_replicate_is_not_an_unplanned_record():
    reg = registry()
    rows = campaign(reg) + [rec(M1, "F30", 1000, "faulted", "bare", "pass", rep=1)]
    assert stage2.check_complete(rows, reg).unplanned == []


def test_missingness_table_flags_a_gap_of_more_than_two_points():
    reg = registry()
    rows = campaign(reg)
    layer = [
        r
        for r in rows
        if r["model"] == M1 and r["arm"] == "block-nonote" and stage2.cell_of(r) == PRIMARY
    ]
    for r in layer[:4]:  # 4 of 160: 2.5 points
        r["outcome"] = "infra_error"
    table = stage2.missingness(rows, reg)
    mine = [t for t in table if t["model"] == M1]
    assert [t["lost"] for t in mine] == [0, 4] and mine[1]["lost_share"] == pytest.approx(4 / 160)
    assert all(t["binding"] for t in mine) and not any(
        t["binding"] for t in table if t["model"] == M2
    )
    layer[0]["outcome"] = "pass"  # 3 of 160: 1.9 points
    assert not any(t["binding"] for t in stage2.missingness(rows, reg) if t["model"] == M1)
    assert all(t["refused"] == 0 for t in table)


# --- the registered decision ---------------------------------------------------------------------


def test_a_layer_that_turns_every_silent_error_into_a_pass_is_effective():
    result = decide(campaign(registry()))
    assert result.effective and len(result.models_passing) == 3 and result.claim
    for d in result.models:
        assert d.drop.estimate == pytest.approx(1.0) and d.gain.estimate == pytest.approx(1.0)
        assert d.p_holm < 0.05 and d.delivered_loss.estimate == 0 and d.passes
        assert d.p == max(d.drop.p, d.gain.p)
    assert result.n_primary == 10 and result.n_safety == 16
    assert result.mean_clean_pass_loss == 0 and result.mean_delivered_loss == 0


def test_a_layer_that_changes_nothing_is_not_shown():
    result = decide(campaign(registry(), behave(bare_silent=0.6, layer_silent=0.6)))
    assert not result.effective and result.models_passing == ()
    assert all(abs(d.drop.estimate) < 1e-9 for d in result.models)


def test_trading_silent_errors_for_undelivered_answers_is_not_a_win():
    """The review's case: the layer makes the agent stop answering. Silent drops by 100 points,
    nothing is gained in passes and the delivered share collapses."""
    result = decide(campaign(registry(), behave(layer_undelivered=1.0)))
    for d in result.models:
        assert d.drop.estimate == pytest.approx(1.0)
        assert d.gain.estimate == 0 and d.delivered_loss.estimate == pytest.approx(1.0)
        assert not d.checks["pass_gain"] and not d.checks["delivered_loss"] and not d.passes
    assert not result.effective


def test_half_the_silent_errors_becoming_undeliverable_still_fails_the_delivered_guard():
    # bare silent on all 5; the layer passes 3 and loses 2 answers:
    # drop 100, gain 60, delivered loss 40
    result = decide(campaign(registry(), behave(layer_undelivered=0.4)))
    d = result.models[0]
    assert d.drop.estimate == pytest.approx(1.0) and d.gain.estimate == pytest.approx(0.6)
    assert d.delivered_loss.estimate == pytest.approx(0.4) and not d.checks["delivered_loss"]
    assert not result.effective


def test_two_of_three_models_are_enough_and_one_is_not():
    reg = registry()
    good = behave()
    flat = behave(bare_silent=0.6, layer_silent=0.6)

    def mixed(winners):
        return lambda model, *a: (good if model in winners else flat)(model, *a)

    assert decide(campaign(reg, mixed({M1, M2}))).effective
    one = decide(campaign(reg, mixed({M1})))
    assert not one.effective and one.models_passing == (M1,)


def test_a_big_drop_is_not_enough_when_clean_tasks_get_worse():
    reg = registry()
    rows = campaign(reg)
    for r in rows:
        if r["twin"] == "clean" and r["arm"] == "block-nonote" and r["seed"] % 10 == 0:
            r["outcome"] = "silent_wrong"  # one clean seed in five now fails: 20 points lost
    result = decide(rows)
    assert result.mean_clean_pass_loss == pytest.approx(0.2)
    assert not result.effective and all(not d.checks["clean_loss_gated"] for d in result.models)


def test_the_clean_guard_looks_at_p_and_at_every_gated_family_separately():
    reg = registry()
    rows = campaign(reg)
    for r in rows:  # the clean twin is hurt on a family outside P only
        if r["twin"] == "clean" and r["arm"] == "block-nonote" and r["family"] in Q3:
            r["outcome"] = "protocol_fail"
    d = decide(rows).models[0]
    assert d.clean_loss_primary.estimate == 0 and d.clean_loss_gated.estimate == pytest.approx(
        2 / 16
    )
    assert d.checks["clean_loss_primary"] and not d.checks["clean_loss_gated"] and not d.passes


def test_a_modest_but_real_drop_is_enough_and_a_flat_layer_is_not():
    reg = registry()
    assert decide(campaign(reg, behave(bare_silent=0.6, layer_silent=0.4))).effective  # 20 points
    assert not decide(campaign(reg, behave(bare_silent=0.2, layer_silent=0.2))).effective


MANY = [f"F{n}" for n in (*range(29, 34), *range(35, 40), *range(41, 47), *range(53, 67))]  # thirty


def drop_in(n_families):
    """Bare is silent on one seed in five for the first n families and the layer removes it."""
    quiet, loud = behave(bare_silent=0.0), behave(bare_silent=0.2)

    def outcome(model, family, seed, twin, arm, cell):
        use = loud if family in MANY[:n_families] else quiet
        return use(model, family, seed, twin, arm, cell)

    return outcome


def test_ten_points_is_the_bar_and_a_tiny_p_does_not_replace_it():
    reg = stage2.Registry.make(MANY)
    exactly = decide(campaign(reg, drop_in(15)), reg).models[
        0
    ]  # 15 families by 20 points: 10 points
    assert exactly.drop.estimate == pytest.approx(0.10) and exactly.checks["silent_drop"]
    assert exactly.p_holm < 0.05 and exactly.passes
    short = decide(campaign(reg, drop_in(8)), reg).models[0]  # 8 by 20 points: 5.3 points
    assert short.drop.estimate == pytest.approx(8 * 0.2 / 30)
    assert short.p_holm < 0.05 and not short.checks["silent_drop"] and not short.passes


def test_few_families_cannot_reach_a_holm_p_below_five_percent():
    # five families by 20 points is exactly 10 points, but five sign flips cannot give p < 0.017
    reg = registry()
    one = decide(campaign(reg, drop_in_p(5)), reg).models[0]
    assert one.drop.estimate == pytest.approx(0.10) and one.checks["silent_drop"]
    assert one.p == pytest.approx(1 / 32) and one.p_holm > 0.05 and not one.passes


def drop_in_p(n_families):
    quiet, loud = behave(bare_silent=0.0), behave(bare_silent=0.2)

    def outcome(model, family, seed, twin, arm, cell):
        use = loud if family in P_SET[:n_families] else quiet
        return use(model, family, seed, twin, arm, cell)

    return outcome


def test_the_pass_gain_needs_its_own_p_value_and_the_larger_one_counts():
    """Silent errors vanish in every family, but only in half of them into passes (the rest become
    declines): the drop has p = 2^-10, the gain 2^-5. A model passes only if both are small."""
    quiet = behave(bare_silent=0.2, layer_silent=0.0)

    def declines_in_half(model, family, seed, twin, arm, cell):
        outcome = quiet(model, family, seed, twin, arm, cell)
        if arm == "block-nonote" and twin == "faulted" and family in P_SET[5:] and seed % 10 == 0:
            return "flagged_wrong"  # a decline: delivered, not silent, not a pass
        return outcome

    d = decide(campaign(registry(), declines_in_half)).models[0]
    assert d.drop.estimate == pytest.approx(0.2) and d.gain.estimate == pytest.approx(0.1)
    assert d.delivered_loss.estimate == 0 and d.checks["silent_drop"] and d.checks["pass_gain"]
    assert d.drop.p == pytest.approx(2**-10) and d.gain.p == pytest.approx(2**-5)
    assert d.p == pytest.approx(2**-5) and d.p_holm == pytest.approx(3 * 2**-5) and not d.passes


def test_the_safety_losses_are_per_model_not_averaged():
    reg = registry()
    good, hurt = behave(), behave(clean_layer="protocol_fail")

    def mixed(model, *a):
        return (hurt if model == M3 else good)(model, *a)

    result = decide(campaign(reg, mixed))
    assert result.models_passing == (M1, M2) and result.effective  # M3 fails its own guard
    assert result.mean_clean_pass_loss == pytest.approx(1 / 3)  # the mean would have hidden it


def test_a_model_without_data_has_p_one_and_the_holm_family_stays_three():
    reg = registry()
    rows = campaign(reg, models=(M1, M2))
    result = decide(rows)
    missing = result.models[2]
    assert missing.p is None and missing.p_holm == 1.0 and not missing.passes
    assert missing.drop.estimate != missing.drop.estimate  # NaN fails every condition
    assert not any(missing.checks.values())
    complete = decide(campaign(reg))
    # the same p-value with three models in the family: Holm multiplies the smallest p by 3
    assert result.models[0].p_holm == pytest.approx(complete.models[0].p_holm)
    assert result.models_passing == (M1, M2) and result.effective


def test_the_primary_set_and_the_safety_set_are_separate():
    reg = registry()
    result = decide(campaign(reg))
    d = result.models[0]
    assert d.drop.families == 10 and d.clean_loss_gated.families == 16
    assert decide(campaign(reg), primary_set=P_SET[:4]).models[0].drop.families == 4


def test_the_warn_arm_has_its_own_holm_family_and_no_claim():
    reg = registry()
    rows = campaign(reg)
    result = stage2.warn_vs_bare(rows, reg, flips=FLIPS, draws=DRAWS)
    assert not result.claim and result.arm == "warn-nonote"
    assert result.n_safety == len(reg.bite)  # the guards look where the warn arm ran
    assert result.models[0].drop.estimate == pytest.approx(1.0)
    on_bite = stage2.warn_bite(rows, reg, flips=FLIPS, draws=DRAWS)
    assert on_bite[M1].families == 12 and on_bite[M1].p is None  # P and Q1, no test
    assert stage2.warn_bite([r for r in rows if r["arm"] != "warn-nonote"], reg) is None


def test_the_worst_case_flips_a_verdict_that_rests_on_few_trials():
    reg = registry()
    rows = campaign(
        reg, behave(bare_silent=0.4)
    )  # silent on 2 seeds of 5, all passes with the layer
    assert stage2.sensitivity_missing(rows, reg, flips=FLIPS, draws=0)["effective"]  # nothing lost
    for r in rows:  # the layer trials of the three seeds where bare was right are lost
        if r["arm"] == "block-nonote" and r["twin"] == "faulted" and r["seed"] % 10 >= 2:
            r["outcome"] = "infra_error"
    normal = decide(rows)
    assert normal.effective and normal.models[0].drop.estimate == pytest.approx(1.0)
    worst = stage2.sensitivity_missing(rows, reg, flips=FLIPS, draws=0)
    assert not worst[
        "effective"
    ]  # each lost pair counts silent against a bare trial that was right
    assert (
        worst["primary"].models[0].drop.estimate == pytest.approx(-0.2)
        and worst["primary"].worst_case
    )
    binding = any(t["binding"] for t in stage2.missingness(rows, reg))
    assert binding  # 30 lost layer trials in 160 against none in bare
    text = stage2.final_verdict(normal, worst["primary"], binding, False, False)
    assert text.startswith("NOT SHOWN") and "worst case" in text
    assert stage2.final_verdict(normal, worst["primary"], False, False, False) == "EFFECTIVE"
    assert stage2.final_verdict(normal, None, False, True, False).startswith("DESCRIPTIVE")
    assert "incomplete" in stage2.final_verdict(normal, None, False, False, True)


def test_the_split_says_where_the_silent_errors_went():
    rows = []
    for k, layer in enumerate(["pass", "pass", "protocol_fail", "flagged_wrong", "silent_wrong"]):
        rows += pair(M1, "F30", 1000 + k, "faulted", "silent_wrong", layer)
    rows += pair(M1, "F31", 1000, "faulted", "pass", "silent_wrong")  # a new silent error
    rows[5]["reason"] = "result.json missing"  # the layer's protocol_fail of seed 1002
    split = stage2.outcome_split(rows, model=M1, families=["F30", "F31"])
    assert (split["bare_silent"], split["to_pass"], split["to_undelivered"]) == (5, 2, 1)
    assert (split["to_flagged"], split["to_silent"], split["new_silent"]) == (1, 1, 1)
    assert split["undelivered_kinds"] == {"no_file": 1}
    assert (
        split["per_family"]["F30"]["to_pass"] == 2 and split["per_family"]["F31"]["new_silent"] == 1
    )


# --- the descriptive outputs ----------------------------------------------------------------------


def test_regret_counts_pairs_that_were_right_without_the_layer():
    rows = (
        pair(M1, "F1", 1020, "faulted", "pass", "silent_wrong")  # regretted
        + pair(M1, "F1", 1021, "faulted", "pass", "pass")
        + pair(M1, "F1", 1022, "faulted", "silent_wrong", "pass")  # a gain, not regret
        + pair(M1, "F1", 1020, "clean", "pass", "protocol_fail")  # regretted
        + pair(M1, "F1", 1021, "clean", "pass", "pass")
    )
    assert stage2.regret(rows, M1, "block-nonote", PRIMARY, ["F1"]) == {
        "faulted": (1, 2, 3),
        "clean": (1, 2, 2),
    }


def test_the_replicate_gives_the_regret_noise_floor():
    rows = [
        rec(M1, "F1", 1020, "faulted", "bare", "pass"),
        rec(M1, "F1", 1020, "faulted", "bare", "silent_wrong", rep=1),  # bare against bare
        rec(M1, "F1", 1021, "faulted", "bare", "pass"),
        rec(M1, "F1", 1021, "faulted", "bare", "pass", rep=1),
    ]
    floor = stage2.regret(rows, M1, "bare", PRIMARY, ["F1"], "bare", 0, 1)
    assert floor["faulted"] == (1, 2, 2)


def test_heterogeneity_groups_families_and_small_groups_get_no_interval():
    reg = stage2.Registry.make(["F30", "F31", "F101"])
    rows = campaign(reg, models=(M1,))
    by_pack = stage2.heterogeneity(
        rows, M1, "block-nonote", ["F30", "F31", "F101"], stage2.pack_of, draws=DRAWS
    )
    assert set(by_pack) == {"alignment", "single-cell"}
    assert by_pack["alignment"].families == 2 and by_pack["alignment"].estimate == pytest.approx(
        1.0
    )
    assert by_pack["alignment"].low != by_pack["alignment"].low  # two families: no interval
    big = stage2.heterogeneity(
        campaign(registry(), models=(M1,)), M1, "block-nonote", P_SET, stage2.pack_of, draws=DRAWS
    )
    assert big["alignment"].families == 10 and big["alignment"].low == pytest.approx(1.0)
    assert stage2.pack_of("F40") == "quantification" and stage2.pack_of("F112") == "single-cell"


def test_pooled_numbers_are_labelled_and_use_the_common_families():
    reg = registry()
    result = decide(campaign(reg))
    pooled = stage2.pooled([d.drop for d in result.models], draws=DRAWS)
    assert (
        pooled["label"] == "pooled over models"
        and pooled["models"] == 3
        and pooled["families"] == 10
    )
    assert pooled["estimate"] == pytest.approx(1.0) and pooled["low"] == pytest.approx(1.0)
    assert stage2.pooled([], draws=10)["families"] == 0


def test_mcnemar_companion_counts_the_discordant_pairs():
    reg = registry()
    rows = campaign(reg, models=(M1,))
    table = stage2.mcnemar_seed_level(rows, reg, models=(M1,))
    assert table[0]["fixed"] == 50 and table[0]["broken"] == 0 and table[0]["p"] < 1e-6


def test_safety_table_has_friction_delivery_cost_and_the_layer_record():
    reg = registry()
    rows = campaign(reg, models=(M1,))
    for r in rows:
        if (
            r["arm"] == "block-nonote"
            and r["twin"] == "clean"
            and r["family"] == "F30"
            and r["seed"] == 1000
        ):
            r["layer"]["actions"]["block"] = 2
    table = stage2.safety_table(rows, reg, models=(M1,), arms=("bare", "block-nonote"))
    clean = next(
        t
        for t in table
        if t["arm"] == "block-nonote" and t["twin"] == "clean" and t["set"] == "gated"
    )
    assert (
        clean["n"] == 80
        and clean["friction"] == pytest.approx(1 / 80)
        and clean["layer_blocks"] == 2
    )
    faulted = next(
        t for t in table if t["arm"] == "bare" and t["twin"] == "faulted" and t["set"] == "P"
    )
    assert faulted["silent_planned"] == 1.0 and faulted["silent_delivered"] == 1.0
    assert faulted["median_turns"] == 7 and faulted["usd"] == pytest.approx(0.5)


def test_bypass_splits_the_trapped_route_by_whether_a_rule_fired():
    reg = registry()
    rows = [
        r
        for r in campaign(reg, models=(M1,))
        if r["arm"] == "block-nonote" and stage2.cell_of(r) == PRIMARY
    ]
    faulted = [r for r in rows if r["twin"] == "faulted" and r["family"] in reg.primary]
    for r in faulted[:6]:
        r["exposed"] = True
    for r in faulted[:2]:
        r["layer"]["fired"] = ["x.rule:code:block"]
    faulted[6].update(exposed_issued=True, exposed=False)
    row = stage2.bypass(rows, reg, models=(M1,))[0]
    assert (row["route_ran"], row["route_ran_after_a_fire"], row["route_ran_no_fire"]) == (6, 2, 4)
    assert row["blocked_only"] == 1 and row["faulted"] == 50


def test_the_default_cell_is_a_harm_check_and_f4_is_reported_apart():
    reg = registry()
    rows = campaign(reg, behave(clean_layer="protocol_fail"))
    out = stage2.default_cell(rows, reg, flips=FLIPS, draws=DRAWS)
    d = out["models"][M1]
    assert d["clean_pass_loss"].estimate == pytest.approx(1.0) and d["harm_flag"]
    assert d["silent_drop"].p is None  # no test on the faulted silent rate
    assert d["silent_drop"].families == 16 and "F4" not in d["silent_drop"].per_family
    assert out["F4"][M1]["bare"]["clean"] == {"pass": 3}


def test_sensitivities_recompute_the_decision_on_other_sets():
    reg = registry(weak=("F44",), weak_reachable=("F44",))
    rows = campaign(reg)
    names = [s["name"] for s in stage2.sensitivities(rows, reg, flips=FLIPS)]
    assert names[0] == "registered P" and "without the five stage-1 families" in names
    assert (
        "without the 1 most influential families" in names
        and "every bite family (no reachability)" in names
    )
    assert not any("leave out the alignment pack" in n for n in names)  # nothing would be left of P
    assert any("weak bites" in n for n in names)
    assert all(s["effective"] for s in stage2.sensitivities(rows, reg, flips=FLIPS))


def test_sensitivity_drops_layer_trials_whose_layer_was_not_intact():
    reg = registry()
    rows = campaign(reg)
    for r in rows[:300]:
        if r["arm"] == "block-nonote":
            r["layer"]["touched_layer"] = 1
    assert stage2.layer_flag(next(r for r in rows[:300] if r["arm"] == "block-nonote"))
    assert any("touched layer" in s["name"] for s in stage2.sensitivities(rows, reg, flips=FLIPS))


def test_lenient_rescoring_reads_a_bare_list_as_the_answer(tmp_path):
    row = rec(
        M1,
        "F30",
        1000,
        "faulted",
        "bare",
        "protocol_fail",
        reason="result.json is not a JSON object",
    )
    trial = tmp_path / "trials" / row["trial"]
    (trial / "work").mkdir(parents=True)
    (trial / "task.json").write_text(
        json.dumps({"truth": {"kind": "int_list", "expected": [3, 7]}})
    )
    (trial / "work" / "result.json").write_text("[3, 7]")
    relaxed, unavailable = stage2.lenient_rows(
        [row, rec(M1, "F31", 1000, "faulted", "bare", "pass")], [tmp_path]
    )
    assert relaxed[0]["outcome"] == "pass" and unavailable == 1


def test_the_final_message_and_the_expected_value_are_read_from_a_transcript():
    anthropic = [{"role": "assistant", "content": [{"type": "text", "text": "The count is 39."}]}]
    openai = [{"role": "assistant", "content": "I made it 24."}, {"role": "tool", "content": "x"}]
    assert stage2.final_message(anthropic) == "The count is 39."
    assert stage2.final_message(openai) == "I made it 24." and stage2.final_message([]) == ""
    count = {"kind": "int", "expected": 24}
    assert stage2.contains_expected("so the right count is 24, not 39", count)
    assert not stage2.contains_expected("so it is 124 or 24.5", count)
    ids = {"kind": "id_list", "expected": ["v01", "v05"]}
    assert stage2.contains_expected("v01 and v05", ids) and not stage2.contains_expected("v01", ids)


def test_disclosure_counts_silent_trials_whose_message_gave_the_cause_away(tmp_path):
    reg = registry()
    rows = []
    for k, text in enumerate(["the count is 24 really", "I am unsure about this", "done"]):
        row = rec(M1, "F30", 1000 + k, "faulted", "bare", "silent_wrong")
        trial = tmp_path / "trials" / row["trial"]
        trial.mkdir(parents=True)
        (trial / "task.json").write_text(json.dumps({"truth": {"kind": "int", "expected": 24}}))
        (trial / "transcript.json").write_text(
            json.dumps([{"role": "assistant", "content": [{"type": "text", "text": text}]}])
        )
        rows.append(row)
    out = stage2.disclosure(rows, [tmp_path], reg, models=(M1,))
    assert out["models"][M1] == {
        "silent": 3,
        "expected_value": 1,
        "doubt": 1,
        "undisclosed": 1,
        "unavailable": 0,
    }
    assert len(out["audit"]) == 3


# --- the command line -----------------------------------------------------------------------------


@pytest.fixture
def frozen(tmp_path):
    """A registry as a sets file and a complete campaign on disk, one directory per model."""
    reg = registry()
    sets = {
        "format": stage2.SETS_FORMAT,
        "primary": list(reg.primary),
        "q1": list(reg.q1),
        "q2": list(reg.q2),
        "q3": list(reg.q3),
        "weak": [],
        "excluded": ["F34"],
        "default_only": ["F4"],
        "seeds": {f: list(s) for f, s in reg.seeds.items()},
        "default_seeds": {f: list(s) for f, s in reg.default_seeds.items()},
        "power": {"underpowered": False},
    }
    path = tmp_path / "sets.json"
    path.write_text(json.dumps(sets))
    dirs = []
    for model in stage2.MODELS:
        root = tmp_path / "runs" / model.replace("/", "_")
        rows = campaign(reg, models=(model,))
        write_run(root, rows)
        ids = {r["trial"]: {"model": model} for r in rows}
        (root / "plan.json").write_text(
            json.dumps({"format": 1, "ids": list(ids), "specs": ids, "jobs": []})
        )
        dirs.append(str(root))
    return reg, path, dirs


def run_cli(*args):
    return stage2.main([*args])


def test_analyze_prints_the_verdict_and_every_section(frozen, tmp_path, capsys):
    _reg, sets, dirs = frozen
    out = tmp_path / "numbers.json"
    code = run_cli(
        "analyze",
        "--out",
        *dirs,
        "--sets",
        str(sets),
        "--json",
        str(out),
        "--flips",
        "500",
        "--bootstraps",
        "100",
    )
    text = capsys.readouterr().out
    assert code == 0 and text.startswith("VERDICT: EFFECTIVE")
    for section in (
        "attrition",
        "strata",
        "sensitivity",
        "default cell",
        "McNemar",
        "regret",
        "pooled",
        "heterogeneity",
        "bypass",
    ):
        assert section in text
    data = json.loads(out.read_text())
    assert data["verdict"] == "EFFECTIVE" and data["primary"]["effective"] is True
    assert data["completeness"]["absent"] == [] and data["specificity"]["P_minus_Q1"][M1]["P"] == 10
    assert data["split"][0]["to_pass"] == 50 and data["coverage"]["P"] == 10


def test_analyze_refuses_when_a_planned_trial_is_absent(frozen, capsys):
    _reg, sets, dirs = frozen
    victim = next((Path(dirs[0]) / "trials").glob("*/*/score.json"))
    victim.unlink()
    code = run_cli(
        "analyze", "--out", *dirs, "--sets", str(sets), "--flips", "300", "--bootstraps", "50"
    )
    err = capsys.readouterr().err
    assert code == 2 and "ABORT" in err and "no record" in err and "--allow-missing" in err
    code = run_cli(
        "analyze",
        "--out",
        *dirs,
        "--sets",
        str(sets),
        "--flips",
        "300",
        "--bootstraps",
        "50",
        "--allow-missing",
    )
    out = capsys.readouterr().out
    assert code == 0 and "WARNING: --allow-missing" in out and "EFFECTIVE (incomplete data" in out


def test_analyze_refuses_records_that_are_not_planned_or_have_a_provenance_problem(frozen, capsys):
    _reg, sets, dirs = frozen
    stray = rec(
        M1, "F30", 1000, "faulted", "block-nonote", "pass", cell=("bare", "explicit", "free")
    )
    write_run(Path(dirs[0]), [stray])
    assert run_cli("analyze", "--out", *dirs, "--sets", str(sets), "--allow-missing") == 2
    assert "neither the primary" in capsys.readouterr().err


def test_analyze_refuses_a_directory_of_another_campaign(frozen, capsys):
    _reg, sets, dirs = frozen
    assert run_cli("analyze", "--out", *dirs, "--sets", str(sets), "--code-digest", "f" * 64) == 2
    assert "code_digest differs" in capsys.readouterr().err


def test_analyze_refuses_a_sets_file_without_a_registry(tmp_path, frozen, capsys):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"primary": ["F30"], "complement": []}))
    assert run_cli("analyze", "--out", frozen[2][0], "--sets", str(old)) == 2
    assert "seed registry" in capsys.readouterr().err


def test_health_prints_counts_and_no_outcome_rate(frozen, tmp_path, capsys):
    _reg, sets, dirs = frozen
    out = tmp_path / "health.json"
    assert run_cli("health", "--out", *dirs, "--sets", str(sets), "--json", str(out)) == 0
    text = capsys.readouterr().out
    assert "planned" in text and "providers" in text and "Wafer" in text
    for word in ("silent", "pass ", "EFFECTIVE", "NOT SHOWN", "drop"):
        assert word not in text
    table = json.loads(out.read_text())["table"]
    assert len(table) == 15 and all(t["absent"] == 0 for t in table)
    assert sum(t["recorded"] for t in table) == len(stage2.load_rows(dirs))


def test_a_sets_file_that_is_not_confirmatory_needs_the_override(frozen, tmp_path, capsys):
    _reg, sets, dirs = frozen
    flagged = json.loads(sets.read_text()) | {"flags": ["lenient"]}
    path = tmp_path / "flagged.json"
    path.write_text(json.dumps(flagged))
    assert run_cli("analyze", "--out", *dirs, "--sets", str(path)) == 2
    assert "not confirmatory" in capsys.readouterr().err
    code = run_cli(
        "analyze",
        "--out",
        *dirs,
        "--sets",
        str(path),
        "--allow-missing",
        "--flips",
        "300",
        "--bootstraps",
        "50",
    )
    assert code == 0
    assert "WARNING: --allow-missing: the sets file is not confirmatory" in capsys.readouterr().out


def test_a_family_that_lost_every_pair_stops_the_analysis(frozen, capsys):
    """The ids are all on disk but the family has no valid pair: it would vanish from the mean."""
    _reg, sets, dirs = frozen
    pattern = "*/F30-s*-faulted-implicit-forced-bare-block-nonote-r0/score.json"
    for path in (Path(dirs[0]) / "trials").glob(pattern):
        record = json.loads(path.read_text()) | {"outcome": "infra_error", "attempt": MAX_ATTEMPTS}
        path.write_text(json.dumps(record))
    assert (
        run_cli(
            "analyze", "--out", *dirs, "--sets", str(sets), "--flips", "300", "--bootstraps", "50"
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "no valid pair for F30" in err and M1 in err
    code = run_cli(
        "analyze",
        "--out",
        *dirs,
        "--sets",
        str(sets),
        "--allow-missing",
        "--flips",
        "300",
        "--bootstraps",
        "50",
    )
    assert code == 0 and "no valid pair for F30" in capsys.readouterr().out


def test_the_registry_filter_keeps_only_registered_trials():
    reg = registry()
    good = rec(M1, "F30", 1000, "faulted", "bare", "pass")
    assert stage2.usable(good, reg) and stage2.usable(good)
    assert not stage2.usable(rec(M1, "F30", 1000, "faulted", "bare", "infra_error"), reg)
    assert not stage2.usable(rec(M1, "F30", 1020, "faulted", "bare", "pass"), reg)  # wrong block
    assert not stage2.usable(rec(M1, "F30", 1000, "faulted", "bare", "pass", rep=1), reg)
    assert not stage2.usable(
        rec(M1, "F30", 1000, "faulted", "warn-nonote", "pass", cell=DEFAULT), reg
    )
    assert not stage2.usable(rec(M1, "F30", 1000, "faulted", "bare", "pass", harness="opencode"))
    assert stage2.registered(
        rec(M1, "F30", 1000, "faulted", "bare", "pass", rep=1), reg
    )  # A/A on P
    assert not stage2.registered(rec(M1, "F41", 1000, "faulted", "bare", "pass", rep=1), reg)
    assert stage2.registered(rec(M1, "F4", 1002, "faulted", "bare", "pass", cell=DEFAULT), reg)
    assert not stage2.registered(rec(M1, "F4", 1003, "faulted", "bare", "pass", cell=DEFAULT), reg)


def test_analysis_never_counts_an_unregistered_seed_even_if_it_is_handed_one():
    reg = registry()
    rows = campaign(reg, models=(M1,)) + pair(
        M1, "F30", 7, "faulted", "silent_wrong", "silent_wrong"
    )
    result = stage2.analyze(rows, reg, flips=300, draws=50, files=False)
    assert result["primary"].models[0].drop.pairs == 50  # the seed-7 pair is not among them


def test_the_pack_clustered_interval_resamples_whole_packs():
    per_family = {"F30": 1.0, "F31": 1.0, "F41": 0.0, "F42": 0.0, "F101": 0.5}
    low, high, packs = stage2.cluster_bootstrap_ci(per_family, stage2.pack_of, draws=400)
    assert packs == 3 and low <= 0.5 <= high and low < high
    assert all(x != x for x in stage2.cluster_bootstrap_ci({"F30": 1.0}, stage2.pack_of)[:2])
    reg = registry()
    out = stage2.analyze(campaign(reg), reg, flips=300, draws=100, files=False)
    assert (
        out["cluster"][M1]["packs"] == 1
    )  # all of P is in the alignment pack: nothing to resample


def test_the_scan_finds_registered_seeds_that_an_agent_has_already_run(tmp_path, capsys):
    reg = registry()
    sets = {
        "format": stage2.SETS_FORMAT,
        "primary": list(reg.primary),
        "q1": list(reg.q1),
        "q2": list(reg.q2),
        "q3": list(reg.q3),
        "seeds": {f: list(s) for f, s in reg.seeds.items()},
        "default_seeds": {f: list(s) for f, s in reg.default_seeds.items()},
        "default_only": ["F4"],
    }
    sets_path = tmp_path / "sets.json"
    sets_path.write_text(json.dumps(sets))
    dev, old, own = (
        tmp_path / "runs" / "pilot",
        tmp_path / "runs" / "stage1b",
        tmp_path / "runs" / "stage2",
    )
    for root, rows in (
        (dev, [rec(M1, "F30", 3, "faulted", "bare", "pass")]),  # a development seed
        (
            old,
            [
                rec(M1, "F30", 1002, "faulted", "bare", "pass"),
                rec(M2, "F44", 1004, "clean", "bare", "pass"),
            ],
        ),
        (own, [rec(M1, "F30", 1002, "faulted", "bare", "pass")]),  # the job's own directory
    ):
        write_run(root, rows)
    runs = [str(dev), str(old), str(own)]
    assert stage2.main(["scan", "--sets", str(sets_path), "--runs", *runs, "--own", str(own)]) == 2
    text = capsys.readouterr().out
    assert "F30 seed 1002" in text and "F44 seed 1004" in text and "seed 3" not in text
    assert str(own) not in text and "2 already run" in text
    assert (
        stage2.main(
            ["scan", "--sets", str(sets_path), "--runs", str(dev), str(own), "--own", str(own)]
        )
        == 0
    )
    assert stage2.seen_pairs([old])[("F30", 1002)] == {str(old): 1}
    assert ("F30", 1002) in stage2.registered_pairs(reg) and (
        "F4",
        1002,
    ) in stage2.registered_pairs(reg)
