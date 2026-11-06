"""The bite set per model, reachability from a family's own rules, the strata P, Q1, Q2, Q3, the
family-to-rule map and the `sets` command, on synthetic pilot records and replays."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biowarden.bench import pilot as pilot_module
from biowarden.bench import stage2
from biowarden.bench.runner.trial import TrialSpec
from biowarden.bench.stage2 import FamilyRule

M1, M2, M3 = stage2.MODELS
PRIMARY = stage2.PRIMARY_CELL
PROVENANCE = {
    "scorer": 3,
    "code_digest": "d" * 64,
    "run_config_sha": "c" * 64,
    "tool_manifest_sha": "t" * 64,
}
SERVED_BY = {M2: ["Wafer"], M3: ["Near AI"]}
FLIPS, DRAWS = 2000, 200


def trial_row(model, family, seed, twin, outcome, exposed=False, signature=None, arm="bare"):
    spec = TrialSpec(model, family, seed, twin, 0, "implicit", "forced", "bare", "minimal", arm)
    row = {
        "trial": spec.trial_id,
        "model": model,
        "family": family,
        "seed": seed,
        "twin": twin,
        "arm": arm,
        "rep": 0,
        "harness": "minimal",
        "contract": "bare",
        "tier": "implicit",
        "route": "forced",
        "outcome": outcome,
        "exposed": exposed,
        "signature": signature,
        **PROVENANCE,
    }
    if model in stage2.PIN_TAGS:
        row |= {
            "provider_pin": [stage2.PIN_TAGS[model]],
            "providers": SERVED_BY[model],
            "pin_match": True,
        }
    return row


def family_rows(
    model,
    family,
    exposed=5,
    silent=2,
    clean_silent=0,
    clean_undelivered=0,
    seeds=range(5),
    undelivered=0,
):
    """Five seeds of one family for one model: the first `exposed` faulted trials took the route,
    the first `silent` of them ended silent_wrong, `undelivered` more are protocol_fail."""
    rows = []
    for seed in seeds:
        k = seed % 5
        outcome = "silent_wrong" if k < silent else "pass"
        if silent <= k < silent + undelivered:
            outcome = "protocol_fail"
        rows.append(trial_row(model, family, seed, "faulted", outcome, exposed=k < exposed))
        clean = "silent_wrong" if k < clean_silent else "pass"
        if clean_silent <= k < clean_silent + clean_undelivered:
            clean = "protocol_fail"
        rows.append(trial_row(model, family, seed, "clean", clean))
    return rows


def pilot(spec, default=None):
    """spec: {family: {model: kwargs of family_rows}}; a model without an entry handles the trap
    (5 exposed, no silent trial)."""
    rows = []
    for family, per_model in spec.items():
        for model in stage2.MODELS:
            kwargs = per_model.get(model, default or {"exposed": 5, "silent": 0})
            rows += family_rows(model, family, **kwargs)
    return rows


def replay(
    trial, family, fired=(), seed=0, crashed=False, problem="", twin="faulted", route="forced"
):
    return {
        "trial": trial,
        "family": family,
        "twin": twin,
        "seed": seed,
        "route": route,
        "fired": list(fired),
        "blocked": any(f.endswith(":block") for f in fired),
        "warned": any(f.endswith(":warn") for f in fired),
        "crashed": crashed,
        "problem": problem,
    }


def replays_of(rows, fired_by_family, **kw):
    """One replay per exposed faulted pilot trial, firing what the family's entry says."""
    return [
        replay(r["trial"], r["family"], fired_by_family.get(r["family"], ()), r["seed"], **kw)
        for r in rows
        if r["twin"] == "faulted" and (r["exposed"] or r["signature"])
    ]


# --- bite per model -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exposed", "silent", "label"),
    [
        (5, 5, "bites"),
        (5, 2, "bites"),
        (3, 2, "bites"),  # two silent among exactly three exposed
        (5, 1, "weak bite"),
        (3, 1, "weak bite"),
        (5, 0, "control (handled)"),
        (2, 2, "control (routed around)"),  # fewer than three took the route
        (0, 0, "control (routed around)"),
    ],
)
def test_the_registered_bite_rule(exposed, silent, label):
    rows = family_rows(M1, "F30", exposed=exposed, silent=min(silent, exposed))
    entry = stage2.classify_model(rows, M1, ["F30"])["F30"]
    assert entry["label"] == label


def test_one_silent_trial_is_a_weak_bite_not_a_bite():
    # the cut 'at least 10%' was one trial for up to nine exposed trials; the registered cut is two
    assert pilot_module.label_of(5, 1) == "weak bite" and pilot_module.label_of(5, 2) == "bites"
    assert pilot_module.label_of(25, 2) == "weak bite"  # 8%: below the share floor as well


def test_a_trap_signature_answer_counts_as_having_taken_the_route():
    rows = family_rows(M1, "F30", exposed=0, silent=0)
    for r in rows:  # three silent trials the exposed() heuristic missed, but with the trap's answer
        if r["twin"] == "faulted" and r["seed"] < 3:
            r.update(outcome="silent_wrong", signature=True)
    entry = stage2.classify_model(rows, M1, ["F30"])["F30"]
    assert (
        entry["exposed"] == 3 and entry["exposed_by_signature"] == 3 and entry["label"] == "bites"
    )
    assert entry["silent_unexposed"] == 0
    rows[0].update(signature=False)  # silent, not the trap's answer, route not seen: unexplained
    entry = stage2.classify_model(rows, M1, ["F30"])["F30"]
    assert entry["silent_unexposed"] == 1 and entry["label"] == "control (routed around)"


def test_models_are_never_pooled():
    rows = family_rows(M1, "F30", exposed=2, silent=1) + family_rows(M2, "F30", exposed=2, silent=1)
    entries = pilot_module.classify(rows)
    assert {(e["model"], e["label"]) for e in entries} == {
        (M1, "control (routed around)"),
        (M2, "control (routed around)"),
    }  # pooled, four exposed trials with two silent would have "bitten"
    assert {e["model"] for e in pilot_module.classify(rows[:10])} == {M1}


def test_only_the_five_pilot_seeds_count():
    rows = family_rows(M1, "F30", exposed=5, silent=0)
    rows += family_rows(
        M1, "F30", exposed=5, silent=5, seeds=range(5, 12)
    )  # development seeds 5-11
    assert stage2.classify_model(rows, M1, ["F30"])["F30"]["label"] == "control (handled)"
    assert pilot_module.classify(rows)[0]["label"] == "bites"  # without a seed filter they would


def test_an_incomplete_pilot_raises_instead_of_shrinking_the_bite_set():
    rows = family_rows(M1, "F30")
    gone = rows.pop(3)
    assert stage2.pilot_gaps(rows, M1, ["F30"]) == [("F30", gone["twin"], gone["seed"])]
    with pytest.raises(ValueError, match="pilot cell"):
        stage2.classify_model(rows, M1, ["F30"])
    assert stage2.classify_model(rows, M1, ["F30"], strict=False)["F30"]["trials"] == 9
    rows += [trial_row(M1, "F30", 9, "clean", "infra_error")]  # a lost trial is not a counted cell
    assert len(stage2.pilot_gaps(rows, M1, ["F30"])) == 1


def test_undelivered_trials_are_not_exposure_and_are_reported():
    rows = family_rows(M1, "F30", exposed=5, silent=2, undelivered=2)
    entry = stage2.classify_model(rows, M1, ["F30"])["F30"]
    assert (
        entry["undelivered"] == 2
        and entry["exposed"] == 3
        and entry["silent_given_exposed"] == (2, 3)
    )


def test_the_clean_twin_is_screened():
    rows = family_rows(M1, "F34", exposed=5, silent=0, clean_silent=4)  # clean 4 of 5 silent
    entry = stage2.classify_model(rows, M1, ["F34"])["F34"]
    assert entry["clean_flag"] and entry["clean_silent"] == (4, 5) and entry["clean_pass"] == (1, 5)
    assert entry["label"] == "control (handled)"
    one = stage2.classify_model(family_rows(M1, "F30", clean_silent=1), M1, ["F30"])["F30"]
    assert not one["clean_flag"]


# --- reachability ---------------------------------------------------------------------------------

RULES = {
    "F30": FamilyRule(("a.own",)),
    "F31": FamilyRule(("b.own",)),
    "F22": FamilyRule(("p.own",), seed_parity="even"),
    "F23": FamilyRule((), {"free": ("f.own",)}),
}


def reach_of(family, fired, rules=RULES, blocking=None, **kw):
    rows = family_rows(M1, family, exposed=5, silent=2)
    replays = replays_of(rows, {family: fired}, **kw)
    return stage2.reachable(replays, rows, own_rules=rules, blocking=blocking)[family]


def test_an_own_rule_that_would_block_makes_the_family_reachable():
    e = reach_of("F30", ["a.own:code:block"])
    assert (
        e["block"] and not e["warn_only"] and not e["foreign"] and e["n"] == 5 and e["missing"] == 0
    )


def test_a_foreign_rule_does_not_make_a_family_reachable():
    e = reach_of("F30", ["x.foreign:code:block"])
    assert not e["block"] and e["foreign"] and not e["warn_only"]
    both = reach_of("F30", ["x.foreign:code:block", "a.own:code:block"])
    assert both["block"] and both["foreign"]


def test_an_own_rule_that_only_warns_is_reported_apart():
    e = reach_of("F30", ["a.own:cannot_tell:warn"])
    assert e["warn_only"] and not e["block"]
    # a block anywhere wins over a warning elsewhere
    rows = family_rows(M1, "F30", exposed=5, silent=2)
    replays = replays_of(rows, {})
    replays[0]["fired"] = ["a.own:cannot_tell:warn"]
    replays[1]["fired"] = ["a.own:code:block"]
    e = stage2.reachable(replays, rows, own_rules=RULES)["F30"]
    assert e["block"] and not e["warn_only"]


def test_an_own_rule_that_the_arm_would_not_block_is_not_reachability():
    e = reach_of("F30", ["a.own:code:block"], blocking={"other.rule"})
    assert not e["block"] and e["warn_only"]
    assert reach_of("F30", ["a.own:code:block"], blocking={"a.own"})["block"]


def test_without_a_rule_map_any_rule_is_own():
    rows = family_rows(M1, "F30", exposed=5, silent=2)
    e = stage2.reachable(replays_of(rows, {"F30": ["whatever:c:block"]}), rows)["F30"]
    assert e["block"] and not e["foreign"]


def test_seed_parity_and_route_limit_the_own_rules():
    even_rows = family_rows(M1, "F22", exposed=5, silent=2)
    fired = {"F22": ["p.own:phred64_undeclared:block"]}
    e = stage2.reachable(replays_of(even_rows, fired), even_rows, own_rules=RULES)["F22"]
    assert e["block"]  # seeds 0, 2, 4 are even
    odd = [r for r in even_rows if r["seed"] % 2]
    only_odd = [r for r in replays_of(even_rows, fired) if r["seed"] % 2]
    e = stage2.reachable(only_odd, odd, own_rules=RULES)["F22"]
    assert not e["block"] and e["foreign"]  # on odd seeds the rule is not this family's
    assert RULES["F22"].own(2) == {"p.own"} and RULES["F22"].own(3) == frozenset()
    assert RULES["F23"].own(None, "forced") == frozenset() and RULES["F23"].own(None, "free") == {
        "f.own"
    }
    forced = reach_of("F23", ["f.own:code:block"])
    assert not forced["block"]  # the strandedness rules fire on the free route only
    assert reach_of("F23", ["f.own:code:block"], route="free")["block"]


def test_only_exposed_faulted_pilot_trials_are_replayed():
    rows = family_rows(M1, "F30", exposed=2, silent=2)
    fired = {"F30": ["a.own:code:block"]}
    replays = [replay(r["trial"], "F30", fired["F30"], r["seed"], twin=r["twin"]) for r in rows]
    e = stage2.reachable(replays, rows, own_rules=RULES)["F30"]
    assert e["n"] == 2 and e["missing"] == 0  # not the clean twin, not the three that routed around
    assert "F31" not in stage2.reachable(replays, rows, own_rules=RULES)


def test_a_replay_error_and_a_missing_replay_are_counted_not_ignored():
    rows = family_rows(M1, "F30", exposed=5, silent=2)
    replays = replays_of(rows, {})
    replays[0]["crashed"] = True
    replays[1]["problem"] = "transcript: ValueError"
    del replays[4]
    e = stage2.reachable(replays, rows, own_rules=RULES)["F30"]
    assert (e["crashed"], e["n"], e["missing"]) == (2, 4, 1) and not e["block"]


def test_a_replay_without_fired_falls_back_to_the_blocking_rules():
    rows = family_rows(M1, "F30", exposed=5, silent=2)
    old = [
        {"trial": r["trial"], "family": "F30", "twin": "faulted", "blocking_rules": ["a.own"]}
        for r in rows[::2]
    ]
    assert stage2.reachable(old, rows, own_rules=RULES)["F30"]["block"]


# --- the sets -------------------------------------------------------------------------------------

FAMILIES = ["F4", "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37", "F38"]
BITES = {"exposed": 5, "silent": 3}


def reference_spec():
    return {
        "F30": {M1: BITES},  # bites for Haiku
        "F31": {M2: BITES},  # bites for DeepSeek only
        "F32": {M3: BITES},  # bites for gpt-oss, no rule fires
        "F33": {},  # handled everywhere
        "F34": {M1: BITES},  # named in the exclusions
        "F35": {},  # handled, no rule
        "F36": {M1: {"exposed": 5, "silent": 1}},  # a weak bite
        "F37": {M2: {"exposed": 1, "silent": 1}},  # routed around
        "F38": {},
    }


def reach_reference():
    block = {
        "block": True,
        "warn_only": False,
        "foreign": False,
        "crashed": 0,
        "n": 5,
        "missing": 0,
    }
    none = {**block, "block": False}
    return {"F30": block, "F31": block, "F32": none, "F33": block, "F36": block, "F37": none}


def reference_sets(**kw):
    return stage2.build_sets(
        pilot(reference_spec()), FAMILIES, reach_reference(), excluded=["F34"], **kw
    )


def test_the_strata_follow_from_bite_and_reach():
    sets = reference_sets()
    assert sets["bite"] == ["F30", "F31", "F32"] and sets["weak"] == ["F36"]
    assert sets["primary"] == ["F30", "F31"]
    assert sets["q1"] == ["F32"] and sets["bite_unreachable"] == ["F32"]  # bite, no own rule blocks
    assert sets["q2"] == ["F33", "F36"]  # an own rule blocks, no (strong) bite
    assert sets["q3"] == ["F35", "F37", "F38"]  # neither
    assert sets["complement"] == ["F32", "F33", "F35", "F36", "F37", "F38"]
    assert sets["weak_reachable"] == ["F36"]
    assert sets["excluded"] == ["F34"] and sets["exclusion_reasons"] == {
        "F34": "named before launch"
    }
    assert sets["default_only"] == ["F4"] and "F4" not in sets["complement"]
    assert sets["flags"] == [] and sets["incomplete"] == {}


def test_each_model_has_its_own_bite_families_and_p():
    sets = reference_sets()
    assert sets["per_model"][M1]["bite"] == ["F30"] and sets["per_model"][M1]["weak"] == ["F36"]
    assert sets["per_model"][M2]["primary"] == ["F31"]
    assert sets["per_model"][M3]["bite"] == ["F32"] and sets["per_model"][M3]["primary"] == []
    assert sets["power"]["primary_by_model"] == {M1: 1, M2: 1, M3: 0}


def test_the_counts_behind_every_label_are_stored():
    sets = reference_sets()
    assert sets["counts"][M1]["F30"] == {
        "trials": 10, "undelivered": 0, "delivered_faulted": 5, "exposed": 5, "silent": 3,
        "silent_unexposed": 0, "exposed_by_signature": 0, "clean_silent": 0, "clean_delivered": 5,
        "clean_pass": 5, "clean_trials": 5,
    }  # fmt: skip
    assert sets["labels"][M2]["F31"] == "bites" and sets["labels"][M1]["F31"] == "control (handled)"
    assert sets["pilot"]["rows"] == 9 * 3 * 10 and len(sets["pilot"]["digest"]) == 64
    assert sets["pilot"]["digest"] == reference_sets()["pilot"]["digest"]  # deterministic


def test_the_sets_carry_the_seed_registry_and_round_trip_into_a_registry():
    sets = stage2.build_sets(
        pilot({"F1": {M1: BITES}, "F30": {M1: BITES}}),
        ["F1", "F30", "F4"],
        {"F1": True, "F30": True},
    )
    assert sets["seeds"]["F1"] == [1020, 1021, 1022, 1023, 1024]
    assert sets["seeds"]["F30"] == [1000, 1001, 1002, 1003, 1004]
    assert sets["default_seeds"]["F1"] == [1020, 1021, 1022] and sets["default_seeds"]["F4"] == [
        1000,
        1001,
        1002,
    ]
    reg = stage2.Registry.from_sets(json.loads(json.dumps(sets)))
    assert (
        reg.primary == ("F1", "F30") and reg.gated == ("F1", "F30") and reg.default_only == ("F4",)
    )
    assert reg.underpowered  # two families in P


def test_legacy_boolean_reach_still_works():
    sets = stage2.build_sets(
        pilot({"F30": {M1: BITES}, "F31": {M1: BITES}}), ["F30", "F31"], {"F30": True}
    )
    assert sets["primary"] == ["F30"] and sets["q1"] == ["F31"]


def test_an_incomplete_pilot_or_undecided_reach_raises_unless_allowed():
    rows = pilot(reference_spec())
    del rows[7]
    with pytest.raises(ValueError, match="incomplete pilot"):
        stage2.build_sets(rows, FAMILIES, reach_reference(), excluded=["F34"])
    sets = stage2.build_sets(rows, FAMILIES, reach_reference(), excluded=["F34"], strict=False)
    assert sets["flags"] == ["incomplete"] and sets["incomplete"]
    undecided = reach_reference()
    undecided["F32"] = {**undecided["F32"], "crashed": 1}  # not reachable, and a replay failed
    with pytest.raises(ValueError, match="undecided"):
        stage2.build_sets(pilot(reference_spec()), FAMILIES, undecided, excluded=["F34"])
    undecided["F32"] = {**undecided["F32"], "block": True}  # reachable anyway: the error is moot
    assert (
        "F32"
        in stage2.build_sets(pilot(reference_spec()), FAMILIES, undecided, excluded=["F34"])[
            "primary"
        ]
    )
    missing = reach_reference()
    missing["F30"] = {**missing["F30"], "block": False, "missing": 2}
    with pytest.raises(ValueError, match="undecided"):
        stage2.build_sets(pilot(reference_spec()), FAMILIES, missing, excluded=["F34"])


def test_a_family_whose_clean_twin_fails_for_two_models_is_excluded_by_the_rule():
    spec = {
        "F30": {
            M1: {"exposed": 5, "silent": 0, "clean_silent": 3},
            M2: {"exposed": 5, "silent": 0, "clean_undelivered": 3},
        },
        "F31": {M1: {"exposed": 5, "silent": 0, "clean_silent": 3}},  # only one model: stays in
        "F32": {
            M1: {"exposed": 5, "silent": 0, "clean_silent": 2}
        },  # flagged, pass 3 of 5 = 60%: stays
    }
    sets = stage2.build_sets(pilot(spec), ["F30", "F31", "F32"], {})
    assert sets["excluded"] == ["F30"] and "below 60%" in sets["exclusion_reasons"]["F30"]
    assert M1 in sets["exclusion_reasons"]["F30"] and "3/5" not in sets["exclusion_reasons"]["F30"]
    assert sets["clean_screen"] == {
        "F30": [M1],
        "F31": [M1],
        "F32": [M1],
    }  # flagged: two silent or more
    assert "F30" not in sets["q3"] and {"F31", "F32"} <= set(sets["q3"])


def test_the_stop_rule_declares_a_small_p_underpowered():
    many = [f"F{n}" for n in (*range(29, 34), *range(35, 40), *range(41, 47), *range(53, 67))]  # 30
    everyone = {f: {m: BITES for m in stage2.MODELS} for f in many}
    reach = {f: True for f in many}
    powered = stage2.build_sets(pilot(everyone), many, reach)
    assert len(powered["primary"]) == 30 and not powered["power"]["underpowered"]
    haiku_only = {f: {M1: BITES} for f in many}
    one_model = stage2.build_sets(pilot(haiku_only), many, reach)
    assert (
        len(one_model["primary"]) == 30 and one_model["power"]["underpowered"]
    )  # only one model has 15
    few = stage2.build_sets(
        pilot({f: {m: BITES for m in stage2.MODELS} for f in many[:19]}), many[:19], reach
    )
    assert few["power"]["underpowered"]  # 19 families: P under 20
    twenty = stage2.build_sets(
        pilot({f: {m: BITES for m in stage2.MODELS} for f in many[:20]}), many[:20], reach
    )
    assert not twenty["power"]["underpowered"]


# --- the contrasts on the strata ------------------------------------------------------------------


def campaign_rows(reg, layer_effect):
    """layer_effect(family) -> (bare silent seeds out of 5, layer silent seeds out of 5)."""
    rows = []
    for model in stage2.MODELS:
        for arm, cell in stage2.CAMPAIGN:
            for family in reg.families_for(arm, cell):
                for seed in reg.seeds_for(cell)[family]:
                    k = seed % 10
                    n_bare, n_layer = layer_effect(family)
                    for twin in ("clean", "faulted"):
                        silent = twin == "faulted" and k < (n_bare if arm == "bare" else n_layer)
                        spec = TrialSpec(
                            model, family, seed, twin, 0, cell[1], cell[2], cell[0], "minimal", arm
                        )
                        row = {
                            "trial": spec.trial_id,
                            "model": model,
                            "family": family,
                            "seed": seed,
                            "twin": twin,
                            "arm": arm,
                            "rep": 0,
                            "harness": "minimal",
                            "outcome": "silent_wrong" if silent else "pass",
                            "contract": cell[0],
                            "tier": cell[1],
                            "route": cell[2],
                        }
                        rows.append(row)
    return rows


def test_the_strata_are_tested_two_sided_so_a_harmful_layer_shows():
    reg = stage2.Registry.make(
        ["F30", "F31", "F32", "F33", "F35", "F36", "F37", "F38"], ["F41", "F42"], ["F43", "F44"],
        ["F45", "F46", "F53", "F54", "F55", "F56", "F57", "F58"],
    )  # fmt: skip
    q3 = set(reg.q3)
    rows = campaign_rows(reg, lambda f: (0, 5) if f in q3 else (5, 0))  # the layer hurts on Q3
    out = stage2.specificity(rows, reg, flips=FLIPS, draws=DRAWS)
    harm = out["strata"]["Q3"]["models"][M1]["silent"]
    assert harm.estimate == pytest.approx(-1.0) and harm.sided == "two-sided"
    assert harm.p == pytest.approx(2 / 2**8)  # eight families all in the same direction, both tails
    assert (
        stage2.sign_flip_p(list(harm.per_family.values())) > 0.9
    )  # a one-sided 'greater' test misses it
    good = out["strata"]["P"]["models"][M1]["silent"]
    assert good.estimate == pytest.approx(1.0) and out["strata"]["P"]["families"] == 8
    assert out["strata"]["Q1"]["models"][M1]["baseline_silent"] == pytest.approx(1.0)
    diff = out["P_minus_Q1"][M1]
    assert diff["estimate"] == pytest.approx(0.0) and diff["P"] == 8 and diff["Q1"] == 2


def test_p_minus_q1_is_the_gap_between_what_the_rules_reach_and_what_they_do_not():
    reg = stage2.Registry.make(["F30", "F31", "F32", "F33", "F35"], ["F41", "F42", "F43"])
    q1 = set(reg.q1)
    rows = campaign_rows(reg, lambda f: (5, 5) if f in q1 else (5, 0))
    diff = stage2.specificity(rows, reg, flips=FLIPS, draws=DRAWS)["P_minus_Q1"][M2]
    assert diff["estimate"] == pytest.approx(1.0) and diff["low"] == pytest.approx(1.0)
    assert (
        stage2.specificity(rows, reg, flips=FLIPS, draws=DRAWS)["strata"]["Q1"]["models"][M2][
            "silent"
        ].p
        == 1.0
    )


def test_coverage_and_the_replication_row():
    reg = stage2.Registry.make(["F1", "F30"], ["F7"], ["F43"], ["F45", "F46"], weak=("F43",))
    cov = stage2.coverage(reg)
    assert (cov["P"], cov["bite"], cov["gated"]) == (2, 3, 6) and cov[
        "P_over_bite"
    ] == pytest.approx(2 / 3)
    assert cov["bite_over_gated"] == 0.5 and (cov["q1"], cov["q2"], cov["q3"], cov["weak"]) == (
        1,
        1,
        2,
        1,
    )
    rows = campaign_rows(reg, lambda f: (5, 0))
    sets = stage2.family_sets(rows, reg, "block-nonote", flips=FLIPS, draws=DRAWS)
    assert sets["stage1_replication"]["families"] == 2  # F1 and F7 of the five stage-1 families
    assert sets["all_bite"]["families"] == 3 and sets["all_gated"]["families"] == 6
    assert sets["all_gated"]["models"][M1].estimate == pytest.approx(1.0)


# --- the family-to-rule map -----------------------------------------------------------------------


def test_the_map_names_real_rules_and_every_family():
    from biowarden.bench.tasks.families import all_families
    from biowarden.engine.spec import packaged_specs

    rules = stage2.load_family_rules()
    known = {rule.id for spec in packaged_specs() for rule in spec.rules}
    named = {
        r
        for fr in rules.values()
        for r in (*fr.rules, *(x for v in fr.route_rules.values() for x in v))
    }
    assert named <= known, sorted(named - known)
    assert set(rules) == set(all_families())  # all 105 families, no other


def test_the_rule_free_families_are_the_review_list():
    rules = stage2.load_family_rules()
    free = sorted((f for f, fr in rules.items() if not fr.rules), key=stage2.fam_no)
    assert free == [
        "F2", "F3", "F4", "F5", "F6", "F8", "F9", "F11", "F12", "F14", "F16", "F17", "F18", "F19",
        "F20", "F21", "F23", "F24", "F25", "F26", "F27", "F28",
    ]  # fmt: skip
    assert len(free) == 22
    assert all(rules[f].rules for f in rules if stage2.fam_no(f) >= 29)  # every F29-F112 has a rule
    assert rules["F1"].rules == ("bcftools.regions_file_base", "bcftools.region_semantics")
    assert rules["F22"].seed_parity == "even" and rules["F23"].route_rules["free"]
    assert rules["F44"].rules == (
        "htseq.min_mapq",
    )  # htseq.strandedness fires on it but is foreign


def test_blocking_rules_are_the_promoted_ones_for_the_current_spec(tmp_path):
    from biowarden.engine.spec import packaged_specs

    hashes = {r.id: s.content_hash() for s in packaged_specs() for r in s.rules}
    table = {
        "fastq.phred_offset": {"mode": "block", "spec_hash": hashes["fastq.phred_offset"]},
        "bcftools.region_semantics": {"mode": "block", "spec_hash": "stale"},  # spec changed since
        "matrix.log_scale_state": {"mode": "warn"},
        "sc.de_raw_default": {"mode": "block"},  # no hash recorded: taken as current
    }
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps(table))
    assert stage2.blocking_rules(path) == {"fastq.phred_offset", "sc.de_raw_default"}
    assert len(stage2.blocking_rules()) >= 100  # the packaged table


# --- the sets command -----------------------------------------------------------------------------


def write_pilot(root, rows):
    for r in rows:
        directory = root / "trials" / r["trial"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "score.json").write_text(json.dumps(r))


@pytest.fixture
def pilot_dir(tmp_path):
    """A pilot of eight real families whose own rules are in the packaged map."""
    spec = {
        "F30": {M1: BITES},  # bwa.mem_min_score
        "F31": {M2: BITES},  # star.genecounts_column
        "F32": {M3: BITES},  # samtools.stats_nm_missing
        "F33": {},
        "F34": {M1: BITES},  # excluded by default
        "F35": {},
    }
    rows = pilot(spec)
    write_pilot(tmp_path / "pilot", rows)
    fired = {
        "F30": ["bwa.mem_min_score:min_score_default:block"],
        "F31": ["star.genecounts_column:wrong_column:block", "x.foreign:code:block"],
        "F32": ["x.foreign:code:block"],  # no own rule blocks: Q1
        "F33": ["bowtie2.maxins_default:declaration_missing:block"],  # own rule, handled: Q2
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replays_of(rows, fired)))
    own = ["bwa.mem_min_score", "star.genecounts_column", "bowtie2.maxins_default"]
    (tmp_path / "promotion.json").write_text(json.dumps({r: {"mode": "block"} for r in own}))
    return tmp_path, rows


def test_the_sets_command_writes_the_frozen_file(pilot_dir, capsys):
    tmp, _rows = pilot_dir
    out = tmp / "sets.json"
    families = ["F30", "F31", "F32", "F33", "F34", "F35"]
    code = stage2.main(
        ["sets", "--pilot", str(tmp / "pilot"), "--replay", str(tmp / "replay.json"),
         "--families", *families, "--promotion", str(tmp / "promotion.json"), "--out", str(out)]
    )  # fmt: skip
    text = capsys.readouterr().out
    sets = json.loads(out.read_text())
    assert code == 0 and text.startswith(
        "6 families: bite 3, primary 2, bite but no own rule blocks 1"
    )
    assert sets["primary"] == ["F30", "F31"] and sets["q1"] == ["F32"] and sets["q2"] == ["F33"]
    assert sets["q3"] == ["F35"] and sets["excluded"] == ["F34"]  # F34 is excluded by default
    assert sets["pilot"]["dirs"] == [str(tmp / "pilot")] and len(sets["family_rules_sha"]) == 64
    assert len(sets["promotion_sha"]) == 64 and sets["blocking_rules"] == 3
    assert "excluded F34: named before launch" in text and "UNDERPOWERED" in text
    assert stage2.Registry.from_sets(sets).primary == ("F30", "F31")
    assert sets["reach"]["F31"]["foreign"] and sets["reach"]["F31"]["block"]


def test_an_explicit_exclusion_list_replaces_the_default(pilot_dir):
    tmp, _rows = pilot_dir
    out = tmp / "sets.json"
    args = ["sets", "--pilot", str(tmp / "pilot"), "--replay", str(tmp / "replay.json"),
            "--families", "F30", "F34", "--exclude", "--promotion", str(tmp / "promotion.json"),
            "--out", str(out)]  # fmt: skip
    assert stage2.main(args) == 0
    assert (
        json.loads(out.read_text())["excluded"] == []
        and "F34" in json.loads(out.read_text())["bite"]
    )


def test_the_sets_command_refuses_unpinned_or_incomplete_pilots(pilot_dir, capsys):
    tmp, rows = pilot_dir
    out = tmp / "sets.json"
    base = ["sets", "--pilot", str(tmp / "pilot"), "--replay", str(tmp / "replay.json"),
            "--families", "F30", "F31", "--promotion", str(tmp / "promotion.json"),
            "--out", str(out)]  # fmt: skip
    victim = next(r for r in rows if r["model"] == M2)
    path = tmp / "pilot" / "trials" / victim["trial"] / "score.json"
    path.write_text(json.dumps({**victim, "provider_pin": None}))  # an unpinned OpenRouter trial
    assert stage2.main(base) == 2 and "provider pin" in capsys.readouterr().err
    assert stage2.main([*base, "--lenient"]) == 0  # a dry run: written, flagged, not confirmatory
    sets = json.loads(out.read_text())
    assert sets["flags"] == ["lenient"] and sets["provenance_problems"]
    assert stage2.Registry.from_sets(sets).flags == ("lenient",)
    path.write_text(json.dumps(victim))
    gone = tmp / "pilot" / "trials" / next(r for r in rows if r["model"] == M1)["trial"]
    (gone / "score.json").unlink()
    assert stage2.main(base) == 2 and "incomplete pilot" in capsys.readouterr().err
    assert stage2.main([*base, "--allow-incomplete"]) == 0
    assert json.loads(out.read_text())["flags"] == ["incomplete"]


def test_the_sets_command_refuses_exposed_trials_without_a_replay(pilot_dir, capsys):
    tmp, _rows = pilot_dir
    replay_path = tmp / "replay.json"
    replays = json.loads(replay_path.read_text())
    replay_path.write_text(json.dumps([r for r in replays if r["family"] != "F32"]))
    args = ["sets", "--pilot", str(tmp / "pilot"), "--replay", str(replay_path),
            "--families", "F30", "F32", "--promotion", str(tmp / "promotion.json"),
            "--out", str(tmp / "sets.json")]  # fmt: skip
    assert stage2.main(args) == 2 and "undecided" in capsys.readouterr().err
    assert Path(tmp / "sets.json").exists() is False


# ---- audit of the stage-2 fixes: descriptive outputs and provenance -------------------------


@pytest.mark.parametrize(
    ("message", "truth", "found"),
    [
        ("The answer is 24.", {"kind": "int", "expected": 24}, True),
        ("It is 0.42.", {"kind": "float", "expected": 0.42}, True),
        ("a1 and b2.", {"kind": "str_list", "expected": ["a1", "b2"]}, True),
        ("version 24.5 is out", {"kind": "int", "expected": 24}, False),
        ("x240 is a name", {"kind": "int", "expected": 24}, False),
        ("value 0.42.7", {"kind": "float", "expected": 0.42}, False),
    ],
)
def test_the_disclosure_regex_accepts_a_sentence_final_value_and_rejects_part_of_a_longer_token(
    message, truth, found
):
    assert stage2.contains_expected(message, truth) is found
