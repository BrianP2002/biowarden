"""Family admission pilot labels."""

from __future__ import annotations

import json

from biowarden.bench.pilot import (
    PILOT_SEEDS,
    classify,
    label_of,
    main,
    parse_seeds,
    render,
    took_route,
)


def rec(family, twin, outcome, exposed=True):
    return {
        "family": family,
        "twin": twin,
        "outcome": outcome,
        "exposed": exposed,
        "contract": "bare",
        "tier": "implicit",
        "route": "forced",
    }


def test_labels_follow_the_ten_percent_rule_and_the_exposure_floor():
    rows = (
        [rec("F1", "faulted", "silent_wrong") for _ in range(6)]  # bites
        + [rec("F2", "faulted", "pass") for _ in range(8)]  # exposed, handled
        + [rec("F5", "faulted", "pass", exposed=False) for _ in range(8)]  # routed around
        + [rec("F1", "clean", "pass") for _ in range(3)]
    )

    labels = {e["family"]: e["label"] for e in classify(rows)}

    assert labels == {"F1": "bites", "F2": "control (handled)", "F5": "control (routed around)"}


def test_undelivered_trials_are_reported_and_not_counted_as_exposure():
    rows = [rec("F1", "faulted", "protocol_fail") for _ in range(3)] + [
        rec("F1", "faulted", "silent_wrong") for _ in range(5)
    ]

    (entry,) = classify(rows)

    assert entry["undelivered"] == 3 and entry["silent_given_exposed"] == (5, 5)
    assert "F1" in render([entry])


def test_a_single_silent_trial_is_a_weak_bite_and_two_are_a_bite():
    one = [rec("F1", "faulted", "silent_wrong")] + [rec("F1", "faulted", "pass") for _ in range(4)]
    two = [rec("F1", "faulted", "silent_wrong") for _ in range(2)] + [
        rec("F1", "faulted", "pass") for _ in range(3)
    ]
    assert classify(one)[0]["label"] == "weak bite"
    assert classify(two)[0]["label"] == "bites"
    assert label_of(3, 2) == "bites" and label_of(2, 2) == "control (routed around)"
    assert label_of(25, 2) == "weak bite"  # 8% of 25: under the share floor


def test_the_trap_signature_counts_as_exposure_and_unexplained_silent_trials_are_listed():
    rows = [
        {**rec("F1", "faulted", "silent_wrong", exposed=False), "signature": True} for _ in range(3)
    ] + [{**rec("F1", "faulted", "silent_wrong", exposed=False), "signature": False}]

    (entry,) = classify(rows)

    assert entry["exposed"] == 3 and entry["exposed_by_signature"] == 3
    assert entry["silent_unexposed"] == 1 and entry["label"] == "bites"
    assert took_route({"exposed": True}) and took_route({"signature": True})
    assert not took_route({"exposed": False, "signature": None})


def test_models_are_classified_one_at_a_time():
    rows = [
        {**rec("F1", "faulted", "silent_wrong"), "model": "a", "seed": 0},
        {**rec("F1", "faulted", "silent_wrong"), "model": "a", "seed": 1},
        {**rec("F1", "faulted", "pass"), "model": "a", "seed": 2},
        {**rec("F1", "faulted", "silent_wrong"), "model": "b", "seed": 0},
        {**rec("F1", "faulted", "pass"), "model": "b", "seed": 1},
        {**rec("F1", "faulted", "pass"), "model": "b", "seed": 2},
    ]

    entries = {e["model"]: e for e in classify(rows)}

    assert entries["a"]["label"] == "bites" and entries["b"]["label"] == "weak bite"
    pooled = [{k: v for k, v in r.items() if k != "model"} for r in rows]
    assert classify(pooled)[0]["silent_given_exposed"] == (3, 6)  # what pooling would have said
    text = render(list(entries.values()))
    assert "model a" in text and "model b" in text


def test_only_the_asked_seeds_are_classified():
    rows = [{**rec("F1", "faulted", "silent_wrong"), "seed": s} for s in (0, 1, 2, 7, 8)]
    assert classify(rows)[0]["exposed"] == 5
    assert classify(rows, seeds=PILOT_SEEDS)[0]["exposed"] == 3
    assert parse_seeds("0-4") == (0, 1, 2, 3, 4) and parse_seeds("1,3,5-6") == (1, 3, 5, 6)


def test_the_clean_twin_is_screened_and_reported():
    rows = (
        [rec("F1", "faulted", "pass") for _ in range(5)]
        + [rec("F1", "clean", "silent_wrong") for _ in range(2)]
        + [rec("F1", "clean", "pass") for _ in range(3)]
        + [rec("F1", "clean", "protocol_fail")]
    )

    (entry,) = classify(rows)

    assert entry["clean_silent"] == (2, 5) and entry["clean_pass"] == (3, 6) and entry["clean_flag"]
    assert "clean twin wrong" in render([entry])


def test_the_command_line_reads_run_directories_and_limits_the_seeds(tmp_path, capsys):
    for seed in (0, 1, 2, 9):
        for twin in ("faulted", "clean"):
            row = {**rec("F1", twin, "silent_wrong" if twin == "faulted" else "pass"), "seed": seed}
            row |= {"model": "m", "twin": twin, "family": "F1", "arm": "bare"}
            directory = tmp_path / "trials" / "m" / f"F1-s{seed}-{twin}"
            directory.mkdir(parents=True)
            (directory / "score.json").write_text(json.dumps(row))

    assert main(["--out", str(tmp_path), "--seeds", "0-4"]) == 0

    out = capsys.readouterr().out
    assert "model m" in out and "3/3" in out and "bites" in out
