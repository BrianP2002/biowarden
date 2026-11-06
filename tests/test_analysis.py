"""The clean-versus-faulted comparison on synthetic trial records."""

from __future__ import annotations

import json

import pytest

from biowarden.bench.analysis import (
    cell_of,
    compare,
    efficiency,
    load_records,
    mcnemar_exact,
    render,
    render_efficiency,
    undelivered_kind,
)


def rec(twin, seed, outcome, model="m", **extra):
    base = {"family": "F1", "twin": twin, "seed": seed, "model": model, "outcome": outcome}
    return base | extra


def test_paired_discordance_counts_each_direction_and_ignores_undelivered_pairs():
    rows = []
    for seed in range(10):
        rows.append(rec("clean", seed, "pass"))
        rows.append(
            rec(
                "faulted",
                seed,
                "silent_wrong" if seed < 6 else "pass",
                signature=True,
                exposed=True,
            )
        )
    rows.append(rec("clean", 10, "silent_wrong"))
    rows.append(rec("faulted", 10, "pass"))
    rows.append(rec("clean", 11, "protocol_fail"))
    rows.append(rec("faulted", 11, "silent_wrong"))  # no delivered clean twin: not a pair

    [entry] = compare(rows)

    assert entry["faulted"] == (7, 12) and entry["clean"] == (1, 11)
    assert (entry["faulted_only"], entry["clean_only"]) == (6, 1)
    assert entry["not_delivered"] == (0, 12, 1, 12)
    assert entry["mcnemar_p"] == 0.125 and entry["signature_share"] == 6 / 7


def test_cells_are_kept_apart_and_f4_is_excluded():
    rows = [rec("faulted", 0, "pass", contract="bare", tier="implicit", route="forced")]
    rows += [rec("faulted", 0, "pass"), {**rec("faulted", 0, "silent_wrong"), "family": "F4"}]
    rows += [rec("faulted", 0, "pass", harness="opencode")]

    entries = compare(rows)

    assert {e["cell"] for e in entries} == {
        "bare/implicit/forced",
        "full/explicit/free",
        "opencode:full/explicit/free",
    }
    assert cell_of({"harness": "opencode"}) == "opencode:full/explicit/free"
    assert "F4" not in render(entries)


def test_mcnemar_exact_matches_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(0, 8) - 2 / 256) < 1e-12


def test_every_family_except_the_must_detect_one_enters_the_trap_comparison():
    rows = [
        rec("clean", 0, "pass", family="F9"),
        rec("faulted", 0, "silent_wrong", family="F9"),
        rec("faulted", 0, "silent_wrong", family="F4"),
    ]

    (entry,) = compare(rows)

    assert entry["faulted"] == (1, 1) and entry["clean"] == (0, 1)


def test_efficiency_counts_budgeted_successes_against_all_planned_trials():
    rows = [
        rec("clean", 0, "pass", correct=True, turns=5, usd=0.002, stop="end_turn"),
        rec("clean", 1, "pass", correct=True, turns=25, usd=0.004, stop="end_turn"),
        rec("clean", 2, "protocol_fail", correct=False, turns=30, usd=0.006, stop="max_turns"),
        rec("clean", 3, "infra_error"),
    ]

    (entry,) = efficiency(rows)

    assert (entry["n"], entry["correct"], entry["within_budget"], entry["cap_hit"]) == (3, 2, 1, 1)
    assert entry["median_turns"] == 25
    assert abs(entry["usd_per_correct"] - 0.006) < 1e-9
    assert "ok<=10t" in render_efficiency([entry])


def write_trial(root, model, name, record):
    directory = root / "trials" / model / name
    directory.mkdir(parents=True)
    (directory / "score.json").write_text(record if isinstance(record, str) else json.dumps(record))
    return directory / "score.json"


def test_records_are_loaded_in_path_order_not_in_file_system_order(tmp_path):
    for name in ("F30-s02", "F9-s01", "F30-s01"):
        write_trial(tmp_path, "m", name, {"trial": name, "outcome": "pass"})
    write_trial(tmp_path, "a", "F1-s00", {"trial": "a/F1-s00", "outcome": "pass"})

    rows = load_records([tmp_path])

    assert [r["trial"] for r in rows] == ["a/F1-s00", "F30-s01", "F30-s02", "F9-s01"]


def test_a_truncated_or_vanished_record_is_skipped_and_can_be_listed(tmp_path):
    write_trial(tmp_path, "m", "good", {"trial": "good", "outcome": "pass"})
    bad = write_trial(tmp_path, "m", "bad", '{"trial": "bad", "outc')
    write_trial(tmp_path, "m", "list", "[1, 2]")
    unreadable = []

    rows = load_records([str(tmp_path)], unreadable)

    assert [r["trial"] for r in rows] == ["good"] and bad in unreadable and len(unreadable) == 2
    assert load_records([tmp_path / "no-such-run"]) == []
    assert [r["trial"] for r in load_records([tmp_path])] == ["good"]  # tolerant without the list


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        ({"outcome": "refused"}, "refused"),
        ({"outcome": "format_error", "reason": "answer type"}, "wrong_type"),
        ({"outcome": "protocol_fail", "reason": "result.json missing"}, "no_file"),
        (
            {"outcome": "protocol_fail", "reason": "result.json is not valid JSON: x"},
            "invalid_json",
        ),
        ({"outcome": "protocol_fail", "reason": "result.json is not a JSON object"}, "not_object"),
        (
            {"outcome": "protocol_fail", "reason": "result.json unreadable: Too many levels"},
            "unreadable",
        ),
        ({"outcome": "protocol_fail", "reason": "result.json too large"}, "unreadable"),
        ({"outcome": "protocol_fail", "reason": "answer is missing"}, "no_answer"),
        (
            {"outcome": "protocol_fail", "reason": "status must be 'ok' or 'needs_input'"},
            "bad_shape",
        ),
        (
            {"outcome": "protocol_fail", "reason": "result.json missing", "stop": "max_turns"},
            "turn_cap",
        ),
        (
            {"outcome": "protocol_fail", "reason": "result.json missing", "stop": "budget"},
            "budget_cap",
        ),
        ({"outcome": "protocol_fail", "reason": "", "stop": "end_turn"}, "other"),
    ],
)
def test_an_undelivered_answer_is_filed_under_the_reason_it_did_not_arrive(record, kind):
    assert undelivered_kind(record) == kind
