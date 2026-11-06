"""tools/blind_coverage.py: case checks, ground truth, the layer's answer and the tables."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "blind_coverage", ROOT / "tools" / "blind_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blind_coverage"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bc():
    return load()


def case(**changes):
    base = {
        "id": "tst-01",
        "domain": "tst",
        "tool": "wc",
        "trap_class": "line count",
        "intent": "count the records",
        "silent_error": "the last line has no newline, so wc undercounts it",
        "files": {"in.txt": "1\n2\n3"},
        "setup": ["cp in.txt copy.txt"],
        "faulted": "wc -l < in.txt > n.txt",
        "clean": "awk 'END { print NR }' copy.txt > n.txt",
        "observe": "cat n.txt",
    }
    return {**base, **changes}


# ---- the case file ----------------------------------------------------------------------------


def test_a_well_formed_case_has_no_problems(bc):
    assert bc.check_case(case(), "tst-01") == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"id": "tst-02"}, "id is not the file name"),
        ({"id": "Tst_1"}, "id is not the file name"),
        ({"faulted": ""}, "faulted is not one command"),
        ({"clean": "x" * 2001}, "clean is not one command"),
        ({"clean": "wc -l < in.txt > n.txt"}, "the same command"),
        ({"files": {"../escape": "x"}}, "not a plain relative name"),
        ({"files": {"/abs": "x"}}, "not a plain relative name"),
        ({"files": {"a/b/c": "x"}}, "not a plain relative name"),
        ({"files": {"big.txt": "x" * 50_001}}, "up to 50000 characters"),
        ({"setup": "cp a b"}, "setup is not a list"),
        ({"observe": " "}, "observe is empty"),
    ],
)
def test_a_defective_case_is_named(bc, changes, fragment):
    problems = bc.check_case(case(**changes), "tst-01")

    assert any(fragment in p for p in problems), problems


def test_a_missing_field_is_named(bc):
    broken = case()
    del broken["observe"]

    assert bc.check_case(broken, "tst-01") == ["missing field observe"]


def test_cases_are_loaded_from_a_directory_and_the_rest_is_reported(bc, tmp_path):
    (tmp_path / "tst-01.json").write_text(json.dumps(case()))
    (tmp_path / "tst-02.json").write_text("{not json")
    (tmp_path / "tst-03.json").write_text(json.dumps(case(id="tst-09")))

    cases, rejected = bc.load_cases(tmp_path)

    assert [c["id"] for c in cases] == ["tst-01"]
    assert set(rejected) == {"tst-02", "tst-03"}


# ---- the ground truth (no layer) --------------------------------------------------------------


@pytest.fixture(scope="module")
def runner(bc):
    scan = bc.load_scan()
    return bc.Variant(scan, scan.tool_environment(None), None)


def test_a_silent_error_is_verified(bc, runner):
    entry = bc.verify_case(runner, case())

    assert entry["verified"] and entry["reasons"] == []
    assert entry["observed"] == {"faulted": "2", "clean": "3"}
    assert entry["tool_speaks"] is False


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"clean": "wc -l < in.txt > n.txt"}, None),  # the same command: refused by check_case
        ({"clean": "wc -l < in.txt > n.txt; true"}, "give the same observation"),
        ({"faulted": "exit 3"}, "faulted does not exit 0"),
        ({"observe": "echo $RANDOM$RANDOM$RANDOM"}, "differs between two runs"),
        ({"observe": "true"}, "observe prints nothing"),
        ({"setup": ["false"]}, "setup failed"),
    ],
)
def test_what_is_not_a_repeatable_silent_error_is_not_verified(bc, runner, changes, reason):
    if reason is None:
        assert bc.check_case(case(**changes), "tst-01")
        return

    entry = bc.verify_case(runner, case(**changes))

    assert not entry["verified"]
    assert any(reason in r for r in entry["reasons"]), entry["reasons"]


# ---- the layer's answer -----------------------------------------------------------------------


def run_with(rows=(), exit_code=0, stderr="", observed="2"):
    return {"ledger": list(rows), "exit": exit_code, "stderr": stderr, "observed": observed}


def ledger_row(action, *fired):
    outcomes = [
        {"rule": r, "code": c, "effective": e, "verdict": "fail", "note": ""} for r, c, e in fired
    ]
    return {"action": action, "tool": "bcftools", "argv": ["view"], "outcomes": outcomes}


FINDING = ("bcftools.regions_file_base", "bed_content_read_one_based", "block")
REQUEST = ("bcftools.region_semantics", "declaration_missing", "block")
WARNING = ("bedtools.merge_bookended", "merge_adjacent_undeclared", "warn")


def test_findings_requests_and_warnings_are_told_apart(bc):
    both = bc.decide(run_with([ledger_row("block", FINDING, REQUEST)], 97, "BLOCKED"))
    asked = bc.decide(run_with([ledger_row("block", REQUEST)], 97, "BLOCKED"))
    warned = bc.decide(run_with([ledger_row("warn", WARNING)]))
    quiet = bc.decide(run_with([ledger_row("allow")]))
    unseen = bc.decide(run_with([], 97, "BLOCKED"))

    assert both["decision"] == "block" and len(both["findings"]) == 1 and len(both["requests"]) == 1
    assert asked["decision"] == "block" and asked["findings"] == [] and len(asked["requests"]) == 1
    assert warned["decision"] == "warn" and warned["warnings"] == [list(WARNING)]
    assert quiet["decision"] == "allow" and quiet["calls_checked"] == 1
    assert unseen["decision"] == "block"  # exit 97 is a block even without a ledger row
    assert both["message"] == "BLOCKED" and quiet["message"] == ""


class Scripted:
    """A runner whose answers are given: (which variant) -> a guarded run."""

    def __init__(self, faulted, clean):
        self.answers = {"faulted": faulted, "clean": clean}

    def run(self, case, which, guarded):
        assert guarded
        return self.answers[which]


TRUTH = {"observed": {"faulted": "2", "clean": "3"}}
BLOCK = [ledger_row("block", FINDING)]
ASK = [ledger_row("block", REQUEST)]


@pytest.mark.parametrize(
    ("faulted", "clean", "outcome", "false_block", "clean_asked"),
    [
        (
            run_with(BLOCK, 97),
            run_with([ledger_row("allow")], observed="3"),
            "caught",
            False,
            False,
        ),
        (run_with(BLOCK, 97), run_with(ASK, 97), "caught", False, True),
        (run_with(BLOCK, 97), run_with(BLOCK, 97), "both", True, False),
        (run_with(ASK, 97), run_with(ASK, 97), "declaration", False, True),
        (
            run_with([ledger_row("warn", WARNING)]),
            run_with([], observed="3"),
            "warned",
            False,
            False,
        ),
        (run_with([ledger_row("allow")]), run_with([], observed="3"), "missed", False, False),
    ],
)
def test_the_outcome_of_a_case(bc, faulted, clean, outcome, false_block, clean_asked):
    entry = bc.replay_case(Scripted(faulted, clean), case(), TRUTH)

    assert entry["outcome"] == outcome
    assert entry["false_block"] is false_block and entry["clean_asked"] is clean_asked


def test_an_allowed_call_must_give_the_result_it_gives_without_the_layer(bc):
    same = bc.replay_case(Scripted(run_with(), run_with(observed="3")), case(), TRUTH)
    changed = bc.replay_case(
        Scripted(run_with(observed="9"), run_with(observed="3")), case(), TRUTH
    )

    assert same["faulted"]["same_result_as_unguarded"] and same["clean"]["same_result_as_unguarded"]
    assert not changed["faulted"]["same_result_as_unguarded"]


def test_a_real_guarded_run_of_an_unwatched_command_is_missed(bc, runner):
    """awk is shimmed but no rule applies to this call: the layer lets it through unchanged."""
    awk = case(
        faulted="awk 'BEGIN { print 2 }' > n.txt",
        clean="awk 'BEGIN { print 3 }' > n.txt",
        setup=[],
        tool="awk",
    )
    truth = bc.verify_case(runner, awk)

    entry = bc.replay_case(runner, awk, truth)

    assert truth["verified"]
    assert entry["outcome"] == "missed" and not entry["false_block"]
    assert entry["faulted"]["tools_seen"] == ["awk"]
    assert (
        entry["faulted"]["same_result_as_unguarded"] and entry["clean"]["same_result_as_unguarded"]
    )


# ---- the tables -------------------------------------------------------------------------------


def test_the_table_counts_outcomes_by_domain_with_an_interval(bc):
    def result(id_, domain, tool, outcome, false_block=False, clean_asked=False):
        return {
            "id": id_,
            "domain": domain,
            "tool": tool,
            "outcome": outcome,
            "false_block": false_block,
            "clean_asked": clean_asked,
            "faulted": {"calls_checked": 0 if tool == "mystery" else 1},
        }

    replay = {
        "watched": ["bcftools", "samtools"],
        "cases": [
            result("a-01", "a", "bcftools", "caught", clean_asked=True),
            result("a-02", "a", "bcftools", "missed"),
            result("b-01", "b", "samtools", "caught"),
            result("b-02", "b", "mystery", "missed"),
            result("b-03", "b", "samtools", "both", false_block=True),
        ],
    }

    table = bc.summary(replay).splitlines()
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in table[2:]
    }

    assert rows["a"][:2] == ["2", "1/2 (50%, 9-91%)"]
    assert rows["b"][:2] == ["3", "1/3 (33%, 6-79%)"]
    assert rows["all"][0] == "5" and rows["all"][1].startswith("2/5 (40%")
    assert rows["call seen by a shim"][0] == "4"
    assert rows["all"][-2:] == ["1", "1"]  # one false block, one clean call that was asked
    filtered = bc.summary(replay, accepted={"a-01", "a-02"}).splitlines()
    kept = {line.split("|")[1].strip(): line.split("|")[2].strip() for line in filtered[2:]}

    assert kept == {"a": "2", "all": "2", "call seen by a shim": "2"}


# ---- the reviewers ------------------------------------------------------------------------------


def review(id_, verdict="accept", duplicate_of="", **answers):
    base = {"faulted_wrong": True, "clean_right": True, "plausible": True, "silent": True}
    return {"id": id_, "verdict": verdict, "duplicate_of": duplicate_of, **base, **answers}


def test_a_case_is_accepted_only_when_every_answer_is_yes(bc):
    reviews = [
        review("a-01"),
        review("a-02", verdict="reject"),
        review("a-03", silent=False),  # an accept that contradicts the answers does not count
        review("a-04", plausible=False, verdict="reject"),
    ]

    assert bc.accepted_ids(reviews) == ["a-01"]


def test_accepted_duplicates_count_once(bc):
    reviews = [
        review("a-01"),
        review("a-02", duplicate_of="a-01"),
        review("a-03", duplicate_of="a-02"),
        review("a-04", duplicate_of="a-09"),  # the other case was rejected: this one stays
        review("a-09", verdict="reject"),
        review("b-01"),
    ]

    assert bc.accepted_ids(reviews) == ["a-01", "a-04", "b-01"]


def test_cases_that_repeat_a_family_trap_are_counted_apart(bc):
    def result(id_, outcome):
        return {
            "id": id_,
            "domain": "a",
            "tool": "bcftools",
            "outcome": outcome,
            "false_block": False,
            "clean_asked": False,
            "faulted": {"calls_checked": 1},
        }

    replay = {
        "watched": ["bcftools"],
        "cases": [result("a-01", "caught"), result("a-02", "caught"), result("a-03", "missed")],
    }

    table = bc.summary(replay, families={"a-01": "F30", "a-02": "", "a-03": ""}).splitlines()
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in table[2:]
    }

    assert rows["repeats a family trap"][:2] == ["1", "1/1 (100%, 21-100%)"]
    assert rows["new trap"][:2] == ["2", "1/2 (50%, 9-91%)"]


def test_findings_on_clean_commands_are_counted_by_judgement(bc):
    def result(id_, false_block):
        return {
            "id": id_,
            "domain": "a",
            "tool": "bcftools",
            "outcome": "both" if false_block else "missed",
            "false_block": false_block,
            "clean_asked": False,
            "faulted": {"calls_checked": 1},
        }

    replay = {
        "watched": ["bcftools"],
        "cases": [
            result("a-01", True),
            result("a-02", True),
            result("a-03", True),
            result("a-04", False),
        ],
    }
    clean = {"a-01": "valid", "a-02": "false_positive", "a-03": "valid", "a-04": "n/a"}

    text = bc.summary(replay, clean=clean)

    assert (
        "Findings on the clean command, judged (3): 2 valid, 1 false positive, 0 unclear." in text
    )
