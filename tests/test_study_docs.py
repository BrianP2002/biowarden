"""docs/real-code-scan.md says what docs/results/real-code-scan.json holds."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from biowarden.engine.spec import packaged_specs

ROOT = Path(__file__).resolve().parents[1]
ASKS = ("declaration_missing", "declaration_invalid")
PAGE = (ROOT / "docs" / "real-code-scan.md").read_text()
DATA = json.loads((ROOT / "docs" / "results" / "real-code-scan.json").read_text())
FLAT = " ".join(PAGE.split())


def number(text: str) -> int:
    return int(text.replace(",", ""))


def rows_under(heading: str) -> list[list[str]]:
    """The cells of the first table after `heading`, without the header and the rule line."""
    start = PAGE.index(heading)
    lines = PAGE[start:].splitlines()
    table = [ln for ln in lines[next(i for i, ln in enumerate(lines) if ln.startswith("|")) :]]
    out = []
    for line in table:
        if not line.startswith("|"):
            break
        out.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return out[2:]


def test_the_results_table_is_the_aggregate():
    rows = {row[0].split()[0]: row for row in rows_under("## Results")}
    total = {"ok": 0, "dist": 0, "calls": 0, "allow": 0, "warn": 0, "block": 0}

    for name, suite in DATA["suites"].items():
        _, ok, disturbed, calls, allowed, warned, blocked = rows[name]
        actions = suite["calls_by_action"]
        assert ok == f"{suite['baseline_ok']:,} of {suite['tests']:,}", name
        assert number(disturbed) == suite["disturbed"], name
        assert number(calls) == suite["calls_checked"] == sum(actions.values()), name
        assert [number(allowed), number(warned), number(blocked)] == [
            actions.get("allow", 0),
            actions.get("warn", 0),
            actions.get("block", 0),
        ], name
        assert suite["version"] in rows[name][0] and suite["flaky"] == 0
        total["ok"] += suite["baseline_ok"]
        total["dist"] += suite["disturbed"]
        total["calls"] += suite["calls_checked"]
        for key in ("allow", "warn", "block"):
            total[key] += actions.get(key, 0)

    _, ok, disturbed, calls, allowed, warned, blocked = rows["all"]
    assert [number(c) for c in (ok, disturbed, calls, allowed, warned, blocked)] == [
        total["ok"],
        total["dist"],
        total["calls"],
        total["allow"],
        total["warn"],
        total["block"],
    ]
    share = {k: f"{100 * total[k] / total['calls']:.1f}%" for k in ("allow", "warn", "block")}
    assert f"{total['calls']:,} calls the layer checked, {share['allow']} passed untouched" in FLAT
    assert f"{share['warn']} got a warning and {share['block']} were blocked" in FLAT
    assert f"{total['dist']} of the {total['ok']:,} tests that pass without the layer" in FLAT
    assert f"({100 * total['dist'] / total['ok']:.1f}%)" in FLAT


def blocking(rule: str) -> tuple[int, set[str]]:
    count, codes = 0, set()
    for suite in DATA["suites"].values():
        for key, times in suite["firing"].items():
            name, code, effect = key.split("|")
            if name == rule and effect == "block":
                count += times
                codes.add(code)
    return count, codes


def test_the_blocks_are_counted_by_rule_and_kind():
    rows = rows_under("### What blocked")
    requests = findings = 0

    for rule, kind, calls, _ in rows:
        count, codes = blocking(rule.strip("`"))
        assert number(calls) == count and count > 0, rule
        assert (kind == "request") == bool(codes & set(ASKS)), rule
        requests += count if kind == "request" else 0
        findings += count if kind == "finding" else 0

    blocked = sum(s["calls_by_action"].get("block", 0) for s in DATA["suites"].values())
    assert requests + findings == blocked
    assert f"{requests} of the {blocked} blocks are requests and {findings} are findings" in FLAT
    everything = {
        k.split("|")[0]
        for s in DATA["suites"].values()
        for k in s["firing"]
        if k.endswith("|block")
    }
    assert everything == {row[0].strip("`") for row in rows}


def test_the_warnings_are_counted_by_rule():
    section = FLAT[FLAT.index("### Warnings") : FLAT.index("### Time")]
    listed = {rule: int(n) for rule, n in re.findall(r"`([a-z0-9_.]+)` \((\d+)\)", section)}
    fired: dict[str, int] = {}
    for suite in DATA["suites"].values():
        for key, times in suite["firing"].items():
            name, _, effect = key.split("|")
            if effect == "warn":
                fired[name] = fired.get(name, 0) + times
    warned = sum(s["calls_by_action"].get("warn", 0) for s in DATA["suites"].values())

    assert listed == fired
    assert f"{warned} calls got a warning and ran" in section


def test_the_times_are_the_aggregates_rounded():
    section = FLAT[FLAT.index("### Time") : FLAT.index("## Reproduce")]
    ms = {name: round(s["ms_added_per_call"]) for name, s in DATA["suites"].items()}

    assert f"about {ms['bedtools']} ms to each call it checked for bedtools" in section
    assert f"{ms['samtools']} ms for samtools and {ms['bcftools']} ms for bcftools" in section


def test_the_run_is_the_shipped_layer_in_its_default_mode():
    rules = sum(len(spec.rules) for spec in packaged_specs())
    environment = DATA["environment"]

    assert environment["rules"] == rules and environment["mode"] == "promoted"
    assert f"frozen rules ({rules}, all promoted" in FLAT
    assert DATA["sandbox"] is True
    for name in ("bedtools", "samtools", "bcftools"):
        assert DATA["suites"][name]["version"] in environment[name]


# ---- docs/blind-coverage.md and docs/blind-cases/ ------------------------------------------------

CASES = ROOT / "docs" / "blind-cases"
RESULT = ROOT / "docs" / "results" / "blind-coverage.json"


def blind_tool():
    spec = importlib.util.spec_from_file_location(
        "blind_coverage", ROOT / "tools" / "blind_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blind_coverage"] = module
    spec.loader.exec_module(module)
    return module


BC = blind_tool()
BLIND = json.loads(RESULT.read_text())
BLIND_PAGE = (ROOT / "docs" / "blind-coverage.md").read_text()
BLIND_FLAT = " ".join(BLIND_PAGE.split())
CASE_FILES = sorted(p for p in CASES.glob("*.json") if re.fullmatch(r"[a-z]+-\d+", p.stem))
REVIEWS = json.loads((CASES / "reviews.json").read_text())
LABELS = json.loads((CASES / "labels.json").read_text())
ACCEPTED = json.loads((CASES / "accepted.json").read_text())
VERIFIED = json.loads((CASES / "verified.json").read_text())["cases"]


def test_every_case_file_is_a_case_the_harness_can_run():
    assert CASE_FILES
    for path in CASE_FILES:
        case = json.loads(path.read_text())
        assert BC.check_case(case, path.stem) == [], path.name


def test_the_decisions_are_the_reviews_and_the_labels_cover_the_accepted_cases():
    ids = {p.stem for p in CASE_FILES}

    assert {r["id"] for r in REVIEWS} == ids == {c["id"] for c in VERIFIED if c["verified"]}
    assert ACCEPTED == BC.accepted_ids(REVIEWS)
    assert set(LABELS) == set(ACCEPTED)
    families = set(re.findall(r"^\| (F\d+) \|", (ROOT / "docs" / "families.md").read_text(), re.M))
    assert {item["family"] for item in LABELS.values()} <= families | {""}


def test_the_replay_covers_the_verified_cases_and_the_shim_changed_no_result():
    assert {c["id"] for c in BLIND["cases"]} == {p.stem for p in CASE_FILES}
    for case in BLIND["cases"]:
        for which in ("faulted", "clean"):
            run = case[which]
            assert run["decision"] == "block" or run["same_result_as_unguarded"], (
                case["id"],
                which,
            )


def test_the_watched_programs_are_the_ones_the_layer_has_now():
    from biowarden.engine.tools import watched_tools

    assert BLIND["watched"] == sorted(watched_tools(packaged_specs()))


def test_the_page_table_is_the_summary_of_the_replay():
    on_target = {i: {"on_target": item["on_target"] == "yes"} for i, item in LABELS.items()}
    families = {i: item["family"] for i, item in LABELS.items()}
    clean = {i: item["clean_finding"] for i, item in LABELS.items()}

    text = BC.summary(BLIND, set(ACCEPTED), on_target, families, clean)

    for line in text.splitlines():
        if line.strip():
            assert line in BLIND_PAGE or " ".join(line.split()) in BLIND_FLAT, line


def test_the_page_states_the_counts_of_the_study():
    accepted = set(ACCEPTED)
    rows = [c for c in BLIND["cases"] if c["id"] in accepted]
    caught = sum(1 for c in rows if c["outcome"] == "caught")
    repeats = [c for c in rows if LABELS[c["id"]]["family"]]
    news = [c for c in rows if not LABELS[c["id"]]["family"]]
    caught_repeats = sum(1 for c in repeats if c["outcome"] == "caught")
    caught_new = sum(1 for c in news if c["outcome"] == "caught")

    assert f"{len(CASE_FILES)} pairs of commands" in BLIND_FLAT
    assert f"accepted {len(accepted)}" in BLIND_FLAT
    assert f"{caught} of the {len(rows)} faulted commands" in BLIND_FLAT
    assert f"{caught_repeats} of the {len(repeats)} that repeat" in BLIND_FLAT
    assert f"{caught_new} of the {len(news)} that do not" in BLIND_FLAT
    environment = BLIND["provenance"]
    assert sum(environment["slices"].values()) == len(CASE_FILES)
    assert environment["audit"]["flagged"] == 0


def test_the_rules_that_blocked_a_correct_command_all_take_a_declaration():
    rules = {r.id: r for spec in packaged_specs() for r in spec.rules}
    wrong = {
        finding[0]
        for case in BLIND["cases"]
        if case["id"] in set(ACCEPTED) and case["false_block"]
        for finding in case["clean"]["findings"]
    }

    assert len(wrong) == 4 and all(rules[rule].declare is not None for rule in wrong)
    assert "The four rules involved" in BLIND_FLAT


def test_the_page_states_the_audit_the_dropped_ideas_and_the_programs_without_a_shim():
    provenance = BLIND["provenance"]
    audit = provenance["audit"]
    unseen = {c["tool"] for c in BLIND["cases"] if c["faulted"]["calls_checked"] == 0}

    calls = f"{audit['authors']['tool_calls']} calls of the authors"
    assert f"{calls} and {audit['reviewers']['tool_calls']} of the reviewers" in BLIND_FLAT
    assert f"dropped {provenance['dropped_ideas']} ideas" in BLIND_FLAT
    assert unseen == {"STAR", "pdb_selatom", "pdb_reres"}
    assert all(name in BLIND_FLAT for name in unseen)
    assert BLIND["provenance"]["model"] == "claude-sonnet-5-5" and "Sonnet 5.5" in BLIND_FLAT


def test_the_one_new_trap_that_was_caught_belongs_to_a_rule_without_a_family():
    caught = [
        c for c in BLIND["cases"] if c["outcome"] == "caught" and not LABELS[c["id"]]["family"]
    ]
    rules = (ROOT / "docs" / "rules.md").read_text()

    assert [c["id"] for c in caught] == ["intervals-06"]
    row = next(line for line in rules.splitlines() if "bedops.element_of_amount" in line)
    assert caught[0]["faulted"]["findings"][0][0] == "bedops.element_of_amount"
    assert [cell.strip() for cell in row.strip("|").split("|")][6] == "none"
    assert "`intervals-06`" in BLIND_FLAT and "no family lists as its own" in BLIND_FLAT
