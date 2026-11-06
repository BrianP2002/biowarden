"""The rule engine core: calls, argv readers, spec validation, modes, retry cap, admission maths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import Cell, Stats
from biowarden.engine import (
    Call,
    CheckResult,
    Engine,
    Promotion,
    Spec,
    checker,
)
from biowarden.engine.argv import first_positional, has_option, option_values
from biowarden.engine.call import DeclarationError
from biowarden.engine.decide import BlockState
from biowarden.engine.render import render_block, render_warning
from biowarden.ledger import Ledger, verify_ledger

REGIONS_RULE = {
    "id": "test.regions",
    "slot": "interval_semantics",
    "kind": "call",
    "tier": "T0",
    "applies": {"tools": ["bcftools"], "subcommands": ["view"], "options_any": ["-R", "-T"]},
    "declare": {"key": "region_semantics", "values": ["overlap", "start_only"]},
    "check": "option_implies_declared",
    "params": {
        "option_map": {"-R": "overlap", "-T": "start_only"},
        "fix_option": {"overlap": "-R", "start_only": "-T"},
    },
    "message": "-R selects overlapping records, -T only records that start inside a region",
}


def spec_of(*rules: dict) -> Spec:
    return Spec.model_validate({"name": "test", "rules": list(rules)})


def call(argv: list[str], declared: dict[str, str] | None = None, cwd: Path = Path("/tmp")) -> Call:
    return Call("bcftools", tuple(argv), cwd, declared or {})


# -- calls and argv ------------------------------------------------------------------------


def test_declarations_are_split_from_argv_and_override_the_environment():
    built = Call.from_argv(
        "/usr/bin/bcftools",
        ["--declare", "a=1", "view", "--declare=b=2", "-R", "r.bed"],
        "/work",
        env={"BIOWARDEN_DECLARE": "a=0;c=3"},
    )

    assert built.tool == "bcftools"
    assert built.argv == ("view", "-R", "r.bed")
    assert dict(built.declared) == {"a": "1", "b": "2", "c": "3"}


@pytest.mark.parametrize("bad", [["--declare"], ["--declare", "novalue"], ["--declare=k="]])
def test_malformed_declarations_are_rejected(bad):
    with pytest.raises(DeclarationError):
        Call.from_argv("bcftools", bad, "/work", env={})


def test_call_hash_changes_with_the_declaration():
    assert call(["view"]).call_hash() == call(["view"]).call_hash()
    assert call(["view"]).call_hash() != call(["view"], {"k": "v"}).call_hash()


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["view", "-R", "a.bed"], ["a.bed"]),
        (["view", "-Ra.bed"], ["a.bed"]),
        (["view", "--regions-file", "a.bed"], ["a.bed"]),
        (["view", "--regions-file=a.bed"], ["a.bed"]),
        (["view", "-R"], []),
        (["view", "--", "-R", "a.bed"], []),
        (["view", "-T", "t.bed"], []),
    ],
)
def test_option_values_reads_every_spelling(argv, expected):
    assert option_values(argv, "-R", "--regions-file") == expected


def test_has_option_and_first_positional():
    assert has_option(["view", "-HR", "x"], "-H") is True
    assert has_option(["view", "--regions-file=x"], "--regions-file") is True
    assert has_option(["view", "x"], "-R") is False
    assert first_positional(["-q", "view", "x"]) == "view"
    assert first_positional(["--version"]) is None


# -- spec validation -----------------------------------------------------------------------


def test_spec_shape_rules():
    spec_of(REGIONS_RULE)
    no_declare = {k: v for k, v in REGIONS_RULE.items() if k != "declare"}
    with pytest.raises(ValidationError, match="needs a declaration"):
        spec_of(no_declare)
    with pytest.raises(ValidationError, match="duplicate rule id"):
        spec_of(REGIONS_RULE, REGIONS_RULE)
    fact_t0 = {
        **no_declare,
        "kind": "fact",
        "applies": {"file_patterns": [r"\w+\.fq"]},
        "declare": {"key": "k", "values": ["a"]},
    }
    with pytest.raises(ValidationError, match="cannot be tier T0"):
        spec_of(fact_t0)
    with pytest.raises(ValidationError, match="needs applies.tools"):
        spec_of({**REGIONS_RULE, "applies": {}})


def test_declaration_schema_lists_enums_and_required_call_keys():
    schema = spec_of(REGIONS_RULE).declaration_schema()

    assert schema["properties"]["region_semantics"]["enum"] == ["overlap", "start_only"]
    assert schema["required"] == ["region_semantics"]
    assert schema["additionalProperties"] is False


def test_unknown_checker_name_is_rejected_when_the_engine_is_built():
    with pytest.raises(ValueError, match="unknown checkers"):
        Engine([spec_of({**REGIONS_RULE, "check": "no_such_checker"})])


# -- decisions -----------------------------------------------------------------------------

GOOD = call(["view", "-R", "r.bed"], {"region_semantics": "overlap"})
BAD = call(["view", "-T", "r.bed"], {"region_semantics": "overlap"})
UNDECLARED = call(["view", "-R", "r.bed"])


@pytest.mark.parametrize(
    "mode,expected",
    [("off", "allow"), ("warn", "warn"), ("block", "block"), ("promoted", "warn")],
)
def test_a_failing_rule_is_handled_per_mode(mode, expected):
    engine = Engine([spec_of(REGIONS_RULE)], mode=mode)

    assert engine.evaluate(BAD).action == expected
    assert engine.evaluate(GOOD).action == "allow"


def test_promoted_mode_blocks_only_rules_the_table_promotes():
    promotion = Promotion({"test.regions": {"mode": "block"}})
    engine = Engine([spec_of(REGIONS_RULE)], mode="promoted", promotion=promotion)

    assert engine.evaluate(BAD).action == "block"


def test_a_missing_declaration_is_a_failure_with_the_declaration_as_the_fix():
    decision = Engine([spec_of(REGIONS_RULE)], mode="block").evaluate(UNDECLARED)

    (outcome,) = decision.firing()
    assert outcome.result.code == "declaration_missing"
    assert "--declare region_semantics=<overlap|start_only>" in render_block(decision)


def test_an_invalid_declaration_value_is_a_failure():
    bad_value = call(["view", "-R", "r.bed"], {"region_semantics": "whatever"})

    (outcome,) = Engine([spec_of(REGIONS_RULE)], mode="block").evaluate(bad_value).firing()

    assert outcome.result.code == "declaration_invalid"


def test_the_fix_leads_the_block_message_and_names_the_matching_option():
    text = render_block(Engine([spec_of(REGIONS_RULE)], mode="block").evaluate(BAD))

    assert text.splitlines()[0] == "FIX: use -R instead"
    assert "BLOCKED, the command did not run" in text


def test_warnings_are_rendered_and_never_stop_the_call():
    decision = Engine([spec_of(REGIONS_RULE)], mode="warn").evaluate(BAD)

    assert decision.action == "warn"
    assert render_warning(decision).startswith("WARNING test.regions")


def test_unrelated_calls_do_not_match():
    engine = Engine([spec_of(REGIONS_RULE)], mode="block")

    assert engine.evaluate(Call("samtools", ("view", "-R", "x"), Path("/tmp"))).outcomes == ()
    assert engine.evaluate(call(["query", "-R", "x"])).outcomes == ()
    assert engine.evaluate(call(["view", "x.vcf"])).outcomes == ()


@checker("test_unknown_checker")
def _always_unknown(call, match, ctx):
    return CheckResult("unknown", "cannot_tell", "cannot tell from here")


@checker("test_crashing_checker")
def _crashes(call, match, ctx):
    raise RuntimeError("boom")


@checker("test_reads_by_name")
def _reads_by_name(call, match, ctx):
    return CheckResult("pass", "ok", note=next(ctx.lines("a.dat")))


@checker("test_reads_a_file")
def _reads(call, match, ctx):
    first = next(ctx.lines(match.files[0]))
    return CheckResult("fail", "risky", f"first line {first!r}")


def fact_rule(check: str, tier: str = "T1", **extra) -> dict:
    return {
        "id": f"test.{check}",
        "slot": "units",
        "kind": "fact",
        "tier": tier,
        "applies": {"file_patterns": [r"[\w./-]+\.dat"]},
        "check": check,
        "message": "m",
        **extra,
    }


def test_unknown_only_warns_even_in_block_mode(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact_rule("test_unknown_checker"))], mode="block")

    decision = engine.evaluate(Call("awk", ("{print}", "a.dat"), tmp_path))

    assert decision.action == "warn"
    assert render_warning(decision).startswith("UNVERIFIED")


def test_a_t2_rule_never_blocks(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact_rule("test_reads_a_file", tier="T2"))], mode="block")

    # a T2 checker may not read files, so it crashes, and even a crash only warns for T2
    assert engine.evaluate(Call("awk", ("a.dat",), tmp_path)).action == "warn"


def test_a_crashing_checker_fails_closed_in_block_and_promoted_modes(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    spec = spec_of(fact_rule("test_crashing_checker"))
    run = Call("awk", ("a.dat",), tmp_path)

    assert Engine([spec], mode="block").evaluate(run).action == "block"
    assert Engine([spec], mode="promoted").evaluate(run).action == "block"
    assert Engine([spec], mode="warn").evaluate(run).action == "warn"


def test_fact_rules_find_file_names_inside_code_strings(tmp_path):
    (tmp_path / "reads.dat").write_text("first\n")
    engine = Engine([spec_of(fact_rule("test_reads_a_file"))], mode="block")

    hit = engine.evaluate(Call("python3", ("-c", "print(open('reads.dat').read())"), tmp_path))
    miss = engine.evaluate(Call("python3", ("-c", "print(open('other.dat').read())"), tmp_path))

    assert hit.action == "block" and hit.outcomes[0].files == (str(tmp_path / "reads.dat"),)
    assert miss.outcomes == ()


def test_a_t0_checker_cannot_read_files(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    rule = {**fact_rule("test_reads_by_name"), "kind": "call", "tier": "T0"}
    rule["applies"] = {"tools": ["awk"]}
    rule["declare"] = {"key": "k", "values": ["v"]}

    decision = Engine([spec_of(rule)], mode="block").evaluate(
        Call("awk", ("a.dat",), tmp_path, {"k": "v"})
    )

    assert decision.outcomes[0].result.code == "checker_crashed"
    assert "may not read files" in decision.outcomes[0].result.message


def test_the_same_call_blocked_repeatedly_ends_in_a_stop_message():
    engine = Engine([spec_of(REGIONS_RULE)], mode="block", retry_cap=2)

    decisions = [engine.evaluate(BAD) for _ in range(3)]

    assert [d.stop for d in decisions] == [False, False, True]
    assert all(d.action == "block" for d in decisions)
    assert "needs_input" in render_block(decisions[2])
    assert "needs_input" not in render_block(decisions[0])


def test_block_state_is_shared_through_a_file(tmp_path):
    a, b = BlockState(tmp_path / "s.json"), BlockState(tmp_path / "s.json")

    a.record("h")
    b.record("h")

    assert a.count("h") == b.count("h") == 2


def test_decisions_are_written_to_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    engine = Engine([spec_of(REGIONS_RULE)], mode="block", ledger=ledger)

    engine.evaluate(BAD)

    record = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert record["event"] == "engine_decision" and record["action"] == "block"
    assert record["outcomes"][0]["rule"] == "test.regions"
    assert verify_ledger(tmp_path / "l.jsonl").ok


def test_promotion_round_trips_through_json(tmp_path):
    promotion = Promotion({"r": {"mode": "block"}})
    promotion.dump(tmp_path / "p.json")

    assert Promotion.load(tmp_path / "p.json").mode_of("r") == "block"
    assert Promotion().mode_of("unseen") == "warn"


# -- admission arithmetic ------------------------------------------------------------------


def stats_with(right_declared: tuple[int, int], wrong_failed: int = 5) -> Stats:
    stats = Stats("r", seeds=1)
    stats.cells[("right", True)] = Cell(n=right_declared[1], failed=right_declared[0])
    stats.cells[("wrong", False)] = Cell(n=10, failed=wrong_failed)
    return stats


def test_promotion_needs_enough_calls_and_a_small_upper_bound():
    assert stats_with((0, 200)).promotion_entry()["mode"] == "block"
    assert stats_with((0, 199)).promotion_entry()["mode"] == "warn"
    assert (
        stats_with((1, 400)).promotion_entry()["mode"] == "warn"
    )  # admitted needs zero false fails


def test_a_rule_that_kills_nothing_is_not_admitted():
    assert stats_with((0, 300), wrong_failed=0).admitted() is False
    assert stats_with((0, 300), wrong_failed=0).promotion_entry()["mode"] == "warn"


def test_upper_bound_for_zero_failures_in_200_calls_is_about_1_8_percent():
    assert 0.017 < Cell(n=200, failed=0).upper() < 0.019


def test_a_conditional_rule_is_also_gated_on_friction_for_correct_undeclared_calls():
    stats = stats_with((0, 300))
    stats.conditional = True
    stats.cells[("right", False)] = Cell(n=300, failed=3)

    assert stats.admitted() is False
    assert stats.promotion_entry()["mode"] == "warn"

    stats.cells[("right", False)] = Cell(n=300, failed=0)
    assert stats.promotion_entry()["mode"] == "block"


def test_a_relative_working_directory_does_not_double_the_path(tmp_path, monkeypatch):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.dat").write_text("first\n")
    monkeypatch.chdir(tmp_path)
    engine = Engine([spec_of(fact_rule("test_reads_a_file"))], mode="block")

    decision = engine.evaluate(Call("awk", ("a.dat",), Path("w")))

    assert decision.action == "block"
    assert decision.outcomes[0].result.message == "first line 'first'"
    assert Call.from_argv("awk", ["x"], "w").cwd == tmp_path / "w"


def test_a_huge_argument_is_scanned_quickly(tmp_path):
    import time

    from biowarden.engine.spec import packaged_specs

    engine = Engine(packaged_specs(), mode="warn")
    huge = "x" * 40_000
    started = time.monotonic()
    engine.evaluate(Call("python3", ("-c", huge), tmp_path))

    assert time.monotonic() - started < 2.0


def test_generic_style_says_only_that_a_convention_may_be_misread_and_lets_a_repeat_through():
    from biowarden.engine.render import GENERIC_BLOCK, GENERIC_WARNING, render_warning

    engine = Engine([spec_of(REGIONS_RULE)], mode="block", style="generic")

    first = engine.evaluate(BAD)
    text = render_block(first, style="generic")
    second = engine.evaluate(BAD)

    assert first.action == "block" and text == GENERIC_BLOCK
    for secret in ("-R", "-T", "overlap", "start_only", "FIX", "region_semantics", "declare"):
        assert secret not in text
    assert second.action == "allow" and second.acknowledged is True and second.stop is False
    warn = Engine([spec_of(REGIONS_RULE)], mode="warn", style="generic").evaluate(BAD)
    assert render_warning(warn, style="generic") == GENERIC_WARNING


def test_full_style_is_unchanged_by_the_generic_option():
    engine = Engine([spec_of(REGIONS_RULE)], mode="block")

    assert [engine.evaluate(BAD).action for _ in range(3)] == ["block"] * 3


def test_option_value_rule_reads_the_value_of_the_option(tmp_path):
    rule = {
        "id": "test.strand",
        "slot": "strand",
        "kind": "call",
        "tier": "T0",
        "applies": {"tools": ["aligner"], "options_any": ["--strand"]},
        "declare": {"key": "library", "values": ["forward", "reverse"]},
        "check": "option_value_implies_declared",
        "params": {
            "option_values": {"--strand": {"F": "forward", "R": "reverse"}},
            "fix_value": {"forward": "--strand F", "reverse": "--strand R"},
        },
        "message": "--strand states the library strand",
    }
    engine = Engine([spec_of(rule)], mode="block")

    def decide(value, declared):
        return engine.evaluate(
            Call("aligner", ("--strand", value), tmp_path, {"library": declared})
        )

    assert decide("F", "forward").action == "allow"
    bad = decide("F", "reverse")
    assert bad.action == "block" and render_block(bad).startswith("FIX: use --strand R instead")
    assert decide("X", "forward").action == "warn"  # a value the rule does not know only warns


def test_ctx_reads_raw_bytes_only_for_tier_t1(tmp_path):
    from biowarden.engine.checks import Ctx, TierViolation

    (tmp_path / "a.bin").write_bytes(b"\x1f\x8b" + b"x" * 100)

    assert Ctx(tmp_path, "T1").read_bytes("a.bin", 4) == b"\x1f\x8bxx"
    with pytest.raises(TierViolation):
        Ctx(tmp_path, "T0").read_bytes("a.bin")


def test_single_dash_long_options_are_not_split_into_a_short_option_and_a_value():
    assert option_values(["intersect", "-bed", "x.bed"], "-b") == ["ed"]  # the default, for -Rfile
    assert option_values(["intersect", "-bed", "x.bed"], "-b", attached=False) == []
    assert (
        has_option(["-bed"], "-b") is True and has_option(["-bed"], "-b", attached=False) is False
    )


def test_an_optional_declaration_that_was_not_made_passes_the_option_rules(tmp_path):
    rule = {**REGIONS_RULE, "declare": {**REGIONS_RULE["declare"], "required": False}}

    decision = Engine([spec_of(rule)], mode="block").evaluate(
        call(["view", "-T", "r.bed"], cwd=tmp_path)
    )

    assert decision.action == "allow"
