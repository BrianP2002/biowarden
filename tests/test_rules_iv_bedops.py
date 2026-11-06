"""Rules bedops.sorted_input, bedops.bedmap_score_column, bedops.element_of_amount (pack iv).

Unit tests on small BED files written here, the real BEDOPS programs on the traps the rules name,
the F70 / F71 workspaces, and the admission measurement (calls labelled by running the real tools,
bench/admission/cases_iv_bedops.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SORTED_ID = "bedops.sorted_input"
SCORE_ID = "bedops.bedmap_score_column"
ELEMENT_ID = "bedops.element_of_amount"
VALUE_KEY = "bedmap_value_column"
AMOUNT_KEY = "overlap_amount"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or len(env.real_bins(["bedmap", "bedops", "sort-bed"])) < 3:
        pytest.skip("BEDOPS is not installed")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("iv_bedops")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def outcome(engine: Engine, rule_id: str, tool: str, argv, cwd: Path, declared=None):
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def write(path: Path, rows: list[str]) -> Path:
    path.write_text("".join(row + "\n" for row in rows))
    return path


SORT_BED = ["chr1\t100\t200\ta", "chr10\t100\t200\tb", "chr2\t100\t200\tc", "chr9\t5\t6\td"]
GENOME = ["chr1\t100\t200\ta", "chr2\t100\t200\tc", "chr9\t5\t6\td", "chr10\t100\t200\tb"]
BACKWARDS = ["chr1\t500\t600\ta", "chr1\t100\t200\tb"]
END_TIES = ["chr1\t100\t300\ta", "chr1\t100\t200\tb"]


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_has_the_three_rules(spec):
    assert spec.name == "iv_bedops"
    rules = {r.id: r for r in spec.rules}
    assert set(rules) == {SORTED_ID, SCORE_ID, ELEMENT_ID}
    assert (rules[SORTED_ID].kind, rules[SORTED_ID].tier, rules[SORTED_ID].declare) == (
        "call",
        "T1",
        None,
    )
    assert rules[SORTED_ID].applies.tools == ["bedmap", "bedops", "closest-features"]
    assert (rules[SCORE_ID].kind, rules[SCORE_ID].tier) == ("call", "T1")
    assert rules[SCORE_ID].declare is not None and rules[SCORE_ID].declare.required is False
    assert (rules[SCORE_ID].declare.key, rules[SCORE_ID].declare.values) == (VALUE_KEY, ["5"])
    assert (rules[ELEMENT_ID].kind, rules[ELEMENT_ID].tier) == ("call", "T0")
    assert rules[ELEMENT_ID].declare is not None and rules[ELEMENT_ID].declare.required is False
    engine_for(spec)  # imports the rule modules and checks the checker names
    for rule in spec.rules:
        assert rule.check in known_checkers()
    schema = spec.declaration_schema()
    assert schema["required"] == []
    assert set(schema["properties"]) == {VALUE_KEY, AMOUNT_KEY}


def test_spec_rejects_a_wrong_shape(spec):
    data = spec.model_dump(mode="json")
    data["rules"][2]["declare"] = None  # a T0 rule is decided from the call: it needs a declaration
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][0]["applies"]["tools"] = []
    with pytest.raises(ValidationError, match="needs applies.tools"):
        Spec.model_validate(data)


# ---- bedops.sorted_input -----------------------------------------------------------------------


def test_a_file_in_sort_bed_order_passes_for_every_bedops_program(spec, tmp_path):
    write(tmp_path / "a.bed", SORT_BED)
    write(tmp_path / "b.bed", SORT_BED)
    eng = engine_for(spec)
    for tool, argv in (
        ("bedmap", ["--echo", "--count", "a.bed", "b.bed"]),
        ("bedmap", ["--count", "a.bed"]),
        ("bedops", ["-m", "a.bed", "b.bed"]),
        ("bedops", ["-u", "a.bed", "b.bed"]),
        ("closest-features", ["a.bed", "b.bed"]),
    ):
        decision, result = outcome(eng, SORTED_ID, tool, argv, tmp_path)
        assert decision.action == "allow" and result.verdict == "pass", (tool, argv)


@pytest.mark.parametrize("tool", ["bedmap", "bedops", "closest-features"])
def test_genome_order_fails_with_the_chromosome_names_and_a_sort_bed_fix(spec, tmp_path, tool):
    write(tmp_path / "ref.bed", SORT_BED)
    write(tmp_path / "karyo.bed", GENOME)
    argv = (
        ["--echo", "--count", "ref.bed", "karyo.bed"]
        if tool == "bedmap"
        else ["ref.bed", "karyo.bed"]
    )
    if tool == "bedops":
        argv = ["-e", "1", "ref.bed", "karyo.bed"]
    decision, result = outcome(engine_for(spec), SORTED_ID, tool, argv, tmp_path)
    assert result.verdict == "fail" and result.code == "not_in_sort_bed_order"
    assert decision.action == "block"
    text = render_block(decision)
    assert text.splitlines()[0].startswith("FIX: sort-bed karyo.bed > karyo.sorted.bed")
    for needle in ("karyo.bed", "data line 4", "chr10", "chr9", "as text", "--ec"):
        assert needle in text, needle


def test_a_start_that_goes_backwards_or_an_end_tie_in_the_wrong_order_fails(spec, tmp_path):
    eng = engine_for(spec)
    for name, rows, line in (
        ("back.bed", BACKWARDS, "data line 2"),
        ("tie.bed", END_TIES, "data line 2"),
    ):
        write(tmp_path / name, rows)
        _, result = outcome(eng, SORTED_ID, "bedmap", ["--count", name, "other.bed"], tmp_path)
        assert result.verdict == "fail" and line in result.message
        assert "must not go backwards" in result.message


def test_ec_makes_the_call_loud_so_the_rule_stays_silent(spec, tmp_path):
    write(tmp_path / "karyo.bed", GENOME)
    argv = ["--ec", "--echo", "--count", "karyo.bed"]
    assert outcome(engine_for(spec), SORTED_ID, "bedmap", argv, tmp_path)[1].verdict == "pass"


def test_only_existing_text_files_are_judged(spec, tmp_path):
    write(tmp_path / "karyo.bed", GENOME)
    eng = engine_for(spec)
    assert (
        outcome(eng, SORTED_ID, "bedmap", ["--count", "-", "absent.bed"], tmp_path)[1].verdict
        == "pass"
    )
    assert outcome(eng, SORTED_ID, "bedmap", ["--count", "a.starch"], tmp_path)[1].verdict == "pass"
    write(tmp_path / "x.starch", GENOME)
    assert outcome(eng, SORTED_ID, "bedmap", ["--count", "x.starch"], tmp_path)[1].verdict == "pass"
    write(
        tmp_path / "hdr.bed",
        ["track name=x", "#comment", "", "browser position chr1:1-2", *SORT_BED],
    )
    assert outcome(eng, SORTED_ID, "bedmap", ["--count", "hdr.bed"], tmp_path)[1].verdict == "pass"
    (tmp_path / "text.txt").write_text("not a bed file\nat all\n")
    assert outcome(eng, SORTED_ID, "bedmap", ["--count", "text.txt"], tmp_path)[1].verdict == "pass"
    # an option value that is not a file never counts
    assert (
        outcome(eng, SORTED_ID, "bedmap", ["--delim", ";", "--count", "karyo.bed"], tmp_path)[
            1
        ].verdict
        == "fail"
    )


def test_the_second_file_is_checked_too_and_a_clean_pair_passes(spec, tmp_path):
    write(tmp_path / "ok.bed", SORT_BED)
    write(tmp_path / "karyo.bed", GENOME)
    eng = engine_for(spec)
    assert (
        outcome(eng, SORTED_ID, "bedmap", ["--count", "ok.bed", "karyo.bed"], tmp_path)[1].verdict
        == "fail"
    )
    assert (
        outcome(eng, SORTED_ID, "bedmap", ["--count", "karyo.bed", "ok.bed"], tmp_path)[1].verdict
        == "fail"
    )
    assert (
        outcome(eng, SORTED_ID, "bedmap", ["--count", "ok.bed", "ok.bed"], tmp_path)[1].verdict
        == "pass"
    )


def test_the_checkers_that_read_files_refuse_a_t0_context(spec, tmp_path):
    write(tmp_path / "a.bed", SORT_BED)
    for rule_id, check in (
        (SORTED_ID, "iv_bedops_sorted_input"),
        (SCORE_ID, "iv_bedops_bedmap_score_column"),
    ):
        rule = next(r for r in spec.rules if r.id == rule_id)
        call = Call("bedmap", ("--mean", "a.bed"), tmp_path)
        with pytest.raises(TierViolation):
            known_checkers()[check](call, Match(rule), Ctx(tmp_path, "T0"))
    # the amount rule is T0: it decides from the arguments and never touches a file
    rule = next(r for r in spec.rules if r.id == ELEMENT_ID)
    call = Call("bedops", ("-e", "a.bed", "b.bed"), tmp_path)
    assert (
        known_checkers()["iv_bedops_element_of_amount"](
            call, Match(rule), Ctx(tmp_path, "T0")
        ).verdict
        == "fail"
    )


# ---- bedops.bedmap_score_column ----------------------------------------------------------------

REF = ["chr1\t100\t200\tr1", "chr1\t300\t400\tr2"]
BEDGRAPH = ["chr1\t110\t120\t3.5", "chr1\t130\t140\t4.5", "chr1\t310\t330\t10"]
BED5 = ["chr1\t110\t120\tm1\t3.5", "chr1\t130\t140\tm2\t4.5", "chr1\t310\t330\tm3\t10"]
BED6 = ["chr1\t110\t120\tm1\t3.5\t+", "chr1\t310\t330\tm3\t10\t-"]
NARROW = [
    "chr1\t110\t210\tp1\t143\t.\t4.7\t9.1\t7.2\t40",
    "chr1\t310\t430\tp2\t210\t.\t8.5\t12.0\t10.1\t60",
]
SCORE_OPERATIONS = [
    "--mean",
    "--sum",
    "--median",
    "--min",
    "--max",
    "--stdev",
    "--wmean",
    "--echo-map-score",
]


@pytest.mark.parametrize("operation", SCORE_OPERATIONS)
def test_a_bedgraph_map_file_fails_for_every_score_operation(spec, tmp_path, operation):
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "sig.bedgraph", BEDGRAPH)
    decision, result = outcome(
        engine_for(spec),
        SCORE_ID,
        "bedmap",
        ["--echo", operation, "ref.bed", "sig.bedgraph"],
        tmp_path,
    )
    assert result.verdict == "fail" and result.code == "no_score_column"
    text = render_block(decision)
    assert text.startswith("FIX: awk ")
    assert "only 4 columns" in text and "0.000000 or NAN" in text


def test_a_declaration_does_not_rescue_a_file_without_column_5(spec, tmp_path):
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "sig.bedgraph", BEDGRAPH)
    _, result = outcome(
        engine_for(spec),
        SCORE_ID,
        "bedmap",
        ["--mean", "ref.bed", "sig.bedgraph"],
        tmp_path,
        {VALUE_KEY: "5"},
    )
    assert result.verdict == "fail" and result.code == "no_score_column"


def test_three_columns_and_a_text_fifth_column_fail_but_plain_scores_pass(spec, tmp_path):
    eng = engine_for(spec)
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "three.bed", ["chr1\t110\t120"])
    write(tmp_path / "text5.bed", ["chr1\t110\t120\tm1\tabc"])
    write(tmp_path / "dot5.bed", ["chr1\t110\t120\tm1\t.\t+"])
    write(tmp_path / "bed5.bed", BED5)
    write(tmp_path / "bed6.bed", BED6)
    write(tmp_path / "sci.bed", ["chr1\t110\t120\tm1\t1.5e-3", "chr1\t130\t140\tm2\t-2"])
    for name in ("three.bed", "text5.bed", "dot5.bed"):
        assert (
            outcome(eng, SCORE_ID, "bedmap", ["--mean", "ref.bed", name], tmp_path)[1].verdict
            == "fail"
        ), name
    for name in ("bed5.bed", "bed6.bed", "sci.bed"):
        decision, result = outcome(eng, SCORE_ID, "bedmap", ["--mean", "ref.bed", name], tmp_path)
        assert decision.action == "allow" and result.verdict == "pass", name


def test_a_narrowpeak_file_fails_until_the_scaled_score_is_declared(spec, tmp_path):
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "peaks.narrowPeak", NARROW)
    write(tmp_path / "peaks.tsv", NARROW)  # recognised by its shape, whatever the name
    eng = engine_for(spec)
    for name in ("peaks.narrowPeak", "peaks.tsv"):
        argv = ["--echo", "--mean", "ref.bed", name]
        decision, result = outcome(eng, SCORE_ID, "bedmap", argv, tmp_path)
        assert result.verdict == "fail" and result.code == "peak_score_column", name
        text = render_block(decision)
        assert text.splitlines()[0].startswith("FIX: awk ") and "$5=$7" in text
        assert f"--declare {VALUE_KEY}=5" in text and "column 7" in text
        assert (
            outcome(eng, SCORE_ID, "bedmap", argv, tmp_path, {VALUE_KEY: "5"})[1].verdict == "pass"
        )
    write(tmp_path / "peaks.broadPeak", NARROW[:1])
    assert (
        outcome(eng, SCORE_ID, "bedmap", ["--sum", "ref.bed", "peaks.broadPeak"], tmp_path)[1].code
        == "peak_score_column"
    )


def test_no_score_operation_no_match_and_the_single_file_form(spec, tmp_path):
    eng = engine_for(spec)
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "sig.bedgraph", BEDGRAPH)
    for argv in (
        ["--count", "ref.bed", "sig.bedgraph"],
        ["--echo-map-id", "ref.bed", "sig.bedgraph"],
        ["--indicator", "ref.bed", "sig.bedgraph"],
    ):
        decision, result = outcome(eng, SCORE_ID, "bedmap", argv, tmp_path)
        assert result is None and decision.action == "allow", argv  # the rule does not match
    # one file: it is both reference and map file
    _, result = outcome(eng, SCORE_ID, "bedmap", ["--mean", "sig.bedgraph"], tmp_path)
    assert result.verdict == "fail"
    _, result = outcome(eng, SCORE_ID, "bedmap", ["--mean", "-"], tmp_path)
    assert result.verdict == "pass"  # standard input is not seen
    # a pipe feeds the map file: the reference file (BED4) is not the one to judge
    for argv in (
        ["--echo", "--mean", "ref.bed", "-"],
        ["--delim", ";", "--echo", "--mean", "ref.bed", "-"],
        ["--prec", "3", "--echo", "--tmean", "0.1", "0.1", "ref.bed", "-"],
        ["--echo", "--mad", "2", "--mean", "ref.bed", "-"],
    ):
        assert outcome(eng, SCORE_ID, "bedmap", argv, tmp_path)[1].verdict == "pass", argv
    # option values are never mistaken for the map file
    for argv in (
        ["--prec", "3", "--echo", "--mean", "ref.bed", "sig.bedgraph"],
        ["--chrom", "chr1", "--mad", "--mean", "ref.bed", "sig.bedgraph"],
        ["--bp-ovr", "5", "--echo", "--mean", "ref.bed", "sig.bedgraph"],
    ):
        assert outcome(eng, SCORE_ID, "bedmap", argv, tmp_path)[1].verdict == "fail", argv


# ---- bedops.element_of_amount ------------------------------------------------------------------


@pytest.mark.parametrize("option", ["-e", "--element-of", "-n", "--not-element-of"])
def test_an_explicit_amount_passes_and_the_default_needs_a_declaration(spec, tmp_path, option):
    eng = engine_for(spec)
    for amount in ("1", "50%", "100%", "25", "0.5%"):
        decision, result = outcome(
            eng, ELEMENT_ID, "bedops", [option, amount, "r.bed", "m.bed"], tmp_path
        )
        assert decision.action == "allow" and result.verdict == "pass", (option, amount)
    decision, result = outcome(eng, ELEMENT_ID, "bedops", [option, "r.bed", "m.bed"], tmp_path)
    assert result.verdict == "fail" and result.code == "amount_not_stated"
    text = render_block(decision)
    assert f"bedops {option} 1 reference.bed map.bed" in text.splitlines()[0]
    assert "100 percent" in text and f"--declare {AMOUNT_KEY}=full" in text
    argv = [option, "r.bed", "m.bed"]
    assert (
        outcome(eng, ELEMENT_ID, "bedops", argv, tmp_path, {AMOUNT_KEY: "full"})[1].verdict
        == "pass"
    )
    decision, result = outcome(eng, ELEMENT_ID, "bedops", argv, tmp_path, {AMOUNT_KEY: "any"})
    assert result.verdict == "fail" and result.code == "declared_mismatch"
    assert "bedops " + option + " 1" in render_block(decision)


def test_element_of_after_other_options_and_other_bedops_operations(spec, tmp_path):
    eng = engine_for(spec)
    argv = ["--ec", "--range", "10", "-e", "r.bed", "m.bed"]
    assert outcome(eng, ELEMENT_ID, "bedops", argv, tmp_path)[1].code == "amount_not_stated"
    for argv in (
        ["-m", "a.bed"],
        ["-u", "a.bed", "b.bed"],
        ["-i", "a.bed", "b.bed"],
        ["-d", "a.bed", "b.bed"],
    ):
        decision, result = outcome(eng, ELEMENT_ID, "bedops", argv, tmp_path)
        assert result is None and decision.action == "allow", argv
    _, result = outcome(eng, ELEMENT_ID, "bedops", ["-e", "-", "m.bed"], tmp_path)
    assert result.code == "amount_not_stated"
    _, result = outcome(
        eng, ELEMENT_ID, "bedops", ["-e", "1", "r.bed", "m.bed"], tmp_path, {AMOUNT_KEY: "full"}
    )
    assert result.verdict == "pass"  # an explicit amount decides; the declaration is not compared


def test_an_invalid_declaration_is_refused_by_the_engine(spec, tmp_path):
    _, result = outcome(
        engine_for(spec),
        ELEMENT_ID,
        "bedops",
        ["-e", "r.bed", "m.bed"],
        tmp_path,
        {AMOUNT_KEY: "most"},
    )
    assert result.code == "declaration_invalid"
    write(tmp_path / "ref.bed", REF)
    write(tmp_path / "m.bed", BED5)
    _, result = outcome(
        engine_for(spec),
        SCORE_ID,
        "bedmap",
        ["--mean", "ref.bed", "m.bed"],
        tmp_path,
        {VALUE_KEY: "7"},
    )
    assert result.code == "declaration_invalid"


# ---- the traps on the real programs ------------------------------------------------------------


def run(tools, argv, cwd):
    return tools.run(argv, cwd)


def test_bedmap_on_genome_order_files_is_silent_wrong_and_ec_stops(tools, tmp_path):
    write(tmp_path / "ref.bed", GENOME)
    write(
        tmp_path / "map.bed",
        ["chr1\t150\t160\tm1", "chr2\t150\t160\tm2", "chr9\t5\t6\tm9", "chr10\t120\t130\tm10"],
    )
    proc = run(tools, ["bedmap", "--echo", "--count", "ref.bed", "map.bed"], tmp_path)
    assert proc.returncode == 0 and proc.stderr == ""
    wrong = [line.rsplit("|", 1)[1] for line in proc.stdout.splitlines()]
    assert wrong[-1] == "0"  # chr10 has a map element inside it
    for name in ("ref.bed", "map.bed"):
        sorted_copy = run(tools, ["sort-bed", name], tmp_path).stdout
        (tmp_path / f"s_{name}").write_text(sorted_copy)
    good = run(tools, ["bedmap", "--echo", "--count", "s_ref.bed", "s_map.bed"], tmp_path)
    counts = {
        line.split("\t")[3].split("|")[0]: line.rsplit("|", 1)[1]
        for line in good.stdout.splitlines()
    }
    assert counts == {"a": "1", "b": "1", "c": "1", "d": "1"}
    loud = run(tools, ["bedmap", "--ec", "--echo", "--count", "ref.bed", "map.bed"], tmp_path)
    assert loud.returncode == 1 and "not properly sorted by first column" in loud.stderr


def test_bedmap_mean_on_a_bedgraph_is_silent_garbage_and_on_bed5_is_right(tools, tmp_path):
    write(tmp_path / "ref.bed", [*REF, "chr1\t500\t600\tr3"])
    write(tmp_path / "sig.bedgraph", BEDGRAPH)
    write(tmp_path / "sig5.bed", BED5)
    proc = run(tools, ["bedmap", "--mean", "ref.bed", "sig.bedgraph"], tmp_path)
    assert proc.returncode == 0 and proc.stderr == ""
    assert proc.stdout.split() == ["0.000000", "NAN", "NAN"]  # r2 holds the 10 and gets NAN
    proc = run(tools, ["bedmap", "--mean", "ref.bed", "sig5.bed"], tmp_path)
    assert proc.stdout.split() == ["4.000000", "10.000000", "NAN"]
    write(tmp_path / "narrow.narrowPeak", NARROW)
    proc = run(tools, ["bedmap", "--mean", "ref.bed", "narrow.narrowPeak"], tmp_path)
    assert proc.stderr == "" and proc.stdout.split()[:2] == ["143.000000", "210.000000"]


def test_bedops_element_of_defaults_to_100_percent(tools, tmp_path):
    write(tmp_path / "r.bed", ["chr1\t100\t200\tr1", "chr1\t300\t400\tr2", "chr1\t500\t600\tr3"])
    write(tmp_path / "m.bed", ["chr1\t150\t160\tm1", "chr1\t300\t400\tm2", "chr1\t450\t650\tm3"])
    names = lambda argv: [  # noqa: E731
        line.split("\t")[3] for line in run(tools, ["bedops", *argv], tmp_path).stdout.splitlines()
    ]
    assert names(["-e", "r.bed", "m.bed"]) == ["r2", "r3"]
    assert names(["-e", "100%", "r.bed", "m.bed"]) == ["r2", "r3"]
    assert names(["-e", "1", "r.bed", "m.bed"]) == ["r1", "r2", "r3"]
    assert names(["-n", "r.bed", "m.bed"]) == ["r1"]
    assert names(["-n", "1", "r.bed", "m.bed"]) == []
    proc = run(tools, ["bedops", "-e", "r.bed", "m.bed"], tmp_path)
    assert proc.returncode == 0 and proc.stderr == ""


# ---- the F70 / F71 workspaces ------------------------------------------------------------------


def test_the_family_commands_through_the_rules(spec, tools, tmp_path):
    eng = engine_for(spec)
    faulted = tmp_path / "f70f"
    build_task("F70", 5, "faulted", faulted, tools, "implicit")
    clean = tmp_path / "f70c"
    build_task("F70", 5, "clean", clean, tools, "implicit")
    argv = ["--echo", "--count", "regions.bed", "peaks.bed"]
    decision, result = outcome(eng, SORTED_ID, "bedmap", argv, faulted)
    assert result.verdict == "fail" and decision.action == "block"
    assert "chr9" in result.message and "chr10" in result.message
    assert outcome(eng, SORTED_ID, "bedmap", argv, clean)[1].verdict == "pass"
    assert outcome(eng, SORTED_ID, "bedmap", ["--ec", *argv], faulted)[1].verdict == "pass"
    f71f, f71c = tmp_path / "f71f", tmp_path / "f71c"
    build_task("F71", 5, "faulted", f71f, tools, "implicit")
    build_task("F71", 5, "clean", f71c, tools, "implicit")
    assert outcome(eng, SCORE_ID, "bedmap", ["--mean", "regions.bed", "peaks.narrowPeak"], f71f)[
        1
    ].code == ("peak_score_column")
    assert (
        outcome(eng, SCORE_ID, "bedmap", ["--mean", "regions.bed", "peaks.bed"], f71c)[1].verdict
        == "pass"
    )


# ---- admission ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_id", "min_wrong_failed"),
    [(SORTED_ID, 5), (SCORE_ID, 5), (ELEMENT_ID, 4)],
)
def test_admission(spec, tools, tmp_path, rule_id, min_wrong_failed):
    stats = measure(rule_id, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    right_undeclared = stats.cells[("right", False)]
    assert stats.conditional
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert stats.killed() >= ADMISSION_SEEDS * min_wrong_failed
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert promotion_is_consistent(stats)
