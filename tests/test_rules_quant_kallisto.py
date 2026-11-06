"""Rule kallisto.strand_option: the built-in option_implies_declared check on --fr/--rf-stranded.

T0: the call alone decides, so no workspace is needed for the verdict tests. The admission test
builds paired FASTQ of a dUTP and of an unstranded library, indexes them and runs the real kallisto
once per option (bench/admission/cases_quant_kallisto.py).
"""

from __future__ import annotations

import pytest
from test_rules_quant_common import engine_for, outcome, quant_tools

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_quant_kallisto import (
    OPTIONS,
    TEMPLATES,
    build_workspace,
    kallisto_counts,
)
from biowarden.bench.score import answer_matches
from biowarden.engine import Call, Spec
from biowarden.engine.checks import Ctx, Match, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

RULE = "kallisto.strand_option"
QUANT = ["quant", "-i", "t.idx", "-o", "out"]
FQ = ["r1.fastq", "r2.fastq"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_kallisto")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_kallisto")


def test_spec_uses_the_builtin_option_check_with_a_required_declaration(spec):
    (rule,) = spec.rules
    assert (rule.id, rule.kind, rule.tier, rule.slot) == (RULE, "call", "T0", "strand")
    assert rule.check == "option_implies_declared" and rule.check in known_checkers()
    assert rule.applies.tools == ["kallisto"] and rule.applies.subcommands == ["quant"]
    assert rule.applies.options_any == ["--fr-stranded", "--rf-stranded"]
    assert rule.params["option_map"] == {"--fr-stranded": "forward", "--rf-stranded": "reverse"}
    assert (rule.declare.key, rule.declare.required) == ("library_strandedness", True)
    assert rule.declare.values == ["unstranded", "forward", "reverse"]
    assert spec.declaration_schema()["required"] == ["library_strandedness"]


@pytest.mark.parametrize(
    ("option", "declared", "verdict"),
    [
        ("--rf-stranded", "reverse", "pass"),
        ("--fr-stranded", "forward", "pass"),
        ("--rf-stranded", "forward", "fail"),
        ("--fr-stranded", "reverse", "fail"),
        ("--rf-stranded", "unstranded", "fail"),
        ("--fr-stranded", "unstranded", "fail"),
        ("--rf-stranded", None, "fail"),  # declaration_missing: the default is dangerous
        ("--fr-stranded", None, "fail"),
    ],
)
def test_strand_options_need_a_matching_declaration(tmp_path, eng, option, declared, verdict):
    decl = {"library_strandedness": declared} if declared else None
    decision, result = outcome(eng, RULE, "kallisto", [*QUANT, option, *FQ], tmp_path, decl)
    assert result.verdict == verdict, result
    if verdict == "fail":
        assert (decision.action == "block") and result.fix
    if declared is None:
        assert result.code == "declaration_missing"


def test_the_fix_names_the_option_that_matches_the_declaration(tmp_path, eng):
    decision, result = outcome(
        eng,
        RULE,
        "kallisto",
        [*QUANT, "--rf-stranded", *FQ],
        tmp_path,
        {"library_strandedness": "forward"},
    )
    assert result.fix == ("use --fr-stranded instead",)
    assert render_block(decision, eng.retry_cap, eng.style).startswith("FIX: use --fr-stranded")
    _, unstranded = outcome(
        eng,
        RULE,
        "kallisto",
        [*QUANT, "--fr-stranded", *FQ],
        tmp_path,
        {"library_strandedness": "unstranded"},
    )
    assert "without --fr-stranded/--rf-stranded" in unstranded.fix[0]
    _, missing = outcome(eng, RULE, "kallisto", [*QUANT, "--rf-stranded", *FQ], tmp_path)
    assert "library_strandedness=<unstranded|forward|reverse>" in missing.fix[0]


@pytest.mark.parametrize(
    "argv",
    [
        QUANT + FQ,  # kallisto is unstranded by default: nothing dangerous, no declaration
        ["index", "-i", "t.idx", "t.fa"],
        [
            "quant",
            "--plaintext",
            "-i",
            "t.idx",
            "-o",
            "out",
            "--single",
            "-l",
            "200",
            "-s",
            "20",
            "r.fq",
        ],
        ["version"],
        ["cite"],
    ],
)
def test_calls_without_a_strand_option_are_not_matched(tmp_path, eng, argv):
    _, result = outcome(eng, RULE, "kallisto", argv, tmp_path)
    assert result is None


def test_both_options_at_once_cannot_be_decided_and_only_warn(tmp_path, eng):
    decision, result = outcome(
        eng,
        RULE,
        "kallisto",
        [*QUANT, "--fr-stranded", "--rf-stranded", *FQ],
        tmp_path,
        {"library_strandedness": "reverse"},
    )
    assert result.verdict == "unknown" and result.code == "ambiguous_call"
    assert decision.action == "warn"


def test_other_options_and_positions_do_not_change_the_verdict(tmp_path, eng):
    for argv in (
        ["quant", "--rf-stranded", "-i", "t.idx", "-o", "out", "-t", "1", *FQ],
        [
            "quant",
            "-t",
            "4",
            "-b",
            "10",
            "--seed=3",
            "--rf-stranded",
            "-i",
            "t.idx",
            "-o",
            "o",
            *FQ,
        ],
        ["quant", "-i", "t.idx", "-o", "out", *FQ, "--rf-stranded"],
    ):
        _, result = outcome(
            eng, RULE, "kallisto", argv, tmp_path, {"library_strandedness": "reverse"}
        )
        assert result.verdict == "pass", argv


def test_the_check_reads_nothing(tmp_path, spec):
    rule = spec.rules[0]
    call = Call(
        "kallisto", (*QUANT, "--rf-stranded", *FQ), tmp_path, {"library_strandedness": "reverse"}
    )
    assert (
        known_checkers()["option_implies_declared"](call, Match(rule), Ctx(tmp_path, "T0")).verdict
        == "pass"
    )


def test_every_template_spelling_gives_the_numbers_of_its_key(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        truth = build_workspace(0, twin, work, tools)
        by_key = {k: kallisto_counts(work, tools, truth, *opt) for k, opt in OPTIONS.items()}
        right = {k: v is not None and answer_matches(truth, v) for k, v in by_key.items()}
        assert right["none"]  # the default is right on both libraries
        assert right["rf"] == (twin == "clean") and not right["fr"]
        for t in TEMPLATES:
            opts = tuple(a for a in t.argv if a.startswith("--") and a.endswith("-stranded"))
            key = {(): "none", ("--rf-stranded",): "rf", ("--fr-stranded",): "fr"}[opts]
            assert key == t.key, t.name


def test_unstranded_reads_with_a_strand_option_lose_about_half_without_an_error(tools, tmp_path):
    work = tmp_path / "w"
    truth = build_workspace(1, "faulted", work, tools)
    proc = tools.run(
        [
            "kallisto",
            "quant",
            "-i",
            "transcripts.idx",
            "-o",
            "o",
            "--rf-stranded",
            "reads_1.fastq",
            "reads_2.fastq",
        ],
        work,
    )
    assert proc.returncode == 0
    total = sum(truth["expected"].values())
    kept = sum(kallisto_counts(work, tools, truth, "--rf-stranded").values())
    assert 0.3 * total < kept < 0.7 * total
    assert "reads pseudoaligned" in proc.stderr


def test_admission(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    assert not stats.conditional  # a required declaration: only right_declared is gated
    assert stats.admitted() and stats.killed() > 0
    right_declared = stats.cells[("right", True)]
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed > 0
    # a wrong option declared as it is (--rf-stranded + reverse on unstranded reads) passes the rule
    assert stats.cells[("wrong", True)].failed < stats.cells[("wrong", True)].n
    assert promotion_is_consistent(stats)
