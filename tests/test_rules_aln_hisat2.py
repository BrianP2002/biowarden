"""Rule hisat2.rna_strandness (family F34): unit tests, evidence against hisat2 + StringTie.

A T0 rule with a required declaration, checked on real argv. The evidence tests run the installed
hisat2 2.2.3, samtools 1.24 and StringTie on F34 workspaces to tie the module docstring's claims
to the tools (a wrong value flips every strand; one letter on pairs breaks them). The admission
test measures the rule over ADMISSION_SEEDS seeds with calls labelled by running hisat2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f34_hisat2_strand import _align, assembled_strands
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, known_checkers
from biowarden.engine.rules import aln_hisat2 as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_hisat2")
(RULE,) = SPEC.rules
KEY = "library_strandedness"
PAIRS = ["-x", "idx", "-1", "a_1.fq", "-2", "a_2.fq"]
SINGLE = ["-x", "idx", "-U", "a.fq"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not {"hisat2", "stringtie"} <= set(env.real_bins(["hisat2", "stringtie"])):
        pytest.skip("samtools, hisat2 and stringtie are not available")
    return env


def run(argv, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(
        Call("hisat2", tuple(argv), Path("/nonexistent-workspace"), declared or {})
    )
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_required_t0_declaration_when_the_option_is_used():
    assert RULE.id == "hisat2.rna_strandness" and (RULE.kind, RULE.tier) == ("call", "T0")
    assert RULE.applies.tools == ["hisat2"] and RULE.applies.options_any == ["--rna-strandness"]
    assert RULE.declare.key == KEY and RULE.declare.required
    assert RULE.declare.values == ["forward", "reverse"]
    assert RULE.check in known_checkers()


def test_the_checker_touches_no_file():
    call = Call(
        "hisat2", ("--rna-strandness", "FR", *PAIRS), Path("/nonexistent"), {KEY: "reverse"}
    )
    result = known_checkers()[RULE.check](call, Match(RULE), Ctx(Path("/nonexistent"), "T0"))
    assert result.verdict == "fail"


@pytest.mark.parametrize(
    ("tail", "value", "paired"),
    [
        (["--rna-strandness", "RF", *PAIRS], "RF", True),
        (["--rna-strandness=fr", *PAIRS], "FR", True),
        (["--rna-strandness", "R", *SINGLE], "R", False),
        (["--rna-strandness", "F", "-p", "4", *SINGLE], "F", False),
        ([*PAIRS], None, True),
        (["--rna-strandness", "F", "--rna-strandness", "RF", *PAIRS], "RF", True),
    ],
)
def test_strandness_value_and_pairing_are_read_in_every_spelling(tail, value, paired):
    assert rules.strandness(Call("hisat2", tuple(tail), Path("."))) == (value, paired)


@pytest.mark.parametrize(
    ("value", "reads", "declared", "expected"),
    [
        ("RF", PAIRS, "reverse", "pass"),
        ("FR", PAIRS, "forward", "pass"),
        ("FR", PAIRS, "reverse", "fail"),
        ("RF", PAIRS, "forward", "fail"),
        ("R", SINGLE, "reverse", "pass"),
        ("F", SINGLE, "forward", "pass"),
        ("F", SINGLE, "reverse", "fail"),
        ("R", SINGLE, "forward", "fail"),
        ("rf", PAIRS, "reverse", "pass"),
        ("R", PAIRS, "reverse", "unknown"),  # one letter on pairs: the outcome depends on the data
        ("F", PAIRS, "forward", "unknown"),
    ],
)
def test_declaration_against_the_value(value, reads, declared, expected):
    assert run(["--rna-strandness", value, *reads], {KEY: declared})[1].verdict == expected


def test_contradiction_names_the_value_to_use_and_the_consequence():
    decision, result = run(["--rna-strandness", "FR", *PAIRS], {KEY: "reverse"})
    assert result.code == "strandness_contradicts_declaration" and decision.action == "block"
    assert result.fix == ("use --rna-strandness RF for a reverse-stranded library",)
    assert "StringTie" in result.message and "dUTP" not in result.fix[0]
    _, single = run(["--rna-strandness", "F", *SINGLE], {KEY: "reverse"})
    assert single.fix == ("use --rna-strandness R for a reverse-stranded library",)


def test_a_single_letter_on_pairs_warns_with_the_two_letter_fix():
    decision, result = run(["--rna-strandness", "R", *PAIRS], {KEY: "reverse"})
    assert result.code == "single_letter_on_pairs" and decision.action == "warn"
    assert result.fix == ("use --rna-strandness RF",)


def test_missing_declaration_is_refused_with_the_library_the_value_implies():
    decision, result = run(["--rna-strandness", "FR", *PAIRS])
    assert result.code == "declaration_missing" and decision.action == "block"
    assert "forward-stranded" in result.fix[0] and "RF" in result.fix[0]
    decision, result = run(["--rna-strandness", "R", *SINGLE])
    assert "reverse-stranded" in result.fix[0] and "F" in result.fix[0]


def test_calls_without_the_option_and_other_tools_are_not_matched():
    assert run(PAIRS)[1] is None
    assert run(["--dta", *PAIRS])[1] is None
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    assert eng.evaluate(Call("hisat2-build", ("a.fa", "idx"), Path("."))).action == "allow"
    assert eng.evaluate(Call("stringtie", ("--rf", "a.bam"), Path("."))).action == "allow"


def test_an_unknown_value_is_left_to_hisat2():
    assert run(["--rna-strandness", "XY", *PAIRS], {KEY: "forward"})[1].verdict == "pass"


# ---- F34 workspaces and the tools --------------------------------------------------------------


@pytest.fixture(scope="module")
def f34(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f34")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F34", 2, twin, work, tools, "implicit"), work)
    return out


def flip(strands: dict[str, str]) -> dict[str, str]:
    return {g: "-" if s == "+" else "+" for g, s in strands.items()}


def test_a_wrong_value_aligns_equally_well_and_flips_every_assembled_strand(f34, tools):
    task, work = f34["faulted"]  # dUTP: RF is right
    rates = {}
    for value in ("RF", "FR"):
        sam = _align(work, tools, f"ev_{value}", "--rna-strandness", value)
        rates[value] = sum(
            1 for x in sam.splitlines() if not x.startswith("@") and int(x.split("\t")[1]) & 4 == 0
        )
        # XS:A is written on every alignment once the option is given
        mapped = [
            x for x in sam.splitlines() if not x.startswith("@") and int(x.split("\t")[1]) & 4 == 0
        ]
        assert all("\tXS:A:" in line for line in mapped)
    assert rates["RF"] == rates["FR"]
    right = assembled_strands(work, tools, "ev_RF")
    wrong = assembled_strands(work, tools, "ev_FR")
    assert right == task.truth["expected"] and wrong == flip(right) == task.truth["naive"]


def test_one_letter_on_pairs_does_not_give_the_right_strands(f34, tools):
    task, work = f34["faulted"]
    _align(work, tools, "ev_R", "--rna-strandness", "R")
    assert assembled_strands(work, tools, "ev_R") != task.truth["expected"]


def test_without_the_option_the_strands_come_from_the_splice_motifs_and_are_right(f34, tools):
    task, work = f34["faulted"]
    _align(work, tools, "ev_none")
    assert assembled_strands(work, tools, "ev_none") == task.truth["expected"]


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tools, tmp_path):
    seeds = max(
        8, ADMISSION_SEEDS // 4
    )  # each seed runs hisat2, samtools sort and StringTie 18 times
    stats = measure(RULE.id, [SPEC], range(seeds), tools, tmp_path)
    assert stats.admitted() and not stats.conditional
    assert stats.cells[("right", True)].n >= MIN_RIGHT_CALLS * seeds // ADMISSION_SEEDS
    assert stats.cells[("right", True)].failed == 0
    assert stats.killed() > 0 and all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # a declared library that contradicts the value is caught in every case
    assert stats.cells[("wrong", True)].failed > 0
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
