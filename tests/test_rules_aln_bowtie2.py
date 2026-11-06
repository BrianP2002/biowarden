"""Rule bowtie2.maxins_default (family F33): unit tests, evidence against bowtie2, admission.

A T0 rule with a required declaration: the unit tests call the engine on real argv. The evidence
tests run the installed bowtie2 2.5.5 (and hisat2 2.2.3, which prints the same default but does not
show the trap) on F33 workspaces to tie the docstring's claims to the tools. The admission test
measures the rule over ADMISSION_SEEDS seeds with calls labelled by running bowtie2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import aln_bowtie2 as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_bowtie2")
(RULE,) = SPEC.rules
KEY = "fragment_length_max"
PAIRED = ["-x", "idx", "-1", "a_1.fq", "-2", "a_2.fq"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "bowtie2" not in env.real_bins(["bowtie2"]):
        pytest.skip("samtools and bowtie2 are not available")
    return env


def run(argv, declared=None, cwd="/nonexistent-workspace"):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("bowtie2", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_required_t0_declaration_on_paired_runs():
    assert RULE.id == "bowtie2.maxins_default" and (RULE.kind, RULE.tier) == ("call", "T0")
    assert RULE.applies.tools == ["bowtie2"] and RULE.applies.options_any == ["-1", "-2"]
    assert RULE.declare.key == KEY and RULE.declare.required
    assert RULE.declare.values == ["up_to_500", "over_500"]
    assert RULE.check in known_checkers()
    assert SPEC.declaration_schema()["required"] == [KEY]


def test_the_checker_touches_no_file():
    call = Call("bowtie2", tuple(PAIRED), Path("/nonexistent-workspace"), {KEY: "over_500"})
    result = known_checkers()[RULE.check](call, Match(RULE), Ctx(Path("/nonexistent"), "T0"))
    assert result.verdict == "fail"


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], 500),
        (["-X", "800"], 800),
        (["-X800"], 800),
        (["--maxins", "1000"], 1000),
        (["--maxins=1200"], 1200),
        (["-X", "300", "-X", "900"], 900),
        (["-p", "4", "-I", "50", "-X", "700"], 700),
        (["--local", "--very-sensitive-local", "-X", "650"], 650),
    ],
)
def test_maxins_in_every_spelling(options, expected):
    assert rules.maxins(Call("bowtie2", tuple([*options, *PAIRED]), Path("."))) == expected


@pytest.mark.parametrize(
    ("options", "declared", "expected"),
    [
        ([], "over_500", "fail"),
        (["-X", "500"], "over_500", "fail"),
        (["-X", "501"], "over_500", "pass"),
        (["-X", "1000"], "over_500", "pass"),
        (["--maxins=1000"], "over_500", "pass"),
        ([], "up_to_500", "pass"),
        (["-X", "400"], "up_to_500", "pass"),
        (["-X", "1000"], "up_to_500", "pass"),  # a larger -X than needed is harmless
    ],
)
def test_declaration_against_the_option(options, declared, expected):
    _, result = run([*options, *PAIRED], {KEY: declared})
    assert result.verdict == expected


def test_a_default_call_declared_over_500_gets_the_x_fix_first():
    decision, result = run(PAIRED, {KEY: "over_500"})
    assert result.code == "maxins_too_small" and decision.action == "block"
    assert result.fix == ("add -X <the longest fragment of the library>, for example -X 1000",)
    assert render_block(decision).startswith("FIX: add -X")
    assert "flag 0x2 clear" in result.message


def test_a_missing_declaration_is_refused_with_a_fix_that_fits_the_call():
    decision, result = run(PAIRED)
    assert result.code == "declaration_missing" and decision.action == "block"
    assert "up_to_500" in result.fix[0] and "-X 1000" in result.fix[0]
    decision, result = run(["-X", "900", *PAIRED])
    assert "this call sets -X 900" in result.fix[0] and "over_500" in result.fix[0]


def test_single_end_runs_and_other_tools_are_not_matched():
    assert run(["-x", "idx", "-U", "a.fq"])[1] is None
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    assert eng.evaluate(Call("bowtie2-build", ("a.fa", "idx"), Path("."))).action == "allow"
    assert eng.evaluate(Call("hisat2", tuple(PAIRED), Path("."))).action == "allow"


def test_an_invalid_declaration_is_refused():
    decision, result = run(PAIRED, {KEY: "huge"})
    assert result.code == "declaration_invalid" and decision.action == "block"


# ---- F33 workspaces and the tool -------------------------------------------------------------


@pytest.fixture(scope="module")
def f33(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f33")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F33", 1, twin, work, tools, "implicit"), work)
    return out


def flags(proc) -> list[int]:
    return [int(x.split("\t")[1]) for x in proc.stdout.splitlines() if not x.startswith("@")]


def test_bowtie2_calls_long_fragments_discordant_by_default_and_concordant_with_x(f33, tools):
    task, work = f33["faulted"]
    base = ["bowtie2", "-p", "1", "-x", "bt2idx", "-1", "reads_1.fastq", "-2", "reads_2.fastq"]
    default = tools.run(base, work)
    wide = tools.run([*base, "-X", "1000"], work)
    assert default.returncode == 0 == wide.returncode
    proper = lambda proc: sum((f & 0x43) == 0x43 for f in flags(proc))  # noqa: E731
    assert proper(default) == task.truth["naive"] < task.truth["expected"] == proper(wide)
    assert "aligned discordantly 1 time" in default.stderr
    assert not any(
        f & 0x4 for f in flags(default)
    )  # every mate is aligned, only unpaired as a pair
    discordant = [f for f in flags(default) if f & 0x1 and not f & 0x2]
    assert len(discordant) == 2 * (task.truth["expected"] - task.truth["naive"])


def test_hisat2_prints_the_same_default_but_reports_the_long_pairs_concordant(f33, tools):
    task, work = f33["faulted"]
    build = tools.run(["hisat2-build-s", "-q", "genome.fa", "hs2idx"], work)
    assert build.returncode == 0
    run_hisat2 = tools.run(
        ["hisat2", "-p", "1", "-x", "hs2idx", "-1", "reads_1.fastq", "-2", "reads_2.fastq"], work
    )
    proper = sum((f & 0x43) == 0x43 for f in flags(run_hisat2))
    assert proper == task.truth["expected"]
    usage = tools.run(["hisat2", "--help"], work)
    assert "maximum fragment length (500)" in usage.stdout + usage.stderr


def test_family_commands_with_the_right_declaration_pass_the_rule_which_cannot_see_the_data(f33):
    base = ["-p", "1", "-x", "bt2idx", "-1", "reads_1.fastq", "-2", "reads_2.fastq"]
    for twin in ("clean", "faulted"):
        _, work = f33[twin]
        assert run(base, {KEY: "up_to_500"}, work)[1].verdict == "pass"  # T0: cannot know better
        assert run(base, {KEY: "over_500"}, work)[1].verdict == "fail"
        assert run([*base, "-X", "1000"], {KEY: "over_500"}, work)[1].verdict == "pass"


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.admitted() and not stats.conditional  # a required declaration
    assert stats.cells[("right", True)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", True)].failed == 0
    assert stats.killed() > 0 and all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # every undeclared wrong call is refused by the engine; a declared over_500 catches the default
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert stats.cells[("wrong", True)].failed > 0
