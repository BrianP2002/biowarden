"""Rule stringtie.strandedness: --rf / --fr against the strand of the reads.

Workspaces come from the quant simulator (cases_quant_stringtie.py): single-end BAM + GTF of a dUTP
library and of an unstranded one, plus a paired forward-stranded one built here.
"""

from __future__ import annotations

import random

import pytest
from test_rules_quant_common import engine_for, outcome, quant_tools

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_quant_stringtie import (
    FLAGS,
    TEMPLATES,
    build_workspace,
    stringtie_reads,
)
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import common
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    FORWARD,
    gtf_text,
    make_fragments,
    make_layout,
    sam_genome,
)
from biowarden.engine import Call, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

RULE = "stringtie.strandedness"
COMMON = ["-G", "genes.gtf", "-e", "-A", "ab.tab", "-o", "out.gtf", "-p", "1"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_stringtie")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_stringtie")


@pytest.fixture(scope="module")
def libraries(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("stringtie")
    for name, twin in (("dutp", "clean"), ("unstranded", "faulted")):
        build_workspace(5, twin, base / name, tools)
    layout = make_layout("st-5", 5)
    forward = base / "forward"
    forward.mkdir()
    frags = make_fragments(random.Random("fwd"), layout, FORWARD)
    (forward / "genes.gtf").write_text(gtf_text(layout))
    common.write_bam(forward, sam_genome(layout, frags, True), tools, "reads")  # paired
    return base / "dutp", base / "unstranded", forward


def test_spec_is_a_conditional_t1_call_rule(spec):
    (rule,) = spec.rules
    assert (rule.id, rule.kind, rule.tier, rule.slot) == (RULE, "call", "T1", "strand")
    assert rule.applies.tools == ["stringtie"] and rule.check in known_checkers()
    assert (rule.declare.key, rule.declare.required) == ("library_strandedness", False)
    assert rule.incidents == ["GEN-23", "GEN-20"]


@pytest.mark.parametrize(
    ("which", "flags", "verdict"),
    [
        (0, [], "pass"),  # dUTP without a flag: not judged, never fails a call without a flag
        (0, ["--rf"], "pass"),
        (0, ["--fr"], "fail"),  # most genes get no coverage
        (1, [], "pass"),
        (1, ["--rf"], "fail"),  # half of the reads
        (1, ["--fr"], "fail"),
        (2, ["--fr"], "pass"),  # forward, paired
        (2, ["--rf"], "fail"),
        (2, [], "pass"),
    ],
)
def test_verdicts(libraries, eng, which, flags, verdict):
    work = libraries[which]
    for argv in (["reads.bam", *COMMON, *flags], [*flags, "reads.bam", *COMMON]):
        _, result = outcome(eng, RULE, "stringtie", argv, work)
        assert result.verdict == verdict, (argv, result)
        if verdict == "fail":
            assert result.code == "stranded_option_mismatch"


def test_message_and_fix(libraries, eng):
    decision, result = outcome(eng, RULE, "stringtie", ["reads.bam", *COMMON, "--fr"], libraries[0])
    assert "--fr tells stringtie the library is forward" in result.message
    assert "reverse-stranded (dUTP)" in result.message and "reads.bam" in result.message
    assert result.fix[0] == "re-run with --rf (declare library_strandedness=reverse)"
    assert render_block(decision, eng.retry_cap, eng.style).startswith("FIX: re-run with --rf")
    _, half = outcome(eng, RULE, "stringtie", ["reads.bam", *COMMON, "--rf"], libraries[1])
    assert "only about half of the reads would be counted" in half.message
    assert half.fix[0].startswith("re-run with no --rf/--fr flag")


def test_declarations(libraries, eng):
    for flags, declared, verdict in (
        (["--rf"], "reverse", "pass"),
        (["--rf"], "forward", "fail"),
        (["--fr"], "reverse", "fail"),
        ([], "reverse", "fail"),  # no flag means unstranded
        ([], "unstranded", "fail"),  # ... which the reads contradict
    ):
        _, result = outcome(
            eng,
            RULE,
            "stringtie",
            ["reads.bam", *COMMON, *flags],
            libraries[0],
            {"library_strandedness": declared},
        )
        assert result.verdict == verdict, (flags, declared, result)


def test_merge_calls_both_flags_and_missing_inputs_are_not_judged(libraries, eng):
    work = libraries[1]
    cases = [
        ["--merge", "-G", "genes.gtf", "-o", "merged.gtf", "a.gtf", "b.gtf", "--rf"],
        ["reads.bam", *COMMON, "--rf", "--fr"],
        ["reads.bam", "-e", "-o", "out.gtf", "--rf"],  # no -G: nothing to compare with
        ["missing.bam", *COMMON, "--rf"],
        ["reads.bam", "-G", "missing.gtf", "-e", "--rf"],
        ["--version"],
    ]
    for argv in cases:
        _, result = outcome(eng, RULE, "stringtie", argv, work)
        assert result.verdict == "pass", argv


def test_the_checker_reads_the_bam_so_a_t0_context_refuses(libraries, spec):
    work = libraries[1]
    call = Call("stringtie", ("reads.bam", *COMMON, "--rf"), work)
    check = known_checkers()["stringtie_strandedness"]
    with pytest.raises(TierViolation):
        check(call, Match(spec.rules[0]), Ctx(work, "T0"))
    assert check(call, Match(spec.rules[0]), Ctx(work, "T1")).verdict == "fail"


def test_reads_recovered_from_coverage_match_the_simulation_and_the_strand_effects(
    tools, libraries
):
    """The admission labels rest on reads = coverage * length / 50 (the docstring evidence)."""
    for work, twin in ((libraries[0], "clean"), (libraries[1], "faulted")):
        truth = build_workspace(5, twin, work, tools)
        got = {k: stringtie_reads(work, tools, truth, *flags) for k, flags in FLAGS.items()}
        assert answer_matches(truth, got["none"])  # exact without a flag on both libraries
        assert answer_matches(truth, got["rf"]) == (twin == "clean")
        assert not answer_matches(truth, got["fr"])
        if twin == "clean":
            assert sum(got["fr"].values()) < 0.5 * sum(truth["expected"].values())
        else:
            assert (
                0.3 * sum(truth["expected"].values())
                < sum(got["rf"].values())
                < 0.7 * sum(truth["expected"].values())
            )


def test_every_template_spelling_has_the_flag_of_its_key():
    for t in TEMPLATES:
        flags = tuple(a for a in t.argv if a in ("--rf", "--fr"))
        assert {(): "none", ("--rf",): "rf", ("--fr",): "fr"}[flags] == t.key, t.name


def test_a_stranded_flag_on_unstranded_reads_exits_0_with_nothing_on_stderr(tools, libraries):
    proc = tools.run(["stringtie", "reads.bam", *COMMON, "--rf"], libraries[1])
    assert proc.returncode == 0 and proc.stderr == ""


def test_admission(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert (
        stats.cells[("right", True)].n >= MIN_RIGHT_CALLS
        and stats.cells[("right", True)].failed == 0
    )
    assert (
        stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
        and stats.cells[("right", False)].failed == 0
    )
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert promotion_is_consistent(stats)
