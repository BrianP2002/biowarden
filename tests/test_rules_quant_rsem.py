"""Rule rsem.strandedness: the transcriptome-BAM sniff and the stranded options of RSEM.

The BAMs are written by the quant simulator (paired transcriptome alignments of a dUTP library and
of an unstranded one, bench/admission/cases_quant_rsem.py). RSEM costs about 1 s per run, so the
admission test uses 10 seeds.
"""

from __future__ import annotations

import random

import pytest
from test_rules_quant_common import engine_for, outcome, quant_tools

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_quant_rsem import TEMPLATES, build_workspace, rsem_counts
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    FORWARD,
    make_fragments,
    make_layout,
    sam_transcriptome,
    write_transcriptome_bam,
)
from biowarden.engine import Call, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, promotion_is_consistent

RULE = "rsem.strandedness"
SEEDS = min(ADMISSION_SEEDS, 10)
BASE = ["--alignments", "--paired-end", "-p", "1", "--no-bam-output"]
ARGS = ["reads_tx.bam", "rsem_ref", "out"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_rsem")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_rsem")


@pytest.fixture(scope="module")
def libraries(tools, tmp_path_factory):
    """(dUTP workspace, unstranded workspace, forward workspace), each with reads_tx.bam."""
    base = tmp_path_factory.mktemp("rsem")
    out = []
    for name, twin in (("dutp", "clean"), ("unstranded", "faulted")):
        build_workspace(3, twin, base / name, tools)
        out.append(base / name)
    layout = make_layout("rs-3", 5)
    forward = base / "forward"
    forward.mkdir()
    frags = make_fragments(random.Random("fwd"), layout, FORWARD)
    write_transcriptome_bam(forward, sam_transcriptome(layout, frags, True), tools)
    return (*out, forward)


def test_spec_is_a_conditional_t1_call_rule(spec):
    (rule,) = spec.rules
    assert (rule.id, rule.kind, rule.tier, rule.slot) == (RULE, "call", "T1", "strand")
    assert rule.applies.tools == ["rsem-calculate-expression"] and rule.check in known_checkers()
    assert (rule.declare.key, rule.declare.required) == ("library_strandedness", False)
    assert rule.incidents == ["GEN-20"]


@pytest.mark.parametrize(
    ("which", "option", "verdict"),
    [
        (0, [], "pass"),  # dUTP, the default none: not judged (never fails a none call)
        (0, ["--strandedness", "none"], "pass"),
        (0, ["--strandedness", "reverse"], "pass"),
        (0, ["--strandedness=reverse"], "pass"),
        (0, ["--forward-prob", "0"], "pass"),
        (0, ["--strandedness", "forward"], "fail"),
        (0, ["--forward-prob", "1"], "fail"),
        (1, [], "pass"),  # unstranded
        (1, ["--strandedness", "reverse"], "fail"),
        (1, ["--strandedness", "forward"], "fail"),
        (1, ["--forward-prob", "0"], "fail"),
        (1, ["--forward-prob", "0.5"], "pass"),
        (2, ["--strandedness", "forward"], "pass"),  # forward library
        (2, ["--strandedness", "reverse"], "fail"),
    ],
)
def test_verdicts_on_transcriptome_bams(libraries, eng, which, option, verdict):
    work = libraries[which]
    _, result = outcome(eng, RULE, "rsem-calculate-expression", [*BASE, *option, *ARGS], work)
    assert result.verdict == verdict, result
    if verdict == "fail":
        assert result.fix and result.fix[0].startswith("re-run with --strandedness")


def test_message_names_the_option_the_reads_and_the_fix_comes_first(libraries, eng):
    decision, result = outcome(
        eng,
        RULE,
        "rsem-calculate-expression",
        [*BASE, "--strandedness", "reverse", *ARGS],
        libraries[1],
    )
    assert result.code == "stranded_option_mismatch"
    assert "--strandedness reverse" in result.message and "unstranded" in result.message
    assert "only about half of the reads would be counted" in result.message
    assert render_block(decision, eng.retry_cap, eng.style).startswith(
        "FIX: re-run with --strandedness none"
    )


def test_declarations(libraries, eng):
    argv = [*BASE, "--strandedness", "reverse", *ARGS]
    for work, declared, verdict in (
        (libraries[0], "reverse", "pass"),
        (libraries[0], "unstranded", "fail"),
        (libraries[0], "forward", "fail"),
        (libraries[1], "reverse", "fail"),
    ):
        _, result = outcome(
            eng, RULE, "rsem-calculate-expression", argv, work, {"library_strandedness": declared}
        )
        assert result.verdict == verdict, (declared, result)


def test_calls_on_reads_and_unreadable_inputs_are_not_judged(libraries, eng):
    work = libraries[1]
    cases = [
        ["--paired-end", "--strandedness", "reverse", "r1.fq", "r2.fq", "rsem_ref", "out"],  # reads
        [*BASE, "--strandedness", "reverse", "missing.bam", "rsem_ref", "out"],
        [*BASE, "--strandedness", "sideways", *ARGS],  # RSEM refuses it
        [*BASE, "--forward-prob", "high", *ARGS],
        [*BASE, "--strandedness", "reverse", "-", "rsem_ref", "out"],
    ]
    for argv in cases:
        _, result = outcome(eng, RULE, "rsem-calculate-expression", argv, work)
        assert result.verdict == "pass", argv


def test_the_checker_reads_the_bam_so_a_t0_context_refuses(libraries, spec):
    work = libraries[1]
    call = Call("rsem-calculate-expression", (*BASE, "--strandedness", "reverse", *ARGS), work)
    check = known_checkers()["rsem_strandedness"]
    with pytest.raises(TierViolation):
        check(call, Match(spec.rules[0]), Ctx(work, "T0"))
    assert check(call, Match(spec.rules[0]), Ctx(work, "T1")).verdict == "fail"


def test_every_template_spelling_gives_the_numbers_of_its_key(tools, tmp_path):
    work = tmp_path / "w"
    truth = build_workspace(2, "faulted", work, tools)
    by_key = {}
    for key, option in (
        ("none", ["--strandedness", "none"]),
        ("forward", ["--strandedness", "forward"]),
        ("reverse", ["--strandedness", "reverse"]),
    ):
        by_key[key] = rsem_counts(work, tools, truth, f"out_{key}", *option)
    for t in TEMPLATES:
        proc = tools.run([t.tool, *t.argv[:-1], f"s_{t.name}"], work)
        got = None
        if proc.returncode == 0:
            rows = [
                r.split("\t")
                for r in (work / f"s_{t.name}.genes.results").read_text().splitlines()[1:]
            ]
            got = {r[0]: float(r[4]) for r in rows}
        want = by_key[t.key]
        right = got is not None and answer_matches(truth, got)
        assert right == (want is not None and answer_matches(truth, want)), t.name


def test_a_stranded_option_on_an_unstranded_library_halves_the_counts_silently(tools, tmp_path):
    """The evidence of the module docstring: exit 0, empty stderr, about half of the fragments."""
    work = tmp_path / "w"
    truth = build_workspace(4, "faulted", work, tools)
    argv = ["rsem-calculate-expression", *BASE, "--strandedness", "reverse", *ARGS]
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and proc.stderr == ""
    total = sum(truth["expected"].values())
    kept = sum(rsem_counts(work, tools, truth, "again", "--strandedness", "reverse").values())
    assert 0.3 * total < kept < 0.7 * total


def test_admission(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(SEEDS), tools, tmp_path)
    needed = max(1, 200 * SEEDS // 300)
    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert stats.cells[("right", True)].n >= needed and stats.cells[("right", True)].failed == 0
    assert stats.cells[("right", False)].n >= needed and stats.cells[("right", False)].failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert promotion_is_consistent(stats)
