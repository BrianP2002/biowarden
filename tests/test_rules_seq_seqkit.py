"""Rule seqkit.rmdup_strands (F82): the checker on real argv and sequence files, the seqkit facts it
relies on (run with the real program), and admission over calls labelled by running seqkit.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f77_blastn_task import (
    fasta_text,
    rand_dna,
    rand_prot,
    revcomp,
)
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_seqkit")
(RULE,) = SPEC.rules
KEY = "duplicate_strand_policy"
A1 = "ACGTACGTAAGGCCTTAA"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "seqkit" not in env.real_bins(["seqkit"]):
        pytest.skip("seqkit is not available")
    return env


def evaluate(argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("seqkit", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def six(path: Path) -> Path:
    """a1, its reverse complement a3, an exact copy a4, a lower-case copy a5, two unrelated."""
    records = [("a1", A1), ("a2", "GAATTCAAGTGAATTCAA"), ("a3", revcomp(A1)), ("a4", A1)]
    records += [("a5", A1.lower()), ("a6", "TTTTGGGGCCCCAAAA")]
    path.write_text(fasta_text(records))
    return path


def test_spec_is_a_conditional_t1_call_rule_on_seqkit():
    assert RULE.id == "seqkit.rmdup_strands" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["seqkit"] and RULE.declare.key == KEY
    assert not RULE.declare.required and set(RULE.declare.values) == {
        "both_strands",
        "same_strand_only",
    }
    assert RULE.check in known_checkers()


def test_the_checker_reads_the_sequence_file_so_it_needs_tier_t1(tmp_path):
    six(tmp_path / "s.fa")
    call = Call("seqkit", ("rmdup", "-s", "s.fa"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("args", "verdict"),
    [
        (["rmdup", "-s", "s.fa"], "fail"),
        (["rmdup", "--by-seq", "s.fa"], "fail"),
        (["rmdup", "-si", "s.fa"], "fail"),
        (["rmdup", "-s", "-i", "s.fa"], "fail"),
        (["rmdup", "-j", "2", "-s", "s.fa"], "fail"),
        (["rmdup", "-s", "-o", "out.fa", "s.fa"], "fail"),
        (["rmdup", "-s", "-d", "dups.fa", "s.fa"], "fail"),
        (["-j", "2", "rmdup", "-s", "s.fa"], "fail"),
        (["rmdup", "-s", "-P", "s.fa"], "pass"),
        (["rmdup", "-sP", "s.fa"], "pass"),
        (["rmdup", "--by-seq", "--only-positive-strand", "s.fa"], "pass"),
        (["rmdup", "s.fa"], "pass"),  # by ID: sequences are not compared
        (["rmdup", "-n", "s.fa"], "pass"),
        (["rmdup", "-s", "absent.fa"], "pass"),
        (["rmdup", "-s", "-"], "pass"),
        (["stats", "s.fa"], "pass"),
    ],
)
def test_the_strand_flag_decides_when_a_reverse_complement_pair_exists(tmp_path, args, verdict):
    six(tmp_path / "s.fa")
    decision, result = evaluate(args, tmp_path)
    assert result.verdict == verdict, (args, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_pairs_are_looked_for_in_the_right_case_and_alphabet(tmp_path):
    rng = random.Random(3)
    (tmp_path / "lower.fa").write_text(fasta_text([("x", A1), ("y", revcomp(A1).lower())]))
    # case is compared as written unless -i is given (verified below with seqkit)
    assert evaluate(["rmdup", "-s", "lower.fa"], tmp_path)[1].verdict == "pass"
    assert evaluate(["rmdup", "-s", "-i", "lower.fa"], tmp_path)[1].verdict == "fail"
    (tmp_path / "none.fa").write_text(fasta_text([(f"s{i}", rand_dna(rng, 20)) for i in range(8)]))
    assert evaluate(["rmdup", "-s", "none.fa"], tmp_path)[1].verdict == "pass"
    (tmp_path / "pal.fa").write_text(
        fasta_text([("p1", "GAATTC"), ("p2", "GAATTC")])
    )  # palindromes
    assert evaluate(["rmdup", "-s", "pal.fa"], tmp_path)[1].verdict == "pass"
    (tmp_path / "prot.fa").write_text(fasta_text([(f"p{i}", rand_prot(rng, 40)) for i in range(5)]))
    assert evaluate(["rmdup", "-s", "prot.fa"], tmp_path)[1].verdict == "pass"
    (tmp_path / "r.fq").write_text(f"@r1\n{A1}\n+\n{'I' * 18}\n@r2\n{revcomp(A1)}\n+\n{'I' * 18}\n")
    assert evaluate(["rmdup", "-s", "r.fq"], tmp_path)[1].verdict == "fail"
    (tmp_path / "zero.fa").write_text("")
    assert evaluate(["rmdup", "-s", "zero.fa"], tmp_path)[1].verdict == "pass"


def test_the_message_has_the_fix_first_and_the_instance_facts(tmp_path):
    six(tmp_path / "s.fa")
    decision, result = evaluate(["rmdup", "-s", "s.fa"], tmp_path)
    text = render_block(decision)
    assert text.startswith("FIX: add -P (--only-positive-strand) to compare sequences as written")
    assert result.code == "reverse_complement_duplicates"
    assert "s.fa holds 1 such pair(s)" in result.message
    assert A1 in result.message and revcomp(A1) in result.message


def test_declarations_of_the_duplicate_strand_policy(tmp_path):
    six(tmp_path / "s.fa")
    call = ["rmdup", "-s", "s.fa"]
    assert evaluate(call, tmp_path, {KEY: "both_strands"})[1].verdict == "pass"
    assert evaluate(call, tmp_path, {KEY: "same_strand_only"})[1].verdict == "fail"
    assert (
        evaluate(["rmdup", "-s", "-P", "s.fa"], tmp_path, {KEY: "same_strand_only"})[1].verdict
        == "pass"
    )
    assert evaluate(call, tmp_path, {KEY: "x"})[0].action == "block"


# ---- seqkit facts ----------------------------------------------------------------------------


def kept(tools, work, *options):
    proc = tools.run(["seqkit", "rmdup", *options, "s.fa"], work)
    assert proc.returncode == 0
    return [line[1:] for line in proc.stdout.splitlines() if line.startswith(">")], proc


def test_rmdup_by_sequence_drops_the_reverse_complement_unless_p_is_given(tools, tmp_path):
    six(tmp_path / "s.fa")
    both, proc = kept(tools, tmp_path, "-s")
    assert both == ["a1", "a2", "a5", "a6"] and "2 duplicated records removed" in proc.stderr
    assert kept(tools, tmp_path, "-s", "-P")[0] == ["a1", "a2", "a3", "a5", "a6"]
    assert kept(tools, tmp_path, "-s", "-i")[0] == ["a1", "a2", "a6"]
    assert kept(tools, tmp_path, "-n")[0] == ["a1", "a2", "a3", "a4", "a5", "a6"]
    assert "warn" not in proc.stderr.lower()


def test_family_workspaces_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        task = build_task("F82", 5, twin, work, tools, "implicit")
        verdict = evaluate(["rmdup", "-s", "guides.fasta"], work)[1].verdict
        assert verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate(["rmdup", "-s", "-P", "guides.fasta"], work)[1].verdict == "pass"
        both = tools.run(["seqkit", "rmdup", "-s", "guides.fasta"], work).stdout.count(">")
        assert both == task.truth["naive"]


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_seqkit(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print(
        "ADMISSION",
        RULE.id,
        "seeds",
        ADMISSION_SEEDS,
        cells,
        "recall",
        round(stats.recall() or 0, 3),
    )
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= MIN_RIGHT_CALLS
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
