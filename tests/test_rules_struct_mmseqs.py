"""Rule mmseqs.cluster_coverage: the fragment detector, the checker on real argv and files, the
MMseqs2 facts it relies on (run with the real program), and admission over calls labelled by
running mmseqs (8 seeds: one run costs several seconds)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra import f94_mmseqs_coverage as f94
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_mmseqs as sm
from biowarden.engine.spec import load_packaged
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "mmseqs.cluster_coverage"
KEY = "fragment_policy"
BASE = ["easy-cluster", "seqs.fasta", "clu", "tmp", "--min-seq-id", "0.9"]
SEEDS = 8


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "mmseqs" not in env.real_bins(["mmseqs"]):
        pytest.skip("mmseqs is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_mmseqs")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_set(cwd: Path, twin: str, seed: int = 1) -> int:
    recs, n_fam, n_frag = f94.make_records(seed)
    (cwd / "seqs.fasta").write_text(f94.fasta(recs, twin))
    return n_frag if twin == "faulted" else 0


def outcome(eng: Engine, argv, cwd: Path, declared=None):
    decision = eng.evaluate(Call("mmseqs", tuple(argv), cwd, declared or {}))
    return decision, decision.outcomes[0].result if decision.outcomes else None


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "call" and rule.tier == "T1"
    assert rule.applies.tools == ["mmseqs"] and rule.check in known_checkers()
    assert set(rule.applies.subcommands) == {"easy-cluster", "cluster", "easy-linclust", "linclust"}
    assert rule.declare.key == KEY and rule.declare.required is False


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_set(tmp_path, "faulted")
    call = Call("mmseqs", tuple(BASE), tmp_path)
    with pytest.raises(TierViolation):
        sm.mmseqs_cluster_coverage(call, Match(spec.rules[0]), Ctx(tmp_path, "T0"))


def test_fragment_detector_finds_pieces_of_longer_sequences_only():
    parent = "MKTAYIAKQRQISFVKSHFSRQAGLDEMKTAYIAKQRQISFVKSHFSRQAGLDE"
    other = "GSHMLEDPVGTLKQWERTYIPASDFGHKLCVNMQWERTYIPLKJHGFDSAQWE"
    records = [("p", parent), ("frag", parent[5:30]), ("q", other), ("near", parent[:-3])]
    found = sm.fragments(records)
    assert found == [("frag", "p")]  # `near` is only 6% shorter than p; `other` is unrelated
    assert sm.fragments([("a", parent), ("b", parent)]) == []
    assert sm.read_fasta(">a desc\nMKT\nAYI\n>b\nGG\n") == [("a", "MKTAYI"), ("b", "GG")]


@pytest.mark.parametrize(
    "argv",
    [
        BASE,
        [*BASE, "-c", "0.3"],
        [*BASE, "--cov-mode", "0"],
        [*BASE, "--threads", "1", "-v", "1"],
        ["easy-linclust", "seqs.fasta", "clu", "tmp", "--min-seq-id", "0.9"],
        ["easy-cluster", "./seqs.fasta", "clu", "tmp"],
    ],
)
def test_fragments_with_the_default_coverage_mode_fail_until_declared(spec, tmp_path, argv):
    n_frag = write_set(tmp_path, "faulted")
    eng = engine_for(spec)
    decision, result = outcome(eng, argv, tmp_path)
    assert decision.action == "block" and result.code == "fragment_policy_undeclared"
    text = render_block(decision)
    assert text.startswith("FIX: add --cov-mode 1 and --declare fragment_policy=with_parent")
    assert f"holds {n_frag} fragment(s)" in text or "fragment(s) such as" in text
    assert outcome(eng, argv, tmp_path, {KEY: "separate"})[0].action == "allow"
    decision, result = outcome(eng, argv, tmp_path, {KEY: "with_parent"})
    assert result.code == "declared_mismatch" and decision.action == "block"
    assert "you declared fragment_policy=with_parent" in render_block(decision)


@pytest.mark.parametrize(
    "extra",
    [["--cov-mode", "1"], ["--cov-mode=1"], ["--cov-mode", "2", "-c", "0.3"],
     ["-c", "0.5", "--cov-mode", "1"]],
)  # fmt: skip
def test_member_or_representative_coverage_modes_stay_silent(spec, tmp_path, extra):
    write_set(tmp_path, "faulted")
    decision, result = outcome(engine_for(spec), [*BASE, *extra], tmp_path)
    assert decision.action == "allow" and result.verdict == "pass"


def test_sets_without_fragments_other_subcommands_and_databases_stay_silent(spec, tmp_path):
    write_set(tmp_path, "clean")
    eng = engine_for(spec)
    for argv in (BASE, [*BASE, "-c", "0.8"]):
        decision, result = outcome(eng, argv, tmp_path)
        assert decision.action == "allow" and result.verdict == "pass"
    write_set(tmp_path, "faulted")
    # easy-search is not a clustering call: the rule does not apply
    decision, result = outcome(
        eng, ["easy-search", "seqs.fasta", "seqs.fasta", "o.m8", "tmp"], tmp_path
    )
    assert decision.outcomes == () and decision.action == "allow"
    (tmp_path / "db").write_bytes(b"\x01\x00\x02binary database")
    assert outcome(eng, ["cluster", "db", "clu", "tmp"], tmp_path)[0].action == "allow"
    assert (
        outcome(eng, ["easy-cluster", "missing.fasta", "clu", "tmp"], tmp_path)[0].action == "allow"
    )


def test_default_coverage_loses_fragments_and_cov_mode_one_keeps_them(tools, tmp_path):
    recs, n_fam, n_frag = f94.make_records(2)
    text = f94.fasta(recs, "faulted")
    default = f94.cluster_count(tools, text)
    member = f94.cluster_count(tools, text, "--cov-mode", "1")
    assert (default, member) == (n_fam + n_frag, n_fam)
    assert f94.cluster_count(tools, f94.fasta(recs, "clean")) == n_fam


def test_admission_on_calls_labelled_by_running_mmseqs(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(SEEDS), tools, tmp_path)
    print(
        "ADMISSION",
        stats.rule_id,
        "seeds",
        stats.seeds,
        {
            f"{k[0]}_{'declared' if k[1] else 'undeclared'}": [c.n, c.failed, c.unknown, c.crashed]
            for k, c in stats.cells.items()
        },
        "recall",
        round(stats.recall() or 0, 3),
        "killed",
        stats.killed(),
    )
    assert stats.admitted(), stats.cells
    assert stats.right_declared.n >= 200 * SEEDS // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
