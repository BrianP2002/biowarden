"""Rule tmalign.normalisation: the checker on real argv and files, the TMalign facts it relies on
(run with the real program), the pipeline pre-check, and admission over calls labelled by running
TMalign."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Chain,
    make_residues,
    mmcif_text,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_tmalign as st
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "tmalign.normalisation"
KEY = "tmscore_normalisation"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "TMalign" not in env.real_bins(["TMalign"]):
        pytest.skip("TMalign is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_tmalign")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_pair(cwd: Path, n_model: int, n_ref: int, two_chains: bool = False) -> None:
    rng = random.Random(7)
    ref = make_residues(rng, random_sequence(rng, n_ref), 1)
    chains = [Chain("A", ref)]
    if two_chains:
        chains.append(Chain("B", make_residues(rng, random_sequence(rng, 10), 1)))
    (cwd / "reference.pdb").write_text(pdb_text(chains))
    (cwd / "model.pdb").write_text(pdb_text([Chain("A", ref[:n_model])]))


def outcome(eng: Engine, argv, cwd: Path, declared=None):
    decision = eng.evaluate(Call("TMalign", tuple(argv), cwd, declared or {}))
    return decision, decision.outcomes[0].result if decision.outcomes else None


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "call" and rule.tier == "T1"
    assert rule.applies.tools == ["TMalign"] and rule.check in known_checkers()
    assert rule.declare.key == KEY and rule.declare.required is False
    assert set(rule.declare.values) == {"structure_1", "structure_2", "average", "given_length"}


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_pair(tmp_path, 20, 30)
    call = Call("TMalign", ("model.pdb", "reference.pdb"), tmp_path)
    with pytest.raises(TierViolation):
        st.tmalign_normalisation(call, Match(spec.rules[0]), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    "argv",
    [
        ["model.pdb", "reference.pdb"],
        ["./model.pdb", "./reference.pdb"],
        ["model.pdb", "reference.pdb", "-fast"],
        ["-fast", "model.pdb", "reference.pdb"],
        ["model.pdb", "reference.pdb", "-d", "5"],
        ["model.pdb", "reference.pdb", "-m", "matrix.txt", "-o", "sup.pdb"],
        ["model.pdb", "reference.pdb", "-ter", "3"],
    ],
)
def test_different_lengths_fail_until_the_normalisation_is_declared(spec, tmp_path, argv):
    write_pair(tmp_path, 20, 30)
    eng = engine_for(spec)
    decision, result = outcome(eng, argv, tmp_path)
    assert decision.action == "block" and result.code == "tm_norm_undeclared"
    text = render_block(decision)
    assert "structure_1 = " in text and "(20 residues)" in text and "(30 residues)" in text
    assert "normalized by length of Structure_2" in text and f"--declare {KEY}=structure_2" in text
    for value in ("structure_1", "structure_2"):
        assert outcome(eng, argv, tmp_path, {KEY: value})[0].action == "allow"


def test_options_that_ask_for_a_normalisation_and_their_declarations(spec, tmp_path):
    write_pair(tmp_path, 20, 30)
    eng = engine_for(spec)
    base = ["model.pdb", "reference.pdb"]
    for extra, value in ((["-a", "T"], "average"), (["-u", "30"], "given_length"),
                         (["-L", "30"], "given_length"), (["-a", "True"], "average")):  # fmt: skip
        assert outcome(eng, [*base, *extra], tmp_path)[0].action == "allow"
        assert outcome(eng, [*base, *extra], tmp_path, {KEY: value})[0].action == "allow"
    # a declaration the call cannot back
    for value, option in (("average", "-a T"), ("given_length", "-u <length>")):
        decision, result = outcome(eng, base, tmp_path, {KEY: value})
        assert result.code == "declared_mismatch" and decision.action == "block"
        assert option in render_block(decision)
    # -a F does not ask for the average
    assert outcome(eng, [*base, "-a", "F"], tmp_path)[1].code == "tm_norm_undeclared"


def test_equal_lengths_other_modes_and_unreadable_files_stay_silent(spec, tmp_path):
    write_pair(tmp_path, 30, 30)
    eng = engine_for(spec)
    for argv in (["model.pdb", "reference.pdb"], ["reference.pdb", "model.pdb", "-a", "T"]):
        decision, result = outcome(eng, argv, tmp_path)
        assert decision.action == "allow" and result.verdict == "pass"
    assert outcome(eng, ["-dir", "chains/", "list.txt"], tmp_path)[0].action == "allow"
    assert outcome(eng, ["-h"], tmp_path)[0].action == "allow"
    decision, result = outcome(eng, ["model.pdb", "missing.pdb"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_only_the_first_chain_counts_by_default(spec, tmp_path):
    write_pair(tmp_path, 30, 30, two_chains=True)  # reference.pdb: 30 + 10 residues in two chains
    eng = engine_for(spec)
    assert outcome(eng, ["model.pdb", "reference.pdb"], tmp_path)[0].action == "allow"
    decision, result = outcome(eng, ["model.pdb", "reference.pdb", "-ter", "0"], tmp_path)
    assert result.code == "tm_norm_undeclared" and "(40 residues)" in render_block(decision)


def test_mmcif_input_is_counted_through_gemmi(spec, tmp_path):
    rng = random.Random(5)
    ref = make_residues(rng, random_sequence(rng, 30), 1)
    (tmp_path / "reference.cif").write_text(mmcif_text([Chain("A", ref)], "r"))
    (tmp_path / "model.pdb").write_text(pdb_text([Chain("A", ref[:20])]))
    decision, result = outcome(engine_for(spec), ["model.pdb", "reference.cif"], tmp_path)
    assert result.code == "tm_norm_undeclared" and "(30 residues)" in render_block(decision)


def test_the_precheck_evaluates_the_tmalign_stage_of_a_pipeline(spec, tmp_path):
    write_pair(tmp_path, 20, 30)
    eng = engine_for(spec)
    result = precheck("TMalign model.pdb reference.pdb | grep -m1 TM-score", tmp_path, eng)
    assert result.action == "block" and "tm_norm_undeclared" not in result.text
    assert "TMalign" not in result.text or "normalized by" in result.text
    ok = precheck(
        f"TMalign model.pdb reference.pdb --declare {KEY}=structure_2 | grep Structure_2",
        tmp_path,
        eng,
    )
    assert ok.action == "allow"


def test_tmalign_prints_two_scores_and_options_only_add_lines(tools, tmp_path):
    write_pair(tmp_path, 20, 30)
    lines = {}
    for name, extra in (
        ("plain", []),
        ("avg", ["-a", "T"]),
        ("user", ["-u", "30"]),
        ("d0", ["-d", "5"]),
    ):
        proc = tools.run(["TMalign", "model.pdb", "reference.pdb", *extra], tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        lines[name] = re.findall(r"TM-score=\s*([0-9.]+)\s*\((.*?)[:)]", proc.stdout)
    first, second = lines["plain"]
    assert "Structure_1" in first[1] and "Structure_2" in second[1]
    assert float(first[0]) > float(second[0])  # the model-normalised score is the larger one
    for name in ("avg", "user", "d0"):
        assert lines[name][:2] == lines["plain"] and len(lines[name]) == 3
    assert "average length" in lines["avg"][2][1] and "user-specified" in lines["user"][2][1]
    assert lines["user"][2][0] == second[0]  # -u 30 is the reference length: the same value


def test_tmalign_aligns_only_the_first_chain_by_default(tools, tmp_path):
    write_pair(tmp_path, 30, 30, two_chains=True)
    default = tools.run(["TMalign", "model.pdb", "reference.pdb"], tmp_path).stdout
    whole = tools.run(["TMalign", "model.pdb", "reference.pdb", "-ter", "0"], tmp_path).stdout
    assert "Length of Structure_2: 30 residues" in default
    assert "Length of Structure_2: 40 residues" in whole


def test_admission_on_calls_labelled_by_running_tmalign(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
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
    assert stats.right_declared.n >= MIN_RIGHT_CALLS
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
