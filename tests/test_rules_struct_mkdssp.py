"""Rule mkdssp.poly_seq_scheme: the checker on real argv and files, the mkdssp facts it relies on
(run with the real program), and admission over calls labelled by running mkdssp."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Chain,
    make_residues,
    mmcif_text,
    pdb_text,
    random_fold,
    random_sequence,
)
from biowarden.bench.tasks.extra.f98_mkdssp_scheme import dssp_helix_count
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_mkdssp as sk
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "mkdssp.poly_seq_scheme"
KEY = "mkdssp_input"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "mkdssp" not in env.real_bins(["mkdssp"]):
        pytest.skip("mkdssp is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_mkdssp")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_model(cwd: Path, scheme: bool, name: str = "model.cif") -> list:
    rng = random.Random(3)
    helix = ([-57.0] * 12, [-47.0] * 12)
    rest = random_fold(rng, 28)
    fold = (helix[0] + rest[0], helix[1] + rest[1])
    res = make_residues(rng, random_sequence(rng, 40), 1, fold=fold)
    (cwd / name).write_text(mmcif_text([Chain("A", res)], "m", poly_scheme=scheme))
    (cwd / "model.pdb").write_text(pdb_text([Chain("A", res)]))
    return res


def outcome(eng: Engine, argv, cwd: Path, declared=None):
    decision = eng.evaluate(Call("mkdssp", tuple(argv), cwd, declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "call" and rule.tier == "T1"
    assert rule.applies.tools == ["mkdssp"] and rule.check in known_checkers()
    assert rule.declare.key == KEY and rule.declare.values == ["mmcif_complete", "pdb_converted"]
    assert rule.declare.required is False


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_model(tmp_path, scheme=False)
    call = Call("mkdssp", ("model.cif", "o.dssp"), tmp_path)
    with pytest.raises(TierViolation):
        sk.mkdssp_poly_seq_scheme(call, Match(spec.rules[0]), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    "argv",
    [
        ["model.cif", "o.dssp"],
        ["./model.cif", "o.dssp"],
        ["--quiet", "model.cif", "o.dssp"],
        ["-v", "model.cif", "o.dssp"],
        ["--output-format", "dssp", "model.cif", "o.txt"],
        ["--output-format", "mmcif", "model.cif"],
        ["--min-pp-stretch", "4", "model.cif", "o.dssp"],
    ],
)
def test_an_mmcif_without_the_residue_table_fails_until_converted(spec, tmp_path, argv):
    write_model(tmp_path, scheme=False)
    eng = engine_for(spec)
    decision, result = outcome(eng, argv, tmp_path)
    assert decision.action == "block" and result.code == "poly_seq_scheme_missing"
    text = render_block(decision)
    assert text.startswith("FIX: convert first: python-struct -c")
    assert (
        "write_pdb('model.pdb')" in text
        and "0 residues" in text
        and "_pdbx_poly_seq_scheme" in text
    )
    for value in ("mmcif_complete", "pdb_converted"):  # still the same .cif: both claims are false
        decision, result = outcome(eng, argv, tmp_path, {KEY: value})
        assert result.code == "declared_mismatch" and decision.action == "block"


def test_complete_mmcif_pdb_input_and_unrelated_calls_stay_silent(spec, tmp_path):
    write_model(tmp_path, scheme=True, name="full.cif")
    write_model(tmp_path, scheme=False, name="bare.cif")
    eng = engine_for(spec)
    for argv in (["full.cif", "o.dssp"], ["model.pdb", "o.dssp"], ["bare.pdb", "o.dssp"],
                 ["--version"], ["--help"], ["missing.cif", "o.dssp"], []):  # fmt: skip
        decision, result = outcome(eng, argv, tmp_path)
        assert decision.action == "allow", argv
    for value in ("mmcif_complete", "pdb_converted"):
        assert outcome(eng, ["full.cif", "o.dssp"], tmp_path, {KEY: value})[0].action == "allow"
        assert outcome(eng, ["model.pdb", "o.dssp"], tmp_path, {KEY: value})[0].action == "allow"
    (tmp_path / "junk.cif").write_text("\x00\x01 not a cif")
    decision, result = outcome(eng, ["junk.cif", "o.dssp"], tmp_path)
    assert decision.action in ("allow", "warn")


def test_gzipped_mmcif_is_read(spec, tmp_path):
    import gzip

    write_model(tmp_path, scheme=False)
    with gzip.open(tmp_path / "model.cif.gz", "wt") as handle:
        handle.write((tmp_path / "model.cif").read_text())
    decision, result = outcome(engine_for(spec), ["model.cif.gz", "o.dssp"], tmp_path)
    assert result.code == "poly_seq_scheme_missing"


def test_the_precheck_checks_the_mkdssp_stage(spec, tmp_path):
    write_model(tmp_path, scheme=False)
    result = precheck("mkdssp model.cif o.dssp && grep -c ' H ' o.dssp", tmp_path, engine_for(spec))
    assert result.action == "block" and "no _pdbx_poly_seq_scheme table" in result.text
    ok = precheck("mkdssp model.pdb o.dssp && grep -c ' H ' o.dssp", tmp_path, engine_for(spec))
    assert ok.action == "allow"


def test_mkdssp_reads_residues_from_the_scheme_table(tools, tmp_path):
    write_model(tmp_path, scheme=True, name="full.cif")
    write_model(tmp_path, scheme=False, name="bare.cif")
    counts = {}
    for name in ("full.cif", "bare.cif", "model.pdb"):
        proc = tools.run(["mkdssp", name, f"{name}.dssp"], tmp_path)
        assert proc.returncode == 0
        rows = (tmp_path / f"{name}.dssp").read_text()
        counts[name] = (dssp_helix_count(rows), rows.splitlines()[6].split()[0])
    assert counts["full.cif"] == counts["model.pdb"] and counts["full.cif"][0] >= 8
    assert counts["bare.cif"] == (0, "0")  # a valid, empty result
    # the output format follows the extension of the output file
    tools.run(["mkdssp", "model.pdb", "o.txt"], tmp_path)
    assert (tmp_path / "o.txt").read_text().startswith("data_")
    shutil.copy(tmp_path / "o.txt", tmp_path / "again.cif")


def test_admission_on_calls_labelled_by_running_mkdssp(spec, tools, tmp_path):
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
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.5
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
