"""Rule mdtraj.resid_index: the checker on real argv and files, the mdtraj facts it relies on (run
with the real library), and admission over calls labelled by running the libraries."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Chain,
    make_residues,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_mdtraj as sm
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "mdtraj.resid_index"
KEY = "residue_numbering"
PY = "python-struct"
LOAD = "import mdtraj as md; top = md.load_pdb('p.pdb').topology; "


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_mdtraj")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_pdb(path: Path, start: int, n: int = 20, gap_after: int | None = None) -> None:
    rng = random.Random(4)
    res = make_residues(rng, random_sequence(rng, n), start)
    if gap_after is not None:
        res = [r for r in res if r.num != gap_after]
    path.write_text(pdb_text([Chain("A", res)]))


def outcome(eng: Engine, code: str, cwd: Path, declared=None):
    decision = eng.evaluate(Call(PY, ("-c", code), cwd, declared or {}))
    mine = [o for o in decision.outcomes if o.rule.id == RULE]
    return decision, (mine[0].result if mine else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "fact" and rule.tier == "T1"
    assert rule.check in known_checkers() and rule.declare.key == KEY
    assert rule.declare.required is False and set(rule.declare.values) == {
        "pdb_number",
        "residue_index",
    }


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb", 1)
    call = Call(PY, ("-c", LOAD + "top.select('resid 3')"), tmp_path)
    with pytest.raises(TierViolation):
        sm.mdtraj_resid_index(
            call, Match(spec.rules[0], (tmp_path / "p.pdb",)), Ctx(tmp_path, "T0")
        )


@pytest.mark.parametrize(
    "select",
    ["resid 5 and name CA", "name CA and resid 5", "(resid 5 or resid 9) and name CA",
     "resid 5 to 9", "resid 5 and backbone"],
)  # fmt: skip
def test_resid_on_a_chain_that_starts_at_one_fails_until_declared(spec, tmp_path, select):
    write_pdb(tmp_path / "p.pdb", 1)  # index 5 is residue 6
    eng = engine_for(spec)
    code = LOAD + f"print(top.select({select!r}))"
    decision, result = outcome(eng, code, tmp_path)
    assert decision.action == "block" and result.code == "resid_index_undeclared"
    text = render_block(decision)
    assert text.startswith('FIX: select by PDB number: top.select("resSeq 5 and name CA")')
    assert "0-based residue index" in text and "residue number 6" in text
    assert outcome(eng, code, tmp_path, {KEY: "residue_index"})[0].action == "allow"
    decision, result = outcome(eng, code, tmp_path, {KEY: "pdb_number"})
    assert decision.action == "block" and result.code == "declared_mismatch"
    assert "you declared residue_numbering=pdb_number" in render_block(decision)


def test_numbering_from_zero_resSeq_and_other_libraries_stay_silent(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb", 0)  # index = number
    write_pdb(tmp_path / "q.pdb", 1)
    eng = engine_for(spec)
    for code in (
        LOAD + "top.select('resid 5 and name CA')",  # numbered from 0: index = number
        "import mdtraj as md; top = md.load_pdb('q.pdb').topology; top.select('resSeq 5')",
        "import mdtraj as md; top = md.load_pdb('q.pdb').topology; top.select('name CA')",
        "import MDAnalysis as mda; u = mda.Universe('q.pdb'); u.select_atoms('resid 5')",
        "import prody; prody.parsePDB('q.pdb').select('resnum 5')",
        "import gemmi; print(gemmi.read_structure('q.pdb')[0]['A'][5].name)",
    ):
        decision, result = outcome(eng, code, tmp_path)
        assert decision.action == "allow", code
        assert result is None or result.verdict == "pass"


def test_gaps_in_the_numbering_and_indices_beyond_the_chain(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb", 1, n=20, gap_after=7)  # 19 residues: numbers 1..6, 8..20
    eng = engine_for(spec)
    # index 6 is residue 8, and index 25 does not exist
    for n, expected in ((6, "residue number 8"), (25, "no residue")):
        code = LOAD + f"top.select('resid {n} and name CA')"
        decision, result = outcome(eng, code, tmp_path)
        assert result.code == "resid_index_undeclared" and expected in render_block(decision)
    # index 0..5 are residues 1..6 (shifted by one: still wrong), only a chain from 0 is right
    assert (
        outcome(eng, LOAD + "top.select('resid 0')", tmp_path)[1].code == "resid_index_undeclared"
    )


def test_unreadable_files_are_unknown(spec, tmp_path):
    (tmp_path / "p.pdb").write_text("hello\n")
    decision, result = outcome(engine_for(spec), LOAD + "top.select('resid 3')", tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_mdtraj_resid_is_the_index_and_resSeq_is_the_number(tools, tmp_path):
    write_pdb(tmp_path / "p.pdb", 19, n=22, gap_after=27)
    code = (
        "import json, mdtraj as md, gemmi; t = md.load_pdb('p.pdb'); top = t.topology; "
        "num = lambda sel: [top.atom(i).residue.resSeq for i in top.select(sel)]; "
        "ch = gemmi.read_structure('p.pdb')[0]['A']; "
        "ca = lambda n: [r['CA'][0].pos for r in ch if r.seqid.num == n][0]; "
        "d = md.compute_distances(t, [[top.select('resSeq 20 and name CA')[0], "
        "top.select('resSeq 30 and name CA')[0]]])[0, 0]; "
        "print(json.dumps({'resid8': num('resid 8 and name CA'), 'resid25': num('resid 25'), "
        "'resSeq25': num('resSeq 25 and name CA'), 'nm': round(float(d), 3), "
        "'gemmi': round(ca(20).dist(ca(30)), 2)}))"
    )
    proc = tools.run([PY, "-s", "-c", code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["resid25"] == []  # index 25 does not exist: an empty selection, no error
    assert out["resid8"] == [28]  # index 8 is residue 28: 19..26 are indices 0..7, 27 is absent
    assert out["resSeq25"] == [25]
    assert abs(out["nm"] * 10 - out["gemmi"]) < 0.01  # mdtraj distances are nanometres


def test_admission_on_calls_labelled_by_the_real_libraries(spec, tools, tmp_path):
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
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.2
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
