"""Rule mda.protein_selection: the checker on real argv and files, the residue-list facts it relies
on (run with the real libraries), and admission over calls labelled by running them."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Atom,
    Chain,
    Residue,
    make_residues,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_mda as sm
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "mda.protein_selection"
KEY = "protein_selection"
PY = "python-struct"
MDA = "import MDAnalysis as mda; u = mda.Universe('model.pdb'); "


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_mda")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_model(path: Path, names: dict[int, str]) -> None:
    rng = random.Random(8)
    res = make_residues(rng, random_sequence(rng, 16), 1)
    for i, name in names.items():
        res[i].name, res[i].het = name, True
    water = Residue("HOH", 101, [Atom("O", (3.0, 3.0, 3.0))], het=True)
    path.write_text(pdb_text([Chain("A", res), Chain("A", [water], polymer=False)]))


def outcome(eng: Engine, code: str, cwd: Path, declared=None):
    decision = eng.evaluate(Call(PY, ("-c", code), cwd, declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "fact" and rule.tier == "T1"
    assert rule.check in known_checkers() and rule.declare.key == KEY and not rule.declare.required
    assert set(rule.declare.values) == {"includes_modified", "standard_only"}
    assert set(rule.applies.tools) >= {"python-struct", "python3"}


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_model(tmp_path / "model.pdb", {3: "SEP"})
    call = Call(PY, ("-c", MDA + "u.select_atoms('protein')"), tmp_path)
    with pytest.raises(TierViolation):
        sm.mda_protein_selection(
            call, Match(spec.rules[0], (tmp_path / "model.pdb",)), Ctx(tmp_path, "T0")
        )


@pytest.mark.parametrize(
    "selection",
    ["protein", "protein and name CA", "backbone and name CA", "protein and not name H*",
     "nucleic or protein"],
)  # fmt: skip
def test_the_protein_keyword_on_a_file_with_phosphorylated_residues_fails_until_declared(
    spec, tmp_path, selection
):
    write_model(tmp_path / "model.pdb", {3: "SEP", 9: "TPO", 11: "MSE"})
    eng = engine_for(spec)
    code = MDA + f"print(u.select_atoms({selection!r}).n_residues)"
    decision, result = outcome(eng, code, tmp_path)
    assert decision.action == "block" and result.code == "protein_selection_undeclared"
    text = render_block(decision)
    assert "(SEP, TPO)" in text and "MSE" not in text.split("BLOCKED")[0]  # MSE is in the list
    assert text.startswith(
        "FIX: select by residue name as well: select_atoms('protein or resname SEP TPO')"
    )
    assert outcome(eng, code, tmp_path, {KEY: "standard_only"})[0].action == "allow"
    decision, result = outcome(eng, code, tmp_path, {KEY: "includes_modified"})
    assert result.code == "declared_mismatch" and decision.action == "block"


def test_residue_name_selections_other_libraries_and_clean_files_stay_silent(spec, tmp_path):
    write_model(tmp_path / "model.pdb", {3: "SEP"})
    write_model(tmp_path / "plain.pdb", {11: "MSE"})  # MSE is in MDAnalysis's list
    eng = engine_for(spec)
    for code in (
        MDA + "u.select_atoms('protein or resname SEP')",
        MDA + "u.select_atoms('not resname HOH')",
        MDA + "u.select_atoms('name CA')",
        MDA + "u.select_atoms('protein or name CA')",
        MDA + "u.atoms.n_residues",
        "import mdtraj as md; md.load_pdb('model.pdb').topology.select('protein')",
        "import prody; prody.parsePDB('model.pdb').select('protein')",
        MDA.replace("model.pdb", "plain.pdb") + "u.select_atoms('protein')",
    ):
        decision, result = outcome(eng, code, tmp_path)
        assert decision.action == "allow", code


def test_script_files_and_unreadable_structures(spec, tmp_path):
    write_model(tmp_path / "model.pdb", {3: "SEP"})
    (tmp_path / "go.py").write_text(MDA + "print(u.select_atoms('protein').n_residues)\n")
    decision = engine_for(spec).evaluate(Call(PY, ("go.py", "model.pdb"), tmp_path))
    assert decision.action == "block"
    (tmp_path / "bad.pdb").write_text("hello\n")
    decision, result = outcome(
        engine_for(spec),
        MDA.replace("model.pdb", "bad.pdb") + "u.select_atoms('protein')",
        tmp_path,
    )
    assert result.verdict == "unknown" and decision.action == "warn"


def test_the_copied_residue_list_matches_the_installed_library(tools, tmp_path):
    code = (
        "import json, warnings; warnings.simplefilter('ignore'); "
        "from MDAnalysis.core.selection import ProteinSelection as P; "
        "print(json.dumps(sorted(P.prot_res)))"
    )
    proc = tools.run([PY, "-I", "-c", code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert set(json.loads(proc.stdout.strip().splitlines()[-1])) == set(sm.MDA_PROTEIN)


def test_libraries_disagree_on_phosphorylated_residues(tools, tmp_path):
    write_model(tmp_path / "model.pdb", {3: "SEP", 9: "TPO", 11: "MSE"})
    code = (
        "import json, warnings; warnings.simplefilter('ignore'); import MDAnalysis as mda, "
        "mdtraj as md, prody; u = mda.Universe('model.pdb'); t = md.load_pdb('model.pdb'); "
        "print(json.dumps({'mda': u.select_atoms('protein').n_residues, 'all': u.atoms.n_residues, "
        "'mdtraj': len({t.topology.atom(i).residue.index for i in t.topology.select('protein')}), "
        "'prody': prody.parsePDB('model.pdb').select('protein').numResidues()}))"
    )
    proc = tools.run([PY, "-s", "-c", code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["all"] == 17 and out["mda"] == 14  # 16 residues + 1 water; SEP and TPO are missing
    assert out["mdtraj"] == out["prody"] == 16


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
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.5
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
