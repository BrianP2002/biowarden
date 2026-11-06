"""Rule rdkit.pdb_bond_orders: the checker on real argv and files, the RDKit facts it relies on (run
with the real library), and admission over calls labelled by running them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra import f100_rdkit_bonds as f100
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_rdkit as sr
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "rdkit.pdb_bond_orders"
KEY = "bond_orders"
PY = "python-struct"
RD = "from rdkit import Chem; from rdkit.Chem import Descriptors; "
MOLWT = RD + "m = Chem.MolFromPDBFile('ligand.pdb'); print(Descriptors.MolWt(m))"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_rdkit")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_ligand(path: Path, twin: str, seed: int = 0) -> None:
    path.write_text(f100.pdb_text(f100.make_ligand(seed), twin))


def outcome(eng: Engine, code: str, cwd: Path, declared=None):
    decision = eng.evaluate(Call(PY, ("-c", code), cwd, declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "fact" and rule.tier == "T1"
    assert rule.check in known_checkers() and rule.declare.key == KEY and not rule.declare.required
    assert rule.declare.values == ["conect", "template", "single_bonds"]


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_ligand(tmp_path / "ligand.pdb", "faulted")
    call = Call(PY, ("-c", MOLWT), tmp_path)
    with pytest.raises(TierViolation):
        sr.rdkit_pdb_bond_orders(
            call, Match(spec.rules[0], (tmp_path / "ligand.pdb",)), Ctx(tmp_path, "T0")
        )


@pytest.mark.parametrize("seed", range(6))
def test_every_family_ligand_is_flagged_without_conect_and_silent_with_it(spec, tmp_path, seed):
    eng = engine_for(spec)
    write_ligand(tmp_path / "ligand.pdb", "faulted", seed)
    decision, result = outcome(eng, MOLWT, tmp_path)
    assert decision.action == "block" and result.code == "bond_orders_undeclared"
    write_ligand(tmp_path / "ligand.pdb", "clean", seed)
    assert outcome(eng, MOLWT, tmp_path)[0].action == "allow"


@pytest.mark.parametrize(
    "property_call",
    ["Descriptors.MolWt(m)", "Descriptors.ExactMolWt(m)", "rdMolDescriptors.CalcMolFormula(m)",
     "sum(a.GetIsAromatic() for a in m.GetAtoms())", "Chem.MolToSmiles(m)"],
)  # fmt: skip
def test_bond_order_properties_on_a_ligand_without_conect_fail_until_declared(
    spec, tmp_path, property_call
):
    write_ligand(tmp_path / "ligand.pdb", "faulted", 1)
    eng = engine_for(spec)
    code = RD + f"m = Chem.MolFromPDBFile('ligand.pdb', removeHs=False); print({property_call})"
    decision, result = outcome(eng, code, tmp_path)
    assert decision.action == "block" and result.code == "bond_orders_undeclared"
    text = render_block(decision)
    assert text.startswith("FIX: assign the orders from a template")
    assert "no CONECT record repeats a serial" in text and "ligand.pdb" in text
    assert outcome(eng, code, tmp_path, {KEY: "single_bonds"})[0].action == "allow"
    decision, result = outcome(eng, code, tmp_path, {KEY: "conect"})
    assert result.code == "declared_mismatch" and decision.action == "block"
    decision, result = outcome(eng, code, tmp_path, {KEY: "template"})
    assert result.code == "declared_mismatch"  # no template code in the command


def test_template_code_sanitize_off_and_weight_sums_stay_silent(spec, tmp_path):
    write_ligand(tmp_path / "ligand.pdb", "faulted", 2)
    eng = engine_for(spec)
    silent = [
        RD + "m = Chem.MolFromPDBFile('ligand.pdb', sanitize=False, removeHs=False); "
        "print(Descriptors.MolWt(m))",
        RD + "m = Chem.MolFromPDBFile('ligand.pdb', removeHs=False); "
        "print(sum(a.GetMass() for a in m.GetAtoms()))",
        RD + "from rdkit.Chem import AllChem; t = Chem.MolFromSmiles('NC(=O)c1cccnc1'); "
        "m = AllChem.AssignBondOrdersFromTemplate(t, Chem.MolFromPDBFile('ligand.pdb')); "
        "print(Descriptors.MolWt(m))",
        RD + "m = Chem.MolFromPDBFile('ligand.pdb'); print(m.GetNumAtoms())",
        "import gemmi; st = gemmi.read_structure('ligand.pdb'); print(len(st[0]['A']))",
    ]
    for code in silent:
        decision, result = outcome(eng, code, tmp_path)
        assert decision.action == "allow", code
    template = silent[2]
    assert outcome(eng, template, tmp_path, {KEY: "template"})[0].action == "allow"
    for tool, argv in (("awk", ["{print $1}", "ligand.pdb"]), ("head", ["ligand.pdb"])):
        assert eng.evaluate(Call(tool, tuple(argv), tmp_path)).action == "allow"


def test_saturated_ligands_and_protein_atoms_cost_nothing(spec, tmp_path):
    # cyclohexanol-like geometry: every heavy-atom bond is a single bond length
    lines = []
    for i, (name, el, x) in enumerate([("C1", "C", 0.0), ("C2", "C", 1.53), ("O1", "O", 2.96)], 1):
        coords = f"{x:8.3f}{0:8.3f}{0:8.3f}"
        lines.append(f"HETATM{i:5d} {name:<4} LIG A   1    {coords}  1.00 20.00          {el:>2}")
    (tmp_path / "ligand.pdb").write_text("\n".join([*lines, "END"]) + "\n")
    assert outcome(engine_for(spec), MOLWT, tmp_path)[0].action == "allow"
    (tmp_path / "bad.pdb").write_text("hello\n")
    code = MOLWT.replace("ligand.pdb", "bad.pdb")
    decision, result = outcome(engine_for(spec), code, tmp_path)
    assert decision.action == "allow"  # no HETATM atoms: nothing to say


def test_rdkit_reads_conect_orders_and_ignores_their_absence(tools, tmp_path):
    results = {}
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        work.mkdir()
        write_ligand(work / "ligand.pdb", twin, 0)  # benzoic acid
        code = (
            RD + "from rdkit.Chem import rdMolDescriptors as D; import json; "
            "m = Chem.MolFromPDBFile('ligand.pdb'); "
            "mk = Chem.MolFromPDBFile('ligand.pdb', removeHs=False); "
            "print(json.dumps({'mw': round(Descriptors.MolWt(m), 3), "
            "'mw_keep': round(Descriptors.MolWt(mk), 3), 'formula': D.CalcMolFormula(m), "
            "'aromatic': sum(a.GetIsAromatic() for a in m.GetAtoms())}))"
        )
        proc = tools.run([PY, "-s", "-c", code], work)
        assert proc.returncode == 0 and proc.stderr == ""
        results[twin] = json.loads(proc.stdout.strip().splitlines()[-1])
    assert results["clean"] == {
        "mw": 122.123,
        "mw_keep": 122.123,
        "formula": "C7H6O2",
        "aromatic": 6,
    }
    assert results["faulted"] == {
        "mw": 130.187,
        "mw_keep": 130.187,
        "formula": "C7H14O2",
        "aromatic": 0,
    }


def test_the_listed_atoms_sum_to_the_true_weight(tools, tmp_path):
    for seed in range(6):
        ligand = f100.make_ligand(seed)
        code = (
            "from rdkit import Chem; from rdkit.Chem import Descriptors; "
            f"print(Descriptors.MolWt(Chem.MolFromSmiles({ligand['smiles']!r})))"
        )
        proc = tools.run([PY, "-s", "-c", code], tmp_path)
        assert float(proc.stdout) == pytest.approx(f100.molecular_weight(ligand["atoms"]), abs=0.01)


def test_admission_on_calls_labelled_by_the_real_library(spec, tools, tmp_path):
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
