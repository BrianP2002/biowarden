"""Rule struct.altloc_policy: the checker on real argv and files, the library facts it relies on
(run with the real libraries), and admission over calls labelled by running them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra import f90_altloc as f90
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_altloc as sa
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "struct.altloc_policy"
KEY = "altloc_policy"
PY = "python-struct"

PRODY = (
    "import prody; ag = prody.parsePDB('model.pdb'); c = ag.select('resnum 12 name CA').getCoords()"
)
GEMMI_REMOVE = (
    "import gemmi; st = gemmi.read_structure('model.pdb'); st.remove_alternative_conformations(); "
    "print(st[0]['A'][5]['CA'][0].pos)"
)
GEMMI_RAW = (
    "import gemmi; st = gemmi.read_structure('model.pdb'); print(st[0]['A'][5]['CA'][0].pos)"
)
TEXT_FIRST = (
    "rows = [l for l in open('model.pdb') if l[12:16] == ' CA ']; "
    "print([float(l[30:38]) for l in rows][:3])"
)
BIOPYTHON = (
    "from Bio.PDB import PDBParser; "
    "s = PDBParser(QUIET=True).get_structure('x', 'model.pdb'); "
    "print(s[0]['A'][12]['CA'].get_coord())"
)
BY_OCC = (
    "import prody; ag = prody.parsePDB('model.pdb'); print(ag.getOccupancies()); "
    "print(ag.getCoords()[0])"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_altloc")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_model(path: Path, twin: str, seed: int = 1) -> None:
    path.write_text(f90.pdb_text(f90.make_model(seed), twin))


def outcome(eng: Engine, code: str, cwd: Path, declared=None):
    decision = eng.evaluate(Call(PY, ("-c", code), cwd, declared or {}))
    mine = [o for o in decision.outcomes if o.rule.id == RULE]
    return decision, (mine[0].result if mine else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "fact" and rule.tier == "T1"
    assert rule.check in known_checkers() and rule.declare.key == KEY
    assert rule.declare.required is False
    assert set(rule.declare.values) == {"highest_occupancy", "first_listed", "all_conformers"}


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_model(tmp_path / "model.pdb", "faulted")
    call = Call(PY, ("-c", PRODY), tmp_path)
    with pytest.raises(TierViolation):
        sa.struct_altloc_policy(
            call, Match(spec.rules[0], (tmp_path / "model.pdb",)), Ctx(tmp_path, "T0")
        )


@pytest.mark.parametrize("code", [PRODY, GEMMI_REMOVE, TEXT_FIRST])
def test_first_conformer_code_on_a_file_whose_major_conformer_is_b_fails_until_declared(
    spec, tmp_path, code
):
    write_model(tmp_path / "model.pdb", "faulted")
    eng = engine_for(spec)
    decision, result = outcome(eng, code, tmp_path)
    assert decision.action == "block" and result.code == "altloc_policy_undeclared"
    text = render_block(decision)
    assert "altloc A (0." in text and "is listed first, altloc B (0." in text
    assert "has the higher occupancy" in text and "Bio.PDB.PDBParser" in text
    assert f"--declare {KEY}=highest_occupancy" in text
    for value in ("first_listed", "all_conformers"):
        assert outcome(eng, code, tmp_path, {KEY: value})[0].action == "allow"


@pytest.mark.parametrize("code", [PRODY, GEMMI_REMOVE])
def test_declared_highest_occupancy_with_a_first_conformer_default_contradicts(
    spec, tmp_path, code
):
    write_model(tmp_path / "model.pdb", "faulted")
    decision, result = outcome(engine_for(spec), code, tmp_path, {KEY: "highest_occupancy"})
    assert decision.action == "block" and result.code == "declared_mismatch"
    assert "you declared altloc_policy=highest_occupancy" in render_block(decision)


@pytest.mark.parametrize("code", [BIOPYTHON, BY_OCC])
def test_code_that_names_occupancy_or_uses_biopython_is_handled(spec, tmp_path, code):
    write_model(tmp_path / "model.pdb", "faulted")
    eng = engine_for(spec)
    assert outcome(eng, code, tmp_path)[0].action == "allow"
    for value in ("highest_occupancy", "first_listed"):
        assert outcome(eng, code, tmp_path, {KEY: value})[0].action == "allow"


def test_major_conformer_a_and_files_without_altlocs_stay_silent(spec, tmp_path):
    write_model(tmp_path / "model.pdb", "clean")  # A is the major conformer
    eng = engine_for(spec)
    for code in (PRODY, GEMMI_REMOVE, TEXT_FIRST):
        decision, result = outcome(eng, code, tmp_path)
        assert decision.action == "allow" and (result is None or result.verdict == "pass")
    model = f90.make_model(2)
    model["shift"] = {}  # no altloc residue at all
    (tmp_path / "plain.pdb").write_text(f90.pdb_text(model, "faulted"))
    code = PRODY.replace("model.pdb", "plain.pdb")
    assert outcome(eng, code, tmp_path)[0].action == "allow"


def test_inspection_and_other_tools_stay_silent(spec, tmp_path):
    write_model(tmp_path / "model.pdb", "faulted")
    eng = engine_for(spec)
    for code in (
        "print([l[:27] for l in open('model.pdb') if l[16] in 'AB'][:4])",
        "import gemmi; st = gemmi.read_structure('model.pdb'); print(len(st[0]['A']))",
        "import prody; ag = prody.parsePDB('model.pdb'); print(ag.numAtoms())",
        "print(open('model.pdb').read()[:100])",
    ):
        decision, result = outcome(eng, code, tmp_path)
        assert decision.action == "allow", code
    for tool, argv in (("head", ["model.pdb"]), ("awk", ["{print $6}", "model.pdb"]),
                       ("grep", ["-c", "ATOM", "model.pdb"])):  # fmt: skip
        assert eng.evaluate(Call(tool, tuple(argv), tmp_path)).action == "allow"


def test_script_files_are_read_and_unreadable_structures_are_unknown(spec, tmp_path):
    write_model(tmp_path / "model.pdb", "faulted")
    (tmp_path / "go.py").write_text(PRODY + "\nprint(c)\n")
    decision = engine_for(spec).evaluate(Call(PY, ("go.py", "model.pdb"), tmp_path))
    assert decision.action == "block"
    (tmp_path / "bad.pdb").write_text("nothing here\n")
    code = PRODY.replace("model.pdb", "bad.pdb")
    decision, result = outcome(engine_for(spec), code, tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_library_defaults_keep_different_conformers(tools, tmp_path):
    write_model(tmp_path / "model.pdb", "faulted", seed=3)
    model = f90.make_model(3)
    p = model["p"]
    code = (
        "import json, prody, gemmi; from Bio.PDB import PDBParser; "
        f"ca = lambda ag: ag.select('resnum {p} name CA'); "
        "d = prody.parsePDB('model.pdb'); "
        "st = gemmi.read_structure('model.pdb'); n_raw = sum(len(r) for r in st[0]['A']); "
        "st.remove_alternative_conformations(); n_one = sum(len(r) for r in st[0]['A']); "
        "bio = PDBParser(QUIET=True).get_structure('x', 'model.pdb')[0]['A']; "
        "out = {'prody_occ': float(ca(d).getOccupancies()[0]), "
        f"'prody_B': float(ca(prody.parsePDB('model.pdb', altloc='B')).getOccupancies()[0]), "
        f"'bio_occ': bio[{p}]['CA'].get_occupancy(), 'gemmi_raw': n_raw, 'gemmi_one': n_one, "
        f"'gemmi_kept_occ': [a.occ for r in st[0]['A'] if r.seqid.num == {p} for a in r "
        "if a.name == 'CA'][0]}; print(json.dumps(out))"
    )
    proc = tools.run([PY, "-s", "-c", code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    major = model["major"][p]
    assert out["prody_occ"] == pytest.approx(round(1 - major, 2), abs=1e-6)  # the minor one
    assert out["prody_B"] == pytest.approx(major, abs=1e-6)
    assert out["bio_occ"] == pytest.approx(major, abs=1e-6)  # Biopython: highest occupancy
    assert out["gemmi_kept_occ"] == pytest.approx(round(1 - major, 2), abs=1e-6)
    assert out["gemmi_raw"] > out["gemmi_one"]


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
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
