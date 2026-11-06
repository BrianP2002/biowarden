"""Rules prody.chain_scheme and prody.range_end: the checkers on real argv and files, the tool
facts they rely on (run with the real libraries), the engine path, and admission.

The unit tests write small structure files here (the helpers of the F89 family module) and run the
checkers through an Engine in block mode; the tool-fact tests run ProDy, gemmi, Biopython and
MDAnalysis through python-struct; the admission test measures both rules on calls labelled by
running the libraries (bench/admission/cases_struct_prody.py).
"""

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
    mmcif_text,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_prody as sp
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
CHAIN, RANGE = "prody.chain_scheme", "prody.range_end"
PY = "python-struct"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_prody")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def residues(n: int = 20, start: int = 1, seed: int = 1):
    rng = random.Random(seed)
    return make_residues(rng, random_sequence(rng, n), start)


def write_cif(path: Path, swapped: bool) -> None:
    """Two chains; swapped: author A is label B and author B is label A."""
    a, b = residues(18, 1, 1), residues(12, 1, 2)
    labels = ("B", "A") if swapped else ("A", "B")
    chains = [Chain("A", a, label=labels[0], entity=1), Chain("B", b, label=labels[1], entity=2)]
    chains.sort(key=lambda c: c.label)
    path.write_text(mmcif_text(chains, "x"))


def outcome(eng: Engine, argv, cwd: Path, rule: str, declared=None, tool: str = PY):
    decision = eng.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    mine = [o for o in decision.outcomes if o.rule.id == rule]
    return decision, (mine[0].result if mine else None)


# ---- spec and tier ---------------------------------------------------------------------------


def test_spec_is_valid_and_checkers_are_registered(spec):
    assert [r.id for r in spec.rules] == [CHAIN, RANGE]
    registry = known_checkers()
    for rule in spec.rules:
        assert rule.check in registry and rule.kind == "fact" and rule.tier == "T1"
        assert rule.declare is not None and rule.declare.required is False
        assert set(rule.applies.tools) >= {"python-struct", "python3"}


def test_checkers_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_cif(tmp_path / "c.cif", swapped=True)
    rule = spec.rules[0]
    ctx = Ctx(tmp_path, "T0")
    call = Call(PY, ("-c", "import prody; prody.parseMMCIF('c.cif')"), tmp_path)
    with pytest.raises(TierViolation):
        sp.prody_chain_scheme(call, Match(rule, (tmp_path / "c.cif",)), ctx)


# ---- prody.chain_scheme ----------------------------------------------------------------------

PARSERS = [
    "import prody; ag = prody.parseMMCIF('c.cif')",
    "import prody; ag = prody.parseCIF('c.cif')",
    "from prody import parseMMCIF; ag = parseMMCIF('c.cif')",
    "import prody; ag = prody.parsePDB('c.cif')",
    "import prody\nag = prody.parseMMCIF(\n    'c.cif'\n)",
    "import prody; f = 'c.cif'; ag = prody.parseMMCIF(f)",
]


@pytest.mark.parametrize("code", PARSERS)
def test_default_prody_parse_of_a_relabelled_cif_fails_until_declared(spec, tmp_path, code):
    write_cif(tmp_path / "c.cif", swapped=True)
    eng = engine_for(spec)
    decision, result = outcome(eng, ["-c", code], tmp_path, CHAIN)
    assert decision.action == "block" and result.code == "chain_scheme_undeclared"
    text = render_block(decision)
    assert text.startswith("FIX: use the author IDs: prody.parseMMCIF('c.cif', unite_chains=True)")
    for needle in ("author A is label B", "author B is label A", "c.cif", "gemmi"):
        assert needle in text, needle
    # declared label: the caller wants the archive IDs
    assert outcome(eng, ["-c", code], tmp_path, CHAIN, {"chain_id_scheme": "label"})[0].action == (
        "allow"
    )
    # declared auth with ProDy's default contradicts the data
    decision, result = outcome(eng, ["-c", code], tmp_path, CHAIN, {"chain_id_scheme": "auth"})
    assert decision.action == "block" and result.code == "declared_mismatch"
    assert "you declared chain_id_scheme=auth" in render_block(decision)


def test_safe_routes_and_unaffected_data_stay_silent(spec, tmp_path):
    write_cif(tmp_path / "c.cif", swapped=True)
    write_cif(tmp_path / "same.cif", swapped=False)
    (tmp_path / "p.pdb").write_text(pdb_text([Chain("A", residues())]))
    eng = engine_for(spec)
    silent = [
        (["-c", "import prody; prody.parseMMCIF('c.cif', unite_chains=True)"], "c.cif"),
        (["-c", "import prody; prody.parseMMCIF('c.cif', unite_chains = True)"], "c.cif"),
        (["-c", "import gemmi; gemmi.read_structure('c.cif')"], "c.cif"),
        (["-c", "from Bio.PDB import MMCIFParser; MMCIFParser().get_structure('x', 'c.cif')"], "c"),
        (["-c", "import prody; prody.parseMMCIF('same.cif')"], "same.cif"),  # label = auth
        (["-c", "import prody; prody.parsePDB('p.pdb')"], "p.pdb"),
        (["-c", "print(open('c.cif').read()[:50])"], "c.cif"),
    ]
    for argv, _ in silent:
        decision, result = outcome(eng, argv, tmp_path, CHAIN)
        assert decision.action == "allow", argv
        assert result is None or result.verdict == "pass"
    # a declaration of either value never blocks a safe call
    for value in ("auth", "label"):
        argv = ["-c", "import gemmi; gemmi.read_structure('c.cif')"]
        assert outcome(eng, argv, tmp_path, CHAIN, {"chain_id_scheme": value})[0].action == "allow"
    assert (
        outcome(
            eng,
            ["-c", "import prody; prody.parseMMCIF('c.cif', unite_chains=True)"],
            tmp_path,
            CHAIN,
            {"chain_id_scheme": "auth"},
        )[0].action
        == "allow"
    )


def test_ligand_and_water_labels_do_not_trigger_the_polymer_check(spec, tmp_path):
    # author A holds protein (label A) plus a water whose label differs: only polymer residues count
    rng = random.Random(3)
    protein = make_residues(rng, random_sequence(rng, 10), 1)
    chains = [Chain("A", protein, label="A", entity=1)]
    (tmp_path / "w.cif").write_text(mmcif_text(chains, "w"))
    eng = engine_for(spec)
    code = "import prody; prody.parseMMCIF('w.cif')"
    assert outcome(eng, ["-c", code], tmp_path, CHAIN)[0].action == "allow"


def test_script_files_are_read_and_unreadable_cif_is_unknown(spec, tmp_path):
    write_cif(tmp_path / "c.cif", swapped=True)
    (tmp_path / "go.py").write_text("import prody\nag = prody.parseMMCIF('c.cif')\n")
    eng = engine_for(spec)
    decision, result = outcome(eng, ["go.py", "c.cif"], tmp_path, CHAIN)
    assert result.code == "chain_scheme_undeclared"
    (tmp_path / "bad.cif").write_text("data_x\nloop_\n_atom_site.id\n1 2 3 4\n")
    decision, result = outcome(
        eng, ["-c", "import prody; prody.parseMMCIF('bad.cif')"], tmp_path, CHAIN
    )
    assert result.verdict == "unknown" and decision.action == "warn"


# ---- prody.range_end -------------------------------------------------------------------------


def write_pdb(path: Path, start: int = 10, n: int = 20) -> None:
    path.write_text(pdb_text([Chain("A", residues(n, start))]))  # residues start .. start+n-1


RANGE_CALLS = [
    "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum 12:18')",
    "import prody; ag = prody.parsePDB('p.pdb'); ag.select('chain A and resnum 12:18 and name CA')",
    "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resid 12 : 18')",
    "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum 12:18:2')",
    'import prody; ag = prody.parsePDB("p.pdb"); ag.select("calpha and resnum 12:18")',
]


@pytest.mark.parametrize("code", RANGE_CALLS)
def test_a_colon_range_whose_end_exists_fails_until_declared(spec, tmp_path, code):
    write_pdb(tmp_path / "p.pdb")
    eng = engine_for(spec)
    decision, result = outcome(eng, ["-c", code], tmp_path, RANGE)
    assert decision.action == "block" and result.code == "range_end_undeclared"
    text = render_block(decision)
    assert "residue 18 is left out" in text and " to 18" in text and "p.pdb" in text
    assert outcome(eng, ["-c", code], tmp_path, RANGE, {"range_end": "exclusive"})[0].action == (
        "allow"
    )
    decision, result = outcome(eng, ["-c", code], tmp_path, RANGE, {"range_end": "inclusive"})
    assert decision.action == "block" and result.code == "declared_mismatch"


def test_to_ranges_other_libraries_and_absent_ends_stay_silent(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb", start=10, n=20)  # residues 10..29
    eng = engine_for(spec)
    silent = [
        "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum 12 to 18')",
        "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum 12:40')",  # 40 is absent
        "import MDAnalysis as mda; u = mda.Universe('p.pdb'); u.select_atoms('resid 12:18')",
        "import gemmi; st = gemmi.read_structure('p.pdb'); print(len(st[0]['A']))",
        "import prody; ag = prody.parsePDB('p.pdb'); ag.select('index 3:9')",
        "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum 12 18')",
    ]
    for code in silent:
        decision, result = outcome(eng, ["-c", code], tmp_path, RANGE)
        assert decision.action == "allow", code
        assert result is None or result.verdict == "pass"
    # an awk or grep over the file is outside the interpreters the rule watches
    for tool, argv in (("awk", ["{print $6}", "p.pdb"]), ("grep", ["-c", "CA", "p.pdb"])):
        assert eng.evaluate(Call(tool, tuple(argv), tmp_path)).action == "allow"


def test_negative_numbers_and_unreadable_files(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb", start=-5, n=20)  # residues -5..14
    eng = engine_for(spec)
    code = "import prody; ag = prody.parsePDB('p.pdb'); ag.select('resnum -3:4')"
    assert outcome(eng, ["-c", code], tmp_path, RANGE)[1].code == "range_end_undeclared"
    (tmp_path / "bad.pdb").write_text("not a structure\n")
    code = "import prody; ag = prody.parsePDB('bad.pdb'); ag.select('resnum 1:5')"
    decision, result = outcome(eng, ["-c", code], tmp_path, RANGE)
    assert decision.action in ("allow", "warn")  # an empty parse has no residue 5: nothing to say


def test_the_precheck_sees_bash_wrapped_calls(spec, tmp_path):
    write_pdb(tmp_path / "p.pdb")
    eng = engine_for(spec)
    command = (
        'bash -c "python-struct -c \\"import prody; '
        "prody.parsePDB('p.pdb').select('resnum 12:18')\\\"\""
    )
    result = precheck(command, tmp_path, eng)
    assert result.action == "block" and "BLOCKED" in result.text
    assert "residue 18 is left out" in result.text


# ---- the facts the rules rest on, with the real libraries -------------------------------------


def run_py(tools: ToolEnv, cwd: Path, code: str) -> str:
    proc = tools.run([PY, "-s", "-c", code], cwd)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_prody_names_chains_by_label_and_the_others_by_author(tools, tmp_path):
    write_cif(tmp_path / "c.cif", swapped=True)
    code = (
        "import json, prody, gemmi; from Bio.PDB import MMCIFParser; "
        "ag = prody.parseMMCIF('c.cif'); "
        "out = {'prody': {c.getChid(): c.numResidues() for c in ag.getHierView().iterChains()}, "
        "'prody_pdb': sorted(set(map(str, prody.parsePDB('c.cif').getChids()))), "
        "'united': sorted(set(map(str, prody.parseMMCIF('c.cif', unite_chains=True).getChids()))), "
        "'gemmi': {c.name: len(c) for c in gemmi.read_structure('c.cif')[0]}, "
        "'bio': {c.id: len(c) for c in MMCIFParser(QUIET=True).get_structure('x', 'c.cif')[0]}}; "
        "print(json.dumps(out))"
    )
    out = json.loads(run_py(tools, tmp_path, code))
    # author A has 18 residues, author B has 12; ProDy swaps them
    assert out["gemmi"] == out["bio"] == {"A": 18, "B": 12}
    assert out["prody"] == {"A": 12, "B": 18} and out["prody_pdb"] == ["A", "B"]
    assert out["united"] == ["A", "B"]


def test_prody_colon_range_excludes_the_end_and_to_includes_it(tools, tmp_path):
    write_pdb(tmp_path / "p.pdb", start=10, n=20)
    code = (
        "import json, prody, warnings; warnings.simplefilter('ignore'); import MDAnalysis as mda; "
        "ag = prody.parsePDB('p.pdb'); u = mda.Universe('p.pdb'); "
        "n = lambda s: ag.select(s + ' and name CA').numAtoms(); "
        "print(json.dumps({'to': n('resnum 12 to 18'), 'colon': n('resnum 12:18'), "
        "'resid_colon': n('resid 12:18'), "
        "'mda': u.select_atoms('resid 12:18 and name CA').n_atoms}))"
    )
    assert json.loads(run_py(tools, tmp_path, code)) == {
        "to": 7, "colon": 6, "resid_colon": 6, "mda": 7,
    }  # fmt: skip


# ---- admission -------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", [CHAIN, RANGE])
def test_admission_on_calls_labelled_by_the_real_libraries(spec, tools, tmp_path, rule_id):
    stats = measure(rule_id, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
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
