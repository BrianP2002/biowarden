"""Rules pdbtools.selaltloc_digits and pdbtools.modified_residues: the port of pdb_selaltloc's
grouping logic against the real tool, the checkers on real argv and files, the tool facts they rely
on (run with the real programs), and admission over calls labelled by running them."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Atom,
    Chain,
    make_residues,
    mmcif_text,
    pdb_atom_line,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_pdbtools as sp
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SEL, MOD = "pdbtools.selaltloc_digits", "pdbtools.modified_residues"
PY = "python-struct"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not env.real_bins(["pdb_selaltloc", "python-struct"]).keys() >= {
        "pdb_selaltloc",
        "python-struct",
    }:
        pytest.skip("pdb-tools or python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_pdbtools")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def split_pdb(path: Path, start: int, split: int, whole: bool = True, n: int = 14) -> list[str]:
    """Residues start..start+n-1; residue `split` has altlocs A 0.4 / B 0.6 (every atom or CB)."""
    rng = random.Random(2)
    lines, serial = [], 1
    for res in make_residues(rng, random_sequence(rng, n), start):
        for atom in res.atoms:
            if res.num == split and (whole or atom.name == "CB"):
                for label, occ, dx in (("A", 0.4, 0.0), ("B", 0.6, 0.4)):
                    a = Atom(atom.name, (atom.xyz[0] + dx, *atom.xyz[1:]), occ=occ, altloc=label)
                    lines.append(pdb_atom_line(serial, a, res, "A"))
                    serial += 1
            else:
                lines.append(pdb_atom_line(serial, atom, res, "A"))
                serial += 1
    path.write_text("\n".join([*lines, "TER", "END"]) + "\n")
    return lines


def outcome(eng: Engine, tool: str, argv, cwd: Path, rule: str, declared=None, context=()):
    call = Call(tool, tuple(argv), cwd, declared or {}, tuple(context))
    decision = eng.evaluate(call)
    mine = [o for o in decision.outcomes if o.rule.id == rule]
    return decision, (mine[0].result if mine else None)


def test_spec_is_valid_and_the_checkers_are_registered(spec):
    sel, mod = spec.rules
    assert (sel.id, sel.kind, sel.tier, sel.applies.tools) == (SEL, "call", "T1", ["pdb_selaltloc"])
    assert (mod.id, mod.kind, mod.tier) == (MOD, "fact", "T1")
    assert "pdb_tofasta" in mod.applies.tools and "python-struct" in mod.applies.tools
    assert sel.check in known_checkers() and mod.check in known_checkers()
    assert sel.declare.values == ["verified"] and mod.declare.values == [
        "parent_letter",
        "X",
        "omitted",
    ]
    assert sel.declare.required is False and mod.declare.required is False


def test_checkers_cannot_read_files_below_tier_t1(spec, tmp_path):
    split_pdb(tmp_path / "model.pdb", 1, 9)
    ctx = Ctx(tmp_path, "T0")
    with pytest.raises(TierViolation):
        sp.pdbtools_selaltloc_digits(
            Call("pdb_selaltloc", ("model.pdb",), tmp_path), Match(spec.rules[0]), ctx
        )
    call = Call(PY, ("-c", "from Bio.SeqUtils import seq1; seq1('x'); open('model.pdb')"), tmp_path)
    with pytest.raises(TierViolation):
        sp.pdbtools_modified_residues(call, Match(spec.rules[1], (tmp_path / "model.pdb",)), ctx)


# ---- pdb_selaltloc ---------------------------------------------------------------------------


def test_the_port_of_the_grouping_logic_agrees_with_the_real_tool(tools, tmp_path):
    rng = random.Random(11)
    mismatches = 0
    for _ in range(60):
        start = rng.choice([1, 5, 8, 9, 10, 95, 98, 99, 100, 995])
        path = tmp_path / "t.pdb"
        lines = []
        for res in make_residues(rng, random_sequence(rng, rng.randint(6, 16)), start):
            mode = rng.choice(["plain", "plain", "side", "whole", "mixed"])
            for atom in res.atoms:
                split = (
                    mode == "whole"
                    or (mode == "side" and atom.name == "CB")
                    or (mode == "mixed" and atom.name in ("CB", "O"))
                )
                labels = rng.choice([("A", "B"), ("A", "B", "C"), ("B", "A")]) if split else ("",)
                for label in labels:
                    occ = round(rng.uniform(0.2, 0.7), 2) if split else 1.0
                    a = Atom(atom.name, atom.xyz, occ=occ, altloc=label)
                    lines.append(pdb_atom_line(len(lines) + 1, a, res, "A"))
        path.write_text("\n".join([*lines, "END"]) + "\n")
        real = tools.run(["pdb_selaltloc", "t.pdb"], tmp_path).stdout.splitlines()
        mismatches += real != sp.emulate_selaltloc([*lines, "END"])
    assert mismatches == 0


@pytest.mark.parametrize(("start", "split", "whole", "lost"), [
    (1, 9, True, 5), (1, 10, True, 5), (91, 99, True, 5), (91, 100, True, 5), (1, 9, False, 1),
    (1, 5, True, 0), (1, 12, True, 0), (4, 12, True, 0), (1, 11, False, 0), (1, 8, False, 0),
])  # fmt: skip
def test_atoms_are_lost_exactly_at_digit_boundaries_and_the_rule_agrees(
    tools, spec, tmp_path, start, split, whole, lost
):
    lines = split_pdb(tmp_path / "model.pdb", start, split, whole, n=14)
    expected = sp.distinct_atoms(lines)
    out = tools.run(["bash", "-c", "pdb_selaltloc model.pdb | grep -c '^ATOM'"], tmp_path)
    assert int(out.stdout) == expected - lost, (out.stdout, expected, lost)
    decision, result = outcome(engine_for(spec), "pdb_selaltloc", ["model.pdb"], tmp_path, SEL)
    if lost:
        assert decision.action == "block" and result.code == "selaltloc_drops_atoms"
        text = render_block(decision)
        assert f"{lost} atom(s) fewer" in text and "compares residue numbers as text" in text
        assert "Bio.PDB.PDBParser" in text and "--declare selaltloc_residues=verified" in text
    else:
        assert decision.action == "allow" and result.verdict == "pass"


def test_declaration_options_pipelines_and_other_formats(spec, tmp_path):
    split_pdb(tmp_path / "model.pdb", 1, 9)
    eng = engine_for(spec)
    for argv in (["model.pdb"], ["-A", "model.pdb"], ["-B", "model.pdb"], ["./model.pdb"]):
        assert outcome(eng, "pdb_selaltloc", argv, tmp_path, SEL)[1].code == "selaltloc_drops_atoms"
    decision, result = outcome(
        eng, "pdb_selaltloc", [], tmp_path, SEL, context=("cat", "model.pdb")
    )
    assert result.code == "selaltloc_drops_atoms"  # the file is named in the first stage
    declared = {"selaltloc_residues": "verified"}
    assert (
        outcome(eng, "pdb_selaltloc", ["model.pdb"], tmp_path, SEL, declared)[0].action == "allow"
    )
    assert outcome(eng, "pdb_selaltloc", ["-h"], tmp_path, SEL)[0].action == "allow"
    # an mmCIF is not analysed, an unreadable file only warns, no file named: nothing to say
    (tmp_path / "m.cif").write_text(
        mmcif_text([Chain("A", make_residues(random.Random(1), "MKTAYIA", 1))])
    )
    assert outcome(eng, "pdb_selaltloc", ["m.cif"], tmp_path, SEL)[0].action == "allow"
    (tmp_path / "bad.pdb").write_text("ATOM  short\n")
    decision, result = outcome(eng, "pdb_selaltloc", ["bad.pdb"], tmp_path, SEL)
    assert decision.action in ("allow", "warn")
    assert outcome(eng, "pdb_selaltloc", [], tmp_path, SEL)[0].action == "allow"


def test_the_precheck_checks_the_pdb_selaltloc_stage_of_a_pipeline(spec, tmp_path):
    split_pdb(tmp_path / "model.pdb", 1, 9)
    result = precheck("pdb_selaltloc model.pdb | grep -c '^ATOM'", tmp_path, engine_for(spec))
    assert result.action == "block" and "pdb_selaltloc would write 5 atom(s) fewer" in result.text
    result = precheck("cat model.pdb | pdb_selaltloc | grep -c '^ATOM'", tmp_path, engine_for(spec))
    assert result.action == "block"


# ---- modified residues -----------------------------------------------------------------------


def write_mod(path: Path, modified: bool = True) -> None:
    rng = random.Random(6)
    res = make_residues(rng, "MKTAYMAKQRQSFVKSHFAGLDE", 1)
    if modified:
        res[5].name, res[5].het = "MSE", True
        res[11].name, res[11].het = "SEP", True
    path.write_text(pdb_text([Chain("A", res)]))


X_CODE = [
    "import gemmi; st = gemmi.read_structure('model.pdb'); "
    "print(gemmi.one_letter_code([r.name for r in st[0]['A']]))",
    "from Bio.SeqUtils import seq1; from Bio.PDB import PDBParser; "
    "ch = PDBParser().get_structure('x', 'model.pdb')[0]['A']; "
    "print(seq1(''.join(r.get_resname() for r in ch)))",
]
OMIT_CODE = [
    "from Bio.PDB import PDBParser, PPBuilder; st = PDBParser().get_structure('x', 'model.pdb'); "
    "print([str(p.get_sequence()) for p in PPBuilder().build_peptides(st)])",
]
PARENT_CODE = [
    "import prody; print(prody.parsePDB('model.pdb').select('calpha').getSequence())",
]


@pytest.mark.parametrize("code", X_CODE)
def test_x_writing_extractors_fail_until_declared(spec, tmp_path, code):
    write_mod(tmp_path / "model.pdb")
    eng = engine_for(spec)
    decision, result = outcome(eng, PY, ["-c", code], tmp_path, MOD)
    assert decision.action == "block" and result.code == "modified_residues_undeclared"
    text = render_block(decision)
    assert (
        "(MSE, SEP)" in text and "write" not in text.split("BLOCKED")[0] or "map modified" in text
    )
    assert "writes them as X" in text and "find_tabulated_residue" in text
    assert (
        outcome(eng, PY, ["-c", code], tmp_path, MOD, {"modified_residues": "X"})[0].action
        == "allow"
    )
    decision, result = outcome(
        eng, PY, ["-c", code], tmp_path, MOD, {"modified_residues": "parent_letter"}
    )
    assert (
        result.code == "declared_mismatch"
        and "you declared modified_residues=parent_letter" in (render_block(decision))
    )


def test_pdb_tofasta_and_peptide_builders(spec, tmp_path):
    write_mod(tmp_path / "model.pdb")
    eng = engine_for(spec)
    decision, result = outcome(eng, "pdb_tofasta", ["model.pdb"], tmp_path, MOD)
    assert result.code == "modified_residues_undeclared" and "writes them as X" in render_block(
        decision
    )
    assert (
        outcome(eng, "pdb_tofasta", ["model.pdb"], tmp_path, MOD, {"modified_residues": "X"})[
            0
        ].action
        == "allow"
    )
    decision, result = outcome(eng, PY, ["-c", OMIT_CODE[0]], tmp_path, MOD)
    assert result.code == "modified_residues_undeclared" and "skip hetero residues" in render_block(
        decision
    )
    assert (
        outcome(eng, PY, ["-c", OMIT_CODE[0]], tmp_path, MOD, {"modified_residues": "omitted"})[
            0
        ].action
        == "allow"
    )


def test_parent_letter_extractors_handled_code_and_clean_files_stay_silent(spec, tmp_path):
    write_mod(tmp_path / "model.pdb")
    write_mod(tmp_path / "plain.pdb", modified=False)
    eng = engine_for(spec)
    handled = [
        *PARENT_CODE,
        "import gemmi; st = gemmi.read_structure('model.pdb'); "
        "print(''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() "
        "for r in st[0]['A']))",
        "m = {'MSE': 'M', 'SEP': 'S'}; from Bio.SeqUtils import seq1; seq1('A'); open('model.pdb')",
        "from Bio.Data.IUPACData import protein_letters_3to1_extended as t; seq1('x'); "
        "open('model.pdb')",
        "from Bio.SeqUtils import seq1; seq1('x')  # MSE maps to its parent; open('model.pdb')",
    ]
    for code in handled:
        decision, result = outcome(eng, PY, ["-c", code], tmp_path, MOD)
        assert decision.action == "allow", code
    for code in X_CODE:  # a file without modified residues is unaffected
        decision, result = outcome(
            eng, PY, ["-c", code.replace("model.pdb", "plain.pdb")], tmp_path, MOD
        )
        assert decision.action == "allow"
    for tool, argv in (
        ("head", ["model.pdb"]),
        ("grep", ["HETATM", "model.pdb"]),
        ("awk", ["{print $4}", "model.pdb"]),
    ):
        assert eng.evaluate(Call(tool, tuple(argv), tmp_path)).action == "allow"
    (tmp_path / "bad.pdb").write_text("hello\n")
    decision, result = outcome(
        eng, PY, ["-c", X_CODE[0].replace("model.pdb", "bad.pdb")], tmp_path, MOD
    )
    assert result.verdict == "unknown" and decision.action == "warn"


def test_extractors_disagree_on_modified_residues(tools, tmp_path):
    write_mod(tmp_path / "model.pdb")
    code = (
        "import json, gemmi, prody, warnings; warnings.simplefilter('ignore'); "
        "from Bio.SeqUtils import seq1; from Bio.PDB import PDBParser, PPBuilder; "
        "st = gemmi.read_structure('model.pdb'); names = [r.name for r in st[0]['A']]; "
        "bio = PDBParser(QUIET=True).get_structure('x', 'model.pdb'); "
        "print(json.dumps({'gemmi': gemmi.one_letter_code(names), 'seq1': seq1(''.join(names)), "
        "'ppb': [str(p.get_sequence()) for p in PPBuilder().build_peptides(bio)], "
        "'prody': prody.parsePDB('model.pdb').select('calpha').getSequence()}))"
    )
    proc = tools.run([PY, "-s", "-c", code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["gemmi"] == out["seq1"] and out["gemmi"][5] == "X" and out["gemmi"][11] == "X"
    assert out["prody"][5] == "M" and out["prody"][11] == "S" and len(out["prody"]) == 23
    assert len("".join(out["ppb"])) == 21  # both modified residues are missing from the peptides
    fasta = tools.run(["pdb_tofasta", "model.pdb"], tmp_path).stdout.splitlines()[1]
    assert fasta == out["gemmi"]


@pytest.mark.parametrize("rule_id", [SEL, MOD])
def test_admission_on_calls_labelled_by_running_the_tools(spec, tools, tmp_path, rule_id):
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


def test_family_workspaces_hold_the_documented_states(tools, tmp_path):
    for twin, lost in (("clean", 0), ("faulted", 5)):
        work = tmp_path / twin
        task = build_task("F95", 4, twin, work, tools, "implicit", "free")
        out = tools.run(["bash", "-c", "pdb_selaltloc model.pdb | grep -c '^ATOM'"], work)
        assert int(out.stdout) == task.truth["expected"] - lost
