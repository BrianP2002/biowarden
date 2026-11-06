"""Family-specific tests for the `struct` pack (F89-F99): the trap size on seeds 0-29 from the
recorded answers, the twins and tiers, exposure on realistic commands, the real tools against the
recorded naive answer, and byte reproducibility across processes.

The generic gates (reference passes, naive falls into the trap on seeds 0-4, forced route, truth out
of the workspace, locus) live in test_extra_families.py. Skipped when the tools are missing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.reference import solve

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [f"F{n}" for n in range(89, 101)]
SEEDS = range(30)
WORK = Path("/nonexistent-workspace")  # the exposure checks read only the command text
NEEDS = {"python-struct", "TMalign", "mmseqs", "mkdssp", "pdb_selaltloc", "pdb_tofasta"}
FLOAT_TOL = {"F90": 0.05, "F92": 0.05, "F93": 0.02, "F100": 0.05}


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not set(env.real_bins(sorted(NEEDS))) >= NEEDS:
        pytest.skip("the extended structure environment is not available")
    return env


SPECS = load()


def built(tmp_path, family, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


# ---- registration and design ---------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_is_registered_with_a_locus_a_source_and_flat_twin_files(family):
    spec = SPECS[family]
    assert spec.locus in ("tool_default", "data") and spec.source and spec.slot
    assert spec.twin_files and all("/" not in name for name in spec.twin_files)
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")


def test_at_least_three_library_families_and_three_command_line_families():
    library = {
        "F89",
        "F90",
        "F91",
        "F92",
        "F99",
        "F100",
    }  # the trap is in a python-struct library call
    cli = {"F93", "F94", "F95", "F96", "F98"}  # the trap is in a command-line tool
    assert library | cli <= set(SPECS) and len(library) >= 3 and len(cli) >= 3


@pytest.mark.parametrize("family", FAMILIES)
def test_trap_size_from_the_recorded_answers_over_30_seeds_and_both_tiers(tmp_path, tools, family):
    spec = SPECS[family]
    tol = FLOAT_TOL.get(family)
    for tier in spec.tiers:
        for seed in SEEDS:
            _, clean = built(tmp_path, family, seed, "clean", tools, tier)
            _, faulted = built(tmp_path, family, seed, "faulted", tools, tier)
            as_naive = {**faulted.truth, "expected": faulted.truth["naive"]}
            assert answer_matches(clean.truth, clean.truth["naive"]), (family, seed, "clean")
            assert not answer_matches(faulted.truth, faulted.truth["naive"]), (family, seed, tier)
            assert answer_matches(as_naive, faulted.truth["naive"])
            if tol is not None:  # a float answer: the wrong value is far outside the tolerance
                assert abs(faulted.truth["naive"] - faulted.truth["expected"]) > 3 * tol


@pytest.mark.parametrize("family", FAMILIES)
def test_twins_differ_only_in_twin_files_and_tiers_only_in_the_readme(tmp_path, tools, family):
    spec = SPECS[family]
    for seed in (0, 7, 19):
        work_c, clean = built(tmp_path, family, seed, "clean", tools, "implicit")
        work_f, faulted = built(tmp_path, family, seed, "faulted", tools, "implicit")
        assert (
            clean.prompt == faulted.prompt and clean.truth["expected"] == faulted.truth["expected"]
        )
        names = {p.name for p in work_c.iterdir()} | {p.name for p in work_f.iterdir()}
        changed = {
            n
            for n in names
            if not (work_c / n).exists()
            or not (work_f / n).exists()
            or (work_c / n).read_bytes() != (work_f / n).read_bytes()
        }
        assert changed and changed <= spec.twin_files, (family, seed, changed)
        assert all((work_f / n).is_file() for n in names if (work_f / n).exists())  # no subdirs
        explicit, _ = built(tmp_path, family, seed, "faulted", tools, "explicit")
        diff = {
            n
            for n in names
            if (work_f / n).exists() and (explicit / n).exists()
            and (work_f / n).read_bytes() != (explicit / n).read_bytes()
        }  # fmt: skip
        assert diff <= {"README.md"}, (family, diff)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_prompt_names_the_files_and_the_forced_route_names_the_tool(tmp_path, tools, family):
    work, free = built(tmp_path, family, 3, "faulted", tools, "explicit", "free")
    _, forced = built(tmp_path, family, 3, "faulted", tools, "explicit", "forced")
    files = {p.name for p in work.iterdir()}
    mentioned = [n for n in files if n in free.prompt]
    assert mentioned, (family, free.prompt)
    assert forced.prompt != free.prompt and forced.truth == free.truth
    assert free.task_id.startswith(f"{family}-s03-faulted")


# ---- the real tools against the recorded naive answer --------------------------------------


@pytest.mark.parametrize(
    ("family", "seeds"),
    [("F93", range(30)), ("F95", range(30)), ("F96", range(30)), ("F97", range(30)),
     ("F98", range(30)), ("F89", range(6)), ("F90", range(6)), ("F91", range(6)),
     ("F92", range(6)), ("F99", range(4)), ("F100", range(6))],
)  # fmt: skip
def test_the_naive_route_really_produces_the_recorded_trap_answer_silently(
    tmp_path, tools, family, seeds
):
    for seed in seeds:
        work, task = built(tmp_path, family, seed, "faulted", tools)
        naive = solve(task, work, tools, reference=False)
        recorded = {**task.truth, "expected": task.truth["naive"]}
        assert answer_matches(recorded, naive["answer"]), (family, seed, naive["answer"])
        assert not answer_matches(task.truth, naive["answer"])
        assert naive["status"] == "ok" and naive["flags"] == []


def run(tools, cwd, argv, **kw):
    proc = tools.run(argv, cwd, **kw)
    return proc


def test_the_traps_are_silent_exit_zero_and_no_warning_on_stderr(tmp_path, tools):
    work, task = built(tmp_path, "F93", 1, "faulted", tools)
    proc = run(tools, work, ["TMalign", "model.pdb", "reference.pdb"])
    assert proc.returncode == 0 and proc.stderr == "" and proc.stdout.count("TM-score=") == 2
    work, task = built(tmp_path, "F95", 1, "faulted", tools)
    proc = run(tools, work, ["pdb_selaltloc", "model.pdb"])
    assert proc.returncode == 0 and proc.stderr == ""
    work, task = built(tmp_path, "F96", 1, "faulted", tools)
    proc = run(tools, work, ["pdb_tofasta", "model.pdb"])
    assert proc.returncode == 0 and proc.stderr == "" and "X" in proc.stdout
    work, task = built(tmp_path, "F98", 1, "faulted", tools)
    proc = run(tools, work, ["mkdssp", "model.cif", "o.dssp"])
    assert proc.returncode == 0 and proc.stderr == ""
    assert (work / "o.dssp").read_text().splitlines()[6].split()[0] == "0"
    work, task = built(tmp_path, "F94", 1, "faulted", tools)
    from biowarden.bench.tasks.extra import f94_mmseqs_coverage as f94

    proc = run(
        tools, work, ["mmseqs", "easy-cluster", "seqs.fasta", "clu", "tmp", "--min-seq-id", "0.9",
                      "--threads", "1", "-v", "1"],
    )  # fmt: skip
    clusters = len({r.split("\t")[0] for r in (work / "clu_cluster.tsv").read_text().splitlines()})
    assert proc.returncode == 0 and proc.stderr == "" and clusters == task.truth["naive"]
    del f94


def test_the_clean_twins_of_the_cli_families_give_the_right_answer_on_the_naive_route(
    tmp_path, tools
):
    for family in ("F93", "F95", "F96", "F97", "F98"):
        work, task = built(tmp_path, family, 2, "clean", tools)
        naive = solve(task, work, tools, reference=False)
        assert answer_matches(task.truth, naive["answer"]), family


# ---- exposure ------------------------------------------------------------------------------

EXPOSED = {
    "F89": [
        "python-struct -c \"import prody; ag = prody.parseMMCIF('complex.cif'); print(ag)\"",
        "python-struct -c \"import prody; prody.parsePDB('complex.cif')\"",
        'python-struct -c "from Bio.PDB.MMCIF2Dict import MMCIF2Dict as M; '
        "M('complex.cif')['_atom_site.label_asym_id']\"",
        "python-struct -c \"import prody; ag = prody.parseMMCIF('complex.cif'); "
        'print(ag.getHierView().numChains())"',
    ],
    "F90": [
        "python-struct -c \"import prody; ag = prody.parsePDB('model.pdb'); ag.select('name CA')\"",
        "python-struct -c \"import gemmi; s = gemmi.read_structure('model.pdb'); "
        's.remove_alternative_conformations()"',
        "awk '$3==\"CA\" {print $7, $8, $9}' model.pdb",
        "grep ' CA ' model.pdb",
    ],
    "F91": [
        "python-struct -c \"import prody; ag = prody.parsePDB('protein.pdb'); "
        "ag.select('resnum 15:24 and name CA')\"",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb').select('resid 15:24')\"",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb').select('chain A and "
        "resnum 3:9 and calpha')\"",
        "python-struct -c \"import prody; a, b = 15, 24; prody.parsePDB('protein.pdb')"
        ".select('resnum %d:%d' % (a, b))\"",
        "python-struct -c \"import prody; a, b = 15, 24; prody.parsePDB('protein.pdb')"
        ".select(f'resnum {a}:{b + 1} and name CA')\"",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb')"
        ".select('resnum {}:{}'.format(a, b))\"",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb').select('resnum $a:$b')\"",
    ],
    "F92": [
        "python-struct -c \"import mdtraj as md; t = md.load_pdb('protein.pdb'); "
        "t.topology.select('resid 12 and name CA')\"",
        "python-struct -c \"import mdtraj as md; md.load('protein.pdb').topology.residue(12)\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        "t.topology.select('resid 5 or resid 9')\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        "t.topology.select('resid %d and name CA' % p)\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        "t.topology.select(f'resid {p} and name CA')\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        "t.topology.select('resid {}'.format(p))\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        'print(t.topology.residue(p).name)"',
        "python-struct -c \"import mdtraj as md; md.load('protein.pdb').topology.residue(p - 1)\"",
    ],
    "F93": [
        "TMalign model.pdb reference.pdb",
        "TMalign model.pdb reference.pdb | grep -m1 TM-score",
        "TMalign ./model.pdb ./reference.pdb -fast | head -20",
        "TMalign model.pdb reference.pdb | awk '/TM-score/ {print $2; exit}'",
        "TMalign model.pdb reference.pdb -a T",
        "TMalign model.pdb reference.pdb -a T | grep -m1 TM-score",
    ],
    "F94": [
        "mmseqs easy-cluster seqs.fasta clu tmp --min-seq-id 0.9",
        "mmseqs easy-cluster seqs.fasta clu tmp --min-seq-id 0.9 -c 0.3",
        "mmseqs easy-linclust seqs.fasta clu tmp --min-seq-id 0.9",
        "cd /w && mmseqs easy-cluster seqs.fasta out tmp --min-seq-id 0.9 --threads 1",
    ],
    "F95": [
        "pdb_selaltloc model.pdb | grep -c '^ATOM'",
        "cat model.pdb | pdb_selaltloc | wc -l",
        "pdb_selaltloc -A model.pdb > clean.pdb",
        "pdb_selaltloc model.pdb | pdb_tofasta",
    ],
    "F96": [
        "pdb_tofasta model.pdb",
        "python-struct -c \"import gemmi; s = gemmi.read_structure('model.pdb'); "
        "print(gemmi.one_letter_code([r.name for r in s[0]['A']]))\"",
        'python-struct -c "from Bio.PDB import PDBParser, PPBuilder; '
        "PPBuilder().build_peptides(PDBParser().get_structure('x', 'model.pdb'))\"",
        "python-struct -c \"from Bio.SeqUtils import seq1; seq1('ALAGLY')  # model.pdb\"",
        "python-struct -c \"d = {'ALA': 'A', 'GLY': 'G'}; "
        "print(''.join(d.get(l[17:20], 'X') for l in open('model.pdb') if ' CA ' in l))\"",
        "python-struct -c \"aa = {'ALA':'A','CYS':'C'}; "
        "[aa[l[17:20]] for l in open('model.pdb') if l[12:16] == ' CA ']\"",
        "python-struct -c \"print(''.join(t.get(l[17:20].strip(), 'X') for l in "
        "open('model.pdb') if l[12:16] == ' CA '))\"",
        "python-struct -c \"print(''.join(three.get(r.name, 'X') for r in "
        "gemmi.read_structure('model.pdb')[0]['A']))\"",
    ],
    "F97": [
        "awk '$3==\"CA\" && substr($0,61,6)>70' predicted.pdb | wc -l",
        "python-struct -c \"print(sum(float(l[60:66]) > 70 for l in open('predicted.pdb') "
        "if l[12:16] == ' CA '))\"",
        "python-struct -c \"import prody; prody.parsePDB('predicted.pdb').select('beta > 70')\"",
        "python-struct -c \"import prody; ag = prody.parsePDB('predicted.pdb'); "
        '(ag.getBetas() >= 90).sum()"',
        "awk '$3==\"CA\" && $(NF-1)>70' predicted.pdb | wc -l",
        "awk '$3==\"CA\" && $(NF - 1) > 70 {n++} END{print n}' predicted.pdb",
        "python-struct -c \"import gemmi; s = gemmi.read_structure('predicted.pdb'); "
        "print(sum(a.b_iso > 70 for r in s[0]['A'] for a in r if a.name == 'CA'))\"",
    ],
    "F98": [
        "mkdssp model.cif out.dssp",
        "mkdssp --output-format dssp model.cif out.txt",
        "mkdssp model.cif out.dssp && grep -c ' H ' out.dssp",
        'python-struct -c "from Bio.PDB import MMCIFParser, DSSP; '
        "DSSP(MMCIFParser().get_structure('x', 'model.cif')[0], 'model.cif')\"",
    ],
    "F99": [
        "python-struct -c \"import MDAnalysis as mda; u = mda.Universe('model.pdb'); "
        "print(u.select_atoms('protein').n_residues)\"",
        "python-struct -c \"import MDAnalysis as mda; u = mda.Universe('model.pdb'); "
        "print(len(u.select_atoms('backbone and name CA')))\"",
        "python-struct -c \"import MDAnalysis as mda; mda.Universe('model.pdb')"
        ".select_atoms('protein and not name H*')\"",
        "grep '^ATOM' model.pdb | awk '$3==\"CA\"' | wc -l",
        'awk \'$1=="ATOM" && $3=="CA" {n++} END{print n}\' model.pdb',
        "awk '/^ATOM/ && $3 == \"CA\" {n++} END {print n}' model.pdb",
        "grep '^ATOM' model.pdb | cut -c18-26 | sort -u | wc -l",
        "python-struct -c \"print(sum(1 for l in open('model.pdb') if l.startswith('ATOM') "
        "and l[12:16] == ' CA '))\"",
    ],
    "F100": [
        'python-struct -c "from rdkit import Chem; from rdkit.Chem import Descriptors; '
        "print(Descriptors.MolWt(Chem.MolFromPDBFile('ligand.pdb')))\"",
        'python-struct -c "from rdkit import Chem; from rdkit.Chem import rdMolDescriptors as D; '
        "print(D.CalcMolFormula(Chem.MolFromPDBFile('ligand.pdb', removeHs=False)))\"",
        'python-struct -c "from rdkit import Chem; '
        "print(Chem.MolToSmiles(Chem.MolFromPDBFile('ligand.pdb')))\"",
    ],
}
NOT_EXPOSED = {
    "F89": [
        "python-struct -c \"import prody; prody.parseMMCIF('complex.cif', unite_chains=True)\"",
        "python-struct -c \"import gemmi; s = gemmi.read_structure('complex.cif'); "
        'print([c.name for c in s[0]])"',
        "python-struct -c \"from Bio.PDB import MMCIFParser; MMCIFParser().get_structure('x', "
        "'complex.cif')\"",
        "python-struct -c \"d = MMCIF2Dict('complex.cif'); d['_atom_site.auth_asym_id']\"",
        "head -30 complex.cif",
    ],
    "F90": [
        'python-struct -c "from Bio.PDB import PDBParser; '
        "PDBParser().get_structure('x', 'model.pdb')\"",
        "python-struct -c \"import prody; prody.parsePDB('model.pdb', altloc='B')\"",
        "awk '$3==\"CA\" && substr($0,55,6) > 0.5' model.pdb",
        "python-struct -c \"import gemmi; max(a.occ for a in s[0]['A'][5]); "
        "read_structure('model.pdb')\"",
        "head -20 model.pdb",
        "wc -l model.pdb",
    ],
    "F91": [
        'python-struct -c "import prody; '
        "prody.parsePDB('protein.pdb').select('resnum 15 to 24')\"",
        "python-struct -c \"import MDAnalysis as mda; mda.Universe('protein.pdb')"
        ".select_atoms('resid 15:24')\"",
        "python-struct -c \"import gemmi; s = gemmi.read_structure('protein.pdb'); "
        "print([r.name for r in s[0]['A'] if 15 <= r.seqid.num <= 24])\"",
        "awk '$6>=15 && $6<=24' protein.pdb",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb')"
        ".select(f'resnum {a} to {b}')\"",
        "python-struct -c \"import MDAnalysis as mda; mda.Universe('protein.pdb')"
        ".select_atoms(f'resid {a}:{b}')\"",
    ],
    "F92": [
        "python-struct -c \"import mdtraj as md; md.load('protein.pdb').topology.select("
        "'resSeq 12 and name CA')\"",
        "python-struct -c \"import gemmi; gemmi.read_structure('protein.pdb')\"",
        "python-struct -c \"import prody; prody.parsePDB('protein.pdb').select('resnum 12')\"",
        "head protein.pdb",
        "python-struct -c \"import mdtraj as md; md.load('protein.pdb').topology.select("
        "f'resSeq {p} and name CA')\"",
        "python-struct -c \"import mdtraj as md; t = md.load('protein.pdb'); "
        'print([r for r in t.topology.residues if r.resSeq == p])"',
    ],
    "F93": [
        "TMalign reference.pdb model.pdb",
        "TMalign model.pdb reference.pdb -a T | sed -n 2p",
        "TMalign model.pdb reference.pdb -u 54 | grep user-specified",
        "TMalign model.pdb reference.pdb | sed -n 2p",
        "TMalign model.pdb reference.pdb | grep 'Structure_2'",
        "ls -la; cat README.md",
    ],
    "F94": [
        "mmseqs easy-cluster seqs.fasta clu tmp --min-seq-id 0.9 --cov-mode 1",
        "mmseqs easy-cluster seqs.fasta clu tmp --cov-mode 1 -c 0.5 --min-seq-id 0.9",
        "mmseqs easy-search seqs.fasta seqs.fasta out.m8 tmp",
        "grep -c '>' seqs.fasta",
    ],
    "F95": [
        "python-struct -c \"import gemmi; gemmi.read_structure('model.pdb')\"",
        "awk '$1==\"ATOM\"' model.pdb | wc -l",
        "pdb_tofasta model.pdb",
        "ls -la; cat README.md",
    ],
    "F96": [
        "python-struct -c \"import prody; print(prody.parsePDB('model.pdb').select('calpha')"
        '.getSequence())"',
        "python-struct -c \"import gemmi; names = [r.name for r in s[0]['A']]; "
        "m = {'MSE': 'M'}; one_letter_code(names)  # model.pdb\"",
        "grep HETATM model.pdb | head",
        "head -5 model.pdb",
        "python-struct -c \"d = {'ALA': 'A', 'MSE': 'M'}; "
        "print(''.join(d.get(l[17:20], 'X') for l in open('model.pdb') if ' CA ' in l))\"",
    ],
    "F97": [
        "awk '$3==\"CA\" {print substr($0,61,6)}' predicted.pdb | sort -n | tail -1",
        "awk '$3==\"CA\" && substr($0,61,6)*100>70' predicted.pdb | wc -l",
        "python-struct -c \"b = [float(l[60:66]) for l in open('predicted.pdb') "
        "if l[12:16] == ' CA ']; s = 100 if max(b) <= 1 else 1; "
        'print(sum(x * s > 70 for x in b))"',
        "head predicted.pdb",
        "awk '$3==\"CA\" && $(NF-1)*100>70' predicted.pdb | wc -l",
        "awk '$3==\"CA\" {print $(NF-1)}' predicted.pdb | sort -n | tail -1",
    ],
    "F98": [
        "mkdssp model.pdb out.dssp",
        "mkdssp --version",
        'python-struct -c "import gemmi; '
        "gemmi.read_structure('model.cif').write_pdb('model.pdb')\"",
        "head -20 model.cif",
    ],
    "F99": [
        "python-struct -c \"import MDAnalysis as mda; u = mda.Universe('model.pdb'); "
        "print(u.select_atoms('not resname HOH').n_residues)\"",
        "python-struct -c \"import MDAnalysis as mda; u = mda.Universe('model.pdb'); "
        "print(len(u.select_atoms('name CA')))\"",
        'python-struct -c "import mdtraj as md; '
        "md.load_pdb('model.pdb').topology.select('protein')\"",
        "python-struct -c \"import prody; prody.parsePDB('model.pdb').select('protein')\"",
        "python-struct -c \"import MDAnalysis as mda; mda.Universe('model.pdb')"
        ".select_atoms('protein or resname SEP TPO')\"",
        "grep ' CA ' model.pdb | wc -l",
        "grep '^ATOM' model.pdb | head -3",
        "grep -c '^ATOM' model.pdb",
        "awk '$1==\"ATOM\"' model.pdb | wc -l",
        "grep -E '^(ATOM|HETATM)' model.pdb | awk '$3==\"CA\"' | wc -l",
    ],
    "F100": [
        "python-struct -c \"from rdkit import Chem; m = Chem.MolFromPDBFile('ligand.pdb', "
        'sanitize=False, removeHs=False); print(sum(a.GetMass() for a in m.GetAtoms()))"',
        "python-struct -c \"import gemmi; st = gemmi.read_structure('ligand.pdb'); "
        "print(sum(gemmi.Element(a.element.name).weight for r in st[0]['A'] for a in r))\"",
        'python-struct -c "from rdkit import Chem; from rdkit.Chem import AllChem; '
        "AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles('Oc1ccccc1'), "
        "Chem.MolFromPDBFile('ligand.pdb'))\"",
        'python-struct -c "from rdkit import Chem; '
        "print(Chem.MolFromPDBFile('ligand.pdb').GetNumAtoms())\"",
        "head -5 ligand.pdb",
    ],
}


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_is_true_on_the_trapped_route_per_command(family):
    for command in EXPOSED[family]:
        assert SPECS[family].exposed([command], WORK) is True, command
    safe = NOT_EXPOSED[family][0]
    # one trapped command among safe ones is enough: the session executed the trap
    assert SPECS[family].exposed([safe, EXPOSED[family][0]], WORK) is True


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_is_false_on_safe_routes_and_unrelated_commands(family):
    for command in NOT_EXPOSED[family]:
        assert SPECS[family].exposed([command], WORK) is False, command
    assert SPECS[family].exposed(["ls -la; cat README.md"], WORK) is False
    assert SPECS[family].exposed([], WORK) is False


# ---- fixes of the final review: F89-F94 (FD-3, FD-10, FD-11, FD-12) ------------------------


def readme_of(tmp_path, family, seed, twin, tools, tier="implicit"):
    work, _ = built(tmp_path, family, seed, twin, tools, tier)
    return (work / "README.md").read_text()


def test_f89_implicit_readme_names_the_authors_chain_letters_but_no_mmcif_field(tmp_path, tools):
    for twin in ("clean", "faulted"):
        implicit = readme_of(tmp_path, "F89", 3, twin, tools)
        assert "the chain letters the authors use in the paper" in implicit
        assert "auth_asym_id" not in implicit and "label_asym_id" not in implicit
        assert "(auth_asym_id)" in readme_of(tmp_path, "F89", 3, twin, tools, "explicit")


def test_f90_no_altloc_residue_sits_at_a_pdb_selaltloc_digit_boundary():
    from biowarden.bench.tasks.extra import f90_altloc as f90

    for seed in [*range(300), *range(1000, 1010), *range(1020, 1025)]:
        model = f90.make_model(seed)
        assert not set(model["shift"]) & {9, 10, 99, 100}, (seed, sorted(model["shift"]))
        assert {model["p"], model["q"]} <= set(model["shift"])  # the two query residues


def test_f90_pdb_selaltloc_keeps_every_atom_of_the_models_it_is_listed_safe_on(tmp_path, tools):
    def atoms(text):
        return {(line[22:26], line[12:16]) for line in text.splitlines() if line[:4] == "ATOM"}

    for seed in [*range(40), *range(1000, 1005), *range(1020, 1025)]:
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F90", seed, twin, tools)
            out = run(tools, work, ["pdb_selaltloc", "model.pdb"]).stdout
            listed = atoms((work / "model.pdb").read_text())
            kept = [line for line in out.splitlines() if line[:4] == "ATOM"]
            assert len(kept) == len(listed) == len(atoms(out)), (seed, twin)


def test_f91_the_answer_varies_over_two_to_four_and_the_naive_drops_the_last_residue(
    tmp_path, tools
):
    answers = set()
    for seed in range(30):
        _, clean = built(tmp_path, "F91", seed, "clean", tools)
        _, faulted = built(tmp_path, "F91", seed, "faulted", tools)
        expected = clean.truth["expected"]
        assert expected == faulted.truth["expected"] and 2 <= expected <= 4, seed
        assert clean.truth["naive"] == expected and faulted.truth["naive"] == expected - 1
        answers.add(expected)
    assert answers == {2, 3, 4}


def test_f92_readme_states_the_unit_in_both_tiers_and_twins(tmp_path, tools):
    for tier in ("implicit", "explicit"):
        for twin in ("clean", "faulted"):
            text = readme_of(tmp_path, "F92", 2, twin, tools, tier)
            assert "Coordinates are in angstroms, and so is the distance to report." in text


def test_f94_ids_are_neutral_and_only_the_header_marks_a_fragment(tmp_path, tools):
    for seed in (0, 1, 7, 1000, 1004):
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F94", seed, twin, tools)
            heads = [x[1:] for x in (work / "seqs.fasta").read_text().splitlines() if x[0] == ">"]
            assert all(re.fullmatch(r"prot\d\d( partial sequence)?", h) for h in heads)
            # numbered in file order, 1..N, with no gap and no name for a family or a member
            assert [h.split()[0] for h in heads] == [f"prot{i + 1:02d}" for i in range(len(heads))]
            partial = sum(h.endswith("partial sequence") for h in heads)
            assert partial == task.truth["naive"] - task.truth["expected"], (seed, twin)


# ---- reproducibility -----------------------------------------------------------------------

SNIPPET = r"""
import hashlib, json, sys, tempfile
from pathlib import Path
from biowarden.bench.tasks import ToolEnv, build_task
tools = ToolEnv.discover(Path(sys.argv[1]))
out = {}
for family in sys.argv[2:]:
    for twin in ("clean", "faulted"):
        work = Path(tempfile.mkdtemp())
        task = build_task(family, 5, twin, work, tools, "implicit", "free")
        digest = hashlib.sha256()
        for p in sorted(work.iterdir()):
            digest.update(p.name.encode() + p.read_bytes())
        out[f"{family}-{twin}"] = [digest.hexdigest(), json.dumps(task.truth, sort_keys=True)]
print(json.dumps(out, sort_keys=True))
"""


def test_builds_are_byte_reproducible_across_processes_and_hash_seeds(tools):
    results = []
    for hash_seed in ("0", "12345"):
        env = {"PATH": tools.path_var(), "PYTHONHASHSEED": hash_seed, "LC_ALL": "C"}
        proc = subprocess.run(
            [sys.executable, "-c", SNIPPET, str(ROOT), *FAMILIES],
            capture_output=True, text=True, env=env, check=False, cwd=ROOT,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr[-800:]
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1] and len(results[0]) == 2 * len(FAMILIES)
    assert hashlib.sha256(json.dumps(results[0]).encode()).hexdigest()
