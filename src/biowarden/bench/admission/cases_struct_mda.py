"""Admission cases for mda.protein_selection (family F99: amino-acid residues of a chain with
crystal waters, phosphorylated residues in the faulted twin).

Workspaces are the benchmark's own (implicit tier). Every call is a `python-struct -c` snippet run
with the real libraries (MDAnalysis 2.9.0, mdtraj, ProDy, gemmi, Biopython); the printed residue
count is compared with `task.truth`; labels never come from the rule.

  MDAnalysis `protein`, `protein and name CA` and `backbone and name CA` (wrong on the faulted
  twin), MDAnalysis by residue name (`not resname HOH`, `name CA`, `protein or resname SEP TPO`:
  right),
  mdtraj, ProDy, gemmi and Biopython protein tests (right), a hand-written CA count (right), and a
  printed residue-name summary (inspection).
Declarations: includes_modified for right calls; includes_modified and standard_only for the calls
that use the keyword.
"""

from __future__ import annotations

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_struct_common import Tpl, build_cases, last_int, py
from biowarden.bench.tasks import ToolEnv

MDA = (
    "import warnings; warnings.simplefilter('ignore'); import MDAnalysis as mda; "
    "u = mda.Universe('model.pdb'); "
)
TEMPLATES = [
    py(
        "mda_protein",
        MDA + "print(u.select_atoms('protein').n_residues)",
        "standard_only",
        last_int,
    ),
    py(
        "mda_protein_ca",
        MDA + "print(len(u.select_atoms('protein and name CA')))",
        "standard_only",
        last_int,
    ),
    py(
        "mda_backbone_ca",
        MDA + "print(len(u.select_atoms('backbone and name CA')))",
        "standard_only",
        last_int,
    ),
    py(
        "mda_not_water",
        MDA + "print(u.select_atoms('not resname HOH').n_residues)",
        "includes_modified",
        last_int,
    ),
    py("mda_ca_only", MDA + "print(len(u.select_atoms('name CA')))", "includes_modified", last_int),
    py(
        "mda_protein_or_phospho",
        MDA + "print(u.select_atoms('protein or resname SEP TPO').n_residues)",
        "includes_modified",
        last_int,
    ),
    py(
        "mdtraj_protein",
        "import mdtraj as md; t = md.load_pdb('model.pdb'); "
        "print(len({t.topology.atom(i).residue.index for i in t.topology.select('protein')}))",
        "includes_modified",
        last_int,
    ),
    py(
        "prody_protein",
        "import prody; print(prody.parsePDB('model.pdb').select('protein').numResidues())",
        "includes_modified",
        last_int,
    ),
    py(
        "gemmi_is_amino_acid",
        "import gemmi; st = gemmi.read_structure('model.pdb'); "
        "print(sum(1 for r in st[0]['A'] if gemmi.find_tabulated_residue(r.name) "
        "and gemmi.find_tabulated_residue(r.name).is_amino_acid()))",
        "includes_modified",
        last_int,
    ),
    py(
        "biopython_is_aa",
        "from Bio.PDB import PDBParser; from Bio.PDB.Polypeptide import is_aa; "
        "st = PDBParser(QUIET=True).get_structure('x', 'model.pdb'); "
        "print(sum(is_aa(r) for r in st[0]['A']))",
        "includes_modified",
        last_int,
    ),
    py(
        "text_ca_count",
        "print(sum(1 for l in open('model.pdb') if l[12:16] == ' CA '))",
        "includes_modified",
        last_int,
    ),
    py(
        "mda_resname_summary",
        MDA + "print(sorted(set(u.residues.resnames)), u.atoms.n_residues)",
        None,
        None,
    ),
]


def templates(task) -> list[Tpl]:
    return TEMPLATES


@case_source("mda.protein_selection")
def protein_selection_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F99",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=templates,
        key="protein_selection",
        needed=lambda twin, task: "includes_modified",
        values=("includes_modified", "standard_only"),
    )
