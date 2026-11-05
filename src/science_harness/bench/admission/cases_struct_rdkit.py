"""Admission cases for rdkit.pdb_bond_orders (family F100: molecular weight of a ligand).

Workspaces are the benchmark's own (implicit tier): the clean twin's ligand.pdb has CONECT records
with repeated serials for double and aromatic bonds, the faulted twin's has none. Every call is a
`python-struct -c` snippet run with the real RDKit 2024.09.2 (or gemmi); the printed weight is
compared with `task.truth` (tolerance 0.05); labels never come from the rule.

  RDKit MolFromPDBFile + MolWt (also removeHs=False and AddHs: right on the clean twin, 8-10 g/mol
  too heavy on the faulted twin), RDKit with sanitize=False (explicit atoms only: right), RDKit
  ExactMolWt (monoisotopic: wrong on both twins), RDKit MolWt of a molecule built from a SMILES
  that is not in the workspace (not generated), gemmi element weights and a plain table summed over
  the HETATM records (right), a weight read from the atom list of RDKit (GetMass: right), and the
  atom count of the RDKit molecule (inspection).
Declarations: conect on the clean twin and template on the faulted twin for right calls; for the
RDKit property calls on the faulted twin also conect (what a caller who trusts the file assumes).
"""

from __future__ import annotations

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_struct_common import Tpl, build_cases, last_float, py
from science_harness.bench.tasks import ToolEnv

RD = "from rdkit import Chem; from rdkit.Chem import Descriptors; "
LOAD = "m = Chem.MolFromPDBFile('ligand.pdb'{}); "
KEEP = LOAD.format(", removeHs=False")
TABLE = "w = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999}; "
GEMMI = (
    "import gemmi; st = gemmi.read_structure('ligand.pdb'); "
    "print(sum(gemmi.Element(a.element.name).weight for r in st[0]['A'] for a in r))"
)
TEMPLATES = [
    py("rdkit_molwt", RD + LOAD.format("") + "print(Descriptors.MolWt(m))", "conect", last_float),
    py("rdkit_molwt_keep_hs", RD + KEEP + "print(Descriptors.MolWt(m))", "conect", last_float),
    py(
        "rdkit_molwt_addhs",
        RD + LOAD.format("") + "m = Chem.AddHs(m); print(Descriptors.MolWt(m))",
        "conect",
        last_float,
    ),
    py(
        "rdkit_exact_molwt",
        RD + LOAD.format("") + "print(Descriptors.ExactMolWt(m))",
        "conect",
        last_float,
    ),
    py(
        "rdkit_unsanitized",
        RD
        + LOAD.format(", sanitize=False, removeHs=False")
        + "print(sum(a.GetMass() for a in m.GetAtoms()))",
        None,
        last_float,
    ),
    py(
        "rdkit_atom_masses",
        RD + KEEP + "print(sum(a.GetMass() for a in m.GetAtoms()))",
        None,
        last_float,
    ),
    py("gemmi_weights", GEMMI, None, last_float),
    py(
        "python_table",
        TABLE + "print(sum(w[l[76:78].strip()] for l in open('ligand.pdb') if l[:6] == 'HETATM'))",
        None,
        last_float,
    ),
    py("rdkit_atom_count", RD + KEEP + "print(m.GetNumAtoms())", None, None),
]


def templates(task) -> list[Tpl]:
    return TEMPLATES


@case_source("rdkit.pdb_bond_orders")
def bond_orders_cases(seed: int, scratch, tools: ToolEnv) -> list[Case]:
    return build_cases(
        family="F100",
        seed=seed,
        scratch=scratch,
        tools=tools,
        templates=templates,
        key="bond_orders",
        needed=lambda twin, task: "conect" if twin == "clean" else "template",
        values=("conect", "template", "single_bonds"),
    )
