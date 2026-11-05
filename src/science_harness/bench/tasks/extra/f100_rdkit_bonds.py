"""F100 rdkit_bonds: RDKit's PDB reader makes every bond single unless CONECT records say otherwise.

Source R2-STR-45 (RDKit issue 6154: proximity bonding overrides CONECT and makes single bonds) and
the reader behaviour verified here (RDKit 2024.09.2) on benzoic acid with all hydrogens, exported
once with its CONECT records (a bond of order 2 is a repeated serial, `CONECT 2 3 3 4`) and once
without them, as ligand coordinates from a crystal structure come:

  with CONECT     Chem.MolFromPDBFile -> C7H6O2, MolWt 122.123, 6 aromatic atoms, 1 double bond
  without CONECT  Chem.MolFromPDBFile -> C7H14O2, MolWt 130.187, 0 aromatic atoms, 0 double bonds
                  (removeHs=False gives the same: the explicit hydrogens are kept and the
                  valences are filled up with implicit ones, 15 explicit + 8 implicit H)
Exit 0 and an empty stderr. The same holds for paracetamol (151.165 against 159.229) and the other
ligands of the family. The molecular weight of a file that lists every atom is the sum of the
atomic weights of the listed atoms, whatever the bond orders.

Twins: the same ligand with all its hydrogens, at the same coordinates. Clean: the file carries
CONECT records with the doubled serials for double and aromatic bonds. Faulted: no CONECT records
(coordinates only). The answer is the molecular weight in g/mol (tolerance 0.05).

  explicit  the README says that every atom of the ligand is listed, hydrogens included
  implicit  the README only says "ligand coordinates"
  forced    the task sentence dictates RDKit
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f89_chain_ids import TOOLS_NOTE, python_struct
from science_harness.bench.tasks.extra.f93_tmalign_norm import apply, rotation
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"ligand.pdb", "README.md"})
WEIGHTS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}  # RDKit's average atomic weights
TOL = 0.05

# name -> SMILES, atoms (name, element, x, y, z) with all hydrogens, CONECT serials (RDKit export)
LIGANDS: dict[str, dict] = {
    "benzoic acid": {
        "smiles": "OC(=O)c1ccccc1",
        "atoms": [
            ("O1", "O", -2.511, -1.057, 0.936),
            ("C1", "C", -2.096, -0.126, 0.058),
            ("O2", "O", -2.891, 0.527, -0.591),
            ("C2", "C", -0.619, -0.012, -0.007),
            ("C3", "C", -0.063, 0.925, -0.887),
            ("C4", "C", 1.323, 1.065, -0.98),
            ("C5", "C", 2.157, 0.271, -0.194),
            ("C6", "C", 1.61, -0.665, 0.684),
            ("C7", "C", 0.224, -0.808, 0.78),
            ("H1", "H", -3.488, -1.019, 0.87),
            ("H2", "H", -0.709, 1.549, -1.503),
            ("H3", "H", 1.75, 1.794, -1.664),
            ("H4", "H", 3.237, 0.38, -0.267),
            ("H5", "H", 2.263, -1.283, 1.296),
            ("H6", "H", -0.188, -1.541, 1.469),
        ],
        "conect": [
            [1, 2, 10],
            [2, 3, 3, 4],
            [4, 5, 5, 9],
            [5, 6, 11],
            [6, 7, 7, 12],
            [7, 8, 13],
            [8, 9, 9, 14],
            [9, 15],
        ],
    },
    "paracetamol": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "atoms": [
            ("C1", "C", 3.695, 0.311, -0.141),
            ("C2", "C", 2.26, 0.775, -0.112),
            ("O1", "O", 1.988, 1.97, -0.126),
            ("N1", "N", 1.371, -0.282, -0.04),
            ("C3", "C", -0.037, -0.219, 0.008),
            ("C4", "C", -0.729, -1.436, 0.09),
            ("C5", "C", -2.125, -1.459, 0.141),
            ("C6", "C", -2.833, -0.266, 0.11),
            ("O2", "O", -4.193, -0.331, 0.161),
            ("C7", "C", -2.167, 0.951, 0.028),
            ("C8", "C", -0.768, 0.971, -0.023),
            ("H1", "H", 4.318, 1.084, -0.6),
            ("H2", "H", 4.037, 0.127, 0.881),
            ("H3", "H", 3.799, -0.603, -0.734),
            ("H4", "H", 1.77, -1.212, -0.011),
            ("H5", "H", -0.196, -2.382, 0.115),
            ("H6", "H", -2.657, -2.404, 0.204),
            ("H7", "H", -4.546, 0.574, 0.132),
            ("H8", "H", -2.708, 1.891, 0.003),
            ("H9", "H", -0.279, 1.939, -0.087),
        ],
        "conect": [
            [1, 2, 12, 13, 14],
            [2, 3, 3, 4],
            [4, 5, 15],
            [5, 6, 6, 11],
            [6, 7, 16],
            [7, 8, 8, 17],
            [8, 9, 10],
            [9, 18],
            [10, 11, 11, 19],
            [11, 20],
        ],
    },
    "nicotinamide": {
        "smiles": "NC(=O)c1cccnc1",
        "atoms": [
            ("N1", "N", 2.634, -0.221, -0.468),
            ("C1", "C", 1.683, 0.637, 0.014),
            ("O1", "O", 2.003, 1.719, 0.49),
            ("C2", "C", 0.27, 0.199, -0.02),
            ("C3", "C", -0.659, 0.802, 0.831),
            ("C4", "C", -1.99, 0.392, 0.797),
            ("C5", "C", -2.347, -0.607, -0.095),
            ("N2", "N", -1.478, -1.204, -0.941),
            ("C6", "C", -0.19, -0.779, -0.891),
            ("H1", "H", 2.467, -1.209, -0.595),
            ("H2", "H", 3.594, 0.064, -0.316),
            ("H3", "H", -0.348, 1.587, 1.517),
            ("H4", "H", -2.73, 0.843, 1.449),
            ("H5", "H", -3.372, -0.962, -0.158),
            ("H6", "H", 0.462, -1.26, -1.613),
        ],
        "conect": [
            [1, 2, 10, 11],
            [2, 3, 3, 4],
            [4, 5, 5, 9],
            [5, 6, 12],
            [6, 7, 7, 13],
            [7, 8, 14],
            [8, 9, 9],
            [9, 15],
        ],
    },
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "atoms": [
            ("C1", "C", 3.193, -1.025, 0.417),
            ("C2", "C", 1.972, -0.406, -0.194),
            ("O1", "O", 1.802, -0.257, -1.397),
            ("O2", "O", 1.103, -0.032, 0.824),
            ("C3", "C", -0.087, 0.516, 0.339),
            ("C4", "C", -0.163, 1.914, 0.328),
            ("C5", "C", -1.323, 2.543, -0.122),
            ("C6", "C", -2.407, 1.779, -0.549),
            ("C7", "C", -2.335, 0.383, -0.518),
            ("C8", "C", -1.177, -0.269, -0.063),
            ("C9", "C", -1.214, -1.752, -0.043),
            ("O3", "O", -2.12, -2.436, -0.489),
            ("O4", "O", -0.161, -2.336, 0.554),
            ("H1", "H", 3.869, -1.353, -0.378),
            ("H2", "H", 2.91, -1.898, 1.011),
            ("H3", "H", 3.712, -0.289, 1.036),
            ("H4", "H", 0.682, 2.511, 0.659),
            ("H5", "H", -1.38, 3.629, -0.143),
            ("H6", "H", -3.311, 2.267, -0.905),
            ("H7", "H", -3.195, -0.197, -0.85),
            ("H8", "H", -0.368, -3.292, 0.484),
        ],
        "conect": [
            [1, 2, 14, 15, 16],
            [2, 3, 3, 4],
            [4, 5],
            [5, 6, 6, 10],
            [6, 7, 17],
            [7, 8, 8, 18],
            [8, 9, 19],
            [9, 10, 10, 20],
            [10, 11],
            [11, 12, 12, 13],
            [13, 21],
        ],
    },
    "phenol": {
        "smiles": "Oc1ccccc1",
        "atoms": [
            ("O1", "O", 2.448, -0.419, -0.224),
            ("C1", "C", 1.106, -0.203, -0.107),
            ("C2", "C", 0.606, 0.998, 0.387),
            ("C3", "C", -0.776, 1.174, 0.489),
            ("C4", "C", -1.642, 0.152, 0.098),
            ("C5", "C", -1.13, -1.048, -0.395),
            ("C6", "C", 0.25, -1.226, -0.498),
            ("H1", "H", 2.915, 0.375, 0.084),
            ("H2", "H", 1.268, 1.801, 0.694),
            ("H3", "H", -1.178, 2.108, 0.873),
            ("H4", "H", -2.718, 0.291, 0.178),
            ("H5", "H", -1.803, -1.845, -0.699),
            ("H6", "H", 0.655, -2.158, -0.881),
        ],
        "conect": [
            [1, 2, 8],
            [2, 3, 3, 7],
            [3, 4, 9],
            [4, 5, 5, 10],
            [5, 6, 11],
            [6, 7, 7, 12],
            [7, 13],
        ],
    },
    "imidazole propionate": {
        "smiles": "OC(=O)CCc1cnc[nH]1",
        "atoms": [
            ("O1", "O", 3.093, 0.687, -1.218),
            ("C1", "C", 2.024, 0.461, -0.428),
            ("O2", "O", 1.726, 1.248, 0.456),
            ("C2", "C", 1.311, -0.812, -0.803),
            ("C3", "C", 0.242, -1.204, 0.214),
            ("C4", "C", -1.002, -0.394, 0.092),
            ("C5", "C", -2.247, -0.672, -0.43),
            ("N1", "N", -3.079, 0.411, -0.313),
            ("C6", "C", -2.362, 1.34, 0.276),
            ("N2", "N", -1.099, 0.9, 0.548),
            ("H1", "H", 3.467, 1.535, -0.899),
            ("H2", "H", 2.064, -1.608, -0.836),
            ("H3", "H", 0.88, -0.691, -1.803),
            ("H4", "H", 0.633, -1.126, 1.237),
            ("H5", "H", -0.016, -2.259, 0.059),
            ("H6", "H", -2.588, -1.593, -0.885),
            ("H7", "H", -2.696, 2.338, 0.526),
            ("H8", "H", -0.35, 1.438, 0.97),
        ],
        "conect": [
            [1, 2, 11],
            [2, 3, 3, 4],
            [4, 5, 12, 13],
            [5, 6, 14, 15],
            [6, 7, 7, 10],
            [7, 8, 16],
            [8, 9, 9],
            [9, 10, 17],
            [10, 18],
        ],
    },
}


NAMES = tuple(LIGANDS)


def make_ligand(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f100-{seed}")
    name = NAMES[seed % len(NAMES)]
    ligand = LIGANDS[name]
    rot, shift = rotation(rng), tuple(rng.uniform(-10, 10) for _ in range(3))
    atoms = [
        (n, e, *(round(c, 3) for c in apply(rot, shift, (x, y, z))))
        for n, e, x, y, z in ligand["atoms"]
    ]
    return {"name": name, "atoms": atoms, "conect": ligand["conect"], "smiles": ligand["smiles"]}


def molecular_weight(atoms) -> float:
    return sum(WEIGHTS[e] for _, e, *_ in atoms)


def pdb_text(ligand: dict[str, Any], twin: str) -> str:
    lines = []
    for i, (name, element, x, y, z) in enumerate(ligand["atoms"], 1):
        label = f" {name:<3}" if len(name) < 4 else name
        lines.append(
            f"HETATM{i:5d} {label} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00"
            f"          {element:>2}"
        )
    if twin == "clean":
        lines += ["CONECT" + "".join(f"{n:5d}" for n in row) for row in ligand["conect"]]
    return "\n".join([*lines, "END"]) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    ligand = make_ligand(seed)
    (work / "ligand.pdb").write_text(pdb_text(ligand, twin))
    line = "ligand.pdb: coordinates of one small-molecule ligand (residue LIG), PDB format."
    if tier == "explicit":
        line += " Every atom of the ligand is listed, hydrogens included."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = ("the file has CONECT records; double bonds are repeated serials",)
    else:
        evidence = ("the file has no CONECT records: only atom coordinates",)
    prompt = common.prompt_for(
        route,
        "report the molecular weight in g/mol of the ligand in ligand.pdb (every atom of the "
        "ligand, hydrogens included, is listed in the file)",
        "use RDKit to report the molecular weight in g/mol of the ligand in ligand.pdb (every "
        "atom of the ligand, hydrogens included, is listed in the file)",
        "a JSON number",
    )
    expected = round(molecular_weight(ligand["atoms"]), 3)
    wrong = naive_weight(ligand)  # single bonds only: valences filled up with implicit hydrogens
    truth = {
        "kind": "float",
        "expected": expected,
        "naive": expected if twin == "clean" else wrong,
        "tol": TOL,
    }
    task_id = common.task_id("F100", seed, twin, tier, route)
    return Task(task_id, "F100", "bond_order", twin, seed, prompt, truth, evidence, tier, route)


VALENCE = {"C": 4, "N": 3, "O": 2}


def naive_weight(ligand: dict[str, Any]) -> float:
    """RDKit's weight when every proximity bond is single: valences filled with implicit H."""
    atoms = ligand["atoms"]
    heavy = [i for i, (_, e, *_) in enumerate(atoms) if e != "H"]
    degree = {i: 0 for i in heavy}
    for i in heavy:
        for j in range(len(atoms)):
            if j != i and math.dist(atoms[i][2:], atoms[j][2:]) < _cutoff(atoms[i][1], atoms[j][1]):
                degree[i] += 1
    extra = sum(max(0, VALENCE[atoms[i][1]] - degree[i]) for i in heavy)
    return round(molecular_weight(atoms) + extra * WEIGHTS["H"], 3)


def _cutoff(a: str, b: str) -> float:
    return {"HH": 0.0, "CH": 1.2, "NH": 1.15, "OH": 1.1}.get("".join(sorted((a, b))), 1.7)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    elements = [
        line[76:78].strip()
        for line in (work / "ligand.pdb").read_text().splitlines()
        if line.startswith("HETATM")
    ]
    return _result(round(sum(WEIGHTS[e] for e in elements), 3), "sum of the atomic weights")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = (
        "from rdkit import Chem; from rdkit.Chem import Descriptors; "
        "m = Chem.MolFromPDBFile('ligand.pdb'); print(Descriptors.MolWt(m))"
    )
    proc = python_struct(tools, work, code)
    return _result(round(float(proc.stdout.strip().splitlines()[-1]), 3), "RDKit MolFromPDBFile")


# Exposure: RDKit reads ligand.pdb and a bond-order dependent property is taken from the molecule
# (weight, formula, aromaticity, hybridisation, charges) without a template, SMILES, sanitize=False
# or a sum over the listed atoms. Judged per command.
RDKIT_READ = re.compile(r"MolFromPDB(?:File|Block)|MolFromPDB\b")
PROPERTY = re.compile(
    r"MolWt|ExactMolWt|CalcMolFormula|GetIsAromatic|NumAromatic|GetHybridization|FormalCharge"
    r"|GetNumImplicitHs|TotalNumHs|CalcNumRings|MolToSmiles"
)
SAFE = re.compile(
    r"AssignBondOrdersFromTemplate|MolFromSmiles|sanitize\s*=\s*False|proximityBonding\s*=\s*False"
    r"|GetAtomicNum|GetMass|atomic_weight|\.weight\b"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        "ligand.pdb" in c and RDKIT_READ.search(c) and PROPERTY.search(c) and not SAFE.search(c)
        for c in commands
    )


SPEC = FamilySpec(
    id="F100",
    slot="bond_order",
    source="R2-STR-45",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
