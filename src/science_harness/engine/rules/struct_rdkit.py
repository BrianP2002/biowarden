"""Rule rdkit.pdb_bond_orders (slot bond_order, family F100, incident R2-STR-45): RDKit's PDB
reader makes every bond single unless CONECT records say otherwise.

The trap (RDKit issue 6154: proximity bonding overrides CONECT and creates single bonds).
Verified here (RDKit 2024.09.2) on ligands exported with all hydrogens, once with their CONECT
records (a bond of order 2 is a repeated serial, `CONECT 2 3 3 4`) and once without:

  with CONECT     Chem.MolFromPDBFile -> benzoic acid C7H6O2, MolWt 122.123, 6 aromatic atoms
  without CONECT  Chem.MolFromPDBFile -> C7H14O2, MolWt 130.187, 0 aromatic atoms, 0 double bonds
  (removeHs=False gives the same; paracetamol: 151.165 with CONECT, 159.229 without)
Exit 0 and an empty stderr. Crystallographic ligand coordinates normally come without bond orders.

What the rule does. A FACT rule over .pdb/.ent files named in a command run by python-struct,
python-bio, python3 or python. It looks at the code first: RDKit's `MolFromPDBFile` /
`MolFromPDBBlock` together with a property that depends on bond orders (`MolWt`, `ExactMolWt`,
`CalcMolFormula`, aromaticity, hybridisation, `FormalCharge`, `MolToSmiles`, ring counts). Code that
assigns orders (`AssignBondOrdersFromTemplate`), starts from a SMILES, reads with
`sanitize=False` / `proximityBonding=False`, or sums atomic numbers / masses / weights itself is
taken to have handled the question. Then at the data: the non-water HETATM atoms of the file. The
state is risky when the file has no CONECT record that repeats a serial (no bond order is written)
and some bond between two heavy ligand atoms is shorter than a single bond can be (C-C below 1.46,
C-N below 1.41, C-O below 1.34, N-N below 1.34, N-O below 1.33 angstroms): a saturated ligand costs
nothing.
  risky data, no declaration                                    -> fail (bond_orders_undeclared)
  declared conect (orders come from CONECT) but there are none  -> fail (declared_mismatch)
  declared template without template code                       -> fail (declared_mismatch)
  declared single_bonds, or declared template with template code -> pass
Declaration `bond_orders` (optional): conect | template | single_bonds. `unknown`: unreadable file.

Not seen: ligands read from mmCIF, SDF/MOL2 files (they carry orders), molecules built in an
earlier command, and the Open Babel / PyMOL readers.
"""

from __future__ import annotations

import math
import re

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.struct_common import command_text, file_text

READ = re.compile(r"MolFromPDB(?:File|Block)\b")
PROPERTY = re.compile(
    r"MolWt|ExactMolWt|CalcMolFormula|GetIsAromatic|NumAromatic|GetHybridization|FormalCharge"
    r"|GetNumImplicitHs|TotalNumHs|CalcNumRings|MolToSmiles|GetBondType"
)
TEMPLATE = re.compile(r"AssignBondOrdersFromTemplate|MolFromSmiles|MolFromMolFile|MolFromMol2")
SAFE = re.compile(
    r"sanitize\s*=\s*False|proximityBonding\s*=\s*False|GetAtomicNum|GetMass\b|atomic_weight"
    r"|\.weight\b|GetAtomWithIdx\(\d+\)\.GetSymbol"
)
SHORTER_THAN = {
    ("C", "C"): 1.46, ("C", "N"): 1.41, ("C", "O"): 1.34, ("N", "N"): 1.34, ("N", "O"): 1.33,
}  # fmt: skip


def multiple_bonds(text: str) -> tuple[bool, list[str]]:
    """(has order information in CONECT, short bonds between heavy ligand atoms)."""
    atoms: dict[int, tuple[str, tuple[float, float, float], str]] = {}
    ordered = False
    for line in text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() not in ("HOH", "WAT"):
            try:
                serial = int(line[6:11])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            element = (line[76:78].strip() or line[12:16].strip()[:1]).capitalize()
            atoms[serial] = (element, xyz, line[12:16].strip())
        elif line.startswith("CONECT"):
            serials = [line[i : i + 5].strip() for i in range(6, len(line.rstrip()), 5)]
            if len(serials) > 1 and len(set(serials[1:])) < len(serials[1:]):
                ordered = True
    short: list[str] = []
    heavy = [a for a in atoms.values() if a[0] != "H"]
    for i, (e1, p1, n1) in enumerate(heavy):
        for e2, p2, n2 in heavy[i + 1 :]:
            limit = SHORTER_THAN.get(tuple(sorted((e1, e2))))
            d = math.dist(p1, p2)
            if limit and d < limit:
                short.append(f"{n1}-{n2} {d:.2f} A")
    return ordered, short


@checker("rdkit_pdb_bond_orders")
def rdkit_pdb_bond_orders(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not (READ.search(text) and PROPERTY.search(text)) or SAFE.search(text):
        return passed()
    templated = TEMPLATE.search(text) is not None
    declared = call.declared.get(key)
    if declared == "single_bonds" or (declared == "template" and templated):
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        content = file_text(ctx, path)
        if content is None:
            unreadable.append(path.name)
            continue
        ordered, short = multiple_bonds(content)
        if ordered or not short:
            continue
        if templated and declared is None:
            continue
        facts = (
            f"{path.name} lists ligand atoms without bond orders (no CONECT record repeats a "
            f"serial) and has bonds such as {short[0]}, i.e. double or aromatic ones; RDKit's "
            "PDB reader makes every bond single, fills the valences with implicit hydrogens "
            "and finds no aromatic atom, so the weight and the formula come out wrong."
        )
        fix = (
            "assign the orders from a template (Chem.AllChem.AssignBondOrdersFromTemplate) or "
            "read an SDF/SMILES, or sum the atomic weights of the listed atoms",
        )
        if declared is not None:
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}={declared} but {facts}",
                expected="orders from CONECT or a template",
                got="single bonds",
                fix=fix,
            )
        return CheckResult(
            "fail",
            "bond_orders_undeclared",
            facts,
            expected=list(rule.declare.values),
            got=None,
            fix=(*fix, f"then --declare {key}=template (or single_bonds if that is intended)"),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: its bond information cannot be checked.",
        )
    return passed()
