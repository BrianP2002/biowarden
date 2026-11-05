"""Rule struct.altloc_policy (slot altloc_policy, family F90): which alternate location a structure
library keeps by default.

The trap (R2-STR-05: ProteinMPNN and RFdiffusion keep different conformers of the same atom). When
an atom has alternate locations A, B, ... each with an occupancy, "the" coordinates depend on the
reader. Verified here (ProDy 2.6.1, gemmi 0.7.5, Biopython 1.88, pdb-tools 2.7.0) on a PDB whose
residues 12 and 30 carry altlocs A (occupancy 0.35) and B (0.65) on every atom:

  prody.parsePDB(f)                altloc A, 88 atoms; CA of residue 12 is the 0.35 one
  prody.parsePDB(f, altloc='B')    altloc B
  prody.parsePDB(f, altloc=True)   both, as two coordinate sets
  gemmi.read_structure(f)          97 atoms, both conformers; the first CA of the residue is A
  Structure.remove_alternative_conformations()
                                   88 atoms, the first conformer (A), whatever its occupancy
  Bio.PDB.PDBParser                a DisorderedAtom whose selected atom has the HIGHEST
                                   occupancy (B, 0.65)
  pdb_selaltloc                    the highest occupancy (but see F95 for its bug)

All exit 0 with empty stderr (ProDy prints its "@> ... parsed" line). The CA-CA distance of two
residues with altlocs differs by 2.4-3.6 angstroms between the A and the B conformers in the family.

What the rule does. A FACT rule over .pdb/.ent/.cif/.mmcif files named in a command run by
python-struct, python-bio, python3 or python. It looks at the code first: it must use coordinates
(`getCoords`, `.pos`, `.xyz`, `.positions`, `dist(`, `compute_distances`, `float(`, `.split(`...);
viewers and code that only counts or lists atoms are left alone.
Then at the data (gemmi): an atom name with two or more alternate locations in one residue where
the conformer with the highest occupancy is not the one listed first (the risky state; files whose
major conformer is A, and files without altlocs, cost nothing).
  risky data, no declaration, code that handles it (names occupancy / altloc / selaltloc /
  `PDBParser`)                                              -> pass
  risky data, no declaration, other code                    -> fail (altloc_policy_undeclared)
  declared highest_occupancy, code that uses the ProDy / gemmi first-conformer default and
  never names occupancy or altloc                           -> fail (declared_mismatch)
  declared highest_occupancy otherwise, first_listed, all_conformers   -> pass
`unknown`: a structure file gemmi cannot read.

Declaration `altloc_policy` (optional): highest_occupancy | first_listed | all_conformers. A
declaration states the intent; the rule cannot verify that the code applies the policy it declares.

Not seen: files read by awk/grep/perl (outside the interpreters watched), selections made in an
earlier command whose output file the command names, and altlocs of different residue names
(microheterogeneity), which gemmi keeps as separate residues.
"""

from __future__ import annotations

import re

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.struct_common import (
    command_text,
    read_structure,
    reads_structure_as_code,
)

READS = re.compile(
    r"getCoords|get_coord|\.pos\b|\.xyz\b|\.positions\b|\bdist\(|calcDistance|calcRMSD|calcCenter"
    r"|compute_distances|superpose|float\(|\.split\(|substr\(|\[30:\s*38\]|\$[6-8]\b"
    r"|getCoordsets|get_vector|\bcoord\b"
)
FIRST_DEFAULT = re.compile(
    r"\bparse(?:PDB|MMCIF|CIF)\b|remove_alternative_conformations|read_structure|read_pdb"
)
HANDLED = re.compile(
    r"(?i)occupanc|\bocc\b|altloc|selaltloc|get_occupancy|getOccupancies|\bPDBParser\b"
    r"|\bMMCIFParser\b|\[54:60\]|substr\(\$0\s*,\s*55"
)


def inversions(structure) -> list[str]:
    """Descriptions of atoms whose highest-occupancy conformer is not the one listed first."""
    found: list[str] = []
    for chain in structure[0]:
        for res in chain:
            by_name: dict[str, list] = {}
            for atom in res:
                if atom.altloc != "\0":
                    by_name.setdefault(atom.name, []).append(atom)
            for name, atoms in by_name.items():
                if len(atoms) < 2:
                    continue
                major = max(atoms, key=lambda a: a.occ)
                if major.occ > atoms[0].occ + 1e-6:
                    first = atoms[0]
                    found.append(
                        f"{res.name} {res.seqid.num} {name}: altloc {first.altloc} "
                        f"({first.occ:.2f}) is listed first, altloc {major.altloc} "
                        f"({major.occ:.2f}) has the higher occupancy"
                    )
    return found


@checker("struct_altloc_policy")
def struct_altloc_policy(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not reads_structure_as_code(call.tool, text, READS):
        return passed()
    declared = call.declared.get(key)
    handled = HANDLED.search(text) is not None
    first_default = FIRST_DEFAULT.search(text) is not None and not handled
    if declared in ("first_listed", "all_conformers") or (
        declared == "highest_occupancy" and not first_default
    ):
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        found = inversions(structure)
        if not found:
            continue
        shown = f"{found[0]}" + (f" (and {len(found) - 1} more atoms)" if len(found) > 1 else "")
        facts = (
            f"{path.name} has alternate locations whose major conformer is not the first listed: "
            f"{shown}. ProDy's parsePDB and gemmi's remove_alternative_conformations keep the "
            "first conformer, Biopython keeps the highest-occupancy one."
        )
        if declared == "highest_occupancy":
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}=highest_occupancy but this code keeps the first conformer. "
                + facts,
                expected="highest_occupancy",
                got="first_listed",
                fix=(
                    "read the file with Bio.PDB.PDBParser (highest occupancy) or choose the atom "
                    "with the largest occupancy yourself",
                ),
            )
        if handled:
            continue
        return CheckResult(
            "fail",
            "altloc_policy_undeclared",
            facts,
            expected=["highest_occupancy", "first_listed", "all_conformers"],
            got=None,
            fix=(
                "keep the highest-occupancy conformer (Bio.PDB.PDBParser does; or pick the atom "
                f"with the largest occupancy) and --declare {key}=highest_occupancy",
                f"or --declare {key}=first_listed if altloc A is what you want",
            ),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: the alternate locations cannot be checked.",
        )
    return passed()
