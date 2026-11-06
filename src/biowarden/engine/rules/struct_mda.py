"""Rule mda.protein_selection (slot selection_scope, family F99): MDAnalysis's `protein` keyword
is a fixed residue-name list that leaves out phosphorylated and other modified residues.

The trap (`MDAnalysis.core.selection.ProteinSelection.prot_res`, MDAnalysis 2.9.0). Checked against
mdtraj 1.10.3, ProDy 2.6.1, gemmi 0.7.5 and Biopython 1.88 on a 25-residue chain with MSE at residue
6, SEP at residue 15 and one water:

  MDAnalysis  select_atoms("protein").n_residues   23   (SEP is not in the list)
  mdtraj      select("protein")                    24
  ProDy       select("protein")                    24
  gemmi       residues with is_amino_acid()        24
  Biopython   is_aa(residue)                       24

The MDAnalysis list holds MSE, HYP, HID, HIE, CYX, ASH, GLH, NME and ACE but not SEP, TPO, PTR, CSO
or MLY; mdtraj's list holds SEP, TPO, PTR, MSE, CSO, HYP, MLY, GLH, NME, ACE. The `backbone` keyword
uses the same residue list. Exit 0 and no warning. The MDAnalysis list is copied in this module
(MDAnalysis is not installed on the host that runs the engine).

What the rule does. A FACT rule over .pdb/.ent/.cif/.mmcif files named in a command run by
python-struct, python-bio, python3 or python. It looks at the code first: MDAnalysis
(`MDAnalysis`, `mda.`) with a `select_atoms("...")` string that uses `protein`, `backbone`,
`nucleic` or `nucleicbackbone`; code that adds `or resname ...` / `or name ...` to the selection,
or names SEP/TPO/PTR/CSO/HETATM/`is_amino_acid`, is taken to have handled modified residues.
Then at the data (gemmi): a polymer residue whose name is a tabulated amino acid (gemmi's residue
table) that is not in MDAnalysis's list (SEP, TPO, PTR, CSO, MLY, ...).
  risky data, no declaration                  -> fail (protein_selection_undeclared)
  declared includes_modified                  -> fail (declared_mismatch: the keyword excludes them)
  declared standard_only                      -> pass
  no such residue in the file                 -> pass
Declaration `protein_selection` (optional): includes_modified | standard_only. `unknown`: an
unreadable structure.

Not seen: selections stored in a variable, `u.select_atoms(sel)` where `sel` is built elsewhere,
and AtomGroup.select_atoms with `protein` on a sub-selection of another command.
"""

from __future__ import annotations

import re

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.struct_common import command_text, read_structure

# MDAnalysis 2.9.0 ProteinSelection.prot_res
MDA_PROTEIN = frozenset(
    "ACE ALA ALAD ARG ARGN ASF ASH ASN ASN1 ASP ASPH CALA CARG CASF CASN CASP CCYS CCYX CGLN CGLU "
    "CGLY CHID CHIE CHIP CILE CLEU CLYS CME CMET CPHE CPRO CSER CTHR CTRP CTYR CVAL CYM CYS CYS1 "
    "CYS2 CYSH CYX DAB GLH GLN GLU GLUH GLY HID HIE HIP HIS HIS1 HIS2 HISA HISB HISD HISE HISH HSD "
    "HSE HSP HYP ILE LEU LYN LYS LYSH MET MSE NALA NARG NASN NASP NCYS NCYX NGLN NGLU NGLY NHID "
    "NHIE NHIP NILE NLEU NLYS NME NMET NPHE NPRO NSER NTHR NTRP NTYR NVAL ORN PGLU PHE PRO QLN SER "
    "THR TRP TYR VAL".split()
)
LIB = re.compile(r"\bMDAnalysis\b|\bmda\.")
KEYWORD = re.compile(
    r"""select_atoms\(\s*[rf]?['"][^'"]*\b(?:protein|backbone|nucleic|nucleicbackbone)\b"""
)
HANDLED = re.compile(
    r"\bSEP\b|\bTPO\b|\bPTR\b|\bCSO\b|HETATM|is_amino_acid|\bor\s+(?:resname|name)\b"
)


def missing_from_protein(structure) -> list[str]:
    """Polymer residue names that are amino acids for gemmi but not in MDAnalysis's `protein`."""
    import gemmi

    found: list[str] = []
    for chain in structure[0]:
        for res in chain:
            if res.name in MDA_PROTEIN or res.is_water() or res.name in found:
                continue
            info = gemmi.find_tabulated_residue(res.name)
            if info is not None and info.is_amino_acid():
                found.append(res.name)
    return found


@checker("mda_protein_selection")
def mda_protein_selection(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not (LIB.search(text) and KEYWORD.search(text)) or HANDLED.search(text):
        return passed()
    declared = call.declared.get(key)
    if declared == "standard_only":
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        names = missing_from_protein(structure)
        if not names:
            continue
        shown = ", ".join(names)
        facts = (
            f"{path.name} holds modified amino acids ({shown}) that MDAnalysis's `protein` and "
            "`backbone` keywords do not include (a fixed residue-name list); mdtraj, ProDy and "
            "gemmi count them as protein."
        )
        fix = (
            f"select by residue name as well: select_atoms('protein or resname {' '.join(names)}')"
            ", or select_atoms('not resname HOH') for a chain without ligands",
        )
        if declared == "includes_modified":
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}=includes_modified but {facts}",
                expected="includes_modified",
                got="standard_only",
                fix=fix,
            )
        return CheckResult(
            "fail",
            "protein_selection_undeclared",
            facts,
            expected=list(rule.declare.values),
            got=None,
            fix=(*fix, f"then --declare {key}=includes_modified"),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: its residue names cannot be checked.",
        )
    return passed()
