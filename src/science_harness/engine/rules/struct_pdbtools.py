"""Rules pdbtools.selaltloc_digits (family F95) and pdbtools.modified_residues (family F96): two
silent traps in the sequence/atom handling around pdb-tools and the Python structure libraries.

pdbtools.selaltloc_digits (slot record_count, incident R2-STR-16: pdb-tools issue 153, residues
whose atoms all carry alternate locations disappear; maintainers call it a bug). The cause is in
the code of pdb-tools 2.7.0: `pdb_selaltloc.is_another_altloc_group` tests `ru1 > ru0` on the
residue numbers as STRINGS, so '10' > '9' is False; the boundary between residue 9 and residue 10
(99/100, 999/1000, or any pair whose next number sorts lower as text) is not seen, the two residues
land in one group and the group is flushed as if it were one residue.
Verified here (pdb-tools 2.7.0, 110-residue chain, N CA C O CB per residue, exit 0, empty stderr;
expected = 550 atoms): CB alone carrying altlocs A 0.4 / B 0.6 at residue 3, 8, 10, 11, 19, 98, 100,
101: 550 atoms; at residue 9 or 99: 549 atoms; every atom of the residue split, at residue 3, 8,
11, 19, 98, 101: 550; at residue 9, 10, 99 or 100: 545 (a whole residue missing). gemmi, ProDy and
Biopython keep all 550.
The rule is a CALL rule on `pdb_selaltloc` (tier T1). The structure file is the argument, or a
file named earlier in the pipeline (`cat model.pdb | pdb_selaltloc`). It re-runs the grouping logic
of pdb-tools 2.7.0 on the file (a faithful port of `select_altloc`, `is_another_altloc_group` and
the four flush functions, for the default highest-occupancy mode and for `-X` altloc selection)
and compares the number of ATOM/HETATM lines it would write with the number of distinct atoms in the
file. Fewer lines -> fail (selaltloc_drops_atoms). A file without altlocs, or with altlocs away
from digit boundaries, costs nothing. PDB-format files only; an mmCIF passes. Declaration
`selaltloc_residues` (optional): verified (the caller compares the output with the input). A
declaration is a claim about the caller's next step; the rule cannot check it.

pdbtools.modified_residues (slot residue_alphabet, incidents R2-STR-03 and R2-STR-28: ProteinMPNN
and LigandMPNN turn residues they do not know into gaps or drop them). A modified residue
(selenomethionine MSE, phosphoserine SEP, TPO, PTR ...) is written as HETATM inside the polymer.
Verified here (pdb-tools 2.7.0, gemmi 0.7.5, Biopython 1.88, ProDy 2.6.1) on a 25-residue chain
with MSE at 6, SEP at 15, a gap and one water:
  pdb_tofasta                                  MKTAYXAKQRQISFXKSHFAGLDEX  (X for MSE, SEP and HOH)
  gemmi.one_letter_code(names), Bio seq1       the same string without the water
  Bio.PDB.PPBuilder().build_peptides           'MKTAY','AKQRQISF','KSHF','AGLDE': both residues
                                               missing from every peptide
  ProDy chain.getSequence()                    MKTAYMAKQRQISFSKSHFAGLDE  (parent letters M and S)
The rule is a FACT rule over structure files named in any command. It fires for code that extracts
a sequence with one of the above (`pdb_tofasta`, `one_letter_code`, `seq1`, `three_to_one`,
`PPBuilder`/`build_peptides`, `getSequence`) on a file where a polymer residue is a tabulated amino
acid other than the 20 standard ones (gemmi's residue table). Behaviour of the extractor: X
(pdb_tofasta, one_letter_code, seq1), omitted (PPBuilder), parent_letter (getSequence); code that
names MSE/SEP/TPO/PTR/CSO, HETATM, `modified`, `parent`, `find_tabulated_residue` or
`protein_letters_3to1_extended` is taken to have handled them.
  risky file, no declaration, extractor writing X or nothing -> fail (modified_residues_undeclared)
  no declaration, extractor writing the parent letter (getSequence) -> pass
  declared value different from the extractor's behaviour -> fail (declared_mismatch)
  declared value equal to it, or handled code             -> pass
Declaration `modified_residues`: parent_letter | X | omitted.

Not seen (both rules): file names built at run time, pipelines whose first stage is not in the
same command line, alternate readers of the sequence (awk over SEQRES, Bio SeqIO pdb-seqres).
"""

from __future__ import annotations

import re

from science_harness.engine.argv import has_option
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.struct_common import (
    STRUCT_RE,
    command_text,
    existing_files,
    file_text,
    is_cif,
    read_structure,
)

# -- pdb_selaltloc: a port of the grouping logic of pdb-tools 2.7.0 -----------------------------

RECORDS = ("ATOM", "HETATM", "ANISOU")
TERMINATORS = ("TER", "END", "CONECT", "END", "ENDMDL")


def _all_same_residue(altloc_lines: dict[str, list[str]]) -> bool:
    residues = {(ln[17:20], ln[22:26].strip()) for v in altloc_lines.values() for ln in v}
    return len(residues) == 1


def _partial(altloc_lines: dict[str, list[str]]) -> bool:
    return " " in altloc_lines and _all_same_residue(altloc_lines)


def _is_another(a1, a0, ru1, ru0, ra1, ra0, rl, rploc) -> bool:
    rv = list(rploc.values())
    return bool(
        all((a0, ra0, ru0))
        and (
            (a0 != a1 and a1 == " " and ru1 > ru0)
            or (a0 == " " and a1 != " " and ru1 > ru0)
            or (a0 == " " and a1 == " " and (ru1 != ru0 or ra1 != ra0))
            or (
                a0 == a1
                and a0 != " "
                and a1 in rl
                and ru1 > ru0
                and len(rl) > 1
                and all(len(v) == len(rv[0]) for v in rv[1:])
            )
        )
    )


def _flush_resloc(selloc, altloc_lines):
    if selloc in altloc_lines:
        return [ln[:16] + " " + ln[17:] for ln in altloc_lines[selloc]]
    return [ln for lines in altloc_lines.values() for ln in lines]


def _flush_resloc_occ(selloc, altloc_lines):
    highest, altloc = 0.0, " "
    for key, lines in altloc_lines.items():
        occ = float(lines[0][54:60])
        if occ > highest:
            altloc, highest = key, occ
    return [ln[:16] + " " + ln[17:] for ln in altloc_lines[altloc]]


def _sorted_atoms(altloc_lines):
    atoms: dict[tuple, tuple[int, list[str]]] = {}
    for lines in altloc_lines.values():
        for ln in lines:
            key = (int(ln[22:26]), ln[17:20].strip(), ln[12:16], ln[21])
            atoms.setdefault(key, (int(ln[6:11]), []))[1].append(ln)
    return sorted(atoms.items(), key=lambda x: (x[0][0], x[1][0]))


def _flush_id_same_residue(selloc, altloc_lines):
    out: list[str] = []
    for _, (_, lines) in _sorted_atoms(altloc_lines):
        chosen = [ln for ln in lines if ln[16] == selloc]
        out.extend([ln[:16] + " " + ln[17:] for ln in chosen] if chosen else lines)
    return out


def _flush_occ_same_residue(selloc, altloc_lines):
    out: list[str] = []
    for _, (_, lines) in _sorted_atoms(altloc_lines):
        atoms = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
        aniso = [ln for ln in lines if ln.startswith("ANISOU")]
        if aniso:
            pairs = sorted(
                zip(atoms, aniso, strict=False), key=lambda x: float(x[0][54:60]), reverse=True
            )
            out += [
                pairs[0][0][:16] + " " + pairs[0][0][17:],
                pairs[0][1][:16] + " " + pairs[0][1][17:],
            ]
        else:
            atoms.sort(key=lambda x: float(x[54:60]), reverse=True)
            out.append(atoms[0][:16] + " " + atoms[0][17:])
    return out


def emulate_selaltloc(lines: list[str], selloc: str | None = None) -> list[str]:
    """The lines pdb-tools 2.7.0 `pdb_selaltloc` writes (selloc None: highest occupancy)."""
    byocc = selloc is None
    multi = _flush_resloc_occ if byocc else _flush_resloc
    single = _flush_occ_same_residue if byocc else _flush_id_same_residue
    out: list[str] = []
    groups: dict[str, list[str]] = {}
    per_loc: dict[str, set] = {}
    prev = ("", "", "")  # altloc, resname, resnum

    def flush():
        nonlocal groups, per_loc
        out.extend((single if _partial(groups) else multi)(selloc, groups))
        groups, per_loc = {}, {}

    for line in lines:
        if line.startswith(RECORDS):
            altloc, resname, resnum = line[16], line[17:20], line[22:26].strip()
            if _is_another(altloc, prev[0], resnum, prev[2], resname, prev[1], groups, per_loc):
                flush()
            groups.setdefault(altloc, []).append(line)
            per_loc.setdefault(altloc, set()).add((resname, resnum))
            prev = (altloc, resname, resnum)
        elif line.startswith(TERMINATORS):
            if groups:
                flush()
            prev = ("", "", "")
            out.append(line)
        else:
            prev = ("", "", "")
            out.append(line)
    if groups:
        flush()
    return out


def distinct_atoms(lines: list[str]) -> int:
    return len(
        {(ln[21], ln[22:27], ln[12:16]) for ln in lines if ln.startswith(("ATOM", "HETATM"))}
    )


def kept_atoms(lines: list[str]) -> int:
    return sum(1 for ln in lines if ln.startswith(("ATOM", "HETATM")))


def first_missing(lines: list[str], written: list[str]) -> str:
    """A residue that lost atoms, as 'RES 9'."""
    have = {(ln[21], ln[22:27], ln[12:16]) for ln in written if ln.startswith(("ATOM", "HETATM"))}
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")) and (ln[21], ln[22:27], ln[12:16]) not in have:
            return f"{ln[17:20].strip()} {ln[22:27].strip()}"
    return "a residue"


@checker("pdbtools_selaltloc_digits")
def pdbtools_selaltloc_digits(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    if call.declared.get(key) == "verified" or has_option(call.argv, "-h", "--help"):
        return passed()
    selloc = next((a[1] for a in call.argv if len(a) == 2 and a[0] == "-" and a[1] != "-"), None)
    unreadable: list[str] = []
    for path in existing_files(call, ctx, STRUCT_RE):
        if is_cif(path):
            continue
        text = file_text(ctx, path)
        if text is None:
            unreadable.append(path.name)
            continue
        lines = text.splitlines()
        try:
            written = emulate_selaltloc(lines, selloc)
        except (ValueError, IndexError):
            unreadable.append(path.name)
            continue
        lost = distinct_atoms(lines) - kept_atoms(written)
        if lost > 0:
            where = first_missing(lines, written)
            return CheckResult(
                "fail",
                "selaltloc_drops_atoms",
                f"pdb_selaltloc would write {lost} atom(s) fewer than the {distinct_atoms(lines)} "
                f"distinct atoms of {path.name} (the first lost is in residue {where}): it "
                "compares residue numbers as text, so '10' sorts before '9' and an altloc "
                "residue next to a change of digit count is merged with its neighbour.",
                expected=distinct_atoms(lines),
                got=kept_atoms(written),
                fix=(
                    "keep the conformer with Bio.PDB.PDBParser (highest occupancy) and write it "
                    "with PDBIO, or renumber the residues so none sits at 9/10, 99/100",
                    f"or compare the atom count of the output with the input and "
                    f"--declare {key}=verified",
                ),
            )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: pdb_selaltloc's grouping cannot be checked.",
        )
    return passed()


# -- modified residues in sequence extraction ----------------------------------------------------

EXTRACTORS = {
    "X": re.compile(r"\bpdb_tofasta\b|\bone_letter_code\b|\bseq1\s*\(|\bthree_to_one\b"),
    "omitted": re.compile(r"\bPPBuilder\b|\bCaPPBuilder\b|\bbuild_peptides\b"),
    "parent_letter": re.compile(r"\bgetSequence\b"),
}
MODIFIED_HANDLED = re.compile(
    r"\bMSE\b|\bSEP\b|\bTPO\b|\bPTR\b|\bCSO\b|HETATM|(?i:modified)|(?i:parent)"
    r"|find_tabulated_residue|protein_letters_3to1_extended|aa_only\s*=\s*False"
)
STANDARD = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK".split()
)


def modified_residues(structure) -> list[str]:
    """Names of polymer residues that are tabulated amino acids but not one of the 20 standard."""
    import gemmi

    found: list[str] = []
    for chain in structure[0]:
        for res in chain:
            if res.name in STANDARD or res.is_water():
                continue
            info = gemmi.find_tabulated_residue(res.name)
            if info is not None and info.is_amino_acid() and res.name not in found:
                found.append(res.name)
    return found


@checker("pdbtools_modified_residues")
def pdbtools_modified_residues(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    behaviour = next((b for b, rx in EXTRACTORS.items() if rx.search(text)), None)
    if behaviour is None or MODIFIED_HANDLED.search(text):
        return passed()
    declared = call.declared.get(key)
    if declared == behaviour or (declared is None and behaviour == "parent_letter"):
        return passed()  # the parent letter is the convention a protein sequence needs
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        names = modified_residues(structure)
        if not names:
            continue
        shown = ", ".join(names)
        facts = (
            f"{path.name} holds modified residues ({shown}) inside the polymer; this way of "
            f"extracting the sequence writes them as {_describe(behaviour)}."
        )
        if declared is not None:
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}={declared} but {facts}",
                expected=declared,
                got=behaviour,
                fix=(_fix(behaviour),),
            )
        return CheckResult(
            "fail",
            "modified_residues_undeclared",
            facts,
            expected=list(rule.declare.values),
            got=None,
            fix=(
                _fix(behaviour),
                f"then --declare {key}=parent_letter (or {behaviour} if intended)",
            ),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: its modified residues cannot be checked.",
        )
    return passed()


def _describe(behaviour: str) -> str:
    return {
        "X": "X (the residue name is not in the 20-letter table)",
        "omitted": "nothing (Biopython's peptide builders skip hetero residues, split the chain)",
        "parent_letter": "the letter of the parent residue (M for MSE, S for SEP)",
    }[behaviour]


def _fix(behaviour: str) -> str:
    if behaviour == "parent_letter":
        return "ProDy's getSequence already maps modified residues to their parent letters"
    return (
        "map modified residues to their parent letter, e.g. "
        "gemmi.find_tabulated_residue(name).one_letter_code.upper(), before joining the letters"
    )
