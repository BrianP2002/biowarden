"""Rules prody.chain_scheme (family F89) and prody.range_end (family F91): two ProDy conventions
that differ from the other structure libraries and fail without a message.

prody.chain_scheme (slot chain_id_scheme, incident R2-STR-07: ProDy issue 1985, LigandMPNN pull 81).
An mmCIF holds two chain names: label_asym_id (assigned by the archive) and auth_asym_id (the
author's chain ID, the one in the paper and in the PDB file). gemmi and Biopython's MMCIFParser
report the author ID; ProDy's `parseMMCIF` (also `parseCIF`, and `parsePDB` on a .cif, which falls
back to it with the warning "Trying to parse as mmCIF file instead") reports label_asym_id unless
`unite_chains=True`. Verified here (ProDy 2.6.1, gemmi 0.7.5, Biopython 1.88) on an mmCIF with
author chains E and I and label chains A and B: gemmi E, I; Biopython E, I; `parseMMCIF(f)` A, B;
`parseMMCIF(f, unite_chains=True)` E, I; exit 0, stderr only ProDy's "@> 79 atoms ... parsed" line.

prody.range_end (slot interval_closure, source: the ProDy selection documentation). In a ProDy
selection string a numeric range written `resnum 10:15` follows Python slicing and EXCLUDES the end;
`resnum 10 to 15` includes it. Verified on a chain with residues 10..15 (and 12A): `resnum 10 to 15
and name CA` 7 atoms, `resnum 10:15 and name CA` 6 atoms, the same without residue 15. MDAnalysis
`resid 10:15` and `resid 10-15` include the end (7 atoms). Exit 0, no warning.

What the rules do. Both are FACT rules: they apply to every existing .cif/.mmcif (chain_scheme) or
structure file (range_end: .pdb .ent .cif .mmcif) named anywhere in a command run by python-struct,
python-bio, python3 or python (inline code, a heredoc, a script the command runs; `.py` scripts are
read). They look at the command first (zero friction on everything else):
  chain_scheme  fires only for code that calls ProDy `parseMMCIF`, `parseCIF` or `parsePDB` and
                does not pass `unite_chains=True` or read the segment names (ProDy keeps
                auth_asym_id there: `getSegnames`, `segment X`)
  range_end     fires only for code that mentions ProDy (`prody`, `parsePDB`, `parseMMCIF`,
                `Select(`) and a selection `resnum A:B` / `resid A:B` (optional third number,
                negative numbers allowed), and not for `select_atoms` (MDAnalysis)
and then at the data (gemmi):
  chain_scheme  a chain whose polymer residues carry a label_asym_id other than the chain's author
                name (waters and ligands are left out when the entity types are known)
  range_end     a residue numbered B exists in the file (then ':' leaves it out; if it does not,
                the two spellings select the same atoms and the rule stays silent)

Declarations (optional, `required: false`):
  chain_id_scheme  auth | label    the chain names the code means. Risky data and no declaration ->
                   fail; declared label -> pass (the caller wants the archive labels); declared
                   auth with ProDy's default -> fail (contradiction: ProDy returns the label IDs).
  range_end        inclusive | exclusive    Risky data and no declaration -> fail; declared
                   exclusive -> pass; declared inclusive -> fail (ProDy's ':' excludes the end).
`unknown`: a structure file gemmi cannot read.

Known false positive: a range written by hand to include its end (`resnum 15:25` for residues 15 to
24) is flagged until it declares range_end=exclusive.

Not seen: file names built at run time or read from a variable, ProDy calls through an alias that
hides the function name (`from prody import parseMMCIF as p` is seen, `getattr` is not), `resnum`
ranges inside a file the command does not run, and selections with a `:` range on other keywords.
"""

from __future__ import annotations

import re

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.struct_common import command_text, is_cif, read_structure

PARSE_CALL = re.compile(r"\bparse(?:MMCIF|CIF|PDB)\s*\(|\bparse(?:MMCIF|CIF)\b")
UNITE = re.compile(r"unite_chains\s*=\s*True")
SEGMENT = re.compile(r"getSegnames|\bsegname\b|\bsegment\s+\w|\bgetSegname")
PRODY_MARK = re.compile(r"\bprody\b|\bparse(?:PDB|MMCIF|CIF)\b|\bSelect\(")
SLICE = re.compile(r"\b(?:resnum|resid)\s+(-?\d+)\s*:\s*(-?\d+)(?:\s*:\s*-?\d+)?")


def label_pairs(structure) -> list[tuple[str, list[str]]]:
    """(author chain name, label_asym_ids of its polymer residues) for chains where they differ."""
    import gemmi

    out: list[tuple[str, list[str]]] = []
    for chain in structure[0]:
        residues = [r for r in chain if not r.is_water()]
        polymer = [r for r in residues if r.entity_type == gemmi.EntityType.Polymer]
        labels = sorted({r.subchain for r in (polymer or residues)})
        if labels and labels != [chain.name]:
            out.append((chain.name, labels))
    return out


@checker("prody_chain_scheme")
def prody_chain_scheme(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not PARSE_CALL.search(text):
        return passed()
    # ProDy keeps auth_asym_id in the segment names, so code that reads those is not affected
    united = UNITE.search(text) is not None or SEGMENT.search(text) is not None
    declared = call.declared.get(key)
    unreadable: list[str] = []
    for path in match.files:
        if not is_cif(path):
            continue
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        pairs = label_pairs(structure)
        if not pairs or united or declared == "label":
            continue
        shown = ", ".join(f"author {a} is label {'/'.join(b)}" for a, b in pairs)
        name = path.name
        fix_code = f"prody.parseMMCIF('{name}', unite_chains=True)"
        if declared == "auth":
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}=auth but ProDy's parseMMCIF default names the chains of "
                f"{name} by label_asym_id ({shown}).",
                expected="auth",
                got="label",
                fix=(f"pass unite_chains=True: {fix_code}", "or read the chain names with gemmi"),
            )
        return CheckResult(
            "fail",
            "chain_scheme_undeclared",
            f"{name} has author chain names that differ from its label_asym_id ({shown}); "
            "ProDy's default returns the label IDs, gemmi and Biopython return the author IDs, "
            "so the same chain name is another chain.",
            expected=["auth", "label"],
            got=None,
            fix=(
                f"use the author IDs: {fix_code}, then --declare {key}=auth",
                f"or --declare {key}=label if the label_asym_id names are what you mean",
            ),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: the chain names cannot be compared.",
        )
    return passed()


def residue_numbers(structure) -> set[int]:
    return {res.seqid.num for chain in structure[0] for res in chain}


@checker("prody_range_end")
def prody_range_end(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not PRODY_MARK.search(text) or "select_atoms" in text:
        return passed()
    ranges = [(int(m.group(2)), m.group(0)) for m in SLICE.finditer(text)]
    if not ranges:
        return passed()
    declared = call.declared.get(key)
    if declared == "exclusive":
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        numbers = residue_numbers(structure)
        for end, written in ranges:
            if end not in numbers:
                continue
            fixed = written.replace(":", " to ", 1).replace("  ", " ")
            if declared == "inclusive":
                return CheckResult(
                    "fail",
                    "declared_mismatch",
                    f"you declared {key}=inclusive but ProDy's '{written}' stops before residue "
                    f"{end}, which {path.name} has.",
                    expected="inclusive",
                    got="exclusive",
                    fix=(f"write '{fixed}' (the 'to' range includes both ends)",),
                )
            return CheckResult(
                "fail",
                "range_end_undeclared",
                f"ProDy reads '{written}' like a Python slice: residue {end} is left out, and "
                f"{path.name} has a residue {end}. MDAnalysis would include it.",
                expected=["inclusive", "exclusive"],
                got=None,
                fix=(
                    f"write '{fixed}' for both ends included, or '{written.split(':')[0]}:"
                    f"{end + 1}' by hand",
                    f"then --declare {key}=inclusive",
                ),
            )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: the end of the range cannot be checked.",
        )
    return passed()
