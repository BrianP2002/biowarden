"""Rule mdtraj.resid_index (slot index_base, family F92): in an mdtraj selection `resid N` is the
0-based residue INDEX, not the residue number of the PDB file (that is `resSeq N`).

The trap (mdtraj selection language, `Topology.select`). The word `resid` looks like the PDB
"residue ID", and in MDAnalysis and ProDy it is the PDB residue number. In mdtraj it is
`residue.index`, counted from 0 over the whole topology (all chains, waters included); the PDB
number is `residue.resSeq`.

Verified here (mdtraj 1.10.3, gemmi 0.7.5) on a 22-residue chain numbered 19..44 with a gap after
residue 27:
  top.select("resid 25 and name CA")     [] (index 25 does not exist; an empty array, no error)
  top.select("resSeq 25 and name CA")    the CA of PDB residue 25 (index 6)
  top.select("resid 8 and name CA")      the CA of PDB residue 30 (index 8)
and on a chain numbered 1..N: `resid 12` is PDB residue 13. Distances come back in nanometres
(1.0665 against 10.665 angstroms read with gemmi), a separate mdtraj convention. Exit 0 and no
warning in every case.

What the rule does. A FACT rule over .pdb/.ent/.cif/.mmcif files named in a command run by
python-struct, python-bio, python3 or python. It looks at the command first: code that mentions
mdtraj (`mdtraj`, `md.load`) and a selection with `resid N` (also `resid N to M`; every number is
checked). Then at the file (gemmi): the residue numbers in file order; `resid N` is risky when the
residue with index N has another number than N, or no residue has index N.
  risky, no declaration                       -> fail (resid_index_undeclared)
  declared residue_index                      -> pass (the caller means the 0-based index)
  declared pdb_number with `resid`            -> fail (declared_mismatch)
  not risky (the first residue is numbered 0 and the numbers run on without gaps), `resSeq`
  selections, other libraries                 -> pass
`unknown`: a file gemmi cannot read.

Declaration `residue_numbering` (optional): pdb_number | residue_index. The fix names the safe
spelling: `top.select("resSeq 25 and name CA")`.

Not seen: indices built at run time (`f"resid {n}"` with a variable), `top.residue(n)` lookups and
`atom_slice` with index lists, selections in another file than the one named in the command.
"""

from __future__ import annotations

import re

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.struct_common import command_text, read_structure

LIB = re.compile(r"\bmdtraj\b|\bmd\.load")
RESID = re.compile(r"\bresid\s+(-?\d+(?:\s+(?:to|-)\s+-?\d+)?)")
NUMBERS = re.compile(r"-?\d+")


def numbers_in_order(structure) -> list[int]:
    return [res.seqid.num for chain in structure[0] for res in chain]


@checker("mdtraj_resid_index")
def mdtraj_resid_index(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not LIB.search(text):
        return passed()
    wanted = sorted({int(n) for m in RESID.finditer(text) for n in NUMBERS.findall(m.group(1))})
    if not wanted:
        return passed()
    declared = call.declared.get(key)
    if declared == "residue_index":
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        nums = numbers_in_order(structure)
        bad = [(n, nums[n] if 0 <= n < len(nums) else None) for n in wanted]
        bad = [(n, got) for n, got in bad if got != n]
        if not bad:
            continue
        n, got = bad[0]
        what = (
            f"residue number {got}"
            if got is not None
            else f"no residue (the file has {len(nums)} residues, indices 0..{len(nums) - 1})"
        )
        first = nums[0] if nums else "?"
        message = (
            f"mdtraj's `resid {n}` is the 0-based residue index, not the residue number: in "
            f"{path.name} (first residue numbered {first}) it is {what}."
        )
        fix = (
            f'select by PDB number: top.select("resSeq {n} and name CA") ("resid" is the index)',
        )
        if declared == "pdb_number":
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}=pdb_number but {message}",
                expected="pdb_number",
                got="residue_index",
                fix=fix,
            )
        return CheckResult(
            "fail",
            "resid_index_undeclared",
            message,
            expected=["pdb_number", "residue_index"],
            got=None,
            fix=(*fix, f"or --declare {key}=residue_index if the 0-based index is what you mean"),
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: the residue numbering cannot be compared.",
        )
    return passed()
