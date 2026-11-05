"""Rule struct.bfactor_scale (slot metric_scale, family F97, incidents STR-19, STR-20, STR-21).

The B-factor column of a predicted structure holds pLDDT on a 0-100 scale (AlphaFold) or on a 0-1
scale (ESMFold2; Boltz writes 0-100 in the B-factor column but 0-1 in `_ma_qa_metric_local`), and
RFdiffusion writes a 0/1 design flag there.

The trap. Nothing in a PDB or mmCIF file says which scale the column uses. A confidence filter
written for AlphaFold (`B > 70`, `plddt >= 90`) is valid awk, Python or ProDy and selects nothing
on a 0-1 file; the reverse (a cutoff of 0.7 on a 0-100 file) selects everything. Both exit 0 with
an empty stderr. Verified in the F97 family on a 44-60 residue chain written on both scales: awk
`substr($0,61,6)+0>70` counts 0 residues on the 0-1 file and the true number on the 0-100 file;
rescaling by max(B) <= 1 gives the same count on both.

What the rule does. A FACT rule over .pdb/.ent/.cif/.mmcif files named in any command. It looks at
the command first: code that reads the B-factor column (`substr($0,61`, `$11`, `b_factor`,
`bfactor`, `get_bfactor`, `getBetas`, `beta` in a ProDy selection, `b_iso`, `plddt`, `[60:66]`)
and compares it with a number: a cutoff of 10 to 100 (the 0-100 convention) or one in (0, 1]
(the 0-1 convention). Code that rescales (`* 100`, `/ 100`, `max(`, `min(`, `scale`, `normalis`)
or only prints values is left alone. Then at the data (gemmi): the largest B-factor of the file.
  data on a 0-1 scale (max <= 1.0), cutoff written for 0-100        -> fail
  data on a 0-100 scale (max > 1.5), cutoff written for 0-1         -> fail
    (no declaration: bfactor_scale_undeclared; declared the other scale: declared_mismatch)
  declared experimental_b (the column is a crystallographic B)      -> pass
  declared value equal to the data's scale, cutoff for the same     -> pass
  declared value equal to the data's scale, cutoff for the other    -> fail
                                           (cutoff_contradicts_declaration)
The message names the min/max, and says so when every value is 0 or 1 (a design flag, not a
confidence). Declaration `bfactor_scale`: 0_1 | 0_100 | experimental_b. Experimental B-factors of
crystal structures (max above 1.5, cutoff written for 0-100) are indistinguishable from pLDDT by
value; they need `experimental_b` or a rescaling marker to pass.

Not seen: the confidence JSON files next to the structure, a cutoff held in a variable
(`thr = 70`), per-residue arrays built by another command.
"""

from __future__ import annotations

import re

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.struct_common import command_text, read_structure

BFACTOR = re.compile(
    r"(?i)b_?factor|getBetas|get_bfactor|\.bfactor|\bbeta\b|b_iso|plddt|substr\(\s*\$0\s*,\s*61"
    r"|\$11\b|\[60:\s*66\]|\[60:\]|B_iso_or_equiv"
)
COMPARISON = re.compile(r"[<>]=?\s*(\d+(?:\.\d+)?)\b")
RESCALED = re.compile(r"(?i)\*\s*100|/\s*100|\bmax\s*\(|\bscale|normali[sz]|\bmin\s*\(")


def cutoffs(text: str) -> list[float]:
    return [float(m.group(1)) for m in COMPARISON.finditer(text)]


def b_range(structure) -> tuple[float, float, bool]:
    """(min, max, all values are 0 or 1) over the atoms of the first model."""
    values = [atom.b_iso for chain in structure[0] for res in chain for atom in res]
    if not values:
        return 0.0, 0.0, False
    flag = all(abs(v) < 1e-6 or abs(v - 1.0) < 1e-6 for v in values)
    return min(values), max(values), flag


@checker("struct_bfactor_scale")
def struct_bfactor_scale(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not BFACTOR.search(text) or RESCALED.search(text):
        return passed()
    limits = cutoffs(text)
    wants_100 = any(10 <= c <= 100 for c in limits)
    wants_1 = any(0 < c <= 1.0 for c in limits)
    if not (wants_100 or wants_1):
        return passed()
    declared = call.declared.get(key)
    if declared == "experimental_b":
        return passed()
    unreadable: list[str] = []
    for path in match.files:
        structure = read_structure(ctx, path)
        if structure is None:
            unreadable.append(path.name)
            continue
        low, high, flag = b_range(structure)
        data = "0_1" if high <= 1.0 else ("0_100" if high > 1.5 else None)
        if wants_100 and wants_1:
            continue
        cutoff_scale = "0_100" if wants_100 else "0_1"
        if data is None or (declared is None and cutoff_scale == data):
            continue
        what = (
            "every value is 0 or 1 (a design flag such as RFdiffusion's, not a confidence)"
            if flag
            else f"the values run from {low:g} to {high:g}"
        )
        shown = ", ".join(f"{c:g}" for c in limits)
        facts = (
            f"in {path.name} {what}, a {'0-1' if data == '0_1' else '0-100'} scale, but the "
            f"cutoff in this command ({shown}) is written for "
            f"{'0-100' if cutoff_scale == '0_100' else '0-1'}, so it selects "
            f"{'nothing' if data == '0_1' else 'everything'}."
        )
        fix = (
            "multiply the B-factors by 100 before the cutoff, or use the cutoff 0.7 for 0-1 data"
            if data == "0_1"
            else "divide the cutoff by 100, or keep it with the 0-100 values"
        )
        if declared is None:
            return CheckResult(
                "fail",
                "bfactor_scale_undeclared",
                facts,
                expected=list(rule.declare.values),
                got=None,
                fix=(fix, f"then --declare {key}={data}"),
            )
        if declared != data:
            return CheckResult(
                "fail",
                "declared_mismatch",
                f"you declared {key}={declared} but {facts}",
                expected=data,
                got=declared,
                fix=(fix, f"declare {key}={data}"),
            )
        if cutoff_scale != declared:
            return CheckResult(
                "fail",
                "cutoff_contradicts_declaration",
                f"you declared {key}={declared} but the cutoff in this command ({shown}) is "
                f"written for the {'0-100' if cutoff_scale == '0_100' else '0-1'} scale.",
                expected=declared,
                got=cutoff_scale,
                fix=(fix,),
            )
    if unreadable:
        return CheckResult(
            "unknown",
            "unreadable_structure",
            f"could not read {', '.join(unreadable)}: the B-factor scale cannot be checked.",
        )
    return passed()
