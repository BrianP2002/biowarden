"""Rule tmalign.normalisation (slot metric_normalisation, family F93, incident STR-22): TMalign
prints two TM-scores, normalised by the length of structure 1 and of structure 2.

The trap (US-align / TM-align documentation: "You should use TM-score normalized by length of the
reference structure"). Verified here (TM-align 20240303, `TMalign model.pdb reference.pdb`, a
40-residue model against a 54-residue reference, exit 0, empty stderr):

  Length of Structure_1: 40 residues     Length of Structure_2: 54 residues
  TM-score= 0.93048 (normalized by length of Structure_1: L=40, d0=1.83)
  TM-score= 0.70964 (normalized by length of Structure_2: L=54, d0=2.41)

The first line is the model-normalised value (0.93), the reference-normalised one is 0.71. Options
add lines and never remove these two: `-a T` appends "(if normalized by average length of two
structures: L=47.0)" (0.80657), `-u 54` / `-L 54` appends "normalized by user-specified L=54.00"
(0.70964), `-d 5` appends "scaled by user-specified d0=5.00" (0.73315). The two lines agree when
the structures have the same length. By default (`-ter 3`) TMalign reads only the first chain of
each file (up to TER, ENDMDL, END or a change of chain ID), so the lengths are those of chain 1.

What the rule does. It is a CALL rule on `TMalign`, tier T1 (it reads the two structure files).
It takes the first two arguments that are not options (an option and its value are skipped; the
batch modes `-dir`, `-dir1`, `-dir2` are left alone), counts the residues of the first chain of
each (CA records of a PDB file; the first chain of an mmCIF through gemmi; `-ter 0` counts the whole
file) and stays silent when the two counts are equal or a file cannot be read (unknown).
When the lengths differ, the caller must say which length normalises the score it reports:
  no declaration, no `-a`/`-u`/`-L`       -> fail (tm_norm_undeclared)
  declared structure_1 | structure_2       -> pass (the rule cannot see which line is read)
  declared average without `-a T`          -> fail (declared_mismatch)
  declared given_length without `-u/-L`    -> fail (declared_mismatch)
  `-a T` or `-u N` / `-L N` given, no declaration -> pass (the call itself asks for that value)
Declaration `tmscore_normalisation` (optional): structure_1 | structure_2 | average |
given_length. A call that puts the reference first (`TMalign reference.pdb model.pdb`) and reads
the first line is right, but the rule cannot know which file is the reference: it is flagged until
it declares structure_1 (documented limit; the admission cases use the model-first order that
the task wording induces).

Not seen: pipelines downstream of TMalign (the line a `grep`, `sed` or Python parser keeps),
`-i/-I` user alignments, RNA mode, and the US-align program.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.struct_common import file_text, is_cif, read_structure

WITH_VALUE = frozenset(
    "-a -u -L -d -i -I -m -o -ter -atom -mol -suffix -dir -dir1 -dir2 -infmt1 -infmt2".split()
)
BATCH = frozenset({"-dir", "-dir1", "-dir2"})
TRUE_WORDS = frozenset({"T", "t", "True", "true", "TRUE", "1"})


def parse_argv(argv: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """(positional arguments, option -> value) of a TMalign command line."""
    positional: list[str] = []
    options: dict[str, str] = {}
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        if token in WITH_VALUE:
            options[token] = items[i + 1] if i + 1 < len(items) else ""
            i += 2
            continue
        if token.startswith("-") and len(token) > 1:
            options[token] = ""
        else:
            positional.append(token)
        i += 1
    return positional, options


def first_chain_length(ctx: Ctx, path: Path, ter: int) -> int | None:
    """Residues TMalign reads from a file: CA atoms of the first chain (the whole file if ter=0)."""
    if is_cif(path):
        structure = read_structure(ctx, path)
        if structure is None:
            return None
        chains = list(structure[0])
        picked = chains if ter == 0 else chains[:1]
        return sum(sum(1 for r in c if r.find_atom("CA", "*") is not None) for c in picked)
    text = file_text(ctx, path)
    if text is None:
        return None
    count, chain = 0, None
    for line in text.splitlines():
        if line.startswith(("ENDMDL", "END")) and ter != 0:
            break
        if line.startswith("TER") and ter == 3 and count:
            break
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[16] in " A1":
            if ter in (2, 3) and chain is not None and line[21] != chain:
                break
            chain = line[21]
            count += 1
    return count


@checker("tmalign_normalisation")
def tmalign_normalisation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    positional, options = parse_argv(call.argv)
    if BATCH & set(options) or len(positional) < 2:
        return passed()
    ter_text = options.get("-ter", "3")
    ter = int(ter_text) if ter_text.isdigit() else 3
    lengths: list[int] = []
    for name in positional[:2]:
        path = ctx.resolve(name)
        n = first_chain_length(ctx, path, ter) if path.is_file() else None
        if not n:
            return CheckResult(
                "unknown",
                "unreadable_structure",
                f"could not count the residues of {name}: the normalisations cannot be compared.",
            )
        lengths.append(n)
    if lengths[0] == lengths[1]:
        return passed()
    implied = None
    if options.get("-a") in TRUE_WORDS:
        implied = "average"
    elif options.get("-u") or options.get("-L"):
        implied = "given_length"
    declared = call.declared.get(key)
    names = (
        f"structure_1 = {positional[0]} ({lengths[0]} residues), "
        f"structure_2 = {positional[1]} ({lengths[1]} residues)"
    )
    if declared in ("average", "given_length") and declared != implied:
        option = "-a T" if declared == "average" else "-u <length>"
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}={declared} but this call has no {option}, so TMalign prints only "
            f"the two chain-length normalisations ({names}).",
            expected=declared,
            got=implied or "structure_1/structure_2",
            fix=(f"add {option} (and read the matching line) or declare structure_1/structure_2",),
        )
    if declared is not None or implied is not None:
        return passed()
    return CheckResult(
        "fail",
        "tm_norm_undeclared",
        f"TMalign prints two TM-scores that differ here: {names}. The first line is normalised by "
        "structure_1, the second by structure_2; the value to quote is the one normalised by the "
        "reference structure.",
        expected=list(rule.declare.values),
        got=None,
        fix=(
            f"read the line 'normalized by length of Structure_2' if {positional[1]} is the "
            f"reference, then --declare {key}=structure_2",
            f"or --declare {key}=structure_1 if {positional[0]} is the reference "
            "(or use -a T for the average, -u <length> for a given one)",
        ),
    )
