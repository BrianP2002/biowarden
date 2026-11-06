"""Rule mkdssp.poly_seq_scheme (slot input_format, family F98): mkdssp 4 reads the residues of an
mmCIF from `_pdbx_poly_seq_scheme` and returns a valid, empty result when the table is missing.

The trap (libcifpp / mkdssp 4.5.3 behaviour). Verified here on a 54-residue chain: the PDB file
gives 54 residues; the mmCIF that mkdssp itself writes (`mkdssp ref.pdb ref_mk.cif`) read back gives
54; remove ONE category from it and the answer changes: without `_pdbx_poly_seq_scheme` the DSSP
header reads `0  1  0 ... TOTAL NUMBER OF RESIDUES, NUMBER OF CHAINS` and the residue table is
empty; removing `_atom_type`, `_chem_comp`, `_entity`, `_entity_poly`, `_entity_poly_seq`,
`_struct_asym`, `_struct_conf`, `_software` or `_audit_conform` one at a time still gives 54.
mmCIF files written by gemmi (`make_mmcif_document`), Biopython (`MMCIFIO`) and pdb-tools
(`pdb_tocif`) do not contain the table: mkdssp gives 0 residues for all three, exit 0 and no stderr
(the file is complete as far as gemmi, ProDy and Biopython are concerned). A downstream count of
helix residues is then 0 rather than an error. The output format follows the extension of the
output file: `.dssp` is classic DSSP, any other name (or none) writes mmCIF.

What the rule does. A CALL rule on `mkdssp` (tier T1). It takes the input file (the first argument
that is not an option or an option value), and only a .cif/.mmcif input (optionally .gz) is
analysed: the file is parsed with gemmi and passes when it has a non-empty `_pdbx_poly_seq_scheme`
loop. A PDB input passes (mkdssp builds the table itself).
  mmCIF without the table, no declaration              -> fail (poly_seq_scheme_missing)
  mmCIF without the table, declared mmcif_complete     -> fail (declared_mismatch)
  mmCIF without the table, declared pdb_converted      -> fail (declared_mismatch: still a .cif)
  mmCIF with the table, or a PDB file                  -> pass
`unknown`: an mmCIF gemmi cannot parse. Declaration `mkdssp_input` (optional): mmcif_complete |
pdb_converted. Fix: convert with gemmi (`gemmi.read_structure(f).write_pdb('model.pdb')`) and run
mkdssp on the PDB file, then check that the DSSP header lists the expected number of residues.

Not seen: the Biopython `DSSP(...)` wrapper (it calls mkdssp itself), input piped on stdin, and
the `dssp` executable name.
"""

from __future__ import annotations

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.struct_common import cif_block, is_cif, positionals

VALUE_OPTIONS = frozenset({"--output-format", "--min-pp-stretch", "--mmcif-dictionary"})


@checker("mkdssp_poly_seq_scheme")
def mkdssp_poly_seq_scheme(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    args = positionals(call.argv, VALUE_OPTIONS)
    if not args or not is_cif(args[0]):
        return passed()
    path = ctx.resolve(args[0])
    if not path.is_file():
        return passed()
    block = cif_block(ctx, path)
    if block is None:
        return CheckResult(
            "unknown",
            "unreadable_mmcif",
            f"could not parse {args[0]} as mmCIF: its tables are unknown.",
        )
    table = block.find_mmcif_category("_pdbx_poly_seq_scheme")
    if table is not None and len(table) > 0:
        return passed()
    declared = call.declared.get(key)
    facts = (
        f"{args[0]} has no _pdbx_poly_seq_scheme table, which mkdssp 4 reads its residues from, "
        "so it reports 0 residues and an empty (but valid) DSSP result without an error."
    )
    fix = (
        f"convert first: python-struct -c \"import gemmi; gemmi.read_structure('{args[0]}')"
        ".write_pdb('model.pdb')\" and run mkdssp on model.pdb",
    )
    if declared is not None:
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}={declared} but {facts}",
            expected="a PDB input or an mmCIF with the table",
            got=declared,
            fix=fix,
        )
    return CheckResult(
        "fail",
        "poly_seq_scheme_missing",
        facts,
        expected=list(rule.declare.values),
        got=None,
        fix=(*fix, f"then --declare {key}=pdb_converted"),
    )
