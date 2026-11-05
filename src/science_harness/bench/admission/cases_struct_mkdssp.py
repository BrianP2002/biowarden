"""Admission cases for mkdssp.poly_seq_scheme (family F98: residues in alpha-helix, state H).

Workspaces are the benchmark's own (implicit tier): the clean twin is an archive-style mmCIF with
the residue tables, the faulted twin the same atoms without `_pdbx_poly_seq_scheme`. A copy of the
model in PDB format (`model.pdb`, written with gemmi) is added to each workspace for the
conversion route. Every call is a real mkdssp run in the workspace; the helix count of its classic
DSSP output is compared with `task.truth` (bench.score.answer_matches); labels never come from the
rule.

  mkdssp model.cif out.dssp (also with ./ paths, --quiet, -v, --output-format dssp and an output
  name that is not .dssp), and the same on model.pdb (the converted route: right on both twins).
Declarations: mmcif_complete for the calls on the mmCIF on the clean twin, pdb_converted for the
calls on the PDB file; the wrong calls on the faulted twin carry both.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.bench.tasks.extra.f98_mkdssp_scheme import dssp_helix_count
from science_harness.engine import Call

KEY = "mkdssp_input"
CALLS = {
    "cif": ("model.cif", "out.dssp", ()),
    "cif_dot_slash": ("./model.cif", "./out.dssp", ()),
    "cif_quiet": ("model.cif", "out.dssp", ("--quiet",)),
    "cif_verbose": ("model.cif", "out.dssp", ("-v",)),
    "cif_format_flag": ("model.cif", "out.txt", ("--output-format", "dssp")),
    "pdb": ("model.pdb", "out.dssp", ()),
    "pdb_quiet": ("model.pdb", "out.dssp", ("--quiet",)),
}


@case_source("mkdssp.poly_seq_scheme")
def poly_seq_scheme_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    import gemmi

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F98", seed, twin, work, tools, "implicit", "free")
        gemmi.read_structure(str(work / "model.cif")).write_pdb(str(work / "model.pdb"))
        for name, (source, target, flags) in CALLS.items():
            argv = (*flags, source, target)
            (work / "out.dssp").unlink(missing_ok=True)
            (work / "out.txt").unlink(missing_ok=True)
            proc = tools.run(["mkdssp", *argv], work)
            produced = work / target.removeprefix("./")
            ok = proc.returncode == 0 and produced.exists()
            count = dssp_helix_count(produced.read_text()) if ok else None
            label: Label = "right" if ok and answer_matches(task.truth, count) else "wrong"
            uses = "pdb_converted" if source.endswith(".pdb") else "mmcif_complete"
            out.append(Case(Call("mkdssp", argv, work), label, False, "F98", seed, twin, name))
            needed = (
                uses
                if label == "right"
                else ("mmcif_complete" if twin == "clean" else "pdb_converted")
            )
            for value in {needed, uses}:
                declared = Call("mkdssp", argv, work, {KEY: value})
                out.append(Case(declared, label, True, "F98", seed, twin, name))
    return out
