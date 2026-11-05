"""F98 mkdssp_scheme: mkdssp 4 reads the residues of an mmCIF from `_pdbx_poly_seq_scheme`, and
returns a valid, empty result when the table is missing.

Source: libcifpp / mkdssp 4.5.3 behaviour, verified here. mkdssp's mmCIF input is read through
libcifpp; the same chain given as PDB gives 54 residues, and as the mmCIF mkdssp wrote itself
(`mkdssp ref.pdb ref_mk.cif`, then `mkdssp ref_mk.cif x.dssp`) also 54. Removing ONE category from
that mmCIF changes the answer: without `_pdbx_poly_seq_scheme` the DSSP header reads
`0  1  0 ... TOTAL NUMBER OF RESIDUES, NUMBER OF CHAINS` and the residue table is empty
(`mkdssp -v` prints "Considering 0 pairs of residues"); removing `_atom_type`, `_chem_comp`,
`_entity`, `_entity_poly`, `_entity_poly_seq`, `_struct_asym`, `_struct_conf`, `_software` or
`_audit_conform` one at a time leaves 54. Exit 0 and nothing on stderr either way. mmCIF files
written by gemmi (`make_mmcif_document`), by Biopython (`MMCIFIO`) and by pdb-tools (`pdb_tocif`)
do not contain the table, and mkdssp reports 0 residues for all three (tested on the same 54-residue
chain); gemmi, ProDy and Biopython read every one of those files without complaint. The mkdssp
output format follows the extension of the output file: `.dssp` is classic DSSP, anything else
(`.txt`, `.out`, `.cif`, no file at all) is mmCIF.

Twins: the same model chain. Clean: `model.cif` is a complete archive-style mmCIF with
`_entity_poly_seq`, `_struct_asym` and `_pdbx_poly_seq_scheme`. Faulted: the same atoms and entity
tables, without the three residue tables (what a prediction or conversion pipeline writes). The
answer is the number of residues DSSP assigns to alpha-helix (H), taken from the clean twin.

  explicit  the README says how the file was written (with or without the residue tables)
  implicit  the README only says "mmCIF model"
  forced    the task sentence dictates mkdssp
"""

from __future__ import annotations

import functools
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f89_chain_ids import (
    TOOLS_NOTE,
    Chain,
    make_residues,
    mmcif_text,
    random_fold,
    random_sequence,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.cif", "README.md"})


@functools.lru_cache(maxsize=64)
def make_chain(seed: int) -> Chain:
    rng = random.Random(f"f98-{seed}")
    n = rng.randint(46, 60)
    helix = (
        [-57.0 + rng.gauss(0, 4) for _ in range(12)],
        [-47.0 + rng.gauss(0, 4) for _ in range(12)],
    )
    rest = random_fold(rng, n - 12)
    fold = (helix[0] + rest[0], helix[1] + rest[1])
    return Chain("A", make_residues(rng, random_sequence(rng, n), 1, fold=fold))


def dssp_helix_count(text: str) -> int:
    """Residues with state H in a classic DSSP file (structure column 17 of the residue table)."""
    rows = text.splitlines()
    start = next(i for i, line in enumerate(rows) if line.startswith("  #  RESIDUE"))
    return sum(1 for line in rows[start + 1 :] if len(line) > 16 and line[16] == "H")


def run_dssp(tools: ToolEnv, name: str, text: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        (cwd / name).write_text(text)
        proc = tools.run(["mkdssp", name, "out.dssp"], cwd)
        if proc.returncode != 0 or not (cwd / "out.dssp").exists():
            raise RuntimeError(proc.stderr)
        return dssp_helix_count((cwd / "out.dssp").read_text())


@functools.lru_cache(maxsize=64)
def expected_helix(seed: int, tools: ToolEnv) -> int:
    """H residues of the complete mmCIF (the form mkdssp reads fully)."""
    return run_dssp(tools, "model.cif", mmcif_text([make_chain(seed)], "model", poly_scheme=True))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    chain = make_chain(seed)
    text = mmcif_text([chain], "model", poly_scheme=(twin == "clean"))
    (work / "model.cif").write_text(text)
    line = "model.cif: protein model, one chain (A), mmCIF."
    if tier == "explicit":
        if twin == "clean":
            line += (
                " Archive-style file: it has the entity and residue tables (_pdbx_poly_seq_scheme)."
            )
        else:
            line += " Written by a modelling script: atom_site and entity tables only."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = ("model.cif has _entity_poly_seq, _struct_asym and _pdbx_poly_seq_scheme",)
    else:
        evidence = ("model.cif has no _pdbx_poly_seq_scheme table (and no _entity_poly_seq)",)
    prompt = common.prompt_for(
        route,
        "count the residues of model.cif that DSSP assigns to alpha-helix (state H)",
        "use mkdssp to count the residues of model.cif that DSSP assigns to alpha-helix (state H)",
        "a JSON integer",
    )
    expected = expected_helix(seed, tools)
    truth = {"kind": "int", "expected": expected, "naive": expected if twin == "clean" else 0}
    task_id = common.task_id("F98", seed, twin, tier, route)
    return Task(task_id, "F98", "input_format", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    import gemmi

    st = gemmi.read_structure(str(work / "model.cif"))
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        st.write_pdb(str(cwd / "model.pdb"))
        tools.run(["mkdssp", "model.pdb", "out.dssp"], cwd)
        count = dssp_helix_count((cwd / "out.dssp").read_text())
    return _result(count, "mkdssp on a PDB written from the mmCIF by gemmi")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(
        run_dssp(tools, "model.cif", (work / "model.cif").read_text()), "mkdssp model.cif out.dssp"
    )


# Exposure: mkdssp (or the Biopython DSSP wrapper) run directly on the mmCIF. A command that also
# names a PDB file (a converted copy) took the other route. Judged per command.
RUN = re.compile(r"\bmkdssp\b|\bdssp\b|\bDSSP\(")


def exposed(commands: list[str], work: Path) -> bool:
    return any(RUN.search(c) and "model.cif" in c and ".pdb" not in c for c in commands)


SPEC = FamilySpec(
    id="F98",
    slot="input_format",
    source="mkdssp 4.5.3 / libcifpp (pdbx_poly_seq_scheme)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
