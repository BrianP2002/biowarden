"""F99 mda_protein: MDAnalysis's `protein` keyword is a fixed residue-name list that leaves out
phosphorylated and other modified residues.

Source: `MDAnalysis.core.selection.ProteinSelection.prot_res` (MDAnalysis 2.9.0), checked against
mdtraj 1.10.3, ProDy 2.6.1, gemmi 0.7.5 and Biopython 1.88 on a 25-residue chain with MSE at
residue 6, SEP at residue 15 and one water:

  MDAnalysis  select_atoms("protein").n_residues   23   (SEP is not in the list)
  mdtraj      select("protein")                    24
  ProDy       select("protein")                    24
  gemmi       residues with is_amino_acid()        24
  Biopython   is_aa(residue)                       24

The MDAnalysis list holds MSE, HYP, HID, HIE, CYX, ASH, GLH, NME and ACE but not SEP, TPO, PTR,
CSO or MLY; the mdtraj list holds SEP, TPO, PTR, MSE, CSO, HYP, MLY, GLH, NME, ACE. Exit 0 and no
warning (the PDB reader's "Unit cell dimensions not found" warning is the only noise).

Twins: the same chain. Clean: serines and threonines are ordinary ATOM records. Faulted: two or
three of them are phosphorylated (SEP and TPO HETATM records, the same backbone and CB atoms), a
common modification in kinase structures. Both twins carry a few crystal waters, so the number of
protein residues is not the number of residues in the file. The answer is the number of amino-acid
residues of chain A (phosphorylated ones included, waters excluded).

  explicit  the README says phosphorylated residues are part of the protein
  implicit  the prompt only asks for the protein residues; the HETATM records are in the file
  forced    the task sentence dictates MDAnalysis
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f89_chain_ids import (
    AA_LETTERS,
    TOOLS_NOTE,
    Atom,
    Chain,
    Residue,
    make_residues,
    pdb_text,
    random_fold,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.pdb", "README.md"})
PHOSPHO = {"S": "SEP", "T": "TPO"}
LETTERS = AA_LETTERS.replace("S", "").replace("T", "")


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f99-{seed}")
    n = rng.randint(30, 44)
    seq = [rng.choice(LETTERS) for _ in range(n)]
    sites = rng.sample(range(2, n - 2), rng.randint(5, 7))  # serines and threonines
    for i in sites:
        seq[i] = rng.choice("ST")
    phospho = sites[: rng.randint(2, 3)]
    waters = [(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6)) for _ in range(3)]
    return {
        "n": n,
        "seq": "".join(seq),
        "phospho": phospho,
        "waters": waters,
        "fold": random_fold(rng, n),
    }


def pdb_file(plan: dict[str, Any], twin: str) -> str:
    residues = make_residues(random.Random(0), plan["seq"], 1, fold=plan["fold"])
    if twin == "faulted":
        for i in plan["phospho"]:
            residues[i].name = PHOSPHO[plan["seq"][i]]
            residues[i].het = True
    waters = [
        Residue("HOH", 101 + k, [Atom("O", tuple(round(c, 3) for c in xyz))], het=True)
        for k, xyz in enumerate(plan["waters"])
    ]
    return pdb_text([Chain("A", residues), Chain("A", waters, polymer=False)])


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "model.pdb").write_text(pdb_file(plan, twin))
    line = (
        "model.pdb: kinase fragment, one protein chain (A) with a few crystal waters, PDB format."
    )
    if tier == "explicit":
        line += " Phosphorylated residues (SEP, TPO) are part of the protein."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = ("every protein residue is an ATOM record; the HETATM records are waters",)
    else:
        where = ", ".join(str(i + 1) for i in plan["phospho"])
        evidence = (f"residues {where} are SEP/TPO HETATM records (phosphorylated)",)
    prompt = common.prompt_for(
        route,
        "report how many amino-acid residues the protein chain of model.pdb has (waters are not "
        "residues of the protein)",
        "use MDAnalysis to report how many amino-acid residues the protein chain of model.pdb "
        "has (waters are not residues of the protein)",
        "a JSON integer",
    )
    n = plan["n"]
    truth = {
        "kind": "int",
        "expected": n,
        "naive": n if twin == "clean" else n - len(plan["phospho"]),
    }
    task_id = common.task_id("F99", seed, twin, tier, route)
    return Task(task_id, "F99", "selection_scope", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = sum(
        1
        for line in (work / "model.pdb").read_text().splitlines()
        if line[:6] in ("ATOM  ", "HETATM") and line[12:16].strip() == "CA" and line[17:20] != "HOH"
    )
    return _result(n, "CA records of residues other than water")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = (
        "import warnings; warnings.simplefilter('ignore'); import MDAnalysis as mda; "
        "u = mda.Universe('model.pdb'); print(u.select_atoms('protein').n_residues)"
    )
    proc = tools.run(["python-struct", "-s", "-c", code], work)
    return _result(int(proc.stdout.strip().splitlines()[-1]), "MDAnalysis select_atoms('protein')")


# Exposure: an MDAnalysis selection by the `protein` or `backbone` keyword (fixed residue-name
# list), or a residue count over the ATOM records only (`grep ^ATOM | awk '$3=="CA"' | wc -l`: SEP
# and TPO are HETATM records, so that count is short by the same residues; counting the atoms of
# the file is not a residue count). Selections by residue name, by atom name alone (`name CA`),
# `not resname HOH` or any mention of SEP/TPO/HETATM count as handled. Judged per command.
LIB = re.compile(r"\bMDAnalysis\b|\bmda\.")
KEYWORD = re.compile(r"""select_atoms\(\s*['"][^'"]*\b(?:protein|backbone|nucleic)\b""")
ATOM_ONLY = re.compile(r"""\^ATOM|\$1\s*==\s*['"]ATOM['"]|startswith\(\s*['"]ATOM""")
RESIDUES = re.compile(r"\bCA\b|\buniq\b|\bsort\s+-\w*u|\bset\(|\$6\b|\[22:26\]|\[22:27\]")
HANDLED = re.compile(r"\bSEP\b|\bTPO\b|\bPTR\b|HETATM|is_amino_acid|\bor\s+(?:resname|name)\b")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        "model.pdb" in c
        and not HANDLED.search(c)
        and ((LIB.search(c) and KEYWORD.search(c)) or (ATOM_ONLY.search(c) and RESIDUES.search(c)))
        for c in commands
    )


SPEC = FamilySpec(
    id="F99",
    slot="selection_scope",
    source="MDAnalysis selection keywords (protein)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
