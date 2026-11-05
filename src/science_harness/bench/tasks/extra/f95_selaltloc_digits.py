"""F95 selaltloc_digits: `pdb_selaltloc` drops atoms of an altloc residue numbered 9, 99, 999 (and
the residue after a fully split one), because it compares residue numbers as strings.

Source R2-STR-16 (pdb-tools issue 153: residues whose atoms all carry alternate locations vanish;
maintainers call it a bug) and the code of pdb-tools 2.7.0 (`pdb_selaltloc.is_another_altloc_group`
tests `ru1 > ru0` on the stripped residue-number STRINGS, so '10' > '9' is False and the group
boundary between residue 9 and residue 10 is not seen: the two residues are flushed together and
the lower-occupancy-or-not logic keeps only one residue's lines).

Verified here (pdb-tools 2.7.0, a 110-residue chain, ATOM lines N, CA, C, O, CB per residue,
`pdb_selaltloc model.pdb`, exit 0, empty stderr), expected = every atom once = 550:
  CB alone carrying altlocs A 0.4 / B 0.6, at residue 3, 8, 10, 11, 19, 98, 100, 101   550 atoms
  CB alone at residue 9 or 99                                                          549 atoms
  every atom of the residue carrying altlocs, at residue 3, 8, 11, 19, 98, 101         550 atoms
  every atom of the residue at residue 9, 10, 99 or 100                                545 atoms
i.e. a whole residue (5 atoms) is missing from the output. gemmi, ProDy and Biopython keep all 550.

Twins: the same model; only the residue numbering moves. Faulted: the chain starts at 1, so the
fully split residue (index 8) is residue 9. Clean: the chain starts at 4, so it is residue 12.
Both twins also have side-chain altlocs elsewhere (never at a digit boundary). Task: the number of
atoms after keeping the highest-occupancy conformer of every atom.

  explicit  the README says that pdb_selaltloc keeps the highest-occupancy conformer
  implicit  the README only describes the file
  forced    the task sentence dictates pdb_selaltloc
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
    LETTERS_NO_G,
    TOOLS_NOTE,
    Atom,
    Residue,
    make_residues,
    pdb_atom_line,
    random_fold,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.pdb", "README.md"})
WHOLE = 8  # index of the fully split residue


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f95-{seed}")
    n = rng.randint(34, 44)
    seq = "".join(rng.choice(LETTERS_NO_G) for _ in range(n))
    side = rng.sample([i for i in range(14, n - 2)], 3)  # CB-only altlocs, never near 9/10 or 99
    return {"n": n, "seq": seq, "side": side, "fold": random_fold(rng, n), "seed": seed}


def residues(plan: dict[str, Any], start: int) -> list[Residue]:
    return make_residues(random.Random(0), plan["seq"], start, fold=plan["fold"])


def pdb_text(plan: dict[str, Any], twin: str) -> str:
    start = 1 if twin == "faulted" else 4
    lines: list[str] = []
    serial = 1
    for i, res in enumerate(residues(plan, start)):
        for atom in res.atoms:
            split = i == WHOLE or (i in plan["side"] and atom.name == "CB")
            if not split:
                lines.append(pdb_atom_line(serial, atom, res, "A"))
                serial += 1
                continue
            for label, occ, dx in (("A", 0.4, 0.0), ("B", 0.6, 0.5)):
                x, y, z = atom.xyz
                alt = Atom(atom.name, (x + dx, y, z), atom.element, occ, atom.bfac, label)
                lines.append(pdb_atom_line(serial, alt, res, "A"))
                serial += 1
    return "\n".join([*lines, "TER", "END"]) + "\n"


def atoms_once(plan: dict[str, Any]) -> int:
    return sum(len(r.atoms) for r in residues(plan, 1))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "model.pdb").write_text(pdb_text(plan, twin))
    first = 1 if twin == "faulted" else 4
    lines = [f"model.pdb: crystal structure, one protein chain (A), residues {first} onward."]
    if tier == "explicit":
        lines.append("pdb_selaltloc keeps the highest-occupancy conformer of each atom.")
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    split_num = first + WHOLE
    if twin == "faulted":
        evidence = (
            f"residue {split_num} (just before residue {split_num + 1}) has altlocs A and B",
        )
    else:
        evidence = (f"the residue with altlocs on every atom is residue {split_num}",)
    prompt = common.prompt_for(
        route,
        "keep only the highest-occupancy conformer of every atom of model.pdb and report how "
        "many atoms the resulting structure has",
        "use `pdb_selaltloc` to keep only the highest-occupancy conformer of every atom of "
        "model.pdb and report how many atoms the resulting structure has",
        "a JSON integer",
    )
    expected = atoms_once(plan)
    truth = {
        "kind": "int",
        "expected": expected,
        "naive": expected if twin == "clean" else expected - len(residues(plan, 1)[WHOLE].atoms),
    }
    task_id = common.task_id("F95", seed, twin, tier, route)
    return Task(task_id, "F95", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    seen = {
        (line[22:27], line[12:16])
        for line in (work / "model.pdb").read_text().splitlines()
        if line.startswith("ATOM")
    }
    return _result(len(seen), "distinct (residue, atom name) pairs: one conformer per atom")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["bash", "-c", "pdb_selaltloc model.pdb | grep -c '^ATOM'"], work)
    return _result(int(proc.stdout.strip()), "pdb_selaltloc model.pdb | grep -c ^ATOM")


# Exposure: any run of pdb_selaltloc over model.pdb. The bug depends on the residue numbers in the
# file, not on options, so every use is the trapped route; a Python/gemmi/awk route is not.
SELALTLOC = re.compile(r"\bpdb_selaltloc\b")


def exposed(commands: list[str], work: Path) -> bool:
    return any(SELALTLOC.search(c) and "model.pdb" in c for c in commands)


SPEC = FamilySpec(
    id="F95",
    slot="record_count",
    source="R2-STR-16",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
