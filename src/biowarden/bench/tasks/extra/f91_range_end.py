"""F91 range_end: in a ProDy selection `resnum 20:31` the end is excluded, `resnum 20 to 31` is not.

Source: the ProDy selection documentation (`prody.atomic.select`, "Number ranges": a range of
numbers using 'to' or Python style slicing with ':'), verified here on ProDy 2.6.1 with a chain
numbered 1-22 that contains residue 12 and its insertion-code partner 12A:
  select('resnum 10 to 15 and name CA')   7 atoms, residues 10, 11, 12, 12A, 13, 14, 15
  select('resnum 10:15 and name CA')      6 atoms, the same without residue 15
MDAnalysis `resid 10:15` and `resid 10-15` include 15 (7 atoms) and mdtraj has no ':' range, so
the same-looking selection string means different things in the three libraries. Exit 0 and no
warning (ProDy prints "@> N atoms ... parsed" on stderr only).

Twins: a 36-44 residue chain with two to four cysteines inside the asked range (the answer, drawn
per seed) and two outside. Faulted: one of them is the LAST residue of the range, so a Python-style
`resnum a:b` drops it and counts one cysteine too few. Clean: the last residue of the range is not
modelled (absent from the file, as unmodelled residues are) and all the cysteines sit in its
interior, so both spellings select the same atoms. The answer is the same in both twins.

  explicit  the README states that residue numbers are the PDB residue numbers of chain A
  implicit  the README only describes the file
  forced    the task sentence dictates ProDy
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f89_chain_ids import (
    AA3,
    AA_LETTERS,
    TOOLS_NOTE,
    Chain,
    make_residues,
    pdb_text,
    python_struct,
    random_fold,
)
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"protein.pdb", "README.md"})
LETTERS = AA_LETTERS.replace("C", "").replace("G", "")


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f91-{seed}")
    n = rng.randint(36, 44)
    start = rng.randint(1, 15)
    a = start + rng.randint(3, 6)
    b = a + rng.randint(7, 10)
    seq = [rng.choice(LETTERS) for _ in range(n)]
    inner = range(a + 1, b - 1)  # positions strictly inside the range
    interior = rng.sample(inner, 2)
    interior += rng.sample([x for x in inner if x not in interior], rng.randint(0, 2))  # 2-4 in all
    outside = [a - 2, b + 3]
    return {
        "n": n,
        "start": start,
        "a": a,
        "b": b,
        "seq": seq,
        "interior": interior,
        "out": outside,
    }


def sequence(plan: dict[str, Any], twin: str) -> str:
    seq = list(plan["seq"])
    start = plan["start"]
    inside = list(plan["interior"])
    if twin == "faulted":
        inside[1] = plan["b"]  # the second cysteine is the last residue of the range
    for pos in (*inside, *plan["out"]):
        seq[pos - start] = "C"
    return "".join(seq)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    fold = random_fold(random.Random(f"f91-fold-{seed}"), plan["n"])
    residues = make_residues(random.Random(0), sequence(plan, twin), plan["start"], fold=fold)
    if twin == "clean":  # the last residue of the range is not modelled
        residues = [r for r in residues if r.num != plan["b"]]
    (work / "protein.pdb").write_text(pdb_text([Chain("A", residues)]))
    line = "protein.pdb: one protein chain (A), PDB format, backbone and CB atoms only."
    if tier == "explicit":
        line += " Residue numbers are the PDB residue numbers of chain A."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    a, b = plan["a"], plan["b"]
    if twin == "clean":
        evidence = (f"residue {b}, the last of the range {a}-{b}, is not in the file",)
    else:
        evidence = (f"residue {b}, the last of the range {a}-{b}, is a cysteine",)
    prompt = common.prompt_for(
        route,
        f"count the cysteine residues among residues {a} to {b}, both ends included, of "
        "protein.pdb (PDB residue numbers)",
        f"use ProDy to count the cysteine residues among residues {a} to {b}, both ends "
        "included, of protein.pdb (PDB residue numbers)",
        "a JSON integer",
    )
    found = len(plan["interior"])
    truth = {"kind": "int", "expected": found, "naive": found if twin == "clean" else found - 1}
    task_id = common.task_id("F91", seed, twin, tier, route)
    return Task(
        task_id, "F91", "interval_closure", twin, seed, prompt, truth, evidence, tier, route
    )


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def query_range(task: Task) -> tuple[int, int]:
    m = re.search(r"residues (\d+) to (\d+)", task.prompt)
    assert m
    return int(m.group(1)), int(m.group(2))


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    a, b = query_range(task)
    count = sum(
        1
        for line in (work / "protein.pdb").read_text().splitlines()
        if line.startswith("ATOM")
        and line[12:16].strip() == "CA"
        and line[17:20] == AA3["C"]
        and a <= int(line[22:26]) <= b
    )
    return _result(count, "residues a..b inclusive from the PDB records")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    a, b = query_range(task)
    code = (
        "import prody; ag = prody.parsePDB('protein.pdb'); "
        f"sel = ag.select('resnum {a}:{b} and name CA and resname CYS'); "
        "print(0 if sel is None else sel.numAtoms())"
    )
    proc = python_struct(tools, work, code)
    return _result(int(proc.stdout.strip().splitlines()[-1]), "ProDy select resnum a:b")


# Exposure: a ProDy-style selection with a `:` range on a residue attribute. The end bound cannot
# be compared with the task here, so `resnum 20:32` (a correct +1 by hand) counts as exposed too;
# `resnum 20 to 31` and other libraries' inclusive ranges do not. A bound is a number or a format
# placeholder (`%d`, `{a}`, `{}`, `$a`), so `resnum %d:%d` and f-strings are seen. Judged per
# command.
BOUND = r"(?:-?\d+|%[-+0-9.]*[dsi]|\{[^{}]*\}|\$\{?\w+\}?)"
SLICE = re.compile(rf"\b(?:resnum|resid|resindex|serial|index)\s+{BOUND}\s*:\s*{BOUND}")


def exposed(commands: list[str], work: Path) -> bool:
    return any(SLICE.search(c) and "protein.pdb" in c and "MDAnalysis" not in c for c in commands)


SPEC = FamilySpec(
    id="F91",
    slot="interval_closure",
    source="ProDy selection docs (number ranges)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
