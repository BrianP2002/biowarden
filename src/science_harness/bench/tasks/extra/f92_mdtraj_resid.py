"""F92 mdtraj_resid: in an mdtraj atom selection `resid N` is the 0-based residue INDEX, not the PDB
residue number (that is `resSeq N`).

Source: the mdtraj selection language (`Topology.select`), verified here on mdtraj 1.10.3 with a
22-residue chain numbered 19..44 with a gap: `top.select("resid 25 and name CA")` returns no atom
(index 25 does not exist), `top.select("resSeq 25 and name CA")` returns the CA of residue 25 (index
6). With a chain numbered from 1 and a query inside the chain, `resid N` silently returns the CA of
PDB residue N+1. Distances come back in nanometres (`md.compute_distances`, `xyz`), not angstroms
(1.0665 against 10.665 for the same pair read with gemmi). Exit 0, nothing on stderr.

Twins: the same coordinates for residues 1..N. Clean: the file also has an N-terminal residue
numbered 0 (index 0), so index and number coincide and `resid N` happens to be residue N. Faulted:
the chain starts at residue 1, so `resid N` is residue N+1.

  explicit  the README says residue numbers are the PDB residue numbers (and where they start)
  implicit  the README only describes the file
  forced    the task sentence dictates mdtraj

The distance is asked in angstroms, the unit of the PDB coordinates, while mdtraj answers in
nanometres: a unit slip would fail both twins, so the README of both tiers states the unit.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f89_chain_ids import (
    TOOLS_NOTE,
    Chain,
    make_residues,
    pdb_text,
    python_struct,
    random_fold,
    random_sequence,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"protein.pdb", "README.md"})


def make_plan(seed: int) -> dict[str, Any]:
    """Residues p < q far enough apart that the pair shifted by one residue has another distance
    (in a regular helix two residues one turn apart keep their distance when both move on)."""
    rng = random.Random(f"f92-{seed}")
    n = rng.randint(30, 40)  # residues 1..n; the clean twin adds residue 0
    res = residues({"n": n}, seed)
    while True:
        p = rng.randint(4, 12)
        q = p + rng.randint(8, 16)
        if q + 1 <= n and abs(ca_distance(res, p, q) - ca_distance(res, p + 1, q + 1)) >= 1.0:
            return {"n": n, "p": p, "q": q}


def residues(plan: dict[str, Any], seed: int):
    rng = random.Random(f"f92-seq-{seed}")
    n = plan["n"]
    fold = random_fold(random.Random(f"f92-fold-{seed}"), n + 1)
    return make_residues(rng, random_sequence(rng, n + 1), start=0, fold=fold)  # numbers 0..n


def ca_distance(res, p: int, q: int) -> float:
    by_num = {r.num: next(a.xyz for a in r.atoms if a.name == "CA") for r in res}
    return math.dist(by_num[p], by_num[q])


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    res = residues(plan, seed)
    kept = res if twin == "clean" else res[1:]
    (work / "protein.pdb").write_text(pdb_text([Chain("A", kept)]))
    first = kept[0].num
    line = f"protein.pdb: one protein chain (A), PDB format. The first residue is numbered {first}."
    if tier == "explicit":
        line += " Residue numbers are the PDB residue numbers of the file."
    unit = "Coordinates are in angstroms, and so is the distance to report."
    (work / "README.md").write_text(common.readme([line, unit, TOOLS_NOTE]))
    p, q = plan["p"], plan["q"]
    if twin == "clean":
        evidence = ("the first residue is numbered 0, so mdtraj's residue index equals the number",)
    else:
        evidence = ("the first residue is numbered 1, so mdtraj's 0-based index is the number - 1",)
    prompt = common.prompt_for(
        route,
        f"report the distance in angstroms between the CA atoms of residues {p} and {q} "
        "(PDB residue numbers) of protein.pdb",
        f"use mdtraj to report the distance in angstroms between the CA atoms of residues {p} and "
        f"{q} (PDB residue numbers) of protein.pdb",
        "a JSON number",
    )
    expected = round(ca_distance(res, p, q), 3)
    wrong = round(ca_distance(res, p + 1, q + 1), 3)  # resid N in the faulted twin is number N+1
    truth = {
        "kind": "float",
        "expected": expected,
        "naive": expected if twin == "clean" else wrong,
        "tol": 0.05,
    }
    task_id = common.task_id("F92", seed, twin, tier, route)
    return Task(task_id, "F92", "index_base", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def query_residues(task: Task) -> tuple[int, int]:
    m = re.search(r"residues (\d+) and (\d+)", task.prompt)
    assert m
    return int(m.group(1)), int(m.group(2))


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    p, q = query_residues(task)
    ca = {}
    for line in (work / "protein.pdb").read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca[int(line[22:26])] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return _result(round(math.dist(ca[p], ca[q]), 3), "CA coordinates by PDB residue number")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    p, q = query_residues(task)
    code = (
        "import mdtraj as md; t = md.load_pdb('protein.pdb'); "
        f"a = t.topology.select('resid {p} and name CA')[0]; "
        f"b = t.topology.select('resid {q} and name CA')[0]; "
        "print(float(md.compute_distances(t, [[a, b]])[0, 0]) * 10)"
    )
    proc = python_struct(tools, work, code)
    return _result(round(float(proc.stdout.strip().splitlines()[-1]), 3), "mdtraj resid N")


# Exposure: an mdtraj selection or lookup by 0-based index where a PDB number was meant:
# `resid N` in a selection string, or `topology.residue(N)`. The number may be a literal or a
# format placeholder (`resid %d`, `resid {n}`, `resid {}`, `resid $n`) and the lookup argument a
# variable or an expression (`topology.residue(p)`, `residue(p - 1)`). `resSeq` is the safe route.
PLACEHOLDER = r"%[-+0-9.]*[dsi]|\{[^{}]*\}|\$\{?\w+\}?"
INDEXED = re.compile(rf"\bresid\s+(?:\d+|{PLACEHOLDER})|\.residue\(\s*[\w(+-]")
LIB = re.compile(r"\bmdtraj\b|\bmd\.load")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        "protein.pdb" in c and LIB.search(c) and INDEXED.search(c) and "resSeq" not in c
        for c in commands
    )


SPEC = FamilySpec(
    id="F92",
    slot="index_base",
    source="mdtraj selection language (resid is the residue index)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
