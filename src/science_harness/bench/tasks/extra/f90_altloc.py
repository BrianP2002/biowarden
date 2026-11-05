"""F90 altloc: which alternate location a structure library keeps by default.

Source R2-STR-05 (ProteinMPNN and RFdiffusion keep different conformers) and the library defaults
verified here (ProDy 2.6.1, gemmi 0.7.5, Biopython 1.88, pdb-tools 2.7.0) on a PDB whose residues
12 and 30 carry alternate locations A (occupancy 0.35) and B (occupancy 0.65) on every atom:
  prody.parsePDB(f)                          keeps altloc A (docs: `altloc='A'` is the default),
                                             so CA of residue 12 is the 0.35 conformer;
  gemmi Structure.remove_alternative_conformations()   keeps the first conformer listed (A);
  gemmi without it                           holds both conformers (97 atoms for 88 listed once);
  Bio.PDB.PDBParser                          a DisorderedAtom whose selected child is the
                                             HIGHEST-occupancy atom (B, 0.65);
  pdb_selaltloc                              the highest occupancy, unless it hits the digit-count
                                             boundary bug of F95 (a fully split residue numbered 9,
                                             10, 99 or 100 loses its atoms): no altloc residue of
                                             this family carries one of those numbers.
Everything exits 0 with an empty stderr (ProDy prints one "@> ... parsed" line on stderr).

Twins: the same two conformers per residue; only the labels differ. Clean: A is the major
conformer (occupancy 0.55-0.75), so the first-listed conformer is also the right one. Faulted:
B is the major conformer, A the minor one, so a library that keeps the first conformer measures
the distance between the minor conformers. Task: the CA-CA distance of two residues that both
carry alternate locations (their conformers move along the line joining them, so the two
conformer pairs differ by 2.4-3.6 A).

  explicit  the README says to use the conformer with the higher occupancy
  implicit  the prompt only says "main conformation"; the occupancies are in the file
  forced    the task sentence dictates ProDy
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
    Atom,
    make_residues,
    pdb_atom_line,
    python_struct,
    random_sequence,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.pdb", "README.md"})
TOL = 0.05
DIGIT_BUG = frozenset({9, 10, 99, 100})  # numbers at which pdb_selaltloc loses a split residue


def _unit(a: tuple[float, float, float], b: tuple[float, float, float]) -> list[float]:
    d = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    return [x / n for x in d]


def make_model(seed: int) -> dict[str, Any]:
    """Residues, the altloc residues with their displacement, the two query residues."""
    rng = random.Random(f"f90-{seed}")
    n = rng.randint(36, 46)
    residues = make_residues(rng, random_sequence(rng, n))
    while True:  # a redraw only when a number hits the digit bug: the other seeds keep their data
        p = rng.randint(6, 12)
        q = p + rng.randint(8, 14)
        if not {p, q} & DIGIT_BUG:
            break
    ca = {r.num: next(a.xyz for a in r.atoms if a.name == "CA") for r in residues}
    shift = {}
    for num, other in ((p, q), (q, p)):
        shift[num] = [x * rng.uniform(0.6, 0.9) for x in _unit(ca[num], ca[other])]
    pool = [r.num for r in residues if r.num not in (p, q) and 3 < r.num < n - 3]
    extra = rng.sample(pool, 2)
    while set(extra) & DIGIT_BUG:
        extra = rng.sample(pool, 2)
    for num in extra:
        shift[num] = [rng.uniform(-0.7, 0.7) for _ in range(3)]
    occ_major = {num: round(rng.uniform(0.55, 0.75), 2) for num in shift}
    return {"residues": residues, "p": p, "q": q, "shift": shift, "major": occ_major}


def conformer_xyz(xyz, shift, sign: int):
    return tuple(xyz[i] + sign * shift[i] for i in range(3))


def pdb_text(model: dict[str, Any], twin: str) -> str:
    """Major conformer = +shift. Clean: major is A. Faulted: major is B (labels swapped)."""
    major_label, minor_label = ("A", "B") if twin == "clean" else ("B", "A")
    lines: list[str] = []
    serial = 1
    for res in model["residues"]:
        for atom in res.atoms:
            if res.num not in model["shift"]:
                lines.append(pdb_atom_line(serial, atom, res, "A"))
                serial += 1
                continue
            occ = model["major"][res.num]
            confs = [
                (major_label, occ, conformer_xyz(atom.xyz, model["shift"][res.num], 1)),
                (
                    minor_label,
                    round(1 - occ, 2),
                    conformer_xyz(atom.xyz, model["shift"][res.num], -1),
                ),
            ]
            for label, o, xyz in sorted(confs, key=lambda c: c[0]):
                alt = Atom(atom.name, xyz, atom.element, o, atom.bfac, label)
                lines.append(pdb_atom_line(serial, alt, res, "A"))
                serial += 1
    return "\n".join([*lines, "TER", "END"]) + "\n"


def distances(model: dict[str, Any]) -> tuple[float, float]:
    """(major-major, minor-minor) CA-CA distance."""
    ca = {r.num: next(a.xyz for a in r.atoms if a.name == "CA") for r in model["residues"]}

    def d(sign: int) -> float:
        a = conformer_xyz(ca[model["p"]], model["shift"][model["p"]], sign)
        b = conformer_xyz(ca[model["q"]], model["shift"][model["q"]], sign)
        return math.dist(a, b)

    return d(1), d(-1)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    model = make_model(seed)
    (work / "model.pdb").write_text(pdb_text(model, twin))
    line = "model.pdb: crystal structure, one protein chain (A), PDB format."
    if tier == "explicit":
        line += (
            " Where an atom has alternate locations, use the conformer with the higher occupancy."
        )
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    p, q = model["p"], model["q"]
    occ = model["major"][p]
    if twin == "clean":
        evidence = (f"altloc A is the major conformer ({occ}) of residues {p} and {q}",)
    else:
        evidence = (
            f"altloc B is the major conformer ({occ}) of residues {p} and {q}; "
            f"A has {round(1 - occ, 2)}",
        )
    prompt = common.prompt_for(
        route,
        f"report the distance in angstroms between the CA atoms of residues {p} and {q} of the "
        "crystal structure model.pdb (its main conformation)",
        f"use ProDy to report the distance in angstroms between the CA atoms of residues {p} and "
        f"{q} of the crystal structure model.pdb (its main conformation)",
        "a JSON number",
    )
    major, minor = distances(model)
    truth = {
        "kind": "float",
        "expected": round(major, 3),
        "naive": round(major if twin == "clean" else minor, 3),
        "tol": TOL,
    }
    task_id = common.task_id("F90", seed, twin, tier, route)
    return Task(task_id, "F90", "altloc_policy", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def query_residues(task: Task) -> tuple[int, int]:
    m = re.search(r"residues (\d+) and (\d+)", task.prompt)
    assert m
    return int(m.group(1)), int(m.group(2))


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    p, q = query_residues(task)
    best: dict[int, tuple[float, tuple[float, float, float]]] = {}
    for line in (work / "model.pdb").read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and int(line[22:26]) in (p, q):
            occ = float(line[54:60])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            num = int(line[22:26])
            if num not in best or occ > best[num][0]:
                best[num] = (occ, xyz)
    d = math.dist(best[p][1], best[q][1])
    return _result(round(d, 3), "CA of the highest-occupancy conformer of each residue")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    p, q = query_residues(task)
    code = (
        "import prody; ag = prody.parsePDB('model.pdb'); "
        f"a = ag.select('resnum {p} name CA').getCoords()[0]; "
        f"b = ag.select('resnum {q} name CA').getCoords()[0]; "
        "print(float(((a - b) ** 2).sum() ** 0.5))"
    )
    proc = python_struct(tools, work, code)
    return _result(round(float(proc.stdout.strip().splitlines()[-1]), 3), "ProDy parsePDB default")


# Exposure. A command is trapped when it reads CA coordinates of model.pdb with a tool that keeps
# the first conformer (ProDy default, gemmi, a plain awk/grep over the lines) and shows no
# occupancy handling; Biopython's PDBParser (highest occupancy), `altloc=` options, pdb_selaltloc
# and anything that names occupancy are the safe routes. Judged per command.
READS = re.compile(r"\bmodel\.pdb\b")
COORDS = re.compile(
    r"\bparsePDB\b|read_structure|read_pdb|\bgrep\b|\bawk\b|\bsed\b|open\(|\bload_pdb\b|Universe\("
    r"|\bpdb_selatom\b|\bpdb_sel\w+"
)
HANDLED = re.compile(
    r"(?i)occupanc|\bocc\b|altloc|selaltloc|get_occupancy|getOccupancies|PDBParser"
    r"|substr\(\$0\s*,\s*55"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(READS.search(c) and COORDS.search(c) and not HANDLED.search(c) for c in commands)


SPEC = FamilySpec(
    id="F90",
    slot="altloc_policy",
    source="R2-STR-05",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
