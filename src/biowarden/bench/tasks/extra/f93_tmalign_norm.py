"""F93 tmalign_norm: TMalign prints two TM-scores, normalised by structure 1 and by structure 2.

Source STR-22 (US-align/TM-align documentation: "You should use TM-score normalized by length of
the reference structure"). Verified here with TM-align 20240303 (`TMalign model.pdb reference.pdb`):

  Length of Structure_1: 40 residues      Length of Structure_2: 54 residues
  TM-score= 0.93048 (normalized by length of Structure_1: L=40, d0=1.83)
  TM-score= 0.70964 (normalized by length of Structure_2: L=54, d0=2.41)

Exit 0, empty stderr. A reader who takes the first `TM-score=` line reports the model-normalised
value (0.93) as the quality of a model of a 54-residue protein; the reference-normalised value is
0.71. The two lines agree when the structures have the same length. `TMalign -a T` normalises by
the average length, `-u N` by a given length.

Twins: the same 70% core in both. Faulted: model.pdb holds only the core (m residues), the
reference has L > m, so the first line is the model-normalised, inflated score. Clean: model.pdb
has L residues, the last L - m of them folded somewhere far from the reference, so both lines give
the core's score (the unaligned tail adds ~0 to the sum). The recorded answer is the
reference-normalised score TMalign prints for the faulted pair.

  explicit  the README says to report the TM-score normalised by the reference length
  implicit  the README only says which file is the reference
  forced    the task sentence dictates TMalign
"""

from __future__ import annotations

import functools
import math
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f89_chain_ids import (
    AA3,
    TOOLS_NOTE,
    Atom,
    Chain,
    Residue,
    backbone,
    pdb_text,
    random_fold,
    random_sequence,
)
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"model.pdb", "README.md"})
TOL = 0.02
LINE = re.compile(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_([12])")


def rotation(rng: random.Random) -> list[list[float]]:
    """A random proper rotation matrix (Rodrigues)."""
    axis = [rng.gauss(0, 1) for _ in range(3)]
    n = math.sqrt(sum(x * x for x in axis))
    x, y, z = (a / n for a in axis)
    th = rng.uniform(0.5, 2.5)
    c, s = math.cos(th), math.sin(th)
    return [
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ]


def apply(rot, shift, xyz):
    return tuple(sum(rot[i][j] * xyz[j] for j in range(3)) + shift[i] for i in range(3))


@functools.lru_cache(maxsize=64)
def make_pair(seed: int) -> tuple[str, str, str, int, int]:
    """(reference PDB, model core-only PDB, model with tail PDB, L, m)."""
    rng = random.Random(f"f93-{seed}")
    length = rng.randint(54, 66)
    core = round(0.7 * length)
    seq = random_sequence(rng, length)
    ref_bb = backbone(*random_fold(rng, length))
    ref_res = [
        Residue(
            AA3[seq[i]],
            i + 1,
            [Atom(n, ref_bb[i][n]) for n in ("N", "CA", "C", "O")],
        )
        for i in range(length)
    ]
    rot, shift = rotation(rng), tuple(rng.uniform(-8, 8) for _ in range(3))
    model_res = []
    for i in range(core):
        atoms = [
            Atom(n, apply(rot, shift, tuple(v + rng.gauss(0, 0.3) for v in ref_bb[i][n])))
            for n in ("N", "CA", "C", "O")
        ]
        model_res.append(Residue(AA3[seq[i]], i + 1, atoms))
    tail_bb = backbone(*random_fold(rng, length - core))
    far = tuple(rng.choice((-1, 1)) * rng.uniform(45, 60) for _ in range(3))
    tail_res = [
        Residue(
            AA3[seq[core + i]],
            core + i + 1,
            [
                Atom(n, tuple(tail_bb[i][n][k] + far[k] for k in range(3)))
                for n in ("N", "CA", "C", "O")
            ],
        )
        for i in range(length - core)
    ]
    ref = pdb_text([Chain("A", ref_res)])
    return (
        ref,
        pdb_text([Chain("A", model_res)]),
        pdb_text([Chain("A", model_res + tail_res)]),
        length,
        core,
    )


def scores(tools: ToolEnv, cwd: Path) -> tuple[float, float]:
    """(normalised by structure 1, by structure 2) printed by `TMalign model.pdb reference.pdb`."""
    proc = tools.run(["TMalign", "model.pdb", "reference.pdb"], cwd)
    found = {int(m.group(2)): float(m.group(1)) for m in LINE.finditer(proc.stdout)}
    return found[1], found[2]


@functools.lru_cache(maxsize=64)
def recorded(seed: int, tools: ToolEnv) -> tuple[float, float]:
    """TMalign on the core-only (faulted) pair: (naive, expected)."""
    ref, core, _, _, _ = make_pair(seed)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "reference.pdb").write_text(ref)
        (Path(tmp) / "model.pdb").write_text(core)
        return scores(tools, Path(tmp))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    ref, core, full, length, m = make_pair(seed)
    (work / "reference.pdb").write_text(ref)
    (work / "model.pdb").write_text(core if twin == "faulted" else full)
    lines = [
        f"reference.pdb: experimental structure, one chain of {length} residues (PDB format).",
        "model.pdb: predicted model of the same protein.",
    ]
    if tier == "explicit":
        lines.append("The TM-score to report is normalised by the length of the reference.")
    (work / "README.md").write_text(common.readme([*lines, TOOLS_NOTE]))
    if twin == "clean":
        evidence = (f"model.pdb and reference.pdb both have {length} residues",)
    else:
        evidence = (f"model.pdb has {m} residues, reference.pdb has {length}",)
    prompt = common.prompt_for(
        route,
        "report the TM-score of the model model.pdb against the experimental structure "
        "reference.pdb",
        "use TMalign to report the TM-score of the model model.pdb against the experimental "
        "structure reference.pdb",
        "a JSON number",
    )
    naive_score, expected = recorded(seed, tools)
    truth = {
        "kind": "float",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_score,
        "tol": TOL,
    }
    task_id = common.task_id("F93", seed, twin, tier, route)
    return Task(
        task_id, "F93", "metric_normalisation", twin, seed, prompt, truth, evidence, tier, route
    )


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    _, by_reference = scores(tools, work)
    return _result(
        by_reference, "TMalign model.pdb reference.pdb: TM-score by Structure_2 (reference)"
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    first, _ = scores(tools, work)
    return _result(first, "TMalign model.pdb reference.pdb: first TM-score line")


# Exposure: a TMalign run whose result is read without choosing a normalisation. Handled routes:
# `-u/-L` (given length), a pipeline that names Structure_2 or the second TM-score (`sed -n 2p`,
# `Structure_2`, `tail`, `grep -A1`, `awk` on NR==2), or TMalign with the reference as the FIRST
# structure (then the first line is the reference-normalised one). `-a T` is not a route: it only
# adds a third, average-length line (0.794 against 0.683) and leaves the first two as they were.
RUN = re.compile(r"\bTMalign\b")
CHOSEN = re.compile(
    r"-u\s+\d|-L\s+\d|Structure_2|Structure_1:\s*L|\bsed\s+-n\s+['\"]?2p|NR\s*==\s*2"
    r"|tail\s+-\d?|grep\s+-[A-Za-z]*A\s*1|normalized by length of Structure_2|\[1\]|\[-1\]"
)
REF_FIRST = re.compile(r"TMalign\s+(?:\./)?reference\.pdb\s+(?:\./)?model\.pdb")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        RUN.search(c) and not CHOSEN.search(c) and not REF_FIRST.search(c)
        for c in commands
        if "model.pdb" in c and "reference.pdb" in c
    )


SPEC = FamilySpec(
    id="F93",
    slot="metric_normalisation",
    source="STR-22",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
