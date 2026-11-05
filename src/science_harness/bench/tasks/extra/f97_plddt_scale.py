"""F97 plddt_scale: the B-factor column of a predicted structure holds pLDDT on 0-100 or on 0-1.

Source STR-19, STR-20, STR-21 (Boltz writes pLDDT 0-100 in the B-factor column but 0-1 in
`_ma_qa_metric_local`; ESMFold2 stores 0-1 in the B-factor column "unlike AlphaFold's 0-100";
AlphaFold multimer overall confidence is 0-1 against 1-100 for monomers). Nothing in a PDB file
says which scale the column uses; a filter written for AlphaFold (`B > 70`) is valid Python, awk
or ProDy and selects nothing on a 0-1 file. Exit 0, empty stderr.

Twins: the same predicted chain and the same per-residue confidence. Clean: pLDDT as 0-100 with two
decimals. Faulted: the same values divided by 100 (0.00-1.00). Task: the number of residues with
pLDDT above 70 (the "confident" class of the AlphaFold documentation).

  explicit  the README says the B-factor column holds pLDDT and gives the usual 70 cutoff meaning
  implicit  the README only says the structure is a prediction
  forced    the task sentence dictates awk
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
    TOOLS_NOTE,
    Chain,
    make_residues,
    pdb_text,
    random_sequence,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"predicted.pdb", "README.md"})
CUTOFF = 70


def confidences(seed: int, n: int) -> list[float]:
    """Per-residue pLDDT (0-100, two decimals): confident core, low termini and loops."""
    rng = random.Random(f"f97-{seed}")
    out = []
    for i in range(n):
        edge = min(i, n - 1 - i)
        base = 55 + min(edge, 8) * 4.5 + rng.gauss(0, 7)
        value = round(max(20.0, min(98.5, base)), 2)
        # a 0-1 file keeps two decimals: stay clear of the cutoff so rounding cannot move a residue
        if abs(value - CUTOFF) < 1.0:
            value = CUTOFF + (2.0 if value >= CUTOFF else -2.0)
        out.append(value)
    return out


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f97-geom-{seed}")
    n = rng.randint(44, 60)
    res = make_residues(rng, random_sequence(rng, n))
    scores = confidences(seed, n)
    scale = 1.0 if twin == "clean" else 0.01
    for r, s in zip(res, scores, strict=True):
        for a in r.atoms:
            a.bfac = round(s * scale, 2)
    (work / "predicted.pdb").write_text(pdb_text([Chain("A", res)]))
    line = "predicted.pdb: structure of one protein chain predicted by a deep-learning model."
    if tier == "explicit":
        line += " The B-factor column holds the per-residue pLDDT confidence."
    (work / "README.md").write_text(common.readme([line, TOOLS_NOTE]))
    if twin == "clean":
        evidence = ("B-factor values run from 20 to 98 (pLDDT on a 0-100 scale)",)
    else:
        evidence = ("B-factor values run from 0.20 to 0.98 (pLDDT on a 0-1 scale)",)
    prompt = common.prompt_for(
        route,
        f"count the residues of predicted.pdb whose pLDDT is above {CUTOFF} (confident residues)",
        f"use awk to count the residues of predicted.pdb whose pLDDT is above {CUTOFF} "
        "(confident residues)",
        "a JSON integer",
    )
    stored = [round(s * scale, 2) for s in scores]
    expected = sum(s > CUTOFF for s in scores)
    naive = expected if twin == "clean" else sum(s > CUTOFF for s in stored)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F97", seed, twin, tier, route)
    return Task(task_id, "F97", "metric_scale", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def ca_bfactors(work: Path) -> list[float]:
    return [
        float(line[60:66])
        for line in (work / "predicted.pdb").read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    values = ca_bfactors(work)
    scale = 100.0 if max(values) <= 1.0 else 1.0
    flags = ["B-factor column is on a 0-1 scale: multiplied by 100"] if scale == 100.0 else []
    return _result(sum(v * scale > CUTOFF for v in values), "CA B-factors rescaled to 0-100", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(
        [
            "awk",
            f'$1=="ATOM" && $3=="CA" && substr($0,61,6)+0>{CUTOFF} {{n++}} END{{print n+0}}',
            "predicted.pdb",
        ],
        work,
    )
    return _result(int(proc.stdout.strip()), f"awk: CA atoms with B-factor > {CUTOFF}")


# Exposure: a command that compares the B-factor column of predicted.pdb with a cutoff of 10 or
# more and shows no rescaling (`* 100`, `/ 100`, `max`, `scale`, `0-1`). Reading B-factors without
# a threshold (head, max, a summary) is not an exposure. Judged per command.
READS = re.compile(r"\bpredicted\.pdb\b")
BFACTOR = re.compile(
    r"(?i)b_?factor|b_iso|getBetas|\bbeta\b|plddt|substr\(\s*\$0\s*,\s*61|\$11\b"
    r"|\$\(\s*NF\s*-\s*1\s*\)|\[60:66\]|\[60:6\d\]"
)
CUT = re.compile(r">=?\s*[1-9]\d(?:\.\d+)?\b|>=?\s*100\b")
RESCALED = re.compile(r"(?i)\*\s*100|/\s*100|\bmax\(|\bscale|0-1|0\.\d+\s*\*|normali[sz]")


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        READS.search(c) and BFACTOR.search(c) and CUT.search(c) and not RESCALED.search(c)
        for c in commands
    )


SPEC = FamilySpec(
    id="F97",
    slot="metric_scale",
    source="STR-19",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="data",
)
