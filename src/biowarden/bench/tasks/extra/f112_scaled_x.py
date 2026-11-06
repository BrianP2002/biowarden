"""F112 scaled_x: X holds z-scores (sc.pp.scale) and the log-normalised values live in .raw.

Source SCD-22 / the scanpy tutorials: `adata.raw = adata` before `sc.pp.scale`, so a processed
object has centred, unit-variance X and the expression values in `.raw`. Reading X as expression
runs without a word: `adata[:, g].X > 0` is true for the cells above the gene's mean, whatever the
gene's detection (for a sparsely detected gene far fewer cells than express it); `mean()` is near 0.

The twins hold the same cells and genes. Clean: X is log1p of counts per 10,000 and .raw a copy.
Faulted: X is the z-score of each gene (clipped at 10, as `sc.pp.scale(max_value=10)`), .raw the
log-normalised matrix. Task: how many cells of one type express a gene (log-normalised expression
above zero). The gene is chosen so that scaled X > 0 holds for much fewer cells than detection.

  explicit  the README says X is scaled and .raw is log-normalised
  implicit  the README says "processed object"; the negative values in X are the evidence
  forced    the task sentence dictates `adata[:, gene].X > 0`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from biowarden.bench.runner.exposure import statements
from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f101_de_use_raw import (
    CELL_TYPES,
    PY_HINT,
    log1p_cp10k,
    make_rng,
    result,
    solve_with,
    synth,
    write_h5ad,
)
from biowarden.bench.tasks.tools import ToolEnv

MIN_SHARE = 0.25  # detected cells minus cells with z > 0, as a share of the type


def zscore(logged: np.ndarray, clip: float = 10.0) -> np.ndarray:
    x = logged.astype(np.float64)
    sd = x.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return np.clip((x - x.mean(axis=0)) / sd, -clip, clip).astype(np.float32)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(112, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(rng, int(rng.integers(100, 151)), int(rng.integers(120, 181)), labels, mean_log=0.6)
    target = labels[int(rng.integers(0, 3))]
    logged = log1p_cp10k(s.counts)
    scaled = zscore(logged)
    in_type = np.array([lab == target for lab in s.labels])
    detected = (logged[in_type] > 0).sum(axis=0)
    positive = (scaled[in_type] > 0).sum(axis=0)
    share = (detected - positive) / in_type.sum()
    gene = int(np.argmax(np.where(detected >= 10, share, -1.0)))
    expected, naive_count = int(detected[gene]), int(positive[gene])
    x = logged if twin == "clean" else scaled
    write_h5ad(
        work,
        "cells.h5ad",
        x,
        s.cells,
        s.genes,
        obs={"celltype": s.labels},
        raw={"X": logged},
        sparse=False,
        uns={"log1p": {"base": None}},
        tools=tools,
    )
    stated = tier == "explicit"
    line = "cells.h5ad: processed scanpy AnnData (cells x genes), cell types in obs['celltype']."
    if stated:
        line += (
            " X is log1p of counts per 10,000; .raw is a copy of X."
            if twin == "clean"
            else " X is scaled (z-scores per gene, clipped at 10); .raw holds the log1p of "
            "counts per 10,000."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = f"X has negative values (min {scaled.min():.1f}); .raw.X does not"
    if twin == "clean":
        evidence = ("README says X is log-normalised",) if stated else ()
    else:
        evidence = (
            ("README says X is scaled and .raw is log-normalised", seen) if stated else (seen,)
        )
    name = s.genes[gene]
    free = (
        f"count the '{target}' cells of cells.h5ad that express gene {name} (log-normalised "
        f"expression above zero) ({PY_HINT})"
    )
    forced = (
        f"count the '{target}' cells of cells.h5ad for which `adata[:, '{name}'].X` is above "
        f"zero, i.e. that express gene {name} ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {
        "kind": "int",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_count,
    }
    task_id = common.task_id("F112", seed, twin, tier, route)
    return Task(task_id, "F112", "matrix_state", twin, seed, prompt, truth, evidence, tier, route)


def _inputs(task: Task) -> tuple[str, str]:
    found = re.search(r"'([^']+)' cells of cells\.h5ad.*?gene (\w+)", task.prompt)
    assert found is not None
    return found.group(1), found.group(2)


_SCRIPT = """
import json, numpy as np, anndata as ad, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
{source}
x = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
print(json.dumps(int((x[(a.obs["celltype"] == {group!r}).values] > 0).sum())))
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, gene = _inputs(task)
    code = _SCRIPT.format(source=f"x = a.raw[:, {gene!r}].X", group=group)
    flags = ["X holds scaled z-scores: expression read from .raw"]
    return result(
        solve_with(tools, code, work),
        "log-normalised values from .raw",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, gene = _inputs(task)
    code = _SCRIPT.format(source=f"x = a[:, {gene!r}].X", group=group)
    return result(solve_with(tools, code, work), "expression read from X")


READS_X = re.compile(r"\.X\b|\bto_df\(|\bobs_df\(")
HANDLED = re.compile(r"\.raw\b|\buse_raw\s*=\s*True|\blayers\s*\[|\blayer\s*=\s*['\"]")
POSITIVE = re.compile(r">\s*0|!=\s*0|\bnonzero\b|\bnnz\b|\bastype\(\s*bool")  # "expressed"
NAMES = re.compile(r"\b(\w+)\s*=(?!=)")
WORDS = re.compile(r"\w+")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command reads expression from X of cells.h5ad, not from .raw / a layer, and
    counts what is above zero (`> 0`, `!= 0`, nonzero, nnz) in what it read.

    A statement that reads .raw (or a layer) is set aside. What comes from X is followed through
    the variables it is assigned to, so `x = a.raw[:, g].X; print((x > 0).sum())` is the safe route
    and `print(a.X.min())` is a look at the matrix, not the trapped count.
    """
    if "cells.h5ad" not in "\n".join(commands):
        return False
    for command in commands:
        from_x: set[str] = set()
        for statement in statements(command):
            if HANDLED.search(statement):
                continue
            if READS_X.search(statement) or from_x.intersection(WORDS.findall(statement)):
                from_x.update(NAMES.findall(statement))
                if POSITIVE.search(statement):
                    return True
    return False


SPEC = FamilySpec(
    id="F112",
    slot="matrix_state",
    source="SCD-22",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
