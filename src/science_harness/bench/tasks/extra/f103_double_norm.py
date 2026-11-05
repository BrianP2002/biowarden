"""F103 double_norm: normalize_total + log1p applied to an X that is already log-normalised.

Source SCD-19 and the scanpy source (`sc.pp.log1p`): the function warns "adata.X seems to be
already log-transformed" only when `adata.uns` has a 'log1p' key. An .h5ad written by another tool
(Seurat export, a database download, a collaborator's script) usually has none. Verified here
(scanpy 1.11.5, anndata 0.11.4): `log1p` on such an object exits 0 with an empty stderr and takes
the maximum from 6.1 to 1.96; `normalize_total` on the log values then gives every cell the same
total and a mean of the wrong quantity. With `uns['log1p']` present scanpy does print the warning.

The twins hold the same biology. Clean: X is the integer count matrix. Faulted: X is
log1p(counts per 10,000) as float32 and uns has no 'log1p' key. Task: the mean of log1p(counts per
10,000) of one marker gene in one cell type. A scanpy user runs the usual pair (normalize_total
target_sum=1e4, log1p); on the faulted twin that normalises a second time.

  explicit  the README states what X holds
  implicit  the README only says "expression matrix"; non-integer values with a maximum near 8
            are the evidence
  forced    the task sentence dictates `sc.pp.normalize_total` and `sc.pp.log1p`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f101_de_use_raw import (
    CELL_TYPES,
    PY_HINT,
    cp10k,
    log1p_cp10k,
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
    write_h5ad,
)
from science_harness.bench.tasks.tools import ToolEnv

TOL = 0.01


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(103, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(rng, int(rng.integers(120, 181)), int(rng.integers(150, 221)), labels)
    target = labels[int(rng.integers(0, 3))]
    logged = log1p_cp10k(s.counts)
    again = np.log1p(cp10k(logged.astype(np.float64)).astype(np.float64))
    mask = np.array([lab == target for lab in s.labels])
    # the marker whose mean moves most under a second transform (it can coincide for some genes)
    gene = max(
        s.markers[target],
        key=lambda g: abs(
            float(again[mask, g].mean()) - float(logged[mask, g].astype(np.float64).mean())
        ),
    )
    x = s.counts.astype(np.float32) if twin == "clean" else logged
    write_h5ad(work, "cells.h5ad", x, s.cells, s.genes, obs={"celltype": s.labels}, tools=tools)
    stated = tier == "explicit"
    line = "cells.h5ad: scanpy AnnData (cells x genes), cell types in obs['celltype']."
    if stated:
        line += " X holds " + (
            "raw UMI counts." if twin == "clean" else "log1p of counts per 10,000 (natural log)."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = f"X has non-integer values (max {logged.max():.1f}) and uns has no 'log1p' key"
    if twin == "clean":
        evidence = ("README states X holds raw counts",) if stated else ()
    else:
        evidence = ("README states X is log1p of CP10K", seen) if stated else (seen,)
    expected = float(logged[mask, gene].astype(np.float64).mean())
    naive = expected if twin == "clean" else float(again[mask, gene].mean())
    name = s.genes[gene]
    free = (
        f"report the mean of log1p(counts per 10,000) (natural log) of gene {name} across the "
        f"'{target}' cells of cells.h5ad ({PY_HINT})"
    )
    forced = (
        f"use `sc.pp.normalize_total` (target_sum=1e4) and `sc.pp.log1p` on cells.h5ad, then "
        f"report the mean of gene {name} across the '{target}' cells ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON number")
    truth = {"kind": "float", "expected": expected, "naive": naive, "tol": TOL}
    task_id = common.task_id("F103", seed, twin, tier, route)
    return Task(task_id, "F103", "matrix_state", twin, seed, prompt, truth, evidence, tier, route)


def _inputs(task: Task) -> tuple[str, str]:
    found = re.search(r"gene (\w+) across the '([^']+)' cells", task.prompt)
    assert found is not None
    return found.group(1), found.group(2)


_SCRIPT = """
import json, numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
{check}
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
sub = a[(a.obs["celltype"] == {group!r}).values, {gene!r}].X
print(json.dumps(float(np.asarray(sub.todense() if sp.issparse(sub) else sub).mean())))
"""
_CHECK = """v = a.X.data if sp.issparse(a.X) else np.asarray(a.X)
if np.allclose(v, np.round(v)):"""
_NO_CHECK = "if True:"


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    gene, group = _inputs(task)
    code = _SCRIPT.format(check=_CHECK, gene=gene, group=group)
    flags = ["X already holds non-integer log-scale values: not normalised a second time"]
    return result(
        solve_with(tools, code, work),
        "normalised only integer counts",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    gene, group = _inputs(task)
    code = _SCRIPT.format(check=_NO_CHECK, gene=gene, group=group)
    return result(solve_with(tools, code, work), "normalize_total and log1p on X")


NORMALISE = re.compile(
    r"\bnormalize_total\s*\(|\blog1p\s*\(|\bnormalize_per_cell\s*\(|\blog\s*\(\s*1\s*\+"
    r"|\blog\s*\([^()\n]*\+\s*1\s*\)"
)
HANDLED = re.compile(r"\bexpm1\b|\ballclose\b|\bis_integer\b|\.max\s*\(\s*\)\s*[<>]|\blayers\s*\[")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command applies normalize_total / log1p (or log(1 + x)) to X before checking
    whether X is already normalised (a back-transform, an integer test, a comparison of X.max()).

    A check handles the statements after it in the same command, not the ones before: a model that
    normalised, noticed and then back-transformed still executed the trapped route. A print of
    such a value (`print(np.allclose(...))`) is a look, not a check.
    """
    return runs_trapped_route(commands, NORMALISE, HANDLED)


SPEC = FamilySpec(
    id="F103",
    slot="matrix_state",
    source="SCD-19",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
