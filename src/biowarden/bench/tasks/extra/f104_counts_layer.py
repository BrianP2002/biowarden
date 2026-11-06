"""F104 counts_layer: UMI-count steps run on a normalised X while the counts sit in a layer.

Source SCD-19 / SCD-23 (a normalised matrix and a counts matrix side by side; the tool reads the
default one). Verified here (scanpy 1.11.5, anndata 0.11.4, 60 cells of 31 to 489 UMIs):
`sc.pp.filter_cells(a, min_counts=150)` on X = log1p(counts per 10,000) keeps 55 cells where the
integer counts keep 30, and `sc.pp.calculate_qc_metrics` reports `total_counts` of 89 to 211 (sums
of logs). Exit 0, empty stderr; both functions read X unless told `layer=`.

The twins hold the same cells. Clean: X is the integer count matrix. Faulted: X is
log1p(counts per 10,000) and layers['counts'] holds the integers (the layout that scanpy tutorials
and most pipelines save). Task: how many cells have at least T UMIs.

  explicit  the README says where the counts are
  implicit  the README only says "UMI counts"; layers['counts'] and the float X are the evidence
  forced    the task sentence dictates sc.pp.filter_cells(a, min_counts=T)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f101_de_use_raw import (
    CELL_TYPES,
    PY_HINT,
    log1p_cp10k,
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
    write_h5ad,
)
from biowarden.bench.tasks.tools import ToolEnv


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(104, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(
        rng,
        int(rng.integers(100, 161)),
        int(rng.integers(150, 221)),
        labels,
        size_sd=0.5,
        mean_log=0.5,
    )
    counts = s.counts.astype(np.float32)
    totals = s.counts.sum(axis=1)
    logged = log1p_cp10k(s.counts)
    log_sums = logged.astype(np.float64).sum(axis=1)
    # a threshold (multiple of 10) with 20-80% of the cells above it where the count of cells with
    # that many UMIs and the count of cells whose log-sum passes it differ most; the log-sum count
    # is kept off the extremes (0 or every cell) when some threshold allows it
    n = len(totals)
    grid = range(int(np.quantile(totals, 0.2) // 10) * 10 + 10, int(np.quantile(totals, 0.8)), 10)
    usable = [
        t for t in grid if np.min(np.abs(totals - t)) >= 1 and np.min(np.abs(log_sums - t)) >= 0.5
    ]
    centred = [t for t in usable if 0.15 * n <= (log_sums >= t).sum() <= 0.85 * n]

    def spread(t: int) -> tuple[int, int]:
        return abs(int((log_sums >= t).sum()) - int((totals >= t).sum())), -t

    threshold = max(centred or usable, key=spread)
    if twin == "clean":
        write_h5ad(work, "cells.h5ad", counts, s.cells, s.genes, tools=tools)
    else:
        write_h5ad(
            work,
            "cells.h5ad",
            logged,
            s.cells,
            s.genes,
            layers={"counts": counts},
            uns={"log1p": {"base": None}},
            tools=tools,
        )
    stated = tier == "explicit"
    line = "cells.h5ad: scanpy AnnData (cells x genes), single-cell UMI counts."
    if stated:
        line += (
            " X holds the raw UMI counts."
            if twin == "clean"
            else " X holds log1p of counts per 10,000; the raw UMI counts are in layers['counts']."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = f"X has non-integer values (max {logged.max():.1f}) and the object has layers['counts']"
    if twin == "clean":
        evidence = ("README states X holds raw counts",) if stated else ()
    else:
        evidence = ("README says the counts are in layers['counts']", seen) if stated else (seen,)
    free = f"count the cells in cells.h5ad with at least {threshold} UMIs in total ({PY_HINT})"
    forced = (
        f"use `sc.pp.filter_cells(a, min_counts={threshold}, inplace=False)` on cells.h5ad to "
        f"count the cells with at least {threshold} UMIs in total ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    expected = int((totals >= threshold).sum())
    naive = expected if twin == "clean" else int((log_sums >= threshold).sum())
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F104", seed, twin, tier, route)
    return Task(task_id, "F104", "matrix_state", twin, seed, prompt, truth, evidence, tier, route)


def _threshold(task: Task) -> int:
    found = re.search(r"at least (\d+) UMIs", task.prompt)
    assert found is not None
    return int(found.group(1))


_SCRIPT = """
import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{layer}keep, _ = sc.pp.filter_cells(a, min_counts={threshold}, inplace=False)
print(json.dumps(int(keep.sum())))
"""
_LAYER = "if 'counts' in a.layers:\n    a.X = a.layers['counts'].copy()\n"


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(layer=_LAYER, threshold=_threshold(task))
    flags = ["X is normalised; integer counts taken from layers['counts']"]
    return result(
        solve_with(tools, code, work),
        "filter_cells on the counts layer",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(layer="", threshold=_threshold(task))
    return result(solve_with(tools, code, work), "filter_cells(min_counts) on X")


COUNTS_STEP = re.compile(
    r"\bmin_counts\s*=|\bcalculate_qc_metrics\s*\(|total_counts"
    r"|\.X\b[^\n;]{0,300}\bsum\s*\(|\bsum\s*\([^\n;]{0,300}\.X\b"
    r"|(?<![\w.])X\.(?:\w+\(\s*\)\.)*sum\s*\("
)
# the counts layer is put into X: handles the statements after it in the same command
COUNTS_INTO_X = re.compile(r"(?<![\w])(?:\w+\.)?X\s*=(?!=)[^\n;]{0,300}\blayers\s*\[")
# the call names a layer (or raw) itself
OWN_LAYER = re.compile(r"\blayers\s*\[|\blayer\s*=\s*['\"]|\.raw\b|\buse_raw\s*=\s*True")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command applies a UMI-count step to X without naming the counts layer.

    A statement that names a layer, raw or use_raw=True is set aside; the counts layer put into X
    handles the steps after it in that command. Printing the layer does not: a command that shows
    layers['counts'] and then filters X still ran the trapped step.
    """
    return runs_trapped_route(commands, COUNTS_STEP, COUNTS_INTO_X, OWN_LAYER)


SPEC = FamilySpec(
    id="F104",
    slot="matrix_state",
    source="SCD-19",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
