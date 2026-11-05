"""F102 int16_qc: `sc.pp.calculate_qc_metrics` on a sparse int16 count matrix.

Source R2-OMX-16 (scanpy issue 4440, open): the percent-of-counts-in-top-N metrics accumulate the
cell total in the dtype of the matrix. Verified here (scanpy 1.11.5, anndata 0.11.4, 8 cells of
32,000 to 33,000 UMIs, percent_top=[20]): a csr or csc int16 matrix gives
pct_counts_in_top_20_genes of -9.25 for the cell whose total passes 32,767 and 9.26 for the cells
below it; the same counts as dense int16, float32 or int32 give 9.23 to 9.4 for every cell. Exit
0, empty stderr, `total_counts` is right (it is summed elsewhere), only the percentages of
the cells above 32,767 UMIs are wrong: for totals 32,768..65,535 they come out negative, for
65,536..98,303 above 100. The type survives an .h5ad round trip, so a file saved as int16 carries
the trap to every later session. uint16 raises an AttributeError (loud).

The twins hold the same counts: X is sparse float32 (clean) or sparse int16 (faulted). Library
sizes are about 30,000 UMIs (deep full-length-like or multiplexed data), so about a third of the
cells are above 32,767. Task: count the cells whose top-20 genes hold more than T percent of the
counts, with scanpy's metric. The naive count only sees the cells below 32,768.

  explicit  the README states the dtype of X
  implicit  the README says "UMI counts"; adata.X.dtype is the evidence
  forced    the task sentence dictates sc.pp.calculate_qc_metrics(a, percent_top=[20], inplace=True)
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
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
    write_h5ad,
)
from science_harness.bench.tasks.tools import ToolEnv

TOP = 20
MARGIN = 0.02  # no true percentage within this many points of the threshold (float rounding)


def wrap16(values: np.ndarray) -> np.ndarray:
    return ((values.astype(np.int64) + 32768) % 65536 - 32768).astype(np.int64)


def top_pct(counts: np.ndarray, wrapped: bool) -> np.ndarray:
    """pct_counts_in_top_20_genes: exact, or as scanpy computes it on a sparse int16 matrix."""
    top = np.sort(counts, axis=1)[:, ::-1][:, :TOP].sum(axis=1).astype(np.int64)
    total = counts.sum(axis=1).astype(np.int64)
    if wrapped:
        top, total = wrap16(top), wrap16(total)
    return top / total * 100


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(102, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(
        rng,
        int(rng.integers(60, 91)),
        int(rng.integers(250, 301)),
        labels,
        n_markers=8,
        fold=(3.0, 6.0),
        size_sd=0.18,
        target_total=31000.0,
        size_clip=(0.6, 1.7),
    )
    totals = s.counts.sum(axis=1)
    assert totals.max() < 65536
    true = top_pct(s.counts, wrapped=False)
    naive_pct = top_pct(s.counts, wrapped=True)
    # the threshold where the cells that wrap (more than 32,767 UMIs) change the count most
    best: tuple[int, float] | None = None
    for q in np.linspace(0.3, 0.7, 21):
        candidate = float(np.round(np.quantile(true, q), 2))
        if np.min(np.abs(true - candidate)) < MARGIN:
            continue
        lost = int((true > candidate).sum()) - int((naive_pct > candidate).sum())
        if best is None or lost > best[0]:
            best = (lost, candidate)
    assert best is not None and best[0] >= 2, "no threshold separates the wrapped cells"
    threshold = best[1]
    dtype = np.float32 if twin == "clean" else np.int16
    write_h5ad(work, "cells.h5ad", s.counts.astype(dtype), s.cells, s.genes, tools=tools)
    stated = tier == "explicit"
    line = "cells.h5ad: scanpy AnnData (cells x genes), UMI counts in X."
    if stated:
        line += f" X is stored as a sparse {'float32' if twin == 'clean' else 'int16'} matrix."
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    above = int((totals > 32767).sum())
    seen = f"X is a sparse int16 matrix and {above} cells have more than 32767 UMIs"
    if twin == "clean":
        evidence = ("README states X is float32",) if stated else ()
    else:
        evidence = ("README states X is int16", seen) if stated else (seen,)
    free = (
        f"count the cells whose pct_counts_in_top_{TOP}_genes, as scanpy's calculate_qc_metrics "
        f"reports it with percent_top=[{TOP}], is above {threshold} in cells.h5ad ({PY_HINT})"
    )
    forced = (
        f"run `sc.pp.calculate_qc_metrics(a, percent_top=[{TOP}], inplace=True)` on cells.h5ad "
        f"and count the cells whose pct_counts_in_top_{TOP}_genes is above {threshold} "
        f"({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    expected = int((true > threshold).sum())
    naive = expected if twin == "clean" else int((naive_pct > threshold).sum())
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F102", seed, twin, tier, route)
    return Task(task_id, "F102", "dtype_overflow", twin, seed, prompt, truth, evidence, tier, route)


_SCRIPT = """
import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{cast}sc.pp.calculate_qc_metrics(a, percent_top=[{top}], inplace=True)
print(json.dumps(int((a.obs["pct_counts_in_top_{top}_genes"] > {threshold}).sum())))
"""


def _threshold(task: Task) -> float:
    import re

    found = re.search(r"above (\d+(?:\.\d+)?)", task.prompt)
    assert found is not None
    return float(found.group(1))


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(cast="a.X = a.X.astype('float64')\n", top=TOP, threshold=_threshold(task))
    flags = ["X is sparse int16: cast to float64 before calculate_qc_metrics"]
    return result(solve_with(tools, code, work), "calculate_qc_metrics on float64", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(cast="", top=TOP, threshold=_threshold(task))
    return result(solve_with(tools, code, work), "calculate_qc_metrics on X as stored")


QC_CALL = re.compile(r"\bcalculate_qc_metrics\s*\(")
CAST = re.compile(r"\.astype\s*\(|\bdtype\s*=")  # not the words float32 / float64: dtype is printed


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff calculate_qc_metrics runs on X before a cast of X to a wider type.

    A cast (`astype(`, `dtype=`) handles the calls after it in the same command; a cast that comes
    after the call, or the word float32 in a print, does not.
    """
    return runs_trapped_route(commands, QC_CALL, CAST)


SPEC = FamilySpec(
    id="F102",
    slot="dtype_overflow",
    source="R2-OMX-16",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
