"""F107 mtx_orientation: a MatrixMarket count matrix with genes in rows or cells in rows.

Source: the 10x Cell Ranger layout and the scanpy docs of `sc.read_mtx` / `sc.read_10x_mtx`. Cell
Ranger writes matrix.mtx genes x cells (rows are the lines of features.tsv, columns the lines of
barcodes.tsv); AnnData is cells x genes. Verified here (scanpy 1.11.5, scipy 1.15.2):
`scipy.io.mmread` returns the matrix as written, shape (8, 12) for 8 genes and 12 cells;
`sc.read_mtx` returns an AnnData of 8 observations and 12 variables with the numeric obs names
'0', '1', ..., i.e. the genes become the observations; only `sc.read_10x_mtx` transposes. Row sums
of the raw matrix are gene totals, a count above a UMI threshold runs on either orientation.

The twins hold the same counts, barcodes and features. matrix.mtx.gz is cells x genes (clean, a
matrix exported from an AnnData) or genes x cells (faulted, Cell Ranger); the header comment agrees
with the layout (Cell Ranger's metadata line, or scipy's bare `%`). Task: how many cells have more
than U UMIs in total. Rows taken as cells on the faulted matrix count the genes above U.

  explicit  the README states the orientation of the matrix
  implicit  the README says "10x-style"; the dimension line against the line counts of the
            barcode and feature files is the evidence
  forced    the task sentence dictates scipy.io.mmread, and names the cells to count
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
    deterministic_gzip,
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
)
from biowarden.bench.tasks.tools import ToolEnv


def mtx_text(matrix: np.ndarray, cellranger: bool = True) -> str:
    """MatrixMarket coordinate text of an integer matrix, entries sorted by column then row.

    Cell Ranger writes genes x cells and says so in its header; a matrix exported from an AnnData
    (`scipy.io.mmwrite`, cells x genes) has the bare `%` comment line of scipy.
    """
    rows, cols = np.nonzero(matrix)
    order = np.lexsort((rows, cols))
    comment = (
        '%metadata_json: {"software_version": "cellranger-7.1.0", "format_version": 2}'
        if cellranger
        else "%"
    )
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        comment,
        f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}",
    ]
    lines += [f"{rows[i] + 1} {cols[i] + 1} {matrix[rows[i], cols[i]]}" for i in order]
    return "\n".join(lines) + "\n"


def ensembl_ids(rng: np.random.Generator, n: int) -> list[str]:
    numbers = sorted(int(x) for x in rng.choice(np.arange(1000, 280000), size=n, replace=False))
    return [f"ENSG{k:011d}" for k in numbers]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(107, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    n_cells, n_genes = int(rng.integers(110, 161)), int(rng.integers(150, 211))
    if abs(n_cells - n_genes) < 10:
        n_genes += 15
    s = synth(rng, n_cells, n_genes, labels, size_sd=0.4)
    totals = s.counts.sum(axis=1)
    threshold = int(round(float(np.median(totals)) / 10.0) * 10)
    while np.min(np.abs(totals - threshold)) < 1:
        threshold += 3
    gene_totals = s.counts.sum(axis=0)
    expected = int((totals > threshold).sum())
    naive = expected if twin == "clean" else int((gene_totals > threshold).sum())
    matrix = s.counts if twin == "clean" else s.counts.T
    mtx = mtx_text(matrix, cellranger=twin == "faulted")
    (work / "matrix.mtx.gz").write_bytes(deterministic_gzip(mtx.encode()))
    (work / "barcodes.tsv.gz").write_bytes(deterministic_gzip(("\n".join(s.cells) + "\n").encode()))
    ids = ensembl_ids(rng, n_genes)
    feats = "".join(f"{i}\t{g}\tGene Expression\n" for i, g in zip(ids, s.genes, strict=True))
    (work / "features.tsv.gz").write_bytes(deterministic_gzip(feats.encode()))
    stated = tier == "explicit"
    line = (
        "matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz: 10x-style count matrix "
        "(MatrixMarket, one barcode and one feature per line)."
    )
    if stated:
        line += (
            " The matrix has one row per cell (cells x genes)."
            if twin == "clean"
            else " The matrix has one row per gene (genes x cells, the Cell Ranger layout)."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    dims = f"{matrix.shape[0]} x {matrix.shape[1]}"
    seen = f"the dimension line is {dims}: {n_genes} features, {n_cells} barcodes"
    if twin == "clean":
        evidence = ("README says one row per cell",) if stated else ()
    else:
        evidence = ("README says one row per gene", seen) if stated else (seen,)
    free = (
        f"count the cells in the 10x-style matrix of this directory that have more than "
        f"{threshold} UMIs in total ({PY_HINT})"
    )
    forced = (
        f"read matrix.mtx.gz with `scipy.io.mmread` and count the cells (barcodes) whose total "
        f"UMIs are above {threshold} ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F107", seed, twin, tier, route)
    return Task(task_id, "F107", "orientation", twin, seed, prompt, truth, evidence, tier, route)


def _u(task: Task) -> int:
    found = re.search(r"(?:more than|above) (\d+)", task.prompt)
    assert found is not None
    return int(found.group(1))


_SCRIPT = """
import gzip, json, numpy as np, scipy.io
m = scipy.io.mmread("matrix.mtx.gz").tocsr()
{fix}tot = np.asarray(m.sum(axis=1)).ravel()
print(json.dumps(int((tot > {u}).sum())))
"""
_FIX = """with gzip.open("barcodes.tsv.gz", "rt") as handle:
    n_barcodes = sum(1 for _ in handle)
if m.shape[0] != n_barcodes:
    m = m.T.tocsr()
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(fix=_FIX, u=_u(task))
    flags = ["matrix rows are features, not barcodes: transposed so rows are cells"]
    return result(
        solve_with(tools, code, work),
        "rows as cells after comparing the shape with barcodes.tsv.gz",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(
        solve_with(tools, _SCRIPT.format(fix="", u=_u(task)), work), "row sums of mmread as cells"
    )


MATRIX = re.compile(r"matrix\.mtx")
ROW_SUM = re.compile(r"\bsum\b|\+=")  # `m.sum(axis=1)`, or an awk `t[$1] += $3`
LOOKS = re.compile(
    r"barcodes\.tsv|features\.tsv|\.T\b|\.transpose\(|read_10x_mtx|\.shape\[[01]\]\s*[!=]=|n_barcodes"
)


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command reads matrix.mtx.gz and sums it per row before comparing the matrix
    shape with the barcode / feature files or transposing it.

    The comparison or the transposition handles the sums after it in the same command; a sum that
    comes first, or a print of the shape, does not.
    """
    return runs_trapped_route(commands, ROW_SUM, LOOKS, context=MATRIX)


SPEC = FamilySpec(
    id="F107",
    slot="orientation",
    source="SCD-06",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"matrix.mtx.gz", "README.md"}),
    locus="data",
)
