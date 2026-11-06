"""F105 obs_alignment: cell annotations joined to an AnnData by position instead of by barcode.

Source R2-OMX-14 (anndata issue 713): `adata.obs = table` with the same barcodes in another order
renames the cells without moving X, and `adata.obs['celltype'] = series.values` pastes labels by
position. Verified here (anndata 0.11.4, pandas 2.3.3): for annotations listed in reversed order,
`obs['ct'] = ann['celltype']` aligns on the index and is right; `obs['ct'] = ann['celltype'].values`
gives every cell the label of another cell; `adata.obs = ann` changes obs_names to the new order
while the row of X for c0 keeps the counts of c0 (row 0 sum 83.0 before and after), with no warning
and no exception. A table read without index_col assigns NaN (visible).

The twins hold the same cells and labels. annotations.csv lists the barcodes in the order of
cells.h5ad (clean) or sorted by cell type (faulted, the order a clustering export has). Task: how
many cells of one type have a marker gene well above background (at least k counts).

  explicit  the README says whether annotations.csv follows the order of cells.h5ad
  implicit  the README says "one row per cell, keyed by barcode"; the barcode order is the evidence
  forced    the task sentence dictates `adata.obs['celltype'] = ...` from the table
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
    assignments,
    make_rng,
    result,
    solve_with,
    synth,
    write_h5ad,
)
from biowarden.bench.tasks.tools import ToolEnv


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(105, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(rng, int(rng.integers(100, 161)), int(rng.integers(120, 181)), labels)
    target = labels[int(rng.integers(0, 3))]
    in_target = np.array([lab == target for lab in s.labels])
    gene, depth = max(
        (
            (g, int(np.ceil(np.quantile(s.counts[~in_target, g], 0.93))) + 1)
            for g in s.markers[target]
        ),
        key=lambda item: (s.counts[in_target, item[0]] >= item[1]).mean(),
    )
    write_h5ad(work, "cells.h5ad", s.counts.astype(np.float32), s.cells, s.genes, tools=tools)
    order = list(range(len(s.cells)))
    if twin == "faulted":
        order = sorted(order, key=lambda i: (s.labels[i], s.cells[i]))
    rows = "".join(f"{s.cells[i]},{s.labels[i]}\n" for i in order)
    (work / "annotations.csv").write_text("barcode,celltype\n" + rows)
    stated = tier == "explicit"
    ann = "annotations.csv: cell type per cell, columns barcode and celltype, one row per cell."
    if stated:
        ann += (
            " The rows follow the order of the cells in cells.h5ad."
            if twin == "clean"
            else " The rows are sorted by cell type, not in the order of the cells in cells.h5ad."
        )
    readme = ["cells.h5ad: scanpy AnnData (cells x genes), UMI counts in X.", ann]
    (work / "README.md").write_text(common.readme([*readme, f"Python: {PY_HINT}."]))
    seen = "annotations.csv lists the barcodes in another order than adata.obs_names"
    if twin == "clean":
        evidence = (
            ("README says annotations.csv follows the order of cells.h5ad",) if stated else ()
        )
    else:
        evidence = (
            ("README says annotations.csv is sorted by cell type", seen) if stated else (seen,)
        )
    detected = s.counts[:, gene] >= depth
    true_labels = np.array(s.labels)
    expected = int(((true_labels == target) & detected).sum())
    by_position = np.array([s.labels[i] for i in order])
    naive = expected if twin == "clean" else int(((by_position == target) & detected).sum())
    name = s.genes[gene]
    free = (
        f"count the '{target}' cells (annotations.csv) in which gene {name} has at least {depth} "
        f"counts in cells.h5ad ({PY_HINT})"
    )
    forced = (
        f"copy the cell types of annotations.csv into `adata.obs['celltype']` of cells.h5ad and "
        f"count the '{target}' cells in which gene {name} has at least {depth} counts "
        f"({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F105", seed, twin, tier, route)
    return Task(
        task_id, "F105", "label_alignment", twin, seed, prompt, truth, evidence, tier, route
    )


def _inputs(task: Task) -> tuple[str, str, int]:
    found = re.search(r"'([^']+)' cells.*?gene (\w+) has at least (\d+) counts", task.prompt)
    assert found is not None
    return found.group(1), found.group(2), int(found.group(3))


_SCRIPT = """
import json, pandas as pd, anndata as ad, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
ann = pd.read_csv("annotations.csv", index_col=0)
a.obs["celltype"] = {assign}
x = a[:, {gene!r}].X
x = (x.toarray() if sp.issparse(x) else x).ravel()
print(json.dumps(int(((a.obs["celltype"] == {group!r}).values & (x >= {depth})).sum())))
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, gene, depth = _inputs(task)
    code = _SCRIPT.format(
        assign='ann.loc[a.obs_names, "celltype"].values', gene=gene, group=group, depth=depth
    )
    return result(solve_with(tools, code, work), "labels joined by barcode (ann.loc[obs_names])")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, gene, depth = _inputs(task)
    code = _SCRIPT.format(assign='ann["celltype"].values', gene=gene, group=group, depth=depth)
    return result(solve_with(tools, code, work), "labels pasted by position (.values)")


# adata.obs = table, adata.obs['c'] = v, adata.obs.loc[:, 'c'] = v, adata.obs.c = v
ASSIGN = re.compile(
    r"\.obs(?P<column>\[[^\]]*\]|\.loc\[[^\]]*\]|\.(?!loc\b|assign\b|insert\b)\w+)?"
    r"\s*=(?!=)\s*(?P<rhs>[^\n;]*)"
)
CALL = re.compile(r"\.obs\.(?:assign|insert)\s*\((?P<rhs>[^\n;]*)")  # adata.obs.assign(c=v)
WORDS = re.compile(r"\w+")
POSITIONAL_RHS = re.compile(r"\.values|\.to_numpy\(|\.tolist\(|np\.array\(|\blist\(")
WHOLE_TABLE_RHS = re.compile(r"\s*(?!\w+\.obs\b)\w+\s*['\")]*\s*$|.*\bread_(?:csv|table)\b")
ALIGNED = re.compile(r"\.loc\[|\.reindex\(|\.join\(|\.merge\(|\.map\(|set_index\(")
# a table put in the order of the cells: `ann = ann.loc[adata.obs_names]`
REORDERED = re.compile(
    r"(?:\.loc\[|\.reindex\(|\.join\(|\.merge\()[^\n;]{0,300}(?:obs_names|\.obs\b|\.index\b)"
)


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command copies the annotations into adata.obs by position: a column assigned
    from `.values` / a list / an array, or adata.obs replaced by the table, with no barcode
    alignment (.loc, reindex, join, merge, map) on the right-hand side of that statement or in the
    statement that put the table in the order of the cells.

    Statements are judged one by one, so a command that aligns one assignment and pastes another
    still executed the trapped route. The column can be set as obs['c'], obs.loc[:, 'c'], obs.c,
    obs.assign(c=...) or obs.insert(i, 'c', ...).
    """
    if "annotations.csv" not in "\n".join(commands):
        return False
    for command in commands:
        reordered: set[str] = set()
        for statement in statements(command):
            pasted = [(found["column"], found["rhs"]) for found in ASSIGN.finditer(statement)]
            pasted += [("call", found["rhs"]) for found in CALL.finditer(statement)]
            for column, rhs in pasted:
                if ALIGNED.search(rhs) or reordered.intersection(WORDS.findall(rhs)):
                    continue
                if column is None and WHOLE_TABLE_RHS.match(rhs):
                    return True
                if column is not None and POSITIONAL_RHS.search(rhs):
                    return True
            for name, rhs in assignments(statement):
                if REORDERED.search(rhs):
                    reordered.add(name)
    return False


SPEC = FamilySpec(
    id="F105",
    slot="label_alignment",
    source="R2-OMX-14",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"annotations.csv", "README.md"}),
    locus="data",
)
