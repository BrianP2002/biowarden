"""F111 mito_case: the mitochondrial-gene prefix is 'MT-' for human and 'mt-' for mouse.

Source: the scanpy and Seurat QC tutorials (`adata.var_names.str.startswith("MT-")`,
`PercentageFeatureSet(pattern = "^MT-")`). The match is case-sensitive and mouse genes are written
'mt-Nd1'. Verified here (scanpy 1.11.5): with 13 mouse mitochondrial genes `startswith('MT-')`
flags 0 genes, `calculate_qc_metrics(qc_vars=['mt'])` then gives pct_counts_mt of exactly 0.0 for
every cell, no warning; the case-insensitive match flags the 13 genes.

The twins hold the same counts. The gene symbols are upper case with 'MT-ND1' style mitochondrial
genes (clean, human) or capitalised with 'mt-Nd1' style ones (faulted, mouse). Task: how many cells
have less than T percent of their counts in mitochondrial genes. The tutorial prefix keeps every
cell of the mouse object.

  explicit  the README says the species
  implicit  the README says "single-cell counts"; the gene names are the evidence
  forced    the task sentence dictates `startswith('MT-')` for the mitochondrial genes
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
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
    write_h5ad,
)
from biowarden.bench.tasks.tools import ToolEnv

MITO = (
    "MT-ND1 MT-ND2 MT-CO1 MT-ATP8 MT-ATP6 MT-CO3 MT-ND3 MT-ND4L MT-ND4 MT-ND5 MT-ND6 MT-CYB MT-CO2"
).split()
MARGIN = 0.25  # no cell within this many percentage points of the threshold


def mouse(symbol: str) -> str:
    """HGNC-style upper case symbol to the mouse style: 'MT-ND1' -> 'mt-Nd1', 'ABCD3' -> 'Abcd3'."""
    head, sep, tail = symbol.partition("-")
    if sep:
        return head.lower() + "-" + tail.capitalize()
    return symbol.capitalize()


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(111, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    n_genes = int(rng.integers(150, 221))
    s = synth(rng, int(rng.integers(120, 181)), n_genes, labels)
    counts = s.counts.copy()
    where = sorted(int(i) for i in rng.choice(n_genes, size=len(MITO), replace=False))
    names = list(s.genes)
    for slot, symbol in zip(where, MITO, strict=True):
        names[slot] = symbol
    stress = np.exp(rng.normal(0.0, 0.75, size=len(counts)))
    level = rng.uniform(2.0, 6.0, size=len(MITO))
    counts[:, where] = rng.poisson(level[None, :] * stress[:, None])
    totals = counts.sum(axis=1)
    pct = counts[:, where].sum(axis=1) / totals * 100
    threshold = float(np.round(np.quantile(pct, 0.75), 1))
    while np.min(np.abs(pct - threshold)) < MARGIN:
        threshold += 0.1
    threshold = round(threshold, 1)  # the prompt prints it: 8.7, not 8.699999999999998
    expected = int((pct < threshold).sum())
    naive = expected if twin == "clean" else len(pct)
    var_names = names if twin == "clean" else [mouse(n) for n in names]
    write_h5ad(work, "cells.h5ad", counts.astype(np.float32), s.cells, var_names, tools=tools)
    stated = tier == "explicit"
    line = "cells.h5ad: scanpy AnnData (cells x genes), single-cell UMI counts in X."
    if stated:
        line += " Species: " + ("human." if twin == "clean" else "mouse.")
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = "mitochondrial genes are written mt-Nd1, mt-Co1 (mouse style, lower case prefix)"
    if twin == "clean":
        evidence = ("README says the species is human",) if stated else ()
    else:
        evidence = ("README says the species is mouse", seen) if stated else (seen,)
    free = (
        f"count the cells in cells.h5ad with less than {threshold} percent of their counts in "
        f"mitochondrial genes ({PY_HINT})"
    )
    forced = (
        f"mark the mitochondrial genes of cells.h5ad with `adata.var_names.str.startswith('MT-')`,"
        f" run `sc.pp.calculate_qc_metrics` and count the cells with pct_counts_mt below "
        f"{threshold} ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F111", seed, twin, tier, route)
    return Task(task_id, "F111", "gene_naming", twin, seed, prompt, truth, evidence, tier, route)


def _threshold(task: Task) -> float:
    found = re.search(r"(?:less than|below) (\d+(?:\.\d+)?)", task.prompt)
    assert found is not None
    return float(found.group(1))


_SCRIPT = """
import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
a.var["mt"] = a.var_names{match}.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
print(json.dumps(int((a.obs["pct_counts_mt"] < {t}).sum())))
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(match=".str.upper()", t=_threshold(task))
    flags = ["mouse gene names: mitochondrial prefix matched case-insensitively"]
    return result(
        solve_with(tools, code, work),
        "mitochondrial genes by case-insensitive MT- prefix",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(
        solve_with(tools, _SCRIPT.format(match="", t=_threshold(task)), work),
        "mitochondrial genes by startswith('MT-')",
    )


PREFIX = re.compile(
    r"""startswith\(\s*\(?\s*['"]MT-['"]|\^MT-|str\.match\(\s*['"]MT-|str\.contains\(\s*['"]MT-"""
    r"""|\[:3\]\s*==\s*['"]MT-['"]|['"]MT-['"]\s+in\s+\w"""
)
# the case is folded, or the lower-case prefix listed, in the statement itself
LOCAL = re.compile(
    r"\.str\.(?:upper|lower)\(|re\.I|IGNORECASE|\(\?i\)|['\"]mt-['\"]|\bcase\s*=\s*False|"
    r"\bmt-|\.upper\(\)\.startswith|\.lower\(\)\.startswith"
)
# the gene names are folded before the prefix is looked for: handles the statements after it
FOLDED = re.compile(r"\bvar_names\s*=(?!=)[^\n;]{0,300}\.(?:upper|lower)\(")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command marks mitochondrial genes with the upper-case prefix 'MT-' only
    (startswith, str.contains, str.match, a slice or `in`).

    A statement that folds the case or lists the lower-case prefix is set aside; listing 'mt-'
    in another statement of the command does not hide a statement that uses 'MT-' alone.
    """
    return runs_trapped_route(commands, PREFIX, FOLDED, LOCAL)


SPEC = FamilySpec(
    id="F111",
    slot="gene_naming",
    source="SCD-34",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
