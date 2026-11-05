"""F106 mudata_union: `MuData.n_obs` counts the cells of ANY modality, not of all of them.

Source: the mudata documentation of MuData (the global obs axis is the union of the modalities'
observations; `obsm['rna']` / `obsm['atac']` hold the presence masks) and the scverse multimodal
tutorials. Verified here (mudata 0.3.10, anndata 0.11.4): an RNA modality with cells c0..c9 and an
ATAC modality with cells c3..c12 give `MuData.n_obs == 13` (and `shape == (13, 14)`,
`len(obs_names) == 13`, `obs.shape == (13, 2)`) where 7 cells have both profiles; the modalities
themselves report 10 and 10. Exit 0; reading prints only a FutureWarning about `.update()` pulling
obs columns, which says nothing about cells.

The twins hold the same cells. cells.h5mu has modalities rna (genes) and atac (peaks); clean: every
cell is in both; faulted: the same cells plus cells that only have an RNA or only an ATAC profile.
Task: how many cells have both an RNA and an ATAC profile. `mdata.n_obs` is the answer on the clean
twin and the union on the faulted one.

  explicit  the README says whether every cell is in both modalities
  implicit  the README only says "multimodal MuData"; the modality sizes against n_obs are the
            evidence
  forced    the task sentence dictates `mudata.read_h5mu` and the n_obs of the MuData
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f101_de_use_raw import (
    PY_HINT,
    assignments,
    barcodes,
    gene_symbols,
    make_rng,
    result,
    run_builder,
    runs_trapped_route,
    solve_with,
)
from science_harness.bench.tasks.tools import ToolEnv

_BUILDER = r"""
import json, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, mudata as md
src, out = sys.argv[1], sys.argv[2]
spec = json.load(open(src + "/spec.json"))
arr = np.load(src + "/arr.npz", allow_pickle=False)
mods = {}
for name, info in spec["modalities"].items():
    obs = pd.DataFrame(index=pd.Index(info["obs_names"]))
    var = pd.DataFrame(index=pd.Index(info["var_names"]))
    mods[name] = ad.AnnData(sp.csr_matrix(arr[name]), obs=obs, var=var)
md.MuData(mods).write_h5mu(out)
"""


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(106, seed)
    n_both = int(rng.integers(90, 121))
    only_rna, only_atac = int(rng.integers(8, 25)), int(rng.integers(8, 25))
    cells = barcodes(rng, n_both + only_rna + only_atac)
    both, rna_only, atac_only = (
        cells[:n_both],
        cells[n_both : n_both + only_rna],
        cells[n_both + only_rna :],
    )
    n_genes, n_peaks = int(rng.integers(60, 91)), int(rng.integers(40, 61))
    genes, peaks = (
        gene_symbols(rng, n_genes),
        [f"chr{rng.integers(1, 20)}:{i}" for i in range(n_peaks)],
    )
    rna_cells = list(both) + (list(rna_only) if twin == "faulted" else [])
    atac_cells = list(both) + (list(atac_only) if twin == "faulted" else [])
    rng.shuffle(rna_cells)
    rng.shuffle(atac_cells)
    rna = rng.poisson(rng.uniform(0.5, 4.0, size=n_genes), size=(len(rna_cells), n_genes))
    atac = rng.poisson(rng.uniform(0.2, 2.0, size=n_peaks), size=(len(atac_cells), n_peaks))
    spec = {
        "modalities": {
            "rna": {"obs_names": rna_cells, "var_names": genes},
            "atac": {"obs_names": atac_cells, "var_names": peaks},
        }
    }
    tmp = Path(tempfile.mkdtemp(prefix="sc-build-"))
    try:
        (tmp / "spec.json").write_text(json.dumps(spec))
        np.savez(tmp / "arr.npz", rna=rna.astype(np.float32), atac=atac.astype(np.float32))
        run_builder(
            tools, _BUILDER, [str(tmp), str(work / "cells.h5mu")], work, "building cells.h5mu"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    stated = tier == "explicit"
    line = (
        "cells.h5mu: a MuData (mudata) of one multiome experiment with modalities 'rna' (genes) "
        "and 'atac' (peaks), UMI / fragment counts in X."
    )
    if stated:
        line += (
            " Every cell has both an RNA and an ATAC profile."
            if twin == "clean"
            else " Some cells have only an RNA profile and some only an ATAC profile."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    union = len(set(rna_cells) | set(atac_cells))
    seen = f"rna has {len(rna_cells)} cells, atac {len(atac_cells)}, the MuData has {union}"
    if twin == "clean":
        evidence = ("README says every cell has both profiles",) if stated else ()
    else:
        evidence = ("README says some cells lack a modality", seen) if stated else (seen,)
    free = (
        "count the cells of cells.h5mu that have both an RNA profile and an ATAC profile "
        f"({PY_HINT})"
    )
    forced = (
        "read cells.h5mu with `mudata.read_h5mu` and report the number of cells (n_obs of the "
        f"MuData) that have both an RNA profile and an ATAC profile ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {"kind": "int", "expected": n_both, "naive": n_both if twin == "clean" else union}
    task_id = common.task_id("F106", seed, twin, tier, route)
    return Task(task_id, "F106", "cell_set", twin, seed, prompt, truth, evidence, tier, route)


_SCRIPT = """
import json, mudata as md
m = md.read_h5mu("cells.h5mu")
{count}
"""
_REF = "print(json.dumps(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names))))"
_NAIVE = "print(json.dumps(int(m.n_obs)))"


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    flags = ["the MuData n_obs is the union of the modalities: intersected the obs_names"]
    return result(
        solve_with(tools, _SCRIPT.format(count=_REF), work),
        "intersection of the modalities' obs_names",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(solve_with(tools, _SCRIPT.format(count=_NAIVE), work), "n_obs of the MuData")


MUDATA_FILE = re.compile(r"read_h5mu|cells\.h5mu")
CHAINED = r"read_h5mu\([^)]*\)\s*\.(?:n_obs|obs_names|shape)\b"
# keeps the cells that are in every modality: the n_obs after it is the intersection
INTERSECTED = re.compile(r"\bintersect_obs\b")


def mudata_count(names: set[str]) -> re.Pattern[str]:
    """What counts the cells of a MuData held in a variable of `names`: n_obs, shape, len()."""
    if not names:
        return re.compile(CHAINED)
    var = "|".join(sorted(map(re.escape, names)))
    return re.compile(
        rf"(?<![\w.\]])(?:{var})\.(?:n_obs|shape|obs\.shape)\b"
        rf"|\blen\(\s*(?:{var})(?:\.obs(?:_names)?)?\s*\)|{CHAINED}"
    )


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command counts the cells of the MuData itself (n_obs, shape, len) of
    cells.h5mu before it intersects the modalities (intersect_obs).

    Only the variable that holds the MuData counts, so the shape of a modality or a count of the
    intersection (`len(set(m.mod['rna'].obs_names) & ...)`) is not the trapped route, and a
    command that prints n_obs next to the modalities' own sizes or ends in `2>&1 | tail` is.
    """
    for command in commands:
        names = {name for name, rhs in assignments(command) if MUDATA_FILE.search(rhs)}
        count = mudata_count(names)
        if runs_trapped_route([command], count, INTERSECTED, context=MUDATA_FILE):
            return True
    return False


SPEC = FamilySpec(
    id="F106",
    slot="cell_set",
    source="mudata docs",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5mu", "README.md"}),
    locus="tool_default",
)
