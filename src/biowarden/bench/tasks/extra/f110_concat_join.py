"""F110 concat_join: `anndata.concat` keeps the genes of every batch only if you say join='outer'.

Source SCD-22 (documented, no incident): `anndata.concat(adatas, join="inner")` is the default;
"inner" is the intersection of the variables, so genes that one batch lacks vanish from the merged
object. Verified here (anndata 0.11.4): two batches of 8 genes with 6 genes each (g0-g5 and g2-g7)
give a 22 x 4 object by default and 22 x 8 with join='outer'; no warning, exit 0. A batch file that
was filtered per batch (genes with no counts dropped, the usual `sc.pp.filter_genes(min_cells=1)`)
loses its batch-specific genes.

The twins hold the same counts. Clean: both files keep every gene of the panel (genes with no
counts are zero columns). Faulted: each file drops its zero-count genes. Task: how many genes have
a non-zero total summed over the cells of both batches. Roughly a fifth of the genes are expressed
in one batch only.

  explicit  the README says whether the files were filtered per batch
  implicit  the README only says "two batches"; the unequal var_names are the evidence
  forced    the task sentence dictates `anndata.concat` of the two files with its defaults
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


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(110, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    n_genes = int(rng.integers(150, 211))
    s = synth(rng, int(rng.integers(120, 181)), n_genes, labels, mean_log=0.8)
    in_a = np.arange(len(s.cells)) < len(s.cells) // 2
    counts = s.counts.copy()
    pool = rng.permutation(n_genes)
    k = max(8, n_genes // 10)
    only_a, only_b = pool[:k], pool[k : 2 * k]  # expressed in one batch only
    counts[np.ix_(~in_a, only_a)] = 0
    counts[np.ix_(in_a, only_b)] = 0
    nonzero = counts.sum(axis=0) > 0
    expected = int(nonzero.sum())
    in_both = (counts[in_a].sum(axis=0) > 0) & (counts[~in_a].sum(axis=0) > 0)
    naive = expected if twin == "clean" else int(in_both.sum())
    for name, rows in (("batch1", in_a), ("batch2", ~in_a)):
        block = counts[rows]
        keep = np.ones(n_genes, dtype=bool) if twin == "clean" else block.sum(axis=0) > 0
        write_h5ad(
            work,
            f"{name}.h5ad",
            block[:, keep].astype(np.float32),
            [c for c, r in zip(s.cells, rows, strict=True) if r],
            [g for g, kp in zip(s.genes, keep, strict=True) if kp],
            tools=tools,
        )
    stated = tier == "explicit"
    line = (
        "batch1.h5ad, batch2.h5ad: two sequencing batches of one study, scanpy AnnData, "
        "counts in X."
    )
    if stated:
        line += (
            " Both files keep every gene of the panel."
            if twin == "clean"
            else " Genes with no counts in a batch were dropped from that batch's file."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = "batch1.h5ad and batch2.h5ad have different var_names"
    if twin == "clean":
        evidence = ("README says both files keep every gene",) if stated else ()
    else:
        evidence = (
            ("README says zero-count genes were dropped per batch", seen) if stated else (seen,)
        )
    free = (
        "merge batch1.h5ad and batch2.h5ad into one AnnData and report how many genes have a "
        f"non-zero total count summed over all the cells of both batches ({PY_HINT})"
    )
    forced = (
        "merge batch1.h5ad and batch2.h5ad with `anndata.concat([a, b])` and report how many "
        f"genes have a non-zero total count summed over all the cells of both batches ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON integer")
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F110", seed, twin, tier, route)
    return Task(task_id, "F110", "gene_set", twin, seed, prompt, truth, evidence, tier, route)


_SCRIPT = """
import json, numpy as np, anndata as ad, scipy.sparse as sp
a, b = ad.read_h5ad("batch1.h5ad"), ad.read_h5ad("batch2.h5ad")
m = ad.concat([a, b]{join})
tot = np.asarray(m.X.sum(axis=0)).ravel()
print(json.dumps(int((tot > 0).sum())))
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    code = _SCRIPT.format(join=", join='outer'")
    flags = ["batches have different genes: merged with join='outer'"]
    return result(
        solve_with(tools, code, work),
        "anndata.concat with join='outer'",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(solve_with(tools, _SCRIPT.format(join=""), work), "anndata.concat defaults")


# anndata.concat / AnnData.concatenate, not pandas.concat or numpy.concatenate
CONCAT = re.compile(
    r"(?<!pd)(?<!np)(?<!pandas)(?<!numpy)\.concat(?:enate)?\s*\(|(?<![\w.])concat\s*\("
)
OUTER = re.compile(r"join\s*=\s*['\"]outer['\"]")  # an argument of the call: merge='same' is not
UNION = re.compile(r"\bunion\b")  # a gene union built by hand handles the calls after it


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command concatenates the batches without join='outer'.

    A call that passes join='outer' is set aside, a union built by hand handles the calls after it,
    and join='inner' with merge='same' is still the inner join.
    """
    return runs_trapped_route(commands, CONCAT, UNION, OUTER)


SPEC = FamilySpec(
    id="F110",
    slot="gene_set",
    source="SCD-22",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"batch1.h5ad", "batch2.h5ad", "README.md"}),
    locus="tool_default",
)
