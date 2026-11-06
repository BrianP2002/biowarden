"""F108 control_label: control cells labelled 'ctrl' in one file and 'non-targeting' in another.

Source: perturbation datasets (GEARS, CPA, Perturb-seq screens) name controls differently per
study and per library: 'ctrl', 'control', 'non-targeting', 'NT', 'scramble'. (GEARS documents the
literal 'ctrl' in obs['condition'] as its control label; GEARS is not installed here, so nothing
about it was run.) A filter written for one spelling matches no cell in a file
that uses another: `a[a.obs['perturbation'] == 'ctrl']` returns a 0 x n_vars AnnData with no
warning (verified: anndata 0.11.4; a mean over it raises ZeroDivisionError for sparse X, so the
analysis must pool the files first). When two files are pooled the controls of the file with the
other spelling silently drop out, so the baseline comes from one batch only.

The twins hold the same cells. screen_a.h5ad and screen_b.h5ad (two sgRNA libraries) both label
the controls 'ctrl' (clean), or screen_b.h5ad labels them with another spelling (faulted). Gene G
reads higher in screen_b (a batch effect), so controls of screen_a alone give a different baseline.
Task: the mean log-normalised expression of G in the cells of one perturbation minus the mean in
the control cells, pooling both files.

  explicit  the README states the control label of each file
  implicit  the README says "obs['perturbation'], controls included"; value_counts is the evidence
  forced    the task sentence dictates `adata[adata.obs['perturbation'] == 'ctrl']`
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
    PY_HINT,
    log1p_cp10k,
    make_rng,
    result,
    solve_with,
    synth,
    write_h5ad,
)
from biowarden.bench.tasks.tools import ToolEnv

OTHER_LABELS = ("non-targeting", "control", "NT", "scramble")
TOL = 0.02
MIN_GAP = 0.15


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(108, seed)
    n_genes = int(rng.integers(150, 201))
    s = synth(
        rng,
        int(rng.integers(220, 281)),
        n_genes,
        ["control", "P1", "P2"],
        sizes=[1.6, 1.0, 1.0],
        n_markers=6,
        fold=(1.6, 2.4),
    )
    target_label = "P1"
    in_b = rng.random(len(s.cells)) < 0.5
    gene = s.markers[target_label][int(rng.integers(0, len(s.markers[target_label])))]
    counts = s.counts.copy()
    boost = 1.7
    while True:  # screen_b reads higher for the gene: controls of screen_a alone are lower
        trial = counts.copy()
        trial[in_b, gene] = rng.poisson(counts[in_b, gene] * boost)
        logged = log1p_cp10k(trial).astype(np.float64)
        is_ctrl = np.array([lab == "control" for lab in s.labels])
        is_p = np.array([lab == target_label for lab in s.labels])
        pooled = logged[is_p, gene].mean() - logged[is_ctrl, gene].mean()
        only_a = logged[is_p, gene].mean() - logged[is_ctrl & ~in_b, gene].mean()
        if abs(pooled - only_a) >= MIN_GAP:
            break
        boost += 0.3
    counts = trial
    perts = [s.genes[i] for i in rng.choice(n_genes, size=2, replace=False)]
    name_of = {"control": "ctrl", "P1": perts[0], "P2": perts[1]}
    other = OTHER_LABELS[int(rng.integers(0, len(OTHER_LABELS)))]
    label_b = "ctrl" if twin == "clean" else other
    x32 = logged.astype(np.float32)
    for name, mask, ctrl_label in (("a", ~in_b, "ctrl"), ("b", in_b, label_b)):
        labs = [
            ctrl_label if lab == "control" else name_of[lab]
            for lab, keep in zip(s.labels, mask, strict=True)
            if keep
        ]
        write_h5ad(
            work,
            f"screen_{name}.h5ad",
            x32[mask],
            [c for c, keep in zip(s.cells, mask, strict=True) if keep],
            s.genes,
            obs={"perturbation": labs},
            uns={"log1p": {"base": None}},
            tools=tools,
        )
    stated = tier == "explicit"
    line = (
        "screen_a.h5ad, screen_b.h5ad: two sgRNA libraries of one CRISPRi screen, scanpy AnnData "
        "(cells x genes), X is log1p of counts per 10,000, obs['perturbation'] holds the targeted "
        "gene or the control label."
    )
    if stated:
        line += (
            " Control cells are labelled 'ctrl' in both files."
            if twin == "clean"
            else f" Control cells are labelled 'ctrl' in screen_a.h5ad and '{other}' in screen_b."
        )
    (work / "README.md").write_text(common.readme([line, f"Python: {PY_HINT}."]))
    seen = f"obs['perturbation'] of screen_b.h5ad has '{label_b}' and no 'ctrl'"
    if twin == "clean":
        evidence = ("README says controls are 'ctrl' in both files",) if stated else ()
    else:
        evidence = (f"README says screen_b labels controls '{other}'", seen) if stated else (seen,)
    gname = s.genes[gene]
    free = (
        f"report the mean X of gene {gname} in the cells targeting {perts[0]} minus its mean in "
        f"the control cells, pooling the cells of screen_a.h5ad and screen_b.h5ad ({PY_HINT})"
    )
    forced = (
        f"pool screen_a.h5ad and screen_b.h5ad and select controls with "
        f"`adata[adata.obs['perturbation'] == 'ctrl']`; report the mean X of gene {gname} in the "
        f"cells targeting {perts[0]} minus its mean in those control cells ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON number")
    truth = {
        "kind": "float",
        "expected": float(pooled),
        "naive": float(pooled) if twin == "clean" else float(only_a),
        "tol": TOL,
    }
    task_id = common.task_id("F108", seed, twin, tier, route)
    return Task(task_id, "F108", "control_label", twin, seed, prompt, truth, evidence, tier, route)


def _inputs(task: Task) -> tuple[str, str]:
    found = re.search(r"gene (\w+) in the cells targeting (\w+)", task.prompt)
    assert found is not None
    return found.group(1), found.group(2)


_SCRIPT = """
import json, numpy as np, anndata as ad, scipy.sparse as sp
a = ad.concat([ad.read_h5ad("screen_a.h5ad"), ad.read_h5ad("screen_b.h5ad")])
{ctrl}
col = a[:, {gene!r}].X
col = np.asarray(col.todense() if sp.issparse(col) else col).ravel()
pert = (a.obs["perturbation"] == {target!r}).values
print(json.dumps(float(col[pert].mean() - col[ctrl].mean())))
"""
_CTRL_REF = 'ctrl = (~a.obs["perturbation"].isin(a.var_names)).values'
_CTRL_NAIVE = 'ctrl = (a.obs["perturbation"] == "ctrl").values'


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    gene, target = _inputs(task)
    code = _SCRIPT.format(ctrl=_CTRL_REF, gene=gene, target=target)
    flags = ["controls are the cells whose label is not a gene: one spelling per file"]
    return result(
        solve_with(tools, code, work),
        "pooled controls found by label, not by the literal 'ctrl'",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    gene, target = _inputs(task)
    code = _SCRIPT.format(ctrl=_CTRL_NAIVE, gene=gene, target=target)
    return result(solve_with(tools, code, work), "controls selected with == 'ctrl'")


LABEL = r"(?:ctrl|control|non-targeting|NT)"
HELD = re.compile(rf"""\b(\w+)\s*=(?!=)\s*['"]{LABEL}['"]""")  # L = 'ctrl'
LITERAL = re.compile(rf"""[=!]=\s*['"]{LABEL}['"]|\.isin\(\s*\[\s*['"]{LABEL}['"]\s*\]\s*\)""")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command selects controls with one literal label ('ctrl', 'control', ...):
    `== 'ctrl'`, `!= 'ctrl'`, `.isin(['ctrl'])`, or `== L` for a variable L set to such a label.

    The whole text is searched: a command that lists the labels (value_counts, unique) or matches
    both spellings and also selects with the one literal still executed the trapped route, which
    is how Haiku's three silent F108 trials looked.
    """
    text = "\n".join(commands)
    held = "|".join(sorted({found.group(1) for found in HELD.finditer(text)}))
    named = re.search(rf"[=!]=\s*(?:{held})\b", text) if held else None
    return bool((LITERAL.search(text) or named) and "screen_" in text)


SPEC = FamilySpec(
    id="F108",
    slot="control_label",
    source="SCD-06",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"screen_b.h5ad", "README.md"}),
    locus="data",
)
