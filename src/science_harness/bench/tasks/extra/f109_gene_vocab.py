"""F109 gene_vocab: a model vocabulary of gene symbols against an AnnData indexed by Ensembl ids.

Source: the input contract of single-cell foundation models (scGPT: a vocab.json of gene symbols;
its preprocessing only logs "match 0/N genes in vocabulary" and carries on with no genes) and the
CELLxGENE schema (var index = Ensembl id, symbols in var['feature_name']). Verified here (anndata
0.11.4): `set(adata.var_names) & set(vocab)` on such an object is empty without a warning, and the
symbol column gives the real overlap.

The twins hold the same genes and the same vocabulary. cells.h5ad has gene symbols as var_names
and the Ensembl ids in var['feature_id'] (clean), or Ensembl ids as var_names and the symbols in
var['feature_name'] (faulted). Task: the symbols of the genes the model knows.

  explicit  the README states what the var index and the other column hold
  implicit  the README says "model vocabulary of gene tokens"; the id style of var_names is the
            evidence
  forced    the task sentence dictates `adata.var_names` against the vocabulary keys
"""

from __future__ import annotations

import json
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
    assignments,
    gene_symbols,
    make_rng,
    result,
    runs_trapped_route,
    solve_with,
    synth,
    write_h5ad,
)
from science_harness.bench.tasks.extra.f107_mtx_orientation import ensembl_ids
from science_harness.bench.tasks.tools import ToolEnv

SPECIAL = ("<pad>", "<cls>", "<eoc>")


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(109, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    n_genes = int(rng.integers(150, 221))
    s = synth(rng, int(rng.integers(60, 91)), n_genes, labels)
    ids = ensembl_ids(rng, n_genes)
    known = sorted(rng.choice(n_genes, size=int(n_genes * rng.uniform(0.6, 0.8)), replace=False))
    decoys = gene_symbols(rng, int(rng.integers(80, 121)), avoid=set(s.genes))
    tokens = [*SPECIAL, *(s.genes[i] for i in known), *decoys]
    order = rng.permutation(len(tokens) - len(SPECIAL)) + len(SPECIAL)
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    vocab.update({tokens[k]: i + len(SPECIAL) for i, k in enumerate(order)})
    (work / "vocab.json").write_text(json.dumps(vocab, indent=0, sort_keys=False) + "\n")
    if twin == "clean":
        names, col, other = s.genes, "feature_id", ids
    else:
        names, col, other = ids, "feature_name", s.genes
    write_h5ad(
        work,
        "cells.h5ad",
        s.counts.astype(np.float32),
        s.cells,
        names,
        obs={"celltype": s.labels},
        var={col: other},
        tools=tools,
    )
    stated = tier == "explicit"
    obj = "cells.h5ad: scanpy AnnData (cells x genes), UMI counts in X."
    voc = "vocab.json: vocabulary of a pretrained single-cell model, JSON object token -> id."
    if stated:
        obj += (
            " var_names are gene symbols; var['feature_id'] holds the Ensembl ids."
            if twin == "clean"
            else " var_names are Ensembl gene ids; var['feature_name'] holds the gene symbols."
        )
        voc += " The gene tokens are gene symbols."
    (work / "README.md").write_text(common.readme([obj, voc, f"Python: {PY_HINT}."]))
    seen = f"var_names look like {names[0]} while the vocabulary tokens look like {tokens[3]}"
    if twin == "clean":
        evidence = ("README says var_names are gene symbols",) if stated else ()
    else:
        evidence = ("README says var_names are Ensembl ids", seen) if stated else (seen,)
    expected = sorted(s.genes[i] for i in known)
    free = (
        f"list the gene symbols of the genes in cells.h5ad that are tokens of the model "
        f"vocabulary vocab.json ({PY_HINT})"
    )
    forced = (
        f"intersect `adata.var_names` of cells.h5ad with the keys of vocab.json and list the "
        f"genes in common ({PY_HINT})"
    )
    prompt = common.prompt_for(route, free, forced, "a JSON list of gene symbols")
    truth = {"kind": "id_list", "expected": expected, "naive": expected if twin == "clean" else []}
    task_id = common.task_id("F109", seed, twin, tier, route)
    return Task(task_id, "F109", "id_namespace", twin, seed, prompt, truth, evidence, tier, route)


_SCRIPT = """
import json, anndata as ad
a = ad.read_h5ad("cells.h5ad")
vocab = set(json.load(open("vocab.json")))
{pick}print(json.dumps(sorted(set(names) & vocab)))
"""
_PICK_REF = """cands = {None: list(a.var_names)}
cands.update({c: a.var[c].astype(str).tolist() for c in a.var.columns})
best = max(cands, key=lambda c: len(vocab & set(cands[c])))
names = cands[best]
"""
_PICK_NAIVE = "names = list(a.var_names)\n"


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    flags = ["var_names are not in the vocabulary's namespace: matched on the symbol column"]
    return result(
        solve_with(tools, _SCRIPT.format(pick=_PICK_REF), work),
        "overlap of the vocabulary with the best-matching gene-name column",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(
        solve_with(tools, _SCRIPT.format(pick=_PICK_NAIVE), work),
        "var_names against the vocabulary",
    )


VOCAB = re.compile(r"vocab")
# vn = list(a.var_names): a name for the var_names (or the var index) the next statements match
NAMES_OF = re.compile(r"\bvar_names\b|\bvar\.index\b")
# var_names (or the index) replaced by another column: the names matched afterwards are the symbols
RENAMED = re.compile(r"\.var_names\s*=(?!=)|\bvar\.index\s*=(?!=)")


def matches_names(names: set[str]) -> re.Pattern[str]:
    """A statement that matches the var names (or a variable `names` holds them in) against
    something: &, intersection, isin or `in`."""
    held = "|".join(sorted(map(re.escape, names)))
    return re.compile(
        rf"(?s)\A(?=.*(?:\b(?:{held})\b|\bvar\.index\b))(?=.*(?:&|\bintersection\b|\.isin\(|\bin\b))"
    )


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command matches var_names (or the var index) against the vocabulary, with no
    other var column put in their place before.

    Reading another column in the same command (`print(a.var.columns)`, `a.var['feature_name']`)
    does not set the command aside: the statement that intersects var_names with the vocabulary
    still ran, and in Haiku's pilot trials it printed an empty intersection.
    """
    for command in commands:
        names = {"var_names", *(name for name, rhs in assignments(command) if NAMES_OF.search(rhs))}
        if runs_trapped_route([command], matches_names(names), RENAMED, context=VOCAB):
            return True
    return False


SPEC = FamilySpec(
    id="F109",
    slot="id_namespace",
    source="SCD-34",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="data",
)
