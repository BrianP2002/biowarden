"""F101 de_use_raw: scanpy `rank_genes_groups` reads `.raw` by default.

Source R2-OMX-01, SCD-22 and the scanpy documentation of `tl.rank_genes_groups`: "use_raw: Use
raw attribute of adata if present" and its default `use_raw=None`, which means "use raw if present".
An AnnData whose `.raw` holds another matrix than `.X` therefore gives a differential-expression
table computed on `.raw` without a word. The logfoldchanges column treats whatever it is given as
log1p data: it takes expm1 of the group means. Verified here (scanpy 1.11.5, anndata 0.11.4, seed
0 of this family, Wilcoxon, group against rest): with X = log1p(counts per 10,000) and `.raw.X` =
counts per 10,000 (not logged, non-integer, so scanpy prints no warning; with integer counts in
`.raw` it prints "It seems you use rank_genes_groups on the raw count data"), the default call
returns log2 fold changes of 346, 305 and 358 for three marker genes where `use_raw=False` returns
3.90, 4.29 and 3.96 (exit 0, empty stderr), with the same scores and p-values (the Wilcoxon
statistic only depends on ranks, and log1p is monotone). The reported genes are the markers with
the lowest raw levels: expm1 of a group mean above about 700 overflows to inf.

The twins hold the same cells and the same X. Clean: `.raw` is a copy of X taken after the log1p
(what the scanpy tutorials do: `adata.raw = adata`), so the default is right. Faulted: `.raw` was
saved before the log1p, as counts per 10,000. Task: report the log fold change scanpy gives for
three marker genes of one cell type against all other cells (Wilcoxon, reference "rest").

  explicit  the README says what X and .raw hold
  implicit  the README only describes the object; raw.X max (hundreds) against X max (about 6)
            is the evidence
  forced    the task sentence dictates `sc.tl.rank_genes_groups` with its default arguments

This module also holds the helpers every family of the sc pack shares (synthetic matrices, the
writers for .h5ad / .mtx / .csv / .loom, a runner for python-sc scripts, the statement reader the
`exposed()` checks use).
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from science_harness.bench.runner.exposure import statements
from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

REPO_ROOT = Path(__file__).resolve().parents[5]
CELL_TYPES = (
    "T cell",
    "B cell",
    "NK cell",
    "Monocyte",
    "Dendritic cell",
    "Platelet",
    "Plasma cell",
    "Erythroid",
)
# 14 of 120 Haiku pilot trials took "use `python-sc`" for a directory and ran its bin/python by path
PY_HINT = (
    "run `python-sc script.py`; it is an executable, not a directory, "
    "and has scanpy 1.11 and anndata 0.11"
)


# ---- shared helpers of the sc pack ---------------------------------------------------------------


def sc_tools(tools: ToolEnv | None) -> ToolEnv:
    """The tool environment that knows `python-sc` (a family builder may get tools=None)."""
    if tools is not None:
        return tools
    found = ToolEnv.discover(REPO_ROOT)
    if found is None:
        raise RuntimeError("no tool environment: python-sc is needed to build sc families")
    return found


# A callable (code, cwd) -> (returncode, stdout, stderr) that tests install to run solver snippets
# in a long-lived python-sc process (same interpreter and libraries, no process per call).
RUN_HOOKS: list[Any] = []


def run_py(tools: ToolEnv | None, code: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run `python-sc -c code` in `cwd` the way an agent would (no -I: the cwd is the workspace)."""
    if RUN_HOOKS and not args:
        rc, out, err = RUN_HOOKS[-1](code, cwd)
        return subprocess.CompletedProcess(["python-sc", "-c", code], rc, out, err)
    return sc_tools(tools).run(["python-sc", "-c", code, *args], cwd)


def last_json(stdout: str) -> Any:
    """The JSON value on the last non-empty line of a script's stdout."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def solve_with(tools: ToolEnv | None, code: str, work: Path) -> Any:
    proc = run_py(tools, code, work)
    if proc.returncode != 0:
        raise RuntimeError(f"solver script failed:\n{proc.stderr[-2000:]}")
    return last_json(proc.stdout)


def result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def make_rng(family: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([family, seed])


def gene_symbols(rng: np.random.Generator, n: int, avoid: set[str] = frozenset()) -> list[str]:
    """Unique made-up HGNC-looking symbols (letters then a digit), in a fixed order."""
    letters = np.array(list("ABCDEFGHKLMNPRSTWXYZ"))
    seen: set[str] = set(avoid)
    out: list[str] = []
    while len(out) < n:
        k = int(rng.integers(3, 5))
        name = "".join(rng.choice(letters, size=k)) + str(int(rng.integers(1, 10)))
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def barcodes(rng: np.random.Generator, n: int, suffix: str = "-1") -> list[str]:
    bases = np.array(list("ACGT"))
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        code = "".join(rng.choice(bases, size=16)) + suffix
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


@dataclass(frozen=True)
class Synth:
    counts: np.ndarray  # (cells, genes) int32
    genes: list[str]
    cells: list[str]
    labels: list[str]  # one label per cell
    markers: dict[str, list[int]]  # label -> marker gene indices


def synth(
    rng: np.random.Generator,
    n_cells: int,
    n_genes: int,
    labels: list[str],
    *,
    n_markers: int = 6,
    fold: tuple[float, float] = (6.0, 10.0),
    size_sd: float = 0.3,
    mean_log: float = 1.2,
    sizes: list[float] | None = None,
    target_total: float | None = None,
    size_clip: tuple[float, float] = (0.0, 1e9),
) -> Synth:
    """Counts with planted marker genes per label. Cells are in random order."""
    weights = np.array(sizes if sizes else [1.0] * len(labels), dtype=float)
    weights = weights / weights.sum()
    cell_label = rng.choice(len(labels), size=n_cells, p=weights)
    for k in range(len(labels)):  # every label has at least ten cells
        if (cell_label == k).sum() < 10:
            cell_label[rng.choice(n_cells, size=10, replace=False)] = k
    base = np.exp(rng.normal(mean_log, 0.8, size=n_genes))
    pool = rng.permutation(n_genes)
    markers: dict[str, list[int]] = {}
    mult = np.ones((len(labels), n_genes))
    for k, label in enumerate(labels):
        idx = sorted(int(i) for i in pool[k * n_markers : (k + 1) * n_markers])
        markers[label] = idx
        mult[k, idx] = rng.uniform(fold[0], fold[1], size=len(idx))
    size = np.clip(np.exp(rng.normal(0.0, size_sd, size=n_cells)), *size_clip)
    mu = base[None, :] * mult[cell_label]
    if target_total is not None:  # expected library size = target_total x size, whatever the genes
        mu = mu / mu.sum(axis=1, keepdims=True) * target_total
    mu = mu * size[:, None]
    mu = mu * rng.gamma(6.0, 1.0 / 6.0, size=mu.shape)
    counts = rng.poisson(mu).astype(np.int32)
    genes = gene_symbols(rng, n_genes)
    return Synth(counts, genes, barcodes(rng, n_cells), [labels[k] for k in cell_label], markers)


def cp10k(counts: np.ndarray, target: float = 1e4) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return (counts / np.maximum(totals, 1.0) * target).astype(np.float32)


def log1p_cp10k(counts: np.ndarray) -> np.ndarray:
    return np.log1p(cp10k(counts).astype(np.float64)).astype(np.float32)


def deterministic_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6, mtime=0)


# ---- exposure: the statement reader every family of the pack uses -----------------------------

PRINTS = re.compile(r"(?<![\w.])print\s*\(")
ASSIGNMENT = re.compile(r"[^=\n]*?(?<![\w.])(\w+)\s*=(?!=)\s*(.*)", re.DOTALL)


def assignments(command: str) -> list[tuple[str, str]]:
    """(name, right-hand side) of the first assignment of every statement of a command, in order.

    Read from the left of the statement (`python-sc -c "m = md.read_h5mu(...)` gives `m`), so the
    cost stays linear in the length of the command, which a search for `name = ... something`
    from every `=` does not.
    """
    found = (ASSIGNMENT.match(statement) for statement in statements(command))
    return [(match.group(1), match.group(2)) for match in found if match]


def runs_trapped_route(
    commands: list[str],
    trapped: re.Pattern[str],
    handled: re.Pattern[str] | None = None,
    local: re.Pattern[str] | None = None,
    context: re.Pattern[str] | None = None,
) -> bool:
    """Did some statement of some command run the trapped route with the data left as it was?

    A command is one fresh process, so each is read on its own, statement by statement and in
    order (`statements`: a newline or `;` outside brackets). A statement that matches `handled`
    changes the data for itself and for every statement after it in that command (a cast, a
    back-transform, the counts put into X) unless it only prints; one that matches `local` handles
    only itself (an argument of the trapped call). A statement that matches `trapped` with neither
    in force is the trapped route, even when a safe variant comes later in the same command:
    exposure is "ran it", not "kept the answer". `context` has to match the command as a whole
    (it names the file).
    """
    for command in commands:
        if context is not None and not context.search(command):
            continue
        done = False
        for statement in statements(command):
            if handled is not None and handled.search(statement):
                done = done or not PRINTS.search(statement)
            elif local is not None and local.search(statement):
                continue
            elif trapped.search(statement) and not done:
                return True
    return False


_BUILDER = r"""
import json, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
src, out = sys.argv[1], sys.argv[2]
spec = json.load(open(src + "/spec.json"))
arr = np.load(src + "/arr.npz", allow_pickle=False)

def mat(key, sparse):
    x = arr[key]
    return sp.csr_matrix(x) if sparse else x

def table(cols, names):
    df = pd.DataFrame(index=pd.Index(names, name=None))
    for name, key in cols.items():
        v = arr[key]
        df[name] = v.tolist() if v.dtype.kind in "US" else v
    return df

x = mat("X", spec["sparse"])
obs = table(spec["obs"], spec["obs_names"])
var = table(spec["var"], spec["var_names"])
a = ad.AnnData(x, obs=obs, var=var)
for name in spec["layers"]:
    a.layers[name] = mat("layer_" + name, spec["sparse"])
if spec["raw"] is not None:
    rv = table(spec["raw"]["var"], spec["raw"]["var_names"])
    a.raw = ad.AnnData(mat("raw_X", spec["sparse"]), obs=a.obs[[]], var=rv)
for k, v in spec["uns"].items():
    a.uns[k] = v
a.write_h5ad(out, compression="gzip")
"""


# A callable (code, argv, cwd) -> (returncode, stderr) that the admission cases install to run the
# builder snippet inside a long-lived python-sc process instead of one process per file.
BUILD_HOOKS: list[Any] = []


def run_builder(
    tools: ToolEnv | None, code: str, argv: list[str], work: Path, what: str = "build"
) -> None:
    """Run a builder snippet (python-sc, sys.argv = argv) or raise with its stderr."""
    if BUILD_HOOKS:
        status, err = BUILD_HOOKS[-1](code, argv, work)
    else:
        proc = sc_tools(tools).run(["python-sc", "-I", "-c", code, *argv], work)
        status, err = proc.returncode, proc.stderr
    if status != 0:
        raise RuntimeError(f"{what} failed:\n{err[-2000:]}")


def write_h5ad(
    work: Path,
    name: str,
    X: np.ndarray,
    obs_names: list[str],
    var_names: list[str],
    *,
    obs: dict[str, Any] | None = None,
    var: dict[str, Any] | None = None,
    layers: dict[str, np.ndarray] | None = None,
    raw: dict[str, Any] | None = None,
    uns: dict[str, Any] | None = None,
    sparse: bool = True,
    tools: ToolEnv | None = None,
) -> None:
    """Write an .h5ad with anndata (python-sc): X, obs/var columns, layers, .raw and uns.

    `raw` is {"X": array, "var_names": list (default var_names), "var": {col: values}}.
    """
    arrays: dict[str, np.ndarray] = {"X": np.asarray(X)}
    spec: dict[str, Any] = {
        "sparse": sparse,
        "obs_names": list(obs_names),
        "var_names": list(var_names),
        "obs": {},
        "var": {},
        "layers": [],
        "raw": None,
        "uns": uns or {},
    }
    for prefix, cols in (("obs", obs or {}), ("var", var or {})):
        for col, values in cols.items():
            arrays[f"{prefix}_{col}"] = np.asarray(values)
            spec[prefix][col] = f"{prefix}_{col}"
    for layer, values in (layers or {}).items():
        arrays[f"layer_{layer}"] = np.asarray(values)
        spec["layers"].append(layer)
    if raw is not None:
        arrays["raw_X"] = np.asarray(raw["X"])
        raw_cols = {}
        for col, values in (raw.get("var") or {}).items():
            arrays[f"rawvar_{col}"] = np.asarray(values)
            raw_cols[col] = f"rawvar_{col}"
        spec["raw"] = {"var_names": list(raw.get("var_names") or var_names), "var": raw_cols}
    tmp = Path(tempfile.mkdtemp(prefix="sc-build-"))
    try:
        (tmp / "spec.json").write_text(json.dumps(spec))
        np.savez(tmp / "arr.npz", **arrays)
        run_builder(tools, _BUILDER, [str(tmp), str(work / name)], work, f"building {name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- F101 ----------------------------------------------------------------------------------------

N_REPORT = 3
LFC_TOL = 0.25


def scanpy_lfc(matrix: np.ndarray, in_group: np.ndarray, genes: list[int]) -> dict[int, float]:
    """scanpy's rank_genes_groups logfoldchanges: log2 of the ratio of expm1(mean) over the
    group and over the rest (the input is taken to be log1p data, whatever it holds)."""
    m = matrix.astype(np.float64)
    g1 = m[in_group][:, genes].mean(axis=0)
    g2 = m[~in_group][:, genes].mean(axis=0)
    values = np.log2((np.expm1(g1) + 1e-9) / (np.expm1(g2) + 1e-9))
    return {gene: float(v) for gene, v in zip(genes, values, strict=True)}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = make_rng(101, seed)
    labels = [CELL_TYPES[i] for i in rng.permutation(len(CELL_TYPES))[:3]]
    s = synth(
        rng,
        int(rng.integers(120, 181)),
        int(rng.integers(150, 221)),
        labels,
        sizes=[float(x) for x in rng.uniform(0.8, 1.4, size=3)],
    )
    target = labels[0]
    x = log1p_cp10k(s.counts)
    raw_x = x if twin == "clean" else cp10k(s.counts)
    write_h5ad(
        work,
        "cells.h5ad",
        x,
        s.cells,
        s.genes,
        obs={"celltype": s.labels},
        raw={"X": raw_x},
        uns={"log1p": {"base": None}},
        tools=tools,
    )
    stated = tier == "explicit"
    obj = "cells.h5ad: scanpy AnnData (cells x genes), cell types in obs['celltype']."
    if stated:
        obj += " X is log1p of counts per 10,000."
        obj += (
            " `.raw` is a copy of X taken after the log1p."
            if twin == "clean"
            else " `.raw` holds the same cells before the log transform (counts per 10,000)."
        )
    (work / "README.md").write_text(common.readme([obj, f"Python: {PY_HINT}."]))
    seen = f"raw.X is not log-transformed (max {raw_x.max():.0f}) while X is (max {x.max():.1f})"
    if twin == "clean":
        evidence = ("README says .raw is a copy of the log1p X",) if stated else ()
    else:
        evidence = ("README says .raw holds counts per 10,000", seen) if stated else (seen,)
    in_group = np.array([lab == target for lab in s.labels])
    # the markers with the lowest counts per 10,000: scanpy's expm1 of a group mean overflows
    # (to inf) above about 700, and an inf is no answer
    level = {
        g: float(cp10k(s.counts)[in_group][:, g].astype(np.float64).mean())
        for g in s.markers[target]
    }
    picked = sorted(sorted(s.markers[target], key=level.get)[:N_REPORT])
    assert max(level[g] for g in picked) < 650, "marker genes too deep for the raw matrix"
    names = [s.genes[i] for i in picked]
    expected = {s.genes[i]: v for i, v in scanpy_lfc(x, in_group, picked).items()}
    naive_values = {s.genes[i]: v for i, v in scanpy_lfc(raw_x, in_group, picked).items()}
    free = (
        f"in cells.h5ad, run scanpy's Wilcoxon rank_genes_groups for cell type '{target}' "
        f"(obs['celltype']) against all other cells and report the log fold change scanpy "
        f"gives for the genes {', '.join(names)} ({PY_HINT})"
    )
    forced = (
        f"in cells.h5ad, call `sc.tl.rank_genes_groups` with its default arguments except "
        f"groupby='celltype', groups=['{target}'], reference='rest' and method='wilcoxon', and "
        f"report the log fold change it gives for the genes {', '.join(names)} ({PY_HINT})"
    )
    prompt = common.prompt_for(
        route, free, forced, "a JSON object mapping each gene to its log fold change"
    )
    truth = {
        "kind": "dict_num",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_values,
        "tol": LFC_TOL,
    }
    task_id = common.task_id("F101", seed, twin, tier, route)
    return Task(task_id, "F101", "matrix_state", twin, seed, prompt, truth, evidence, tier, route)


def _task_inputs(task: Task) -> tuple[str, list[str]]:
    group = re.search(r"cell type '([^']+)'|groups=\['([^']+)'\]", task.prompt)
    assert group is not None
    genes = re.search(r"genes ([A-Z0-9, ]+?) \(", task.prompt)
    assert genes is not None
    return group.group(1) or group.group(2), [g.strip() for g in genes.group(1).split(",")]


_SCRIPT = """
import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
sc.tl.rank_genes_groups(
    a, "celltype", groups=[{group!r}], reference="rest", method="wilcoxon"{extra}
)
df = sc.get.rank_genes_groups_df(a, {group!r}).set_index("names")
print(json.dumps({{g: float(df.loc[g, "logfoldchanges"]) for g in {genes!r}}}))
"""


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, genes = _task_inputs(task)
    code = _SCRIPT.format(group=group, genes=genes, extra=", use_raw=False")
    flags = ["adata.raw differs from X: rank_genes_groups run with use_raw=False"]
    raw_differs = task.twin == "faulted"
    return result(
        solve_with(tools, code, work),
        "rank_genes_groups on X (use_raw=False)",
        flags if raw_differs else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    group, genes = _task_inputs(task)
    code = _SCRIPT.format(group=group, genes=genes, extra="")
    return result(solve_with(tools, code, work), "rank_genes_groups with default arguments")


RANK = re.compile(r"\brank_genes_groups\s*\(")
CHOSEN = re.compile(r"\buse_raw\s*=\s*(?:False|0)\b|\blayer\s*=")  # the call picks the matrix
DROPPED = re.compile(r"\b(\w+)\.raw\s*=\s*None\b|\bdel\s+(\w+)\.raw\b")
FIRST_ARG = re.compile(r"\s*(?:adata\s*=\s*)?(\w+)\s*[,)]")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff some call of rank_genes_groups does not choose the matrix.

    A call that passes use_raw=False (or a layer) is set aside, and so is a call on an object whose
    .raw was dropped earlier in the same command. Anything else is the trapped route, also when a
    safe variant follows in that command (the default call, then `b.raw = None` and a second call
    on b): a model that ran the default call has executed the trapped route.
    """
    for command in commands:
        dropped: set[str] = set()
        for statement in statements(command):
            for found in DROPPED.finditer(statement):
                dropped.update(name for name in found.groups() if name)
            if CHOSEN.search(statement):
                continue
            for call in RANK.finditer(statement):
                first = FIRST_ARG.match(statement, call.end())
                if first is None or first.group(1) not in dropped:
                    return True
    return False


SPEC = FamilySpec(
    id="F101",
    slot="matrix_state",
    source="SCD-22",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"cells.h5ad", "README.md"}),
    locus="tool_default",
)
