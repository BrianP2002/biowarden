"""Admission cases for the matrix-state rules of the sc pack (implicit tier, both twins).

  sc.de_raw_default        F101  rank_genes_groups with a .raw that holds CP10K (faulted) or a
                                 copy of the log1p X (clean)
  sc.small_int_qc          F102  calculate_qc_metrics on a sparse int16 (faulted) / float32 X
  sc.double_transform      F103  normalize_total + log1p on a log-normalised (faulted) / counts X
  sc.counts_needed         F104  UMI-count steps on a log-normalised X with layers['counts']
  sc.scaled_x_expression   F112  expression read from a scaled X with the values in .raw

Per seed, both twins of the family are built and every template (python-sc code an agent could
write: the default call, the explicit fix, a hand-rolled numpy version, a script file, a heredoc,
an inspection) is executed with the real libraries; the label is whether the printed answer equals
`task.truth` (cases_sc_common). Some templates are wrong for another reason than the rule's trap
(a median target_sum, `expm1` on counts, an uns check that the file does not satisfy): they stay
in the cases as the rule's honest misses.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_sc_common import Template, make_cases
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import (
    f101_de_use_raw,
    f102_int16_qc,
    f103_double_norm,
    f104_counts_layer,
    f112_scaled_x,
)

# -- F101 ----------------------------------------------------------------------------------------

_DE = """import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{pre}sc.tl.rank_genes_groups(
    a, "celltype", groups=[{group!r}], reference="rest", method="{method}"{args}
)
df = sc.get.rank_genes_groups_df(a, {group!r}).set_index("names")
print(json.dumps({{g: float(df.loc[g, "logfoldchanges"]) for g in {genes!r}}}))
"""
_DE_MANUAL = """import json, numpy as np, anndata as ad, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
x = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
m = (a.obs["celltype"] == {group!r}).values
cols = [list(a.var_names).index(g) for g in {genes!r}]
g1, g2 = x[m][:, cols].mean(0), x[~m][:, cols].mean(0)
lfc = np.log2((np.expm1(g1) + 1e-9) / (np.expm1(g2) + 1e-9))
print(json.dumps(dict(zip({genes!r}, map(float, lfc)))))
"""


def f101_templates(task: Task, twin: str, work: Path) -> list[Template]:
    group, genes = f101_de_use_raw._task_inputs(task)
    fill = {"group": group, "genes": genes, "method": "wilcoxon", "pre": "", "args": ""}
    X, RAW = {"de_input": "X"}, {"de_input": "raw"}

    def code(**kw: str) -> str:
        return _DE.format(**{**fill, **kw})

    default = code()
    return [
        Template("de_default", default, wrong_decl=(X, RAW)),
        Template("de_default_script", default, form="script", wrong_decl=(X,)),
        Template("de_default_heredoc", default, form="heredoc", wrong_decl=(X,)),
        Template("de_use_raw_false", code(args=", use_raw=False"), decl_true=X, evident=True),
        Template(
            "de_use_raw_false_script",
            code(args=", use_raw=False"),
            form="script",
            decl_true=X,
            evident=True,
        ),
        Template(
            "de_use_raw_true", code(args=", use_raw=True"), decl_true=RAW, wrong_decl=(RAW, X)
        ),
        Template("de_raw_dropped", code(pre="a.raw = None\n"), decl_true=X, evident=True),
        Template(
            "de_raw_to_adata",
            code(pre="a = a.raw.to_adata()\n"),
            decl_true=RAW,
            evident=True,
            wrong_decl=(RAW,),
        ),
        Template(
            "de_manual_numpy",
            _DE_MANUAL.format(group=group, genes=genes),
            decl_true=X,
            evident=True,
        ),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print(a.X.max(), a.raw.X.max(), a.obs['celltype'].value_counts())\n",
            inspection=True,
            decl_true=X,
        ),
    ]


@case_source("sc.de_raw_default")
def cases_f101(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases("F101", seed, scratch, tools, f101_templates, lambda twin: {"de_input": "X"})


# -- F102 ----------------------------------------------------------------------------------------

_QC = """import json, numpy as np, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{pre}sc.pp.calculate_qc_metrics(a, percent_top={top}, inplace=True{args})
print(json.dumps(int((a.obs["pct_counts_in_top_20_genes"] > {t}).sum())))
"""
_QC_MANUAL = """import json, numpy as np, anndata as ad, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
x = (a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype("float64")
top = -np.sort(-x, axis=1)[:, :20].sum(1) / x.sum(1) * 100
print(json.dumps(int((top > {t}) .sum())))
"""


def f102_templates(task: Task, twin: str, work: Path) -> list[Template]:
    t = f102_int16_qc._threshold(task)
    true = {"x_dtype": "float32" if twin == "clean" else "int16"}
    fill = {"t": t, "top": "[20]", "pre": "", "args": ""}

    def code(**kw: str) -> str:
        return _QC.format(**{**fill, **kw})

    default = code()
    return [
        Template("qc_default", default, wrong_decl=(true,)),
        Template("qc_default_script", default, form="script", wrong_decl=(true,)),
        Template(
            "qc_tuple_log1p_false", code(top="(20,)", args=", log1p=False"), wrong_decl=(true,)
        ),
        Template("qc_heredoc", default, form="heredoc", wrong_decl=(true,)),
        Template("qc_cast_float32", code(pre="a.X = a.X.astype('float32')\n"), evident=True),
        Template("qc_cast_float64", code(pre="a.X = a.X.astype(np.float64)\n"), evident=True),
        Template("qc_dense_first", code(pre="a.X = a.X.toarray()\n"), evident=True),
        Template("qc_manual_numpy", _QC_MANUAL.format(t=t), evident=True),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print(a.X.dtype, a.X.sum(1).max())\n",
            inspection=True,
        ),
    ]


@case_source("sc.small_int_qc")
def cases_f102(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"x_dtype": "float32" if twin == "clean" else "int16"}

    return make_cases("F102", seed, scratch, tools, f102_templates, true)


# -- F103 ----------------------------------------------------------------------------------------

_NORM = """import json, numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
{pre}{norm}{log}sub = a[(a.obs["celltype"] == {group!r}).values, {gene!r}].X
print(json.dumps(float(np.asarray(sub.todense() if sp.issparse(sub) else sub).mean())))
"""


def f103_templates(task: Task, twin: str, work: Path) -> list[Template]:
    gene, group = f103_double_norm._inputs(task)
    state = {"x_state": "counts" if twin == "clean" else "log_normalized"}
    counts, logged = {"x_state": "counts"}, {"x_state": "log_normalized"}
    fill = {"group": group, "gene": gene, "pre": "", "norm": "", "log": ""}
    n1e4 = "sc.pp.normalize_total(a, target_sum=1e4)\n"
    log = "sc.pp.log1p(a)\n"

    def code(**kw: str) -> str:
        return _NORM.format(**{**fill, **kw})

    pipeline = code(norm=n1e4, log=log)
    guarded = code(
        pre=(
            "v = a.X.data if sp.issparse(a.X) else np.asarray(a.X)\n"
            "if np.allclose(v, np.round(v)):\n"
        ),
        norm="    " + n1e4,
        log="    " + log,
    )
    return [
        Template("pipeline_1e4", pipeline, wrong_decl=(logged, counts)),
        Template("pipeline_1e4_script", pipeline, form="script", wrong_decl=(logged,)),
        Template("pipeline_1e4_heredoc", pipeline, form="heredoc", wrong_decl=(logged,)),
        Template("pipeline_median_target", code(norm="sc.pp.normalize_total(a)\n", log=log)),
        Template("log1p_only_cp10k", code(log=log), wrong_decl=(logged,)),
        Template(
            "uns_check",
            code(
                pre="if 'log1p' not in a.uns:\n",
                norm="    " + n1e4,
                log="    " + log,
            ),
            wrong_decl=(logged,),
        ),
        Template("as_is", code(), decl_true=state, evident=True),
        Template("guarded_integer_test", guarded, decl_true=state, evident=True),
        Template(
            "expm1_first",
            code(pre="a.X = np.expm1(a.X)\n", norm=n1e4, log=log),
            decl_true=state,
            evident=True,
        ),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print(a.X.max(), 'log1p' in a.uns)\n",
            inspection=True,
            decl_true=state,
        ),
    ]


@case_source("sc.double_transform")
def cases_f103(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"x_state": "counts" if twin == "clean" else "log_normalized"}

    return make_cases("F103", seed, scratch, tools, f103_templates, true)


# -- F104 ----------------------------------------------------------------------------------------

_FILTER = """import json, numpy as np, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{pre}{body}"""


def f104_templates(task: Task, twin: str, work: Path) -> list[Template]:
    t = f104_counts_layer._threshold(task)
    state = {"x_state": "counts" if twin == "clean" else "log_normalized"}
    counts, logged = {"x_state": "counts"}, {"x_state": "log_normalized"}
    layer = "if 'counts' in a.layers:\n    a.X = a.layers['counts'].copy()\n"
    filt = (
        f"keep, _ = sc.pp.filter_cells(a, min_counts={t}, inplace=False)\n"
        "print(json.dumps(int(keep.sum())))\n"
    )
    qc = (
        "sc.pp.calculate_qc_metrics(a, percent_top=None, inplace=True)\n"
        f'print(json.dumps(int((a.obs["total_counts"] >= {t}).sum())))\n'
    )
    xsum = (
        f"tot = np.asarray(a.X.sum(axis=1)).ravel()\nprint(json.dumps(int((tot >= {t}).sum())))\n"
    )
    qc_layer = (
        "lay = 'counts' if 'counts' in a.layers else None\n"
        "sc.pp.calculate_qc_metrics(a, layer=lay, percent_top=None, inplace=True)\n"
        f'print(json.dumps(int((a.obs["total_counts"] >= {t}).sum())))\n'
    )
    manual = (
        "src = a.layers['counts'] if 'counts' in a.layers else a.X\n"
        f"tot = np.asarray(src.sum(axis=1)).ravel()\nprint(json.dumps(int((tot >= {t}).sum())))\n"
    )

    def code(body: str, pre: str = "") -> str:
        return _FILTER.format(pre=pre, body=body)

    return [
        Template("filter_cells_default", code(filt), wrong_decl=(logged, counts)),
        Template("filter_cells_script", code(filt), form="script", wrong_decl=(logged,)),
        Template("filter_cells_heredoc", code(filt), form="heredoc", wrong_decl=(logged,)),
        Template("qc_total_counts", code(qc), wrong_decl=(logged,)),
        Template("x_sum_axis1", code(xsum), wrong_decl=(logged,)),
        Template("filter_cells_counts_layer", code(filt, layer), evident=True, decl_true=state),
        Template("qc_layer", code(qc_layer), evident=True, decl_true=state),
        Template("manual_layer_sum", code(manual), evident=True, decl_true=state),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\nprint(a, list(a.layers))\n',
            inspection=True,
            decl_true=state,
        ),
    ]


@case_source("sc.counts_needed")
def cases_f104(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"x_state": "counts" if twin == "clean" else "log_normalized"}

    return make_cases("F104", seed, scratch, tools, f104_templates, true)


# -- F112 ----------------------------------------------------------------------------------------

_EXPR = """import json, numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
{body}
x = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
print(json.dumps(int((x[(a.obs["celltype"] == {group!r}).values] > 0).sum())))
"""


def f112_templates(task: Task, twin: str, work: Path) -> list[Template]:
    group, gene = f112_scaled_x._inputs(task)
    state = {"x_state": "log_normalized" if twin == "clean" else "scaled"}
    scaled, logged = {"x_state": "scaled"}, {"x_state": "log_normalized"}

    def code(body: str) -> str:
        return _EXPR.format(body=body, group=group)

    on_x = code(f"x = a[:, {gene!r}].X")
    return [
        Template("x_gt0", on_x, wrong_decl=(scaled, logged)),
        Template("x_gt0_script", on_x, form="script", wrong_decl=(scaled,)),
        Template("x_gt0_heredoc", on_x, form="heredoc", wrong_decl=(scaled,)),
        Template("x_via_to_df", code(f"x = a.to_df()[{gene!r}].values"), wrong_decl=(scaled,)),
        Template(
            "x_via_obs_df",
            code(f"x = sc.get.obs_df(a, keys=[{gene!r}])[{gene!r}].values"),
            wrong_decl=(scaled,),
        ),
        Template("raw_gt0", code(f"x = a.raw[:, {gene!r}].X"), evident=True, decl_true=state),
        Template(
            "obs_df_use_raw",
            code(f"x = sc.get.obs_df(a, keys=[{gene!r}], use_raw=True)[{gene!r}].values"),
            evident=True,
            decl_true=state,
        ),
        Template(
            "raw_to_adata",
            code(f"x = a.raw.to_adata()[:, {gene!r}].X"),
            evident=True,
            decl_true=state,
        ),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print(a.X.min(), a.raw.X.min() if a.raw is not None else None)\n",
            inspection=True,
            decl_true=state,
        ),
    ]


@case_source("sc.scaled_x_expression")
def cases_f112(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"x_state": "log_normalized" if twin == "clean" else "scaled"}

    return make_cases("F112", seed, scratch, tools, f112_templates, true)
