"""Admission cases for the annotation / join rules of the sc pack (implicit tier, both twins).

  sc.obs_assign_position   F105  an annotation table in barcode order (clean) or sorted by cell
                                 type (faulted), copied into adata.obs
  sc.control_label         F108  controls spelled 'ctrl' in both screens (clean) or 'ctrl' and
                                 another spelling (faulted)
  sc.gene_vocabulary       F109  var_names as symbols (clean) or Ensembl ids (faulted) against a
                                 symbol vocabulary
  sc.concat_join           F110  two batches with the same genes (clean) or different genes
  sc.mito_prefix_case      F111  human MT- (clean) or mouse mt- (faulted) gene names

Labels come from executing each template with the real libraries and comparing the printed answer
with `task.truth` (cases_sc_common). Templates that are wrong for another reason than the rule's
trap (a pasted `.values` that happens to be right, `join='inner'` written on purpose, a lower-case
prefix on human data) stay as the rule's honest misses.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_sc_common import Template, make_cases
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import (
    f105_obs_alignment,
    f108_control_label,
    f111_mito_case,
)

# -- F105 ----------------------------------------------------------------------------------------

_ANN = """import json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
a = ad.read_h5ad("cells.h5ad")
ann = pd.read_csv("annotations.csv", index_col=0)
{assign}
x = a[:, {gene!r}].X
x = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
print(json.dumps(int(((a.obs["celltype"] == {group!r}).values & (x >= {depth})).sum())))
"""


def f105_templates(task: Task, twin: str, work: Path) -> list[Template]:
    group, gene, depth = f105_obs_alignment._inputs(task)
    pos, idx = {"obs_join": "by_position"}, {"obs_join": "by_index"}

    def code(assign: str) -> str:
        return _ANN.format(assign=assign, gene=gene, group=group, depth=depth)

    values = code("a.obs['celltype'] = ann['celltype'].values")
    return [
        Template("paste_values", values, wrong_decl=(pos, idx)),
        Template("paste_values_script", values, form="script", wrong_decl=(pos,)),
        Template("paste_values_heredoc", values, form="heredoc", wrong_decl=(idx,)),
        Template(
            "paste_to_numpy",
            code("a.obs['celltype'] = ann['celltype'].to_numpy()"),
            wrong_decl=(pos,),
        ),
        Template(
            "paste_list", code("a.obs['celltype'] = list(ann['celltype'])"), wrong_decl=(idx,)
        ),
        Template("replace_obs", code("a.obs = ann"), wrong_decl=(pos, idx)),
        Template("aligned_series", code("a.obs['celltype'] = ann['celltype']"), decl_true=idx),
        Template(
            "aligned_loc",
            code("a.obs['celltype'] = ann.loc[a.obs_names, 'celltype'].values"),
            decl_true=idx,
            evident=True,
        ),
        Template(
            "aligned_reindex",
            code("a.obs['celltype'] = ann.reindex(a.obs_names)['celltype'].values"),
            decl_true=idx,
            evident=True,
        ),
        Template("aligned_join", code("a.obs = a.obs.join(ann)"), decl_true=idx, evident=True),
        Template(
            "aligned_map",
            code("a.obs['celltype'] = a.obs_names.map(dict(zip(ann.index, ann['celltype'])))"),
            decl_true=idx,
            evident=True,
        ),
        Template(
            "inspect",
            'import anndata as ad, pandas as pd\na = ad.read_h5ad("cells.h5ad")\n'
            'ann = pd.read_csv("annotations.csv", index_col=0)\n'
            "print(list(a.obs_names[:3]), list(ann.index[:3]))\n",
            inspection=True,
            decl_true=idx,
        ),
    ]


@case_source("sc.obs_assign_position")
def cases_f105(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:  # the clean table is in cell order: position is honest
        return {"obs_join": "by_position" if twin == "clean" else "by_index"}

    return make_cases("F105", seed, scratch, tools, f105_templates, true)


# -- F108 ----------------------------------------------------------------------------------------

_CTRL = """import json, numpy as np, anndata as ad, scipy.sparse as sp
a = ad.concat([ad.read_h5ad("screen_a.h5ad"), ad.read_h5ad("screen_b.h5ad")])
{select}
col = a[:, {gene!r}].X
col = np.asarray(col.todense() if sp.issparse(col) else col).ravel()
pert = (a.obs["perturbation"] == {target!r}).values
print(json.dumps(float(col[pert].mean() - col[ctrl].mean())))
"""
ALL_SPELLINGS = "['ctrl', 'control', 'non-targeting', 'NT', 'scramble']"


def f108_templates(task: Task, twin: str, work: Path) -> list[Template]:
    gene, target = f108_control_label._inputs(task)
    ctrl, multi = {"control_label": "ctrl"}, {"control_label": "multiple"}

    def code(select: str) -> str:
        return _CTRL.format(select=select, gene=gene, target=target)

    literal = code("ctrl = (a.obs['perturbation'] == 'ctrl').values")
    return [
        Template("eq_ctrl", literal, wrong_decl=(ctrl, multi)),
        Template("eq_ctrl_script", literal, form="script", wrong_decl=(ctrl,)),
        Template("eq_ctrl_heredoc", literal, form="heredoc", wrong_decl=(multi,)),
        Template(
            "eq_ctrl_after_value_counts",
            code(
                "print(a.obs['perturbation'].value_counts())\n"
                "ctrl = (a.obs['perturbation'] == 'ctrl').values"
            ),
            wrong_decl=(ctrl,),
        ),
        Template(
            "eq_control_literal",
            code("ctrl = (a.obs['perturbation'] == 'control').values"),
            wrong_decl=(multi,),
        ),
        Template(
            "isin_all_spellings",
            code(f"ctrl = a.obs['perturbation'].isin({ALL_SPELLINGS}).values"),
            decl_true=multi,
            evident=True,
        ),
        Template(
            "not_a_gene",
            code("ctrl = (~a.obs['perturbation'].isin(a.var_names)).values"),
            decl_true=multi,
            evident=True,
        ),
        Template(
            "regex_case_insensitive",
            code(
                "ctrl = a.obs['perturbation'].str.match("
                "'^(ctrl|control|non-targeting|nt|scramble)$', case=False).values"
            ),
            decl_true=multi,
            evident=True,
        ),
        Template(
            "inspect",
            'import anndata as ad\nfor f in ("screen_a.h5ad", "screen_b.h5ad"):\n'
            '    print(ad.read_h5ad(f).obs["perturbation"].value_counts())\n',
            inspection=True,
            decl_true=multi,
        ),
    ]


@case_source("sc.control_label")
def cases_f108(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"control_label": "ctrl" if twin == "clean" else "multiple"}

    return make_cases("F108", seed, scratch, tools, f108_templates, true)


# -- F109 ----------------------------------------------------------------------------------------

_VOCAB = """import json, anndata as ad
a = ad.read_h5ad("cells.h5ad")
vocab = set(json.load(open("vocab.json")))
{names}
print(json.dumps(sorted(set(names) & vocab)))
"""


def f109_templates(task: Task, twin: str, work: Path) -> list[Template]:
    sym, ens = {"gene_namespace": "symbol"}, {"gene_namespace": "ensembl_id"}

    def code(names: str) -> str:
        return _VOCAB.format(names=names)

    plain = code("names = list(a.var_names)")
    return [
        Template("var_names_set", plain, wrong_decl=(sym, ens)),
        Template("var_names_script", plain, form="script", wrong_decl=(sym,)),
        Template("var_names_heredoc", plain, form="heredoc", wrong_decl=(sym,)),
        Template("var_index_list", code("names = [g for g in a.var.index]"), wrong_decl=(sym,)),
        Template(
            "var_names_isin",
            code("names = list(a.var_names[a.var_names.isin(list(vocab))])"),
            wrong_decl=(sym,),
        ),
        Template(
            "symbol_column_if_present",
            code(
                "col = [c for c in a.var.columns if c in ('feature_name', 'gene_symbol')]\n"
                "names = list(a.var[col[0]].astype(str)) if col else list(a.var_names)"
            ),
            decl_true=sym,
            evident=True,
        ),
        Template(
            "best_column_search",
            code(
                "cands = {None: list(a.var_names)}\n"
                "cands.update({c: a.var[c].astype(str).tolist() for c in a.var.columns})\n"
                "best = max(cands, key=lambda c: len(vocab & set(cands[c])))\n"
                "names = cands[best]"
            ),
            decl_true=sym,
            evident=True,
        ),
        Template(
            "inspect",
            'import json, anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print(list(a.var_names[:3]), a.var.columns.tolist())\n"
            "print(list(json.load(open('vocab.json')))[:5])\n",
            inspection=True,
            decl_true=sym,
        ),
    ]


@case_source("sc.gene_vocabulary")
def cases_f109(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases(
        "F109", seed, scratch, tools, f109_templates, lambda twin: {"gene_namespace": "symbol"}
    )


# -- F110 ----------------------------------------------------------------------------------------

_CONCAT = """import json, numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
a, b = ad.read_h5ad("batch1.h5ad"), ad.read_h5ad("batch2.h5ad")
{merge}
tot = np.asarray(m.X.sum(axis=0)).ravel()
print(json.dumps(int((tot > 0).sum())))
"""
_MANUAL_UNION = """import json, numpy as np, anndata as ad
a, b = ad.read_h5ad("batch1.h5ad"), ad.read_h5ad("batch2.h5ad")
seen = set()
for x in (a, b):
    tot = np.asarray(x.X.sum(axis=0)).ravel()
    seen |= {g for g, t in zip(x.var_names, tot) if t > 0}
print(json.dumps(len(seen)))
"""


def f110_templates(task: Task, twin: str, work: Path) -> list[Template]:
    union, inter = {"concat_genes": "union"}, {"concat_genes": "intersection"}

    def code(merge: str) -> str:
        return _CONCAT.format(merge=merge)

    default = code("m = ad.concat([a, b])")
    return [
        Template("concat_default", default, wrong_decl=(union, inter)),
        Template("concat_default_script", default, form="script", wrong_decl=(union,)),
        Template("concat_default_heredoc", default, form="heredoc", wrong_decl=(union,)),
        Template(
            "concat_label_keys",
            code("m = ad.concat([a, b], label='batch', keys=['1', '2'])"),
            wrong_decl=(union,),
        ),
        Template(
            "concat_inner_explicit",
            code("m = ad.concat([a, b], join='inner')"),
            wrong_decl=(union, inter),
        ),
        Template("scanpy_concat", code("m = sc.concat([a, b])"), wrong_decl=(union,)),
        Template("concatenate_method", code("m = a.concatenate(b)"), wrong_decl=(union,)),
        Template(
            "concat_outer",
            code("m = ad.concat([a, b], join='outer')"),
            decl_true=union,
            evident=True,
        ),
        Template(
            "concat_outer_fill",
            code("m = ad.concat([a, b], join='outer', fill_value=0)"),
            decl_true=union,
            evident=True,
        ),
        Template("manual_union", _MANUAL_UNION, decl_true=union, evident=True),
        Template(
            "inspect",
            "import anndata as ad\n"
            'a, b = ad.read_h5ad("batch1.h5ad"), ad.read_h5ad("batch2.h5ad")\n'
            "print(a.n_vars, b.n_vars, len(set(a.var_names) & set(b.var_names)))\n",
            inspection=True,
            decl_true=union,
        ),
    ]


@case_source("sc.concat_join")
def cases_f110(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases(
        "F110", seed, scratch, tools, f110_templates, lambda twin: {"concat_genes": "union"}
    )


# -- F111 ----------------------------------------------------------------------------------------

_MITO = """import json, anndata as ad, scanpy as sc
a = ad.read_h5ad("cells.h5ad")
{mark}
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
print(json.dumps(int((a.obs["pct_counts_mt"] < {t}).sum())))
"""


def f111_templates(task: Task, twin: str, work: Path) -> list[Template]:
    t = f111_mito_case._threshold(task)
    upper, lower, both = {"mito_prefix": "MT-"}, {"mito_prefix": "mt-"}, {"mito_prefix": "both"}

    def code(mark: str) -> str:
        return _MITO.format(mark=mark, t=t)

    plain = code("a.var['mt'] = a.var_names.str.startswith('MT-')")
    return [
        Template("startswith_MT", plain, wrong_decl=(upper, lower)),
        Template("startswith_MT_script", plain, form="script", wrong_decl=(upper,)),
        Template("startswith_MT_heredoc", plain, form="heredoc", wrong_decl=(lower,)),
        Template(
            "str_match_caret",
            code("a.var['mt'] = a.var_names.str.match('^MT-')"),
            wrong_decl=(upper,),
        ),
        Template(
            "str_contains_caret",
            code("a.var['mt'] = a.var_names.str.contains('^MT-')"),
            wrong_decl=(upper,),
        ),
        Template(
            "upper_then_startswith",
            code("a.var['mt'] = a.var_names.str.upper().str.startswith('MT-')"),
            decl_true=both,
            evident=True,
        ),
        Template(
            "lower_then_startswith",
            code("a.var['mt'] = a.var_names.str.lower().str.startswith('mt-')"),
            decl_true=both,
            evident=True,
        ),
        Template(
            "contains_ignore_case",
            code("a.var['mt'] = a.var_names.str.contains('^mt-', case=False)"),
            decl_true=both,
            evident=True,
        ),
        Template(
            "both_prefixes_tuple",
            code("a.var['mt'] = a.var_names.str.startswith(('MT-', 'mt-'))"),
            decl_true=both,
            evident=True,
        ),
        Template("lower_prefix_only", code("a.var['mt'] = a.var_names.str.startswith('mt-')")),
        Template(
            "inspect",
            'import anndata as ad\na = ad.read_h5ad("cells.h5ad")\n'
            "print([g for g in a.var_names if g.lower().startswith('mt')])\n",
            inspection=True,
            decl_true=both,
        ),
    ]


@case_source("sc.mito_prefix_case")
def cases_f111(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def true(twin: str) -> dict[str, str]:
        return {"mito_prefix": "MT-" if twin == "clean" else "mt-"}

    return make_cases("F111", seed, scratch, tools, f111_templates, true)
