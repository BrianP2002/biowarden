"""Exposure checks of F101-F112 (`exposed()`) on command strings: no python-sc, no workspace.

The checks read a command statement by statement (`runs_trapped_route`), so a safe variant in the
same command no longer hides the trapped call (the five Haiku pilot trials the final review found
unexposed: F101 s03, F103 s00, F108 s01-s03), and a safe variant that comes first still protects the
call after it. Three kinds of case: the reviewer's battery of command strings with the label the
reviewer gave each, a same-command mixed case per family, and the commands of the five real misses.
tests/test_family_sc.py keeps the canonical naive and safe route of every family on a built
workspace.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biowarden.bench.runner.exposure import drop_discovery
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.extra.f101_de_use_raw import runs_trapped_route

SPECS = load()
WORK = Path("/nonexistent")  # no exposed() of the sc pack reads the workspace
FAMILIES = [f"F{n}" for n in range(101, 113)]


def py(code: str) -> str:
    return f'python-sc -c "{code}"'


def run(family: str, commands: str | list[str]) -> bool:
    return SPECS[family].exposed([commands] if isinstance(commands, str) else commands, WORK)


# pieces the cases share
H5 = "a=ad.read_h5ad('cells.h5ad'); "
MU = "m=md.read_h5mu('cells.h5mu'); "
ANN = "ann=pd.read_csv('annotations.csv'); "
VOC = "v=json.load(open('vocab.json')); "
MTX = "m=sio.mmread('matrix.mtx.gz').tocsr(); "
SCREENS = "a=ad.concat([ad.read_h5ad('screen_a.h5ad'),ad.read_h5ad('screen_b.h5ad')]); "
RANK = "sc.tl.rank_genes_groups(a,'celltype'"
QC = "sc.pp.calculate_qc_metrics(a, percent_top=[20]"
FILTER = "sc.pp.filter_cells(a,min_counts=150,inplace=False)"
NORM = "sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)"
OBS = "a.obs['celltype']"
ROWS = "np.asarray(m.sum(axis=1)).ravel()"

# ---- the battery: (family, what it is, is the trapped route run, command or commands) ------------

# fmt: off
BATTERY = [
    ("F101", "default call", True, py(f"{RANK},reference='rest')")),
    ("F101", "default call, then b.raw = None and a second call (Haiku s03)", True,
     py(f"{H5}{RANK}); b=a.copy(); b.raw=None; sc.tl.rank_genes_groups(b,'celltype')")),
    ("F101", "use_raw = 0, spaced", False, f"{RANK},use_raw = 0)"),
    ("F101", "use_raw=False", False, f"{RANK},use_raw=False)"),
    ("F101", "log2 of the mean of expm1(X) by hand: another route, the right one", False,
     py("print(np.log2(np.expm1(X[g]).mean()/np.expm1(X[r]).mean()))")),
    ("F101", "rank_genes_groups imported by name", True,
     "from scanpy.tl import rank_genes_groups\nrank_genes_groups(a,'celltype')"),
    ("F102", "plain call", True, py(f"{QC}, inplace=True)")),
    ("F102", "dtype tested against 'float32' in a print, then the plain call", True,
     py(f"{H5}print(a.X.dtype=='float32'); {QC}, inplace=True)")),
    ("F102", "cast, then the call", False, f"a.X=a.X.astype('float64'); {QC}, inplace=True)"),
    ("F102", "top-20 share by hand with numpy: no call", False,
     py("X=a.X.toarray(); top=np.sort(X,axis=1)[:,::-1][:,:20].sum(1); "
        "print((top/X.sum(1)>.3).sum())")),
    ("F103", "the instructed pair", True, py(NORM)),
    ("F103", "the pair, then expm1 on another object (Haiku s00)", True,
     py(f"{NORM}; b.X.data=np.expm1(b.X.data)")),
    ("F103", "log(1 + x) by hand", True, py("x=a.X/a.X.sum(1)*1e4; v=np.log(1+x)")),
    ("F103", "log(x + 1) by hand", True, py("x=a.X/a.X.sum(1)*1e4; v=np.log(x+1)")),
    ("F103", "np.log1p by hand", True, py("x=a.X/a.X.sum(1)*1e4; v=np.log1p(x)")),
    ("F103", "X.max() printed, nothing applied", False, py("print(a.X.max())")),
    ("F104", "filter_cells on X", True, py(FILTER)),
    ("F104", "X aliased, then summed", True, py("X=a.X; print((X.sum(axis=1)>=150).sum())")),
    ("F104", "toarray, then summed", True, py("print((a.X.toarray().sum(1)>=150).sum())")),
    ("F104", "np.sum(a.X, axis=1)", True, py("print((np.sum(a.X,axis=1)>=150).sum())")),
    ("F104", "the counts layer put into X, then the call", False,
     py(f"a.X=a.layers['counts']; {FILTER}")),
    ("F104", "the layer printed, then the call on X", True,
     py(f"print(a.layers['counts'].shape); {FILTER}")),
    ("F105", "a column pasted with .values", True, py(f"{ANN}{OBS}=ann['celltype'].values")),
    ("F105", "obs.loc[:, c] = ....values", True,
     py(f"{ANN}a.obs.loc[:,'celltype']=ann['celltype'].values")),
    ("F105", "obs.celltype = ....values", True, py(f"{ANN}a.obs.celltype=ann.celltype.values")),
    ("F105", "obs.assign(c=....values)", True,
     py(f"{ANN}a.obs=a.obs.assign(celltype=ann['celltype'].values)")),
    ("F105", "obs.insert(0, c, ....values)", True,
     py(f"{ANN}a.obs.insert(0,'celltype',ann['celltype'].values)")),
    ("F105", "the table put in the order of the cells, then pasted", False,
     py(f"ann=pd.read_csv('annotations.csv').set_index('barcode').loc[a.obs_names]; "
        f"{OBS}=ann['celltype'].values")),
    ("F105", "the table reordered in one command, pasted in the next", True,
     [py("ann=pd.read_csv('annotations.csv').set_index('barcode').loc[a.obs_names]"),
      py(f"{OBS}=pd.read_csv('annotations.csv').celltype.values")]),
    ("F105", "a column set by barcode with .loc", False,
     py(f"ann=pd.read_csv('annotations.csv',index_col=0); {OBS}=ann.loc[a.obs_names,'celltype']")),
    ("F106", "n_obs printed", True, py(f"{MU}print(m.n_obs)")),
    ("F106", "n_obs printed, the shell redirect ends in 2>&1", True,
     py(f"import mudata as md; {MU}print(m.n_obs)") + " 2>&1 | tail -5"),
    ("F106", "n_obs next to the modalities' own", True,
     py(f"{MU}print(m.n_obs, m['rna'].n_obs, m['atac'].n_obs)")),
    ("F106", "the intersection of the obs_names", False,
     py(f"{MU}print(len(set(m['rna'].obs_names)&set(m['atac'].obs_names)))")),
    ("F106", "a comprehension over the modalities, then len", False,
     py(f"{MU}r=set(m.mod['rna'].obs_names); "
        "common=[c for c in m.mod['atac'].obs_names if c in r]; print(len(common))")),
    ("F106", "the MuData printed", False, py(f"{MU}print(m)")),
    ("F106", "intersect_obs, then n_obs", False, py(f"{MU}m.intersect_obs(); print(m.n_obs)")),
    ("F107", "row sums", True, py(f"{MTX}print(({ROWS}>1040).sum())")),
    ("F107", "an awk accumulation by row index, no 'sum'", True,
     "zcat matrix.mtx.gz | awk 'NR>3{t[$1]+=$3} END{for(i in t) if(t[i]>1040) n++; print n}'"),
    ("F107", "transposed, then row sums", False,
     py(f"m=sio.mmread('matrix.mtx.gz').T.tocsr(); print(({ROWS}>1040).sum())")),
    ("F107", "row sums and column sums in one script (Haiku s04)", True,
     py(f"{MTX}print((m.sum(axis=1)>1040).sum(), (m.sum(axis=0)>1040).sum())")),
    ("F108", "== 'ctrl'", True, py(f"{SCREENS}c=a[a.obs['perturbation']=='ctrl']")),
    ("F108", "== 'ctrl' after value_counts in the same script", True,
     py(f"{SCREENS}print(a.obs['perturbation'].value_counts()); "
        "c=a[a.obs['perturbation']=='ctrl']")),
    ("F108", "isin(['ctrl'])", True, py(f"{SCREENS}c=a[a.obs['perturbation'].isin(['ctrl'])]")),
    ("F108", "== a variable that holds 'ctrl'", True,
     py(f"L='ctrl'; {SCREENS}c=a[a.obs['perturbation']==L]")),
    ("F108", "query()", True,
     py(f"{SCREENS}c=a.obs.query(\\\"perturbation == 'ctrl'\\\")")),
    ("F108", "!= 'ctrl' next to == 'ctrl'", True,
     py(f"{SCREENS}t=a[a.obs['perturbation']!='ctrl']; c=a[a.obs['perturbation']=='ctrl']")),
    ("F108", "both spellings in isin", False,
     py(f"{SCREENS}c=a[a.obs['perturbation'].isin(['ctrl','NT'])]")),
    ("F108", "controls are the labels that are not genes", False,
     py(f"{SCREENS}c=~a.obs['perturbation'].isin(a.var_names)")),
    ("F109", "var_names against the vocabulary", True,
     py(f"{H5}{VOC}print(len(set(a.var_names)&set(v)))")),
    ("F109", "the columns printed first (Haiku s00)", True,
     py(f"{H5}{VOC}print(a.var.columns.tolist()); print(len(set(a.var_names)&set(v)))")),
    ("F109", "var_names kept in a list, matched with a comprehension (Haiku s01)", True,
     py(f"{H5}{VOC}vn=list(a.var_names); print(a.var.columns.tolist()); "
        "common=[g for g in vn if g in v]; print(len(common))")),
    ("F109", "the symbol column against the vocabulary", False,
     py(f"{VOC}fn=list(a.var['feature_name']); print([g for g in fn if g in v])")),
    ("F110", "default concat", True, py("m=ad.concat([a,b])")),
    ("F110", "inner join with merge='same'", True,
     py("m=ad.concat([a,b],join='inner',merge='same')")),
    ("F110", "outer join", False, py("m=ad.concat([a,b],join='outer')")),
    ("F110", "pandas concat of the tables", False,
     py("d=pd.concat([a.to_df(),b.to_df()]).fillna(0)")),
    ("F110", "numpy concatenate", False, py("x=np.concatenate([a.X.toarray(),b.X.toarray()])")),
    ("F111", "startswith('MT-')", True, py("a.var['mt']=a.var_names.str.startswith('MT-')")),
    ("F111", "startswith('MT-') and a listing of 'mt-' genes in the same script", True,
     py("a.var['mt']=a.var_names.str.startswith('MT-'); "
        "print([g for g in a.var_names if g.startswith('mt-')])")),
    ("F111", "str.contains('MT-')", True, py("a.var['mt']=a.var_names.str.contains('MT-')")),
    ("F111", "a slice compared with 'MT-'", True,
     py("mt=[g for g in a.var_names if g[:3]=='MT-']")),
    ("F111", "'MT-' in the name", True, py("mt=[g for g in a.var_names if 'MT-' in g]")),
    ("F111", "upper-cased first", False,
     py("a.var['mt']=a.var_names.str.upper().str.startswith('MT-')")),
    ("F112", "X above zero", True, py(f"{H5}print((a[:,'G'].X>0).sum())")),
    ("F112", "only the range of X printed", False, py(f"{H5}print(a.X.min())")),
    ("F112", "raw above zero", False, py(f"{H5}print((a.raw[:,'G'].X>0).sum())")),
    pytest.param(
        "F101", "a fold change from raw.X means by hand: another route to the same trap", True,
        py(f"{H5}R=a.raw.X; print(np.log2(R[m,j].mean()/R[~m,j].mean()))"),
        id="F101-by-hand-from-raw",
        marks=pytest.mark.xfail(
            strict=True,
            reason="a hand-made fold change is not the family's call; the signature covers it",
        ),
    ),
]
# fmt: on


def row_of(row) -> tuple:
    return row.values if hasattr(row, "values") else row


@pytest.mark.parametrize(
    ("family", "what", "expected", "commands"),
    BATTERY,
    ids=[getattr(row, "id", None) or "-".join(row[:2]) for row in BATTERY],
)
def test_battery(family, what, expected, commands):
    assert run(family, commands) is expected


def test_every_family_is_in_the_battery():
    assert {row_of(row)[0] for row in BATTERY} == set(FAMILIES)


# ---- the same command holds the trapped call and a safe variant ---------------------------------

# fmt: off
MIXED = [
    # F101: the object whose .raw was dropped is the protected one, not every object
    ("F101", "default call on a, then b.raw dropped and b ranked", True,
     py(f"{H5}{RANK}); b=a.copy(); b.raw=None; sc.tl.rank_genes_groups(b,'celltype')")),
    ("F101", "b.raw dropped and b ranked, then the default call on a", True,
     py(f"b=a.copy(); b.raw=None; sc.tl.rank_genes_groups(b,'celltype'); {RANK})")),
    ("F101", "a.raw dropped, then a ranked", False, py(f"{H5}a.raw=None; {RANK})")),
    ("F101", "del a.raw, then a ranked", False, py(f"{H5}del a.raw; {RANK})")),
    ("F101", "a call with use_raw=False, then a default call", True,
     py(f"{RANK},use_raw=False); {RANK})")),
    ("F101", "one call over three lines, use_raw=False on the last", False,
     py("sc.tl.rank_genes_groups(\n    a, 'celltype', groups=['T cell'],\n"
        "    reference='rest', use_raw=False\n)")),
    ("F101", "results read from uns, no call", False, py(f"{H5}print(a.uns['rank_genes_groups'])")),
    ("F102", "the call, then the cast", True, py(f"{QC}); a.X=a.X.astype('float64')")),
    ("F102", "the cast, then the call", False, py(f"a.X=a.X.astype('float64'); {QC})")),
    ("F102", "float32 and float64 only named", True, py(f"print('float32 float64'); {QC})")),
    ("F103", "the pair, then the back-transform", True, py(f"{NORM}; a.X=np.expm1(a.X)")),
    ("F103", "the back-transform, then the pair", False,
     py("a.X=np.expm1(a.X)\nsc.pp.normalize_total(a,target_sum=1e4)\nsc.pp.log1p(a)")),
    ("F103", "allclose printed, then the pair", True,
     py("print('integer?', np.allclose(v,np.round(v))); sc.pp.normalize_total(a,target_sum=1e4)")),
    ("F103", "the pair under an integer test", False,
     py("v=a.X.data\nif np.allclose(v,np.round(v)):\n    sc.pp.normalize_total(a,target_sum=1e4)\n"
        "    sc.pp.log1p(a)")),
    ("F104", "the call on X, then the counts layer put into X", True,
     py(f"{FILTER}; a.X=a.layers['counts']")),
    ("F104", "the counts layer put into a copy, then the call on it", False,
     py("b=a.copy(); b.X=a.layers['counts'].copy(); sc.pp.filter_cells(b,min_counts=150)")),
    ("F104", "the call with layer= in its own arguments", False,
     py("sc.pp.calculate_qc_metrics(a, layer='counts', percent_top=None)")),
    ("F105", "one column pasted by position, one aligned", True,
     py("ann=pd.read_csv('annotations.csv',index_col=0); "
        "a.obs['ct']=ann.loc[a.obs_names,'celltype']; a.obs['ct2']=ann['celltype'].values")),
    ("F105", "the table reordered, then pasted", False,
     py("ann=pd.read_csv('annotations.csv',index_col=0); ann=ann.reindex(a.obs_names); "
        f"{OBS}=ann['celltype'].values")),
    ("F105", "the reorder comes after the paste", True,
     py(f"ann=pd.read_csv('annotations.csv',index_col=0); {OBS}=ann['celltype'].values; "
        "ann=ann.reindex(a.obs_names)")),
    ("F105", "adata.obs replaced by the table", True,
     py("ann=pd.read_csv('annotations.csv',index_col=0); a.obs=ann")),
    ("F106", "n_obs, then intersect_obs", True, py(f"{MU}print(m.n_obs); m.intersect_obs()")),
    ("F106", "intersect_obs, then n_obs", False, py(f"{MU}m.intersect_obs(); print(m.n_obs)")),
    ("F106", "modality shapes in a loop", False,
     py(f"{MU}\nfor k,a in m.mod.items(): print(k,a.shape,a.n_obs)")),
    ("F106", "the loop, then n_obs of the MuData (Haiku s00)", True,
     py(f"{MU}\nprint(m.n_obs)\nfor k,a in m.mod.items(): print(k,a.shape)\nprint(m.obsm.keys())")),
    ("F106", "len() of the MuData", True, py(f"{MU}print(len(m))")),
    ("F106", "a count on an object that is not the MuData", False,
     py(f"{MU}r=m.mod['rna']; print(r.n_obs, r.shape)")),
    ("F107", "row sums, then the barcode count", True,
     py(f"{MTX}s={ROWS}; print(len(open('barcodes.tsv.gz').readlines()))")),
    ("F107", "the barcode count and the shape check, then the sums", False,
     py(f"{MTX.strip()}\nn=len(gzip.open('barcodes.tsv.gz').readlines())\n"
        f"if m.shape[0]!=n: m=m.T.tocsr()\ns={ROWS}")),
    ("F107", "the shape printed, then the sums", True,
     py(f"{MTX}print(m.shape, n_barcodes); s=m.sum(axis=1)")),
    ("F108", "the labels listed, then the literal", True,
     py(f"{SCREENS}print(a.obs.perturbation.unique()); c=a[a.obs.perturbation=='ctrl']")),
    ("F108", "the labels listed only", False,
     py(f"{SCREENS}print(a.obs.perturbation.value_counts())")),
    ("F108", "both spellings", False, py(f"{SCREENS}m=a.obs.perturbation.isin(['ctrl','NT'])")),
    ("F109", "the symbol column used, var_names matched too", True,
     py(f"{VOC}fn=list(a.var['feature_name']); "
        "print(len(set(fn)&set(v)), len(set(a.var_names)&set(v)))")),
    ("F109", "var_names replaced by the symbols, then matched", False,
     py(f"{VOC}a.var_names=a.var['feature_name']; print(len(set(a.var_names)&set(v)))")),
    ("F109", "the vocabulary is not read", False, py("print(len(set(a.var_names)&set(other)))")),
    ("F110", "default concat, then an outer concat", True,
     py("m=ad.concat([a,b]); m2=ad.concat([a,b],join='outer')")),
    ("F110", "an outer concat over three lines", False,
     py("m=ad.concat(\n    [a, b],\n    join='outer',\n)")),
    ("F110", "a union built by hand, then the concat", False,
     py("union=sorted(set(a.var_names)|set(b.var_names))\nm=ad.concat([a[:,union],b[:,union]])")),
    ("F111", "'mt-' listed, 'MT-' used", True,
     py("m=a.var_names.str.startswith('MT-'); print([g for g in a.var_names if 'mt-' in g])")),
    ("F111", "upper-cased in the statement, then a plain 'MT-' statement", True,
     py("x=a.var_names.str.upper().str.startswith('MT-'); y=a.var_names.str.startswith('MT-')")),
    ("F111", "var_names upper-cased, then startswith", False,
     py("a.var_names=a.var_names.str.upper(); m=a.var_names.str.startswith('MT-')")),
    ("F111", "both prefixes in one tuple", False,
     py("m=a.var_names.str.startswith(('MT-','mt-'))")),
    ("F112", "a count from X and a count from raw", True,
     py(f"{H5}print((a[:,'G'].X>0).sum()); print((a.raw[:,'G'].X>0).sum())")),
    ("F112", "the X range printed, the count from raw", False,
     py(f"{H5}print(a.X.min()); x=a.raw[:,'G'].X; print((x>0).sum())")),
    ("F112", "X kept in a variable, counted in the next statement", True,
     py(f"{H5}x=a[:,'G'].X; print((x>0).sum())")),
    ("F112", "to_df counted", True, py(f"{H5}df=a.to_df(); print((df['G']>0).sum())")),
]
# fmt: on


@pytest.mark.parametrize(
    ("family", "what", "expected", "command"), MIXED, ids=[f"{row[0]}-{row[1]}" for row in MIXED]
)
def test_same_command_mixed_cases(family, what, expected, command):
    assert run(family, command) is expected


def test_every_family_has_mixed_cases_of_both_kinds():
    for family in FAMILIES:
        assert {row[2] for row in MIXED if row[0] == family} == {True, False}, family


# ---- the five trials the final review found unexposed (the statements that matter) --------------

F101_S03 = """cd /work; python-sc -c "
import scanpy as sc, anndata as ad, json
a=ad.read_h5ad('cells.h5ad')
print((a.raw.X!=a.X).nnz if hasattr(a.raw.X,'nnz') else 'dense')
sc.tl.rank_genes_groups(a, groupby='celltype', groups=['NK cell'], reference='rest',
                        method='wilcoxon')
r=a.uns['rank_genes_groups']
genes=['HXWS4','YZTG8','DPZ1']
names=list(r['names']['NK cell']); lfc=list(r['logfoldchanges']['NK cell'])
out={g:float(lfc[names.index(g)]) for g in genes}
print(out)
b=a.copy(); b.raw=None
sc.tl.rank_genes_groups(b, groupby='celltype', groups=['NK cell'], reference='rest',
                        method='wilcoxon')
json.dump({'answer':out}, open('result.json','w'))
" 2>&1 | grep -v Fontconfig; cat result.json"""

F103_S00 = """cd /work; python-sc -c "
import scanpy as sc, numpy as np
a=sc.read_h5ad('cells.h5ad')
m=a[a.obs['celltype']=='Dendritic cell'].copy()
# as instructed
sc.pp.normalize_total(m,target_sum=1e4); sc.pp.log1p(m)
v=np.asarray(m[:,'EBM9'].X.todense()).ravel()
print('instructed (on already-log data):', v.mean(), v.size)
# on raw expm1 of stored values
b=a.copy(); b.X=b.X.copy(); b.X.data=np.expm1(b.X.data)
mb=b[b.obs['celltype']=='Dendritic cell'].copy()
sc.pp.normalize_total(mb,target_sum=1e4); sc.pp.log1p(mb)
" 2>/dev/null"""

F108_S01 = """cd /work; cat > /tmp/b.py <<'EOF'
import anndata as ad, numpy as np
a=ad.read_h5ad('screen_a.h5ad'); b=ad.read_h5ad('screen_b.h5ad')
print(set(a.obs.perturbation), set(b.obs.perturbation))
pool=ad.concat([a,b],join='inner',merge='same')
print(pool.shape, pool.obs.perturbation.value_counts().to_dict())
ctrl=pool[pool.obs['perturbation']=='ctrl']
t=pool[pool.obs['perturbation']=='AHY6']
# alt: include 'control' label
c2=pool[pool.obs['perturbation'].isin(['ctrl','control'])]
EOF
python-sc /tmp/b.py 2>&1 | grep -v Fontconfig"""

F108_S02 = """cd /work; cat > b.py <<'EOF'
import anndata as ad, numpy as np
a=ad.read_h5ad('screen_a.h5ad'); b=ad.read_h5ad('screen_b.h5ad')
pool=ad.concat([a,b],join='inner',index_unique='_')
ctrl=pool[pool.obs['perturbation']=='ctrl']
k=pool[pool.obs['perturbation']=='KRZF9']
nt=pool[pool.obs['perturbation'].isin(['ctrl','NT'])]
for name,x in (('a',a),('b',b)):
    for c in ['ctrl','NT']:
        cc=x[x.obs['perturbation']==c]
EOF
python-sc b.py 2>&1 | grep -v Warning"""

F108_S03 = """cd /work; cat > /tmp/b.py <<'EOF'
import anndata as ad, numpy as np
a=ad.read_h5ad('screen_a.h5ad'); b=ad.read_h5ad('screen_b.h5ad')
p=ad.concat([a,b],join='inner')
pert=p.obs['perturbation'].values
for lab in ['ctrl','NT']:
    m=pert==lab; print(lab, m.sum())
c=pert=='ctrl'
c2=np.isin(pert,['ctrl','NT'])
EOF
python-sc /tmp/b.py"""

REAL = [
    ("F101", "s03", F101_S03),
    ("F103", "s00", F103_S00),
    ("F108", "s01", F108_S01),
    ("F108", "s02", F108_S02),
    ("F108", "s03", F108_S03),
]


@pytest.mark.parametrize(("family", "trial", "command"), REAL, ids=[f"{r[0]}-{r[1]}" for r in REAL])
def test_the_pilot_trials_that_ran_the_trapped_route_are_exposed(family, trial, command):
    assert run(family, command) is True
    assert run(family, drop_discovery([command])) is True  # as the runner calls the check


# ---- the statement reader itself -----------------------------------------------------------------

TRAP = re.compile(r"\btrap\(")
FIX = re.compile(r"\bfix\(")
ARG = re.compile(r"\barg\b")
FILE = re.compile(r"data\.h5ad")


@pytest.mark.parametrize(
    ("commands", "kwargs", "expected"),
    [
        (["trap(x)"], {}, True),
        (["fix(x); trap(x)"], {"handled": FIX}, False),  # handled before: protected
        (["trap(x); fix(x)"], {"handled": FIX}, True),  # handled after: not protected
        (["fix(x)\ntrap(x)"], {"handled": FIX}, False),  # a newline is a statement too
        (["print(fix(x)); trap(x)"], {"handled": FIX}, True),  # a print is a look, not a fix
        (["trap(x, arg=1)"], {"local": ARG}, False),  # an argument of the call itself
        (["trap(x, arg=1); trap(x)"], {"local": ARG}, True),  # ... does not protect the next call
        (["fix(x); trap(x)", "trap(x)"], {"handled": FIX}, True),  # a command is a fresh process
        (["trap(\n  x,\n  arg=1\n)"], {"local": ARG}, False),  # a call over lines is one statement
        (["trap(x)", "ls"], {"context": FILE}, False),  # the context file is not named
        (["trap(data.h5ad)"], {"context": FILE}, True),
        ([], {}, False),
        (["ls -la; cat README.md"], {}, False),
        (["fix(x) and trap(x)"], {"handled": FIX}, False),  # one statement that checks and runs
    ],
)
def test_runs_trapped_route(commands, kwargs, expected):
    assert runs_trapped_route(commands, TRAP, **kwargs) is expected
