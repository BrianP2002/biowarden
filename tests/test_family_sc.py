"""Families F101-F112 (sc pack): registration, trap size over seeds 0-29, twins, README tiers,
reference and naive solvers, exposure heuristics.

Builds and solver snippets run in ONE long-lived python-sc process (the same anndata / scanpy,
without a process start per call); the byte-for-byte reproducibility across processes and the
reference-passes / naive-fails gates are in tests/test_extra_families.py, the real
`python-sc -c` runs (exit code, stderr) are in tests/test_family_sc_tools.py, and the exposure
checks on mixed same-command strings and the reviewer's battery are in
tests/test_family_sc_exposure.py.
"""

from __future__ import annotations

import filecmp
import gzip
import re
from functools import partial
from pathlib import Path

import pytest

from biowarden.bench.admission.cases_sc_common import _build_run, worker_for
from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f101_de_use_raw, load
from biowarden.bench.tasks.extra.f101_de_use_raw import BUILD_HOOKS, PY_HINT, RUN_HOOKS
from biowarden.bench.tasks.locus import LOCI, locus_of
from biowarden.bench.tasks.reference import solve

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"F{n}" for n in range(101, 113)]
SPECS = load()
SEEDS = range(30)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-sc" not in env.real_bins(["python-sc"]):
        pytest.skip("python-sc is not available")
    return env


@pytest.fixture(scope="module")
def fast(tools):
    worker = worker_for(tools)
    BUILD_HOOKS.append(partial(_build_run, worker))
    RUN_HOOKS.append(lambda code, cwd: worker.run(code, cwd))
    yield worker
    BUILD_HOOKS.pop()
    RUN_HOOKS.pop()


@pytest.fixture(scope="module")
def made(fast, tools, tmp_path_factory):
    """Memoised builds: made(family, seed, twin, tier, route) -> (work, task)."""
    root = tmp_path_factory.mktemp("sc-fam")
    cache: dict[tuple, tuple[Path, object]] = {}

    def get(family, seed, twin, tier="explicit", route="free"):
        key = (family, seed, twin, tier, route)
        if key not in cache:
            work = root / "-".join(map(str, key))
            cache[key] = (work, build_task(family, seed, twin, work, tools, tier, route))
        return cache[key]

    return get


def test_all_twelve_families_are_registered_with_a_valid_locus():
    assert [i for i in IDS if i not in SPECS] == []
    for family in IDS:
        spec = SPECS[family]
        assert spec.locus in LOCI and locus_of(family) == spec.locus
        assert spec.slot and spec.source and spec.twin_files
        assert {"tool_default", "data"} >= {spec.locus} or spec.locus in LOCI
    assert {SPECS[i].locus for i in IDS} == {"data", "tool_default"}


# ---- the trap changes the answer on every seed -------------------------------------------------

KINDS = {
    "F101": "dict_num",
    "F102": "int",
    "F103": "float",
    "F104": "int",
    "F105": "int",
    "F106": "int",
    "F107": "int",
    "F108": "float",
    "F109": "id_list",
    "F110": "int",
    "F111": "int",
    "F112": "int",
}


def gap(truth) -> float:
    expected, naive = truth["expected"], truth["naive"]
    kind = truth["kind"]
    if kind == "int":
        return abs(naive - expected)
    if kind == "float":
        return abs(naive - expected)
    if kind == "dict_num":
        return min(abs(naive[k] - expected[k]) for k in expected)
    return float(len(set(expected) ^ set(naive)))


@pytest.mark.parametrize("family", IDS)
def test_trap_changes_the_answer_on_every_seed_and_the_clean_twin_is_unaffected(made, family):
    for seed in SEEDS:
        _, faulted = made(family, seed, "faulted")
        truth = faulted.truth
        assert truth["kind"] == KINDS[family]
        wrong = {**truth, "expected": truth["naive"]}
        assert not answer_matches(truth, truth["naive"]), (family, seed)
        assert answer_matches(wrong, truth["naive"])
        tol = truth.get("tol", 0.0)
        assert gap(truth) > max(4 * tol, 0.9 if truth["kind"] in ("int", "id_list") else 0.0), (
            family,
            seed,
            gap(truth),
        )
        expected = truth["expected"]
        assert expected != 0 and expected != [] and expected != {}, (family, seed)
    for seed in range(5):
        _, clean = made(family, seed, "clean")
        _, faulted = made(family, seed, "faulted")
        assert clean.truth["naive"] == clean.truth["expected"]
        assert clean.truth["expected"] == faulted.truth["expected"]


@pytest.mark.parametrize("family", IDS)
def test_truth_is_the_same_on_both_tiers_and_routes_and_data_is_tiny(made, family):
    work_e, explicit = made(family, 3, "faulted", "explicit")
    work_i, implicit = made(family, 3, "faulted", "implicit")
    _, forced = made(family, 3, "faulted", "explicit", "forced")
    assert explicit.truth == implicit.truth == forced.truth
    assert explicit.prompt == implicit.prompt and forced.prompt != explicit.prompt
    for work in (work_e, work_i):
        assert sum(p.stat().st_size for p in work.iterdir()) < 400_000
        assert not any(p.is_dir() for p in work.iterdir())


# ---- the README states the convention only in the explicit tier --------------------------------

STATED = {  # (twin, substring that must appear in the explicit README and not in the implicit)
    "F101": ("faulted", "before the log transform"),
    "F102": ("faulted", "int16"),
    "F103": ("faulted", "log1p of counts per 10,000"),
    "F104": ("faulted", "layers['counts']"),
    "F105": ("faulted", "sorted by cell type"),
    "F106": ("faulted", "only an RNA profile"),
    "F107": ("faulted", "genes x cells"),
    "F108": ("faulted", "labelled 'ctrl' in screen_a.h5ad and"),
    "F109": ("faulted", "Ensembl gene ids"),
    "F110": ("faulted", "were dropped"),
    "F111": ("faulted", "mouse"),
    "F112": ("faulted", "scaled"),
}


@pytest.mark.parametrize("family", IDS)
def test_readme_states_the_convention_only_in_the_explicit_tier(made, family):
    twin, text = STATED[family]
    explicit = (made(family, 4, twin, "explicit")[0] / "README.md").read_text()
    implicit = (made(family, 4, twin, "implicit")[0] / "README.md").read_text()
    assert text in explicit and text not in implicit
    assert "python-sc" in explicit and "python-sc" in implicit
    assert f"Python: {PY_HINT}." in explicit and f"Python: {PY_HINT}." in implicit
    clean_e = (made(family, 4, "clean", "explicit")[0] / "README.md").read_text()
    clean_i = (made(family, 4, "clean", "implicit")[0] / "README.md").read_text()
    faulted_i = implicit
    assert clean_i == faulted_i  # the implicit README does not tell the twins apart
    assert clean_e != explicit


FORCED = {
    "F101": "default arguments",
    "F102": "inplace=True",
    "F103": "normalize_total",
    "F104": "filter_cells",
    "F105": "adata.obs['celltype']",
    "F106": "read_h5mu",
    "F107": "scipy.io.mmread",
    "F108": "== 'ctrl'",
    "F109": "adata.var_names",
    "F110": "anndata.concat",
    "F111": "startswith('MT-')",
    "F112": ".X` is above",
}


@pytest.mark.parametrize("family", IDS)
def test_forced_route_names_the_tool_whose_default_is_the_trap(made, family):
    _, free = made(family, 2, "faulted", "explicit", "free")
    _, forced = made(family, 2, "faulted", "explicit", "forced")
    assert FORCED[family] in forced.prompt and FORCED[family] not in free.prompt
    assert f"({PY_HINT})" in free.prompt and f"({PY_HINT})" in forced.prompt


@pytest.mark.parametrize("family", IDS)
def test_twins_differ_only_in_the_declared_files(made, family):
    work_c, _ = made(family, 2, "clean", "implicit")
    work_f, _ = made(family, 2, "faulted", "implicit")
    names = {p.name for p in work_c.iterdir()}
    assert names == {p.name for p in work_f.iterdir()}
    same, diff, _ = filecmp.cmpfiles(work_c, work_f, sorted(names), shallow=False)
    assert diff and set(diff) <= SPECS[family].twin_files
    assert "README.md" in same  # implicit tier: the sentence about the data is shared
    assert not (set(diff) & {"README.md"})


# ---- solvers ------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", IDS)
def test_reference_solves_both_twins_and_naive_falls_into_the_recorded_trap(made, tools, family):
    spec = SPECS[family]
    for seed in (0, 1):
        for twin in ("clean", "faulted"):
            work, task = made(family, seed, twin, "implicit")
            ref = solve(task, work, tools, reference=True)
            assert score_result(task.truth, ref).outcome == "pass", (family, seed, twin)
        work, task = made(family, seed, "faulted", "implicit")
        naive = solve(task, work, tools, reference=False)
        score = score_result(task.truth, naive)
        assert score.outcome == "silent_wrong" and score.signature is True, (family, seed)
        work, task = made(family, seed, "clean", "implicit")
        assert score_result(task.truth, solve(task, work, tools, reference=False)).outcome == "pass"
        assert spec.reference is not spec.naive


# ---- exposure ----------------------------------------------------------------------------------

PY = 'python-sc -c "{}"'
ANN = "ann = pd.read_csv('annotations.csv', index_col=0); "
CSV = "df = pd.read_csv('counts.csv', index_col=0)"
H5 = "a = ad.read_h5ad('cells.h5ad'); "
VOC = "v = json.load(open('vocab.json')); "
EXPOSURE = {
    "F101": (
        [PY.format("sc.tl.rank_genes_groups(a, 'celltype', groups=['T cell'], method='wilcoxon')")],
        [
            PY.format("sc.tl.rank_genes_groups(a, 'celltype', use_raw=False)"),
            PY.format("sc.tl.rank_genes_groups(a, 'celltype', layer='x')"),
            PY.format("print(a.raw.X.max(), a.X.max())"),
        ],
    ),
    "F102": (
        [PY.format("sc.pp.calculate_qc_metrics(a, percent_top=[20], inplace=True)")],
        [
            PY.format(
                "a.X = a.X.astype('float32'); sc.pp.calculate_qc_metrics(a, percent_top=[20])"
            ),
            PY.format("print(a.X.dtype)"),
        ],
    ),
    "F103": (
        [
            PY.format("sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)"),
            PY.format("sc.pp.log1p(a)"),
        ],
        [
            PY.format("a.X = np.expm1(a.X); sc.pp.log1p(a)"),
            PY.format("v = a.X.data\nif np.allclose(v, np.round(v)): sc.pp.log1p(a)"),
            PY.format("print(a.X.max())"),
        ],
    ),
    "F104": (
        [
            PY.format("sc.pp.filter_cells(a, min_counts=700, inplace=False)"),
            PY.format("print((a.X.sum(axis=1) >= 700).sum())"),
        ],
        [
            PY.format("a.X = a.layers['counts']; sc.pp.filter_cells(a, min_counts=700)"),
            PY.format("sc.pp.calculate_qc_metrics(a, layer='counts', percent_top=None)"),
            PY.format("print(a)"),
        ],
    ),
    "F105": (
        [
            PY.format(ANN + "a.obs['celltype'] = ann['celltype'].values"),
            PY.format(ANN + "a.obs = ann"),
            PY.format(
                "ann = pd.read_csv('annotations.csv'); a.obs['celltype'] = list(ann.celltype)"
            ),
        ],
        [
            PY.format(ANN + "a.obs['celltype'] = ann.loc[a.obs_names, 'celltype'].values"),
            PY.format(ANN + "a.obs = a.obs.join(ann)"),
            PY.format(ANN + "a.obs['celltype'] = ann['celltype']"),
            "head annotations.csv",
        ],
    ),
    "F106": (
        [
            PY.format("m = md.read_h5mu('cells.h5mu'); print(m.n_obs)"),
            PY.format("m = md.read_h5mu('cells.h5mu'); print(len(m.obs_names))"),
            PY.format("print(md.read_h5mu('cells.h5mu').n_obs)"),
        ],
        [
            PY.format(
                "m = md.read_h5mu('cells.h5mu'); "
                "print(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names)))"
            ),
            PY.format(
                "m = md.read_h5mu('cells.h5mu'); print((m.obsm['rna'] & m.obsm['atac']).sum())"
            ),
            PY.format("m = md.read_h5mu('cells.h5mu'); print(m.mod['rna'].n_obs)"),
            "ls -la cells.h5mu",
        ],
    ),
    "F107": (
        [
            PY.format(
                "m = scipy.io.mmread('matrix.mtx.gz').tocsr(); print((m.sum(axis=1) > 900).sum())"
            )
        ],
        [
            PY.format("m = scipy.io.mmread('matrix.mtx.gz'); print(m.shape)"),
            PY.format("m = scipy.io.mmread('matrix.mtx.gz').T; print((m.sum(axis=1) > 900).sum())"),
            PY.format(
                "n = len(open('barcodes.tsv.gz').readlines()); "
                "m = scipy.io.mmread('matrix.mtx.gz'); print(m.sum(1))"
            ),
            "zcat matrix.mtx.gz | head",
        ],
    ),
    "F108": (
        [
            PY.format("a = ad.read_h5ad('screen_a.h5ad'); c = a[a.obs['perturbation'] == 'ctrl']"),
            PY.format("m = (ad.read_h5ad('screen_b.h5ad').obs.perturbation == 'control')"),
            PY.format(
                "a = ad.read_h5ad('screen_b.h5ad'); print(a.obs.perturbation.unique()); "
                "m = a.obs.perturbation == 'ctrl'"
            ),
        ],
        [
            PY.format("print(ad.read_h5ad('screen_b.h5ad').obs['perturbation'].value_counts())"),
            PY.format("m = a.obs['perturbation'].isin(['ctrl', 'non-targeting']); screen_a"),
            PY.format(
                "print(a.obs.perturbation.unique()); m = ~a.obs.perturbation.isin(a.var_names)"
            ),
        ],
    ),
    "F109": (
        [
            PY.format(H5 + "v = set(json.load(open('vocab.json'))); print(set(a.var_names) & v)"),
            PY.format(H5 + VOC + "print(a.var.index.isin(list(v)).sum())"),
        ],
        [
            PY.format(H5 + VOC + "print(a.var['feature_name'].isin(list(v)).sum())"),
            PY.format("a = ad.read_h5ad('cells.h5ad'); print(a.var.columns, a.var_names[:3])"),
        ],
    ),
    "F110": (
        [PY.format("m = ad.concat([ad.read_h5ad('batch1.h5ad'), ad.read_h5ad('batch2.h5ad')])")],
        [
            PY.format("m = ad.concat([a, b], join='outer')  # batch1.h5ad batch2.h5ad"),
            PY.format("print(a.n_vars, b.n_vars, len(set(a.var_names) & set(b.var_names)))"),
        ],
    ),
    "F111": (
        [
            PY.format("a.var['mt'] = a.var_names.str.startswith('MT-')"),
            PY.format("a.var['mt'] = a.var_names.str.match('^MT-')"),
        ],
        [
            PY.format("a.var['mt'] = a.var_names.str.upper().str.startswith('MT-')"),
            PY.format("a.var['mt'] = a.var_names.str.startswith(('MT-', 'mt-'))"),
            PY.format("print(a.var_names[:5])"),
        ],
    ),
    "F112": (
        [
            PY.format("a = ad.read_h5ad('cells.h5ad'); x = a[:, 'GENE1'].X; print((x > 0).sum())"),
            PY.format(
                "a = ad.read_h5ad('cells.h5ad'); df = a.to_df(); print((df['GENE1'] > 0).sum())"
            ),
        ],
        [
            PY.format(
                "a = ad.read_h5ad('cells.h5ad'); x = a.raw[:, 'GENE1'].X; print((x > 0).sum())"
            ),
            PY.format("a = ad.read_h5ad('cells.h5ad'); print(a.raw.X.max())"),
        ],
    ),
}


@pytest.mark.parametrize("family", IDS)
def test_exposure_sees_the_naive_route_and_not_the_safe_one(made, family):
    work, _ = made(family, 1, "faulted")
    spec = SPECS[family]
    naive_routes, safe_routes = EXPOSURE[family]
    for command in naive_routes:
        assert spec.exposed([command], work) is True, command
    for command in safe_routes:
        assert spec.exposed([command], work) is False, command
    assert spec.exposed(["ls -la; cat README.md"], work) is False
    assert spec.exposed([], work) is False
    # exposure is "ran it", not "kept the answer": the naive command then the fix still counts
    assert spec.exposed([naive_routes[0], safe_routes[0]], work) is True
    assert spec.exposed([safe_routes[0], "ls"], work) is False


# ---- the changes of the final review (FE-1, FE-5, FE-8, FE-9) ------------------------------------


def test_the_python_hint_names_an_executable_and_the_f101_title_names_no_helpers():
    assert "python-sc script.py" in PY_HINT and "not a directory" in PY_HINT
    assert "scanpy 1.11" in PY_HINT and "anndata 0.11" in PY_HINT
    title = f101_de_use_raw.__doc__.splitlines()[0]
    assert title.startswith("F101 de_use_raw:") and "helpers" not in title
    # the one-line error that tools/make_families_doc.py copies into docs/families.md
    assert title.endswith("reads `.raw` by default.")


def test_f107_forced_sentence_says_what_to_count_and_not_how_to_sum(made):
    _, forced = made("F107", 2, "faulted", "explicit", "forced")
    assert "`scipy.io.mmread`" in forced.prompt
    assert "count the cells (barcodes) whose total UMIs are above" in forced.prompt
    assert "sum over the rows" not in forced.prompt and "entries" not in forced.prompt
    _, free = made("F107", 2, "faulted", "explicit", "free")
    assert "more than" in free.prompt  # the free sentence is unchanged


def test_f107_headers_agree_with_the_layout_of_each_twin(made):
    """Cell Ranger writes genes x cells and says so; the cells x genes matrix is a plain export."""
    for seed in (0, 3):
        for twin, cell_ranger in (("faulted", True), ("clean", False)):
            work, _ = made("F107", seed, twin, "implicit")
            with gzip.open(work / "matrix.mtx.gz", "rt") as handle:
                head = [handle.readline() for _ in range(3)]
            with gzip.open(work / "barcodes.tsv.gz", "rt") as handle:
                barcodes = len(handle.read().split())
            assert head[0].startswith("%%MatrixMarket matrix coordinate integer general")
            assert ("cellranger" in head[1]) is cell_ranger, (seed, twin, head)
            rows, cols, _ = map(int, head[2].split())
            assert (cols if twin == "faulted" else rows) == barcodes


def test_f110_defines_the_quantity_over_the_data_and_keeps_the_default_concat(made):
    for route in ("free", "forced"):
        _, task = made("F110", 2, "faulted", "implicit", route)
        assert "non-zero total count summed over all the cells of both batches" in task.prompt
        assert "in the merged object" not in task.prompt
    _, forced = made("F110", 2, "faulted", "implicit", "forced")
    assert "`anndata.concat([a, b])`" in forced.prompt


@pytest.mark.parametrize("route", ["free", "forced"])
def test_f111_threshold_prints_with_one_decimal_on_every_seed(made, route):
    for seed in SEEDS:
        _, task = made("F111", seed, "faulted", "explicit", route)
        printed = re.search(r"(?:less than|below) (\S+?)(?: percent|\s|\))", task.prompt).group(1)
        assert re.fullmatch(r"\d+\.\d", printed), (seed, printed)
