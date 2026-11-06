"""Rules sc.de_raw_default, sc.small_int_qc, sc.double_transform, sc.counts_needed and
sc.scaled_x_expression: the spec, the checkers on small objects and real-looking code, admission.

The unit tests build small AnnData objects with anndata (python-sc) once per module. The admission
tests (test_rules_sc_admission.py) measure each rule over seeds of the benchmark families with
calls labelled by running the code.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_rules_sc_common import (
    build,
    engine_for,
    log_cp10k,
    names,
    small_counts,
    tools,  # noqa: F401  (fixture)
    verdict,
)

from biowarden.engine import known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from biowarden.engine.tools import watched_tools

SPEC = load_packaged("sc_matrix_state")
RULES = {r.id: r for r in SPEC.rules}
ENGINE = engine_for("sc_matrix_state")
PY = "python-sc"


def test_spec_is_valid_and_every_rule_is_a_t1_fact_rule_with_an_optional_declaration():
    assert [r.id for r in SPEC.rules] == [
        "sc.de_raw_default",
        "sc.small_int_qc",
        "sc.double_transform",
        "sc.counts_needed",
        "sc.scaled_x_expression",
    ]
    registry = known_checkers()
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier) == ("fact", "T1")
        assert rule.declare is not None and rule.declare.required is False
        assert rule.check in registry and rule.incidents and rule.message
        assert {"python-sc", "python3"} <= set(rule.applies.tools)
    assert RULES["sc.de_raw_default"].declare.key == "de_input"
    assert RULES["sc.small_int_qc"].declare.key == "x_dtype"
    assert {RULES[i].declare.key for i in RULES if "state" in RULES[i].declare.key} == {"x_state"}


def test_the_shim_would_watch_python_sc():
    assert {"python-sc", "python3", "python"} <= set(watched_tools([SPEC]))


@pytest.fixture(scope="module")
def objs(tools, tmp_path_factory):  # noqa: F811
    work = tmp_path_factory.mktemp("sc-state")
    counts = small_counts()
    logged = log_cp10k(counts)
    cp10k = np.expm1(logged).astype(np.float32)
    z = ((logged - logged.mean(0)) / (logged.std(0) + 1e-6)).astype(np.float32)
    z[:, 0] = logged[:, 0] - logged[:, 0].mean()
    big = np.tile(counts, (4, 1)) * 40  # cells above 32,767 UMIs
    files = {
        "raw_cp10k": build(work, "raw_cp10k.h5ad", tools, logged, raw={"X": cp10k}),
        "raw_log": build(work, "raw_log.h5ad", tools, z, raw={"X": logged}),
        "raw_counts": build(
            work, "raw_counts.h5ad", tools, logged, raw={"X": counts.astype(np.float32)}
        ),
        "raw_copy": build(work, "raw_copy.h5ad", tools, logged, raw={"X": logged}),
        "no_raw": build(work, "no_raw.h5ad", tools, logged),
        "counts": build(work, "counts.h5ad", tools, counts.astype(np.float32)),
        "log_nomark": build(work, "log_nomark.h5ad", tools, logged),
        "log_mark": build(work, "log_mark.h5ad", tools, logged, uns={"log1p": {"base": None}}),
        "scaled_noraw": build(work, "scaled_noraw.h5ad", tools, z),
        "layer": build(
            work, "layer.h5ad", tools, logged, layers={"counts": counts.astype(np.float32)}
        ),
        "i16": build(work, "i16.h5ad", tools, big.astype(np.int16)),
        "i16dense": build(work, "i16dense.h5ad", tools, big.astype(np.int16), sparse=False),
        "i8": build(work, "i8.h5ad", tools, np.minimum(counts, 100).astype(np.int8)),
        "f32": build(work, "f32.h5ad", tools, big.astype(np.float32)),
    }
    (work / "de.py").write_text(
        "import anndata as ad, scanpy as sc\n"
        "a = ad.read_h5ad('raw_cp10k.h5ad')\n"
        "sc.tl.rank_genes_groups(a, 'celltype')\n"
    )
    (work / "de_fixed.py").write_text(
        "import anndata as ad, scanpy as sc\n"
        "a = ad.read_h5ad('raw_cp10k.h5ad')\n"
        "sc.tl.rank_genes_groups(a, 'celltype', use_raw=False)\n"
    )
    files["work"] = work
    return files


def run(objs, rule, code, declared=None, tool=PY):
    return verdict(ENGINE, rule, tool, ["-c", code], objs["work"], declared)


def codes(result):
    return None if result is None else result.code


# ---- sc.de_raw_default -------------------------------------------------------------------------

DE = "import anndata as ad, scanpy as sc\na = ad.read_h5ad('{f}.h5ad')\n{call}\n"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("sc.tl.rank_genes_groups(a, 'g')", "default_reads_raw"),
        ("sc.tl.rank_genes_groups(a, 'g', method='wilcoxon', use_raw=None)", "default_reads_raw"),
        ("sc.tl.rank_genes_groups(a, 'g', use_raw=False)", "ok"),
        ("sc.tl.rank_genes_groups(a, 'g', use_raw=True)", "ok"),
        ("sc.tl.rank_genes_groups(a, 'g', layer='counts')", "ok"),
        ("a.raw = None\nsc.tl.rank_genes_groups(a, 'g')", "ok"),
        ("del a.raw\nsc.tl.rank_genes_groups(a, 'g')", "ok"),
        ("a = a.raw.to_adata()\nsc.tl.rank_genes_groups(a, 'g')", "ok"),
        ("# use_raw=False\nsc.tl.rank_genes_groups(a, 'g')", "default_reads_raw"),
        ("print(a.raw.X.max())", "ok"),
    ],
)
def test_de_raw_default_on_a_raw_of_unlogged_values(objs, call, expected):
    assert codes(run(objs, "sc.de_raw_default", DE.format(f="raw_cp10k", call=call))) == expected


def test_de_raw_default_message_names_both_matrices(objs):
    result = run(
        objs, "sc.de_raw_default", DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g')")
    )
    assert "raw_cp10k.h5ad" in result.message and "normalised, not logged" in result.message
    assert "log-scale" in result.message and "use_raw=False" in " ".join(result.fix)


@pytest.mark.parametrize("f", ["raw_log", "raw_copy", "no_raw"])
def test_de_raw_default_is_silent_when_raw_is_what_a_log_default_needs(objs, f):
    # tutorial layout (X scaled, raw log-normalised), a copy of X, or no raw at all
    call = "sc.tl.rank_genes_groups(a, 'g')"
    assert codes(run(objs, "sc.de_raw_default", DE.format(f=f, call=call))) == "ok"


def test_de_raw_default_on_integer_counts_in_raw(objs):
    call = "sc.tl.rank_genes_groups(a, 'g')"
    assert (
        codes(run(objs, "sc.de_raw_default", DE.format(f="raw_counts", call=call)))
        == "default_reads_raw"
    )


def test_de_raw_default_declarations(objs):
    default = DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g')")
    fixed = DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g', use_raw=False)")
    assert (
        codes(run(objs, "sc.de_raw_default", default, {"de_input": "X"}))
        == "declared_x_but_default_reads_raw"
    )
    assert codes(run(objs, "sc.de_raw_default", fixed, {"de_input": "X"})) == "ok"
    assert (
        codes(run(objs, "sc.de_raw_default", fixed, {"de_input": "raw"}))
        == "declared_raw_but_use_raw_false"
    )
    # declaring raw says the default read of .raw is meant
    assert codes(run(objs, "sc.de_raw_default", default, {"de_input": "raw"})) == "ok"
    bad = run(objs, "sc.de_raw_default", default, {"de_input": "both"})
    assert codes(bad) == "declaration_invalid"


def test_de_raw_default_finds_the_file_in_a_script_and_in_a_heredoc(objs):
    work = objs["work"]
    assert codes(verdict(ENGINE, "sc.de_raw_default", PY, ["de.py"], work)) == "default_reads_raw"
    assert codes(verdict(ENGINE, "sc.de_raw_default", PY, ["de_fixed.py"], work)) == "ok"
    body = (work / "de.py").read_text()
    result = verdict(ENGINE, "sc.de_raw_default", PY, ["-"], work, context=(body,))
    assert codes(result) == "default_reads_raw"
    assert (
        codes(verdict(ENGINE, "sc.de_raw_default", "python3", ["de.py"], work))
        == "default_reads_raw"
    )


def test_de_raw_default_block_message_starts_with_the_fix(objs):
    decision = ENGINE.evaluate(
        __import__("biowarden.engine", fromlist=["Call"]).Call(
            PY,
            ("-c", DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g')")),
            objs["work"],
        )
    )
    text = render_block(decision, ENGINE.retry_cap)
    assert text.startswith("FIX:") and "use_raw=False" in text.splitlines()[0]


# ---- sc.small_int_qc ---------------------------------------------------------------------------

QC = "import anndata as ad, scanpy as sc\na = ad.read_h5ad('{f}.h5ad')\n{pre}{call}\n"
QC_CALL = "sc.pp.calculate_qc_metrics(a, percent_top=[20], inplace=True)"


@pytest.mark.parametrize(
    ("f", "pre", "call", "expected"),
    [
        ("i16", "", QC_CALL, "small_int_sparse_qc"),
        ("i8", "", QC_CALL, "small_int_sparse_qc"),
        ("i16", "", "sc.pp.calculate_qc_metrics(a, inplace=True)", "small_int_sparse_qc"),
        ("i16", "a.X = a.X.astype('float32')\n", QC_CALL, "ok"),
        ("i16", "a.X = a.X.toarray()\n", QC_CALL, "ok"),
        ("i16", "", "sc.pp.calculate_qc_metrics(a, percent_top=None, inplace=True)", "ok"),
        ("i16dense", "", QC_CALL, "ok"),
        ("f32", "", QC_CALL, "ok"),
        ("i16", "", "print(a.X.dtype)", "ok"),
    ],
)
def test_small_int_qc(objs, f, pre, call, expected):
    assert codes(run(objs, "sc.small_int_qc", QC.format(f=f, pre=pre, call=call))) == expected


def test_small_int_qc_message_and_declarations(objs):
    code = QC.format(f="i16", pre="", call=QC_CALL)
    result = run(objs, "sc.small_int_qc", code)
    assert "sparse int16" in result.message and "astype('float32')" in " ".join(result.fix)
    assert (
        codes(run(objs, "sc.small_int_qc", code, {"x_dtype": "float32"}))
        == "declared_dtype_contradicts_data"
    )
    assert codes(run(objs, "sc.small_int_qc", code, {"x_dtype": "int16"})) == "small_int_sparse_qc"
    cast = QC.format(f="i16", pre="a.X = a.X.astype('float32')\n", call=QC_CALL)
    assert codes(run(objs, "sc.small_int_qc", cast, {"x_dtype": "int16"})) == "ok"
    assert (
        codes(
            run(
                objs,
                "sc.small_int_qc",
                QC.format(f="f32", pre="", call=QC_CALL),
                {"x_dtype": "float32"},
            )
        )
        == "ok"
    )


# ---- sc.double_transform -----------------------------------------------------------------------

DT = "import anndata as ad, scanpy as sc, numpy as np\na = ad.read_h5ad('{f}.h5ad')\n{body}\n"


@pytest.mark.parametrize(
    ("f", "body", "expected"),
    [
        (
            "log_nomark",
            "sc.pp.normalize_total(a, target_sum=1e4)\nsc.pp.log1p(a)",
            "second_transform",
        ),
        ("log_nomark", "sc.pp.log1p(a)", "second_transform"),
        ("log_mark", "sc.pp.log1p(a)", "second_transform"),
        ("log_nomark", "a.X = np.log1p(a.X)", "second_transform"),
        ("log_nomark", "if 'log1p' not in a.uns:\n    sc.pp.log1p(a)", "second_transform"),
        ("scaled_noraw", "sc.pp.log1p(a)", "second_transform"),
        ("counts", "sc.pp.normalize_total(a, target_sum=1e4)\nsc.pp.log1p(a)", "ok"),
        ("log_nomark", "a.X = np.expm1(a.X)\nsc.pp.normalize_total(a)\nsc.pp.log1p(a)", "ok"),
        ("log_nomark", "v = a.X.data\nif np.allclose(v, np.round(v)):\n    sc.pp.log1p(a)", "ok"),
        ("layer", "a.X = a.layers['counts'].copy()\nsc.pp.log1p(a)", "ok"),
        ("log_nomark", "print(a.X.max())", "ok"),
        ("raw_cp10k", "sc.pp.log1p(a)", "second_transform"),
    ],
)
def test_double_transform(objs, f, body, expected):
    assert codes(run(objs, "sc.double_transform", DT.format(f=f, body=body))) == expected


def test_double_transform_message_and_declarations(objs):
    code = DT.format(f="log_nomark", body="sc.pp.log1p(a)")
    result = run(objs, "sc.double_transform", code)
    assert "no 'log1p' key" in result.message and "log-scale" in result.message
    assert "--declare x_state=log_normalized" in " ".join(result.fix)
    marked = run(objs, "sc.double_transform", DT.format(f="log_mark", body="sc.pp.log1p(a)"))
    assert "scanpy would warn" in marked.message
    assert (
        codes(run(objs, "sc.double_transform", code, {"x_state": "counts"}))
        == "declared_state_contradicts_data"
    )
    assert (
        codes(run(objs, "sc.double_transform", code, {"x_state": "log_normalized"}))
        == "second_transform"
    )
    ok = DT.format(f="log_nomark", body="print(a.X.mean())")
    assert codes(run(objs, "sc.double_transform", ok, {"x_state": "log_normalized"})) == "ok"


# ---- sc.counts_needed --------------------------------------------------------------------------

CN = "import anndata as ad, scanpy as sc, numpy as np\na = ad.read_h5ad('{f}.h5ad')\n{body}\n"


@pytest.mark.parametrize(
    ("f", "body", "expected"),
    [
        ("layer", "sc.pp.filter_cells(a, min_counts=500)", "counts_step_on_normalized_x"),
        (
            "layer",
            "sc.pp.calculate_qc_metrics(a, percent_top=None, inplace=True)",
            "counts_step_on_normalized_x",
        ),
        (
            "layer",
            "n = (np.asarray(a.X.sum(axis=1)).ravel() > 500).sum()",
            "counts_step_on_normalized_x",
        ),
        ("log_nomark", "sc.pp.filter_cells(a, min_counts=500)", "counts_step_on_normalized_x"),
        ("layer", "a.X = a.layers['counts'].copy()\nsc.pp.filter_cells(a, min_counts=500)", "ok"),
        ("layer", "sc.pp.calculate_qc_metrics(a, layer='counts', percent_top=None)", "ok"),
        (
            "layer",
            "lay = 'counts'\nsc.pp.calculate_qc_metrics(a, layer=lay, percent_top=None)",
            "ok",
        ),
        (
            "layer",
            "sc.pp.calculate_qc_metrics(a, layer=None, percent_top=None)",
            "counts_step_on_normalized_x",
        ),
        ("counts", "sc.pp.filter_cells(a, min_counts=500)", "ok"),
        ("layer", "print(a)", "ok"),
    ],
)
def test_counts_needed(objs, f, body, expected):
    assert codes(run(objs, "sc.counts_needed", CN.format(f=f, body=body))) == expected


def test_counts_needed_names_the_layer_that_holds_the_counts(objs):
    result = run(
        objs, "sc.counts_needed", CN.format(f="layer", body="sc.pp.filter_cells(a, min_counts=5)")
    )
    assert "layers['counts']" in result.message
    bare = run(
        objs,
        "sc.counts_needed",
        CN.format(f="log_nomark", body="sc.pp.filter_cells(a, min_counts=5)"),
    )
    assert "layers['counts']" not in bare.message and "raw count matrix" in " ".join(bare.fix)
    body = "sc.pp.filter_cells(a, min_counts=5)"
    assert (
        codes(run(objs, "sc.counts_needed", CN.format(f="layer", body=body), {"x_state": "counts"}))
        == "declared_state_contradicts_data"
    )


# ---- sc.scaled_x_expression --------------------------------------------------------------------

SX = "import anndata as ad, scanpy as sc, numpy as np\na = ad.read_h5ad('{f}.h5ad')\n{body}\n"


@pytest.mark.parametrize(
    ("f", "body", "expected"),
    [
        ("raw_log", "x = a[:, 'G003'].X\nn = (x > 0).sum()", "scaled_x_read_as_expression"),
        ("raw_log", "df = a.to_df()", "scaled_x_read_as_expression"),
        ("raw_log", "x = sc.get.obs_df(a, keys=['G003'])", "scaled_x_read_as_expression"),
        ("raw_log", "x = a.raw[:, 'G003'].X", "ok"),
        ("raw_log", "x = sc.get.obs_df(a, keys=['G003'], use_raw=True)", "ok"),
        ("raw_log", "x = a.raw.to_adata()[:, 'G003'].X", "ok"),
        ("raw_log", "print(a.obs.shape)", "ok"),
        ("raw_log", "print(a.X.min(), a.X.max(), a.X.shape, a.X.dtype)", "ok"),
        ("raw_log", "x = a.X[:, 3]", "scaled_x_read_as_expression"),
        ("raw_log", "print(a.X.mean())", "scaled_x_read_as_expression"),
        ("raw_copy", "x = a[:, 'G003'].X", "ok"),
        ("scaled_noraw", "x = a[:, 'G003'].X", "scaled_x_no_raw"),
    ],
)
def test_scaled_x_expression(objs, f, body, expected):
    assert codes(run(objs, "sc.scaled_x_expression", SX.format(f=f, body=body))) == expected


def test_scaled_x_unknown_only_warns_and_declarations(objs):
    code = SX.format(f="scaled_noraw", body="x = a[:, 'G003'].X")
    result = run(objs, "sc.scaled_x_expression", code)
    assert result.verdict == "unknown" and "z-scores" in result.message
    decision = ENGINE.evaluate(
        __import__("biowarden.engine", fromlist=["Call"]).Call(PY, ("-c", code), objs["work"])
    )
    assert decision.action == "warn"
    raw = SX.format(f="raw_log", body="x = a[:, 'G003'].X")
    assert (
        codes(run(objs, "sc.scaled_x_expression", raw, {"x_state": "log_normalized"}))
        == "declared_state_contradicts_data"
    )
    handled = SX.format(f="raw_log", body="x = a.raw[:, 'G003'].X")
    assert codes(run(objs, "sc.scaled_x_expression", handled, {"x_state": "scaled"})) == "ok"


# ---- all rules: friction on correct and unrelated commands -------------------------------------


@pytest.mark.parametrize("rule", sorted(RULES))
def test_unrelated_commands_cost_nothing(objs, tmp_path, rule):
    work = objs["work"]
    for argv in (["-c", "print(1 + 1)"], ["-c", "import numpy as np; print(np.arange(3))"]):
        assert verdict(ENGINE, rule, PY, argv, work) is None
    for command in (
        "import anndata as ad\na = ad.read_h5ad('raw_cp10k.h5ad')\nprint(a.shape, a.obs.columns)\n",
        "import h5py\nf = h5py.File('i16.h5ad')\nprint(list(f))\n",
    ):
        got = verdict(ENGINE, rule, PY, ["-c", command], work)
        assert got is None or got.verdict == "pass"
    missing = verdict(ENGINE, rule, PY, ["-c", "ad.read_h5ad('nofile.h5ad')"], tmp_path)
    assert missing is None


@pytest.mark.parametrize("rule", sorted(RULES))
def test_unreadable_files_do_not_crash_or_block(tools, tmp_path, rule):  # noqa: F811
    (tmp_path / "bad.h5ad").write_bytes(b"\x00\x01 not hdf5")
    code = (
        "import scanpy as sc\na = sc.read_h5ad('bad.h5ad')\n"
        "sc.tl.rank_genes_groups(a, 'g')\nsc.pp.log1p(a)\n"
    )
    got = verdict(ENGINE, rule, PY, ["-c", code], tmp_path)
    assert got is None or got.verdict in ("pass", "unknown")


def test_python_3_and_plain_python_are_watched_too(objs):
    code = DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g')")
    for tool in ("python3", "python"):
        assert codes(run(objs, "sc.de_raw_default", code, tool=tool)) == "default_reads_raw"
    assert verdict(ENGINE, "sc.de_raw_default", "Rscript", ["-e", code], objs["work"]) is None


def test_names_helper_is_stable():
    assert names("G", 3) == ["G000", "G001", "G002"]


def test_without_h5py_the_rules_that_need_it_only_warn(objs, monkeypatch):
    import sys

    from biowarden.engine import Call

    monkeypatch.setitem(sys.modules, "h5py", None)  # `import h5py` now raises ImportError
    code = DE.format(f="raw_cp10k", call="sc.tl.rank_genes_groups(a, 'g')")
    result = run(objs, "sc.de_raw_default", code)
    assert result.verdict == "unknown" and result.code == "h5py_missing"
    decision = ENGINE.evaluate(Call(PY, ("-c", code), objs["work"]))
    assert decision.action == "warn"
    # a command that does not do the risky operation is still silent
    assert (
        codes(
            run(objs, "sc.de_raw_default", "import anndata as ad\nprint(1)\nx = 'raw_cp10k.h5ad'")
        )
        == "ok"
    )
