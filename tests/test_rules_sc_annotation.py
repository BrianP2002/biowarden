"""Rules sc.obs_assign_position, sc.control_label, sc.gene_vocabulary, sc.concat_join and
sc.mito_prefix_case: the spec and the checkers on small objects and real-looking code."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_rules_sc_common import (
    build,
    engine_for,
    names,
    small_counts,
    tools,  # noqa: F401  (fixture)
    verdict,
)

from biowarden.engine import known_checkers
from biowarden.engine.spec import load_packaged

SPEC = load_packaged("sc_annotation")
RULES = {r.id: r for r in SPEC.rules}
ENGINE = engine_for("sc_annotation")
PY = "python-sc"
N_CELLS, N_GENES = 24, 30


def test_spec_is_valid():
    assert [r.id for r in SPEC.rules] == [
        "sc.obs_assign_position",
        "sc.control_label",
        "sc.gene_vocabulary",
        "sc.concat_join",
        "sc.mito_prefix_case",
    ]
    registry = known_checkers()
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier, rule.declare.required) == ("fact", "T1", False)
        assert rule.check in registry and rule.incidents and rule.message
        assert "python-sc" in rule.applies.tools and "python3" in rule.applies.tools
    keys = {r.id: r.declare.key for r in SPEC.rules}
    assert keys["sc.obs_assign_position"] == "obs_join"
    assert keys["sc.mito_prefix_case"] == "mito_prefix"


def codes(result):
    return None if result is None else result.code


@pytest.fixture(scope="module")
def world(tools, tmp_path_factory):  # noqa: F811
    work = tmp_path_factory.mktemp("sc-annotation")
    counts = small_counts(N_CELLS, N_GENES).astype(np.float32)
    cells = names("AAAC", N_CELLS)
    build(work, "cells.h5ad", tools, counts, obs_names=cells, obs={"celltype": ["a", "b"] * 12})
    rows = [f"{c},{'T' if i % 2 else 'B'}" for i, c in enumerate(cells)]
    (work / "same.csv").write_text("barcode,celltype\n" + "\n".join(rows) + "\n")
    (work / "reversed.csv").write_text("barcode,celltype\n" + "\n".join(rows[::-1]) + "\n")
    (work / "reversed.tsv").write_text(
        "barcode\tcelltype\n" + "\n".join(r.replace(",", "\t") for r in rows[::-1]) + "\n"
    )
    (work / "other.csv").write_text(
        "barcode,celltype\n" + "\n".join(f"X{i},T" for i in range(N_CELLS)) + "\n"
    )
    (work / "fewer.csv").write_text("barcode,celltype\n" + "\n".join(rows[:10]) + "\n")
    # perturbation screens
    for name, label in (("a", "ctrl"), ("b", "non-targeting"), ("c", "ctrl")):
        labels = [label] * 8 + ["GENE1"] * 8 + ["GENE2"] * 8
        build(
            work,
            f"screen_{name}.h5ad",
            tools,
            counts,
            obs_names=names(name.upper(), N_CELLS),
            obs={"perturbation": labels},
        )
    # vocabulary
    ens = [f"ENSG{100000 + i:011d}" for i in range(N_GENES)]
    symbols = names("SYM", N_GENES)
    build(work, "ens.h5ad", tools, counts, var_names=ens, var={"feature_name": symbols})
    build(work, "sym.h5ad", tools, counts, var_names=symbols, var={"feature_id": ens})
    vocab = {
        "<pad>": 0,
        "<cls>": 1,
        **{s: i + 2 for i, s in enumerate(symbols[:20] + names("DECOY", 40))},
    }
    (work / "vocab.json").write_text(json.dumps(vocab))
    (work / "vocab.txt").write_text("\n".join(vocab) + "\n")
    (work / "tiny.json").write_text(json.dumps({"a": 1, "b": 2}))
    # batches
    build(
        work,
        "batch1.h5ad",
        tools,
        counts[:, :24],
        obs_names=names("B1", N_CELLS),
        var_names=names("G", 24),
    )
    build(
        work,
        "batch2.h5ad",
        tools,
        counts[:, 6:],
        obs_names=names("B2", N_CELLS),
        var_names=names("G", 30)[6:],
    )
    build(work, "same1.h5ad", tools, counts, obs_names=names("S1", N_CELLS))
    build(work, "same2.h5ad", tools, counts, obs_names=names("S2", N_CELLS))
    # mitochondrial prefixes
    human = [*names("G", 20), "MT-ND1", "MT-CO1", "MT-ATP6", *names("H", 7)]
    mouse = [*names("G", 20), "mt-Nd1", "mt-Co1", "mt-Atp6", *names("H", 7)]
    build(work, "human.h5ad", tools, counts, var_names=human)
    build(work, "mouse.h5ad", tools, counts, var_names=mouse)
    build(work, "nomito.h5ad", tools, counts)
    return work


def run(world, rule, code, declared=None, tool=PY):
    return verdict(ENGINE, rule, tool, ["-c", code], world, declared)


# ---- sc.obs_assign_position --------------------------------------------------------------------

OBS = (
    "import pandas as pd, anndata as ad\na = ad.read_h5ad('cells.h5ad')\n"
    "ann = pd.read_csv('{table}', index_col=0{sep})\n{assign}\n"
)
PASTE = [
    "a.obs['ct'] = ann['celltype'].values",
    "a.obs['ct'] = ann['celltype'].to_numpy()",
    "a.obs['ct'] = ann['celltype'].tolist()",
    "a.obs['ct'] = list(ann['celltype'])",
    "a.obs['ct'] = np.array(ann['celltype'])",
    "a.obs = ann",
    "a.obs = pd.read_csv('reversed.csv', index_col=0)",
]
ALIGNED = [
    "a.obs['ct'] = ann['celltype']",
    "a.obs['ct'] = ann.loc[a.obs_names, 'celltype'].values",
    "a.obs['ct'] = ann.reindex(a.obs_names)['celltype'].values",
    "a.obs = a.obs.join(ann)",
    "a.obs['ct'] = a.obs_names.map(dict(zip(ann.index, ann['celltype'])))",
    "a.obs = a.obs.merge(ann, left_index=True, right_index=True)",
]


@pytest.mark.parametrize("assign", PASTE)
def test_positional_paste_of_a_reordered_table_fails(world, assign):
    code = OBS.format(table="reversed.csv", sep="", assign=assign)
    result = run(world, "sc.obs_assign_position", code)
    assert codes(result) == "positional_join_of_reordered_table"
    assert "cells.h5ad" in result.message and "reversed.csv" in result.message
    assert ".loc[adata.obs_names" in " ".join(result.fix)


@pytest.mark.parametrize("assign", ALIGNED)
def test_aligned_joins_pass(world, assign):
    code = OBS.format(table="reversed.csv", sep="", assign=assign)
    assert codes(run(world, "sc.obs_assign_position", code)) == "ok"


@pytest.mark.parametrize("assign", PASTE[:6])
def test_positional_paste_of_a_table_in_cell_order_is_fine(world, assign):
    code = OBS.format(table="same.csv", sep="", assign=assign)
    assert codes(run(world, "sc.obs_assign_position", code)) == "ok"


def test_tables_that_are_not_this_objects_are_not_judged(world):
    for table in ("other.csv", "fewer.csv"):
        code = OBS.format(table=table, sep="", assign=PASTE[0])
        assert codes(run(world, "sc.obs_assign_position", code)) == "ok"


def test_tsv_tables_are_read_with_tabs(world):
    code = OBS.format(table="reversed.tsv", sep=", sep='\\t'", assign=PASTE[0])
    assert codes(run(world, "sc.obs_assign_position", code)) == "positional_join_of_reordered_table"


def test_obs_join_declarations(world):
    paste = OBS.format(table="reversed.csv", sep="", assign=PASTE[0])
    aligned = OBS.format(table="reversed.csv", sep="", assign=ALIGNED[1])
    same = OBS.format(table="same.csv", sep="", assign=PASTE[0])
    assert codes(run(world, "sc.obs_assign_position", aligned, {"obs_join": "by_index"})) == "ok"
    assert (
        codes(run(world, "sc.obs_assign_position", aligned, {"obs_join": "by_position"}))
        == "declared_position_but_orders_differ"
    )
    assert (
        codes(run(world, "sc.obs_assign_position", paste, {"obs_join": "by_index"}))
        == "positional_join_of_reordered_table"
    )
    assert codes(run(world, "sc.obs_assign_position", same, {"obs_join": "by_position"})) == "ok"
    assert (
        codes(run(world, "sc.obs_assign_position", same, {"obs_join": "by_index"}))
        == "declared_index_but_positional_code"
    )


def test_positional_paste_in_a_script_and_a_heredoc(world):
    code = OBS.format(table="reversed.csv", sep="", assign=PASTE[0])
    (world / "join.py").write_text(code)
    assert (
        codes(verdict(ENGINE, "sc.obs_assign_position", PY, ["join.py"], world))
        == "positional_join_of_reordered_table"
    )
    assert (
        codes(verdict(ENGINE, "sc.obs_assign_position", PY, ["-"], world, context=(code,)))
        == "positional_join_of_reordered_table"
    )


# ---- sc.control_label --------------------------------------------------------------------------

CTRL = (
    "import anndata as ad\n"
    "a = ad.concat([ad.read_h5ad('screen_a.h5ad'), ad.read_h5ad('{second}')])\n{sel}\n"
)


@pytest.mark.parametrize(
    ("sel", "expected"),
    [
        ("m = a.obs['perturbation'] == 'ctrl'", "control_literal_matches_nothing"),
        ('m = a[a.obs["perturbation"] == "ctrl"]', "control_literal_matches_nothing"),
        ("m = a.obs['perturbation'] == 'control'", "control_literal_matches_nothing"),
        ("m = a.obs['perturbation'].isin(['ctrl', 'non-targeting'])", "ok"),
        ("m = a.obs['perturbation'].isin(['ctrl', 'control', 'NT', 'non-targeting'])", "ok"),
        ("m = ~a.obs['perturbation'].isin(a.var_names)", "ok"),
        ("print(a.obs['perturbation'].value_counts())", "ok"),
        ("m = a.obs['perturbation'].str.lower().str.contains('ctrl|non')", "ok"),
        ("m = a.obs['perturbation'] == 'GENE1'", "ok"),
    ],
)
def test_control_label_in_files_with_two_spellings(world, sel, expected):
    assert (
        codes(run(world, "sc.control_label", CTRL.format(second="screen_b.h5ad", sel=sel)))
        == expected
    )


def test_control_label_with_one_spelling_everywhere_is_fine(world):
    code = CTRL.format(second="screen_c.h5ad", sel="m = a.obs['perturbation'] == 'ctrl'")
    assert codes(run(world, "sc.control_label", code)) == "ok"
    one = (
        "import anndata as ad\na = ad.read_h5ad('screen_a.h5ad')\n"
        "m = a.obs['perturbation'] == 'ctrl'\n"
    )
    assert codes(run(world, "sc.control_label", one)) == "ok"
    lone_b = one.replace("screen_a", "screen_b")
    assert codes(run(world, "sc.control_label", lone_b)) == "control_literal_matches_nothing"


def test_control_label_message_lists_the_spellings_by_file(world):
    result = run(
        world,
        "sc.control_label",
        CTRL.format(second="screen_b.h5ad", sel="m = a.obs['perturbation'] == 'ctrl'"),
    )
    assert "'non-targeting'" in result.message and "screen_b.h5ad" in result.message
    assert "isin(" in " ".join(result.fix) and "multiple" in " ".join(result.fix)


def test_control_label_declarations(world):
    pair = CTRL.format(
        second="screen_b.h5ad", sel="m = a.obs['perturbation'].isin(['ctrl', 'non-targeting'])"
    )
    assert codes(run(world, "sc.control_label", pair, {"control_label": "multiple"})) == "ok"
    assert (
        codes(run(world, "sc.control_label", pair, {"control_label": "ctrl"}))
        == "declared_label_missing_in_file"
    )
    plain = CTRL.format(second="screen_c.h5ad", sel="m = a.obs['perturbation'] == 'ctrl'")
    assert codes(run(world, "sc.control_label", plain, {"control_label": "ctrl"})) == "ok"
    assert (
        codes(run(world, "sc.control_label", plain, {"control_label": "control"}))
        == "declared_label_missing_in_file"
    )


# ---- sc.gene_vocabulary ------------------------------------------------------------------------

VOC = (
    "import json, anndata as ad\na = ad.read_h5ad('{f}.h5ad')\n"
    "vocab = set(json.load(open('vocab.json')))\n{body}\n"
)


@pytest.mark.parametrize(
    ("f", "body", "expected"),
    [
        ("ens", "hit = set(a.var_names) & vocab", "var_names_not_in_vocabulary_namespace"),
        (
            "ens",
            "hit = [g for g in a.var.index if g in vocab]",
            "var_names_not_in_vocabulary_namespace",
        ),
        ("ens", "hit = a.var_names.isin(list(vocab))", "var_names_not_in_vocabulary_namespace"),
        ("ens", "hit = set(a.var['feature_name']) & vocab", "ok"),
        ("ens", "hit = a.var.feature_name.isin(vocab)", "ok"),
        ("ens", "print(a.var.columns)", "ok"),
        ("sym", "hit = set(a.var_names) & vocab", "ok"),
    ],
)
def test_gene_vocabulary(world, f, body, expected):
    assert codes(run(world, "sc.gene_vocabulary", VOC.format(f=f, body=body))) == expected


def test_gene_vocabulary_reads_text_vocabularies_and_ignores_tiny_ones(world):
    code = (
        "import anndata as ad\na = ad.read_h5ad('ens.h5ad')\n"
        "vocab = set(open('vocab.txt').read().split())\nhit = set(a.var_names) & vocab\n"
    )
    assert codes(run(world, "sc.gene_vocabulary", code)) == "var_names_not_in_vocabulary_namespace"
    tiny = code.replace("vocab.txt", "tiny.json")
    assert codes(run(world, "sc.gene_vocabulary", tiny)) == "ok"
    assert (
        codes(
            run(
                world,
                "sc.gene_vocabulary",
                "import anndata as ad\na = ad.read_h5ad('ens.h5ad')\nprint(a)\n",
            )
        )
        == "ok"
    )


def test_gene_vocabulary_message_and_declarations(world):
    code = VOC.format(f="ens", body="hit = set(a.var_names) & vocab")
    result = run(world, "sc.gene_vocabulary", code)
    assert (
        "Ensembl ids" in result.message
        and "feature_name" in result.message
        and "0% of them" in result.message
    )
    assert "--declare gene_namespace=symbol" in " ".join(result.fix)
    assert (
        codes(run(world, "sc.gene_vocabulary", code, {"gene_namespace": "ensembl_id"}))
        == "declared_namespace_contradicts_vocabulary"
    )
    assert (
        codes(run(world, "sc.gene_vocabulary", code, {"gene_namespace": "symbol"}))
        == "var_names_not_in_vocabulary_namespace"
    )
    good = VOC.format(f="ens", body="hit = set(a.var['feature_name']) & vocab")
    assert codes(run(world, "sc.gene_vocabulary", good, {"gene_namespace": "symbol"})) == "ok"


# ---- sc.concat_join ----------------------------------------------------------------------------

CAT = (
    "import anndata as ad, scanpy as sc, pandas as pd\n"
    "a, b = ad.read_h5ad('{f1}.h5ad'), ad.read_h5ad('{f2}.h5ad')\n{merge}\n"
)


@pytest.mark.parametrize(
    ("merge", "expected"),
    [
        ("m = ad.concat([a, b])", "concat_inner_join_drops_genes"),
        ("m = ad.concat([a, b], label='batch', keys=['1', '2'])", "concat_inner_join_drops_genes"),
        ("m = ad.concat([a, b], join='inner')", "concat_inner_join_drops_genes"),
        ("m = sc.concat([a, b])", "concat_inner_join_drops_genes"),
        ("m = anndata.concat([a, b])", "concat_inner_join_drops_genes"),
        ("m = a.concatenate(b)", "concat_inner_join_drops_genes"),
        ("m = ad.concat([a, b], join='outer')", "ok"),
        ('m = ad.concat([a, b], join="outer", fill_value=0)', "ok"),
        ("m = pd.concat([a.obs, b.obs])", "ok"),
        ("print(a.n_vars, b.n_vars)", "ok"),
    ],
)
def test_concat_join_on_batches_with_different_genes(world, merge, expected):
    assert (
        codes(run(world, "sc.concat_join", CAT.format(f1="batch1", f2="batch2", merge=merge)))
        == expected
    )


def test_concat_join_is_silent_for_identical_gene_sets_and_a_single_file(world):
    code = CAT.format(f1="same1", f2="same2", merge="m = ad.concat([a, b])")
    assert codes(run(world, "sc.concat_join", code)) == "ok"
    single = "import anndata as ad\na = ad.read_h5ad('batch1.h5ad')\nm = ad.concat([a, a])\n"
    assert codes(run(world, "sc.concat_join", single)) == "ok"


def test_concat_join_message_counts_the_genes(world):
    result = run(
        world, "sc.concat_join", CAT.format(f1="batch1", f2="batch2", merge="m = ad.concat([a, b])")
    )
    assert "batch1.h5ad: 24" in result.message and "batch2.h5ad: 24" in result.message
    assert "18 in common out of 30" in result.message and "12 genes disappear" in result.message
    assert "join='outer'" in " ".join(result.fix)


def test_concat_join_declarations(world):
    default = CAT.format(f1="batch1", f2="batch2", merge="m = ad.concat([a, b])")
    outer = CAT.format(f1="batch1", f2="batch2", merge="m = ad.concat([a, b], join='outer')")
    assert codes(run(world, "sc.concat_join", default, {"concat_genes": "intersection"})) == "ok"
    assert (
        codes(run(world, "sc.concat_join", default, {"concat_genes": "union"}))
        == "concat_inner_join_drops_genes"
    )
    assert codes(run(world, "sc.concat_join", outer, {"concat_genes": "union"})) == "ok"
    assert (
        codes(run(world, "sc.concat_join", outer, {"concat_genes": "intersection"}))
        == "declared_intersection_but_outer"
    )


def test_concat_from_import_form(world):
    code = (
        "from anndata import concat\nimport anndata as ad\n"
        "a, b = ad.read_h5ad('batch1.h5ad'), ad.read_h5ad('batch2.h5ad')\nm = concat([a, b])\n"
    )
    assert codes(run(world, "sc.concat_join", code)) == "concat_inner_join_drops_genes"


# ---- sc.mito_prefix_case -----------------------------------------------------------------------

MITO = "import anndata as ad\na = ad.read_h5ad('{f}.h5ad')\n{mark}\n"


@pytest.mark.parametrize(
    ("f", "mark", "expected"),
    [
        ("mouse", "a.var['mt'] = a.var_names.str.startswith('MT-')", "mito_prefix_case_mismatch"),
        ("mouse", 'a.var["mt"] = a.var_names.str.startswith("MT-")', "mito_prefix_case_mismatch"),
        ("mouse", "a.var['mt'] = a.var_names.str.match('^MT-')", "mito_prefix_case_mismatch"),
        ("mouse", "a.var['mt'] = a.var_names.str.contains('^MT-')", "mito_prefix_case_mismatch"),
        ("mouse", "a.var['mt'] = a.var_names.str.upper().str.startswith('MT-')", "ok"),
        ("mouse", "a.var['mt'] = a.var_names.str.lower().str.startswith('mt-')", "ok"),
        ("mouse", "a.var['mt'] = a.var_names.str.startswith(('MT-', 'mt-'))", "ok"),
        ("mouse", "a.var['mt'] = a.var_names.str.contains('^MT-', case=False)", "ok"),
        ("mouse", "a.var['mt'] = a.var_names.str.startswith('mt-')", "ok"),
        ("mouse", "print(a.var_names[:5])", "ok"),
        ("human", "a.var['mt'] = a.var_names.str.startswith('MT-')", "ok"),
        ("nomito", "a.var['mt'] = a.var_names.str.startswith('MT-')", "ok"),
    ],
)
def test_mito_prefix_case(world, f, mark, expected):
    assert codes(run(world, "sc.mito_prefix_case", MITO.format(f=f, mark=mark))) == expected


def test_mito_prefix_message_and_declarations(world):
    code = MITO.format(f="mouse", mark="a.var['mt'] = a.var_names.str.startswith('MT-')")
    result = run(world, "sc.mito_prefix_case", code)
    assert "'mt-Nd1'" in result.message and "3 genes" in result.message
    assert "str.upper()" in " ".join(result.fix)
    assert (
        codes(run(world, "sc.mito_prefix_case", code, {"mito_prefix": "MT-"}))
        == "declared_prefix_contradicts_data"
    )
    assert (
        codes(run(world, "sc.mito_prefix_case", code, {"mito_prefix": "mt-"}))
        == "mito_prefix_case_mismatch"
    )
    folded = MITO.format(
        f="mouse", mark="a.var['mt'] = a.var_names.str.upper().str.startswith('MT-')"
    )
    assert codes(run(world, "sc.mito_prefix_case", folded, {"mito_prefix": "mt-"})) == "ok"
    assert codes(run(world, "sc.mito_prefix_case", folded, {"mito_prefix": "both"})) == "ok"
    assert (
        codes(run(world, "sc.mito_prefix_case", code, {"mito_prefix": "both"}))
        == "mito_prefix_case_mismatch"
    )
    human = MITO.format(f="human", mark="a.var['mt'] = a.var_names.str.startswith('MT-')")
    assert codes(run(world, "sc.mito_prefix_case", human, {"mito_prefix": "MT-"})) == "ok"
    assert (
        codes(run(world, "sc.mito_prefix_case", human, {"mito_prefix": "mt-"}))
        == "declared_prefix_contradicts_data"
    )


# ---- all rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", sorted(RULES))
def test_unrelated_and_missing_inputs_cost_nothing(world, tmp_path, rule):
    for command in ("print(1)", "import anndata as ad\na = ad.read_h5ad('cells.h5ad')\nprint(a)\n"):
        got = run(world, rule, command)
        assert got is None or got.verdict == "pass"
    gone = verdict(ENGINE, rule, PY, ["-c", "ad.read_h5ad('nofile.h5ad')"], tmp_path)
    assert gone is None


def test_without_h5py_the_annotation_rules_only_warn(world, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "h5py", None)
    code = MITO.format(f="mouse", mark="a.var['mt'] = a.var_names.str.startswith('MT-')")
    assert run(world, "sc.mito_prefix_case", code).code == "h5py_missing"
    code = CAT.format(f1="batch1", f2="batch2", merge="m = ad.concat([a, b])")
    assert run(world, "sc.concat_join", code).verdict == "unknown"
    code = CTRL.format(second="screen_b.h5ad", sel="m = a.obs['perturbation'] == 'ctrl'")
    assert run(world, "sc.control_label", code).verdict == "unknown"
