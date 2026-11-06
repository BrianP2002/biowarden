"""Rules sc.csv_orientation, sc.mtx_orientation and sc.loom_orientation: the sniffers and the
checkers on small files written here (no anndata needed), awk and python-sc spellings, gzip,
declarations."""

from __future__ import annotations

import gzip

import numpy as np
import pytest
from test_rules_sc_common import engine_for, verdict

from biowarden.bench.tasks.extra.f107_mtx_orientation import mtx_text
from biowarden.engine import known_checkers
from biowarden.engine.checks import Ctx
from biowarden.engine.rules import sc_orientation as so
from biowarden.engine.spec import load_packaged

SPEC = load_packaged("sc_orientation")
ENGINE = engine_for("sc_orientation")
PY = "python-sc"
CELLS, GENES = "cells_by_genes", "genes_by_cells"
BASES = np.array(list("ACGT"))


def test_spec_is_valid():
    assert [r.id for r in SPEC.rules] == [
        "sc.csv_orientation",
        "sc.mtx_orientation",
        "sc.loom_orientation",
    ]
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier, rule.declare.key) == ("fact", "T1", "matrix_orientation")
        assert rule.declare.required is False and rule.check in known_checkers()
        assert {"python-sc", "python3", "awk"} <= set(rule.applies.tools)


def barcodes(n: int) -> list[str]:
    rng = np.random.default_rng(3)
    return ["".join(rng.choice(BASES, 16)) + "-1" for _ in range(n)]


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    work = tmp_path_factory.mktemp("sc-orient")
    rng = np.random.default_rng(1)
    cells, genes = barcodes(40), [f"GENE{i}" for i in range(50)]
    counts = rng.poisson(2.0, size=(40, 50))
    rows = lambda names, m: [f"{n}," + ",".join(map(str, r)) for n, r in zip(names, m, strict=True)]  # noqa: E731
    (work / "cells_rows.csv").write_text(
        "\n".join(["barcode," + ",".join(genes), *rows(cells, counts)]) + "\n"
    )
    (work / "genes_rows.csv").write_text(
        "\n".join(["gene," + ",".join(cells), *rows(genes, counts.T)]) + "\n"
    )
    (work / "genes_rows.tsv").write_text(
        "\n".join(
            ["gene\t" + "\t".join(cells), *(r.replace(",", "\t") for r in rows(genes, counts.T))]
        )
        + "\n"
    )
    (work / "genes_rows.csv.gz").write_bytes(gzip.compress((work / "genes_rows.csv").read_bytes()))
    (work / "named.csv").write_text(
        "\n".join(["gene," + ",".join(f"S{i}" for i in range(40)), *rows(genes, counts.T)]) + "\n"
    )
    (work / "sheet.csv").write_text(
        "sample,age,sex\n" + "\n".join(f"S{i},{30 + i},F" for i in range(30)) + "\n"
    )
    (work / "short.csv").write_text("gene,c1\nA,1\n")
    # 10x style directories (flat: prefix names)
    n_cells, n_genes = 30, 45
    m = rng.poisson(1.5, size=(n_genes, n_cells))
    bc, ft = (
        barcodes(n_cells),
        "".join(f"ENSG{i:011d}\tGENE{i}\tGene Expression\n" for i in range(n_genes)),
    )
    for prefix, matrix in (("g_", m), ("c_", m.T)):
        (work / f"{prefix}matrix.mtx.gz").write_bytes(gzip.compress(mtx_text(matrix).encode()))
        (work / f"{prefix}barcodes.tsv.gz").write_bytes(
            gzip.compress(("\n".join(bc) + "\n").encode())
        )
        (work / f"{prefix}features.tsv.gz").write_bytes(gzip.compress(ft.encode()))
    (work / "matrix.mtx").write_text(mtx_text(m))
    (work / "barcodes.tsv").write_text("\n".join(bc) + "\n")
    (work / "features.tsv").write_text(ft)
    sq = rng.poisson(1.0, size=(30, 30))  # as many barcodes as features: not decidable
    (work / "s_matrix.mtx").write_text(mtx_text(sq))
    (work / "s_barcodes.tsv").write_text("\n".join(bc) + "\n")
    (work / "s_features.tsv").write_text(
        "\n".join(f"G{i}\tg{i}\tGene Expression" for i in range(30)) + "\n"
    )
    (work / "lonely_matrix.mtx").write_text(mtx_text(m))
    return work


def codes(result):
    return None if result is None else result.code


# ---- the sniffers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "orientation"),
    [
        ("cells_rows.csv", CELLS),
        ("genes_rows.csv", GENES),
        ("genes_rows.tsv", GENES),
        ("genes_rows.csv.gz", GENES),
        ("named.csv", None),
        ("sheet.csv", None),
        ("short.csv", None),
    ],
)
def test_table_orientation(tables, name, orientation):
    got, detail = so.table_orientation(Ctx(tables, "T1"), tables / name)
    assert got == orientation, detail
    if orientation == GENES:
        assert "are barcodes" in detail


@pytest.mark.parametrize(
    ("prefix", "orientation"),
    [("g_", GENES), ("c_", CELLS), ("s_", None), ("lonely_", None), ("", GENES)],
)
def test_mtx_orientation(tables, prefix, orientation):
    path = tables / (f"{prefix}matrix.mtx.gz" if prefix in ("g_", "c_") else f"{prefix}matrix.mtx")
    got, detail = so.mtx_orientation(path)
    assert got == orientation, detail


# ---- sc.csv_orientation ------------------------------------------------------------------------

CSV = "import json, pandas as pd\ndf = pd.read_csv('{f}', index_col=0)\n{body}\n"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("n = ((df > 0).sum(axis=1) >= 20).sum()", "rows_are_genes_not_cells"),
        ("n = ((df > 0).sum(1) >= 20).sum()", "rows_are_genes_not_cells"),
        ("n = sum(1 for _, r in df.iterrows() if (r > 0).sum() >= 20)", "rows_are_genes_not_cells"),
        ("n = df.mean(axis=1)", "rows_are_genes_not_cells"),
        ("df = df.T\nn = ((df > 0).sum(axis=1) >= 20).sum()", "ok"),
        ("n = ((df > 0).sum(axis=0) >= 20).sum()", "ok"),
        ("n = ((df.transpose() > 0).sum(axis=1) >= 20).sum()", "ok"),
        ("print(df.shape)", "ok"),
        ("print(df.head())", "ok"),
    ],
)
def test_csv_orientation_on_a_table_with_cells_in_columns(tables, body, expected):
    code = CSV.format(f="genes_rows.csv", body=body)
    assert codes(verdict(ENGINE, "sc.csv_orientation", PY, ["-c", code], tables)) == expected


def test_csv_orientation_is_silent_for_cells_in_rows_and_for_tables_it_cannot_read(tables):
    for name in ("cells_rows.csv", "named.csv", "sheet.csv", "short.csv"):
        code = CSV.format(f=name, body="n = (df > 0).sum(axis=1)")
        assert codes(verdict(ENGINE, "sc.csv_orientation", PY, ["-c", code], tables)) == "ok"


def test_csv_orientation_in_tsv_gzip_and_scanpy_spellings(tables):
    for name in ("genes_rows.tsv", "genes_rows.csv.gz"):
        code = CSV.format(f=name, body="n = (df > 0).sum(axis=1)")
        assert (
            codes(verdict(ENGINE, "sc.csv_orientation", PY, ["-c", code], tables))
            == "rows_are_genes_not_cells"
        )
    scanpy = (
        "import scanpy as sc\n"
        "a = sc.read_csv('genes_rows.csv', first_column_names=True)\nprint(a.n_obs)\n"
    )
    assert (
        codes(verdict(ENGINE, "sc.csv_orientation", PY, ["-c", scanpy], tables))
        == "rows_are_genes_not_cells"
    )
    fixed = scanpy.replace("print(a.n_obs)", "a = a.T\nprint(a.n_obs)")
    assert codes(verdict(ENGINE, "sc.csv_orientation", PY, ["-c", fixed], tables)) == "ok"


AWK_ROW = "NR>1{n=0; for(i=2;i<=NF;i++) if($i>0) n++; if(n>=20) c++} END{print c}"
AWK_COL = "NR>1{for(i=2;i<=NF;i++) if($i>0) n[i]++} END{for(i in n) if(n[i]>=20) c++; print c}"


def test_csv_orientation_with_awk(tables):
    row = verdict(ENGINE, "sc.csv_orientation", "awk", ["-F,", AWK_ROW, "genes_rows.csv"], tables)
    assert codes(row) == "rows_are_genes_not_cells"
    col = verdict(ENGINE, "sc.csv_orientation", "awk", ["-F,", AWK_COL, "genes_rows.csv"], tables)
    assert codes(col) == "ok"
    ok = verdict(ENGINE, "sc.csv_orientation", "awk", ["-F,", AWK_ROW, "cells_rows.csv"], tables)
    assert codes(ok) == "ok"
    head = verdict(ENGINE, "sc.csv_orientation", "head", ["genes_rows.csv"], tables)
    assert head is None


def test_csv_orientation_message_and_declarations(tables):
    code = CSV.format(f="genes_rows.csv", body="n = (df > 0).sum(axis=1)")
    result = verdict(ENGINE, "sc.csv_orientation", PY, ["-c", code], tables)
    assert "one row per gene" in result.message and "'GENE0'" in result.message
    assert ".T" in " ".join(result.fix) and "genes_by_cells" in " ".join(result.fix)
    wrong = verdict(
        ENGINE, "sc.csv_orientation", PY, ["-c", code], tables, {"matrix_orientation": CELLS}
    )
    assert codes(wrong) == "declared_orientation_contradicts_table"
    same = verdict(
        ENGINE, "sc.csv_orientation", PY, ["-c", code], tables, {"matrix_orientation": GENES}
    )
    assert codes(same) == "rows_are_genes_not_cells"
    fixed = CSV.format(f="genes_rows.csv", body="n = (df.T > 0).sum(axis=1)")
    good = verdict(
        ENGINE, "sc.csv_orientation", PY, ["-c", fixed], tables, {"matrix_orientation": GENES}
    )
    assert codes(good) == "ok"
    cells = CSV.format(f="cells_rows.csv", body="n = (df > 0).sum(axis=1)")
    assert (
        codes(
            verdict(
                ENGINE,
                "sc.csv_orientation",
                PY,
                ["-c", cells],
                tables,
                {"matrix_orientation": GENES},
            )
        )
        == "declared_orientation_contradicts_table"
    )


# ---- sc.mtx_orientation ------------------------------------------------------------------------

MTX = "import numpy as np, scipy.io\nm = scipy.io.mmread('{f}').tocsr()\n{body}\n"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("tot = np.asarray(m.sum(axis=1)).ravel()", "mtx_rows_are_genes_not_cells"),
        ("tot = np.asarray(m.sum(1)).ravel()", "mtx_rows_are_genes_not_cells"),
        ("tot = m.getnnz(1)", "mtx_rows_are_genes_not_cells"),
        ("tot = np.asarray(m.sum(axis=0)).ravel()", "ok"),
        ("m = m.T.tocsr()\ntot = np.asarray(m.sum(axis=1)).ravel()", "ok"),
        ("n_barcodes = 30\nif m.shape[0] != n_barcodes:\n    m = m.T\ntot = m.sum(axis=1)", "ok"),
        ("print(m.shape)", "ok"),
    ],
)
def test_mtx_orientation_with_genes_in_rows(tables, body, expected):
    code = MTX.format(f="g_matrix.mtx.gz", body=body)
    assert codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", code], tables)) == expected


def test_mtx_orientation_is_silent_when_cells_are_rows_or_undecidable(tables):
    for name in ("c_matrix.mtx.gz", "s_matrix.mtx", "lonely_matrix.mtx"):
        code = MTX.format(f=name, body="tot = np.asarray(m.sum(axis=1)).ravel()")
        got = codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", code], tables))
        assert got == "ok"


def test_mtx_orientation_scanpy_spellings_and_plain_files(tables):
    read = "import scanpy as sc\na = sc.read_mtx('g_matrix.mtx.gz')\nprint(a.n_obs)\n"
    assert (
        codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", read], tables))
        == "mtx_rows_are_genes_not_cells"
    )
    transposed = read.replace("mtx.gz')", "mtx.gz').T")
    assert codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", transposed], tables)) == "ok"
    ten_x = "import scanpy as sc\na = sc.read_10x_mtx('.')\nprint(a.n_obs, 'matrix.mtx.gz')\n"
    assert codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", ten_x], tables)) in (None, "ok")
    plain = MTX.format(f="matrix.mtx", body="tot = m.sum(axis=1)")
    assert (
        codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", plain], tables))
        == "mtx_rows_are_genes_not_cells"
    )


def test_mtx_orientation_message_and_declarations(tables):
    code = MTX.format(f="g_matrix.mtx.gz", body="tot = np.asarray(m.sum(axis=1)).ravel()")
    result = verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", code], tables)
    assert "one row per feature" in result.message and "45 x 30" in result.message
    assert "45 features, 30 barcodes" in result.message and "read_10x_mtx" in " ".join(result.fix)
    wrong = verdict(
        ENGINE, "sc.mtx_orientation", PY, ["-c", code], tables, {"matrix_orientation": CELLS}
    )
    assert codes(wrong) == "declared_orientation_contradicts_matrix"
    cells = MTX.format(f="c_matrix.mtx.gz", body="tot = np.asarray(m.sum(axis=1)).ravel()")
    assert (
        codes(
            verdict(
                ENGINE,
                "sc.mtx_orientation",
                PY,
                ["-c", cells],
                tables,
                {"matrix_orientation": CELLS},
            )
        )
        == "ok"
    )
    assert (
        codes(
            verdict(
                ENGINE,
                "sc.mtx_orientation",
                PY,
                ["-c", cells],
                tables,
                {"matrix_orientation": GENES},
            )
        )
        == "declared_orientation_contradicts_matrix"
    )


def test_script_files_and_unrelated_commands(tables):
    (tables / "tot.py").write_text(MTX.format(f="g_matrix.mtx.gz", body="tot = m.sum(axis=1)"))
    assert (
        codes(verdict(ENGINE, "sc.mtx_orientation", PY, ["tot.py"], tables))
        == "mtx_rows_are_genes_not_cells"
    )
    assert verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", "print(1)"], tables) is None
    assert verdict(ENGINE, "sc.csv_orientation", PY, ["-c", "print(1)"], tables) is None
    assert verdict(ENGINE, "sc.mtx_orientation", PY, ["-c", "open('nofile.mtx')"], tables) is None


# ---- sc.loom_orientation -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def looms(tmp_path_factory):
    import h5py

    work = tmp_path_factory.mktemp("sc-loom")
    rng = np.random.default_rng(2)
    text = h5py.string_dtype("utf-8")

    def write(name, matrix, genes, cells):
        with h5py.File(work / name, "w") as f:
            f.create_dataset("matrix", data=matrix.astype("float32"))
            f.create_group("row_attrs").create_dataset(
                "Gene", data=np.array(genes, dtype=object), dtype=text
            )
            f.create_group("col_attrs").create_dataset(
                "CellID", data=np.array(cells, dtype=object), dtype=text
            )

    m = rng.poisson(2.0, size=(45, 30))
    write("cells.loom", m, [f"G{i}" for i in range(45)], barcodes(30))
    write(
        "square.loom", rng.poisson(2.0, size=(30, 30)), [f"G{i}" for i in range(30)], barcodes(30)
    )
    write("odd.loom", m, [f"G{i}" for i in range(44)], barcodes(30))  # attributes do not fit
    (work / "plain.loom").write_bytes(b"not hdf5")
    return work


LOOM = (
    "import json, numpy as np, loompy\n"
    "with loompy.connect('{f}') as ds:\n    m = ds[:, :]\n{body}\n"
)


@pytest.mark.parametrize(
    ("f", "orientation"),
    [("cells.loom", GENES), ("square.loom", None), ("odd.loom", None), ("plain.loom", None)],
)
def test_loom_orientation(looms, f, orientation):
    got, detail = so.loom_orientation(looms / f)
    assert got == orientation, detail


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("tot = m.sum(axis=1)", "loom_rows_are_genes_not_cells"),
        ("tot = m.sum(1)", "loom_rows_are_genes_not_cells"),
        (
            "tot = [r.sum() for _, r in enumerate(m)]\nn = m.mean(axis=1)",
            "loom_rows_are_genes_not_cells",
        ),
        ("tot = m.sum(axis=0)", "ok"),
        ("tot = m.T.sum(axis=1)", "ok"),
        ("print(ds.shape)", "ok"),
    ],
)
def test_loom_rows_are_genes(looms, body, expected):
    code = LOOM.format(f="cells.loom", body=body)
    assert codes(verdict(ENGINE, "sc.loom_orientation", PY, ["-c", code], looms)) == expected


def test_loom_scanpy_and_h5py_spellings_declarations_and_scripts(looms):
    read = "import scanpy as sc\na = sc.read_loom('cells.loom')\ntot = a.X.sum(axis=1)\n"
    assert codes(verdict(ENGINE, "sc.loom_orientation", PY, ["-c", read], looms)) == "ok"
    h5 = "import h5py\nm = h5py.File('cells.loom')['matrix'][:]\ntot = m.sum(axis=1)\n"
    result = verdict(ENGINE, "sc.loom_orientation", PY, ["-c", h5], looms)
    assert codes(result) == "loom_rows_are_genes_not_cells"
    assert "45 x 30" in result.message and "scanpy.read_loom" in " ".join(result.fix)
    wrong = verdict(
        ENGINE, "sc.loom_orientation", PY, ["-c", h5], looms, {"matrix_orientation": CELLS}
    )
    assert codes(wrong) == "declared_orientation_contradicts_loom"
    right = LOOM.format(f="cells.loom", body="tot = m.sum(axis=0)")
    assert (
        codes(
            verdict(
                ENGINE,
                "sc.loom_orientation",
                PY,
                ["-c", right],
                looms,
                {"matrix_orientation": GENES},
            )
        )
        == "ok"
    )
    (looms / "tot.py").write_text(LOOM.format(f="cells.loom", body="tot = m.sum(axis=1)"))
    assert (
        codes(verdict(ENGINE, "sc.loom_orientation", PY, ["tot.py"], looms))
        == "loom_rows_are_genes_not_cells"
    )
    assert verdict(ENGINE, "sc.loom_orientation", PY, ["-c", "print(1)"], looms) is None
