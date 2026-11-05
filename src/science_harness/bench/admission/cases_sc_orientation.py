"""Admission cases for the orientation rules of the sc pack (implicit tier, both twins).

  sc.mtx_orientation    F107  matrix.mtx.gz with one row per barcode (clean) or per feature
                              (faulted, the Cell Ranger layout)
  sc.csv_orientation    F107  counts.csv written from the same matrix: one row per cell (clean) or
                              per gene (faulted)
  sc.loom_orientation   F107  cells.loom written from the same matrix: one row per gene in BOTH
                              twins (the loom format), so both twins are risky

The csv and the loom are extra files written into the F107 workspaces after the build (the task
and its truth are F107's: how many cells have more than U UMIs in total), so the same answer
labels every template. Labels come from running each template (python-sc code, plus awk for the
csv) in the workspace and comparing the printed integer with `task.truth`. Templates that always
transpose, or always sum over axis 0, are right on the faulted twin and wrong on the clean one: the
rule passes them (it only reports the risky state of the data), so on the clean twin they are
honest misses.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_sc_common import Template, make_cases
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import f107_mtx_orientation

CELLS, GENES = {"matrix_orientation": "cells_by_genes"}, {"matrix_orientation": "genes_by_cells"}


def read_workspace(work: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """(cells x genes counts, barcodes, gene symbols) of an F107 workspace, whatever its layout."""
    with gzip.open(work / "matrix.mtx.gz", "rt") as handle:
        lines = [ln for ln in handle if not ln.startswith("%")]
    rows, cols, _ = (int(x) for x in lines[0].split())
    matrix = np.zeros((rows, cols), dtype=np.int64)
    for ln in lines[1:]:
        r, c, v = ln.split()
        matrix[int(r) - 1, int(c) - 1] = int(v)
    with gzip.open(work / "barcodes.tsv.gz", "rt") as handle:
        barcodes = handle.read().split()
    with gzip.open(work / "features.tsv.gz", "rt") as handle:
        genes = [ln.split("\t")[1] for ln in handle.read().splitlines()]
    cells_by_genes = matrix if rows == len(barcodes) else matrix.T
    return cells_by_genes, barcodes, genes


def write_loom(path: Path, matrix: np.ndarray, genes: list[str], cells: list[str]) -> None:
    """A minimal loom 3 file (genes x cells), byte-stable, readable by loompy and scanpy."""
    import h5py

    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "w", libver="earliest") as f:
        attrs = f.create_group("attrs")
        attrs.create_dataset("LOOM_SPEC_VERSION", data="3.0.0", dtype=text)
        attrs.create_dataset("CreationDate", data="20240101T000000.000000Z", dtype=text)
        f.create_dataset(
            "matrix", data=matrix.astype("float32"), chunks=(32, 32), compression="gzip"
        )
        for group in ("layers", "row_graphs", "col_graphs"):
            f.create_group(group)
        f.create_group("row_attrs").create_dataset(
            "Gene", data=np.array(genes, dtype=object), dtype=text
        )
        f.create_group("col_attrs").create_dataset(
            "CellID", data=np.array(cells, dtype=object), dtype=text
        )


def export_tables(work: Path, task: Task, twin: str) -> None:
    """Write counts.csv (cells in rows only on the clean twin) and cells.loom (genes in rows)."""
    cells_by_genes, barcodes, genes = read_workspace(work)
    if twin == "clean":
        lines = ["barcode," + ",".join(genes)]
        lines += [
            f"{b}," + ",".join(map(str, row))
            for b, row in zip(barcodes, cells_by_genes, strict=True)
        ]
    else:
        lines = ["gene," + ",".join(barcodes)]
        lines += [
            f"{g}," + ",".join(map(str, col))
            for g, col in zip(genes, cells_by_genes.T, strict=True)
        ]
    (work / "counts.csv").write_text("\n".join(lines) + "\n")
    write_loom(work / "cells.loom", cells_by_genes.T, genes, barcodes)


# -- sc.csv_orientation --------------------------------------------------------------------------

_CSV = """import json, re, numpy as np, pandas as pd, scanpy as sc
df = pd.read_csv("counts.csv", index_col=0)
{body}
"""


def csv_templates(task: Task, twin: str, work: Path) -> list[Template]:
    u = f107_mtx_orientation._u(task)
    true = CELLS if twin == "clean" else GENES

    def code(body: str) -> str:
        return _CSV.format(body=body)

    rows = code(f"print(json.dumps(int((df.sum(axis=1) > {u}).sum())))")
    return [
        Template("pandas_axis1", rows, wrong_decl=(true, CELLS, GENES)),
        Template("pandas_axis1_script", rows, form="script", wrong_decl=(true,)),
        Template("pandas_axis1_heredoc", rows, form="heredoc", wrong_decl=(true,)),
        Template(
            "pandas_sum1",
            code(f"print(json.dumps(int((df.sum(1) > {u}).sum())))"),
            wrong_decl=(true,),
        ),
        Template(
            "pandas_iterrows",
            code(
                f"n = sum(1 for _, r in df.iterrows() if r.sum() > {u})\nprint(json.dumps(int(n)))"
            ),
            wrong_decl=(true,),
        ),
        Template(
            "scanpy_read_csv",
            "import json, scanpy as sc\n"
            'a = sc.read_csv("counts.csv", first_column_names=True)\n'
            f"print(json.dumps(int((a.X.sum(axis=1) > {u}).sum())))\n",
            wrong_decl=(true,),
        ),
        Template(
            "awk_per_row",
            "",
            tool="awk",
            argv=(
                "-F,",
                f"NR>1{{s=0; for(i=2;i<=NF;i++) s+=$i; if(s>{u}) c++}} END{{print c+0}}",
                "counts.csv",
            ),
            wrong_decl=(true,),
        ),
        Template(
            "awk_per_column",
            "",
            tool="awk",
            argv=(
                "-F,",
                f"NR>1{{for(i=2;i<=NF;i++) s[i]+=$i}} END{{for(i in s) if(s[i]>{u}) c++; "
                "print c+0}",
                "counts.csv",
            ),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "transpose_then_rows",
            code(f"df = df.T\nprint(json.dumps(int((df.sum(axis=1) > {u}).sum())))"),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "axis0",
            code(f"print(json.dumps(int((df.sum(axis=0) > {u}).sum())))"),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "barcode_check",
            code(
                "if re.match(r'^[ACGT]{16}(-\\d+)?$', str(df.columns[0])):\n"
                "    df = df.T\n"
                f"print(json.dumps(int((df.sum(axis=1) > {u}).sum())))"
            ),
            decl_true=true,
            evident=True,
        ),
        Template(
            "inspect",
            "import pandas as pd\n"
            'df = pd.read_csv("counts.csv", index_col=0)\n'
            "print(df.shape, list(df.index[:3]), list(df.columns[:3]))\n",
            inspection=True,
            decl_true=true,
        ),
    ]


@case_source("sc.csv_orientation")
def cases_csv(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases(
        "F107",
        seed,
        scratch,
        tools,
        csv_templates,
        lambda twin: CELLS if twin == "clean" else GENES,
        after_build=export_tables,
    )


# -- sc.mtx_orientation --------------------------------------------------------------------------

_MTX = """import gzip, json, numpy as np, scipy.io, scipy.sparse as sp, scanpy as sc
{load}
{body}
"""


def mtx_templates(task: Task, twin: str, work: Path) -> list[Template]:
    u = f107_mtx_orientation._u(task)
    true = CELLS if twin == "clean" else GENES
    mm = 'm = scipy.io.mmread("matrix.mtx.gz").tocsr()'

    def code(load: str, body: str) -> str:
        return _MTX.format(load=load, body=body)

    count_rows = (
        f"tot = np.asarray(m.sum(axis=1)).ravel()\nprint(json.dumps(int((tot > {u}).sum())))"
    )
    count_cols = (
        f"tot = np.asarray(m.sum(axis=0)).ravel()\nprint(json.dumps(int((tot > {u}).sum())))"
    )
    count_obs = count_rows.replace("m.sum", "a.X.sum")
    rows = code(mm, count_rows)
    return [
        Template("mmread_rows", rows, wrong_decl=(true, CELLS, GENES)),
        Template("mmread_rows_script", rows, form="script", wrong_decl=(true,)),
        Template("mmread_rows_heredoc", rows, form="heredoc", wrong_decl=(true,)),
        Template(
            "mmread_sum1",
            code(
                mm,
                f"tot = np.asarray(m.sum(1)).ravel()\nprint(json.dumps(int((tot > {u}).sum())))",
            ),
            wrong_decl=(true,),
        ),
        Template(
            "scanpy_read_mtx",
            code('a = sc.read_mtx("matrix.mtx.gz")', count_obs),
            wrong_decl=(true,),
        ),
        Template("mmread_axis0", code(mm, count_cols), decl_true=GENES, evident=True),
        Template(
            "mmread_transpose",
            code('m = scipy.io.mmread("matrix.mtx.gz").T.tocsr()', count_rows),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "shape_check",
            code(
                mm + "\nn_barcodes = sum(1 for _ in gzip.open('barcodes.tsv.gz', 'rt'))\n"
                "if m.shape[0] != n_barcodes:\n    m = m.T.tocsr()",
                count_rows,
            ),
            decl_true=true,
            evident=True,
        ),
        Template(
            "read_10x_mtx",
            code("a = sc.read_10x_mtx('.')", count_obs),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "inspect",
            "import scipy.io\nm = scipy.io.mmread('matrix.mtx.gz')\nprint(m.shape)\n",
            inspection=True,
            decl_true=true,
        ),
    ]


@case_source("sc.mtx_orientation")
def cases_mtx(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases(
        "F107",
        seed,
        scratch,
        tools,
        mtx_templates,
        lambda twin: CELLS if twin == "clean" else GENES,
    )


# -- sc.loom_orientation -------------------------------------------------------------------------

_LOOM = """import json, numpy as np, loompy, scanpy as sc
{body}
"""
_LOOMPY = "with loompy.connect('cells.loom') as ds:\n    m = ds[:, :]\n"


def loom_templates(task: Task, twin: str, work: Path) -> list[Template]:
    u = f107_mtx_orientation._u(task)

    def code(body: str) -> str:
        return _LOOM.format(body=body)

    def total(axis: str) -> str:
        return f"tot = m.sum({axis})\nprint(json.dumps(int((tot > {u}).sum())))"

    rows = code(_LOOMPY + total("axis=1"))
    return [
        Template("loompy_axis1", rows, wrong_decl=(GENES, CELLS)),
        Template("loompy_axis1_script", rows, form="script", wrong_decl=(GENES,)),
        Template("loompy_axis1_heredoc", rows, form="heredoc", wrong_decl=(GENES,)),
        Template("loompy_sum1", code(_LOOMPY + total("1")), wrong_decl=(GENES,)),
        Template(
            "h5py_matrix_rows",
            "import json, h5py\n"
            "with h5py.File('cells.loom') as f:\n    m = f['matrix'][:]\n"
            f"tot = m.sum(axis=1)\nprint(json.dumps(int((tot > {u}).sum())))\n",
            wrong_decl=(GENES,),
        ),
        Template(
            "loompy_rows_loop",
            code(
                "with loompy.connect('cells.loom') as ds:\n"
                f"    n = sum(1 for _, _, v in ds.scan(axis=0) if v.sum() > {u})\n"
                "print(json.dumps(int(n)))"
            ),
            wrong_decl=(GENES,),
        ),
        Template("loompy_axis0", code(_LOOMPY + total("axis=0")), decl_true=GENES, evident=True),
        Template(
            "loompy_transpose",
            code(_LOOMPY + "m = m.T\n" + total("axis=1")),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "scanpy_read_loom",
            code(
                "a = sc.read_loom('cells.loom', sparse=False)\n"
                f"tot = a.X.sum(axis=1)\nprint(json.dumps(int((tot > {u}).sum())))"
            ),
            decl_true=GENES,
            evident=True,
        ),
        Template(
            "inspect",
            "import loompy\nwith loompy.connect('cells.loom') as ds:\n    print(ds.shape)\n",
            inspection=True,
            decl_true=GENES,
        ),
    ]


@case_source("sc.loom_orientation")
def cases_loom(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases(
        "F107",
        seed,
        scratch,
        tools,
        loom_templates,
        lambda twin: GENES,
        after_build=export_tables,
        risky_twins=("clean", "faulted"),
    )
