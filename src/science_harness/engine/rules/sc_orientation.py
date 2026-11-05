"""Rules sc.csv_orientation, sc.mtx_orientation and sc.loom_orientation (family F107 of the sc pack;
the admission cases of the csv and loom rules reuse its workspaces).

Slot orientation: a count table has cells in rows (what AnnData, scanpy's `sc.read_csv` and most
habits assume) or cells in columns (GEO supplementary tables; every 10x `matrix.mtx`). Nothing in
the file says which, and a per-row statistic is valid either way: it just describes the genes
instead of the cells.

Verified here (scanpy 1.11.5, scipy 1.15.2, pandas 2.3.3, loompy 3.0.8):
  mtx   `scipy.io.mmread` returns the matrix as written (shape (8, 12) for 8 genes x 12 cells);
        `sc.read_mtx` makes an AnnData of 8 observations with the obs names '0'..'7' (the genes);
        only `sc.read_10x_mtx` transposes. Row sums of such a matrix are gene totals; a threshold
        count over them is a number, not an error.
  csv   `pd.read_csv(index_col=0)` on a genes x cells table: `(df > 0).sum(axis=1)` is the number
        of cells expressing each gene.
  loom  loompy's `ds.shape` is (genes, cells) and `ds[:, :]` is genes x cells (scanpy's
        `read_loom` transposes); `/matrix` of the file is genes x cells by the loom spec.

What the rules read (stdlib only, plain or gzip text, bounded by Ctx.max_bytes):
  csv   the header and up to 60 rows of every `*.csv|tsv|txt` the command names. Cell barcodes are
        10x style (ACGT, 10 or more bases, optional -N suffix). The orientation is
        genes_by_cells when at least 80% of the header names are barcodes and under 20% of the row
        names are; cells_by_genes in the mirrored case; anything else is not judged.
  loom  `/matrix` against `/row_attrs/*` and `/col_attrs/*`: a loom file is genes x cells whenever
        its rows match the row attributes and its columns the column attributes (always, by spec).
  mtx   the dimension line of `*.mtx[.gz]` against the number of lines of the barcode and feature
        files next to it (`barcodes.tsv[.gz]`, `features.tsv[.gz]` or `genes.tsv[.gz]`, same
        prefix). rows == features and columns == barcodes (and the two counts differ) is
        genes_by_cells; the opposite is cells_by_genes.

A rule fires only when the data are genes_by_cells and the code counts per row without handling
it. "Counts per row": axis=1 / sum(1), iterrows, a csv reader, an awk per-line loop (NF), or
`sc.read_csv` / `sc.read_mtx` (they put the rows on the observations axis). "Handles": `.T`,
transpose, axis=0, `read_10x_mtx`, a column accumulator in awk (`c[i]++`), a comparison of
the shape with the barcode file. The test is a regular expression over the command text and the
scripts it runs; it does not read the arithmetic.

Declaration `matrix_orientation` (cells_by_genes | genes_by_cells): what the file holds. A
declaration that contradicts the file fails; a correct one passes when the code handles the
orientation.

Not seen: a table opened through a variable name, a transposed matrix written by the same script,
loom and h5 files, and tables with a single header token (no orientation can be read).
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.sc_common import code_text, data_files, strip_comments

TABLE_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:csv|tsv|txt)(?:\.gz)?(?![\w])")
MTX_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.mtx(?:\.gz)?(?![\w])")
BARCODE = re.compile(r"^[ACGT]{10,}(?:[-_]\d+)?$")
ROWS_SAMPLED = 60
MIN_SHARE = 0.8
MAX_OTHER = 0.2

ROW_WISE = re.compile(
    r"\baxis\s*=\s*1\b|\.(?:sum|mean|max|min|any|all|count_nonzero|getnnz|astype)\(\s*1\s*\)|"
    r"\biterrows\b|\bitertuples\b|\bcsv\.(?:reader|DictReader)\b|\bNF\b|\bsc\.read_csv\b|"
    r"\bread_mtx\s*\(|\bscanpy\.read_csv\b|\bn_obs\b"
)
HANDLED = re.compile(
    r"\.T\b|\.transpose\s*\(|\btranspose\b|\baxis\s*=\s*0\b|\.(?:sum|mean|max|min|any|all)\(\s*0\s*\)|"
    r"\bread_10x_mtx\b|\w\[\s*[a-z]\s*\]\s*(?:\+\+|\+=)|"
    r"\bn_barcodes\b|\bbarcodes\.tsv|\bread_loom\b"
)


def _tokens(line: str, sep: str) -> list[str]:
    return [t.strip().strip("\"'") for t in line.split(sep)]


def _share(tokens: list[str]) -> float:
    return sum(bool(BARCODE.match(t)) for t in tokens) / len(tokens) if tokens else 0.0


def table_orientation(ctx: Ctx, path: Path) -> tuple[str | None, str]:
    """(orientation, detail) of a delimited count table, or (None, why)."""
    name = path.name.lower().removesuffix(".gz")
    lines = [ln for ln in ctx.lines(path, limit=ROWS_SAMPLED + 1) if ln.strip()]
    if len(lines) < 4:
        return None, "too short"
    sep = "," if name.endswith(".csv") else "\t"
    if lines[0].count(sep) < 3:
        return None, "no delimiter"
    header = _tokens(lines[0], sep)
    rows = [_tokens(ln, sep) for ln in lines[1:]]
    names = [r[0] for r in rows]
    width = len(rows[0])
    cells = header[1:] if len(header) == width else header
    body = [t for r in rows[:20] for t in r[1:]]
    numeric = sum(bool(re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", t)) for t in body)
    if not body or numeric / len(body) < 0.9:
        return None, "not a numeric table"
    head_share, name_share = _share(cells), _share(names)
    if head_share >= MIN_SHARE and name_share < MAX_OTHER:
        return "genes_by_cells", f"header {cells[0]!r}.. are barcodes, row names {names[0]!r}.."
    if name_share >= MIN_SHARE and head_share < MAX_OTHER:
        return "cells_by_genes", f"row names {names[0]!r}.. are barcodes"
    return None, "orientation not readable"


@checker("sc_csv_orientation")
def sc_csv_orientation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    tables = data_files(call, match, ctx, TABLE_NAME, (".csv", ".tsv", ".txt", ".gz"))
    if not tables:
        return passed()
    text = strip_comments(code_text(call, ctx))
    for path in tables:
        try:
            orientation, detail = table_orientation(ctx, path)
        except (OSError, EOFError, UnicodeError):
            continue
        if orientation is None:
            continue
        if declared is not None and declared != orientation:
            return CheckResult(
                "fail",
                "declared_orientation_contradicts_table",
                f"you declared {key}={declared} but {path.name} is {orientation} ({detail}).",
                expected=orientation,
                got=declared,
                fix=(f"declare {key}={orientation} and count along the right axis",),
            )
        if orientation == "genes_by_cells" and ROW_WISE.search(text) and not HANDLED.search(text):
            return CheckResult(
                "fail",
                "rows_are_genes_not_cells",
                f"{path.name} has one row per gene and one column per cell ({detail}), but this "
                "code works along its rows as if they were cells (axis=1, a row loop, NF, "
                "sc.read_csv): it describes the genes, and the numbers look plausible.",
                fix=(
                    "transpose first (df = pd.read_csv(path, index_col=0).T) or aggregate along "
                    f"axis=0, and declare --declare {key}=genes_by_cells",
                ),
            )
    return passed()


# -- sc.mtx_orientation --------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:  # type: ignore[operator]
        return sum(1 for line in handle if line.strip())


def _sibling(directory: Path, prefix: str, stems: tuple[str, ...]) -> Path | None:
    for stem in stems:
        for suffix in (".tsv.gz", ".tsv"):
            candidate = directory / f"{prefix}{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def mtx_orientation(path: Path) -> tuple[str | None, str]:
    """(orientation, detail) of a MatrixMarket file from its dimension line and its siblings."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:  # type: ignore[operator]
        dims = None
        for line in handle:
            if line.startswith("%") or not line.strip():
                continue
            dims = line.split()
            break
    if dims is None or len(dims) < 2 or not all(d.isdigit() for d in dims[:2]):
        return None, "no dimension line"
    rows, cols = int(dims[0]), int(dims[1])
    prefix = re.sub(r"matrix\.mtx(\.gz)?$", "", path.name)
    barcodes = _sibling(path.parent, prefix, ("barcodes",))
    features = _sibling(path.parent, prefix, ("features", "genes"))
    if barcodes is None or features is None:
        return None, "no barcode / feature files next to the matrix"
    n_bc, n_ft = _count_lines(barcodes), _count_lines(features)
    detail = f"dimension line {rows} x {cols}; {n_ft} features, {n_bc} barcodes"
    if n_bc == n_ft:
        return None, "as many barcodes as features: " + detail
    if rows == n_ft and cols == n_bc:
        return "genes_by_cells", detail
    if rows == n_bc and cols == n_ft:
        return "cells_by_genes", detail
    return None, "dimensions fit neither orientation: " + detail


@checker("sc_mtx_orientation")
def sc_mtx_orientation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    files = data_files(call, match, ctx, MTX_NAME, (".mtx", ".mtx.gz"))
    if not files:
        return passed()
    text = strip_comments(code_text(call, ctx))
    for path in files:
        try:
            orientation, detail = mtx_orientation(path)
        except (OSError, EOFError, UnicodeError):
            continue
        if orientation is None:
            continue
        if declared is not None and declared != orientation:
            return CheckResult(
                "fail",
                "declared_orientation_contradicts_matrix",
                f"you declared {key}={declared} but {path.name} is {orientation} ({detail}).",
                expected=orientation,
                got=declared,
                fix=(f"declare {key}={orientation} and sum along the right axis",),
            )
        if orientation == "genes_by_cells" and ROW_WISE.search(text) and not HANDLED.search(text):
            return CheckResult(
                "fail",
                "mtx_rows_are_genes_not_cells",
                f"{path.name} has one row per feature and one column per barcode ({detail}), but "
                "this code works along its rows as if they were cells (axis=1, read_mtx without "
                "a transpose): the per-row numbers are gene totals, and a threshold count over "
                "them is a plausible integer.",
                fix=(
                    "use sc.read_10x_mtx(dir) or transpose (mmread(path).T), or sum along axis=0, "
                    f"and declare --declare {key}=genes_by_cells",
                ),
            )
    return passed()


# -- sc.loom_orientation -------------------------------------------------------------------------

LOOM_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.loom(?![\w])")


def loom_orientation(path: Path) -> tuple[str | None, str]:
    """(orientation, detail) of a loom file: always genes_by_cells when its attributes fit."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            matrix = f["matrix"]
            rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
            row_attrs = [len(f["row_attrs"][k]) for k in f["row_attrs"]]
            col_attrs = [len(f["col_attrs"][k]) for k in f["col_attrs"]]
    except (OSError, KeyError, ValueError, IndexError, ImportError):
        return None, "not a loom file"
    if rows == cols:
        return None, "as many rows as columns"
    if row_attrs and col_attrs and row_attrs[0] == rows and col_attrs[0] == cols:
        return (
            "genes_by_cells",
            f"/matrix is {rows} x {cols}, {rows} row attributes, {cols} column attributes",
        )
    return None, "attributes do not fit /matrix"


@checker("sc_loom_orientation")
def sc_loom_orientation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    files = data_files(call, match, ctx, LOOM_NAME, (".loom",))
    if not files:
        return passed()
    text = strip_comments(code_text(call, ctx))
    for path in files:
        orientation, detail = loom_orientation(path)
        if orientation is None:
            continue
        if declared is not None and declared != orientation:
            return CheckResult(
                "fail",
                "declared_orientation_contradicts_loom",
                f"you declared {key}={declared} but {path.name} is {orientation} ({detail}): a "
                "loom file always has one row per gene.",
                expected=orientation,
                got=declared,
                fix=(f"declare {key}={orientation} and sum along the right axis",),
            )
        if ROW_WISE.search(text) and not HANDLED.search(text):
            return CheckResult(
                "fail",
                "loom_rows_are_genes_not_cells",
                f"{path.name} is a loom file: rows are genes and columns are cells ({detail}), but "
                "this code works along its rows as if they were cells (axis=1, a row loop): the "
                "per-row numbers describe the genes and look plausible.",
                fix=(
                    "use scanpy.read_loom (it transposes), transpose the matrix, or aggregate "
                    f"along axis=0, and declare --declare {key}=genes_by_cells",
                ),
            )
    return passed()
