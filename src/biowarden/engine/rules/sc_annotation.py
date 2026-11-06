"""Rules sc.obs_assign_position, sc.control_label, sc.gene_vocabulary, sc.concat_join,
sc.mito_prefix_case (families F105, F108, F109, F110, F111 of the sc pack).

Five silent traps of joining things to an AnnData by name. All fact rules over .h5ad files read
with h5py, plus the other file the join needs (an annotation table, a vocabulary) when it is
named in the same command. Each fires only when the files are in the risky state AND the code
does the risky join; the code test is a regular expression over the command text and the scripts
it runs (comments are not removed, so a commented-out line can fire a rule).

Verified here (anndata 0.11.4, scanpy 1.11.5, pandas 2.3.3, python-sc):
  obs by position   with the barcodes of a table in reversed order: `obs['ct'] = ann['celltype']`
                    aligns on the index (right); `obs['ct'] = ann['celltype'].values` pastes by
                    position (every cell gets another cell's label); `adata.obs = ann` replaces the
                    obs table and renames the cells while the rows of X stay where they are (row 0
                    sums to 83.0 before and after; obs_names now start with the last barcode). No
                    warning in any of them. A table read without index_col gives NaN labels.
  control label     `a[a.obs['perturbation'] == 'ctrl']` on a file whose controls are
                    'non-targeting' is an empty AnnData (0 x n_vars) without a warning (a mean
                    over it raises ZeroDivisionError for sparse X). Pooling two files first keeps
                    the controls of the file that uses the literal spelling only.
  gene vocabulary   `set(adata.var_names) & set(vocab)` with Ensembl ids against symbol tokens is
                    empty without a warning; the symbol column in var gives the overlap.
  concat            `anndata.concat([a, b])` is join='inner': two batches of 6 genes with 4 in
                    common give a 4-gene object, 22 x 4 where join='outer' gives 22 x 8, no
                    warning (a UserWarning appears only for duplicated obs_names).
  mito prefix       `var_names.str.startswith('MT-')` is case-sensitive; mouse genes are
                    'mt-Nd1'; `calculate_qc_metrics(qc_vars=['mt'])` then reports pct_counts_mt
                    of exactly 0.0 for every cell.

Declarations (all optional; a declaration is checked against the files, never the arithmetic):
  obs_join       by_index | by_position   how the table is joined to the cells
  control_label  ctrl | control | non-targeting | NT | scramble | multiple   (multiple: the files
                 spell the control differently and the code covers every spelling)
  gene_namespace symbol | ensembl_id      the namespace of the names matched against the vocabulary
  concat_genes   intersection | union     the gene set the merged object should have
  mito_prefix    MT- | mt- | both         the prefix of the mitochondrial genes in the data
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.sc_common import (
    H5ad,
    code_text,
    column_values,
    data_files,
    h5ad_files,
    read_h5ad,
    require_h5py,
    strip_comments,
)

TABLE_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:csv|tsv|txt)(?:\.gz)?(?![\w])")
VOCAB_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:json|txt|csv|tsv)(?:\.gz)?(?![\w])")
MAX_TABLE = 20_000


def _infos(call: Call, match: Match, ctx: Ctx) -> list[H5ad]:
    return [info for p in h5ad_files(call, match, ctx) if (info := read_h5ad(p)) is not None]


# -- sc.obs_assign_position ----------------------------------------------------------------------

ASSIGN = re.compile(r"\.obs(\[[^\]]*\])?\s*=(?!=)\s*([^\n;]*)")
POSITIONAL_RHS = re.compile(
    r"\.values\b|\.to_numpy\s*\(|\.tolist\s*\(|\bnp\.array\s*\(|\blist\s*\("
)
WHOLE_TABLE_RHS = re.compile(r"\s*(?!\w+\.obs\b)\w+\s*['\")]*\s*$|.*\bread_(?:csv|table)\b")
ALIGNED_RHS = re.compile(
    r"\.loc\[|\.reindex\s*\(|\.join\s*\(|\.merge\s*\(|\.map\s*\(|\.reindex_like"
)


def table_ids(ctx: Ctx, path: Path) -> list[str]:
    """First column of a delimited table without its header line (at most MAX_TABLE ids)."""
    name = path.name.lower().removesuffix(".gz")
    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    ids: list[str] = []
    for i, line in enumerate(ctx.lines(path, limit=MAX_TABLE + 1)):
        if i == 0 or not line.strip():
            continue
        ids.append(line.split(sep)[0].strip().strip("\"'"))
    return ids


def positional_paste(text: str) -> str | None:
    """The first statement that copies a table into adata.obs by position, or None."""
    for found in ASSIGN.finditer(text):
        column, rhs = found.group(1), found.group(2)
        if ALIGNED_RHS.search(rhs):
            continue
        if column is None and WHOLE_TABLE_RHS.match(rhs):
            return found.group(0).strip()
        if column is not None and POSITIONAL_RHS.search(rhs):
            return found.group(0).strip()
    return None


@checker("sc_obs_assign_position")
def sc_obs_assign_position(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not ASSIGN.search(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    tables = data_files(call, match, ctx, TABLE_NAME, (".csv", ".tsv", ".txt", ".gz"))
    pasted = positional_paste(text)
    for info in _infos(call, match, ctx):
        names = info.obs_names
        if not names:
            continue
        for table in tables:
            try:
                ids = table_ids(ctx, table)
            except (OSError, EOFError):
                continue
            if len(ids) != len(names) or set(ids) != set(names):
                continue  # not the table of this object (or a different cell set: pandas errors)
            same_order = ids == names
            if declared == "by_position" and not same_order:
                return CheckResult(
                    "fail",
                    "declared_position_but_orders_differ",
                    f"you declared {key}=by_position but the barcodes of {table.name} are in a "
                    f"different order than adata.obs_names of {info.path.name} (row 1: "
                    f"{ids[0]} against {names[0]}).",
                    expected="by_index",
                    got="by_position",
                    fix=(f"join on the barcode and declare {key}=by_index",),
                )
            if pasted is not None and not same_order:
                return CheckResult(
                    "fail",
                    "positional_join_of_reordered_table",
                    f"`{pasted[:80]}` copies {table.name} into the cells by position, but its "
                    f"barcodes are in a different order than adata.obs_names of {info.path.name} "
                    f"(first row {ids[0]}, first cell {names[0]}): every label lands on the wrong "
                    "cell (and `adata.obs = table` renames the cells without moving X).",
                    fix=(
                        "join on the barcode: ann = pd.read_csv(path, index_col=0); "
                        "adata.obs['celltype'] = ann.loc[adata.obs_names, 'celltype'].values, "
                        f"and declare --declare {key}=by_index",
                    ),
                )
            if declared == "by_index" and pasted is not None:
                return CheckResult(
                    "fail",
                    "declared_index_but_positional_code",
                    f"you declared {key}=by_index but `{pasted[:80]}` pastes by position.",
                    expected="by_index",
                    got="by_position",
                    fix=("align on the barcode with .loc[adata.obs_names] or a join",),
                )
    return passed()


# -- sc.control_label ----------------------------------------------------------------------------

CONTROL_LIKE = frozenset(
    "ctrl control non-targeting nontargeting non_targeting nt scramble scrambled wt wild-type "
    "untreated vehicle dmso mock negative_control neg_ctrl negctrl control_cells".split()
)
CONTROL_LITERAL = re.compile(r"""['"]([\w-]{2,20})['"]""")
LABEL_VALUES = ("ctrl", "control", "non-targeting", "NT", "scramble", "multiple")


def control_labels(path: Path) -> dict[str, set[str]]:
    """obs column -> control-like labels it holds, for the columns that hold any."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            out: dict[str, set[str]] = {}
            for col in f["obs"]:
                values = column_values(f["obs"], col)
                if values is None:
                    continue
                found = {v for v in set(values) if v.lower() in CONTROL_LIKE}
                if found:
                    out[col] = found
            return out
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        return {}


@checker("sc_control_label")
def sc_control_label(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    used = {
        m.group(1) for m in CONTROL_LITERAL.finditer(text) if m.group(1).lower() in CONTROL_LIKE
    }
    declared = call.declared.get(key)
    if (missing := require_h5py()) is not None:
        return missing
    per_file: dict[str, set[str]] = {}
    for path in h5ad_files(call, match, ctx):
        for labels in control_labels(path).values():
            per_file.setdefault(path.name, set()).update(labels)
    if not per_file:
        return passed()
    spellings = {label for labels in per_file.values() for label in labels}
    if declared and declared != "multiple":
        for name, labels in per_file.items():
            if declared not in labels:
                return CheckResult(
                    "fail",
                    "declared_label_missing_in_file",
                    f"you declared {key}={declared} but the control cells of {name} are labelled "
                    f"{', '.join(repr(x) for x in sorted(labels))}.",
                    expected=sorted(labels),
                    got=declared,
                    fix=(
                        f"cover every spelling ({', '.join(sorted(spellings))}) and declare "
                        f"--declare {key}=multiple",
                    ),
                )
    if not used:
        return passed()
    for name, labels in per_file.items():
        if not used & labels:
            where = "; ".join(
                f"{n}: {', '.join(repr(x) for x in sorted(v))}" for n, v in sorted(per_file.items())
            )
            return CheckResult(
                "fail",
                "control_literal_matches_nothing",
                f"the code selects controls with {', '.join(repr(x) for x in sorted(used))} but "
                f"{name} labels its control cells {', '.join(repr(x) for x in sorted(labels))} "
                f"(control labels by file: {where}); that file contributes no controls and an "
                "empty selection raises no error.",
                fix=(
                    f"match every spelling, for example isin({sorted(spellings)!r}), and run again "
                    f"with --declare {key}=multiple",
                ),
            )
    return passed()


# -- sc.gene_vocabulary --------------------------------------------------------------------------

ENSEMBL = re.compile(r"^ENS[A-Z]{0,6}G\d{6,}")
MIN_OVERLAP = 0.05
GOOD_OVERLAP = 0.5
VAR_COLUMN_USE = re.compile(
    r"\.var\s*\[|\.var\s*\.\s*(?:loc|iloc|columns|(?!index\b)[A-Za-z_]\w*)"
    r"|feature_name|gene_symbol|gene_name"
)
VAR_NAMES_USE = re.compile(r"\bvar_names\b|\.var\s*\.\s*index\b|\bvar\.index\b")


def vocabulary_tokens(ctx: Ctx, path: Path) -> set[str]:
    """Tokens of a vocabulary file: keys of a JSON object, items of a JSON list, or the first
    field of each line of a text / csv / tsv file."""
    name = path.name.lower().removesuffix(".gz")
    lines = list(ctx.lines(path, limit=MAX_TABLE))
    if name.endswith(".json"):
        try:
            data = json.loads("\n".join(lines))
        except ValueError:
            return set()
        if isinstance(data, dict):
            inner = data.get("vocab") if isinstance(data.get("vocab"), dict) else data
            return {str(k) for k in inner}
        if isinstance(data, list):
            return {str(x) for x in data}
        return set()
    return {re.split(r"[\t,]", ln)[0].strip().strip("\"'") for ln in lines if ln.strip()}


def share_ensembl(names: list[str]) -> float:
    return sum(bool(ENSEMBL.match(n)) for n in names) / len(names) if names else 0.0


def _string_columns(path: Path, info: H5ad) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        import h5py

        with h5py.File(path, "r") as f:
            for col in info.var_columns:
                values = column_values(f["var"], col)
                if values is not None:
                    out[col] = values
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        pass
    return out


@checker("sc_gene_vocabulary")
def sc_gene_vocabulary(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    declared = call.declared.get(key)
    vocabs = [
        p
        for p in data_files(call, match, ctx, VOCAB_NAME, (".json", ".txt", ".csv", ".tsv"))
        if not p.name.endswith(".h5ad")
    ]
    if not vocabs:
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    for info in _infos(call, match, ctx):
        names = info.var_names
        if not names:
            continue
        for vocab in vocabs:
            try:
                tokens = vocabulary_tokens(ctx, vocab)
            except (OSError, EOFError):
                continue
            if len(tokens) < 10:
                continue
            tok_ens = share_ensembl(sorted(tokens)) >= 0.5
            if declared is not None and (declared == "ensembl_id") != tok_ens:
                return CheckResult(
                    "fail",
                    "declared_namespace_contradicts_vocabulary",
                    f"you declared {key}={declared} but the tokens of {vocab.name} are "
                    f"{'Ensembl ids' if tok_ens else 'gene symbols'}.",
                    expected="ensembl_id" if tok_ens else "symbol",
                    got=declared,
                    fix=(f"declare {key}={'ensembl_id' if tok_ens else 'symbol'}",),
                )
            overlap = sum(n in tokens for n in names) / len(names)
            if overlap >= MIN_OVERLAP:
                continue
            best_col, best = None, 0.0
            for col, values in _string_columns(info.path, info).items():
                share = sum(v in tokens for v in values) / max(len(values), 1)
                if share > best:
                    best_col, best = col, share
            if best_col is None or best < GOOD_OVERLAP:
                continue
            if VAR_COLUMN_USE.search(text) or not VAR_NAMES_USE.search(text):
                continue
            ens = share_ensembl(names) >= 0.5
            fixed = "ensembl_id" if tok_ens else "symbol"
            return CheckResult(
                "fail",
                "var_names_not_in_vocabulary_namespace",
                f"var_names of {info.path.name} ({names[0]}, {names[1] if len(names) > 1 else ''},"
                f" ...) are {'Ensembl ids' if ens else 'another namespace'} and only "
                f"{overlap:.0%} of them are tokens of {vocab.name} ({sorted(tokens)[3]}, ...), "
                f"while var['{best_col}'] matches {best:.0%}: the intersection with var_names is "
                "empty or tiny and nothing warns.",
                fix=(
                    f"match var['{best_col}'] against the vocabulary (adata.var['{best_col}']"
                    f".isin(vocab)), and declare --declare {key}={fixed}",
                ),
            )
    return passed()


# -- sc.concat_join ------------------------------------------------------------------------------

CONCAT_METHOD = re.compile(r"(\b\w+)\.concat\s*\(")
NOT_ANNDATA = frozenset({"pd", "pandas", "np", "numpy", "dd", "dask", "xr", "xarray"})


def calls_anndata_concat(text: str) -> bool:
    if any(m.group(1) not in NOT_ANNDATA for m in CONCAT_METHOD.finditer(text)):
        return True
    if re.search(r"\.concatenate\s*\(", text):
        return True
    return bool(
        re.search(r"from\s+anndata\s+import[^\n]*\bconcat\b", text)
        and re.search(r"(?<![\w.])concat\s*\(", text)
    )


JOIN_ARG = re.compile(r"""\bjoin\s*=\s*['"](\w+)['"]""")


@checker("sc_concat_join")
def sc_concat_join(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not calls_anndata_concat(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    joins = set(JOIN_ARG.findall(text))
    outer = "outer" in joins
    infos = _infos(call, match, ctx)
    if len(infos) < 2:
        return passed()
    sets = [set(i.var_names) for i in infos]
    common_genes = set.intersection(*sets)
    union = set.union(*sets)
    lost = len(union) - len(common_genes)
    if lost == 0:
        return passed()
    sizes = ", ".join(f"{i.path.name}: {len(s)}" for i, s in zip(infos, sets, strict=True))
    if declared == "intersection" and outer:
        return CheckResult(
            "fail",
            "declared_intersection_but_outer",
            f"you declared {key}=intersection but the call passes join='outer'.",
            expected="intersection",
            got="union",
            fix=("drop join='outer', or declare concat_genes=union",),
        )
    if declared == "intersection":
        return passed()
    if outer:
        return passed()
    return CheckResult(
        "fail",
        "concat_inner_join_drops_genes",
        f"anndata.concat defaults to join='inner': the files have different genes ({sizes}), "
        f"{len(common_genes)} in common out of {len(union)}, so {lost} genes disappear from the "
        "merged object without a warning.",
        fix=(
            "pass join='outer' (and fill_value=0 for the missing genes) to keep every gene, and "
            f"declare --declare {key}=union",
            f"or declare {key}=intersection if only the shared genes are wanted",
        ),
    )


# -- sc.mito_prefix_case -------------------------------------------------------------------------

MT_UPPER = re.compile(
    r"""startswith\(\s*\(?\s*['"]MT-['"]|\^MT-|str\.(?:match|contains)\(\s*['"]\^?MT-"""
)
CASE_SAFE = re.compile(
    r"\.str\.(?:upper|lower)\(\)|\.upper\(\)\s*\.startswith|\.lower\(\)\s*\.startswith|re\.I\b|"
    r"IGNORECASE|\(\?i\)|\bcase\s*=\s*False|['\"]mt-['\"]|\^\[Mm\]\[Tt\]|\[Mm\]"
)


@checker("sc_mito_prefix_case")
def sc_mito_prefix_case(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    uses_upper = bool(MT_UPPER.search(text))
    safe = bool(CASE_SAFE.search(text))
    declared = call.declared.get(key)
    if not uses_upper and declared is None:
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    for info in _infos(call, match, ctx):
        names = info.var_names
        upper = sum(n.startswith("MT-") for n in names)
        lower = sum(n[:3].lower() == "mt-" and not n.startswith("MT-") for n in names)
        if upper == 0 and lower == 0:
            continue
        actual = "MT-" if upper >= lower else "mt-"
        if declared in ("MT-", "mt-") and declared != actual:
            return CheckResult(
                "fail",
                "declared_prefix_contradicts_data",
                f"you declared {key}={declared} but the mitochondrial genes of {info.path.name} "
                f"start with {actual!r} ({lower if actual == 'mt-' else upper} genes, for "
                f"example {next(n for n in names if n[:3].lower() == 'mt-')}).",
                expected=actual,
                got=declared,
                fix=(f"declare {key}={actual} and match that prefix",),
            )
        if uses_upper and not safe and upper == 0 and lower > 0:
            example = next(n for n in names if n[:3].lower() == "mt-")
            return CheckResult(
                "fail",
                "mito_prefix_case_mismatch",
                f"the code marks mitochondrial genes with the upper-case prefix 'MT-' but "
                f"{info.path.name} names them like {example!r} ({lower} genes, none upper case): "
                "no gene is flagged, so pct_counts_mt is 0 for every cell and a QC filter on it "
                "keeps all cells.",
                fix=(
                    "match case-insensitively: adata.var_names.str.upper().str.startswith('MT-'), "
                    f"and declare --declare {key}=mt-",
                ),
            )
        if declared == "both" and uses_upper and not safe:
            return CheckResult(
                "fail",
                "declared_both_but_one_prefix",
                f"you declared {key}=both but the code matches only the upper-case 'MT-'.",
                expected="both",
                got="MT-",
                fix=("fold the case before matching",),
            )
    return passed()
