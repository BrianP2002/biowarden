"""Rules sc.de_raw_default, sc.small_int_qc, sc.double_transform, sc.counts_needed,
sc.scaled_x_expression (families F101-F104 and F112 of the sc pack; slot matrix_state, F102
dtype_overflow).

The trap behind all five: an .h5ad is a valid object whatever its matrices hold, and scanpy does not
check what a step needs. The state of X (integer counts, log1p of counts per 10,000, scaled
z-scores) and of the other matrices (`.raw`, `layers`) decides whether the call is right.

Verified here (scanpy 1.11.5, anndata 0.11.4, python-sc, synthetic 100-200 cell objects):
  rank_genes_groups default      use_raw=None means "use .raw if present". With X = log1p(CP10K) and
                                 .raw.X = CP10K (not logged, non-integer) the call exits 0 with an
                                 empty stderr and returns logfoldchanges of 300 to 600 where
                                 use_raw=False returns 3.5 (the column takes expm1 of group means);
                                 scores and p-values are identical (ranks). With integer counts in
                                 .raw scanpy does print "It seems you use rank_genes_groups on the
                                 raw count data".
  calculate_qc_metrics           on a sparse (csr / csc) int8 or int16 matrix the percent-of-counts-
                                 in-top-N columns come out wrong for every cell above 32,767
                                 UMIs: -9.25 where the same counts as float32 give 9.23 (negative
                                 between 32,768 and 65,535, above 100 up to 98,303). total_counts is
                                 right, there is no warning; a dense int16 matrix and float32 or
                                 int32 are fine; uint16 raises (loud). percent_top=None avoids it.
  log1p / normalize_total twice  sc.pp.log1p warns "adata.X seems to be already log-transformed"
                                 only when adata.uns has a 'log1p' key; an .h5ad written by another
                                 tool (no key) is silently log-transformed a second time (max 6.1
                                 -> 1.96); normalize_total on log data runs without a word.
  filter_cells(min_counts=)      on log-normalised X it keeps 55 of 60 cells where the integer
                                 counts keep 30; calculate_qc_metrics reports total_counts of
                                 89..211 for cells of 31..489 UMIs. No warning in either.
  X read as expression           in a processed object X holds z-scores (negative, centred) and
                                 the log-normalised values live in .raw; `X > 0` is then true only
                                 for the cells above the gene's mean (family F112: 3 cells where
                                 34 express the gene), and a mean of X is near 0.

How the rules look at a command (fact rules, tier T1): they apply to every existing `*.h5ad`
named in the command or inside the `*.py` script it runs, for the interpreters python-sc, python3
and python. A rule fires only when the code does the risky operation (the call is found in the
code text, comments are not removed) and the file is in a risky state; everything else passes,
including `head`-style inspection (`print(a)`, `a.X.max()`) and objects with a counts X. A command
"handles" the state when it names the matrix explicitly (use_raw=, layer=, a layer or .raw lookup,
an expm1 before a log1p); the handled test is deliberately loose. The state of X is read from a
bounded sample of stored values (sc_common.classify).

Declarations. `x_state` (counts | log_normalized | normalized | scaled) states what X holds;
`de_input` (X | raw) states which matrix rank_genes_groups should read. A declaration that
contradicts the data fails; an undeclared risky call fails; a declared and consistent call
passes when the code does not repeat the mistake. The rules cannot read the arithmetic of a
script: a wrong script with a correct declaration is a miss.

Not seen: an object built in the same script (`AnnData(...)`), files named through variables or
globs, a read through `sc.read(path)` where the path is built at run time, and Seurat / loom inputs.
"""

from __future__ import annotations

import re

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.sc_common import (
    H5ad,
    code_text,
    h5ad_files,
    read_h5ad,
    require_h5py,
    strip_comments,
    unknown,
)

STATES = ("counts", "log_normalized", "normalized", "scaled")

RANK_CALL = re.compile(r"\brank_genes_groups\s*\(")
USE_RAW_EXPLICIT = re.compile(r"\buse_raw\s*=\s*(?:True|False)\b")
USE_RAW_FALSE = re.compile(r"\buse_raw\s*=\s*False\b")
LAYER_ARG = re.compile(r"\blayer\s*=\s*(?!None\b)['\"\w]")
RAW_DROPPED = re.compile(r"\.raw\s*=\s*None|\bdel\s+\w+\.raw\b|\.raw\.to_adata\s*\(|\.raw\s*\.\s*X")
LOG_OP = re.compile(r"\blog1p\s*\(")
NORM_OP = re.compile(r"\bnormalize_total\s*\(|\bnormalize_per_cell\s*\(")
BACK_TRANSFORM = re.compile(r"\bexpm1\b|\bexp\s*\(")
LOOKS_AT_STATE = re.compile(
    r"\ballclose\b|\bis_integer\b|\bnp\.round\b|\bissubdtype\b|\.max\s*\(\s*\)\s*[<>]"
)
COUNTS_OP = re.compile(
    r"\bmin_counts\s*=|\bmax_counts\s*=|\btotal_counts\b|\bn_counts\b|\.X\s*\.sum\s*\(|"
    r"\bcalculate_qc_metrics\s*\("
)
SOURCE_NAMED = re.compile(
    r"\blayers\s*\[|\blayer\s*=\s*(?!None\b)['\"\w]|\.raw\b|\buse_raw\s*=\s*True|\bX_counts\b"
)
# reads values of X (not its shape, dtype, min or max, which is how one finds the state)
READS_X = re.compile(
    r"\.X\b(?!\s*\.\s*(?:shape|dtype|ndim|nnz|size|min|max)\b)|\bto_df\s*\(|\bobs_df\s*\(|\bX\s*\["
)
RAW_NAMED = re.compile(
    r"\.raw\b|\buse_raw\s*=\s*True|\blayers\s*\[|\blayer\s*=\s*(?!None\b)['\"\w]"
)


def _files_or_none(call: Call, match: Match, ctx: Ctx) -> list[H5ad]:
    return [info for p in h5ad_files(call, match, ctx) if (info := read_h5ad(p)) is not None]


def _declared_contradiction(
    key: str, declared: str | None, info: H5ad, name: str
) -> CheckResult | None:
    """A declaration of the state of X that the data contradict."""
    if declared is None or info.x.state in ("ambiguous", "empty"):
        return None
    if declared == info.x.state:
        return None
    return CheckResult(
        "fail",
        "declared_state_contradicts_data",
        f"you declared {key}={declared} but X of {name} holds {info.x.says()}.",
        expected=info.x.state,
        got=declared,
        fix=(f"declare {key}={info.x.state} and write the step for that state",),
    )


# -- sc.de_raw_default ---------------------------------------------------------------------------


@checker("sc_de_raw_default")
def sc_de_raw_default(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not RANK_CALL.search(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    chosen = bool(
        USE_RAW_EXPLICIT.search(text) or LAYER_ARG.search(text) or RAW_DROPPED.search(text)
    )
    reads_raw_by_default = not chosen
    for info in _files_or_none(call, match, ctx):
        if info.raw is None:
            continue
        name = info.path.name
        differs = info.raw.state != info.x.state or info.raw.state in ("counts", "normalized")
        raw_not_log = info.raw.state in ("counts", "normalized") and info.x.state in (
            "log_normalized",
            "scaled",
        )
        if declared == "X" and reads_raw_by_default and differs:
            return CheckResult(
                "fail",
                "declared_x_but_default_reads_raw",
                f"you declared {key}=X but rank_genes_groups without use_raw reads .raw of "
                f"{name} ({info.raw.says()}), not X ({info.x.says()}).",
                expected="X",
                got="raw",
                fix=("add use_raw=False to the rank_genes_groups call",),
            )
        if declared == "raw" and USE_RAW_FALSE.search(text) and differs:
            return CheckResult(
                "fail",
                "declared_raw_but_use_raw_false",
                f"you declared {key}=raw but the call passes use_raw=False, which reads X of "
                f"{name} ({info.x.says()}), not .raw ({info.raw.says()}).",
                expected="raw",
                got="X",
                fix=("drop use_raw=False, or declare de_input=X",),
            )
        if declared is None and reads_raw_by_default and raw_not_log:
            return CheckResult(
                "fail",
                "default_reads_raw",
                f"rank_genes_groups without use_raw reads .raw of {name} because it exists: "
                f".raw holds {info.raw.says()} while X holds {info.x.says()}; "
                "the logfoldchanges column treats its input as log1p data, so it reports "
                "values far from the log fold change of X (the scores and p-values only use ranks "
                "and look normal).",
                expected=["X", "raw"],
                fix=(
                    "add use_raw=False to read the log-normalised X, and run again with "
                    "--declare de_input=X",
                    "or add use_raw=True on purpose and declare de_input=raw",
                ),
            )
    return passed()


# -- sc.small_int_qc -----------------------------------------------------------------------------

QC_CALL = re.compile(r"\bcalculate_qc_metrics\s*\(")
PERCENT_TOP_NONE = re.compile(r"\bpercent_top\s*=\s*None\b")
CASTS = re.compile(
    r"\.astype\s*\(|\bdtype\s*=|\bastype\b|float32|float64|\bint32\b|\bint64\b|"
    r"\.toarray\s*\(|\.todense\s*\(|\.A\b"
)
SMALL_INTS = frozenset({"int8", "int16"})


@checker("sc_small_int_qc")
def sc_small_int_qc(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not QC_CALL.search(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    handled = bool(CASTS.search(text) or PERCENT_TOP_NONE.search(text))
    for info in _files_or_none(call, match, ctx):
        name = info.path.name
        dtype = info.x.dtype
        if declared is not None and dtype and declared != dtype:
            return CheckResult(
                "fail",
                "declared_dtype_contradicts_data",
                f"you declared {key}={declared} but X of {name} is stored as {dtype}.",
                expected=dtype,
                got=declared,
                fix=(f"declare {key}={dtype} and cast X to float32 before the QC step",),
            )
        if info.x.sparse and dtype in SMALL_INTS and not handled:
            return CheckResult(
                "fail",
                "small_int_sparse_qc",
                f"X of {name} is a sparse {dtype} matrix: calculate_qc_metrics sums the counts "
                f"of a cell in {dtype}, so pct_counts_in_top_N_genes is wrong (negative or above "
                f"100) for every cell with more than {32767 if dtype == 'int16' else 127} UMIs; "
                "total_counts and n_genes_by_counts look right.",
                fix=(
                    "cast first: a.X = a.X.astype('float32'), then run calculate_qc_metrics, and "
                    f"declare --declare {key}={dtype}",
                    "or pass percent_top=None if the top-N percentages are not needed",
                ),
            )
    return passed()


# -- sc.double_transform -------------------------------------------------------------------------


@checker("sc_double_transform")
def sc_double_transform(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    does_log = bool(LOG_OP.search(text))
    does_norm = bool(NORM_OP.search(text))
    if not (does_log or does_norm):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    handled = bool(
        BACK_TRANSFORM.search(text) or SOURCE_NAMED.search(text) or LOOKS_AT_STATE.search(text)
    )
    for info in _files_or_none(call, match, ctx):
        name = info.path.name
        bad = _declared_contradiction(key, declared, info, name)
        if bad is not None:
            return bad
        state = info.x.state
        if state in ("log_normalized", "scaled") and not handled:
            what = "log1p" if does_log else "normalize_total"
            marker = (
                "adata.uns has 'log1p', so scanpy would warn"
                if info.log1p_marked
                else "adata.uns has no 'log1p' key, so scanpy will not warn"
            )
            return CheckResult(
                "fail",
                "second_transform",
                f"X of {name} already holds {info.x.says()}, and this code applies {what} to "
                f"it again ({marker}); the result is a log of a log, not the normalised "
                "expression.",
                fix=(
                    f"use X as it is, and declare --declare {key}={state}",
                    "or start from the counts if the object keeps them (layers['counts'])",
                ),
            )
    return passed()


# -- sc.counts_needed ----------------------------------------------------------------------------


@checker("sc_counts_needed")
def sc_counts_needed(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not COUNTS_OP.search(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    handled = bool(SOURCE_NAMED.search(text))
    for info in _files_or_none(call, match, ctx):
        name = info.path.name
        bad = _declared_contradiction(key, declared, info, name)
        if bad is not None:
            return bad
        source = info.counts_source()
        if info.x.state in ("log_normalized", "normalized", "scaled") and not handled:
            where = f"; the integer counts are in {source}" if source else ""
            return CheckResult(
                "fail",
                "counts_step_on_normalized_x",
                f"this code works on UMI counts (min_counts / total_counts / X.sum) but X of "
                f"{name} holds {info.x.says()}{where}: filter_cells and calculate_qc_metrics "
                "read X and report sums of normalised values.",
                fix=(
                    f"read the counts from {source or 'the raw count matrix'} (for example "
                    "layer='counts') and declare "
                    f"--declare {key}={info.x.state}",
                ),
            )
    return passed()


# -- sc.scaled_x_expression ----------------------------------------------------------------------


@checker("sc_scaled_x_expression")
def sc_scaled_x_expression(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = strip_comments(code_text(call, ctx))
    if not READS_X.search(text):
        return passed()
    if (missing := require_h5py()) is not None:
        return missing
    declared = call.declared.get(key)
    handled = bool(RAW_NAMED.search(text) or RAW_DROPPED.search(text))
    for info in _files_or_none(call, match, ctx):
        name = info.path.name
        bad = _declared_contradiction(key, declared, info, name)
        if bad is not None:
            return bad
        if info.x.state != "scaled" or handled:
            continue
        if info.raw is not None:
            return CheckResult(
                "fail",
                "scaled_x_read_as_expression",
                f"X of {name} holds {info.x.says()}: z-scores, so 'expressed' (X > 0) holds "
                f"for about half of all cells and means are about 0. The expression values are "
                f"in .raw ({info.raw.says()}).",
                fix=(
                    "read a.raw[:, gene].X (or a.raw.to_adata()), and declare "
                    f"--declare {key}=scaled",
                ),
            )
        return unknown(
            "scaled_x_no_raw",
            f"X of {name} holds {info.x.says()} (z-scores) and the object has no .raw: "
            f"values read from X are not expression levels (about half of them are > 0).",
            fix=("look for the unscaled matrix in layers, or ask for it",),
        )
    return passed()
