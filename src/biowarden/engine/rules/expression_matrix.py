"""Rule matrix.log_scale_state (slot matrix_state, family F15, incidents SCD-05 and SCD-19).

Trap: an expression matrix holds log2(count + 1) (or CPM/TPM-like) values and a count-scale
question is answered on them as if they were raw counts. Nothing fails: the file is a valid table.
Evidence (gawk 4.2.1, perl 5, python 3.12 on synthetic F15 workspaces, seed 3, "count genes whose
mean raw count is above 100", truth 31):

  clean twin (raw integer counts)   awk mean of the file values            -> 31   right
                                    awk mean of 2^x-1                      -> 74   wrong (2^count)
  faulted twin (log2(count+1), 4 decimals, max about 17)
                                    awk mean of the file values            -> ""   (no gene
                                                                           passes: n is never set;
                                                                           python prints 0)
                                    awk mean of 2^x-1, or exp(x*log(2))-1  -> 31   right
  perl -F'\t' -lane gives the same four numbers (31, 74, 0, 31) for its file-value and 2**x-1 means.
  awk reads "NA" as 0 and still divides by NF-1 (mean of "NA 200" = 100), so a missing value is
  not an error either; the sniffer therefore skips NA tokens and never lets them decide the state.
  log2(count+1) rounded to 4 decimals prints exact integers for counts 0, 1, 3, 7, 15 ... (0.0000,
  1.0000, 2.0000, 3.0000, 4.0000), so single values cannot tell integers from logs; only a whole
  column can.

What the checker does (fact rule, tier T1: it reads a bounded prefix of the files named in the call)
  1. `sniff` reads the header and at most MAX_ROWS data rows, picks the delimiter, and decides
     whether the file is an expression matrix at all (see `sniff`). A table that is not one (sample
     sheets, DE results, correlation or distance matrices, proportions, long format, fewer than
     MIN_ROWS rows) is no match: zero friction.
  2. It classifies the numeric sample columns: all non-negative integers -> raw_counts (silent pass,
     no declaration needed, nothing else is checked); otherwise non-integer or negative values
     with a maximum below LOG_MAX -> log_transformed, with a larger maximum -> normalized.
  3. For a risky state (log_transformed or normalized):
       no declaration                      -> fail (observed range, why it matters, how to fix)
       declaration equals the state        -> pass
       declaration raw_counts              -> fail (the data are not raw integers)
       the other risky state declared      -> fail, or unknown when the state itself is uncertain
                                              (maximum in [GREY_LOW, GREY_HIGH) or negative values)
     For raw_counts data the only failure is a contradicted declaration: log_transformed declared
     while the integers reach LOG_MAX or more (log values never get that large). normalized on
     integers passes (rounded normalised counts exist).
  4. A file that cannot be read gives `unknown` (warn), never a crash.

Thresholds and why
  LOG_MAX = 30. log2 of the largest possible count (1e9 reads on one gene) is about 30, and real
    log2(count+1), log2(CPM+1) and log2(TPM+1) matrices peak at 12 to 22 (log2 of 1e6 is 20); ln and
    log10 peak lower. CPM, TPM, FPKM and size-factor-normalised counts reach 1e3 to 1e6 in any
    transcriptome-wide matrix, so the two populations are far apart; 30 sits in the empty gap.
  GREY zone [20, 100): a log2 matrix with a very deep gene and a normalised matrix of a small gene
    set can both land here, so a mismatch between two RISKY declarations is only `unknown` there.
    An undeclared risky matrix still fails: either way it is not raw counts.
  Fractional share >= 0.5 of the sample columns decides non-raw; a minority of fractional columns
    among integer ones is annotation (gc fraction, a tpm column) and the matrix stays raw_counts.
  Maximum <= 1 and no negatives is proportions or probabilities, not a count-derived matrix: no
    match.
  Sniffing only the first MAX_ROWS rows can underestimate the maximum, but the log/normalised cut
    sits an order of magnitude from either population, and integer-ness needs a single
    non-integer to show (any fractional column of >= 8 rows has one).

Known false positives and limits
  * Fractional estimated counts (RSEM expected_count, salmon NumReads) are non-integer with a large
    range, so they read as `normalized`; declare matrix_state=normalized once. Their thresholds are
    on the count scale, so this is friction on correct data, accepted and documented.
  * z-scored or centred matrices (negative values) read as log_transformed/normalized by maximum;
    the verdict is `unknown` if a different risky state is declared, a failure when undeclared.
  * Tables whose first column header is a sample label (sample, patient, subject ...) are taken for
    sample sheets and are no match; a transposed sample-by-gene matrix with such a header is
    therefore not checked (a missed detection, never friction).
  * A script that reads the file (`python3 analyze.py`) names no file in its argv: not seen.
  * `cat expr.tsv | awk ...`: the awk call names no file: not seen. `awk ... < expr.tsv` is seen
    only if the front end passes the redirect target along.
  * sort is only checked for numeric sorts (-n, -g, -h, numeric key flags): a lexical sort does not
    interpret the values.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import (
    CheckResult,
    Ctx,
    Match,
    TierViolation,
    Verdict,
    checker,
    passed,
)

KEY = "matrix_state"
RAW, LOG, NORMALIZED = "raw_counts", "log_transformed", "normalized"

MAX_ROWS = 300  # data rows sniffed per file
COMMENT_SLACK = 60  # leading comment lines tolerated on top of MAX_ROWS
MIN_ROWS = 8  # fewer rows than this is not treated as a matrix
MIN_SAMPLE_COLUMNS = 3
LOG_MAX = 30.0
GREY_LOW, GREY_HIGH = 20.0, 100.0
MAX_COLUMN_SPREAD = 200.0  # largest column maximum over the smallest: sample columns are alike
DIGIT_ROWS = 30  # below this many rows the sample column names must carry a digit (S1, GSM42)
MIN_NUMERIC_SHARE = 0.5
MIN_FRACTIONAL_SHARE = 0.5

_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_MISSING = frozenset({"", "na", "n/a", "nan", "null", "none", "-", ".", "nd", "#n/a"})
# column names of statistics tables (DESeq2, edgeR, limma), never an expression matrix
_RESULT_NAMES = re.compile(
    r"p[._ -]?val|padj|adj[._ -]?p|\bfdr\b|q[._ -]?val|log2?[._ -]?fold|log[._ -]?fc|\blfc|"
    r"base[._ -]?mean|fold[._ -]?change|^stat$|^t$|^b$|ave[._ -]?expr|z[._ -]?score",
    re.IGNORECASE,
)
# numeric columns that annotate the rows instead of measuring a sample
_ANNOTATION_NAMES = re.compile(
    r"^(?:chr|chrom|chromosome|start|end|pos|position|strand|length|width|gene[._ -]?length|"
    r"eff[._ -]?length|effective[._ -]?length|gc|gc[._ -]?content|score|index|rank|total|sum|"
    r"mean|median|sd|std|var|variance|cv|n|count|age|weight|height|bmi|dose|time|day|days|"
    r"week|year|(?:pc|dim|umap|tsne|comp|component|factor|axis)[._ -]?\d*)$",
    re.IGNORECASE,
)
# a first column of samples (a sample sheet, QC table) rather than of genes
_SAMPLE_ROWS = re.compile(
    r"^(?:sample|samples|sample[._ -]?(?:id|name)|patient|subject|participant|specimen|case|"
    r"individual|mouse|run|library)$",
    re.IGNORECASE,
)
_ID_HEADER = re.compile(r"id|gene|symbol|entrez|ensembl|probe|feature|name|transcript", re.I)


@dataclass(frozen=True)
class MatrixProfile:
    """What the sniffer saw in a table that is an expression matrix."""

    state: str  # raw_counts | log_transformed | normalized
    ambiguous: bool  # the state is a guess between log_transformed and normalized
    minimum: float  # over the sample columns that decided the state
    maximum: float
    rows: int  # data rows read
    sample_columns: int
    fractional_columns: int


# -- sniffing -------------------------------------------------------------------------------------


def _delimiters(name: str) -> list[str]:
    base = name.lower()
    if base.endswith(".gz"):
        base = base[:-3]
    hint = "," if base.endswith(".csv") else "\t" if base.endswith((".tsv", ".tab")) else None
    order = ["\t", ",", ";", " "]
    return ([hint] if hint else []) + [d for d in order if d != hint]


def _parse(lines: Sequence[str], delim: str) -> list[list[str]]:
    if delim == " ":
        return [line.split() for line in lines]
    try:
        return [[cell.strip() for cell in row] for row in csv.reader(lines, delimiter=delim)]
    except csv.Error:
        return [[cell.strip() for cell in line.split(delim)] for line in lines]


def _split_rows(lines: Sequence[str], name: str) -> list[list[str]] | None:
    """Rows of the table with the first delimiter that gives a consistent width of >= 3."""
    for delim in _delimiters(name):
        probe = _parse(lines[1:40], delim)
        widths = Counter(len(row) for row in probe if row)
        if not widths:
            continue
        width, count = widths.most_common(1)[0]
        if width >= 3 and count >= 0.8 * len(probe):
            return _parse(lines, delim)
    return None


def _number(cell: str) -> float | None:
    if not _NUM.match(cell):
        return None
    value = float(cell)
    return value if math.isfinite(value) else None


@dataclass
class _Column:
    name: str
    numbers: int = 0
    other: int = 0  # cells that are neither numbers nor missing
    fractional: bool = False
    negative: bool = False
    low: float = math.inf
    high: float = -math.inf

    @property
    def present(self) -> bool:
        return (self.numbers + self.other) > 0

    @property
    def numeric(self) -> bool:
        return self.numbers >= 3 and self.numbers >= 0.9 * (self.numbers + self.other)


def _column(name: str, cells: Sequence[str]) -> _Column:
    col = _Column(name)
    values: list[float] = []
    for cell in cells:
        if _NUM.match(cell):
            value = float(cell)
            if value - value == 0.0:  # finite
                values.append(value)
                continue
        elif cell.lower() in _MISSING:
            continue
        col.other += 1
    if values:
        col.numbers = len(values)
        col.low, col.high = min(values), max(values)
        col.negative = col.low < 0
        col.fractional = any(v != int(v) for v in values)
    return col


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def sniff(lines: Sequence[str], name: str = "") -> MatrixProfile | None:
    """Classify the head of a text table, or None when it is not an expression matrix.

    An expression matrix here: a header; one feature id per row in the first column (unique, not
    fractional numbers); >= MIN_ROWS rows of one width; >= MIN_SAMPLE_COLUMNS numeric sample columns
    that make up at least half of the non-empty columns (annotation columns such as Length, Chr,
    gene_name may sit around them); sample column names with a digit (S1, GSM42) unless there are
    >= DIGIT_ROWS rows; column maxima within MAX_COLUMN_SPREAD of each other (samples are alike);
    no statistics-table column names (pvalue, log2FoldChange, baseMean ...); not square with the row
    ids equal to the column names (correlation, distance); first column header not a sample label
    (sample, patient, subject ...: a sample sheet or QC table, which also means a transposed
    sample-by-gene matrix is not recognised).
    """
    kept_lines = [line for line in lines if line.strip() and not line.startswith("#")]
    if len(kept_lines) < MIN_ROWS + 1:
        return None
    rows = _split_rows(kept_lines[: MAX_ROWS + 1], name)
    if rows is None:
        return None
    header, body = rows[0], rows[1:]
    numeric_header = sum(_number(c) is not None for c in header[1:])
    if header[1:] and numeric_header >= 0.8 * len(header[1:]):  # no header row: all data
        header, body = [], rows
    if not body:
        return None
    width = Counter(len(r) for r in body[:100]).most_common(1)[0][0]
    if width < 3:
        return None
    if header and len(header) == width - 1:  # R write.table: no label above the row names
        header = ["", *header]
    if header and len(header) != width:
        return None
    body = [r for r in body if len(r) == width]
    if len(body) < MIN_ROWS:
        return None
    names = header or [""] * width
    if any(_RESULT_NAMES.search(c) for c in names[1:]) or _SAMPLE_ROWS.match(names[0]):
        return None

    ids = [r[0] for r in body]
    id_numbers = [_number(i) for i in ids]
    numeric_ids = [v for v in id_numbers if v is not None]
    if len(numeric_ids) >= 0.9 * len(ids):
        # numeric first column: Entrez-style integer ids only when the header calls it an id
        if any(not v.is_integer() for v in numeric_ids) or not _ID_HEADER.search(names[0]):
            return None
    if len(set(ids)) < 0.9 * len(ids):
        return None

    columns = [_column(names[j], [r[j] for r in body]) for j in range(1, width)]
    present = [c for c in columns if c.present]
    numeric = [c for c in present if c.numeric]
    sample = [c for c in numeric if not _ANNOTATION_NAMES.match(c.name)]
    if len(sample) < MIN_SAMPLE_COLUMNS or len(numeric) < MIN_NUMERIC_SHARE * len(present):
        return None
    if len(body) < DIGIT_ROWS and sum(_has_digit(c.name) for c in sample) < 0.5 * len(sample):
        return None
    if set(ids) and len(set(ids) & set(names[1:])) >= 0.8 * len(body):
        return None  # square, row ids equal column names: correlation or distance matrix

    fractional = [c for c in sample if c.fractional or c.negative]
    if len(fractional) < MIN_FRACTIONAL_SHARE * len(sample):
        return MatrixProfile(
            RAW, False, 0.0, max(c.high for c in sample), len(body), len(sample), 0
        )
    low = min(c.low for c in fractional)
    high = max(c.high for c in fractional)
    if low >= 0 and high <= 1.0:
        return None  # proportions or probabilities
    tops = [c.high for c in fractional if c.high > 0]
    if tops and max(tops) / min(tops) > MAX_COLUMN_SPREAD:
        return None  # columns of unlike scale: a table of different measurements
    state = LOG if high < LOG_MAX else NORMALIZED
    ambiguous = low < 0 or GREY_LOW <= high < GREY_HIGH
    return MatrixProfile(state, ambiguous, low, high, len(body), len(sample), len(fractional))


# -- the checker ----------------------------------------------------------------------------------


def _numeric_sort(argv: Sequence[str]) -> bool:
    """True when a `sort` call interprets the values as numbers."""
    items = list(argv)
    for i, token in enumerate(items):
        if token in ("--numeric-sort", "--general-numeric-sort", "--human-numeric-sort"):
            return True
        if token in ("-k", "--key") and i + 1 < len(items):
            if re.search(r"[ngh]", re.sub(r"[\d,.]", "", items[i + 1])):
                return True
        elif token.startswith("-") and not token.startswith("--"):
            flags = token[1:]
            if flags[:1] in ("t", "T", "S", "o"):  # option with an attached value
                continue
            if re.search(r"[ngh]", re.sub(r"[\d,.]", "", flags)):
                return True
        elif token.startswith("--key=") and re.search(r"[ngh]", re.sub(r"[\d,.]", "", token[6:])):
            return True
    return False


def _range(profile: MatrixProfile) -> str:
    return f"{profile.minimum:.6g} to {profile.maximum:.6g}"


def _where(path: Path, profile: MatrixProfile) -> str:
    return f"{path.name} ({profile.sample_columns} sample columns, first {profile.rows} rows read)"


_WHY = "thresholds, means or fold changes computed on these values are not raw-count statistics"
_BACK = (
    "back-transform first (log2(count+1): 2^x-1, e.g. awk 2^$i-1, python 2**x-1; "
    "ln(count+1): exp(x)-1)"
)


def _undeclared(path: Path, p: MatrixProfile) -> CheckResult:
    if p.state == LOG:
        message = (
            f"{_where(path, p)} holds non-integer values from {_range(p)}, the range of "
            f"log-transformed counts, not raw counts: {_WHY}."
        )
        fix = (f"{_BACK}, then re-run with --declare {KEY}={LOG}; or use the raw count matrix",)
    else:
        message = (
            f"{_where(path, p)} holds non-integer values from {_range(p)}, the scale of "
            f"normalised data (CPM/TPM/FPKM-like), not raw counts: {_WHY}."
        )
        fix = (
            f"use the raw count matrix for count statistics, or re-run with --declare "
            f"{KEY}={NORMALIZED} if the normalised scale is what you intend",
        )
    return CheckResult(
        "fail", "matrix_state_undeclared", message, expected=[LOG, NORMALIZED], got=p.state, fix=fix
    )


def _declared_raw_but_not(path: Path, p: MatrixProfile) -> CheckResult:
    message = (
        f"you declared {KEY}={RAW} but {_where(path, p)} is not raw integers: its values run from "
        f"{_range(p)} ({'log-like' if p.state == LOG else 'normalised-like'} scale); {_WHY}."
    )
    fix = (
        f"declare {KEY}={p.state} and {_BACK if p.state == LOG else 'work on that scale'}; "
        f"or use the raw count matrix and keep {KEY}={RAW}",
    )
    return CheckResult(
        "fail", "declared_raw_not_integer", message, expected=p.state, got=RAW, fix=fix
    )


def _declared_other(path: Path, p: MatrixProfile, declared: str) -> CheckResult:
    verdict: Verdict = "unknown" if p.ambiguous else "fail"
    message = (
        f"you declared {KEY}={declared} but {_where(path, p)} reads as {p.state}: values "
        f"{_range(p)}"
        + (" sit in the range where log and normalised scales overlap" if p.ambiguous else "")
        + f"; {_WHY}."
    )
    return CheckResult(
        verdict,
        "declared_state_mismatch",
        message,
        expected=p.state,
        got=declared,
        fix=(f"declare {KEY}={p.state}",),
    )


def _contradicted(path: Path, p: MatrixProfile) -> CheckResult:
    message = (
        f"you declared {KEY}={LOG} but {_where(path, p)} holds non-negative integers up to "
        f"{p.maximum:.6g}; log-transformed counts do not reach {LOG_MAX:g}, so these are raw "
        "counts."
    )
    return CheckResult(
        "fail",
        "declared_log_but_integers",
        message,
        expected=RAW,
        got=LOG,
        fix=(f"declare {KEY}={RAW}; do not back-transform integers with 2^x-1",),
    )


@checker("expression_matrix_state")
def expression_matrix_state(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    """See the module docstring. params: none; the thresholds are module constants."""
    rule = match.rule
    assert rule.declare is not None
    declared = call.declared.get(rule.declare.key)
    if call.tool == "sort" and not _numeric_sort(call.argv):
        return passed("lexical sort does not interpret the values")

    profiles: list[tuple[Path, MatrixProfile]] = []
    unreadable: list[str] = []
    for path in match.files:
        try:
            lines = list(ctx.lines(path.absolute(), limit=MAX_ROWS + COMMENT_SLACK + 1))
        except TierViolation:
            raise
        except Exception as exc:  # unreadable, truncated gzip, not text: cannot decide
            unreadable.append(f"{path.name}: {type(exc).__name__}")
            continue
        try:
            profile = sniff(lines, path.name)
        except Exception as exc:  # a sniffer defect must warn, never block
            unreadable.append(f"{path.name}: sniff {type(exc).__name__}")
            continue
        if profile is not None:
            profiles.append((path, profile))

    risky = [(path, p) for path, p in profiles if p.state != RAW]
    results: list[CheckResult] = []
    for path, p in risky:
        if declared is None:
            results.append(_undeclared(path, p))
        elif declared == p.state:
            continue
        elif declared == RAW:
            results.append(_declared_raw_but_not(path, p))
        else:
            results.append(_declared_other(path, p, declared))
    if not risky and declared == LOG:
        results += [_contradicted(path, p) for path, p in profiles if p.maximum >= LOG_MAX]

    for verdict in ("fail", "unknown"):
        hits = [r for r in results if r.verdict == verdict]
        if hits:
            return hits[0]
    if unreadable and not profiles:
        return CheckResult("unknown", "unreadable", "could not read " + "; ".join(unreadable))
    if not profiles:
        return passed("no expression matrix among the files named")
    states = ", ".join(f"{path.name}={p.state}" for path, p in profiles)
    return passed(f"matrix state consistent: {states}")
