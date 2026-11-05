"""Rules for BEDOPS (pack iv): input order, the score column, the default overlap amount.

All three were verified on bedops, bedmap and closest-features 2.4.42 (bio-a), exit 0 and an empty
stderr in every case quoted here.

bedops.sorted_input (slot sort_order, CALL rule, tier T1, no declaration; F70). BEDOPS wants every
input in `sort-bed` order: chromosome names compared as text (chr1, chr10, chr11, chr2, ...), then
start, then end. Nothing is checked unless `--ec` is given. Two BED files in genome order
(chr1, chr2, ..., chr9, chr10, chr11, chr12, chrX; what `sort -V`, a genome file or `samtools
faidx` give): `bedmap --echo --count regions.bed peaks.bed` returned 0 (or the count of another
chromosome) for the regions of chr10-chr12, `bedops -e 1` kept 2 of 4 elements and
`closest-features` printed NA for chr10 and chr11; the same files written by `sort-bed` give the
counts of an independent overlap count. A file unsorted on one side only is wrong as well; within
one chromosome, a start that goes backwards gives 0 counts (`bedmap --count` on 500-600 then
100-200 against 110-120: 0 and 0). `--ec` stops with "Bed file not properly sorted by first column"
(exit 1). The rule reads every existing file named in the call (the first 4 MB of text), compares
neighbouring keys (chrom as bytes, start, end as integers) and fails at the first line that goes
backwards, unless `--ec` is among the arguments. Standard input (`-`), process substitutions and
files that do not exist yet are not seen. A sorted-looking prefix of a very large file does not
prove the rest.

bedops.bedmap_score_column (slot score_column, CALL rule, tier T1, optional declaration
`bedmap_value_column`; F71). Every bedmap score operation (--mean, --sum, --median, --min, --max,
--stdev, --wmean, --echo-map-score, ...) reads the value from column 5 of the map file ("<map-file>
5 columns" in `bedmap --help`). A four-column bedGraph (value in column 4): `--mean` prints
0.000000 for a region with one element and NAN for the others (also where elements overlap, and
`--count` in the same call undercounts: 1 0 0 0 for 1 2 3 1), `--echo-map-score` prints nothing; a
BED with a non-numeric fifth column prints 0.000000 / NAN; the same data with a dummy column 4 and
the value in column 5 give 3.5, 7.0, 8.0, 2.0. A MACS3 narrowPeak file holds the scaled score
(0-1000) in column 5 and the fold enrichment in column 7: `bedmap --mean regions peaks.narrowPeak`
averages the scaled scores (143.0 where the fold enrichment is 4.7), plausible numbers. The rule
reads the map file (the last existing file named, or the only one): a line with fewer than five
columns or a fifth column that is not a number fails (code `no_score_column`, whatever is
declared); a peak-format file (name ending in Peak, or ten columns with numeric 7-9 and an integer
column 10) fails (`peak_score_column`) unless the call declares `bedmap_value_column=5` (the agent
says the scaled score is what it wants). A plain BED5/BED6 with a numeric score passes.

bedops.element_of_amount (slot overlap_amount, CALL rule, tier T0, optional declaration
`overlap_amount`; no family). `bedops -e` / `--element-of` and `-n` / `--not-element-of` are
documented with "by default, -e 100% is used": without an amount the reference element must be
entirely covered. Regions R1 100-200, R2 300-400, R3 500-600 against map elements 150-160, 300-400,
450-650: `bedops -e R M` prints R2 and R3, `-e 100%` the same, `-e 50%` the same, `-e 1` prints
R1, R2 and R3 (the elements with any overlap); `-n R M` prints R1 only (the ones not entirely
covered), `-n 1` nothing. The rule is T0: an explicit amount (`-e 1`, `-e 50%`, `-e1`) passes; with
no amount the call must declare `overlap_amount=full` (100 percent is intended) or `any`; `any`
contradicts the call and fails, a missing declaration fails with the fix.
"""

from __future__ import annotations

import re
from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed

_SKIP_PREFIXES = ("#", "track", "browser")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_AMOUNT = re.compile(r"\d+(?:\.\d+)?%?")
PEAK_SUFFIXES = (".narrowpeak", ".broadpeak", ".gappedpeak")


def existing_files(call: Call, ctx: Ctx) -> list[Path]:
    """Existing regular files among the arguments, in order, each once (options are skipped)."""
    seen: list[Path] = []
    for token in call.argv:
        if token.startswith("-") or token == "":
            continue
        path = ctx.resolve(token)
        if path.is_file() and path not in seen:
            seen.append(path)
    return seen


def bed_rows(ctx: Ctx, path: Path, limit: int | None = None):
    """Data lines of a BED-like file as lists of fields (tab, else whitespace)."""
    for line in ctx.lines(path, limit=limit):
        if not line.strip() or line.startswith(_SKIP_PREFIXES):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            fields = line.split()
        yield fields


# ---- bedops.sorted_input ---------------------------------------------------------------------


def first_disorder(ctx: Ctx, path: Path) -> tuple[int, list[str], list[str]] | None:
    """(data line number, previous fields, fields) of the first line out of sort-bed order."""
    prev_key: tuple[str, int, int] | None = None
    prev_fields: list[str] = []
    for number, fields in enumerate(bed_rows(ctx, path), start=1):
        try:
            key = (fields[0], int(fields[1]), int(fields[2]))
        except (IndexError, ValueError):
            return None  # not a BED file: nothing to say
        if prev_key is not None and key < prev_key:
            return number, prev_fields, fields
        prev_key, prev_fields = key, fields
    return None


@checker("iv_bedops_sorted_input")
def iv_bedops_sorted_input(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if "--ec" in call.argv:
        return passed()
    for path in existing_files(call, ctx):
        if path.suffix in (".starch", ".gz", ".bz2", ".bam"):
            continue
        try:
            found = first_disorder(ctx, path)
        except (OSError, UnicodeError):
            continue
        if found is None:
            continue
        number, before, after = found
        where = f"{path.name}: data line {number} ({after[0]} {after[1]}-{after[2]})"
        if before[0] > after[0]:
            why = (
                f"{where} comes after {before[0]}, but {after[0]} sorts before {before[0]} as "
                "text (sort-bed order is chr1, chr10, chr11, chr2, ..., not genome order)"
            )
        else:
            why = (
                f"{where} comes after {before[0]} {before[1]}-{before[2]}: starts and ends must "
                "not go backwards within a chromosome"
            )
        return CheckResult(
            "fail",
            "not_in_sort_bed_order",
            f"{call.tool} reads its inputs in sort-bed order and checks nothing without --ec, "
            "so this call returns wrong results (zero counts, NA, missed overlaps) silently; "
            f"{why}.",
            fix=(
                f"sort-bed {path.name} > {path.stem}.sorted{path.suffix} and use the sorted copy "
                "for every input file",
                "or add --ec so that BEDOPS stops on a file that is out of order",
            ),
        )
    return passed()


# ---- bedops.bedmap_score_column --------------------------------------------------------------

PEAK_NUMERIC_COLUMNS = (6, 7, 8)  # 0-based: signalValue, pValue, qValue of a narrowPeak


BEDMAP_VALUE_OPTIONS = {
    "--chrom": 1,
    "--delim": 1,
    "--multidelim": 1,
    "--prec": 1,
    "--unmapped-val": 1,
    "--bp-ovr": 1,
    "--range": 1,
    "--fraction-ref": 1,
    "--fraction-map": 1,
    "--fraction-both": 1,
    "--fraction-either": 1,
    "--kth": 1,
    "--tmean": 2,
}


def bedmap_positionals(argv: tuple[str, ...]) -> list[str]:
    """The file arguments of a bedmap call (`-` is standard input); option values are skipped."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if token in BEDMAP_VALUE_OPTIONS:
            i += BEDMAP_VALUE_OPTIONS[token]
        elif token == "--mad":  # the multiplier is optional
            if i < len(argv) and _NUMBER.fullmatch(argv[i]):
                i += 1
        elif token == "-" or not token.startswith("-"):
            out.append(token)
    return out


def map_file(call: Call, ctx: Ctx) -> Path | None:
    """The map file: the last file argument, or the only one; None when it is standard input."""
    positionals = bedmap_positionals(call.argv)
    if not positionals or positionals[-1] == "-":
        return None
    path = ctx.resolve(positionals[-1])
    return path if path.is_file() else None


def looks_like_peak_file(path: Path, rows: list[list[str]]) -> bool:
    if path.name.lower().endswith(PEAK_SUFFIXES):
        return True
    first = rows[0] if rows else []
    return (
        len(first) == 10
        and all(_NUMBER.fullmatch(first[i] or "x") for i in PEAK_NUMERIC_COLUMNS)
        and first[9].lstrip("-").isdigit()
    )


@checker("iv_bedops_bedmap_score_column")
def iv_bedops_bedmap_score_column(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    path = map_file(call, ctx)
    if path is None or path.suffix in (".starch", ".gz", ".bz2"):
        return passed()
    try:
        rows = list(bed_rows(ctx, path, limit=500))
    except (OSError, UnicodeError):
        return passed()
    if not rows:
        return passed()
    bad = next((r for r in rows if len(r) < 5 or not _NUMBER.fullmatch(r[4])), None)
    if bad is not None:
        shape = (
            f"only {len(bad)} columns" if len(bad) < 5 else f"{bad[4]!r} in column 5, not a number"
        )
        return CheckResult(
            "fail",
            "no_score_column",
            f"bedmap score operations read the value from column 5 of the map file, but "
            f"{path.name} has {shape} (a bedGraph keeps its value in column 4): the result is "
            "0.000000 or NAN, silently.",
            fix=(
                f'awk \'BEGIN{{OFS="\\t"}}{{print $1,$2,$3,".",$4}}\' {path.name} > '
                f"{path.stem}.score5.bed, then use that file as the map file",
            ),
        )
    if looks_like_peak_file(path, rows) and call.declared.get(rule.declare.key) != "5":
        return CheckResult(
            "fail",
            "peak_score_column",
            f"{path.name} is a narrowPeak-style file: column 5 is the scaled score (0-1000), the "
            "signal value (fold enrichment) is column 7, and bedmap averages column 5 without "
            "saying so.",
            fix=(
                f"awk 'BEGIN{{OFS=\"\\t\"}}{{$5=$7; print}}' {path.name} > {path.stem}.signal.bed "
                "and map that file (column 5 = signalValue)",
                f"or, if the scaled score is what you want, re-run with --declare "
                f"{rule.declare.key}=5",
            ),
        )
    return passed()


# ---- bedops.element_of_amount ----------------------------------------------------------------

ELEMENT_OPTIONS = ("-e", "--element-of", "-n", "--not-element-of")


def amount_of(argv: tuple[str, ...]) -> tuple[str, str | None]:
    """(option, explicit amount or None) of the first -e/-n option in `argv`."""
    for i, token in enumerate(argv):
        for option in ELEMENT_OPTIONS:
            if token == option:
                following = argv[i + 1] if i + 1 < len(argv) else ""
                return option, following if _AMOUNT.fullmatch(following) else None
            if (
                len(option) == 2
                and token.startswith(option)
                and _AMOUNT.fullmatch(token[2:] or "x")
            ):
                return option, token[2:]
    return "", None


@checker("iv_bedops_element_of_amount")
def iv_bedops_element_of_amount(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    option, amount = amount_of(call.argv)
    if not option or amount is not None:
        return passed()
    declared = call.declared.get(key)
    verb = "covered" if option in ("-e", "--element-of") else "not entirely covered"
    if declared == "full":
        return passed()
    if declared == "any":
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=any but bedops {option} without an amount requires 100 percent "
            "of the reference element to be covered, not one base.",
            expected="any",
            got="full",
            fix=(f"add the amount: bedops {option} 1 reference.bed map.bed",),
        )
    return CheckResult(
        "fail",
        "amount_not_stated",
        f"bedops {option} without an amount defaults to 100 percent: only reference elements "
        f"entirely {verb} by the other files qualify, so partly overlapped elements are "
        "silently left out (or, for -n, kept).",
        fix=(
            f"bedops {option} 1 reference.bed map.bed selects on at least one base of overlap "
            "(or give a percentage such as 50%)",
            f"or declare --declare {key}=full if 100 percent is what you want",
        ),
    )
