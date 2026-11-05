"""Admission cases for the BEDOPS rules (pack iv).

bedops.sorted_input (F70). For one seed both twins of the F70 task are built (implicit tier): the
same regions and peaks, in sort-bed order (clean) or in genome order (faulted). The task counts the
peaks that overlap each region. `sort-bed` copies of both files are written next to them. Every
call is run with the real BEDOPS programs in the workspace and its output is compared with
`task.truth["expected"]` (bench.score.answer_matches); the label never comes from the rule. Calls:
`bedmap --count` in several spellings on the raw files (right on the clean twin, wrong on the
faulted one), on the sorted copies (right), with one of the two files sorted (wrong), `--ec` on raw
files (the tool stops: no answer, wrong), and `bedops -e 1` listing the regions with an overlap
(the truth derived from the counts), raw and sorted. The rule has no declaration, so every call is
emitted in both cells (a declared call carries a note the rule ignores).

bedops.bedmap_score_column (F71). Both twins of the F71 task (implicit): the same peaks with the
fold enrichment in column 5 of peaks.bed (clean) or in column 7 of peaks.narrowPeak (faulted). A
bedGraph and a BED5 copy with the fold enrichment in column 5 are written next to them. The task
asks for the mean fold enrichment per region; calls run `bedmap` with a score operation on the raw
peaks (right only on the clean twin), on the bedGraph (always wrong: NAN / 0), on the BED5 copy
(right) and with other statistics (median, weighted mean: wrong). Right calls carry
`bedmap_value_column=5` in the declared cell, wrong calls on the narrowPeak carry it too (the rule
takes the declaration at its word), and the bedGraph calls carry it as well (the file has no
column 5 whatever is declared).

bedops.element_of_amount (F70 files). The sorted copies of the F70 files and a derived task: list
the regions that overlap at least one peak (the regions with a count above 0 in `task.truth`).
Calls: `bedops -e 1` and `--element-of 1` (right), the default `-e` and `--element-of` (100
percent: wrong), explicit `-e 100%` and `-e 50%` (the amount is stated; wrong), `-n 1` (the
complement: wrong; the default `-n` is not used because on these data it
returns the same regions by coincidence). The declared cell carries `overlap_amount=any` for the
right calls and `full` or `any` for the wrong ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

NOTE = {"note": "no declaration exists for this rule"}


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    reads: str  # how the output is read: counts | means | names | none
    answers: str = "task"  # task | derived (regions with an overlap)


def run(tools: ToolEnv, work: Path, template: Template):
    return tools.run([template.tool, *template.argv], work)


def parse_counts(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = re.split(r"[|\t;]", line)
        out[fields[3]] = float(fields[-1])
    return out


def parse_names(stdout: str) -> list[str]:
    return [line.split("\t")[3] for line in stdout.splitlines() if line.strip()]


def parse_means(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = re.split(r"[|\t;]", line)
        out[fields[3]] = 0.0 if fields[-1].upper() == "NAN" else float(fields[-1])
    return out


def label_of(template: Template, task, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    try:
        if template.reads == "counts":
            answer = parse_counts(proc.stdout)
            return "right" if answer_matches(task.truth, answer) else "wrong"
        if template.reads == "means":
            answer = parse_means(proc.stdout)
            return "right" if answer_matches(task.truth, answer) else "wrong"
        if template.reads == "names":
            names = parse_names(proc.stdout)
            derived = {
                "kind": "id_list",
                "expected": [r for r, c in task.truth["expected"].items() if c > 0],
            }
            return "right" if answer_matches(derived, names) else "wrong"
    except (IndexError, ValueError):
        return "wrong"
    return "wrong"


def emit(out, template, task, work, label, seed, twin, declarations, family):
    out.append(
        Case(
            Call(template.tool, template.argv, work),
            label,
            False,
            family,
            seed,
            twin,
            template.name,
        )
    )
    for declared in declarations:
        call = Call(template.tool, template.argv, work, declared)
        out.append(Case(call, label, True, family, seed, twin, template.name))


# ---- bedops.sorted_input ---------------------------------------------------------------------

COUNT = ("--echo", "--count")
RAW = ("regions.bed", "peaks.bed")
SORTED = ("regions.sorted.bed", "peaks.sorted.bed")
SORT_TEMPLATES: tuple[Template, ...] = (
    Template("count_raw", "bedmap", (*COUNT, *RAW), "counts"),
    Template("count_raw_swapped_flags", "bedmap", ("--count", "--echo", *RAW), "counts"),
    Template("count_raw_semicolon", "bedmap", ("--delim", ";", *COUNT, *RAW), "counts"),
    Template("count_raw_tab", "bedmap", ("--delim", "\\t", *COUNT, *RAW), "counts"),
    Template("count_raw_bp_ovr", "bedmap", ("--bp-ovr", "1", *COUNT, *RAW), "counts"),
    Template("count_raw_range0", "bedmap", ("--range", "0", *COUNT, *RAW), "counts"),
    Template("count_raw_regions_sorted", "bedmap", (*COUNT, SORTED[0], RAW[1]), "counts"),
    Template("count_raw_peaks_sorted", "bedmap", (*COUNT, RAW[0], SORTED[1]), "counts"),
    Template("count_raw_ec", "bedmap", ("--ec", *COUNT, *RAW), "counts"),
    Template("count_sorted", "bedmap", (*COUNT, *SORTED), "counts"),
    Template("count_sorted_swapped_flags", "bedmap", ("--count", "--echo", *SORTED), "counts"),
    Template("count_sorted_ec", "bedmap", ("--ec", *COUNT, *SORTED), "counts"),
    Template("count_sorted_range0", "bedmap", ("--range", "0", *COUNT, *SORTED), "counts"),
    Template("overlap_raw", "bedops", ("-e", "1", *RAW), "names"),
    Template("overlap_sorted", "bedops", ("-e", "1", *SORTED), "names"),
    Template("overlap_sorted_ec", "bedops", ("--ec", "-e", "1", *SORTED), "names"),
)


def write_sorted_copies(work: Path, tools: ToolEnv) -> None:
    for name, copy in zip(RAW, SORTED, strict=True):
        proc = tools.run(["sort-bed", name], work)
        (work / copy).write_text(proc.stdout)


@case_source("bedops.sorted_input")
def sorted_input_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F70", seed, twin, work, tools, "implicit", "free")
        write_sorted_copies(work, tools)
        for template in SORT_TEMPLATES:
            label = label_of(template, task, run(tools, work, template))
            emit(out, template, task, work, label, seed, twin, [NOTE], "F70")
    return out


# ---- bedops.bedmap_score_column --------------------------------------------------------------

KEY = "bedmap_value_column"
STAT = ("--echo",)
SCORE_TEMPLATES: tuple[Template, ...] = (
    Template("mean_raw", "bedmap", (*STAT, "--mean", "regions.bed", "{peaks}"), "means"),
    Template("mean_raw_swapped", "bedmap", ("--mean", "--echo", "regions.bed", "{peaks}"), "means"),
    Template(
        "mean_raw_delim",
        "bedmap",
        ("--delim", ";", "--echo", "--mean", "regions.bed", "{peaks}"),
        "means",
    ),
    Template(
        "mean_raw_prec",
        "bedmap",
        ("--prec", "3", "--echo", "--mean", "regions.bed", "{peaks}"),
        "means",
    ),
    Template("median_raw", "bedmap", (*STAT, "--median", "regions.bed", "{peaks}"), "means"),
    Template("wmean_raw", "bedmap", (*STAT, "--wmean", "regions.bed", "{peaks}"), "means"),
    Template(
        "mean_bedgraph", "bedmap", (*STAT, "--mean", "regions.bed", "signal.bedgraph"), "means"
    ),
    Template(
        "median_bedgraph", "bedmap", (*STAT, "--median", "regions.bed", "signal.bedgraph"), "means"
    ),
    Template("mean_score5", "bedmap", (*STAT, "--mean", "regions.bed", "fold.bed"), "means"),
    Template(
        "mean_score5_prec",
        "bedmap",
        ("--prec", "4", *STAT, "--mean", "regions.bed", "fold.bed"),
        "means",
    ),
    Template(
        "mean_score5_ec", "bedmap", ("--ec", *STAT, "--mean", "regions.bed", "fold.bed"), "means"
    ),
    Template("median_score5", "bedmap", (*STAT, "--median", "regions.bed", "fold.bed"), "means"),
)


def write_score_files(work: Path) -> str:
    """signal.bedgraph (value in column 4) and fold.bed (BED5, value in column 5) from the peaks."""
    narrow = work / "peaks.narrowPeak"
    peaks = "peaks.narrowPeak" if narrow.exists() else "peaks.bed"
    rows = [line.split("\t") for line in (work / peaks).read_text().splitlines()]
    column = 6 if narrow.exists() else 4
    (work / "signal.bedgraph").write_text(
        "".join(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[column]}\n" for r in rows)
    )
    (work / "fold.bed").write_text(
        "".join(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[column]}\n" for r in rows)
    )
    return peaks


@case_source("bedops.bedmap_score_column")
def score_column_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F71", seed, twin, work, tools, "implicit", "free")
        peaks = write_score_files(work)
        for base in SCORE_TEMPLATES:
            template = Template(
                base.name,
                base.tool,
                tuple(peaks if a == "{peaks}" else a for a in base.argv),
                base.reads,
            )
            label = label_of(template, task, run(tools, work, template))
            emit(out, template, task, work, label, seed, twin, [{KEY: "5"}], "F71")
    return out


# ---- bedops.element_of_amount ----------------------------------------------------------------

AMOUNT_KEY = "overlap_amount"
ELEMENT_TEMPLATES: tuple[tuple[Template, str], ...] = (
    (Template("element_of_1", "bedops", ("-e", "1", *SORTED), "names"), "any"),
    (Template("element_of_long_1", "bedops", ("--element-of", "1", *SORTED), "names"), "any"),
    (Template("element_of_1_ec", "bedops", ("--ec", "-e", "1", *SORTED), "names"), "any"),
    (Template("element_of_default", "bedops", ("-e", *SORTED), "names"), "full"),
    (Template("element_of_long_default", "bedops", ("--element-of", *SORTED), "names"), "full"),
    (Template("element_of_default_ec", "bedops", ("--ec", "-e", *SORTED), "names"), "full"),
    (Template("element_of_100", "bedops", ("-e", "100%", *SORTED), "names"), "full"),
    (Template("element_of_50", "bedops", ("-e", "50%", *SORTED), "names"), "full"),
    (Template("not_element_of_1", "bedops", ("-n", "1", *SORTED), "names"), "any"),
)


@case_source("bedops.element_of_amount")
def element_of_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F70", seed, twin, work, tools, "implicit", "free")
        write_sorted_copies(work, tools)
        for template, own in ELEMENT_TEMPLATES:
            label = label_of(template, task, run(tools, work, template))
            # declared cell: what the call's own options mean, and (for wrong calls) the other value
            values = [own] if label == "right" else ["full", "any"]
            emit(
                out,
                template,
                task,
                work,
                label,
                seed,
                twin,
                [{AMOUNT_KEY: v} for v in values],
                "F70",
            )
    return out
