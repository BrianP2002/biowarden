"""Admission cases for the tabix rules of the variant pack (family F61, implicit tier).

intervals.bed (BED, 0-based) and windows.tsv (BED coordinates under another name) are the files of
the task; the answer is the names of the intervals that overlap a window, half-open. The faulted
twin adds intervals that only touch a window. Each call is run for real against a `-p bed` index
and the names it prints are compared with `task.truth` (bench.score.answer_matches); labels never
come from a rule. (Indexing by columns without -0 is a trap of the family but has no rule: see
engine/rules/var_tabix.py.)

tabix.regions_file_base: `-R windows.tsv` reads the BED windows as 1-based (wrong on the faulted
twin, right on the clean one); `-R windows.bed` (a copy under a BED name) is right on both.

tabix.region_strings_bed: region strings typed with the BED numbers are wrong on the faulted twin,
strings with the start moved up by one are right on both; `one_based_closed` is the declaration of
the converted numbers, `zero_based_half_open` the belief of the copied ones.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template, emit
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f61_tabix_bed import names

# ---- shared preparation --------------------------------------------------------------------------


def prepare(work: Path, tools: ToolEnv) -> None:
    for argv in (["bgzip", "-k", "-f", "intervals.bed"],):
        proc = tools.run(argv, work)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
    shutil.copy(work / "windows.tsv", work / "windows.bed")
    proc = tools.run(["tabix", "-f", "-p", "bed", "intervals.bed.gz"], work)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


def label_of(task, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    return "right" if answer_matches(task.truth, names(proc.stdout)) else "wrong"


# ---- tabix.regions_file_base ---------------------------------------------------------------------

BASE_KEY = "region_file_base"


def _rf(name: str, regions: str, *options: str) -> Template:
    meaning = "zero_based_half_open"
    return Template(name, "tabix", ("intervals.bed.gz", *options, regions), meaning)


def regions_templates() -> list[Template]:
    return [
        Template("tsv", "tabix", ("intervals.bed.gz", "-R", "windows.tsv"), "zero_based_half_open"),
        Template(
            "tsv_long",
            "tabix",
            ("intervals.bed.gz", "--regions", "windows.tsv"),
            "zero_based_half_open",
        ),
        Template(
            "tsv_targets",
            "tabix",
            ("intervals.bed.gz", "-T", "windows.tsv"),
            "zero_based_half_open",
        ),
        Template(
            "tsv_before_file",
            "tabix",
            ("-R", "windows.tsv", "intervals.bed.gz"),
            "zero_based_half_open",
        ),
        Template("bed", "tabix", ("intervals.bed.gz", "-R", "windows.bed"), "zero_based_half_open"),
        Template(
            "bed_long",
            "tabix",
            ("intervals.bed.gz", "--regions", "windows.bed"),
            "zero_based_half_open",
        ),
        Template(
            "bed_targets",
            "tabix",
            ("intervals.bed.gz", "-T", "windows.bed"),
            "zero_based_half_open",
        ),
    ]


@case_source("tabix.regions_file_base")
def regions_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F61", seed, twin, work, tools, "implicit", "free")
        prepare(work, tools)
        for template in regions_templates():
            label = label_of(task, tools.run([template.tool, *template.argv], work))
            decls = [{BASE_KEY: "zero_based_half_open"}]
            if label == "wrong":
                decls.append({BASE_KEY: "one_based_closed"})
            emit(out, template, label, work, "F61", seed, twin, decls)
    return out


# ---- tabix.region_strings_bed --------------------------------------------------------------------

COORD_KEY = "region_coordinates"


def window_strings(work: Path, shift: int) -> list[str]:
    rows = [line.split("\t") for line in (work / "windows.tsv").read_text().splitlines()]
    return [f"{c}:{int(s) + shift}-{e}" for c, s, e in rows]


@case_source("tabix.region_strings_bed")
def strings_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F61", seed, twin, work, tools, "implicit", "free")
        prepare(work, tools)
        templates = [
            Template(
                "raw",
                "tabix",
                ("intervals.bed.gz", *window_strings(work, 0)),
                "zero_based_half_open",
            ),
            Template(
                "raw_unique",
                "tabix",
                ("-u", "intervals.bed.gz", *window_strings(work, 0)),
                "zero_based_half_open",
            ),
            Template(
                "raw_header",
                "tabix",
                ("-h", "intervals.bed.gz", *window_strings(work, 0)),
                "zero_based_half_open",
            ),
            Template(
                "moved", "tabix", ("intervals.bed.gz", *window_strings(work, 1)), "one_based_closed"
            ),
            Template(
                "moved_unique",
                "tabix",
                ("-u", "intervals.bed.gz", *window_strings(work, 1)),
                "one_based_closed",
            ),
            Template(
                "moved_separate",
                "tabix",
                ("intervals.bed.gz", *window_strings(work, 1)),
                "one_based_closed",
            ),
        ]
        for template in templates:
            label = label_of(task, tools.run([template.tool, *template.argv], work))
            decls = [{COORD_KEY: "one_based_closed"}]
            if label == "wrong":
                decls.append({COORD_KEY: template.meaning})
            emit(out, template, label, work, "F61", seed, twin, decls)
    return out
