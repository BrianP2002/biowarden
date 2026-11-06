"""Admission cases for star.genecounts_column (family F31, slot strand).

For one seed both twins of the F31 task are built (implicit tier: the README only says
"strand-specific"): the same fragments as a forward-stranded library (clean, column 3 is right) and
as a dUTP library (faulted, column 4 is right). STAR is run for real in the workspace
(`--quantMode GeneCounts`, output `run_ReadsPerGene.out.tab`), then each call below is run for real
and its output, read as a gene -> count table, is compared with `task.truth["expected"]`
(bench.score.answer_matches). The label never comes from the rule. Calls that read column 2 are
wrong on both twins (pairs in the shared exon are N_ambiguous), column 3 is right only on the clean
twin, column 4 only on the faulted one, and a call that sums two columns is wrong on both.
Inspection
calls (head, wc, cat) print no table and cannot be wrong.

Each call is emitted without a declaration and with declarations: the column it reads, and the
column the data point to (a right declaration with botched code).
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template, emit, run_all
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f29_mapq_unique import star_map

KEY = "genecounts_column"
TABLE = "run_ReadsPerGene.out.tab"
NAMES = {2: "unstranded", 3: "first_read_strand", 4: "second_read_strand"}


def _awk(name: str, column: int | None, program: str) -> Template:
    meaning = NAMES.get(column, "none") if column else "none"
    return Template(name, "awk", (program, TABLE), meaning)


def _sh(name: str, column: int | None, pipeline: str) -> Template:
    meaning = NAMES.get(column, "none") if column else "none"
    return Template(name, "bash", ("-c", pipeline), meaning)


PY_HEAD = (
    "rows = [l.split('\\t') for l in open('" + TABLE + "').read().splitlines()]\n"
    "rows = [r for r in rows if not r[0].startswith('N_')]\n"
)
PY_COLUMN = {n: PY_HEAD + f"print({{r[0]: int(r[{n - 1}]) for r in rows}})\n" for n in (2, 3, 4)}

TEMPLATES: tuple[Template, ...] = (
    # --- unstranded column: wrong on both twins
    _awk("awk_col2", 2, "!/^N_/ {print $1, $2}"),
    _sh("cut_col2", 2, f"cut -f1,2 {TABLE} | grep -v '^N_'"),
    _sh("tail_col2", 2, f"tail -n +5 {TABLE} | awk '{{print $1,$2}}'"),
    Template("py_col2", "python3", ("-c", PY_COLUMN[2]), "unstranded"),
    # --- first-read-strand column: right on the clean (forward) twin only
    _awk("awk_col3", 3, "!/^N_/ {print $1, $3}"),
    _sh("cut_col3", 3, f"cut -f1,3 {TABLE} | grep -v '^N_'"),
    _sh("tail_col3", 3, f"tail -n +5 {TABLE} | cut -f 1,3"),
    Template("py_col3", "python3", ("-c", PY_COLUMN[3]), "first_read_strand"),
    # --- second-read-strand column: right on the faulted (dUTP) twin only
    _awk("awk_col4", 4, "!/^N_/ {print $1, $4}"),
    _sh("cut_col4", 4, f"cut -f1,4 {TABLE} | grep -v '^N_'"),
    _sh("awk_col4_tail", 4, f"tail -n +5 {TABLE} | awk -v OFS='\\t' '{{print $1, $4}}'"),
    Template("py_col4", "python3", ("-c", PY_COLUMN[4]), "second_read_strand"),
    # --- other reads of the table
    _awk("awk_sum_3_4", None, "!/^N_/ {print $1, $3 + $4}"),
    _awk("awk_max_3_4", None, "!/^N_/ {m = $3 > $4 ? $3 : $4; print $1, m}"),
    _sh("cut_2_3_4", None, f"cut -f1-4 {TABLE} | grep -v '^N_'"),
    # --- inspection: no table of counts, cannot be wrong
    _sh("head", None, f"head -8 {TABLE}"),
    _sh("wc", None, f"wc -l {TABLE}"),
    _sh("cat_ambiguous_row", None, f"grep N_ambiguous {TABLE}"),
    _sh("sort_view", None, f"sort -k2,2nr {TABLE} | head -3"),
)

INSPECTION = {"head", "wc", "cat_ambiguous_row", "sort_view"}


def _table(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        parts = line.replace("\t", " ").split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _python_table(stdout: str) -> dict[str, int]:
    text = stdout.strip()
    if not text.startswith("{"):
        return {}
    try:
        return {k: int(v) for k, v in eval(text, {"__builtins__": {}}).items()}  # noqa: S307
    except Exception:
        return {}


def _label(task, template: Template, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    if template.name in INSPECTION:
        return "right"
    table = _python_table(proc.stdout) if template.tool == "python3" else _table(proc.stdout)
    table = {g: v for g, v in table.items() if not g.startswith("N_")}
    return "right" if answer_matches(task.truth, table) else "wrong"


@case_source("star.genecounts_column")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F31", seed, twin, work, tools, "implicit", "free")
        star_map(
            work,
            tools,
            ["reads_1.fastq", "reads_2.fastq"],
            "run_",
            ("--quantMode", "GeneCounts", "--outSAMtype", "None"),
        )
        right = NAMES[3] if twin == "clean" else NAMES[4]
        procs = run_all(tools, work, TEMPLATES)
        for template, proc in zip(TEMPLATES, procs, strict=True):
            label = _label(task, template, proc)
            if label == "right":
                declarations = [{KEY: right}]
            else:
                declarations = [{KEY: right}]
                if template.meaning in NAMES.values() and template.meaning != right:
                    declarations.append({KEY: template.meaning})
            emit(out, template, label, work, "F31", seed, twin, declarations)
    return out
