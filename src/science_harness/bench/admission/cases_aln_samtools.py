"""Admission cases for the samtools rules of the alignment pack.

Every rule gets its calls from the benchmark workspaces of its family, built for both twins; the
preparatory steps the agent would run (an aligner, a sort) are run here with the real tools, then
each call is run for real and its output is compared with `task.truth` (bench.score.answer_matches).
Labels never come from a rule.

samtools.mapq_unique_cutoff (family F29, implicit tier): STAR aligns reads.fastq to `aln_*`; the
calls are `samtools view -c` with MAPQ cutoffs and flags. A cutoff of 1-3 keeps multi-mapped reads
(STAR gives them MAPQ 3 and 1), so those calls are right on the clean twin (no multi-mapped reads)
and wrong on the faulted one; a cutoff of 4 or more, `-d NH:1` and `-q 255` are right on both;
calls without a cutoff that count every record or every primary record are wrong on the faulted twin
for another reason, which the rule is not meant to see. Each call is emitted without a declaration
and with `unique_only` and `quality_only`.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

# ---- shared helpers for the alignment-pack case modules ------------------------------------------


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    meaning: str  # the value of the declaration this call's author would give it
    note: str = ""


def run_template(tools: ToolEnv, work: Path, template: Template):
    return tools.run([template.tool, *template.argv], work)


def run_all(tools: ToolEnv, work: Path, templates: Sequence[Template]):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(partial(run_template, tools, work), templates))


def int_label(task: Any, proc: Any) -> Label:
    """Right iff the call printed one integer equal to the truth."""
    if proc.returncode != 0:
        return "wrong"
    text = proc.stdout.strip()
    if not text.lstrip("-").isdigit():
        return "wrong"
    return "right" if answer_matches(task.truth, int(text)) else "wrong"


def emit(
    out: list[Case],
    template: Template,
    label: Label,
    work: Path,
    family: str,
    seed: int,
    twin: str,
    declarations: Sequence[dict[str, str]],
) -> None:
    """The call undeclared, then once per declaration."""
    calls: list[tuple[bool, dict[str, str]]] = [(False, {})] + [(True, d) for d in declarations]
    for declared, decl in calls:
        call = Call(template.tool, template.argv, work, decl)
        out.append(Case(call, label, declared, family, seed, twin, template.name))


# ---- samtools.mapq_unique_cutoff -----------------------------------------------------------------

MAPQ_KEY = "mapq_filter_goal"
BAM = "aln_Aligned.out.bam"
SAM = "aln_Aligned.out.sam"


def _view(name: str, goal: str, *args: str, target: str = BAM) -> Template:
    return Template(name, "samtools", ("view", *args, target), goal)


MAPQ_TEMPLATES: tuple[Template, ...] = (
    # --- cutoffs that keep STAR's multi-mapped reads (MAPQ 3 and 1)
    _view("q1_primary", "unique_only", "-c", "-q", "1", "-F", "0x100"),
    _view("q1_records", "unique_only", "-c", "-q", "1"),
    _view("q2_primary", "unique_only", "-c", "-q", "2", "-F", "256"),
    _view("q3_primary", "unique_only", "-c", "-q", "3", "-F", "0x100"),
    _view("min_mq1_primary", "unique_only", "-c", "--min-MQ", "1", "-F", "0x100"),
    _view("bundled_q1", "unique_only", "-cq1", "-F", "0x100"),
    _view("q1_sam", "unique_only", "-c", "-q", "1", "-F", "0x100", target=SAM),
    # --- cutoffs above every multi-mapped MAPQ, or other ways to say unique: right on both
    _view("q4_primary", "unique_only", "-c", "-q", "4", "-F", "0x100"),
    _view("q10_primary", "unique_only", "-c", "-q", "10", "-F", "0x100"),
    _view("q10_records", "unique_only", "-c", "-q", "10"),
    _view("q30_primary", "unique_only", "-c", "-q", "30", "-F", "0x100"),
    _view("q255", "unique_only", "-c", "-q", "255"),
    _view("attached_q20", "unique_only", "-c", "-q20", "-F", "0x100"),
    _view("long_min_mq", "unique_only", "-c", "--min-MQ=10", "-F", "0x100"),
    _view("q4_sam", "unique_only", "-c", "-q", "4", "-F", "0x100", target=SAM),
    _view("nh1_tag", "unique_only", "-c", "-d", "NH:1"),
    _view("q1_and_nh1", "unique_only", "-c", "-q", "1", "-d", "NH:1"),
    _view("expr_nh", "unique_only", "-c", "-e", "[NH]==1"),
    # --- wrong for another reason: no cutoff at all (the rule does not look)
    _view("all_records", "unique_only", "-c"),
    _view("primary_only", "unique_only", "-c", "-F", "0x100"),
    _view("mapped_primary", "unique_only", "-c", "-F", "0x104"),
)


@case_source("samtools.mapq_unique_cutoff")
def mapq_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    from science_harness.bench.tasks.extra.f29_mapq_unique import star_map

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F29", seed, twin, work, tools, "implicit", "free")
        star_map(work, tools, ["reads.fastq"], "aln_", ("--outSAMtype", "BAM", "Unsorted"))
        star_map(work, tools, ["reads.fastq"], "aln_", ("--outSAMtype", "SAM"))
        procs = run_all(tools, work, MAPQ_TEMPLATES)
        for template, proc in zip(MAPQ_TEMPLATES, procs, strict=True):
            label = int_label(task, proc)
            if label == "right":
                declarations = [{MAPQ_KEY: "unique_only"}, {MAPQ_KEY: "quality_only"}]
            else:
                declarations = [{MAPQ_KEY: "unique_only"}, {MAPQ_KEY: "quality_only"}]
            emit(out, template, label, work, "F29", seed, twin, declarations)
    return out


# ---- samtools.stats_nm_missing -------------------------------------------------------------------
# Family F32 (implicit tier): reads.bam from STAR, with NM and MD (clean) or without (faulted).
# The calls are `samtools stats` variants; a call is right when its `SN mismatches:` line equals the
# truth (the total NM of the alignments). A BAM that went through `samtools calmd` has NM on both
# twins, so stats on it is right on both; stats on reads.bam is right only on the clean twin; a
# region restricted stats is wrong on both for another reason.

STATS_KEY = "stats_mismatches"


def _stats(name: str, *args: str) -> Template:
    return Template(name, "samtools", ("stats", *args), "not_used")


STATS_TEMPLATES: tuple[Template, ...] = (
    _stats("default", "reads.bam"),
    _stats("threads", "-@", "2", "reads.bam"),
    _stats("filter_secondary", "-F", "0x100", "reads.bam"),
    _stats("with_reference", "-r", "genome.fa", "reads.bam"),
    _stats("long_reference", "--ref-seq", "genome.fa", "reads.bam"),
    _stats("remove_dups", "-d", "reads.bam"),
    _stats("insert_size", "-i", "2000", "reads.bam"),
    _stats("coverage", "-c", "1,500,1", "reads.bam"),
    _stats("calmd_bam", "reads_calmd.bam"),
    _stats("calmd_sam", "reads_calmd.sam"),
    _stats("calmd_with_reference", "-r", "genome.fa", "reads_calmd.bam"),
    _stats("region_chr1", "reads.bam", "chr1"),
)


def _stats_label(task: Any, proc: Any) -> Label:
    if proc.returncode != 0:
        return "wrong"
    for line in proc.stdout.splitlines():
        if line.startswith("SN\tmismatches:"):
            return "right" if answer_matches(task.truth, int(line.split("\t")[2])) else "wrong"
    return "wrong"


@case_source("samtools.stats_nm_missing")
def stats_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F32", seed, twin, work, tools, "implicit", "free")
        for pipeline in (
            "samtools calmd -b reads.bam genome.fa > reads_calmd.bam",
            "samtools calmd reads.bam genome.fa > reads_calmd.sam",
        ):
            proc = tools.run(["bash", "-c", pipeline], work)
            if proc.returncode != 0:
                raise RuntimeError(f"{pipeline}: {proc.stderr}")
        procs = run_all(tools, work, STATS_TEMPLATES)
        for template, proc in zip(STATS_TEMPLATES, procs, strict=True):
            label = _stats_label(task, proc)
            emit(out, template, label, work, "F32", seed, twin, [{STATS_KEY: "not_used"}])
    return out


# ---- samtools.fixmate_grouping -------------------------------------------------------------------
# Family F36 (implicit tier): reads.bam with duplicate pairs, mates adjacent (clean) or scattered
# (faulted), header SO:unsorted. Each `samtools fixmate -m` call is run for real and followed by the
# rest of the workflow (sort, markdup); the number of reads with flag 0x400 is compared with the
# truth. Fixmate on reads.bam is right on the clean twin and wrong on the faulted one; fixmate on a
# name-sorted or collated copy is right on both. The rule has no declaration: right calls are
# emitted with declared=True and declared=False alike (nothing to declare), wrong calls likewise.


def _fixmate(name: str, source: str, *options: str) -> Template:
    return Template(name, "samtools", ("fixmate", "-m", *options, source, f"o_{name}.bam"), "")


FIXMATE_TEMPLATES: tuple[Template, ...] = (
    _fixmate("grouped_input", "reads.bam"),
    _fixmate("grouped_threads", "reads.bam", "-@", "2"),
    _fixmate("grouped_bam_format", "reads.bam", "-O", "bam"),
    _fixmate("grouped_long_format", "reads.bam", "--output-fmt", "bam"),
    _fixmate("grouped_template_coord", "reads.bam", "-c"),
    _fixmate("grouped_no_proper_check", "reads.bam", "-p"),
    _fixmate("sort_n_first", "names.bam"),
    _fixmate("sort_n_threads", "names.bam", "-@", "2"),
    _fixmate("collated", "coll.bam"),
    _fixmate("collated_remove", "coll.bam", "-r"),
)


def _duplicates_label(task: Any, tools: ToolEnv, work: Path, template: Template) -> Label:
    out = template.argv[-1]
    stem = out[:-4]
    steps = (
        [template.tool, *template.argv],
        ["samtools", "sort", "-o", f"{stem}_s.bam", out],
        ["samtools", "markdup", f"{stem}_s.bam", f"{stem}_m.bam"],
    )
    for argv in steps:
        if tools.run(argv, work).returncode != 0:
            return "wrong"
    proc = tools.run(["samtools", "view", "-c", "-f", "0x400", f"{stem}_m.bam"], work)
    return int_label(task, proc)


@case_source("samtools.fixmate_grouping")
def fixmate_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F36", seed, twin, work, tools, "implicit", "free")
        for argv in (
            ["samtools", "sort", "-n", "-o", "names.bam", "reads.bam"],
            ["samtools", "collate", "-o", "coll.bam", "reads.bam"],
        ):
            proc = tools.run(argv, work)
            if proc.returncode != 0:
                raise RuntimeError(f"{argv}: {proc.stderr}")
        for template in FIXMATE_TEMPLATES:
            label = _duplicates_label(task, tools, work, template)
            for declared in (True, False):  # no declaration exists: both cells hold the same calls
                call = Call(template.tool, template.argv, work, {})
                out.append(Case(call, label, declared, "F36", seed, twin, template.name))
    return out


# ---- samtools.merge_combine_rg -------------------------------------------------------------------
# Family F37 (implicit tier): lane1.bam (sampleA, RG ID 1) and lane2.bam (sampleB, RG ID 2 on the
# clean twin, ID 1 on the faulted one). Each `samtools merge` call is run for real; the merged
# header's @RG lines and the RG tag of every record give reads per sample, compared with the truth.
# With -c the lanes of the faulted twin collapse into sampleA (wrong); with -c on the clean twin,
# and
# without -c on both, every sample survives (right). No declaration exists for this rule: the same
# call is emitted in both cells.


def _merge(name: str, *args: str) -> Template:
    return Template(name, "samtools", ("merge", *args), "")


MERGE_TEMPLATES: tuple[Template, ...] = (
    _merge("combine", "-c", "-f", "m_combine.bam", "lane1.bam", "lane2.bam"),
    _merge("combine_bundled", "-cf", "m_bundled.bam", "lane1.bam", "lane2.bam"),
    _merge("combine_threads", "-c", "-@", "2", "-f", "m_threads.bam", "lane1.bam", "lane2.bam"),
    _merge("combine_o_option", "-c", "-f", "-o", "m_o.bam", "lane1.bam", "lane2.bam"),
    _merge("combine_pg", "-c", "-p", "-f", "m_pg.bam", "lane1.bam", "lane2.bam"),
    _merge("combine_list", "-c", "-f", "-b", "lanes.txt", "m_list.bam"),
    _merge("combine_no_pg", "-c", "-f", "--no-PG", "m_nopg.bam", "lane1.bam", "lane2.bam"),
    _merge("plain", "-f", "m_plain.bam", "lane1.bam", "lane2.bam"),
    _merge("plain_threads", "-@", "2", "-f", "m_plain2.bam", "lane1.bam", "lane2.bam"),
    _merge("attach_rg", "-r", "-f", "m_r.bam", "lane1.bam", "lane2.bam"),
    _merge("plain_list", "-f", "-b", "lanes.txt", "m_plain_list.bam"),
    _merge("plain_u", "-u", "-f", "m_u.bam", "lane1.bam", "lane2.bam"),
)


def _output_of(template: Template) -> str:
    args = template.argv
    if "-o" in args:
        return args[args.index("-o") + 1]
    return next(a for a in args if a.startswith("m_") and a.endswith(".bam"))


@case_source("samtools.merge_combine_rg")
def merge_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    from science_harness.bench.tasks.extra.f37_merge_rg import reads_per_sample

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F37", seed, twin, work, tools, "implicit", "free")
        (work / "lanes.txt").write_text("lane1.bam\nlane2.bam\n")
        for template in MERGE_TEMPLATES:
            proc = tools.run([template.tool, *template.argv], work)
            label: Label = "wrong"
            if proc.returncode == 0:
                answer = reads_per_sample(work, tools, _output_of(template))
                label = "right" if answer_matches(task.truth, answer) else "wrong"
            for declared in (True, False):  # no declaration exists: both cells hold the same calls
                call = Call(template.tool, template.argv, work, {})
                out.append(Case(call, label, declared, "F37", seed, twin, template.name))
    return out


# ---- samtools.idxstats_unmapped ------------------------------------------------------------------
# Family F38 (implicit tier): reads.bam with unmapped reads only on the `*` row (clean) or partly
# beside their mapped mates (faulted). The calls are `samtools idxstats` and pipelines reading its
# output; a call is right when the number it prints equals the unmapped total. The rule looks
# only at
# the idxstats call: it fails when the call declares star_row_only (and the table has placed
# unmapped reads), warns otherwise. Pipelines run through `bash -c` are labelled by their output and
# emitted as the plain idxstats call they contain.

IDX_KEY = "idxstats_unmapped"
IDX_PIPELINES: tuple[tuple[str, str, str], ...] = (
    # (name, pipeline reading the table, what the author reads: contig_rows_included|star_row_only)
    ("star_grep", "samtools idxstats reads.bam | grep '^\\*' | cut -f4", "star_row_only"),
    ("star_tail", "samtools idxstats reads.bam | tail -n 1 | cut -f4", "star_row_only"),
    ("star_awk", "samtools idxstats reads.bam | awk '$1==\"*\" {print $4}'", "star_row_only"),
    ("star_end", "samtools idxstats reads.bam | awk 'END {print $4}'", "star_row_only"),
    (
        "sum_col4",
        "samtools idxstats reads.bam | awk '{u += $4} END {print u}'",
        "contig_rows_included",
    ),
    (
        "sum_cut_paste",
        "samtools idxstats reads.bam | cut -f4 | paste -sd+ | bc",
        "contig_rows_included",
    ),
    ("view_flag4", "samtools view -c -f 4 reads.bam", "contig_rows_included"),
    (
        "flagstat_mapped",
        "samtools flagstat reads.bam | awk 'NR==1 {t=$1} NR==7 {print t-$1}'",
        "contig_rows_included",
    ),
)


@case_source("samtools.idxstats_unmapped")
def idxstats_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F38", seed, twin, work, tools, "implicit", "free")
        for name, pipeline, reads in IDX_PIPELINES:
            proc = tools.run(["bash", "-c", pipeline], work)
            label = int_label(task, proc)
            call_argv = ("idxstats", "reads.bam")
            template = Template(name, "samtools", call_argv, reads)
            declarations = [{IDX_KEY: reads}]
            if label == "wrong" or reads == "star_row_only":
                other = "contig_rows_included" if reads == "star_row_only" else "star_row_only"
                declarations.append({IDX_KEY: other})
            if label == "right":
                declarations = [{IDX_KEY: "contig_rows_included"}]
                if reads == "star_row_only":
                    declarations.append({IDX_KEY: "star_row_only"})  # right on this twin
            emit(out, template, label, work, "F38", seed, twin, declarations)
    return out


# ---- samtools.split_flag_after_bwa_m -------------------------------------------------------------
# Family F39 (implicit tier): reads.fastq aligned with the README command, which has -M on the
# faulted twin. 40-ish chimeric reads give one primary record with an SA tag and one split part:
# flag 0x800 without -M, 0x100 with it. The calls are `samtools view` counts on the BAM the aligner
# wrote (its @PG line keeps the bwa command). A count of 0x800 records is right on the clean twin
# and 0 on the faulted one; counting the SA tag on primaries is right on both; counting 0x100 is
# right only with -M. No declaration exists for this rule: the same call fills both cells.


def _count(name: str, *args: str, target: str = "aln.bam") -> Template:
    return Template(name, "samtools", ("view", "-c", *args, target), "")


SPLIT_TEMPLATES: tuple[Template, ...] = (
    _count("supplementary_hex", "-f", "0x800"),
    _count("supplementary_decimal", "-f", "2048"),
    _count("supplementary_attached", "-f0x800"),
    _count("supplementary_not_secondary", "-f", "0x800", "-F", "0x100"),
    _count("supplementary_sam", "-f", "0x800", target="aln.sam"),
    _count("drop_supplementary_only", "-F", "0x800"),
    _count("sa_tag_on_primaries", "-F", "0x904", "-e", "exists([SA])"),
    _count("sa_tag_on_primaries_sam", "-F", "0x904", "-e", "exists([SA])", target="aln.sam"),
    _count("secondary_count", "-f", "0x100"),
    _count("primary_only", "-F", "0x900"),
    _count("all_records"),
)


@case_source("samtools.split_flag_after_bwa_m")
def split_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    from science_harness.bench.tasks.extra.f39_bwa_split import align

    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F39", seed, twin, work, tools, "implicit", "free")
        align(work, tools, "aln.sam")
        proc = tools.run(["samtools", "view", "-b", "-o", "aln.bam", "aln.sam"], work)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        procs = run_all(tools, work, SPLIT_TEMPLATES)
        for template, proc in zip(SPLIT_TEMPLATES, procs, strict=True):
            label = int_label(task, proc)
            for declared in (True, False):  # no declaration exists: both cells hold the same calls
                call = Call(template.tool, template.argv, work, {})
                out.append(Case(call, label, declared, "F39", seed, twin, template.name))
    return out
