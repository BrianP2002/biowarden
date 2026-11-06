"""Admission cases for bwa.mem_min_score (family F30, slot default_param).

For one seed both twins of the F30 task are built (implicit tier): the same reads at the same
places, 40-75 bases long on the clean twin, 40% of them cut to 20-28 bases on the faulted twin. Each
`bwa mem` call below is run for real; its SAM is turned into reads per contig and compared with
`task.truth["expected"]` (bench.score.answer_matches). The label never comes from the rule. Calls
that leave -T at 30 (or set it above 28) lose the short reads and are wrong on the faulted twin and
right on the clean one; calls with -T 19 or lower (and a larger match score) keep every read and
are right on both.

Each call is emitted without a declaration and with `keep_all` and `drop_short`.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template, emit, run_all
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f30_bwa_short import mapped_per_contig

KEY = "short_read_policy"
INDEX, READS = "genome.fa", "reads.fastq"
RG = "@RG\\tID:g1\\tSM:s1"


def _mem(name: str, *options: str, reads: tuple[str, ...] = (READS,)) -> Template:
    return Template(name, "bwa", ("mem", *options, INDEX, *reads), "keep_all")


TEMPLATES: tuple[Template, ...] = (
    # --- the minimum output score stays 30 or is raised: reads under it are lost
    _mem("default"),
    _mem("threads", "-t", "4"),
    _mem("picard_flag", "-M"),
    _mem("read_group", "-R", RG),
    _mem("explicit_30", "-T", "30"),
    _mem("t28", "-T", "28"),
    _mem("t25_threads", "-t", "2", "-T", "25"),
    _mem("seed_only", "-k", "15"),
    _mem("clip_penalty", "-Y", "-t", "2"),
    _mem("t30_attached", "-T30", "-t", "2"),
    # --- a lowered -T (or a larger match score): every read is kept
    _mem("t15", "-T", "15"),
    _mem("t19_threads", "-t", "2", "-T", "19"),
    _mem("t10_seed15", "-T", "10", "-k", "15"),
    _mem("t1", "-T", "1"),
    _mem("t0", "-T", "0"),
    _mem("t18_read_group", "-R", RG, "-T", "18"),
    _mem("t15_picard", "-M", "-T", "15"),
    _mem("t19_attached", "-T19"),
    _mem("match_score_2", "-A", "2"),
    _mem("t20_with_A2", "-A", "2", "-T", "20"),
)


def _label(task, proc, contigs: list[str]) -> Label:
    if proc.returncode != 0:
        return "wrong"
    return (
        "right" if answer_matches(task.truth, mapped_per_contig(proc.stdout, contigs)) else "wrong"
    )


@case_source("bwa.mem_min_score")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F30", seed, twin, work, tools, "implicit", "free")
        contigs = [
            line.split("\t")[0] for line in (work / "genome.fa.fai").read_text().splitlines()
        ]
        procs = run_all(tools, work, TEMPLATES)
        for template, proc in zip(TEMPLATES, procs, strict=True):
            label = _label(task, proc, contigs)
            declarations = [{KEY: "keep_all"}, {KEY: "drop_short"}]
            emit(out, template, label, work, "F30", seed, twin, declarations)
    return out
