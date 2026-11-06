"""Admission cases for minimap2.short_reads_preset (family F35, slot default_param).

For one seed both twins of the F35 task are built (implicit tier): the same reads at the same
places, 100 bases long (clean) or cut to 47-52 bases (faulted). Each `minimap2` call is run for
real with SAM output; the SAM is turned into reads per contig and compared with
`task.truth["expected"]` (bench.score.answer_matches). The label never comes from the rule. Calls
without `-x sr` (the default, `-x map-ont`, `-x map-pb`, `-x splice`, ...) are right on the clean
twin and wrong on the faulted one; calls with `-x sr` are right on both.

Each call is emitted undeclared and with `keep_all` and `drop_short`.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template, emit, run_all
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f35_mm2_short import mapped_per_contig

KEY = "short_read_policy"


def _mm(name: str, *options: str) -> Template:
    return Template(name, "minimap2", ("-a", "-t", "1", *options, "genome.fa", "reads.fastq"), "")


TEMPLATES: tuple[Template, ...] = (
    # --- no short-read preset: reads under about 60 bases are lost
    _mm("default"),
    _mm("map_ont", "-x", "map-ont"),
    _mm("map_pb", "-x", "map-pb"),
    _mm("map_hifi", "-x", "map-hifi"),
    _mm("splice", "-x", "splice"),
    _mm("asm5", "-x", "asm5"),
    _mm("read_group", "-R", "@RG\\tID:a\\tSM:b"),
    _mm("soft_clip", "-Y"),
    _mm("secondary_off", "--secondary=no"),
    _mm("eqx", "--eqx"),
    _mm("k12", "-k", "12", "-w", "5"),
    # --- the short-read preset: every read of the family maps
    _mm("sr", "-x", "sr"),
    _mm("sr_attached", "-xsr"),
    _mm("sr_long_option", "--preset", "sr"),
    _mm("sr_eqx", "-x", "sr", "--eqx"),
    _mm("sr_secondary_off", "-x", "sr", "--secondary=no"),
    _mm("sr_read_group", "-x", "sr", "-R", "@RG\\tID:a\\tSM:b"),
)


@case_source("minimap2.short_reads_preset")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F35", seed, twin, work, tools, "implicit", "free")
        contigs = [
            line.split("\t")[0] for line in (work / "genome.fa.fai").read_text().splitlines()
        ]
        procs = run_all(tools, work, TEMPLATES)
        for template, proc in zip(TEMPLATES, procs, strict=True):
            label: Label = "wrong"
            if proc.returncode == 0:
                answer = mapped_per_contig(proc.stdout, contigs)
                label = "right" if answer_matches(task.truth, answer) else "wrong"
            emit(
                out,
                template,
                label,
                work,
                "F35",
                seed,
                twin,
                [{KEY: "keep_all"}, {KEY: "drop_short"}],
            )
    return out
