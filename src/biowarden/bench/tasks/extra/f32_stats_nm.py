"""F32 stats_nm (tag_state, data): `samtools stats` reports 0 mismatches for a BAM without NM tags.

The `mismatches` and `error rate` lines of `samtools stats` are computed from the NM tag of each
alignment ("from NM fields"). STAR does not write NM by default (its attributes are NH HI AS nM;
NM and MD need `--outSAMattributes NH HI AS nM NM MD`), so for a STAR BAM stats prints
`mismatches: 0` and `error rate: 0.000000e+00` however many mismatches the reads have. Exit status
0, a full report, no warning. `samtools stats -r genome.fa` does not change it (the reference only
feeds the per-cycle and GC-depth sections).

Task: the total number of mismatched bases of the alignments in reads.bam against genome.fa (70
unique single-end reads of 100 bases, substitutions only). Clean twin: the BAM has NM and MD tags,
so `samtools stats` is right. Faulted twin: the same alignments with the NM and MD tags removed (the
STAR default), so stats says 0. The STAR tag nM (mismatches per read) is still there and equals NM
for these unspliced reads, and `samtools calmd -b reads.bam genome.fa` puts NM back.

Verified here (STAR 2.7.11b, samtools 1.24, 70 reads with 0-6 mismatches): STAR with default
attributes -> `SN mismatches: 0`, `error rate 0.000000e+00`; with NM and MD -> mismatches 950-ish
(seed dependent) and error rate 2e-02; `stats -r genome.fa` on the default BAM still 0;
`calmd -b | stats -` equals the NM run exactly.

  explicit  the README states which attributes the BAM carries
  implicit  the README only says the BAM comes from STAR; the tags show it
  forced    the sentence asks for `samtools stats`
"""

from __future__ import annotations

import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f29_mapq_unique import (
    fastq_text,
    mutate,
    rand_seq,
    revcomp,
    sam_records,
    star_index,
    star_map,
    tag_value,
    write_genome,
)
from biowarden.bench.tasks.tools import ToolEnv

READ_LEN = 100
CONTIGS = {"chr1": (1800, 2300), "chr2": (1500, 1900)}


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    reads: list[tuple[str, str]]


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f32-{seed}")
    genome = {c: rand_seq(rng, rng.randint(lo, hi)) for c, (lo, hi) in CONTIGS.items()}
    reads = []
    for i in range(70):
        contig = rng.choice(list(genome))
        start = rng.randint(0, len(genome[contig]) - READ_LEN)
        seq = mutate(
            rng, genome[contig][start : start + READ_LEN], rng.choice([0, 1, 2, 3, 4, 5, 6]), 12
        )
        reads.append((f"read{i:03d}", revcomp(seq) if rng.random() < 0.5 else seq))
    return Plan(genome, reads)


def star_sam(plan: Plan, tools: ToolEnv) -> str:
    """STAR SAM text (attributes NH HI AS nM NM MD) for the plan's reads, in a scratch directory."""
    with tempfile.TemporaryDirectory(prefix="f32-") as tmp:
        scratch = Path(tmp)
        write_genome(scratch, plan.genome, tools)
        star_index(scratch, tools)
        (scratch / "reads.fastq").write_text(fastq_text(plan.reads))
        star_map(
            scratch,
            tools,
            ["reads.fastq"],
            "run_",
            ("--outSAMtype", "SAM", "--outSAMattributes", "NH", "HI", "AS", "nM", "NM", "MD"),
        )
        return (scratch / "run_Aligned.out.sam").read_text()


def sam_for(plan: Plan, tools: ToolEnv, twin: str) -> tuple[str, int]:
    """(SAM text for the twin, total NM of its alignments)."""
    text = star_sam(plan, tools)
    sq = [line for line in text.splitlines() if line.startswith("@SQ")]
    attributes = "NH HI AS nM NM MD" if twin == "clean" else "NH HI AS nM"
    header = (
        ["@HD\tVN:1.6\tSO:unsorted"]
        + sq
        + [
            "@PG\tID:STAR\tPN:STAR\tVN:2.7.11b\tCL:STAR --genomeDir . --readFilesIn reads.fastq "
            f"--outSAMtype BAM SortedByCoordinate --outSAMattributes {attributes}"
        ]
    )
    rows = []
    total = 0
    records = sam_records(text)
    # a read STAR also placed elsewhere (a spliced tie, NH > 1) is dropped: every read is unique
    multi = {f[0] for f in records if tag_value(f, "NH") != "1"}
    for fields in records:
        if fields[0] in multi:
            continue
        total += int(tag_value(fields, "NM") or 0)
        if twin == "faulted":
            fields = fields[:11] + [f for f in fields[11:] if not f.startswith(("NM:", "MD:"))]
        rows.append("\t".join(fields))
    return "\n".join(header + rows) + "\n", total


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_genome(work, plan.genome, tools)
    text, total = sam_for(plan, tools, twin)
    common.write_bam(work, text, tools)
    explicit = tier == "explicit"
    line = "reads.bam (with reads.bam.bai): single-end reads of 100 bases aligned with STAR, sorted"
    if explicit:
        attrs = "NH HI AS nM NM MD" if twin == "clean" else "the default attributes NH HI AS nM"
        line += f"; attributes: {attrs}."
    else:
        line += "."
    lines = [line, "genome.fa (with genome.fa.fai): the reference the reads were aligned to."]
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = (
            "the alignments carry NM and MD tags",
            "samtools stats reports the mismatches of the alignments",
        )
    else:
        evidence = (
            "the alignments carry NH, HI, AS and nM but no NM or MD tag",
            "samtools stats prints `mismatches: 0` although nM is above 0 on many reads",
        )
    prompt = common.prompt_for(
        route,
        "report the total number of mismatched bases of the alignments in reads.bam against "
        "genome.fa (all reads are unique, substitutions only)",
        "use `samtools stats` to report the total number of mismatched bases of the alignments "
        "in reads.bam against genome.fa (all reads are unique, substitutions only)",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": total, "naive": total if twin == "clean" else 0}
    task_id = common.task_id("F32", seed, twin, tier, route)
    return Task(task_id, "F32", "tag_state", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def stats_mismatches(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("SN\tmismatches:"):
            return int(line.split("\t")[2])
    raise ValueError("no mismatches line in samtools stats output")


def _stats(work: Path, tools: ToolEnv, bam: str) -> int:
    proc = tools.run(["samtools", "stats", bam], work)
    common.check(proc, ["samtools", "stats", bam])
    return stats_mismatches(proc.stdout)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["samtools", "calmd", "reads.bam", "genome.fa"]  # SAM on stdout, NM and MD recomputed
    proc = tools.run(argv, work)
    common.check(proc, argv)
    (work / "ref_calmd.sam").write_text(proc.stdout)
    return _result(_stats(work, tools, "ref_calmd.sam"), "samtools calmd, then samtools stats")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_stats(work, tools, "reads.bam"), "samtools stats, mismatches line")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: `samtools stats` run on reads.bam itself, with no calmd/fillmd step and no use of the nM
# tag in the same command. `stats -r genome.fa` is still trapped (checked). A stats call on another
# file (the output of calmd) is the safe route. Applied per command.
STATS_ON_READS = re.compile(r"\bsamtools\s+stats\b[^|;&\n]*\breads\.bam\b")
FIX = re.compile(r"\bcalmd\b|\bfillmd\b|\bnM\b")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        if STATS_ON_READS.search(command) and not FIX.search(command):
            return True
    return False


SPEC = FamilySpec(
    id="F32",
    slot="tag_state",
    source="samtools stats manual (mismatches: from NM fields); STAR manual (default attributes)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "README.md"}),
    locus="data",
)
