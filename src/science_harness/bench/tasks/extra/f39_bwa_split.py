"""F39 bwa_split (flag_semantics, documentation): `bwa mem -M` flags split hits 0x100, not 0x800.

For a chimeric read, `bwa mem` writes the longest part as the primary alignment and the other parts
as supplementary records (flag 0x800, SA tag on the primary). `-M` ("mark shorter split hits as
secondary", for Picard compatibility, in the GATK best-practice command) writes those parts with
flag 0x100 instead. Nothing marks the difference in the output: a count of supplementary alignments
(`samtools view -c -f 0x800`, flagstat's "supplementary" line) is simply 0 and secondary counts
(0x100) are inflated by the split parts. Both flag counts exit 0 with a plausible number.

Task: align reads.fastq the way the lab README says and report how many reads are chimeric (the
read has more than one alignment, a primary one and at least one more part). Same reads in both
twins; only the README differs. Clean twin: the lab command is `bwa mem -R ... genome.fa`. Faulted
twin: the lab command has `-M` (the GATK/Picard habit), so the parts of the chimeric reads carry
0x100 and a count of 0x800 records is 0.

Verified here (bwa 0.7.19, samtools 1.24; 40 chimeric reads of 70+50 bases across two loci and 40
ordinary reads of 120 bases): default -> flags 0:80, 2048:40; `-M` -> 0:80, 256:40; `-Y` -> 0:80,
2048:40. `samtools view -c -f 0x800` gives 40 and 0, `-c -f 0x100` gives 0 and 40. The SA tag is on
the primary record in both modes, so counting `-F 0x904` records with an SA tag is right either way.

  explicit  the README names the flag change (split hits flagged secondary, not supplementary)
  implicit  the README only gives the command
  forced    the sentence asks for `samtools view -c` with a flag filter; the flag is left to the
            agent (naming `-f 0x800` would contradict the -M of the faulted README)
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f29_mapq_unique import (
    fastq_text,
    rand_seq,
    revcomp,
    sam_records,
    write_genome,
)
from science_harness.bench.tasks.extra.f30_bwa_short import bwa_index
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": (2200, 2600), "chr2": (1800, 2200)}
FIRST, SECOND = 70, 50
SUPPLEMENTARY, SECONDARY = 0x800, 0x100
RG = "@RG\\tID:lane1\\tSM:sampleA"


def make_reads(seed: int) -> tuple[dict[str, str], list[tuple[str, str]], int]:
    rng = random.Random(f"f39-{seed}")
    genome = {c: rand_seq(rng, rng.randint(lo, hi)) for c, (lo, hi) in CONTIGS.items()}
    reads = []
    chimeric = rng.randint(15, 25)
    for i in range(chimeric + rng.randint(55, 75)):
        if i < chimeric:
            # the 70-base part on one contig, the 50-base part on the other, no overlap by design
            (c1, c2) = rng.sample(list(CONTIGS), 2)
            a = rng.randint(0, len(genome[c1]) - FIRST)
            b = rng.randint(0, len(genome[c2]) - SECOND)
            seq = genome[c1][a : a + FIRST] + genome[c2][b : b + SECOND]
        else:
            c = rng.choice(list(CONTIGS))
            a = rng.randint(0, len(genome[c]) - FIRST - SECOND)
            seq = genome[c][a : a + FIRST + SECOND]
        reads.append((f"read{i:03d}", revcomp(seq) if rng.random() < 0.5 else seq))
    rng.shuffle(reads)
    return genome, reads, chimeric


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    genome, reads, chimeric = make_reads(seed)
    write_genome(work, genome, tools)
    bwa_index(work, tools)
    (work / "reads.fastq").write_text(fastq_text(reads))
    explicit = tier == "explicit"
    flag = " -M" if twin == "faulted" else ""
    command = f"`bwa mem{flag} -R '{RG}' genome.fa reads.fastq`"
    lines = [
        "reads.fastq: single-end reads of 120 bases (FASTQ), some of them chimeric.",
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2; its bwa index "
        "(genome.fa.amb, .ann, .bwt, .pac, .sa) is in this directory.",
        f"Our standard alignment command: {command}",
    ]
    if explicit and twin == "faulted":
        lines[-1] += " (-M: shorter split hits are flagged secondary, not supplementary)."
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = (
            "the README command is bwa mem without -M: split hits are supplementary (0x800)",
        )
    else:
        evidence = (
            "the README command is bwa mem -M, which flags split hits as secondary (0x100)",
            "samtools flagstat of the output reports 0 supplementary and as many secondary records "
            "as there are split parts",
        )
    sentence = (
        "align reads.fastq the way the README says and report how many reads are chimeric (the "
        "read has a primary alignment and at least one more part elsewhere)"
    )
    prompt = common.prompt_for(
        route,
        sentence,
        "align reads.fastq the way the README says and use `samtools view -c` with a flag filter "
        "to report how many reads are chimeric (the read has a primary alignment and at least "
        "one more part elsewhere)",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": chimeric, "naive": chimeric if twin == "clean" else 0}
    task_id = common.task_id("F39", seed, twin, tier, route)
    return Task(task_id, "F39", "flag_semantics", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def readme_options(work: Path) -> list[str]:
    """The bwa mem options of the README command (-M or not, and the read group)."""
    text = (work / "README.md").read_text()
    match = re.search(r"`bwa mem([^`]*?) genome\.fa reads\.fastq`", text)
    argv = ["-M"] if match and " -M" in match.group(1) else []
    return [*argv, "-R", RG]


def align(work: Path, tools: ToolEnv, out: str) -> None:
    argv = ["bwa", "mem", "-t", "1", *readme_options(work), "genome.fa", "reads.fastq"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    (work / out).write_text(proc.stdout)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    align(work, tools, "ref.sam")
    chimeric = {
        f[0]
        for f in sam_records((work / "ref.sam").read_text())
        if not int(f[1]) & 0x904 and any(x.startswith("SA:Z:") for x in f[11:])
    }
    return _result(len(chimeric), "bwa mem as the README, primary records with an SA tag")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    align(work, tools, "naive.sam")
    proc = tools.run(["samtools", "view", "-c", "-f", "0x800", "naive.sam"], work)
    common.check(proc, ["samtools", "view", "-c"])
    return _result(int(proc.stdout.strip()), "samtools view -c -f 0x800 (supplementary records)")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped, on an alignment made by bwa mem -M (the -M is in the command or in the README command of
# the workspace): a samtools view with a positive -f that has the 0x800 bit (2048, 0x801, ...),
# whatever -F or the rest of the call says, since it selects nothing; or one that excludes only
# 0x800 with -F (no 0x100 in the -F value) and does not also select 0x100 or test the SA tag. The
# same two tests as the rule samtools.split_flag_after_bwa_m. Every -f / -F of the call is read, the
# long spellings included.
VIEW = re.compile(r"\bsamtools\s+view\b([^|;&\n]*)")
REQUIRED = re.compile(r"\s(?:-[A-Za-z]*f|--(?:incl|require)-flags[=\s])\s*(0[xX][0-9a-fA-F]+|\d+)")
EXCLUDED = re.compile(r"\s(?:-[A-Za-z]*F|--excl(?:ude)?-flags[=\s])\s*(0[xX][0-9a-fA-F]+|\d+)")
SA_TAG = re.compile(r"\bSA:Z\b|\bSA\b")
BWA_M = re.compile(r"\bbwa\s+mem\b[^|;&\n]*\s-[A-Za-z]*M\b")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def _flag(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError:  # "0400": samtools reads a leading zero as decimal here
        return int(text)


def _readme_has_m(work: Path) -> bool:
    try:
        return " -M" in (work / "README.md").read_text()
    except OSError:
        return False


def _flag_view(args: str, sa_tag: bool) -> bool:
    """Does the argument text of one `samtools view` take the trapped route? `sa_tag`: the command
    around it also tests the SA tag."""
    wanted = [_flag(v) for v in REQUIRED.findall(args)]
    excluded = [_flag(v) for v in EXCLUDED.findall(args)]
    if any(v & SUPPLEMENTARY for v in wanted):
        return True
    lone = any(v & SUPPLEMENTARY and not v & SECONDARY for v in excluded)
    return lone and not sa_tag and not any(v & SECONDARY for v in wanted)


def exposed(commands: list[str], work: Path) -> bool:
    used_m = _readme_has_m(work) or any(BWA_M.search(c) for c in commands)
    if not used_m:
        return False
    for command in commands:
        for segment in SEGMENT.split(command):
            sa_tag = bool(SA_TAG.search(segment))
            if any(_flag_view(view.group(1), sa_tag) for view in VIEW.finditer(segment)):
                return True
    return False


SPEC = FamilySpec(
    id="F39",
    slot="flag_semantics",
    source="bwa mem manual (-M: mark shorter split hits as secondary); GATK best practices",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"README.md"}),
    locus="documentation",
)
