"""F25 doc_flag (flag_semantics, locus documentation): the notes swap the two quality letters of
`samtools coverage`, and the tool accepts either letter silently.

Source: samtools NEWS, release 1.12: "Fixed samtools coverage quality thresholding options which
were the wrong way round compared to mpileup (-q is the mapping quality threshold and -Q is base
quality)". `samtools depth` still has the opposite letters (-q base quality, -Q mapping quality), so
notes written for the older coverage (or by analogy with depth) name the wrong letter.

Verified by running the installed tools and, for the history, older releases on the same BAM
(chr1/chr2, 40 reads, MAPQ 3 or 60, base qualities 20 or 40; numreads column):

  samtools 1.24  `coverage -q 20`  chr1 8  chr2 13   (mapping-quality filter: only MAPQ 60 left)
  samtools 1.24  `coverage -Q 20`  chr1 20 chr2 20   (base-quality filter: no read is dropped)
  samtools 1.10  `coverage -Q 20`  mapping-quality filter (covbases 239 and 239, as 1.24 -q 20)
  samtools 1.10  `coverage -q 20`  base-quality filter (covbases 352 and 252, the unfiltered ones)
  samtools 1.24  `coverage --help` lists "-q, --min-MQ INT  mapping quality threshold [0]" and
                 "-Q, --min-BQ INT  base quality threshold [0]"

Both runs exit 0 with an empty stderr. A base-quality cutoff removes bases, never whole reads (every
read here keeps at least ten bases of Q40), so `numreads` is the total read count whatever the
cutoff: the naive answer is "no read was filtered", with covbases and meandepth still changing
(some bases are Q10 or Q20), which looks like a working filter.

Twin axis: `twin_files` is NOTES.md. Both twins hold the same BAM and the same notes except one
sentence: the faulted twin says -Q is the mapping-quality cutoff and -q the base-quality one (as in
`samtools depth`), the clean twin names them the right way round. Task: for each contig in header
order, how many reads does `samtools coverage` count when only alignments with MAPQ >= 30 are used.
Each contig holds 9-14 reads below 30 (including MAPQ 29) and reads at exactly 30, so the naive
list is wrong on both contigs and a `>` instead of `>=` would be caught.

  explicit  the README adds that the notes can be out of date and should be checked
  implicit  the README says nothing about the notes
  forced    the task sentence dictates `samtools coverage`
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": (500, 700), "chr2": (400, 600)}
READ_LEN = 50
CUTOFF = 30
LOW_MAPQ = (0, 1, 3, 10, 20, 29)  # below the cutoff
HIGH_MAPQ = (30, 35, 40, 42, 60)  # at or above it
GOOD_HEAD = 10  # leading bases that are always Q40, so no read vanishes under a base-quality cutoff
TWIN_FILES = frozenset({"NOTES.md"})

CHECK_SENTENCE = (
    "The notes may be out of date: check what they say against the installed tools (their --help "
    "output, or a small test on the data) before relying on it."
)


@dataclass(frozen=True)
class Read:
    name: str
    chrom: str
    pos: int  # 1-based
    flag: int
    mapq: int
    seq: str
    qual: str


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    reads: list[Read]

    def on(self, chrom: str) -> list[Read]:
        return [r for r in self.reads if r.chrom == chrom]

    def sam(self) -> str:
        lines = ["@HD\tVN:1.6\tSO:coordinate"]
        lines += [f"@SQ\tSN:{c}\tLN:{n}" for c, n in self.contigs.items()]
        lines += [
            f"{r.name}\t{r.flag}\t{r.chrom}\t{r.pos}\t{r.mapq}\t{READ_LEN}M\t*\t0\t0\t{r.seq}\t{r.qual}"
            for r in self.reads
        ]
        return "\n".join(lines) + "\n"


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f25-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    raw: list[tuple[int, int, int, int, str, str]] = []
    for index, length in enumerate(contigs.values()):
        n_low = rnd.randint(9, 14)
        n_high = rnd.randint(14, 24)
        mapqs = [rnd.choice(LOW_MAPQ) for _ in range(n_low)]
        mapqs += [rnd.choice(HIGH_MAPQ) for _ in range(n_high)]
        mapqs[0], mapqs[n_low] = 29, 30  # both sides of the cutoff are present
        rnd.shuffle(mapqs)
        for mapq in mapqs:
            seq = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
            tail = rnd.choices("I5+", weights=(6, 3, 1), k=READ_LEN - GOOD_HEAD)
            qual = "I" * GOOD_HEAD + "".join(tail)
            pos = rnd.randint(1, length - READ_LEN + 1)
            raw.append((index, pos, rnd.choice((0, 16)), mapq, seq, qual))
    raw.sort(key=lambda t: (t[0], t[1]))
    names = list(contigs)
    reads = [Read(f"r{i:03d}", names[c], pos, *rest) for i, (c, pos, *rest) in enumerate(raw)]
    return Plan(contigs, reads)


def truth_answers(plan: Plan) -> tuple[list[int], list[int]]:
    """(expected, naive): reads with MAPQ >= 30 per contig, and all reads per contig."""
    expected = [sum(r.mapq >= CUTOFF for r in plan.on(c)) for c in plan.contigs]
    naive = [len(plan.on(c)) for c in plan.contigs]
    return expected, naive


def notes_text(twin: str) -> str:
    if twin == "clean":
        letters = (
            "`-q INT` is the mapping-quality threshold and `-Q INT` the base-quality threshold "
            "(the opposite of `samtools depth`; the letters were swapped in samtools 1.12)."
        )
    else:
        letters = (
            "`-Q INT` is the mapping-quality threshold and `-q INT` the base-quality threshold "
            "(the same letters as `samtools depth`)."
        )
    lines = [
        "# Lab notes: coverage tables",
        "",
        "- One row per contig: `samtools coverage reads.bam`. The header line starts with "
        "`#rname`.",
        "- Columns: rname, startpos, endpos, numreads, covbases, coverage (%), meandepth, "
        "meanbaseq, meanmapq.",
        "- numreads is the number of alignments counted after the filters.",
        "- Rows follow the order of the BAM header.",
        "- Both quality thresholds default to 0. Our standard cutoff for a confidently placed read "
        "is mapping quality 30.",
        f"- Cutoff options of `samtools coverage`: {letters}",
        "- Unmapped, secondary, QC-fail and duplicate alignments are skipped by default.",
    ]
    return "\n".join(lines) + "\n"


def readme_lines(tier: str) -> list[str]:
    lines = [
        "reads.bam: alignments of a small run on chr1 and chr2 (coordinate-sorted, no index).",
        "NOTES.md: the lab's notes on coverage tables; the task says to follow them.",
    ]
    return [*lines, CHECK_SENTENCE] if tier == "explicit" else lines


def _write_bam(work: Path, sam_text: str, tools: ToolEnv, name: str = "reads") -> None:
    (work / f"{name}.sam").write_text(sam_text)
    argv = ["samtools", "view", "-b", "--no-PG", "-o", f"{name}.bam", f"{name}.sam"]
    common.check(tools.run(argv, work), argv)
    (work / f"{name}.sam").unlink()


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    _write_bam(work, plan.sam(), tools)
    (work / "NOTES.md").write_text(notes_text(twin))
    (work / "README.md").write_text(common.readme(readme_lines(tier)))
    if twin == "clean":
        evidence = (
            "NOTES.md names -q as the mapping-quality threshold, as `samtools coverage "
            "--help` does",
        )
    else:
        evidence = (
            "NOTES.md says -Q is the mapping-quality threshold; `samtools coverage --help` lists "
            "-q, --min-MQ as the mapping-quality threshold and -Q, --min-BQ as the base-quality "
            "one",
            "a small test: `samtools coverage -Q 30` leaves numreads at the total, `-q 30` "
            "lowers it",
        )
    prompt = common.prompt_for(
        route,
        "following the lab notes in NOTES.md, report for each contig of reads.bam, in the order "
        f"of the BAM header, how many reads are counted when only alignments with a mapping "
        f"quality of at least {CUTOFF} are used",
        "following the lab notes in NOTES.md, use `samtools coverage` to report for each contig "
        "of reads.bam, in the order of the BAM header, the number of reads it counts (the "
        f"numreads column) when only alignments with a mapping quality of at least {CUTOFF} are "
        "used",
        "a JSON list of integers, one per contig, in the order of the BAM header",
    )
    expected, naive_answer = truth_answers(plan)
    truth = {
        "kind": "int_list",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_answer,
    }
    task_id = common.task_id("F25", seed, twin, tier, route)
    return Task(task_id, "F25", "flag_semantics", twin, seed, prompt, truth, evidence, tier, route)


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _numreads(tools: ToolEnv, work: Path, *options: str) -> list[int]:
    proc = tools.run(["samtools", "coverage", *options, "reads.bam"], work)
    rows = [ln.split("\t") for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]
    return [int(r[3]) for r in rows]


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Ignores the notes: the long option names the filter, whatever the letters mean."""
    return _ok(_numreads(tools, work, "--min-MQ", str(CUTOFF)), "samtools coverage --min-MQ 30")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Uses the letter the notes give for the mapping-quality threshold."""
    text = (work / "NOTES.md").read_text()
    letter = re.search(r"`-([qQ]) INT` is the mapping-quality threshold", text)
    option = f"-{letter.group(1)}" if letter else "-q"
    return _ok(_numreads(tools, work, option, str(CUTOFF)), f"samtools coverage {option} 30")


# Exposure: a samtools coverage command (one pipeline stage) filtered with the short `-Q <n>`, the
# letter the stale notes give for mapping quality. The same stage naming `-q <n>`, --min-MQ or
# --help is not counted: it set the mapping-quality filter some other way or only looked. Checked
# stage by stage and per command, so a later correction does not undo an earlier trapped call.
STAGE_SPLIT = re.compile(r"[|;&\n]+")
COVERAGE = re.compile(r"\bsamtools\s+coverage\b|['\"]samtools['\"]\s*,\s*['\"]coverage['\"]")
SHORT_Q_UPPER = re.compile(r"(?<![\w-])-Q\s*=?\s*['\"]?\d|['\"]-Q['\"]\s*,\s*['\"]?\d")
SAFE = re.compile(
    r"(?<![\w-])-q\s*=?\s*['\"]?\d|['\"]-q['\"]\s*,\s*['\"]?\d|(?i:--min-?mq)|--help"
    r"|(?<![\w-])-h\b"
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for stage in STAGE_SPLIT.split(command):
            if COVERAGE.search(stage) and SHORT_Q_UPPER.search(stage) and not SAFE.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F25",
    slot="flag_semantics",
    source="samtools 1.12 NEWS: coverage -q/-Q were the wrong way round compared to mpileup",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="documentation",
)
