"""F26 doc_default (default_param, locus documentation): the notes say `bcftools mpileup` skips
bases below Q13 by default; since bcftools 1.13 its default minimum base quality is 1.

Source: the bcftools source (mpileup.c) of the installed 1.24 sets `mplp.min_baseQ = 1` and keeps
the old behaviour behind `--config 1.12` ("1.12 and earlier": `mplp.min_baseQ = 13`); the NEWS of
release 1.13 announces "significant changes in BAQ parametrization in `bcftools mpileup`. The
previous behavior can be triggered by providing the `--config 1.12` option". `samtools mpileup`
still defaults to -Q 13, and `samtools depth -q` to 0, so the three tools disagree.

Verified by running the installed 1.24 and the older 1.10.1 on one BAM (80 reads of 40 bases on two
contigs, base qualities 10 or 40, no BAQ: -B). Sum over chr1 of FORMAT/DP from `-a FORMAT/DP`:

  bcftools 1.10.1  (no -Q)   788      bcftools 1.10.1  -Q 1   1600     -Q 13   788
  bcftools 1.24    (no -Q)   1600     bcftools 1.24    -Q 1   1600     -Q 13   788
  bcftools 1.24 `mpileup --help`: "-Q, --min-BQ INT  Skip bases with baseQ/BAQ smaller than INT [1]"
  bcftools 1.24 INFO/DP is 1600 whatever -Q is (raw depth, before the base-quality filter); the
  FORMAT/DP, FORMAT/AD and INFO/I16 counts follow -Q. The python count of bases with Q >= 13 (788)
  matches the -Q 13 run, so the expected answer is computed from the reads alone (the tests check
  it against FORMAT/DP at every position).

Stderr of `bcftools mpileup` is the two informational lines "[mpileup] 1 samples in 1 input
files" and "[mpileup] maximum number of reads per input file set to -d 250", and the exit code is
0 in both routes: the default silently counts the low-quality bases.

Twin axis: `twin_files` is NOTES.md. The data (BAM, reference) are the same in both twins. The notes
give the recipe `bcftools mpileup -f ref.fa -B -a FORMAT/DP reads.bam`, name the lab standard (only
bases with Phred quality 13 or more count) and then say either that the default is -Q 13 (faulted:
nothing to pass) or that the default is -Q 1 and `-Q 13` must be passed (clean). Task: how many
positions have FORMAT/DP at or above a threshold chosen per seed (largest gap between the two
cutoffs). The naive faulted answer counts positions with the Q1 depth.

  explicit  the README adds that the notes can be out of date and should be checked
  implicit  the README says nothing about the notes
  forced    the task sentence dictates `bcftools mpileup`
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": (240, 320), "chr2": (180, 260)}
READS_PER_CONTIG = (30, 40)
READ_LEN = (40, 50)
STANDARD_Q = 13  # the lab standard: bases below this quality are not counted
DEFAULT_Q = 1  # bcftools mpileup's own default since 1.13
MIN_GAP = 15  # smallest wanted difference between the two counts
MIN_COUNT = 12  # smallest wanted count of positions that pass the standard
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
    seq: str
    quals: tuple[int, ...]  # Phred scores

    @property
    def qual(self) -> str:
        return "".join(chr(q + 33) for q in self.quals)


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    genome: dict[str, str]
    reads: list[Read]
    threshold: int  # depth asked for in the task

    def sam(self) -> str:
        lines = ["@HD\tVN:1.6\tSO:coordinate"]
        lines += [f"@SQ\tSN:{c}\tLN:{n}" for c, n in self.contigs.items()]
        lines += [
            f"{r.name}\t{r.flag}\t{r.chrom}\t{r.pos}\t60\t{len(r.seq)}M\t*\t0\t0\t{r.seq}\t{r.qual}"
            for r in self.reads
        ]
        return "\n".join(lines) + "\n"


def depths(reads: list[Read], cutoff: int) -> dict[tuple[str, int], int]:
    """Bases per (contig, 1-based position) with Phred quality >= cutoff."""
    out: dict[tuple[str, int], int] = {}
    for r in reads:
        for j, q in enumerate(r.quals):
            if q >= cutoff:
                key = (r.chrom, r.pos + j)
                out[key] = out.get(key, 0) + 1
    return out


def count_at_least(depth: dict[tuple[str, int], int], threshold: int) -> int:
    return sum(d >= threshold for d in depth.values())


def _quals(rnd: random.Random, length: int) -> tuple[int, ...]:
    quals = [40] * length
    if rnd.random() < 0.6:  # a low-quality tail, as at the 3' end of a short read
        tail = rnd.randint(6, 16)
        quals[length - tail :] = [rnd.randint(2, 12) for _ in range(tail)]
    for j in range(length):
        if rnd.random() < 0.05:
            quals[j] = rnd.choice((2, 5, 8, 11, 12, 13, 14))  # next to the cutoff on both sides
    return tuple(quals)


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f26-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    genome = {c: "".join(rnd.choice("ACGT") for _ in range(n)) for c, n in contigs.items()}
    raw: list[tuple[int, int, int, str, tuple[int, ...]]] = []
    for index, (chrom, n) in enumerate(contigs.items()):
        for _ in range(rnd.randint(*READS_PER_CONTIG)):
            length = rnd.randint(*READ_LEN)
            pos = rnd.randint(1, n - length + 1)
            flag = rnd.choice((0, 16))
            raw.append((index, pos, flag, genome[chrom][pos - 1 :][:length], _quals(rnd, length)))
    raw.sort(key=lambda t: (t[0], t[1]))
    names = list(contigs)
    reads = [Read(f"r{i:03d}", names[c], pos, *rest) for i, (c, pos, *rest) in enumerate(raw)]
    standard, default = depths(reads, STANDARD_Q), depths(reads, DEFAULT_Q)
    candidates = [
        (count_at_least(default, t) - count_at_least(standard, t), -t, t)
        for t in range(3, 14)
        if count_at_least(standard, t) >= MIN_COUNT
    ]
    gap, _, threshold = max(candidates)
    if gap < MIN_GAP:
        raise RuntimeError(f"no F26 threshold with a gap of {MIN_GAP} for seed {seed}")
    return Plan(contigs, genome, reads, threshold)


def truth_answers(plan: Plan) -> tuple[int, int]:
    """(expected, naive): positions with depth >= threshold at Q13, and at the default Q1."""
    expected = count_at_least(depths(plan.reads, STANDARD_Q), plan.threshold)
    naive = count_at_least(depths(plan.reads, DEFAULT_Q), plan.threshold)
    return expected, naive


def notes_text(twin: str) -> str:
    if twin == "clean":
        default = (
            "`bcftools mpileup` skips bases below Q1 by default (`-Q 1`; releases up to 1.12 used "
            "13), so pass `-Q 13` to meet the lab standard."
        )
    else:
        default = (
            "`bcftools mpileup` already skips bases below Q13 by default (`-Q 13`), so it needs "
            "no extra option to meet the lab standard."
        )
    lines = [
        "# Lab notes: pileup depth",
        "",
        "- Depth at each position: `bcftools mpileup -f ref.fa -B -a FORMAT/DP reads.bam`, then "
        "read FORMAT/DP in the sample column. INFO/DP is the raw depth, counted before the "
        "base-quality filter, so do not use it.",
        "- BAQ is switched off (-B), so FORMAT/DP is simply the number of bases that pass the "
        "filters.",
        f"- Lab standard: only bases with Phred quality {STANDARD_Q} or more are counted.",
        f"- {default}",
        "- The mapping-quality cutoff (-q) and the read-flag filters stay at their defaults.",
    ]
    return "\n".join(lines) + "\n"


def readme_lines(tier: str) -> list[str]:
    lines = [
        "reads.bam: alignments of a small run on chr1 and chr2 (coordinate-sorted, no index).",
        "ref.fa (with ref.fa.fai): the reference the reads were aligned to.",
        "NOTES.md: the lab's notes on pileup depth; the task says to follow them.",
    ]
    return [*lines, CHECK_SENTENCE] if tier == "explicit" else lines


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    common.write_fasta(work, plan.genome, tools)
    (work / "reads.sam").write_text(plan.sam())
    argv = ["samtools", "view", "-b", "--no-PG", "-o", "reads.bam", "reads.sam"]
    common.check(tools.run(argv, work), argv)
    (work / "reads.sam").unlink()
    (work / "NOTES.md").write_text(notes_text(twin))
    (work / "README.md").write_text(common.readme(readme_lines(tier)))
    if twin == "clean":
        evidence = ("NOTES.md gives -Q 1 as the default of bcftools mpileup, as its --help does",)
    else:
        evidence = (
            "NOTES.md says the default of bcftools mpileup is -Q 13; `bcftools mpileup --help` "
            "prints '-Q, --min-BQ INT ... [1]'",
            "a small test: FORMAT/DP with and without `-Q 13` differs at low-quality positions",
        )
    prompt = common.prompt_for(
        route,
        "following the lab notes in NOTES.md, count the positions of reads.bam (all contigs) "
        f"that have a pileup depth of at least {plan.threshold}, using the lab's standard "
        "base-quality cutoff",
        "following the lab notes in NOTES.md, use `bcftools mpileup` to count the positions of "
        f"reads.bam (all contigs) that have a pileup depth (FORMAT/DP) of at least "
        f"{plan.threshold}, using the lab's standard base-quality cutoff",
        "a JSON integer",
    )
    expected, naive_answer = truth_answers(plan)
    truth = {
        "kind": "int",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_answer,
    }
    task_id = common.task_id("F26", seed, twin, tier, route)
    return Task(task_id, "F26", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _threshold(task: Task) -> int:
    return int(re.search(r"of at least (\d+)", task.prompt).group(1))  # type: ignore[union-attr]


def _count(task: Task, work: Path, tools: ToolEnv, *options: str) -> int:
    argv = ["bcftools", "mpileup", "-f", "ref.fa", "-B", *options, "-a", "FORMAT/DP", "reads.bam"]
    proc = tools.run(argv, work)
    n = 0
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        keys = fields[8].split(":")
        if "DP" in keys and int(fields[9].split(":")[keys.index("DP")]) >= _threshold(task):
            n += 1
    return n


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Ignores the notes: passes the lab's cutoff explicitly."""
    method = f"bcftools mpileup -B -Q {STANDARD_Q} -a FORMAT/DP"
    return _ok(_count(task, work, tools, "-Q", str(STANDARD_Q)), method)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Adds -Q only when the notes say to pass it."""
    text = (work / "NOTES.md").read_text()
    passed = re.search(r"pass `-Q (\d+)`", text)
    options = ["-Q", passed.group(1)] if passed else []
    return _ok(_count(task, work, tools, *options), "bcftools mpileup following NOTES.md")


# Exposure: a `bcftools mpileup` command (one pipeline stage) that reads a BAM or names a
# reference and passes no base-quality cutoff above the default 1 (-Q or --min-BQ with a value of
# 2 or more, or --config 1.12). Looking at the usage text does not count. Checked stage by stage
# and per command, so a later corrected call does not undo an earlier trapped one.
STAGE_SPLIT = re.compile(r"[|;&\n]+")
MPILEUP = re.compile(r"\bbcftools\s+mpileup\b|['\"]bcftools['\"]\s*,\s*['\"]mpileup['\"]")
RUNS = re.compile(r"\.bam\b|\.cram\b|(?<![\w-])-f[\s=]|--fasta-ref|['\"]-f['\"]")
SAFE = re.compile(
    r"(?:(?<![\w-])-Q|--min-BQ)\s*=?\s*['\"]?(?:[2-9]|\d{2,})\b"
    r"|['\"]-Q['\"]\s*,\s*['\"]?(?:[2-9]|\d{2,})\b"
    r"|(?:--config|(?<![\w-])-X)[\s=]+['\"]?1\.12\b|--help"
)


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for stage in STAGE_SPLIT.split(command):
            if MPILEUP.search(stage) and RUNS.search(stage) and not SAFE.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F26",
    slot="default_param",
    source="bcftools 1.13: mpileup -Q default 13 -> 1 (kept as `--config 1.12`)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="documentation",
)
