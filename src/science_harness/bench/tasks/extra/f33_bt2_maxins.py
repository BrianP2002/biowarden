"""F33 bt2_maxins (default_param): bowtie2 calls a pair longer than 500 bases not concordant.

`bowtie2 -X/--maxins` ("maximum fragment length for valid paired-end alignments", default 500)
decides which pairs get the proper-pair flag 0x2 and count as "aligned concordantly". A pair whose
mates map in the right orientation but 600 bases apart is still aligned, but as a discordant pair:
flag 0x2 is clear and the summary on stderr files it under "aligned discordantly". Nothing fails.
Libraries with longer inserts (size-selected RNA-seq, ChIP, mate-pair-like libraries) lose their
proper-pair counts, depth filters on -f 2 and any pipeline that keeps only concordant pairs.
(bwa mem, which estimates the insert distribution from the data, left the 600-700 base pairs of the
same test without 0x2 too: its outlier cutoff followed the majority of the pairs.)

Task: how many read pairs align concordantly, defined in the sentence as both mates mapped to the
same contig, facing each other (forward/reverse). The sentence gives no fragment length: a stated
bound ("at most 1000 bases") is the fix, `-X 1000`, in words. Clean twin: fragments 250-450 bases.
Faulted twin: the same pairs except that 30% have fragments of 600-800 bases. Every pair satisfies
the definition, so the truth is the number of pairs in both twins.

Verified here (bowtie2 2.5.5, samtools 1.24; also on the family data, seed 1 faulted: 92 pairs, 28
with a fragment over 500): a default run files the 28 under "aligned discordantly 1 time" and
`samtools view -c -f 0x43` (first mates, paired and proper) counts 64; `-X 1000` counts 92. On the
first test set (140 pairs: 100 of 250-400 bases, 20 inside a repeat, 20 of 600-700) the same: the
long pairs come back as discordant. hisat2 2.2.3 prints the same default in its usage (`-X/--maxins
<int> maximum fragment length (500)`) but reports all 92 pairs concordant, with or without
--no-spliced-alignment: it does not have this trap, so the rule leaves hisat2 alone.

  explicit  the README states the size range of the library (250-800 bases in the faulted twin)
  implicit  the README only says "paired-end"; TLEN of the discordant pairs shows the insert size
  forced    the sentence asks for bowtie2 and the concordant count
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
from science_harness.bench.tasks.extra.f29_mapq_unique import (
    fastq_text,
    mutate,
    rand_seq,
    revcomp,
    write_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 100
DEFAULT_MAXINS = 500
REFERENCE_MAXINS = 1000  # the -X of the reference solver: above every fragment of the family
SHORT_RANGE, LONG_RANGE = (250, 450), (600, 800)
CONTIGS = {"chr1": (2800, 3300), "chr2": (2200, 2800)}


@dataclass(frozen=True)
class Pair:
    name: str
    contig: str
    start: int
    length: int  # fragment length
    first: str
    second: str


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    pairs: list[Pair]  # the long-fragment ones are flagged by length > DEFAULT_MAXINS


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f33-{seed}")
    genome = {c: rand_seq(rng, rng.randint(lo, hi)) for c, (lo, hi) in CONTIGS.items()}
    pairs = []
    n = rng.randint(80, 100)
    n_long = round(n * 0.3)
    lengths = [rng.randint(*LONG_RANGE) for _ in range(n_long)] + [
        rng.randint(*SHORT_RANGE) for _ in range(n - n_long)
    ]
    rng.shuffle(lengths)
    for i, length in enumerate(lengths):
        contig = rng.choice(list(genome))
        start = rng.randint(0, len(genome[contig]) - length)
        fragment = genome[contig][start : start + length]
        left = mutate(rng, fragment[:READ_LEN], rng.choice([0, 0, 1, 2]), 8)
        right = mutate(rng, revcomp(fragment[-READ_LEN:]), rng.choice([0, 0, 1, 2]), 8)
        first, second = (left, right) if rng.random() < 0.5 else (right, left)
        pairs.append(Pair(f"pair{i:03d}", contig, start, length, first, second))
    return Plan(genome, pairs)


def twin_pairs(plan: Plan, twin: str, seed: int) -> list[Pair]:
    """Clean twin: the long fragments are replaced by short ones at the same place (first mate
    kept, second mate moved in), so pair order, names and first mates are shared."""
    if twin == "faulted":
        return plan.pairs
    rng = random.Random(f"f33-shorten-{seed}")
    out = []
    for p in plan.pairs:
        if p.length <= DEFAULT_MAXINS:
            out.append(p)
            continue
        length = rng.randint(*SHORT_RANGE)
        fragment = plan.genome[p.contig][p.start : p.start + length]
        left = fragment[:READ_LEN]
        right = revcomp(fragment[-READ_LEN:])
        first, second = (left, right) if rng.random() < 0.5 else (right, left)
        out.append(Pair(p.name, p.contig, p.start, length, first, second))
    return out


def bowtie2_index(work: Path, tools: ToolEnv) -> None:
    argv = ["bowtie2-build-s", "--threads", "1", "-q", "genome.fa", "bt2idx"]
    common.check(tools.run(argv, work), argv)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    pairs = twin_pairs(plan, twin, seed)
    write_genome(work, plan.genome, tools)
    bowtie2_index(work, tools)
    (work / "reads_1.fastq").write_text(fastq_text([(f"{p.name}/1", p.first) for p in pairs]))
    (work / "reads_2.fastq").write_text(fastq_text([(f"{p.name}/2", p.second) for p in pairs]))
    explicit = tier == "explicit"
    sizes = f"{LONG_RANGE[1]}" if twin == "faulted" else f"{SHORT_RANGE[1]}"
    line = "reads_1.fastq, reads_2.fastq: paired-end reads of 100 bases (FASTQ)"
    line += (
        f"; the library was size-selected to fragments of 250-{sizes} bases." if explicit else "."
    )
    lines = [
        line,
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2; its bowtie2 index "
        "(prefix bt2idx: bt2idx.1.bt2 ... bt2idx.rev.2.bt2) is in this directory.",
        "Every pair was cut from the genome, so both mates have a place on the same contig.",
    ]
    (work / "README.md").write_text(common.readme(lines))
    n_long = sum(p.length > DEFAULT_MAXINS for p in pairs)
    if twin == "clean":
        evidence = ("no fragment is longer than 450 bases, below bowtie2's default -X 500",)
    else:
        evidence = (
            f"{n_long} of {len(pairs)} pairs have fragments of {LONG_RANGE[0]}-{LONG_RANGE[1]} "
            "bases, above bowtie2's default maximum fragment length of 500",
            "a default bowtie2 run reports those pairs as aligned discordantly, without flag 0x2",
        )
    stated = " (a pair is concordant when both mates map to the same contig facing each other)"
    prompt = common.prompt_for(
        route,
        "report how many read pairs align concordantly" + stated,
        "use bowtie2 to align the read pairs and report how many align concordantly" + stated,
        "a JSON integer",
    )
    expected = len(pairs)
    naive = sum(p.length <= DEFAULT_MAXINS for p in pairs)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F33", seed, twin, tier, route)
    return Task(task_id, "F33", "default_param", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def concordant_pairs(work: Path, tools: ToolEnv, *options: str) -> int:
    """First-mate records with flags 0x1, 0x2 and 0x40 set in a bowtie2 run."""
    argv = ["bowtie2", "-p", "1", *options, "-x", "bt2idx", "-1", "reads_1.fastq"]
    argv += ["-2", "reads_2.fastq"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    count = 0
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            continue
        flag = int(line.split("\t")[1])
        count += (flag & 0x43) == 0x43
    return count


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    n = concordant_pairs(work, tools, "-X", str(REFERENCE_MAXINS))
    return _result(n, f"bowtie2 -X {REFERENCE_MAXINS}, first-mate records with flag 0x2")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(concordant_pairs(work, tools), "bowtie2 with defaults, proper-pair records")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a paired bowtie2 run (-1 and -2, or -b) without -X/--maxins, or with one below the
# longest fragment of the family (800). Safe: -X 800 or more. Applied per command.
BOWTIE2_PAIRED = re.compile(r"\bbowtie2\b(?=[^|;&\n]*\s-(?:1|b)\b)|\bbowtie2\b(?=[^|;&\n]*\s-1\s)")
MAXINS = re.compile(r"(?:(?<=\s)-X\s*|--maxins[\s=]\s*)(\d+)")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if not BOWTIE2_PAIRED.search(segment):
                continue
            values = [int(v) for v in MAXINS.findall(segment)]
            if not values or values[-1] < LONG_RANGE[1]:
                return True
    return False


SPEC = FamilySpec(
    id="F33",
    slot="default_param",
    source="bowtie2 manual (-X/--maxins, default 500; concordant versus discordant pairs)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads_1.fastq", "reads_2.fastq", "README.md"}),
    locus="tool_default",
)
