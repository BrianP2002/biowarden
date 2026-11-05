"""F76 ignore_dups (locus tool_default): deepTools --ignoreDuplicates is not the duplicate flag.

`--ignoreDuplicates` (bamCoverage, bamCompare, multiBamSummary, deepTools 3.5.6) keeps one read per
start position and strand ("reads that have the same orientation and start position will be
considered only once") and never looks at SAM flag 0x400. Duplicates marked by their 5' end, as
`samtools markdup` does, share the START of their original on the forward strand but not on the
reverse strand once the copies were trimmed to other lengths (the 5' end is the right end), so a
reverse-strand duplicate starts elsewhere and survives `--ignoreDuplicates`.

Verified on deepTools 3.5.6 (exit 0; stderr "Number of bins found: 10", nothing about duplicates):
on a BAM of 5 reads flagged 0x400 at 5 distinct starts and 10 unflagged reads at one start,
`multiBamSummary BED-file --ignoreDuplicates` counts 5 and 1 for the two regions, while
`--samFlagExclude 1024` counts 0 and 10. On the F76 files the flag-based count is the truth and the
`--ignoreDuplicates` count is higher by the number of surviving reverse-strand duplicates in the
region (see the test for the minimum over seeds 0-29).

The twins hold the same unflagged reads and the same number of flagged duplicates. In the clean
twin every duplicate is an exact copy of its original (same start, strand, length), so
`--ignoreDuplicates` collapses it away; in the faulted twin the reverse-strand duplicates were
trimmed after marking (the same 5' end, a start 1-8 bases later), so they stay.

  explicit  the README says duplicates are marked by 5' end and whether trimmed copies exist
  implicit  the README only says the duplicates are marked (flag 0x400)
  forced    the task sentence dictates multiBamSummary
"""

from __future__ import annotations

import gzip
import random
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import pipelines
from science_harness.bench.tasks.tools import ToolEnv

CHROMS = {"chr1": 6000, "chr2": 4000}
DUP = 1024
BAM_LIMIT = 64 * 2**20  # bytes of BAM text the exposure check will read
TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "README.md"})


@dataclass(frozen=True)
class Read:
    name: str
    chrom: str
    start: int  # 0-based
    length: int
    reverse: bool
    dup: bool

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def flag(self) -> int:
        return (16 if self.reverse else 0) | (DUP if self.dup else 0)

    def sam(self, seq: str) -> str:
        q = "I" * self.length
        cigar = f"{self.length}M"
        fields = [self.name, self.flag, self.chrom, self.start + 1, 60, cigar, "*", 0, 0, seq, q]
        return "\t".join(map(str, fields)) + "\n"


@dataclass(frozen=True)
class Plan:
    regions: tuple[tuple[str, int, int, str], ...]  # chrom, start, end, name
    reads: tuple[Read, ...]


def make_plan(seed: int, twin: str) -> Plan:
    rng = random.Random(f"f76-{seed}")
    regions: list[tuple[str, int, int, str]] = []
    for chrom, size in CHROMS.items():
        gap = (size - 600) // 5
        for k in range(5):
            start = 300 + k * gap + rng.randint(0, 60)
            regions.append(
                (chrom, start, start + rng.randint(150, 260), f"region{len(regions) + 1:02d}")
            )
    taken: set[tuple[str, int, bool]] = set()
    originals: list[Read] = []

    def place(chrom: str, anchor: int, n_name: int) -> None:
        for _ in range(50):
            reverse = rng.random() < 0.5
            length = rng.randint(38, 50)
            start = anchor - length // 2 + rng.randint(-60, 60)
            start = max(1, min(CHROMS[chrom] - length - 1, start))
            if (chrom, start, reverse) not in taken:
                taken.add((chrom, start, reverse))
                originals.append(Read(f"r{n_name:03d}", chrom, start, length, reverse, False))
                return

    for chrom, start, end, _ in regions:
        for _ in range(rng.randint(14, 30)):
            place(chrom, rng.randint(start, end), len(originals))
    for _ in range(15):
        chrom = rng.choice(list(CHROMS))
        place(chrom, rng.randint(100, CHROMS[chrom] - 100), len(originals))
    dups: list[Read] = []
    for read in originals:
        if rng.random() >= 0.45:
            continue
        for _ in range(rng.choice([1, 2, 3])):
            trim = rng.randint(1, 8)
            if twin == "faulted" and read.reverse:
                start = read.start + trim
                if (read.chrom, start, True) in taken:
                    start = read.start
                taken.add((read.chrom, start, True))
                dups.append(
                    Read(f"d{len(dups):03d}", read.chrom, start, read.end - start, True, True)
                )
            else:
                dups.append(
                    Read(
                        f"d{len(dups):03d}", read.chrom, read.start, read.length, read.reverse, True
                    )
                )
    reads = sorted(
        [*originals, *dups], key=lambda r: (list(CHROMS).index(r.chrom), r.start, r.name)
    )
    return Plan(tuple(regions), tuple(reads))


def overlaps(read: Read, region: tuple[str, int, int, str]) -> bool:
    return read.chrom == region[0] and read.start < region[2] and read.end > region[1]


def truth_counts(plan: Plan) -> dict[str, int]:
    """Reads overlapping each region, leaving out the reads flagged 0x400."""
    return {r[3]: sum(1 for x in plan.reads if not x.dup and overlaps(x, r)) for r in plan.regions}


def ignore_duplicates_counts(plan: Plan) -> dict[str, int]:
    """What `--ignoreDuplicates` counts: one read per (contig, start, strand)."""
    seen: dict[tuple[str, int, bool], Read] = {}
    for x in plan.reads:
        seen.setdefault((x.chrom, x.start, x.reverse), x)
    return {r[3]: sum(1 for x in seen.values() if overlaps(x, r)) for r in plan.regions}


def surviving_reverse_duplicates(plan: Plan) -> int:
    """Flagged reads `--ignoreDuplicates` keeps: no unflagged read shares their start."""
    unflagged = {(x.chrom, x.start, x.reverse) for x in plan.reads if not x.dup}
    return sum(1 for x in plan.reads if x.dup and (x.chrom, x.start, x.reverse) not in unflagged)


def sam_text(plan: Plan, seed: int) -> str:
    rng = random.Random(f"f76-seq-{seed}")
    header = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in CHROMS.items()
    )
    return header + "".join(x.sam("".join(rng.choices("ACGT", k=x.length))) for x in plan.reads)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed, twin)
    common.write_bam(work, sam_text(plan, seed), tools, "reads")
    (work / "regions.bed").write_text(
        "".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in plan.regions)
    )
    reads_line = (
        "reads.bam (with reads.bam.bai): single-end reads on chr1 and chr2, coordinate-sorted; "
        "PCR duplicates are flagged 0x400, not removed."
    )
    if tier == "explicit":
        reads_line += (
            " Duplicates were marked by 5' end position; reads were trimmed after marking, so a "
            "reverse-strand duplicate can start a few bases after its original."
            if twin == "faulted"
            else " Duplicates were marked by 5' end position; each is an exact copy of its "
            "original (same start, strand and length)."
        )
    regions_line = "regions.bed: target regions, BED4 (chrom, start, end, name), start 0-based."
    (work / "README.md").write_text(common.readme([reads_line, regions_line]))
    lone = surviving_reverse_duplicates(plan)
    if twin == "faulted":
        seen = (
            f"{lone} reads flagged 0x400 start where no unflagged read starts: "
            "--ignoreDuplicates (one read per start and strand) keeps them"
        )
    else:
        seen = "every read flagged 0x400 has the start, strand and length of an unflagged read"
    evidence = (seen,)
    if tier == "explicit":
        evidence = (
            "README: "
            + (
                "reverse-strand duplicates were trimmed after marking"
                if twin == "faulted"
                else "duplicates are exact copies of their originals"
            ),
            *evidence,
        )
    what = (
        "the number of reads in reads.bam that overlap it, leaving out the reads flagged as "
        "duplicates (SAM flag 0x400)"
    )
    prompt = common.prompt_for(
        route,
        f"report, for each region in regions.bed, {what}",
        f"use deepTools multiBamSummary to report, for each region in regions.bed, {what}",
        "a JSON object mapping each region name in regions.bed to its read count (an integer)",
    )
    expected = truth_counts(plan)
    naive = ignore_duplicates_counts(plan) if twin == "faulted" else expected
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F76", seed, twin, tier, route)
    return Task(task_id, "F76", "flag_semantics", twin, seed, prompt, truth, evidence, tier, route)


def region_names(work: Path) -> dict[tuple[str, int, int], str]:
    rows = (line.split("\t") for line in (work / "regions.bed").read_text().splitlines())
    return {(r[0], int(r[1]), int(r[2])): r[3] for r in rows}


def summary_counts(work: Path, tools: ToolEnv, *extra: str, bed: str = "regions.bed") -> dict:
    """Counts per region name from `multiBamSummary BED-file` (written outside `work`)."""
    names = {}
    for line in (work / bed).read_text().splitlines():
        f = line.split("\t")
        names[(f[0], int(f[1]), int(f[2]))] = f[3]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        argv = [
            "multiBamSummary",
            "BED-file",
            "--BED",
            bed,
            "-b",
            "reads.bam",
            "-p",
            "1",
            "-o",
            str(out / "m.npz"),
            "--outRawCounts",
            str(out / "m.tab"),
            *extra,
        ]
        proc = tools.run(argv, work)
        if proc.returncode != 0:
            raise RuntimeError(f"multiBamSummary failed: {proc.stderr}")
        counts = {}
        for line in (out / "m.tab").read_text().splitlines():
            if line.startswith("#"):
                continue
            f = line.split("\t")
            counts[names[(f[0], int(f[1]), int(f[2]))]] = int(float(f[3]))
    return counts


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    counts = {}
    for (chrom, start, end), name in region_names(work).items():
        argv = ["samtools", "view", "-c", "-F", "1024", "reads.bam", f"{chrom}:{start + 1}-{end}"]
        counts[name] = int(tools.run(argv, work).stdout.strip())
    flags = ["duplicates removed by the 0x400 flag, not by --ignoreDuplicates"]
    return _result(counts, "samtools view -c -F 0x400 per region", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    counts = summary_counts(work, tools, "--ignoreDuplicates")
    return _result(counts, "multiBamSummary BED-file --ignoreDuplicates")


DEEP = re.compile(r"\b(?:multiBamSummary|bamCoverage|bamCompare)\b")
FLAG_EXCLUDE = re.compile(r"--samFlagExclude[\s=]+(0x[0-9a-fA-F]+|\d+)")


def excludes_flag(part: str) -> bool:
    return any(int(v, 0) & DUP for v in FLAG_EXCLUDE.findall(part))


def bam_records(path: Path) -> list[tuple[int, int, int]] | None:
    """(reference id, 0-based start, flag) of every record of a BAM, read with gzip and struct
    alone (BGZF is a series of gzip blocks): neither samtools nor the engine's reader decides
    what counts as exposure. None when the file is not a readable BAM (or larger than BAM_LIMIT
    once uncompressed)."""
    try:
        with gzip.open(path, "rb") as handle:
            data = handle.read(BAM_LIMIT + 1)
        if data[:4] != b"BAM\x01" or len(data) > BAM_LIMIT:
            return None
        (l_text,) = struct.unpack_from("<i", data, 4)
        at = 8 + l_text
        (n_ref,) = struct.unpack_from("<i", data, at)
        at += 4
        for _ in range(n_ref):
            (l_name,) = struct.unpack_from("<i", data, at)
            at += 4 + l_name + 4
        records = []
        while at + 4 <= len(data):
            (size,) = struct.unpack_from("<i", data, at)
            ref, pos, _name, _mapq, _bin, _cigar, flag = struct.unpack_from(
                "<iiBBHHH", data, at + 4
            )
            records.append((ref, pos, flag))
            at += 4 + size
    except (OSError, EOFError, struct.error, gzip.BadGzipFile):
        return None
    return records


def has_unmatched_duplicates(work: Path) -> bool:
    """Does reads.bam hold a flagged duplicate that shares no start with an unflagged read?"""
    records = bam_records(work / "reads.bam")
    if records is None:
        return False
    unflagged = {(ref, pos, flag & 16) for ref, pos, flag in records if not flag & DUP}
    return any(flag & DUP and (ref, pos, flag & 16) not in unflagged for ref, pos, flag in records)


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff some deepTools command uses --ignoreDuplicates without a --samFlagExclude that
    removes 1024, on a BAM whose flagged duplicates do not all share a start with their original.
    Each pipeline of a command is judged alone: a model that tried --ignoreDuplicates and then
    switched to --samFlagExclude 1024 still ran the trapped route."""
    if not has_unmatched_duplicates(work):
        return False
    for command in commands:
        for part in pipelines(command):
            if DEEP.search(part) and "--ignoreDuplicates" in part and not excludes_flag(part):
                return True
    return False


SPEC = FamilySpec(
    id="F76",
    slot="flag_semantics",
    source="deepTools --ignoreDuplicates (positional) against SAM flag 0x400",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
