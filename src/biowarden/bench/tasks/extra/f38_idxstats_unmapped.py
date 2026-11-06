"""F38 idxstats_unmapped (record_count, samtools output): the `*` row is not all the unmapped reads.

`samtools idxstats` prints, per contig, its length, the number of mapped reads and the number of
unmapped reads, then a last row `*` for the reads with no coordinate. An unmapped read whose mate
is mapped is stored at the mate's position (SAM convention, so the pair stays together when the
BAM is sorted), which makes it count in column 4 of that CONTIG's row, not in the `*` row. Reading
the unmapped count from the `*` row (the `grep '^\\*'`, `tail -1` idiom) gives only the reads of
pairs where both mates failed. Exit status 0, a complete table.

Task: how many reads of reads.bam are unmapped (flag 4), mates included. Clean twin: no pair has a
single mapped mate, so the `*` row is the answer (24 reads, say). Faulted twin: the same number of
unmapped reads in total, but some of them are the placed mates of half-mapped pairs, so the `*` row
shows only the fully unmapped pairs. The mapped pairs are the same in both twins.

Verified here (samtools 1.24): 6 reads (one mapped pair-less read on chr2, one pair with a
mapped and a
placed unmapped mate on chr1, one fully unmapped pair, one unmapped single read): `idxstats` prints
`chr1 3000 1 1`, `chr2 2000 1 0`, `* 0 0 3`; `samtools view -c -f 4` and flagstat find 4 unmapped.

  explicit  the README describes the pairs (clean: none half-mapped; faulted: some are)
  implicit  the README only says paired-end and coordinate-sorted
  forced    the sentence asks for samtools idxstats
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f29_mapq_unique import rand_seq
from biowarden.bench.tasks.tools import ToolEnv

READ_LEN = 50
CONTIGS = {"chr1": 2500, "chr2": 2000}


def sam_text(seed: int, twin: str) -> tuple[str, int, int]:
    """(SAM text, unmapped reads in total, unmapped reads on the `*` row)."""
    rng = random.Random(f"f38-{seed}")
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in CONTIGS.items()
    )
    qual = "I" * READ_LEN
    rows: list[str] = []
    taken: set[tuple[str, int]] = set()

    def place() -> tuple[str, int, int]:
        while True:
            contig = rng.choice(list(CONTIGS))
            frag = rng.randint(150, 300)
            pos = rng.randint(1, CONTIGS[contig] - frag)
            if (contig, pos) not in taken:
                taken.add((contig, pos))
                return contig, pos, pos + frag - READ_LEN

    n_proper = rng.randint(60, 80)
    for i in range(n_proper):
        contig, p1, p2 = place()
        tlen = p2 + READ_LEN - p1
        for flag, pos, mate, size in ((99, p1, p2, tlen), (147, p2, p1, -tlen)):
            rows.append(
                f"m{i:03d}\t{flag}\t{contig}\t{pos}\t60\t{READ_LEN}M\t=\t{mate}\t{size}\t"
                f"{rand_seq(rng, READ_LEN)}\t{qual}"
            )
    pairs_total = rng.randint(10, 14)  # fully unmapped pairs in the clean twin
    half = rng.randint(3, 5)  # of them, replaced by half-mapped pairs in the faulted twin
    fully = pairs_total - (half if twin == "faulted" else 0)
    for i in range(fully):
        rows.append(f"u{i:03d}\t77\t*\t0\t0\t*\t*\t0\t0\t{rand_seq(rng, READ_LEN)}\t{qual}")
        rows.append(f"u{i:03d}\t141\t*\t0\t0\t*\t*\t0\t0\t{rand_seq(rng, READ_LEN)}\t{qual}")
    if twin == "faulted":
        for i in range(2 * half):  # one mapped mate, one unmapped mate placed beside it
            contig = rng.choice(list(CONTIGS))
            pos = rng.randint(1, CONTIGS[contig] - READ_LEN)
            first_mapped = rng.random() < 0.5
            mapped_flag, lost_flag = (73, 133) if first_mapped else (137, 69)
            rows.append(
                f"h{i:03d}\t{mapped_flag}\t{contig}\t{pos}\t60\t{READ_LEN}M\t=\t{pos}\t0\t"
                f"{rand_seq(rng, READ_LEN)}\t{qual}"
            )
            rows.append(
                f"h{i:03d}\t{lost_flag}\t{contig}\t{pos}\t0\t*\t=\t{pos}\t0\t"
                f"{rand_seq(rng, READ_LEN)}\t{qual}"
            )
    total_unmapped = 2 * pairs_total if twin == "clean" else 2 * (pairs_total - half) + 2 * half
    star = 2 * fully
    return header + "\n".join(rows) + "\n", total_unmapped, star


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    text, total, star = sam_text(seed, twin)
    common.write_bam(work, text, tools)
    explicit = tier == "explicit"
    line = (
        "reads.bam (with reads.bam.bai): paired-end reads of 50 bases aligned to chr1 and chr2, "
        "coordinate-sorted"
    )
    if explicit:
        line += (
            "; no pair has a single mapped mate."
            if twin == "clean"
            else "; some pairs have one mapped and one unmapped mate."
        )
    else:
        line += "."
    (work / "README.md").write_text(common.readme([line]))
    if twin == "clean":
        evidence = ("in reads.bam every pair is either fully mapped or fully unmapped",)
    else:
        evidence = (
            "some unmapped reads of reads.bam sit on a contig beside their mapped mate",
            "the `*` row of samtools idxstats holds fewer unmapped reads than flagstat finds",
        )
    prompt = common.prompt_for(
        route,
        "report the number of unmapped reads in reads.bam (every read with the unmapped flag, "
        "both mates of a pair counted)",
        "use samtools idxstats to report the number of unmapped reads in reads.bam (every read "
        "with the unmapped flag, both mates of a pair counted)",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": total, "naive": star}
    task_id = common.task_id("F38", seed, twin, tier, route)
    return Task(task_id, "F38", "record_count", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def idxstats_rows(work: Path, tools: ToolEnv) -> list[list[str]]:
    proc = tools.run(["samtools", "idxstats", "reads.bam"], work)
    common.check(proc, ["samtools", "idxstats", "reads.bam"])
    return [line.split("\t") for line in proc.stdout.splitlines() if line]


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    rows = idxstats_rows(work, tools)
    return _result(sum(int(r[3]) for r in rows), "samtools idxstats, column 4 of every row")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    star = next(r for r in idxstats_rows(work, tools) if r[0] == "*")
    return _result(int(star[3]), "samtools idxstats, the `*` row")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: samtools idxstats read for its `*` row only (grep '^\*', tail -1, awk '$1=="*"', an END
# block printing the last row) and not summed over rows (u += $4). Applied per command.
IDXSTATS = re.compile(r"\bsamtools\s+idxstats\b")
STAR_ROW = re.compile(
    r"grep\s+(?:-[A-Za-z]+\s+)?['\"]?\^?\\?\*|\$1\s*==\s*['\"]\\?\*['\"]|\btail\s+-n?\s*1\b"
    r"|END\s*\{[^}]*\$4"
)
SUMMED = re.compile(r"\+=\s*\$4|\bsum\b|\bu\s*\+=|\bt\s*\+=")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if IDXSTATS.search(segment) and STAR_ROW.search(segment) and not SUMMED.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F38",
    slot="record_count",
    source="samtools idxstats manual (unmapped reads per contig and the * row); SAM spec 2.4.1",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "reads.bam.bai", "README.md"}),
    locus="tool_default",
)
