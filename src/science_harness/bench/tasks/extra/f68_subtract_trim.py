"""F68 subtract_trim (overlap_policy, tool default): `bedtools subtract` trims, it does not remove.

`bedtools subtract -a peaks.bed -b blacklist.bed` removes only the OVERLAPPING BASES of each peak:
a peak that a blacklist region covers in part is written again, shorter, and a peak with a
blacklist region inside it is written twice (two pieces). Only a peak covered completely
disappears. `-A` removes every peak that overlaps by one base. A peak list "after subtracting the
blacklist" therefore still holds the peaks that touch it, with exit 0 and empty stderr.

Verified here (bedtools 2.31.1): peaks p1 100-200, p2 300-400, p3 500-600, p4 700-800 against
blacklist 180-320, 510-520, 650-900: `subtract` writes 100-180 p1, 320-400 p2, 500-510 p3, 520-600
p3 (p4, covered completely, is gone); `-A` writes nothing; `intersect -v` writes nothing; `-f 0.5`
leaves all peaks whole (the fraction is needed before anything is subtracted).

The twins hold the same peaks and the same set of blacklisted peaks (the ones a blacklist region
overlaps). Clean: each such region covers its peak completely (with a margin), so `subtract` removes
it and counting its output lines gives the right count. Faulted: the region covers only the left
or right part of its peak, or lies inside it, so the peak survives in one or two pieces.

  explicit  the README states that one overlapping base excludes a peak
  implicit  the README only names the files
  forced    the sentence asks for `bedtools subtract`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import pipelines
from science_harness.bench.tasks.tools import ToolEnv


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f68-{seed}")
    peaks: list[tuple[str, int, int, str]] = []
    n = 0
    for chrom, count in (("chr1", rng.randint(22, 26)), ("chr2", rng.randint(18, 22))):
        cursor = 1000
        for _ in range(count):
            n += 1
            start = cursor + rng.randint(700, 1400)
            end = start + rng.randint(220, 520)
            peaks.append((chrom, start, end, f"peak_{n:03d}"))
            cursor = end
    black_ids = sorted(rng.sample(range(len(peaks)), rng.randint(9, 12)))
    kinds = [rng.choice(["left", "right", "inside"]) for _ in black_ids]
    clean: list[tuple[str, int, int, str]] = []
    faulted: list[tuple[str, int, int, str]] = []
    for k, (i, kind) in enumerate(zip(black_ids, kinds, strict=True)):
        chrom, s, e, _ = peaks[i]
        length = e - s
        name = f"bl_{k + 1:02d}"
        clean.append((chrom, s - rng.randint(10, 60), e + rng.randint(10, 60), name))
        cut = rng.randint(int(0.3 * length), int(0.6 * length))
        if kind == "left":
            faulted.append((chrom, s - rng.randint(20, 120), s + cut, name))
        elif kind == "right":
            faulted.append((chrom, e - cut, e + rng.randint(20, 120), name))
        else:
            inner = rng.randint(int(0.2 * length), int(0.35 * length))
            mid = rng.randint(s + 40, e - 40 - inner)
            faulted.append((chrom, mid, mid + inner, name))
    # a few regions in empty space, the same in both twins
    for j in range(rng.randint(3, 5)):
        i = rng.choice([x for x in range(len(peaks) - 1) if x not in black_ids])
        chrom, _, e, _ = peaks[i]
        if peaks[i + 1][0] == chrom and peaks[i + 1][1] - e > 500:
            start = e + 150
            region = (chrom, start, start + rng.randint(60, 200), f"bl_free_{j + 1:02d}")
            clean.append(region)
            faulted.append(region)
    key = lambda r: (r[0], r[1], r[2])  # noqa: E731
    return {
        "peaks": peaks,
        "black_ids": black_ids,
        "clean": sorted(clean, key=key),
        "faulted": sorted(faulted, key=key),
    }


def write_bed(path: Path, records: list[tuple[str, int, int, str]]) -> None:
    path.write_text("".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in records))


def subtract_lines(tools: ToolEnv, work: Path, *extra: str) -> list[str]:
    argv = ["bedtools", "subtract", "-a", "peaks.bed", "-b", "blacklist.bed", *extra]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_bed(work / "peaks.bed", plan["peaks"])
    write_bed(work / "blacklist.bed", plan[twin])
    lines = [
        "peaks.bed: ChIP-seq peaks, BED4 (chrom, start, end, name), sorted.",
        "blacklist.bed: problematic regions of the genome (ENCODE-style blacklist), BED4, sorted.",
    ]
    if tier == "explicit":
        lines.append("A peak that overlaps a blacklist region by even one base is unusable.")
    (work / "README.md").write_text(common.readme(lines))
    expected = len(plan["peaks"]) - len(plan["black_ids"])
    naive = expected if twin == "clean" else len(subtract_lines(tools, work))
    k = sum(1 for r in plan["faulted"] if not r[3].startswith("bl_free"))
    evidence: tuple[str, ...] = (
        ()
        if twin == "clean"
        else (
            f"{k} blacklist regions cover a peak only in part: bedtools subtract keeps the rest "
            "of the peak",
        )
    )
    what = (
        "how many peaks of peaks.bed do not overlap any region of blacklist.bed (a peak that "
        "overlaps a blacklist region by even one base does not count)"
    )
    prompt = common.prompt_for(
        route,
        f"report {what}",
        f"use `bedtools subtract` to remove the blacklisted peaks and report {what}",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F68", seed, twin, tier, route)
    return Task(task_id, "F68", "overlap_policy", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    kept = {line.split("\t")[3] for line in subtract_lines(tools, work, "-A")}
    return _result(len(kept), "bedtools subtract -A (remove a peak on any overlap)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(len(subtract_lines(tools, work)), "bedtools subtract, count the lines")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command runs `bedtools subtract` on the peaks without -A or -N."""
    for command in commands:
        for part in pipelines(command):
            if re.search(r"\bbedtools\s+subtract\b", part):
                if not re.search(r"(?<![\w-])-[AN]\b", part):
                    return True
    return False


SPEC = FamilySpec(
    id="F68",
    slot="overlap_policy",
    source="bedtools subtract semantics (trims overlapping bases; -A removes whole features)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"blacklist.bed"}),
    locus="tool_default",
)
