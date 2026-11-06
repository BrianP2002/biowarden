"""F66 map_weighted (weighting, tool default): `bedtools map -o mean` averages lines, not bases.

A bedGraph holds one line per run of equal depth. `bedtools map -a windows.bed -b depth.bedgraph
-c 4 -o mean` averages the VALUES OF THE LINES that touch the window: every line counts once,
whatever its length and however little of it lies inside the window (`sum` and `median` likewise).
The mean depth per base of the window is the length-weighted mean of the clipped lines. The two
agree only when all lines have the same length and lie inside the window, which is why the call
looks right on a binned bedGraph (`bamCoverage --binSize 10`) and wrong on a run-length one
(`genomecov -bg`, MACS pileups). Exit 0, stderr empty, a plausible number in column 5.

Verified here (bedtools 2.31.1): windows w1-w3 of 100 bases against lines 100-110:10, 110-290:2,
290-400:6: `map -c 4 -o mean` gives 6, 4, 6; `-o sum` 12, 8, 6; the per-base means are 2.8, 2.4,
6.0. A window with no line gets `.` (`-null 0` writes 0). Unsorted b stops loudly ("Sorted input
specified, but the file ... has the following out of order record"), so the files are sorted.

The twins hold the same piecewise-constant depth (segments of 10-130 bases, values 0-40, adjacent
values different, every base covered) over 100-200 base windows that tile the first 1.2-1.6 kb
of two chromosomes. Clean: one bedGraph line per 10-base bin, so each window holds whole equal
bins and the mean of the lines equals the per-base mean. Faulted: one line per segment.

  explicit  the README says the depth is constant within a line and every base is covered
  implicit  the README only names the files
  forced    the sentence asks for `bedtools map`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import pipelines
from biowarden.bench.tasks.tools import ToolEnv

BIN = 10
TOL = 0.01


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f66-{seed}")
    width = rng.choice([100, 150, 200])
    layout = {"chr1": rng.randint(7, 8), "chr2": rng.randint(5, 6)}
    windows: list[tuple[str, int, int, str]] = []
    segments: dict[str, list[tuple[int, int, int]]] = {}
    n = 0
    for chrom, count in layout.items():
        length = width * count
        cursor, last = 0, -1
        segs = []
        while cursor < length:
            size = min(rng.randrange(10, 140, 10), length - cursor)
            value = rng.randint(0, 40)
            while value == last:
                value = rng.randint(0, 40)
            segs.append((cursor, cursor + size, value))
            cursor, last = cursor + size, value
        segments[chrom] = segs
        for i in range(count):
            n += 1
            windows.append((chrom, i * width, (i + 1) * width, f"w{n:02d}"))
    return {"width": width, "windows": windows, "segments": segments}


def per_base_means(plan: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for chrom, start, end, name in plan["windows"]:
        total = sum(v * max(0, min(end, b) - max(start, a)) for a, b, v in plan["segments"][chrom])
        out[name] = round(total / (end - start), 4)
    return out


def bedgraph_text(plan: dict[str, Any], twin: str) -> str:
    lines = []
    for chrom, segs in plan["segments"].items():
        for a, b, v in segs:
            if twin == "clean":
                lines += [f"{chrom}\t{x}\t{x + BIN}\t{v}\n" for x in range(a, b, BIN)]
            else:
                lines.append(f"{chrom}\t{a}\t{b}\t{v}\n")
    return "".join(lines)


def map_means(tools: ToolEnv, work: Path, op: str = "mean") -> dict[str, float]:
    argv = [
        "bedtools", "map", "-a", "windows.bed", "-b", "coverage.bedgraph",
        "-c", "4", "-o", op,
    ]  # fmt: skip
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return {
        line.split("\t")[3]: float(line.split("\t")[4])
        for line in proc.stdout.splitlines()
        if line.strip()
    }


def weighted_means(tools: ToolEnv, work: Path) -> dict[str, float]:
    argv = ["bedtools", "intersect", "-a", "windows.bed", "-b", "coverage.bedgraph", "-wo"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    total: dict[str, float] = {}
    length: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        name, depth, overlap = f[3], float(f[7]), int(f[8])
        total[name] = total.get(name, 0.0) + depth * overlap
        length[name] = int(f[2]) - int(f[1])
    return {k: round(total[k] / length[k], 4) for k in total}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "windows.bed").write_text(
        "".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in plan["windows"])
    )
    (work / "coverage.bedgraph").write_text(bedgraph_text(plan, twin))
    lines = [
        "windows.bed: analysis windows, BED4 (chrom, start, end, name), sorted.",
        "coverage.bedgraph: read depth, bedGraph (chrom, start, end, depth), sorted.",
    ]
    if tier == "explicit":
        lines.append(
            "The depth is constant within a bedGraph line and every base of the windows is "
            "covered by exactly one line."
        )
    (work / "README.md").write_text(common.readme(lines))
    expected = per_base_means(plan)
    naive = expected if twin == "clean" else map_means(tools, work)
    if twin == "clean":
        evidence: tuple[str, ...] = ()
    else:
        evidence = (
            "coverage.bedgraph has run-length lines of 10-130 bases that cross window "
            "boundaries: bedtools map -o mean averages the line values, not the bases",
        )
    what = (
        "the mean per-base depth over the window (the depth of each base of the window, averaged "
        "over all the bases of the window)"
    )
    prompt = common.prompt_for(
        route,
        f"report, for each window in windows.bed, {what}, from coverage.bedgraph",
        f"use `bedtools map` to report, for each window in windows.bed, {what}, from "
        "coverage.bedgraph",
        "a JSON object mapping each window name to its mean depth (a number)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": TOL}
    task_id = common.task_id("F66", seed, twin, tier, route)
    return Task(task_id, "F66", "weighting", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(
        weighted_means(tools, work), "bedtools intersect -wo, depth weighted by the overlap length"
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(map_means(tools, work), "bedtools map -c 4 -o mean")


MAP_OP = re.compile(r"(?<![\w-])-o\s*(?:mean|sum|median)\b")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command runs `bedtools map` with -o mean, sum or median on the track."""
    for command in commands:
        for part in pipelines(command):
            if re.search(r"\bbedtools\s+map\b", part) and MAP_OP.search(part):
                return True
    return False


SPEC = FamilySpec(
    id="F66",
    slot="weighting",
    source="bedtools map semantics (a value per overlapping record, not per base)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"coverage.bedgraph"}),
    locus="tool_default",
)
