"""F69 window_gap (distance_unit, tool default): bedtools distances are one more than the gap.

`bedtools window -w N` extends each a interval by N bases and asks for overlaps, and
`bedtools closest -d` reports the distance as the difference of two coordinates that count the first
base after the gap: an interval whose gap (the bases strictly between the two intervals) is G is at
distance G + 1, book-ended intervals are at distance 1 and only overlap gives 0. So "within 1000
bases" in the sense "at most 1000 bases lie between them" needs `-w 1001` or `$NF <= 1001`; the
literal `-w 1000` / `$NF <= 1000` loses every pair whose gap is exactly 1000. Exit 0, empty stderr.

Verified here (bedtools 2.31.1): peak 1000-1100 with genes at 1100-1200 (gap 0), 1700-1800 (gap
600), peak 5000-5100 with gene 4900-4950: `closest -d` prints 1 for the book-ended gene and 3801 for
a gap of 3800; `window -w 600` does NOT report the gene at 1700-1800 (gap 600) and `-w 601` does;
overlapping features get 0; a chromosome without genes gets `. -1 -1 . -1` (not used here: every
chromosome has genes). Unsorted input stops loudly ("Sorted input specified").

The twins hold the same genes and the same peaks except for the boundary group: in the faulted
twin four to six peaks sit at a gap of exactly 1000 bases from their gene, in the clean twin the
same peaks sit at a gap of 990-999 bases. Both count as "within 1000 bases", so the expected
answer is the same; only the literal `-w 1000` loses the first group. Two peaks per seed sit at
gap 1001 in both twins (not within).

  explicit  the README says that distances are counted in bases between the intervals
  implicit  the README only names the files
  forced    the sentence asks for `bedtools window`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import normalise, pipelines
from science_harness.bench.tasks.tools import ToolEnv

LIMIT = 1000


def make_plan(seed: int) -> dict[str, Any]:
    rng = random.Random(f"f69-{seed}")
    genes: list[tuple[str, int, int, str]] = []
    n = 0
    for chrom, count in (("chr1", rng.randint(16, 18)), ("chr2", rng.randint(14, 16))):
        cursor = 3000
        for _ in range(count):
            n += 1
            start = cursor + rng.randint(9000, 12000)
            end = start + rng.randint(800, 2500)
            genes.append((chrom, start, end, f"gene_{n:02d}"))
            cursor = end
    kinds = (
        ["boundary"] * rng.randint(4, 6)
        + ["just_out"] * 2
        + ["overlap"] * rng.randint(3, 5)
        + ["near"] * rng.randint(7, 9)
        + ["far"] * rng.randint(6, 8)
    )
    kinds += ["none"] * (len(genes) - len(kinds))
    rng.shuffle(kinds)
    side = {g[3]: rng.choice(["up", "down"]) for g in genes}
    peaks_clean, peaks_faulted = [], []
    expected = 0
    for gene, kind in zip(genes, kinds, strict=True):
        if kind == "none":
            continue
        chrom, gs, ge, name = gene
        width = rng.randint(200, 400)
        gap_clean = gap_faulted = 0
        inside = False
        if kind == "boundary":
            gap_faulted, gap_clean = LIMIT, rng.randint(LIMIT - 10, LIMIT - 1)
        elif kind == "just_out":
            gap_faulted = gap_clean = LIMIT + 1
        elif kind == "near":
            gap_faulted = gap_clean = rng.randint(50, 800)
        elif kind == "far":
            gap_faulted = gap_clean = rng.randint(1500, 4000)
        else:
            inside = True
        near = kind in ("boundary", "overlap", "near")
        expected += near
        out = []
        inner = rng.randint(gs, max(gs, ge - width))
        for gap in (gap_clean, gap_faulted):
            if inside:
                out.append((chrom, inner, inner + width))
            elif side[name] == "down":
                out.append((chrom, ge + gap, ge + gap + width))
            else:
                out.append((chrom, gs - gap - width, gs - gap))
        peaks_clean.append(out[0])
        peaks_faulted.append(out[1])
    names = [f"peak_{i + 1:02d}" for i in range(len(peaks_clean))]
    order = lambda p: (p[0], p[1])  # noqa: E731
    clean = [(*p, nm) for p, nm in zip(peaks_clean, names, strict=True)]
    faulted = [(*p, nm) for p, nm in zip(peaks_faulted, names, strict=True)]
    return {
        "genes": genes,
        "clean": sorted(clean, key=order),
        "faulted": sorted(faulted, key=order),
        "expected": expected,
    }


def write_bed(path: Path, records: list[tuple[str, int, int, str]]) -> None:
    path.write_text("".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in records))


def window_count(tools: ToolEnv, work: Path, width: int) -> int:
    argv = [
        "bedtools", "window", "-a", "peaks.bed", "-b", "genes.bed", "-w", str(width), "-u",
    ]  # fmt: skip
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_bed(work / "genes.bed", plan["genes"])
    write_bed(work / "peaks.bed", plan[twin])
    lines = [
        "genes.bed: gene bodies, BED4 (chrom, start, end, name), sorted.",
        "peaks.bed: ChIP-seq peaks, BED4 (chrom, start, end, name), sorted.",
    ]
    if tier == "explicit":
        lines.append("Distances between intervals are counted in bases lying between them.")
    (work / "README.md").write_text(common.readme(lines))
    expected = plan["expected"]
    naive = expected if twin == "clean" else window_count(tools, work, LIMIT)
    evidence: tuple[str, ...] = (
        ()
        if twin == "clean"
        else (
            f"{expected - naive} peaks lie exactly 1000 bases from a gene: bedtools window -w 1000 "
            "reaches only a gap of 999",
        )
    )
    what = (
        "how many peaks of peaks.bed have a gene of genes.bed within 1000 bp (at most 1000 bases "
        "lie between the peak and the gene; a peak that overlaps a gene counts)"
    )
    prompt = common.prompt_for(
        route,
        f"report {what}",
        f"use `bedtools window` to report {what}",
        "a JSON integer",
    )
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F69", seed, twin, tier, route)
    return Task(task_id, "F69", "distance_unit", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(
        window_count(tools, work, LIMIT + 1),
        "bedtools window -w 1001 (a gap of 1000 is distance 1001)",
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(window_count(tools, work, LIMIT), "bedtools window -w 1000")


LITERAL_WINDOW = re.compile(r"\bbedtools\s+window\b.*(?<![\w-])-[wlr]\s*1000(?!\d)")
# the limit 1000 compared with the distance itself (`$NF <= 1000`, `d < 1001`): a shifted distance
# (`$NF-1 <= 1000`) is the corrected form and does not match
LITERAL_CLOSEST = re.compile(r"(?:\$\(?\w+\)?|[A-Za-z_]\w*)\s*(?:<=?\s*1000\b|<\s*1001\b)")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a command reaches 1000 bases with the literal bedtools window size (-w, -l or
    -r 1000), or thresholds closest -d at 1000 (`<= 1000`, `< 1001`) anywhere in the command."""
    for command in commands:
        if any(LITERAL_WINDOW.search(part) for part in pipelines(command)):
            return True
        text = normalise(command)
        if re.search(r"\bbedtools\s+closest\b", text) and LITERAL_CLOSEST.search(text):
            return True
    return False


SPEC = FamilySpec(
    id="F69",
    slot="distance_unit",
    source="bedtools window/closest distance is the gap plus one (documented only for overlap = 0)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"peaks.bed"}),
    locus="tool_default",
)
