"""F61 tabix_bed (index_base, tool default): tabix reads BED coordinates as 1-based unless told.

tabix treats the coordinates of a text file as 1-based and closed. A BED file is 0-based and
half-open, so it needs `tabix -p bed` (or `-0`). Three routes that look natural give a one-base
shift at the edges of the query, silently: (1) `tabix -s 1 -b 2 -e 3 FILE` without -0 (the columns
are the GFF-default reading), (2) a region string typed with the BED window, chr1:100-200 for the
window [100, 200) (a region string is always 1-based and closed, so the interval that ends at 100
is returned), (3) `tabix -R windows.tsv`, because only a name ending in .bed makes -R read BED
(the bcftools -R rule, GEN-01, on tabix). Exit 0, a plausible list.

Task: list the names of the intervals of intervals.bed that overlap at least one window of
windows.tsv; both files hold BED coordinates (0-based start, end exclusive), so an interval that
only touches a window does not overlap it. The clean twin has no interval that touches a window.
The faulted twin adds 5-6 extra intervals that end exactly where a window starts or start exactly
where one ends: not overlapping, but returned by the routes above (the first and the third route
report those that end at a window start, the indexing route those that start at a window end).
Interval names run b000, b001, ... over the position-ordered union of the base intervals and the
extras (the clean twin leaves the extras out, so its numbering skips them): no name marks an extra.

Verified here (tabix/bgzip 1.24, bedtools 2.31.1): see tests/test_family_var.py and the module
docstring of engine/rules/var_tabix.py; on the family data `bedtools intersect -u` gives the truth,
`tabix -p bed` + `-R windows.bed` too, `tabix -R windows.tsv` adds the intervals that end at a
window start.

  explicit  the README says that touching is not overlap (0-based start, end exclusive)
  implicit  the README names the files and their format (BED4, BED-style); that is the only hint
  forced    the sentence asks for bgzip, `tabix -p bed` or -s/-b/-e, and `tabix -R windows.tsv`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import normalise, result, segments
from science_harness.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 4000, "chr2": 3000}
N_WINDOWS = 6


def make_plan(seed: int):
    """(windows, base intervals, extras): windows (chrom, start0, end0); intervals (name, chrom,
    start0, end0); extras are the touching intervals of the faulted twin ((name, ..., 'left' |
    'right') where 'left' ends at a window start and 'right' starts at a window end). The names
    b000... number the position-ordered union of both, the same in the two twins."""
    rng = random.Random(f"f61-{seed}")
    windows: list[tuple[str, int, int]] = []
    for chrom, size in CONTIGS.items():
        starts = sorted(rng.sample(range(200, size - 600, 450), N_WINDOWS // 2))
        windows += [(chrom, s, s + rng.randint(120, 260)) for s in starts]
    base: list[tuple[str, int, int]] = []
    for chrom, size in CONTIGS.items():
        pos = 20
        while pos < size - 150:
            length = rng.randint(20, 110)
            base.append((chrom, pos, pos + length))
            pos += length + rng.randint(15, 140)
    edges = {(c, p) for c, s, e in windows for p in range(s - 4, s + 5)}
    edges |= {(c, p) for c, s, e in windows for p in range(e - 4, e + 5)}
    base = [
        (c, s, e)
        for c, s, e in base
        if not any((c, p) in edges for p in (s, e))
        and not any(c == w[0] and s <= w[1] + 1 and e >= w[1] - 1 for w in windows)
    ]
    # an interval may span a window edge, but none starts or ends within 4 bases of one
    touching = []
    picks = rng.sample(range(len(windows)), min(len(windows), rng.randint(5, 6)))
    for k, w in enumerate(picks):
        c, s, e = windows[w]
        side = "left" if k % 2 == 0 else "right"
        if side == "left":
            touching.append((c, s - rng.randint(20, 60), s, side))
        else:
            touching.append((c, e, e + rng.randint(20, 60), side))
    order = {c: i for i, c in enumerate(CONTIGS)}
    union = sorted(
        [*base, *((c, s, e) for c, s, e, _side in touching)], key=lambda r: (order[r[0]], *r[1:])
    )
    names = {row: f"b{i:03d}" for i, row in enumerate(union)}
    named = [(names[row], *row) for row in base]
    extras = [(names[(c, s, e)], c, s, e, side) for c, s, e, side in touching]
    return windows, named, extras


def overlapping(windows, intervals) -> list[str]:
    return sorted(
        {n for n, c, s, e in intervals for wc, ws, we in windows if c == wc and s < we and e > ws}
    )


def interval_text(named, extras, twin: str) -> str:
    rows = [(c, s, e, n) for n, c, s, e in named]
    if twin == "faulted":
        rows += [(c, s, e, n) for n, c, s, e, _side in extras]
    order = {c: i for i, c in enumerate(CONTIGS)}
    rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    return "".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in rows)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    windows, named, extras = make_plan(seed)
    (work / "intervals.bed").write_text(interval_text(named, extras, twin))
    (work / "windows.tsv").write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in windows))
    lines = [
        "intervals.bed: annotated intervals, BED4 (chrom, start, end, name), sorted.",
        "windows.tsv: query windows, BED-style (chrom, start, end), tab-separated, sorted.",
    ]
    if tier == "explicit":
        lines.append(
            "Both files hold BED coordinates: 0-based start, end exclusive. An interval that "
            "only touches a window (its end is the window's start, or its start the window's "
            "end) does not overlap it."
        )
    (work / "README.md").write_text(common.readme(lines))
    n_left = sum(1 for x in extras if x[4] == "left")
    if twin == "clean":
        evidence = ("no interval of intervals.bed starts or ends at the edge of a window",)
    else:
        evidence = (
            f"{len(extras)} intervals touch a window ({n_left} end at a window start)",
            "tabix reads region strings and -R files with another name than .bed as 1-based",
        )
    prompt = common.prompt_for(
        route,
        "list the names of the intervals in intervals.bed that overlap at least one window of "
        "windows.tsv",
        "use bgzip and tabix to list the names of the intervals in intervals.bed that overlap at "
        "least one window of windows.tsv",
        "a JSON list of interval names",
    )
    expected = overlapping(windows, named)
    touching_left = [x[0] for x in extras if x[4] == "left"]
    naive = sorted(expected + touching_left) if twin == "faulted" else expected
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F61", seed, twin, tier, route)
    return Task(task_id, "F61", "index_base", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def names(stdout: str) -> list[str]:
    return sorted({line.split("\t")[3] for line in stdout.splitlines() if line.count("\t") >= 3})


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bedtools", "intersect", "-u", "-a", "intervals.bed", "-b", "windows.tsv"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return result(names(proc.stdout), "bedtools intersect -u (BED semantics)")


def tabix_names(work: Path, tools: ToolEnv, index: list[str], query: list[str]) -> list[str]:
    for argv in (
        ["bgzip", "-k", "-f", "intervals.bed"],
        ["tabix", "-f", *index, "intervals.bed.gz"],
    ):
        common.check(tools.run(argv, work), argv)
    argv = ["tabix", "intervals.bed.gz", *query]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return names(proc.stdout)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    found = tabix_names(work, tools, ["-p", "bed"], ["-R", "windows.tsv"])
    return result(found, "bgzip + tabix -p bed, tabix -R windows.tsv")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped, per command: (1) tabix indexing by columns (-s/-b/-e) without -p or -0; (2) a tabix query
# with -R/-T on a file that is not named .bed, or with a region string (chr:start-end) typed by
# hand or put together from the three columns of a window ($1":"$2"-"$3, $c:$s-$e, {c}:{s}-{e}),
# and no arithmetic (+ 1, $((...)), awk $2+1) in the command. bedtools and bcftools are safe.
COLUMN_INDEX = re.compile(r"\btabix\b[^|;&\n]*\s-[sbe]\s*\d")
INDEX_FIXED = re.compile(r"\s-p\b|--preset|\s-0\b|--zero-based")
REGIONS_FILE = re.compile(r"\btabix\b[^|;&\n]*(?:\s-[RT]\s*|--regions[\s=]|--targets[\s=])(\S+)")
REGION_STRING = re.compile(r"\btabix\b[^|;&\n]*\s[\w.]+:\d+-\d+")
COLUMN = r"(?:\$\{?\w+\}?|\{\w+\}|%[sd])"
BUILT_REGION = re.compile(rf"{COLUMN}[\"']?:[\"']?{COLUMN}[\"']?-[\"']?{COLUMN}")
ARITHMETIC = re.compile(r"\+\s*1\b|\$\(\(|\$2\s*\+|\bexpr\b|\bbc\b|int\(|\+1\b")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            if "tabix" not in segment:
                continue
            if COLUMN_INDEX.search(segment) and not INDEX_FIXED.search(segment):
                return True
            hit = REGIONS_FILE.search(segment)
            if hit and not hit.group(1).strip("'\"").lower().endswith((".bed", ".bed.gz")):
                return True
            if REGION_STRING.search(segment) and not ARITHMETIC.search(segment):
                return True
        text = normalise(command)
        if re.search(r"\btabix\b", text) and BUILT_REGION.search(text):
            if not ARITHMETIC.search(text):  # the tabix call may sit in another stage
                return True
    return False


SPEC = FamilySpec(
    id="F61",
    slot="index_base",
    source="GEN-04 (htslib#407), GEN-01; tabix manual (-0, -p bed, -R)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"intervals.bed", "README.md"}),
    locus="tool_default",
)
