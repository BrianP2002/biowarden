"""F70 bedops_order (locus data): BEDOPS needs sort-bed order and checks nothing without --ec.

`bedmap`, `bedops` and `closest-features` assume that every input is in the order `sort-bed`
writes: chromosome names compared as plain strings (chr1, chr10, chr11, chr2, ...), then start,
then end. They do not check it. A BED file in genome order (chr1, chr2, ..., chr10, ...), the order
`sort -V`, `samtools faidx` or a genome file give, is accepted and silently mis-processed.

Verified on bedmap 2.4.42 (bio-a), two BED files in genome order against the same files written by
sort-bed: `bedmap --echo --count regions.bed peaks.bed` exits 0 with an empty stderr both ways;
on the genome-order pair the regions of chr10, chr11, chr12 get counts of 0 (or the counts of other
chromosomes), on the sort-bed pair every count equals the overlap count computed in Python.
`bedmap --ec` stops with "Bed file not properly sorted by first column" (exit 1). A file in the
wrong order on one side only is wrong as well. `bedops -e 1` returned 2 of 4 elements and
`closest-features` printed NA for chr10 and chr11 on the genome-order pair; both exit 0 with no
message.

The twins hold the same regions and peaks; only the order of the lines of the two BED files differs
(sort-bed order in the clean twin, genome order in the faulted one). Task: count, for each region,
the peaks that overlap it by at least one base.

  explicit  the README states how the files are sorted
  implicit  the README only says "BED"; the order of the chromosomes is the evidence
  forced    the task sentence dictates bedmap
"""

from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import normalise
from biowarden.bench.tasks.tools import ToolEnv

CHROMS = ("chr1", "chr2", "chr3", "chr4", "chr9", "chr10", "chr11", "chr12", "chrX")
GENOME_ORDER = {c: i for i, c in enumerate(CHROMS)}
TWIN_FILES = frozenset({"regions.bed", "peaks.bed", "README.md"})
Interval = tuple[str, int, int, str]


def sort_bed_key(row: Interval) -> tuple[str, int, int]:
    return (row[0], row[1], row[2])


def genome_key(row: Interval) -> tuple[int, int, int]:
    return (GENOME_ORDER[row[0]], row[1], row[2])


def make_instance(seed: int) -> tuple[list[Interval], list[Interval]]:
    """(regions, peaks): every chromosome carries 2-3 regions and peaks on and between them."""
    rng = random.Random(f"f70-{seed}")
    regions: list[Interval] = []
    peaks: list[Interval] = []
    for chrom in CHROMS:
        n_regions = rng.randint(2, 3)
        starts = sorted(rng.sample(range(1000, 30000, 2000), n_regions))
        for start in starts:
            end = start + rng.randint(300, 700)
            regions.append((chrom, start, end, ""))
            for _ in range(rng.randint(1, 4)):
                length = rng.randint(40, 150)
                p_start = rng.randint(start - 100, end - 20)
                peaks.append((chrom, p_start, p_start + length, ""))
        for _ in range(rng.randint(2, 4)):  # background peaks between regions (never touching one)
            while True:
                p_start = rng.randint(200, 31000)
                if all(
                    not (p_start < r[2] + 200 and p_start + 150 > r[1] - 200)
                    for r in regions
                    if r[0] == chrom
                ):
                    break
            peaks.append((chrom, p_start, p_start + rng.randint(40, 150), ""))
    regions.sort(key=genome_key)
    peaks.sort(key=genome_key)
    named_regions = [(c, s, e, f"reg{i + 1:02d}") for i, (c, s, e, _) in enumerate(regions)]
    named_peaks = [(c, s, e, f"pk{i + 1:03d}") for i, (c, s, e, _) in enumerate(peaks)]
    return named_regions, named_peaks


def overlap_counts(regions: list[Interval], peaks: list[Interval]) -> dict[str, int]:
    """Peaks sharing at least one base with each region (BED half-open intervals)."""
    return {
        name: sum(1 for p in peaks if p[0] == chrom and p[1] < end and p[2] > start)
        for chrom, start, end, name in regions
    }


def bed_text(rows: list[Interval]) -> str:
    return "".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in rows)


def parse_counts(stdout: str) -> dict[str, int]:
    """`bedmap --echo --count` lines: chrom, start, end, name|count."""
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        left, _, count = line.rpartition("|")
        counts[left.split("\t")[3]] = int(count)
    return counts


def bedmap_counts(work: Path, tools: ToolEnv, regions: str, peaks: str) -> dict[str, int]:
    proc = tools.run(["bedmap", "--echo", "--count", regions, peaks], work)
    if proc.returncode != 0:
        raise RuntimeError(f"bedmap failed: {proc.stderr}")
    return parse_counts(proc.stdout)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    regions, peaks = make_instance(seed)
    order = sort_bed_key if twin == "clean" else genome_key
    (work / "regions.bed").write_text(bed_text(sorted(regions, key=order)))
    (work / "peaks.bed").write_text(bed_text(sorted(peaks, key=order)))
    if twin == "clean":
        stated = (
            " Sorted by chromosome name as text, then start (chr1, chr10, chr11, chr12, chr2, ...)."
        )
    else:
        stated = (
            " Sorted in genome order (chr1, chr2, chr3, chr4, chr9, chr10, chr11, chr12, chrX)."
        )
    note = stated if tier == "explicit" else ""
    lines = [
        "regions.bed: regions of interest, tab-separated BED4 (chrom, start, end, name)." + note,
        "peaks.bed: ChIP-seq peaks, tab-separated BED4 (chrom, start, end, name)." + note,
    ]
    (work / "README.md").write_text(common.readme(lines))
    expected = overlap_counts(regions, peaks)
    naive = expected if twin == "clean" else bedmap_counts(work, tools, "regions.bed", "peaks.bed")
    if twin == "faulted":
        evidence = (
            "regions.bed and peaks.bed list the chromosomes in genome order (chr9 before chr10), "
            "not in the text order sort-bed writes (chr10 before chr9)",
        )
    else:
        evidence = ("regions.bed and peaks.bed are in the text order sort-bed writes",)
    if tier == "explicit":
        evidence = ("README states the sort order of the BED files", *evidence)
    what = "the number of peaks in peaks.bed that overlap it by at least one base"
    prompt = common.prompt_for(
        route,
        f"report, for each region in regions.bed, {what}",
        f"use bedmap to report, for each region in regions.bed, {what}",
        "a JSON object mapping each region name (column 4 of regions.bed) to its peak count "
        "(an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F70", seed, twin, tier, route)
    return Task(task_id, "F70", "sort_order", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """sort-bed both files, then bedmap."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name in ("regions.bed", "peaks.bed"):
            proc = tools.run(["sort-bed", name], work)
            if proc.returncode != 0:
                raise RuntimeError(f"sort-bed failed: {proc.stderr}")
            path = Path(tmp) / name
            path.write_text(proc.stdout)
            paths.append(str(path))
        counts = bedmap_counts(work, tools, paths[0], paths[1])
    flags = ["BED files re-sorted with sort-bed before bedmap"] if task.twin == "faulted" else []
    return _result(counts, "sort-bed then bedmap --count", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(bedmap_counts(work, tools, "regions.bed", "peaks.bed"), "bedmap --count")


# ---- exposure --------------------------------------------------------------------------------

_STAGE = re.compile(r"\|\||&&|[|;&\n]")
_SORTED_SUBST = re.compile(r"<\((?=[^)]*\b(?:sort-bed|sort\s+-k\s*1,1|bedtools\s+sort))[^)]*\)")
_PROGRAM = re.compile(r"(?<![\w.-])(?:bedmap|bedops|closest-features)\b")
_EC = re.compile(r"(?<![\w-])--ec(?![\w-])")
_RAW = re.compile(r"(?<![\w./-])(?:\./)?(?:regions|peaks)\.bed(?![\w.-])")


def workspace_is_sort_bed_order(work: Path) -> bool:
    for name in ("regions.bed", "peaks.bed"):
        try:
            rows = [
                tuple(line.split("\t")[:3])
                for line in (work / name).read_text().splitlines()
                if line.strip()
            ]
        except OSError:
            return True
        keys = [(r[0], int(r[1]), int(r[2])) for r in rows]
        if keys != sorted(keys):
            return False
    return True


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a BEDOPS stage reads regions.bed or peaks.bed as they are, without --ec, in a
    workspace whose files are in genome order. A file re-sorted by sort-bed (a process
    substitution, a copy) or a stage that reads standard input only is the safe route; judged per
    stage, so a session that tried the raw files and then re-sorted still counts as exposed. A
    /work/ path, a line continuation and a quoted file name read as the plain command."""
    if workspace_is_sort_bed_order(work):
        return False
    for command in commands:
        for stage in _STAGE.split(_SORTED_SUBST.sub(" ", normalise(command))):
            if _PROGRAM.search(stage) and _RAW.search(stage) and not _EC.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F70",
    slot="sort_order",
    source="BEDOPS docs: input must be in sort-bed order, nothing is checked without --ec",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="data",
)
