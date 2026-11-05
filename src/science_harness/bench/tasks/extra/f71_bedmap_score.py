"""F71 bedmap_score (locus tool_default): bedmap's score operations read column 5 and nothing else.

`bedmap --mean|--sum|--median|--min|--max|...` take the value from column 5 of the map file (its
`--help`: "<map-file> 5 columns"). The map file does not have to hold the quantity you mean there. A
MACS3 narrowPeak file carries the scaled score (an integer, up to 1000) in column 5 and the fold
enrichment in column 7 (signalValue); a bedGraph has no column 5 at all.

Verified on bedmap 2.4.42 (bio-a, exit 0 and an empty stderr every time):
  narrowPeak (10 columns, F71 seed 0): `bedmap --echo --mean regions.bed peaks.narrowPeak` prints
    the mean of column 5 (the scaled scores, 143.0 where the fold enrichment is 4.7), plausible
    numbers and no message;
  bedGraph (chrom, start, end, value): `bedmap --mean` prints 0.000000 or NAN for every region,
    `--count` still counts the elements; `--echo-map-score` prints nothing.
The same peaks written as a BED5 file with the fold enrichment in column 5 give the right means.

Every peak lies wholly inside one region and every region holds 2-5 peaks, so no overlap or
weighting question (the bedtools map trap of F66) arises; only the column decides. The twins hold
the same peaks and values: the clean twin writes `peaks.bed` (BED5, column 5 = fold enrichment),
the faulted twin `peaks.narrowPeak` (the same fold enrichment in column 7, the scaled score in 5).

  explicit  the README lists the narrowPeak columns
  implicit  the README only says "narrowPeak"; the format definition is the evidence
  forced    the task sentence dictates bedmap
"""

from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f53_vcftools_bed import normalise
from science_harness.bench.tasks.tools import ToolEnv

CHROMS = ("chr1", "chr2", "chr3")
TWIN_FILES = frozenset({"peaks.bed", "peaks.narrowPeak", "README.md"})
SCORE_OPERATIONS = (
    "mean|sum|median|min|max|wmean|stdev|variance|tmean|kth|mad|cv|echo-map-score|"
    "min-element|max-element"
)


def make_instance(seed: int) -> tuple[list[tuple], list[tuple]]:
    """(regions, peaks): regions (chrom, start, end, name); peaks (chrom, start, end, name, fold,
    scaled score, -log10 p, -log10 q, summit offset), all inside one region, in sort-bed order."""
    rng = random.Random(f"f71-{seed}")
    regions: list[tuple] = []
    peaks: list[tuple] = []
    number = 0
    for chrom in CHROMS:
        starts = sorted(rng.sample(range(1000, 12000, 1500), rng.randint(2, 3)))
        for start in starts:
            end = start + rng.randint(500, 900)
            regions.append((chrom, start, end, f"win{len(regions) + 1:02d}"))
            for _ in range(rng.randint(2, 5)):
                length = rng.randint(120, 260)
                p_start = rng.randint(start + 5, end - length - 5)
                fold = round(rng.uniform(2.0, 14.0), 2)
                q = round(rng.uniform(4.0, 38.0), 3)
                number += 1
                peaks.append(
                    (
                        chrom,
                        p_start,
                        p_start + length,
                        f"peak{number:03d}",
                        fold,
                        min(1000, int(10 * q)),
                        round(q + rng.uniform(0.5, 3.0), 3),
                        q,
                        rng.randint(20, length - 20),
                    )
                )
    peaks.sort(key=lambda p: (p[0], p[1], p[2]))
    return regions, peaks


def region_of(regions: list[tuple], peak: tuple) -> str:
    return next(r[3] for r in regions if r[0] == peak[0] and r[1] <= peak[1] and peak[2] <= r[2])


def means(regions: list[tuple], peaks: list[tuple], column: int) -> dict[str, float]:
    """Mean of peak[column] over the peaks of each region (4 = fold, 5 = scaled score)."""
    out: dict[str, list[float]] = {r[3]: [] for r in regions}
    for peak in peaks:
        out[region_of(regions, peak)].append(float(peak[column]))
    return {name: round(sum(v) / len(v), 4) for name, v in out.items()}


def narrow_peak_text(peaks: list[tuple]) -> str:
    return "".join(
        f"{c}\t{s}\t{e}\t{n}\t{score}\t.\t{fold}\t{p}\t{q}\t{summit}\n"
        for c, s, e, n, fold, score, p, q, summit in peaks
    )


def bed5_text(peaks: list[tuple]) -> str:
    return "".join(f"{c}\t{s}\t{e}\t{n}\t{fold}\n" for c, s, e, n, fold, *_ in peaks)


def peaks_file(work: Path) -> str:
    return "peaks.narrowPeak" if (work / "peaks.narrowPeak").exists() else "peaks.bed"


def parse_means(stdout: str) -> dict[str, float]:
    """`bedmap --echo --mean` lines: chrom, start, end, name|mean (NAN is read as 0)."""
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        left, _, value = line.rpartition("|")
        out[left.split("\t")[3]] = 0.0 if value.upper() == "NAN" else float(value)
    return out


def bedmap_means(work: Path, tools: ToolEnv, regions: str, peaks: str) -> dict[str, float]:
    proc = tools.run(["bedmap", "--echo", "--mean", regions, peaks], work)
    if proc.returncode != 0:
        raise RuntimeError(f"bedmap failed: {proc.stderr}")
    return parse_means(proc.stdout)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    regions, peaks = make_instance(seed)
    (work / "regions.bed").write_text("".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in regions))
    if twin == "clean":
        (work / "peaks.bed").write_text(bed5_text(peaks))
        peaks_line = (
            "peaks.bed: ChIP-seq peaks with their fold enrichment, tab-separated: chrom, start, "
            "end, name, fold enrichment."
        )
    else:
        (work / "peaks.narrowPeak").write_text(narrow_peak_text(peaks))
        peaks_line = "peaks.narrowPeak: ChIP-seq peaks called by MACS3, narrowPeak format (BED6+4)."
        if tier == "explicit":
            peaks_line += (
                " Columns: chrom, start, end, name, score (scaled -10*log10 q, at most 1000), "
                "strand, signalValue (fold enrichment), -log10 p, -log10 q, summit offset."
            )
    lines = [
        "regions.bed: regions of interest, tab-separated BED4 (chrom, start, end, name); both "
        "files are sorted.",
        peaks_line + " Every peak lies inside one region.",
    ]
    (work / "README.md").write_text(common.readme(lines))
    expected = means(regions, peaks, 4)
    naive = expected if twin == "clean" else means(regions, peaks, 5)
    if twin == "faulted":
        evidence = (
            "peaks.narrowPeak holds the fold enrichment in column 7; column 5 is the scaled "
            "score, which is what bedmap averages",
        )
    else:
        evidence = ("peaks.bed holds the fold enrichment in column 5, where bedmap reads it",)
    if tier == "explicit" and twin == "faulted":
        evidence = ("README lists the narrowPeak columns", *evidence)
    what = "the mean fold enrichment of the ChIP-seq peaks that lie inside it"
    prompt = common.prompt_for(
        route,
        f"report, for each region in regions.bed, {what}",
        f"use bedmap to report, for each region in regions.bed, {what}",
        "a JSON object mapping each region name (column 4 of regions.bed) to the mean fold "
        "enrichment (a number)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.02}
    task_id = common.task_id("F71", seed, twin, tier, route)
    return Task(task_id, "F71", "score_column", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Put the fold enrichment into column 5, then bedmap --mean."""
    name = peaks_file(work)
    if name == "peaks.bed":
        return _result(bedmap_means(work, tools, "regions.bed", name), "bedmap --mean")
    rows = [line.split("\t") for line in (work / name).read_text().splitlines()]
    text = "".join("\t".join([*r[:4], r[6]]) + "\n" for r in rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fold.bed"
        path.write_text(text)
        answer = bedmap_means(work, tools, "regions.bed", str(path))
    flags = [
        "narrowPeak column 5 is the scaled score; fold enrichment (column 7) moved to column 5"
    ]
    return _result(answer, "bedmap --mean on a BED5 with the fold enrichment in column 5", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    answer = bedmap_means(work, tools, "regions.bed", peaks_file(work))
    return _result(answer, "bedmap --mean")


# ---- exposure --------------------------------------------------------------------------------

_STAGE = re.compile(r"\|\||&&|[|;&\n]")
_RESHAPED = re.compile(r"<\((?=[^)]*\b(?:awk|cut|paste|sed|perl|python3?)\b)[^)]*\)")
_BEDMAP = re.compile(r"(?<![\w.-])bedmap\b")
_OPERATION = re.compile(rf"--(?:{SCORE_OPERATIONS})(?![\w-])")
_RAW = re.compile(r"(?<![\w./-])(?:\./)?peaks\.narrowPeak(?![\w.-])")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a bedmap stage applies a score operation to peaks.narrowPeak itself. A column
    moved with awk/cut first (a process substitution, a file, a pipe) is the safe route; judged per
    stage, so a session that tried the raw file and then reshaped it still counts as exposed."""
    if not (work / "peaks.narrowPeak").exists():
        return False
    for command in commands:
        for stage in _STAGE.split(_RESHAPED.sub(" ", normalise(command))):
            if _BEDMAP.search(stage) and _OPERATION.search(stage) and _RAW.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F71",
    slot="score_column",
    source="BEDOPS bedmap --help: score operations read column 5 of the map file",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
