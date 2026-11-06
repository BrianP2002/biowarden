"""F73 matrix_missing (locus tool_default): a bin missing from a bigWig is NaN, not zero.

`computeMatrix` (deepTools 3.5.6) fills a matrix cell with NaN when the bigWig has no interval over
the bin, and only `--missingDataAsZero` turns it into 0. A coverage track written sparsely (a
bedGraph of the covered bins converted to bigWig, `bamCoverage --skipNonCoveredRegions`) therefore
gives NaN for every bin without reads, and the usual averages skip NaN: numpy `nanmean`, pandas
`mean`, the mean profile of `plotProfile`. A gene body without reads vanishes from the average.

The twins hold the same signal (coverage per 50-bp bin over 14-16 gene bodies of 20 bins, 4-5 genes
silent, a quarter of the bins of the others without reads, a little signal outside the genes). The
clean twin stores every bin of the genome (zeros included), the faulted twin stores only the bins
with reads. Task: the mean coverage per bin over all gene bodies, a bin without reads counting 0.

Verified on deepTools 3.5.6 (exit 0, stdout and stderr empty): `computeMatrix scale-regions -S
signal.bw -R genes.bed -b 0 -a 0 --regionBodyLength 1000 --binSize 50` gives a matrix of 20
columns per gene whose cells are `nan` for the absent bins; the mean of the cells that are not NaN
is 1.7 to 2.3 times the true mean on seeds 0-29 (3.5 coverage units too high at least); with
`--missingDataAsZero` the matrix has the zeros and its mean is the truth. The dense twin gives the
truth with or without the flag.

  explicit  the README says whether bins without reads are stored (clean) or absent (faulted)
  implicit  the README only says "read coverage per 50-bp bin"; the bigWig itself is the evidence
  forced    the task sentence dictates computeMatrix
"""

from __future__ import annotations

import json
import math
import random
import re
import struct
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import pipelines
from biowarden.bench.tasks.tools import ToolEnv

BIN = 50
GENE_BINS = 20
GENE_LEN = BIN * GENE_BINS
SLOT = 2500
CHROMS = {"chr1": 25000, "chr2": 15000}
TOL = 0.02
TWIN_FILES = frozenset({"signal.bw", "README.md"})
BIGWIG_MAGIC = 0x888FFC26

WRITER = """
import json, sys
import pyBigWig
spec = json.load(sys.stdin)
bw = pyBigWig.open("signal.bw", "w")
bw.addHeader([tuple(c) for c in spec["chroms"]], maxZooms=0)
bw.addEntries(spec["chrom"], spec["start"], ends=spec["end"], values=spec["value"])
bw.close()
"""


@dataclass(frozen=True)
class Gene:
    name: str
    chrom: str
    start: int
    strand: str
    bins: tuple[int, ...]  # coverage of each of the 20 bins (0 = no reads)


@dataclass(frozen=True)
class Plan:
    genes: tuple[Gene, ...]
    flanks: dict[tuple[str, int], int]  # (chrom, bin start) -> coverage, outside the gene bodies

    def signal(self) -> dict[tuple[str, int], int]:
        out = dict(self.flanks)
        for g in self.genes:
            for i, v in enumerate(g.bins):
                out[(g.chrom, g.start + i * BIN)] = v
        return out


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f73-{seed}")
    slots = [(c, k * SLOT) for c, n in CHROMS.items() for k in range(n // SLOT)]
    n_genes = 14 + seed % 3
    chosen = sorted(rng.sample(range(len(slots)), n_genes))
    silent = set(rng.sample(range(n_genes), 4 + seed % 2))
    genes: list[Gene] = []
    flanks: dict[tuple[str, int], int] = {}
    for i, k in enumerate(chosen):
        chrom, slot = slots[k]
        start = slot + 750 + BIN * rng.randint(0, 10)
        if i in silent:
            bins = (0,) * GENE_BINS
        else:
            while True:
                bins = tuple(0 if rng.random() < 0.25 else rng.randint(2, 15) for _ in range(20))
                if sum(1 for b in bins if b) >= 8:
                    break
        genes.append(Gene(f"gene{i + 1:02d}", chrom, start, rng.choice("+-"), bins))
        for j in range(10):  # 500 bp on each side
            for pos in (start - 500 + j * BIN, start + GENE_LEN + j * BIN):
                if rng.random() < 0.5:
                    flanks[(chrom, pos)] = rng.randint(1, 4)
    return Plan(tuple(genes), flanks)


def truth_values(plan: Plan) -> tuple[float, float]:
    """(true mean over all gene bins, mean over the bins that have an entry in a sparse track)."""
    cells = [v for g in plan.genes for v in g.bins]
    total = sum(cells)
    return total / len(cells), total / sum(1 for v in cells if v)


def track_json(plan: Plan, sparse: bool) -> str:
    signal = plan.signal()
    rows = []
    for chrom, size in CHROMS.items():
        for pos in range(0, size, BIN):
            v = signal.get((chrom, pos), 0)
            if sparse and v == 0:
                continue
            rows.append((chrom, pos, v))
    return json.dumps(
        {
            "chroms": [[c, n] for c, n in CHROMS.items()],
            "chrom": [r[0] for r in rows],
            "start": [r[1] for r in rows],
            "end": [r[1] + BIN for r in rows],
            "value": [float(r[2]) for r in rows],
        }
    )


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    if not tools.real_bins(["python-bio"]):  # pytest reports this as a skip, not a failure
        raise unittest.SkipTest("F73 writes signal.bw with pyBigWig: python-bio is not installed")
    plan = make_plan(seed)
    sparse = twin == "faulted"
    writer = tools.run(["python-bio", "-c", WRITER], work, stdin=track_json(plan, sparse))
    common.check(writer, ["python-bio", "pyBigWig"])
    (work / "genes.bed").write_text(
        "".join(
            f"{g.chrom}\t{g.start}\t{g.start + GENE_LEN}\t{g.name}\t0\t{g.strand}\n"
            for g in plan.genes
        )
    )
    signal_line = (
        "signal.bw (bigWig): read coverage of the sample, one value per 50-bp bin (reads per "
        "bin), contigs chr1 and chr2."
    )
    if tier == "explicit":
        signal_line += (
            " Only bins with reads have an entry; a bin without reads is absent from the file."
            if sparse
            else " Every bin of the genome has an entry; a bin without reads is stored as 0."
        )
    genes_line = (
        "genes.bed: gene bodies of 1000 bp, BED6 (chrom, start, end, name, score, strand), "
        "start 0-based."
    )
    (work / "README.md").write_text(common.readme([signal_line, genes_line]))
    n_bins = sum(CHROMS.values()) // BIN
    stored = sum(1 for v in plan.signal().values() if v) if sparse else n_bins
    if sparse:
        seen = (
            f"signal.bw has entries for {stored} of {n_bins} bins: the bins without reads are "
            "absent, so computeMatrix gives NaN for them"
        )
    else:
        seen = f"signal.bw has an entry for each of the {n_bins} bins, zeros included"
    evidence = (seen,)
    if tier == "explicit":
        evidence = (
            "README states that bins without reads are " + ("absent" if sparse else "stored as 0"),
            *evidence,
        )
    what = (
        "the mean read coverage per 50-bp bin over the bodies of all genes in genes.bed (every "
        "bin of every gene body counts)"
    )
    prompt = common.prompt_for(
        route,
        f"report {what}",
        f"use deepTools computeMatrix to report {what}",
        "a JSON number (the mean coverage per bin)",
    )
    expected, naive = truth_values(plan)
    truth = {
        "kind": "float",
        "expected": round(expected, 4),
        "naive": round(naive if sparse else expected, 4),
        "tol": TOL,
    }
    task_id = common.task_id("F73", seed, twin, tier, route)
    return Task(task_id, "F73", "missing_value", twin, seed, prompt, truth, evidence, tier, route)


def matrix_cells(
    work: Path, tools: ToolEnv, *extra: str, subcommand: str = "scale-regions"
) -> list[float]:
    """Cells of the computeMatrix table for signal.bw over genes.bed (written outside `work`)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        argv = [
            "computeMatrix",
            subcommand,
            "-S",
            "signal.bw",
            "-R",
            "genes.bed",
            "--binSize",
            str(BIN),
            "-p",
            "1",
            "-o",
            str(out / "m.gz"),
            "--outFileNameMatrix",
            str(out / "m.tab"),
        ]
        if subcommand == "scale-regions":
            argv += ["-b", "0", "-a", "0", "--regionBodyLength", str(GENE_LEN)]
        else:
            argv += ["--referencePoint", "TSS", "-b", "0", "-a", str(GENE_LEN)]
        proc = tools.run([*argv, *extra], work)
        if proc.returncode != 0:
            raise RuntimeError(f"computeMatrix failed: {proc.stderr}")
        return parse_matrix((out / "m.tab").read_text())


def parse_matrix(text: str) -> list[float]:
    cells: list[float] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith(("#", "genes:")):
            continue
        cells += [float(x) for x in line.split("\t")]
    return cells


def mean_skipping_nan(cells: list[float]) -> float:
    kept = [c for c in cells if not math.isnan(c)]
    return sum(kept) / len(kept)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    cells = matrix_cells(work, tools, "--missingDataAsZero")
    flags = ["bins absent from the bigWig counted as 0 (--missingDataAsZero)"]
    return _result(
        round(sum(cells) / len(cells), 4),
        "computeMatrix --missingDataAsZero, mean over all bins",
        flags if task.twin == "faulted" else [],
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    cells = matrix_cells(work, tools)
    return _result(round(mean_skipping_nan(cells), 4), "computeMatrix, mean of the matrix cells")


SAFE = re.compile(
    r"missingDataAsZero|nan_to_num|fillna\s*\(\s*0|isnan|np\.where|\bor\s+0\b|if\s+\w+\s+is\s+None"
)
PYTHON_READERS = re.compile(
    r"bigWigSummary|bigWigAverageOverBed|multiBigwigSummary|\.stats\s*\(|\.values\s*\("
    r"|\.intervals\s*\("
)


def bases_covered(path: Path) -> int | None:
    """The bases that carry a value in a (little-endian) bigWig, from the `basesCovered` field of
    its total summary: read from the file with struct alone, so neither pyBigWig nor the engine's
    decoder decides what counts as exposure. None when the file is not such a bigWig."""
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
            if struct.unpack_from("<I", head, 0)[0] != BIGWIG_MAGIC:
                return None
            (summary,) = struct.unpack_from("<Q", head, 44)  # totalSummaryOffset
            if not summary:
                return None
            handle.seek(summary)
            (covered,) = struct.unpack("<Q", handle.read(8))
    except (OSError, struct.error):
        return None
    return covered


def is_sparse(work: Path) -> bool:
    """Does signal.bw leave bins out (the faulted twin)? A dense track covers every base."""
    covered = bases_covered(work / "signal.bw")
    return covered is not None and covered < sum(CHROMS.values())


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff some command averages the bigWig without turning absent bins into zero, on a
    sparse signal.bw: computeMatrix without --missingDataAsZero, or pyBigWig / bigWigSummary style
    readers that skip the bins without data (a command that fills NaN with 0 is safe). Judged per
    pipeline: a fill in another statement of the command does not hide the trapped call."""
    if not is_sparse(work):
        return False
    for command in commands:
        for part in pipelines(command):
            if re.search(r"\bcomputeMatrix\b", part) and "--missingDataAsZero" not in part:
                return True
            if "signal.bw" in part and PYTHON_READERS.search(part) and not SAFE.search(part):
                return True
    return False


SPEC = FamilySpec(
    id="F73",
    slot="missing_value",
    source="deepTools computeMatrix --missingDataAsZero (documented default NaN)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
