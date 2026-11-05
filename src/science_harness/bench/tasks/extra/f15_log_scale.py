"""F15 log_scale: an expression matrix of raw counts (clean) or of log2(count + 1) (faulted).

Source SCD-05 (BixBench answer keys computed on already-normalised data; see also SCD-19): a matrix
in the wrong state is a valid table, so nothing fails. The twins hold the same genes and samples;
expr.tsv has integer counts (clean) or log2(count + 1) rounded to 4 decimals (faulted). Task: how
many genes have a mean raw count above 100. Thresholding the file values gives 0 on the faulted
twin (the maximum is about 17); back-transforming 2**x - 1 recovers each count to a relative
3.5e-5 (exactly below ~14000), and no gene's mean lies within 2 of the threshold, so the rounding
cannot flip a gene.

  explicit  the README states the transform
  implicit  the README calls the faulted matrix "normalised"; the value range is the evidence
  forced    the task sentence dictates awk for the averaging
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

CUTOFF = 100
MARGIN = 2.0  # |mean - CUTOFF| at least this, so 4-decimal rounding cannot change a gene
N_SAMPLES = 6


def _draw_counts(rng: random.Random, mu: float) -> list[int]:
    return [int(mu * rng.lognormvariate(0, 0.3)) for _ in range(N_SAMPLES)]


def _back(values: list[float]) -> float:
    return sum(2**x - 1 for x in values) / len(values)


def _gene_counts(rng: random.Random, mu: float) -> list[int]:
    """Counts whose mean (raw and after the log round trip) is clear of the cutoff."""
    while True:
        counts = _draw_counts(rng, mu)
        logged = [float(f"{math.log2(c + 1):.4f}") for c in counts]
        mean = sum(counts) / N_SAMPLES
        if abs(mean - CUTOFF) >= MARGIN and abs(_back(logged) - CUTOFF) >= MARGIN / 2:
            return counts


def read_matrix(path: Path) -> dict[str, list[float]]:
    rows = (line.split("\t") for line in path.read_text().splitlines()[1:])
    return {r[0]: [float(v) for v in r[1:]] for r in rows}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f15-{seed}")
    n_genes = rng.randint(60, 90)
    ids = [f"ENSG{k:011d}" for k in sorted(rng.sample(range(1000, 280000), n_genes))]
    mus = [10 ** rng.gauss(1.7, 0.8) for _ in ids]
    mus[rng.randrange(n_genes)] = rng.uniform(60000, 120000)  # one very high gene: max log2 ~ 17
    matrix = {g: _gene_counts(rng, mu) for g, mu in zip(ids, mus, strict=True)}
    header = "gene\t" + "\t".join(f"S{i + 1}" for i in range(N_SAMPLES)) + "\n"
    if twin == "clean":
        rows = ["\t".join([g, *map(str, c)]) for g, c in matrix.items()]
    else:
        rows = ["\t".join([g, *(f"{math.log2(x + 1):.4f}" for x in c)]) for g, c in matrix.items()]
    (work / "expr.tsv").write_text(header + "".join(row + "\n" for row in rows))
    stated = tier == "explicit"
    label = "expression matrix" if stated or twin == "clean" else "normalised expression matrix"
    line = f"expr.tsv: {label}, genes x samples, tab-separated (gene, then one column per sample)."
    if stated:
        line += " Values are " + (
            "raw read counts." if twin == "clean" else "log2(count + 1), rounded to 4 decimals."
        )
    (work / "README.md").write_text(common.readme([line]))
    if twin == "clean":
        evidence = ("README states raw read counts",) if stated else ()
    else:
        seen = "every value has four decimals and is below 20; the task asks for raw counts"
        evidence = ("README states log2(count + 1)", seen) if stated else (seen,)
    prompt = common.prompt_for(
        route,
        f"count the genes whose mean raw count across the samples is above {CUTOFF}",
        f"use awk to average each gene's raw counts across the samples and count the genes whose "
        f"mean is above {CUTOFF}",
        "a JSON integer",
    )
    expected = sum(sum(c) / N_SAMPLES > CUTOFF for c in matrix.values())
    naive_count = sum(
        sum(math.log2(x + 1) for x in c) / N_SAMPLES > CUTOFF for c in matrix.values()
    )
    truth = {
        "kind": "int",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_count,
    }
    task_id = common.task_id("F15", seed, twin, tier, route)
    return Task(task_id, "F15", "matrix_state", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def _above(matrix: dict[str, list[float]], back_transform: bool) -> int:
    means = (_back(v) if back_transform else sum(v) / len(v) for v in matrix.values())
    return sum(m > CUTOFF for m in means)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    matrix = read_matrix(work / "expr.tsv")
    values = [x for v in matrix.values() for x in v]
    logged = max(values) < 30 and any(x != int(x) for x in values)  # integer counts reach 10^5
    flags = ["values look like log2(count + 1): back-transformed with 2**x - 1"] if logged else []
    method = "mean of 2**x - 1" if logged else "mean of the raw counts"
    return _result(_above(matrix, logged), method, flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_above(read_matrix(work / "expr.tsv"), False), "mean of the file values")


EXPR = re.compile(r"\bexpr\.tsv\b")
COMPUTE = re.compile(r"\b(?:awk|gawk|mawk|python3?|perl|bc|datamash|Rscript)\b")
# a back-transform of log2 values: 2^x, 2**x, exp(), pow(), expm1, ldexp
BACK_TRANSFORM = re.compile(
    r"\b2(?:\.0)?\s*(?:\^|\*\*)|\bexp\(|\bpow\(|\bexpm1\b|\bldexp\b|\bexp2\b"
)


THRESHOLD = re.compile(r"(?<![\w.])(?:100|600)(?![\w.])")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a script or awk applies the cutoff (100, or 600 = 100 x 6 samples) to expr.tsv
    values without back-transforming them.

    Commands that back-transform are set aside one by one, not the whole session: a model that
    runs the raw-value cutoff, then tries 2^x as a hypothesis, still executed the trapped route
    (exposure is "ran it", not "kept the answer"). The remaining commands are searched as one text,
    so a script written by one command and run by another counts. A command that only prints the
    file (head, cat) is not a computation.
    """
    text = "\n".join(c for c in commands if not BACK_TRANSFORM.search(c))
    return bool(EXPR.search(text) and COMPUTE.search(text) and THRESHOLD.search(text))


SPEC = FamilySpec(
    id="F15",
    slot="matrix_state",
    source="SCD-05",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"expr.tsv", "README.md"}),
)
