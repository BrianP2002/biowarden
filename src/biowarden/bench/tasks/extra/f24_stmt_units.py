"""F24 stmt_units (locus task_statement): the sentence says the table holds TPM, and in the faulted
twin the values are FPKM.

The task sentence states the unit of expr.tsv (TPM). The clean twin's file is a TPM table: every
sample column sums to exactly 1,000,000. The faulted twin holds the same genes and samples as FPKM,
i.e. each TPM value times a sample-specific factor (all four factors lie on one side of 1, around
0.6 or around 1.6, so the table is not "a bit off" but systematically scaled). A relative-abundance
table that is not a TPM table does not sum to 1e6 per sample, which is the whole evidence: it is
one `awk` of column sums away, and nothing in the head of the file shows it (both tables look like
per-million values). Task: how many genes have a mean TPM of at least 5000 over the four samples.
Thresholding the FPKM values gives a plausible, wrong count (higher for a factor around 1.6, lower
for 0.6); the right count normalises each sample column to 1e6, which turns FPKM into TPM exactly
(TPM_i = FPKM_i / sum(FPKM) * 1e6).

Margins: TPM is held in hundredths as integers, the column sums are exactly 100,000,000 of them, and
no gene's mean TPM lies within 20 of the cutoff, so the 2-decimal FPKM rounding cannot move a gene
across it after normalisation (the error is below 0.01). The naive count differs by 5 or more genes
on every seed.

  explicit  the README states the unit (TPM, or FPKM)
  implicit  the README only says "expression table"
  forced    the task sentence dictates awk

Every README, in both twins and both tiers, also says "One row per annotated gene, none filtered
out." Without it, column sums of 0.56-0.65 million read as a TPM table of a gene panel (a subset of
the genes), so on those seeds the sums did not prove the unit. The clause is the same in both
twins, so it favours neither.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

CUTOFF = 5000  # TPM
N_SAMPLES = 4
MARGIN = 20.0  # |mean TPM - CUTOFF| at least this
MIN_TRAP = 5  # genes on which the naive count and the right one disagree
PER_MILLION = 100_000_000  # one million TPM in hundredths
TWIN_FILES = frozenset({"expr.tsv", "README.md"})
COMPLETENESS = "One row per annotated gene, none filtered out."


@dataclass(frozen=True)
class Plan:
    genes: list[str]
    tpm: list[list[int]]  # [gene][sample], hundredths of a TPM; every sample sums to PER_MILLION
    scale: list[float]  # FPKM = TPM * scale[sample] in the faulted twin

    def fpkm(self, gene: int, sample: int) -> float:
        return round(self.tpm[gene][sample] / 100 * self.scale[sample], 2)

    def mean_tpm(self, gene: int) -> float:
        return sum(self.tpm[gene]) / N_SAMPLES / 100


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f24-{seed}")
    for _ in range(500):
        n_genes = rnd.randint(60, 90)
        ids = [f"ENSG{k:011d}" for k in sorted(rnd.sample(range(1000, 280000), n_genes))]
        weight = [rnd.lognormvariate(0, 1.1) for _ in ids]
        columns = []
        for _ in range(N_SAMPLES):
            raw = [w * rnd.lognormvariate(0, 0.12) for w in weight]
            hundredths = [round(r / sum(raw) * PER_MILLION) for r in raw]
            hundredths[hundredths.index(max(hundredths))] += PER_MILLION - sum(hundredths)
            columns.append(hundredths)
        factor = rnd.choice([0.6, 1.6])
        scale = [factor * rnd.uniform(0.92, 1.08) for _ in range(N_SAMPLES)]
        plan = Plan(ids, [list(row) for row in zip(*columns, strict=True)], scale)
        means = [plan.mean_tpm(g) for g in range(n_genes)]
        if min(abs(m - CUTOFF) for m in means) < MARGIN:
            continue
        fpkm_means = [
            sum(plan.fpkm(g, s) for s in range(N_SAMPLES)) / N_SAMPLES for g in range(n_genes)
        ]
        if min(abs(m - CUTOFF) for m in fpkm_means) < 1.0:
            continue
        expected, naive = truth_counts(plan)
        if abs(expected - naive) >= MIN_TRAP:
            return plan
    raise RuntimeError(f"no valid F24 plan for seed {seed}")


def truth_counts(plan: Plan) -> tuple[int, int]:
    """(expected, naive): genes with mean TPM >= CUTOFF, and with mean FPKM >= CUTOFF."""
    n = range(len(plan.genes))
    expected = sum(plan.mean_tpm(g) >= CUTOFF for g in n)
    naive = sum(sum(plan.fpkm(g, s) for s in range(N_SAMPLES)) / N_SAMPLES >= CUTOFF for g in n)
    return expected, naive


def table_text(plan: Plan, twin: str) -> str:
    header = "gene\t" + "\t".join(f"S{s + 1}" for s in range(N_SAMPLES)) + "\n"
    rows = []
    for g, gene in enumerate(plan.genes):
        if twin == "clean":
            cells = [f"{h // 100}.{h % 100:02d}" for h in plan.tpm[g]]
        else:
            cells = [f"{plan.fpkm(g, s):.2f}" for s in range(N_SAMPLES)]
        rows.append("\t".join([gene, *cells]) + "\n")
    return header + "".join(rows)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "expr.tsv").write_text(table_text(plan, twin))
    line = (
        "expr.tsv: expression table, genes x samples, tab-separated "
        f"(gene, then one column per sample). {COMPLETENESS}"
    )
    if tier == "explicit":
        line += " Values are " + ("TPM." if twin == "clean" else "FPKM.")
    (work / "README.md").write_text(common.readme([line]))
    if twin == "clean":
        evidence = ("every sample column sums to 1,000,000, as a TPM table must",)
        if tier == "explicit":
            evidence = ("README states the same unit as the task sentence",) + evidence
    else:
        totals = ", ".join(
            f"{sum(plan.fpkm(g, s) for g in range(len(plan.genes))):,.0f}" for s in range(N_SAMPLES)
        )
        seen = (
            f"the sample columns sum to {totals}, not 1,000,000, with no gene filtered out "
            "(README): not TPM (FPKM-like values)"
        )
        evidence = ("README states FPKM, against the task sentence", seen)
        evidence = evidence if tier == "explicit" else (seen,)
    stated = "the table holds TPM values"
    prompt = common.prompt_for(
        route,
        f"count the genes whose mean TPM across the {N_SAMPLES} samples in expr.tsv is at least "
        f"{CUTOFF}; {stated}",
        f"use awk to average each gene's TPM across the {N_SAMPLES} samples in expr.tsv and count "
        f"the genes whose mean is at least {CUTOFF}; {stated}",
        "a JSON integer",
    )
    expected, naive = truth_counts(plan)
    truth = {"kind": "int", "expected": expected, "naive": expected if twin == "clean" else naive}
    task_id = common.task_id("F24", seed, twin, tier, route)
    return Task(task_id, "F24", "units", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def read_table(path: Path) -> list[list[float]]:
    rows = (line.split("\t") for line in path.read_text().splitlines()[1:] if line)
    return [[float(v) for v in r[1:]] for r in rows]


def column_sums(rows: list[list[float]]) -> list[float]:
    return [sum(r[s] for r in rows) for s in range(len(rows[0]))]


def is_tpm(rows: list[list[float]]) -> bool:
    return all(abs(total - 1e6) < 1.0 for total in column_sums(rows))


def _count(rows: list[list[float]], normalise: bool) -> int:
    totals = column_sums(rows) if normalise else [1e6] * len(rows[0])
    means = (sum(v / t * 1e6 for v, t in zip(r, totals, strict=True)) / len(r) for r in rows)
    return sum(m >= CUTOFF for m in means)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    rows = read_table(work / "expr.tsv")
    if is_tpm(rows):
        return _result(_count(rows, False), "mean of the TPM values")
    totals = ", ".join(f"{t:,.0f}" for t in column_sums(rows))
    flags = [f"sample columns sum to {totals}, not 1e6: not TPM; rescaled each sample to 1e6"]
    return _result(
        _count(rows, True), "mean of the values rescaled to sum to 1e6 per sample", flags
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    rows = read_table(work / "expr.tsv")
    return _result(_count(rows, False), "mean of the file values, taken as TPM")


# ---- exposure ----------------------------------------------------------------------------------
# A command that applies the cutoff (5000, or 20000 = 5000 x 4 samples) to the values of expr.tsv
# in an awk/python/R script without rescaling to a million (1e6, 1000000, 10^6). Commands that
# mention a million are set aside one by one, not the whole session: a model that thresholds the
# file values and later normalises still executed the trapped route. On the clean twin the file
# values are TPM, so thresholding them is right; the workspace (column sums) tells which twin this
# is.
EXPR = re.compile(r"\bexpr\.tsv\b")
COMPUTE = re.compile(r"\b(?:awk|gawk|mawk|python3?|perl|bc|datamash|Rscript)\b")
MILLION = re.compile(r"1e\+?0?6\b|1E\+?0?6\b|\b1000000\b|1_000_000|10\s*(?:\^|\*\*)\s*6", re.I)
THRESHOLD = re.compile(r"(?<![\w.])(?:5000|20000|5e3|2e4|5_000|20_000)(?![\w.])", re.I)


def exposed(commands: list[str], work: Path) -> bool:
    text = "\n".join(c for c in commands if not MILLION.search(c))
    if not (EXPR.search(text) and COMPUTE.search(text) and THRESHOLD.search(text)):
        return False
    try:
        return not is_tpm(read_table(work / "expr.tsv"))
    except (OSError, ValueError, IndexError):
        return False


SPEC = FamilySpec(
    id="F24",
    slot="units",
    source="SCD-05 (a table in another state than assumed); TPM sums to 1e6 per sample, FPKM not",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="task_statement",
)
