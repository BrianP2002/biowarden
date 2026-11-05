"""F42 fc_read_pairs (locus tool_default): `featureCounts -p` counts reads, not read pairs.

Since Subread 2.0.2 `-p` only declares that the input is paired-end; the pairs are counted as
fragments only with the extra `--countReadPairs`. The twins hold the same fragments of the same
genes (an unstranded library). The clean twin keeps read 1 of each fragment as a single-end read
(a fragment is a read, and featureCounts without `-p` is right). The faulted twin keeps both mates
as a properly paired BAM. featureCounts stops with an error when a paired BAM is given without `-p`
(so every agent adds it), and `-p` alone counts every mate: two counts per fragment.

Verified on featureCounts 2.1.1 (exit 0): on a paired BAM of 640 fragments (1280 records)
`featureCounts -p -a genes.gtf -o c.txt reads.bam` writes counts that sum to 1280 and the summary
says `Assigned 1280`; `-p --countReadPairs` sums to 640. Without `-p` the run fails (`ERROR:
Paired-end reads were detected in single-end read library`, exit 255). The stderr banner prints
`Count read pairs : no` and `The reads are assigned on the single-end mode.`, which is the only
sign (featureCounts always prints a banner on stderr, so stderr is never empty).

  explicit  the README states that reads.bam is paired-end (faulted) or single-end (clean)
  implicit  the README only says "RNA-seq reads"; `samtools flagstat` shows the pairing
  forced    the task sentence dictates featureCounts
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
from science_harness.bench.tasks.extra.f41_htseq_strand import (
    UNSTRANDED,
    counts_by_gene,
    gtf_text,
    make_fragments,
    make_layout,
    sam_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "README.md"})


def feature_counts(
    work: Path, tools: ToolEnv, *args: str, bam: str = "reads.bam"
) -> dict[str, int]:
    """Run featureCounts (outputs in a scratch directory) and return {gene: count}."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "counts.txt"
        argv = ["featureCounts", "-a", "genes.gtf", "-o", str(out), *args, bam]
        proc = tools.run(argv, work)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"featureCounts failed ({proc.returncode}): {proc.stderr[-300:]}")
        rows = [r.split("\t") for r in out.read_text().splitlines() if not r.startswith("#")]
    return {r[0]: int(float(r[-1])) for r in rows[1:]}


def is_paired(work: Path, tools: ToolEnv) -> bool:
    proc = tools.run(["samtools", "view", "-c", "-f", "1", "reads.bam"], work)
    return int(proc.stdout.strip() or 0) > 0


def make_plan(seed: int):
    layout = make_layout(f"f42-{seed}", 5 + seed % 2)
    frags = make_fragments(random.Random(f"f42-{seed}-frags"), layout, UNSTRANDED)
    return layout, frags


def truth_answers(layout, frags, twin: str) -> tuple[dict[str, int], dict[str, int]]:
    expected = counts_by_gene(layout, frags)
    factor = 1 if twin == "clean" else 2  # `-p` alone reports every mate
    return expected, {g: n * factor for g, n in expected.items()}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    layout, frags = make_plan(seed)
    paired = twin == "faulted"
    common.write_bam(work, sam_genome(layout, frags, paired, singles=not paired), tools, "reads")
    (work / "genes.gtf").write_text(gtf_text(layout))
    kind = "paired-end" if paired else "single-end"
    reads_line = "reads.bam (with reads.bam.bai): RNA-seq reads aligned to the genome."
    if tier == "explicit":
        reads_line = reads_line.replace("RNA-seq reads", f"{kind} RNA-seq reads")
        reads_line += " The library is unstranded."
    gtf_line = "genes.gtf: gene annotation in GTF format (one transcript per gene)."
    (work / "README.md").write_text(common.readme([reads_line, gtf_line]))
    expected, naive = truth_answers(layout, frags, twin)
    if paired:
        seen = (
            f"reads.bam holds {2 * len(frags)} paired records for {len(frags)} fragments: "
            "featureCounts -p without --countReadPairs reports two counts per fragment"
        )
    else:
        seen = f"reads.bam holds {len(frags)} single-end reads: one record per fragment"
    evidence = (seen,)
    if tier == "explicit":
        evidence = (f"README states {kind} reads", *evidence)
    what = (
        "the number of fragments (sequenced molecules; a read pair is one fragment) in reads.bam "
        "that come from that gene"
    )
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use featureCounts to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its fragment count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F42", seed, twin, tier, route)
    return Task(task_id, "F42", "record_count", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    if not is_paired(work, tools):
        return _result(feature_counts(work, tools), "featureCounts (single-end)")
    counts = feature_counts(work, tools, "-p", "--countReadPairs")
    return _result(
        counts, "featureCounts -p --countReadPairs", ["paired-end: fragments, not reads"]
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    args = ["-p"] if is_paired(work, tools) else []
    return _result(
        feature_counts(work, tools, *args), "featureCounts" + "".join(f" {a}" for a in args)
    )


# A featureCounts command that declares paired-end input (-p, also inside a combined flag such as
# -p, or -T 1 -p) and never asks for fragments: the command line has no --countReadPairs and no
# division of the counts by two is visible to the heuristic.
FEATURE_COUNTS = re.compile(r"\bfeatureCounts\b")
PAIRED_FLAG = re.compile(r"(?<![\w-])-p(?![\w-])")
FRAGMENTS_FLAG = re.compile(r"--countReadPairs")


def _flag_paired(path: Path) -> bool:
    import gzip
    import struct

    try:
        data = gzip.open(path).read()
        (l_text,) = struct.unpack_from("<i", data, 4)
        offset = 8 + l_text
        (n_ref,) = struct.unpack_from("<i", data, offset)
        offset += 4
        for _ in range(n_ref):
            (l_name,) = struct.unpack_from("<i", data, offset)
            offset += 4 + l_name + 4
        (size,) = struct.unpack_from("<i", data, offset)
        flag = struct.unpack_from("<H", data, offset + 4 + 14)[0]
        return bool(flag & 1) and size > 0
    except (OSError, ValueError, struct.error, EOFError):
        return False


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff featureCounts runs with -p, without --countReadPairs, on a paired workspace."""
    if not (work / "reads.bam").exists() or not _flag_paired(work / "reads.bam"):
        return False
    for command in commands:
        for part in re.split(r"[|;&\n]", command):
            if FEATURE_COUNTS.search(part) and PAIRED_FLAG.search(part):
                if not FRAGMENTS_FLAG.search(part):
                    return True
    return False


SPEC = FamilySpec(
    id="F42",
    slot="record_count",
    source="GEN-24 (featureCounts >= 2.0.2: -p alone counts reads, --countReadPairs counts pairs)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
