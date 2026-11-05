"""F44 htseq_mapq (locus tool_default): `htseq-count` skips alignments with MAPQ below 10.

`htseq-count -a/--minaqual` defaults to 10 (featureCounts `-Q` defaults to 0). Reads from a
short-read aligner such as BWA or Bowtie 2 that align uniquely but with mismatches or in
repetitive sequence often carry MAPQ 0-3; htseq-count drops them and only the special counter
`__too_low_aQual` at the bottom of the table says so. The twins hold the same reads (the same
fragments, positions and mismatches, no NH above 1) of a forward-stranded library (read 1 on the
strand of its gene, no leaked reads), so the default `-s yes` of htseq-count is right in both twins
and the MAPQ floor is the only trap. (With an unstranded library the default `-s yes` was a second,
unstated trap: the plain default call was wrong on the clean twin too.) In the clean twin every
MAPQ is 30-60. In the faulted twin 15-30 % of each gene's reads (the ones with mismatches) have
MAPQ 0-3.

Verified on HTSeq 2.1.2 (exit 0, nothing on stderr with `-q`): the plain default call `htseq-count
-q reads.bam genes.gtf` on such a BAM prints counts that are 15-30 % short and `__too_low_aQual
58` (seed 0, `__no_feature` 0); adding `-a 0` gives every read. `featureCounts -s 0` on the same BAM
counts all of them.

  explicit  the README says the BAM comes from BWA-MEM and holds unique alignments whose MAPQ ranges
            from 0 to 60, and that the library is forward-stranded
  implicit  the README only says "RNA-seq reads"; `samtools view | cut -f5` shows the MAPQ values
  forced    the task sentence dictates htseq-count
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f41_htseq_strand import (
    FORWARD,
    LIBRARY_LINE,
    counts_by_gene,
    gtf_text,
    htseq_counts,
    make_fragments,
    make_layout,
    sam_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "README.md"})
LOW_MAPQ = (0, 1, 2, 3)
HTSEQ_DEFAULT_MAPQ = 10


def make_plan(seed: int):
    layout = make_layout(f"f44-{seed}", 5 + seed % 2)
    rnd = random.Random(f"f44-{seed}-frags")
    # forward-stranded, no leaked reads: the default -s yes is right, the MAPQ floor is the trap
    frags = make_fragments(rnd, layout, FORWARD)
    low: set[str] = set()
    for i in range(len(layout.genes)):
        mine = [f.name for f in frags if f.gene == i]
        low |= set(rnd.sample(mine, round(rnd.uniform(0.15, 0.30) * len(mine))))
    # MAPQ per read: the faulted twin gives the mismatching ones 0-3, the clean twin 30-60
    high = {f.name: rnd.randint(30, 60) for f in frags}
    faulted = {n: (rnd.choice(LOW_MAPQ) if n in low else q) for n, q in high.items()}
    return layout, frags, low, {"clean": high, "faulted": faulted}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    layout, frags, low, quality = make_plan(seed)
    sam = sam_genome(
        layout,
        frags,
        paired=False,
        mapq=lambda frag: quality[twin][frag.name],
        mismatches=lambda frag: (3, 11, 29) if frag.name in low else (),
    )
    common.write_bam(work, sam, tools, "reads")
    (work / "genes.gtf").write_text(gtf_text(layout))
    reads_line = "reads.bam (with reads.bam.bai): single-end RNA-seq reads aligned to the genome."
    if tier == "explicit":
        reads_line += (
            " Aligner: BWA-MEM; one primary alignment per read, MAPQ from 0 to 60. "
            + LIBRARY_LINE[FORWARD]
        )
    gtf_line = "genes.gtf: gene annotation in GTF format (one transcript per gene)."
    (work / "README.md").write_text(common.readme([reads_line, gtf_line]))
    expected = counts_by_gene(layout, frags)
    naive = {g.id: expected[g.id] for g in layout.genes}
    if twin == "faulted":
        for f in frags:
            if f.name in low:
                naive[layout.genes[f.gene].id] -= 1
        seen = (
            f"{len(low)} of {len(frags)} reads have MAPQ below {HTSEQ_DEFAULT_MAPQ} "
            f"(0-3): htseq-count skips them by default (__too_low_aQual)"
        )
    else:
        seen = f"every read has MAPQ 30 or more: the default -a {HTSEQ_DEFAULT_MAPQ} drops nothing"
    evidence = (seen,)
    if tier == "explicit":
        evidence = ("README says MAPQ ranges from 0 to 60", *evidence)
    what = "the number of reads in reads.bam that align to that gene (every aligned read counts)"
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use htseq-count to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its read count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F44", seed, twin, tier, route)
    return Task(task_id, "F44", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    counts = htseq_counts(work, tools, "-a", "0")
    return _result(counts, "htseq-count -a 0", ["-a 0: MAPQ is not a filter here"])


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    # the plain default call: -s yes fits the forward-stranded reads, -a 10 is the trap
    return _result(htseq_counts(work, tools), "htseq-count with its defaults (-a 10)")


# htseq-count with an effective MAPQ floor above 0 on a workspace whose BAM holds reads below it
HTSEQ = re.compile(r"\bhtseq-count\b")
MINAQUAL = re.compile(r"(?<![\w-])(?:-a\s*|--minaqual[= ])\s*(\d+)")


def _has_low_mapq(work: Path) -> bool:
    import gzip
    import struct

    try:
        data = gzip.open(work / "reads.bam").read()
        (l_text,) = struct.unpack_from("<i", data, 4)
        offset = 8 + l_text
        (n_ref,) = struct.unpack_from("<i", data, offset)
        offset += 4
        for _ in range(n_ref):
            (l_name,) = struct.unpack_from("<i", data, offset)
            offset += 4 + l_name + 4
        while offset < len(data):
            (size,) = struct.unpack_from("<i", data, offset)
            if data[offset + 13] < HTSEQ_DEFAULT_MAPQ:
                return True
            offset += 4 + size
    except (OSError, ValueError, struct.error, EOFError):
        return False
    return False


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff htseq-count runs with a MAPQ floor above 0 (default 10, or -a N, N > 0) on a
    BAM that holds reads below it."""
    if not (work / "reads.bam").exists() or not _has_low_mapq(work):
        return False
    for command in commands:
        for part in re.split(r"[|;&\n]", command):
            if HTSEQ.search(part):
                opt = MINAQUAL.search(part)
                if opt is None or int(opt.group(1)) > 0:
                    return True
    return False


SPEC = FamilySpec(
    id="F44",
    slot="default_param",
    source="GEN-22 (htseq-count defaults), HTSeq --help: -a defaults to 10, featureCounts -Q to 0",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
