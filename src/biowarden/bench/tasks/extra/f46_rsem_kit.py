"""F46 rsem_kit (locus data): RSEM keeps only the reads in the orientation it is told.

`rsem-calculate-expression --strandedness reverse` (RSEM's own help: "For Illumina TruSeq Stranded
protocols, please use 'reverse'") keeps only alignments whose read 1 maps to the strand opposite to
the transcript; `forward` keeps the others. On a library that is not strand-specific that silently
drops about half of the fragments. The README names the preparation kit (TruSeq Stranded mRNA, in
both twins and both tiers). The twins hold the same genes and the same fragments, as paired
alignments to the transcriptome (the input RSEM wants for `--alignments`). In the clean twin the
library really is dUTP (every read 1 on the opposite strand). In the faulted twin the sample
lost its strand marking: read 1 maps to either strand with equal probability, and the kit named in
the README is no evidence for the orientation. Only the flags of the BAM say so.

Verified on RSEM 1.3.1 (exit 0, empty stderr): on 640 simulated paired fragments of an unstranded
library, `rsem-calculate-expression --alignments --paired-end --strandedness reverse --no-bam-output
reads_tx.bam ref out` gives `sum(expected_count) = 337` (the fragments whose read 1 maps to the
reverse strand), `--strandedness forward` gives 303 and `none` 640; the `.cnt` file still reports
640 alignable reads. On a dUTP library `reverse` and `none` give 640 and `forward` stops with `No
valid read to estimate the length distribution!` (exit 255). The counts of a gene are exactly its
fragments in the matching orientation, so the recorded naive answer is computed without running
RSEM (and tests/test_family_quant.py checks it against the real tool).

  explicit  the README adds the measured library type (reverse-stranded, or unstranded)
  implicit  the README only names the kit; the flags of read 1 are the evidence
  forced    the task sentence dictates RSEM
"""

from __future__ import annotations

import gzip
import random
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    REVERSE,
    UNSTRANDED,
    Layout,
    counts_by_gene,
    genome_fasta,
    gtf_text,
    make_fragments,
    make_layout,
    sam_transcriptome,
    write_transcriptome_bam,
)
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"reads_tx.bam", "README.md"})
KIT = "Library preparation: Illumina TruSeq Stranded mRNA kit."
RSEM_BASE = ["--alignments", "--paired-end", "-p", "1", "--no-bam-output"]


def library_of(twin: str) -> str:
    return REVERSE if twin == "clean" else UNSTRANDED


def make_plan(seed: int, twin: str):
    layout = make_layout(f"f46-{seed}", 5 + seed % 2)
    library = library_of(twin)
    frags = make_fragments(random.Random(f"f46-{seed}-{library}"), layout, library)
    return layout, frags


def truth_answers(layout: Layout, frags, twin: str) -> tuple[dict[str, int], dict[str, int]]:
    """(expected, naive): the fragments of each gene, and what `--strandedness reverse` keeps."""
    expected = counts_by_gene(layout, frags)
    kept = {g.id: 0 for g in layout.genes}
    for frag in frags:
        if frag.o == "anti":  # read 1 on the strand opposite to the transcript
            kept[layout.genes[frag.gene].id] += 1
    return expected, kept


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    layout, frags = make_plan(seed, twin)
    (work / "genome.fa").write_text(genome_fasta(layout))
    (work / "genes.gtf").write_text(gtf_text(layout))
    write_transcriptome_bam(work, sam_transcriptome(layout, frags, True), tools)
    lines = [
        "reads_tx.bam: paired-end RNA-seq reads aligned to the transcriptome (reference names are "
        "the transcript IDs of genes.gtf, mates adjacent), as RSEM's --alignments mode expects.",
        "genome.fa, genes.gtf: the genome and annotation the transcripts come from (one "
        "transcript per gene).",
        KIT,
    ]
    if tier == "explicit":
        lines[2] += (
            " Measured library type: reverse-stranded (dUTP)."
            if twin == "clean"
            else " Measured library type: unstranded (read 1 maps to either strand)."
        )
    (work / "README.md").write_text(common.readme(lines))
    expected, naive = truth_answers(layout, frags, twin)
    opposite = sum(naive.values())
    if twin == "clean":
        seen = (
            f"read 1 maps to the strand opposite to its transcript for all {len(frags)} fragments: "
            "a dUTP library, as the kit says"
        )
    else:
        seen = (
            f"read 1 maps to the opposite strand for {opposite} of {len(frags)} fragments: an "
            "unstranded library, whatever the kit says; --strandedness reverse keeps only those"
        )
    evidence = (seen,)
    if tier == "explicit":
        evidence = ("README states the measured library type", *evidence)
    what = "the number of fragments (read pairs) in reads_tx.bam that come from that gene"
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use RSEM (rsem-calculate-expression) to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its fragment count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 2.0}
    task_id = common.task_id("F46", seed, twin, tier, route)
    return Task(task_id, "F46", "strand", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def rsem_counts(work: Path, tools: ToolEnv, *option: str) -> dict[str, float]:
    """rsem-prepare-reference + rsem-calculate-expression in a scratch folder; gene -> count."""
    with tempfile.TemporaryDirectory() as scratch:
        ref, out = str(Path(scratch) / "ref"), str(Path(scratch) / "out")
        argv = ["rsem-prepare-reference", "--gtf", "genes.gtf", "genome.fa", ref]
        common.check(tools.run(argv, work), argv)
        argv = ["rsem-calculate-expression", *RSEM_BASE, *option, "reads_tx.bam", ref, out]
        common.check(tools.run(argv, work), argv)
        rows = [r.split("\t") for r in Path(out + ".genes.results").read_text().splitlines()[1:]]
    return {r[0]: float(r[4]) for r in rows}


def read1_reverse_share(path: Path, limit: int = 2000) -> float | None:
    """Share of read-1 records of a BAM with the reverse flag (stdlib; None when unreadable)."""
    try:
        with gzip.open(path) as handle:
            if handle.read(4) != b"BAM\x01":
                return None
            handle.read(struct.unpack("<i", handle.read(4))[0])
            for _ in range(struct.unpack("<i", handle.read(4))[0]):
                handle.read(struct.unpack("<i", handle.read(4))[0] + 4)
            reverse = total = 0
            for _ in range(limit):
                head = handle.read(4)
                if len(head) < 4:
                    break
                body = handle.read(struct.unpack("<i", head)[0])
                flag = struct.unpack_from("<H", body, 14)[0]
                if flag & 0x40:
                    total += 1
                    reverse += bool(flag & 0x10)
    except (OSError, EOFError, struct.error):
        return None
    return reverse / total if total else None


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    share = read1_reverse_share(work / "reads_tx.bam")
    option = "none"
    if share is not None and share >= 0.85:
        option = "reverse"
    elif share is not None and share <= 0.15:
        option = "forward"
    flags = [f"read 1 on the reverse strand for {100 * (share or 0):.0f} % of the fragments"]
    counts = rsem_counts(work, tools, "--strandedness", option)
    return _result(counts, f"rsem-calculate-expression --strandedness {option}", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    counts = rsem_counts(work, tools, "--strandedness", "reverse")
    return _result(counts, "rsem-calculate-expression --strandedness reverse (TruSeq Stranded)")


# A stranded RSEM call (--strandedness forward|reverse, --forward-prob 0 or 1) on a workspace whose
# BAM is not strand-specific (the share of reverse-flagged read 1 is between 0.3 and 0.7).
RSEM = re.compile(r"\brsem-calculate-expression\b")
STRANDED = re.compile(
    r"--strandedness[= ]\s*(forward|reverse)\b|--forward-prob[= ]\s*(?:0|1)(?:\.0+)?(?![\d.])"
)


def exposed(commands: list[str], work: Path) -> bool:
    share = read1_reverse_share(work / "reads_tx.bam")
    if share is None or not 0.3 <= share <= 0.7:
        return False
    return any(
        RSEM.search(part) and STRANDED.search(part)
        for command in commands
        for part in re.split(r"[|;&\n]", command)
    )


SPEC = FamilySpec(
    id="F46",
    slot="strand",
    source="GEN-20, GEN-21 (RSEM help: TruSeq Stranded means --strandedness reverse)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="data",
)
