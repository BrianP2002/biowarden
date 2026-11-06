"""F43 fc_antisense (locus tool_default): featureCounts counts unstranded unless told otherwise.

`featureCounts` defaults to `-s 0` (unstranded counting). On a reverse-stranded (dUTP) library that
is harmless while no two genes overlap on opposite strands, and silently loses reads where they
do: a read that overlaps the exons of two genes is `Unassigned_Ambiguity` and counted for neither.
The twins hold the same single-end reads of the same genes (the same fragments). In the clean twin
the last gene (an antisense partner of gene01) lies clear of every other gene; in the faulted twin
its first exon overlaps gene01's middle exon on the opposite strand, so the reads of both genes in
the overlap (about a quarter of gene01's and half of the partner's) are dropped by the default.

Verified on featureCounts 2.1.1 (exit 0, banner only on stderr): on a dUTP BAM with such a pair,
`-s 0` and `-s 1` lose reads, `-s 2` counts every read once; the summary file shows
`Unassigned_Ambiguity` for the first and `Unassigned_NoFeatures` for the wrong strand. The counts
of every gene outside the pair are identical under `-s 0` and `-s 2`, so the totals look healthy.

  explicit  the README states the library type (reverse-stranded, dUTP)
  implicit  the README only says "RNA-seq reads"; the strand of the reads against the genes is
            the evidence (every read maps to the strand opposite to its gene)
  forced    the task sentence dictates featureCounts
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f41_htseq_strand import (
    REVERSE,
    Layout,
    _blocks,
    counts_by_gene,
    gtf_text,
    make_fragments,
    make_layout,
    reads_of,
    sam_genome,
)
from biowarden.bench.tasks.extra.f42_fc_read_pairs import feature_counts
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "genes.gtf", "README.md"})
STRAND_OPT = re.compile(r"(?<![\w-])-s\s*([012])(?![\w-])")


def make_plan(seed: int, twin: str):
    mode = "separate" if twin == "clean" else "overlap"
    layout = make_layout(f"f43-{seed}", 5 + seed % 2, antisense=mode)
    frags = make_fragments(random.Random(f"f43-{seed}-frags"), layout, REVERSE)
    return layout, frags


def unstranded_counts(layout: Layout, frags) -> tuple[dict[str, int], int]:
    """(`featureCounts -s 0` counts, reads dropped as ambiguous): a read counts for a gene only if
    its aligned blocks touch the exons of that gene and of no other gene."""
    out = {g.id: 0 for g in layout.genes}
    ambiguous = 0
    for frag in frags:
        (read,) = reads_of(layout, frag, False)
        gene = layout.genes[frag.gene]
        blocks = _blocks(gene, read.x, read.y)
        hit = {
            other.id
            for other in layout.genes
            if other.chrom == gene.chrom
            and any(a < be and b > ba for a, b in other.exons for ba, be in blocks)
        }
        if len(hit) == 1:
            out[next(iter(hit))] += 1
        elif len(hit) > 1:
            ambiguous += 1
    return out, ambiguous


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    layout, frags = make_plan(seed, twin)
    common.write_bam(work, sam_genome(layout, frags, paired=False), tools, "reads")
    (work / "genes.gtf").write_text(gtf_text(layout))
    reads_line = "reads.bam (with reads.bam.bai): single-end RNA-seq reads aligned to the genome."
    if tier == "explicit":
        reads_line += (
            " Library: reverse-stranded (dUTP): a read maps to the strand opposite its gene."
        )
    gtf_line = "genes.gtf: gene annotation in GTF format (one transcript per gene)."
    (work / "README.md").write_text(common.readme([reads_line, gtf_line]))
    expected = counts_by_gene(layout, frags)
    naive, ambiguous = unstranded_counts(layout, frags)
    first, partner = layout.genes[0], layout.genes[-1]
    if twin == "clean":
        seen = (
            f"{partner.id} ({partner.strand}) lies clear of every gene: unstranded counting "
            "loses nothing"
        )
    else:
        seen = (
            f"{first.id} ({first.strand}) and {partner.id} ({partner.strand}) overlap on opposite "
            f"strands and {ambiguous} reads fall in the overlap: featureCounts -s 0 drops them as "
            "Unassigned_Ambiguity"
        )
    evidence = (seen,)
    if tier == "explicit":
        evidence = ("README states a reverse-stranded library", *evidence)
    what = "the number of reads in reads.bam that come from that gene's transcripts"
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use featureCounts to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its read count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F43", seed, twin, tier, route)
    return Task(task_id, "F43", "strand", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    forward = sum(feature_counts(work, tools, "-s", "1").values())
    reverse = sum(feature_counts(work, tools, "-s", "2").values())
    if max(forward, reverse) == 0 or min(forward, reverse) > 0.4 * max(forward, reverse):
        return _result(feature_counts(work, tools, "-s", "0"), "featureCounts -s 0 (unstranded)")
    option = "1" if forward > reverse else "2"
    flags = [f"stranded library: featureCounts -s {option}, not the default -s 0"]
    return _result(feature_counts(work, tools, "-s", option), f"featureCounts -s {option}", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(feature_counts(work, tools), "featureCounts with its defaults")


def has_antisense_overlap(work: Path) -> bool:
    """Do two genes of genes.gtf overlap on opposite strands (any exon of one touches the other)?"""
    try:
        rows = [
            line.split("\t")
            for line in (work / "genes.gtf").read_text().splitlines()
            if "\texon\t" in line
        ]
    except OSError:
        return False
    exons = [
        (r[0], int(r[3]), int(r[4]), r[6], re.search(r'gene_id "([^"]+)"', r[8])) for r in rows
    ]
    for i, a in enumerate(exons):
        for b in exons[i + 1 :]:
            if a[0] == b[0] and a[3] != b[3] and a[1] <= b[2] and b[1] <= a[2]:
                return True
    return False


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff featureCounts runs unstranded (no -s, -s 0) or forward (-s 1) on a workspace
    whose genes overlap on opposite strands (the library of this family is always reverse)."""
    if not has_antisense_overlap(work):
        return False
    for command in commands:
        for part in re.split(r"[|;&\n]", command):
            if not re.search(r"\bfeatureCounts\b", part):
                continue
            opt = STRAND_OPT.search(part)
            if opt is None or opt.group(1) in ("0", "1"):
                return True
    return False


SPEC = FamilySpec(
    id="F43",
    slot="strand",
    source="GEN-20, GEN-23 (library strandedness not passed to the counter; antisense overlaps)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
