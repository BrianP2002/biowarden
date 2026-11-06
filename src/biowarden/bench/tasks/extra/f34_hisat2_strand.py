"""F34 hisat2_strand (strand, tool default): --rna-strandness writes the XS strand of every read.

`hisat2 --rna-strandness` tells the aligner how the library was made. The values are one letter
for single-end reads (F, R) and two for pairs (FR, RF); dUTP libraries (Illumina TruSeq Stranded)
are RF. Once the option is given, hisat2 writes an XS:A strand tag on every alignment (400 of 400
records here). Nothing checks the value against the reads: a wrong one aligns every read just as
well (100% overall alignment rate) and only flips the XS tag, which StringTie and Cufflinks read to
give each assembled transcript a strand. A single letter on paired reads is also accepted and makes
XS inconsistent between the mates.

Task: assemble the transcripts of a strand-specific paired-end RNA-seq library and report, for each
gene region of genes.bed (BED3, no strand), the strand of the transcript assembled there. Same
fragments as F31 (5 genes, one antisense overlapping pair). Clean twin: forward-stranded library
(FR is right). Faulted twin: dUTP library (RF is right); the naive value FR flips every strand.

Verified here (hisat2 2.2.3, StringTie 3.0.3, samtools 1.24; the 4-gene test
library, dUTP): no --rna-strandness, `RF`: + - + - (right); `FR`: - + - + (all four flipped, 100%
alignment either way, XS:A present on every read); `R` or `F` on the pairs: + - + + - . and
+ + - + (broken: 300 of 400 reads got XS:A:+, one transcript with strand '.'); single-end read 1:
R right, F flipped. `stringtie --rf` on the FR BAM still gives the flipped strands (it trusts XS).
Without --rna-strandness the spliced reads get XS from the splice motifs and the strands come out
right, so the trap needs the option to be set wrongly.

  explicit  the README names the protocol (forward-stranded, or dUTP / reverse-stranded)
  implicit  the README says only "strand-specific"
  forced    the sentence asks for hisat2 with --rna-strandness for the library
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f29_mapq_unique import fastq_text, sam_records, write_genome
from biowarden.bench.tasks.extra.f31_star_genecounts import (
    FORWARD,
    REVERSE,
    make_plan,
    read_pairs,
)
from biowarden.bench.tasks.tools import ToolEnv

CORRECT = {FORWARD: "FR", REVERSE: "RF"}


def hisat2_index(work: Path, tools: ToolEnv) -> None:
    argv = ["hisat2-build-s", "-q", "genome.fa", "hs2idx"]
    common.check(tools.run(argv, work), argv)


def gene_regions(plan) -> dict[str, tuple[str, int, int]]:
    return {
        g.id: (g.chrom, min(a for a, _ in g.exons), max(b for _, b in g.exons)) for g in plan.genes
    }


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    library = FORWARD if twin == "clean" else REVERSE
    write_genome(work, plan.genome, tools)
    hisat2_index(work, tools)
    first, second = read_pairs(plan, library)
    (work / "reads_1.fastq").write_text(fastq_text(first))
    (work / "reads_2.fastq").write_text(fastq_text(second))
    regions = gene_regions(plan)
    (work / "genes.bed").write_text(
        "".join(f"{c}\t{a}\t{b}\t{g}\n" for g, (c, a, b) in regions.items())
    )
    explicit = tier == "explicit"
    protocol = (
        "forward-stranded (read 1 is on the same strand as the transcript)"
        if library == FORWARD
        else "dUTP, reverse-stranded (read 1 is on the opposite strand of the transcript)"
    )
    lines = [
        "reads_1.fastq, reads_2.fastq: paired-end RNA-seq reads of 75 bases from a strand-specific "
        "library" + (f": {protocol}." if explicit else "."),
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2; its hisat2 index "
        "(prefix hs2idx: hs2idx.1.ht2 ... hs2idx.8.ht2) is in this directory.",
        "genes.bed: the five gene regions (chrom, start, end, gene id; BED, no strand column).",
    ]
    (work / "README.md").write_text(common.readme(lines))
    expected = {g.id: g.strand for g in plan.genes}
    flipped = {g: "-" if s == "+" else "+" for g, s in expected.items()}
    if twin == "clean":
        evidence = (
            "a forward-stranded library: read 1 of a spliced pair sits on the gene's strand",
        )
        naive = expected
    else:
        evidence = (
            "a dUTP library: read 1 of a spliced pair sits on the strand opposite to its gene",
            "hisat2's XS tag, written for every read once --rna-strandness is given, follows the "
            "value given, not the reads",
        )
        naive = flipped
    if explicit:
        evidence = (f"README states the protocol: {protocol}",) + evidence
    prompt = common.prompt_for(
        route,
        "assemble the transcripts from the strand-specific reads and report, for each gene "
        "region in genes.bed, the strand ('+' or '-') of the transcript assembled there",
        "use hisat2 with its --rna-strandness option set for the library, then stringtie, to "
        "assemble the transcripts and report, for each gene region in genes.bed, the strand "
        "('+' or '-') of the transcript assembled there",
        "a JSON object mapping each gene ID to '+' or '-'",
    )
    truth = {"kind": "contexts", "expected": expected, "naive": naive}
    task_id = common.task_id("F34", seed, twin, tier, route)
    return Task(task_id, "F34", "strand", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def _align(work: Path, tools: ToolEnv, prefix: str, *options: str) -> str:
    argv = ["hisat2", "-p", "1", *options, "-x", "hs2idx", "-1", "reads_1.fastq"]
    argv += ["-2", "reads_2.fastq", "-S", f"{prefix}.sam"]
    common.check(tools.run(argv, work), argv)
    return (work / f"{prefix}.sam").read_text()


def assembled_strands(work: Path, tools: ToolEnv, prefix: str) -> dict[str, str]:
    """Sort, run StringTie, give each gene region the strand of its best-overlapping transcript."""
    for argv in (
        ["samtools", "sort", "-o", f"{prefix}.bam", f"{prefix}.sam"],
        ["stringtie", f"{prefix}.bam", "-o", f"{prefix}.gtf"],
    ):
        common.check(tools.run(argv, work), argv)
    transcripts = []
    for line in (work / f"{prefix}.gtf").read_text().splitlines():
        f = line.split("\t")
        if len(f) > 8 and f[2] == "transcript":
            transcripts.append((f[0], int(f[3]) - 1, int(f[4]), f[6]))
    out = {}
    for row in (work / "genes.bed").read_text().splitlines():
        chrom, a, b, gene = row.split("\t")[:4]
        best = max(
            ((min(int(b), t[2]) - max(int(a), t[1]), t[3]) for t in transcripts if t[0] == chrom),
            default=(0, "."),
        )
        out[gene] = best[1] if best[0] > 0 else "."
    return out


def library_of(sam_text: str) -> str:
    """forward if read 1 of spliced pairs sits on the strand of the XS tag, else reverse."""
    same = other = 0
    for f in sam_records(sam_text):
        flag = int(f[1])
        xs = next((x[5:] for x in f[11:] if x.startswith("XS:A:")), None)
        if xs is None or not flag & 0x40 or "N" not in f[5]:
            continue
        strand = "-" if flag & 0x10 else "+"
        same += strand == xs
        other += strand != xs
    return FORWARD if same >= other else REVERSE


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    library = library_of(_align(work, tools, "probe"))
    option = CORRECT[library]
    _align(work, tools, "ref", "--rna-strandness", option)
    return _result(
        assembled_strands(work, tools, "ref"),
        f"hisat2 --rna-strandness {option}, stringtie, overlap with genes.bed",
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    _align(work, tools, "naive", "--rna-strandness", "FR")
    return _result(assembled_strands(work, tools, "naive"), "hisat2 --rna-strandness FR, stringtie")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a hisat2 command that gives --rna-strandness F or FR (the forward values; the family's
# faulted twin is a dUTP library, so they flip every strand) or a single letter (R, F) on paired
# reads (-1 and -2). RF is the safe value, and so is leaving the option out. Applied per command.
HISAT2 = re.compile(r"\bhisat2\b")
STRANDNESS = re.compile(r"--rna-strandness[\s=]\s*['\"]?([FR]{1,2})\b")
PAIRED = re.compile(r"(?<=\s)-1\b")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if not HISAT2.search(segment):
                continue
            for value in STRANDNESS.findall(segment):
                if value in ("F", "FR") or (len(value) == 1 and PAIRED.search(segment)):
                    return True
    return False


SPEC = FamilySpec(
    id="F34",
    slot="strand",
    source="hisat2 manual (--rna-strandness F/R/FR/RF; XS:A on every read); GEN-20",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads_1.fastq", "reads_2.fastq", "README.md"}),
    locus="tool_default",
)
