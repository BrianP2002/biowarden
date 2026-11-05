"""F31 star_genecounts (strand, tool default): which column of STAR's ReadsPerGene.out.tab to read.

`STAR --quantMode GeneCounts` writes one table with three count columns per gene: column 2 counts
reads regardless of strand, column 3 counts reads whose FIRST mate maps to the gene's own strand
(the htseq-count -s yes / featureCounts -s 1 convention, forward-stranded libraries), column 4
counts reads whose first mate maps to the opposite strand (-s reverse / -s 2, the dUTP protocol of
Illumina TruSeq Stranded). The file does not say which column suits the library. On a dUTP library
column 3 holds only the few percent of pairs that leak the wrong way, so the genes look lowly
expressed rather than absent, and column 2 loses every pair that overlaps a gene on the other strand
(they are counted as N_ambiguous). Exit status 0, a complete table, no warning.

Task: per gene, the number of read pairs derived from the gene's transcripts in a strand-specific
library, counting a pair for a gene only when its orientation fits the gene's strand under the
library protocol. Clean twin: forward-stranded library (read 1 on the gene's strand), so column 3
is the answer. Faulted twin: the same fragments from dUTP (reverse-stranded) libraries, so the
answer is column 4. The leakage (1-3 pairs per gene at the wrong orientation) is placed outside the
region where the two genes of chr2 overlap on opposite strands, so it never moves a pair between
genes. g4 (+) and g5 (-) share 160 bases of exon: a pair there is ambiguous in column 2.

Verified here (STAR 2.7.11b, 1 thread, `--genomeDir .` with the GTF in the index, 5 genes, 2x75
pairs): forward library: column 3 equals the number of sense pairs per gene, column 4 the leaked
ones (N_noFeature of column 4 holds the sense pairs); dUTP library: the opposite; column 2 equals
column 3 + column 4 per gene except for g4 and g5, where pairs in the shared exon are N_ambiguous.

  explicit  the README names the protocol (forward-stranded, or dUTP / reverse-stranded)
  implicit  the README says only "strand-specific"; the column totals of a first run show it
  forced    the sentence asks for STAR's GeneCounts table
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f29_mapq_unique import (
    clean_up,
    fastq_text,
    rand_seq,
    revcomp,
    star_index,
    star_map,
    write_genome,
)
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 75
FORWARD, REVERSE = "forward", "reverse"
GENES = ("g1", "g2", "g3", "g4", "g5")


@dataclass(frozen=True)
class Gene:
    id: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]  # 0-based half-open

    def transcript(self, genome: dict[str, str]) -> str:
        seq = "".join(genome[self.chrom][a:b] for a, b in self.exons)
        return seq if self.strand == "+" else revcomp(seq)


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    genes: tuple[Gene, ...]
    # (gene index, fragment sequence in transcript orientation, leaked?) per read pair, file order
    fragments: tuple[tuple[int, str, bool], ...]
    sense: dict[str, int]  # gene -> number of sense pairs (the truth)


def _motifs(seq: list[str], gene_strand: str, exons: list[tuple[int, int]]) -> None:
    donor, acceptor = ("GT", "AG") if gene_strand == "+" else ("CT", "AC")
    for (_, end), (start, _) in zip(exons, exons[1:], strict=False):
        seq[end : end + 2] = list(donor)
        seq[start - 2 : start] = list(acceptor)


def make_plan(seed: int, attempt: int = 0) -> Plan:
    """Genome, genes and fragments of one seed; `attempt` > 0 redraws only the fragments."""
    rng = random.Random(f"f31-{seed}")

    def jitter(n: int, k: int = 20) -> int:
        return n + rng.randint(-k, k)

    # chr1: three genes in a row, chr2: g4 (+) and g5 (-) sharing the end of g4's last exon
    g1 = [(300, 300 + jitter(260))]
    g1.append((g1[0][1] + jitter(450), 0))
    g1[1] = (g1[1][0], g1[1][0] + jitter(200))
    g1.append((g1[1][1] + jitter(380), 0))
    g1[2] = (g1[2][0], g1[2][0] + jitter(260))
    g2 = [(g1[2][1] + 400, g1[2][1] + 400 + jitter(240))]
    g2.append((g2[0][1] + jitter(420), 0))
    g2[1] = (g2[1][0], g2[1][0] + jitter(260))
    g2.append((g2[1][1] + jitter(350), 0))
    g2[2] = (g2[2][0], g2[2][0] + jitter(250))
    g3 = [(g2[2][1] + 400, g2[2][1] + 400 + jitter(230))]
    g3.append((g3[0][1] + jitter(500), 0))
    g3[1] = (g3[1][0], g3[1][0] + jitter(270))
    chr1_len = g3[1][1] + 250

    g4 = [(250, 250 + jitter(200))]
    g4.append((g4[0][1] + jitter(420), 0))
    g4[1] = (g4[1][0], g4[1][0] + 260)
    overlap_start = g4[1][1] - 160
    g5 = [(overlap_start, overlap_start + 250)]
    g5.append((g5[0][1] + jitter(450), 0))
    g5[1] = (g5[1][0], g5[1][0] + jitter(240))
    chr2_len = g5[1][1] + 250

    models = [
        ("g1", "chr1", "+", g1),
        ("g2", "chr1", "-", g2),
        ("g3", "chr1", "+", g3),
        ("g4", "chr2", "+", g4),
        ("g5", "chr2", "-", g5),
    ]
    genome_list = {"chr1": list(rand_seq(rng, chr1_len)), "chr2": list(rand_seq(rng, chr2_len))}
    for _, chrom, strand, exons in models:
        _motifs(genome_list[chrom], strand, exons)
    genome = {c: "".join(s) for c, s in genome_list.items()}
    genes = tuple(Gene(i, c, s, tuple(e)) for i, c, s, e in models)

    if attempt:  # the genome stays; a fresh stream draws the fragments again
        rng = random.Random(f"f31-{seed}-retry{attempt}")
    fragments: list[tuple[int, str, bool]] = []
    sense: dict[str, int] = {}
    for gi, gene in enumerate(genes):
        t = gene.transcript(genome)
        # transcript range free of the g4/g5 overlap, where a leaked pair could be counted for the
        # other gene: g4 (+) uses its first exon; g5 (-) is read 5'->3' from its LAST genomic
        # exon, so the part outside the overlap is the first stretch of its transcript
        lo, hi = 0, len(t)
        if gene.id == "g4":
            hi = gene.exons[0][1] - gene.exons[0][0]
        if gene.id == "g5":
            hi = gene.exons[1][1] - gene.exons[1][0]
        n = rng.randint(50, 90)
        sense[gene.id] = n
        for leaked, count in ((False, n), (True, rng.randint(1, 3))):
            for _ in range(count):
                span = (0, len(t)) if not leaked else (lo, hi)
                length = rng.randint(180, min(260, span[1] - span[0]))
                start = rng.randint(span[0], span[1] - length)
                fragments.append((gi, t[start : start + length], leaked))
    rng.shuffle(fragments)
    return Plan(genome, genes, tuple(fragments), sense)


def gtf_text(genes: tuple[Gene, ...]) -> str:
    rows = []
    for g in genes:
        for i, (a, b) in enumerate(g.exons, start=1):
            attrs = f'gene_id "{g.id}"; transcript_id "{g.id}.1"; exon_number "{i}";'
            rows.append(f"{g.chrom}\tsyn\texon\t{a + 1}\t{b}\t.\t{g.strand}\t.\t{attrs}\n")
    return "".join(rows)


def read_pairs(plan: Plan, library: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    first: list[tuple[str, str]] = []
    second: list[tuple[str, str]] = []
    for i, (_, frag, leaked) in enumerate(plan.fragments):
        sense_read = frag[:READ_LEN]
        anti_read = revcomp(frag[-READ_LEN:])
        forward_pair = (sense_read, anti_read)  # read 1 on the transcript's strand
        pair = forward_pair if (library == FORWARD) != leaked else forward_pair[::-1]
        name = f"pair{i:04d}"
        first.append((f"{name}/1", pair[0]))
        second.append((f"{name}/2", pair[1]))
    return first, second


def run_counts(work: Path, tools: ToolEnv, prefix: str) -> dict[str, tuple[int, int, int]]:
    star_map(
        work,
        tools,
        ["reads_1.fastq", "reads_2.fastq"],
        prefix,
        ("--quantMode", "GeneCounts", "--outSAMtype", "None"),
    )
    return parse_counts((work / f"{prefix}ReadsPerGene.out.tab").read_text())


def parse_counts(text: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) == 4:
            out[f[0]] = (int(f[1]), int(f[2]), int(f[3]))
    return out


def column(table: dict[str, tuple[int, int, int]], index: int) -> dict[str, int]:
    return {g: v[index] for g, v in table.items() if g in GENES}


MAX_ATTEMPTS = 12


def pick_plan(seed: int, work: Path, tools: ToolEnv) -> tuple[Plan, dict]:
    """The first fragment draw whose pairs STAR assigns exactly as designed in BOTH libraries.

    A pair can map to two loci by chance (a spliced read with a short overhang in a random 7 kb
    genome: seed 45), which moves it into N_multimapping and the gene count one below the design.
    The draw is redone from a fresh stream until forward and dUTP versions both come out exact, so
    the clean and faulted twins always share the truth. Returns the plan and the counts by library.
    """
    for attempt in range(MAX_ATTEMPTS):
        plan = make_plan(seed, attempt)
        tables = {}
        exact = True
        for library in (FORWARD, REVERSE):
            first, second = read_pairs(plan, library)
            (work / "reads_1.fastq").write_text(fastq_text(first))
            (work / "reads_2.fastq").write_text(fastq_text(second))
            tables[library] = run_counts(work, tools, "tmp_")
            clean_up(work, "tmp_*")
            right = 1 if library == FORWARD else 2
            exact &= column(tables[library], right) == plan.sense
            exact &= (column(tables[library], 1) == plan.sense) == (library == FORWARD)
        if exact:
            return plan, tables
    raise RuntimeError(f"F31 seed {seed}: no fragment draw gave exact STAR counts")


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    library = FORWARD if twin == "clean" else REVERSE
    genome_plan = make_plan(seed)  # the genome and the genes do not depend on the attempt
    write_genome(work, genome_plan.genome, tools)
    (work / "genes.gtf").write_text(gtf_text(genome_plan.genes))
    star_index(work, tools, gtf="genes.gtf", overhang=READ_LEN - 1)
    plan, tables = pick_plan(seed, work, tools)
    first, second = read_pairs(plan, library)
    (work / "reads_1.fastq").write_text(fastq_text(first))
    (work / "reads_2.fastq").write_text(fastq_text(second))
    table = tables[library]
    totals = [sum(column(table, i).values()) for i in range(3)]
    explicit = tier == "explicit"
    protocol = (
        "forward-stranded (read 1 is on the same strand as the gene)"
        if library == FORWARD
        else "dUTP, reverse-stranded (read 1 is on the opposite strand of the gene)"
    )
    lines = [
        "reads_1.fastq, reads_2.fastq: paired-end RNA-seq reads of 75 bases from a strand-specific "
        "library" + (f": {protocol}." if explicit else "."),
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2. genes.gtf: five "
        "genes (exon lines with gene_id).",
        "The STAR 2.7.11b index of genome.fa built with genes.gtf is in this directory (Genome, "
        "SA, SAindex, ...): use --genomeDir .",
    ]
    (work / "README.md").write_text(common.readme(lines))
    evidence = ((f"README states the protocol: {protocol}",) if explicit else ()) + (
        f"column totals of ReadsPerGene.out.tab over the genes: unstranded {totals[0]}, "
        f"column 3 {totals[1]}, column 4 {totals[2]}",
    )
    stated = (
        " The library is strand-specific: a pair counts for a gene only if its orientation "
        "fits the gene's strand under the library protocol, so use the strand information."
    )
    prompt = common.prompt_for(
        route,
        "report, for each gene in genes.gtf, the number of read pairs derived from that "
        "gene's transcripts, counting each pair once." + stated,
        "use STAR with --quantMode GeneCounts (ReadsPerGene.out.tab) to report, for each gene in "
        "genes.gtf, the number of read pairs derived from that gene's transcripts, counting each "
        "pair once." + stated,
        "a JSON object mapping each gene ID to its pair count (an integer)",
    )
    expected = dict(plan.sense)
    naive = column(table, 1)  # column 3 of the file: the first-read-strand (forward) column
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0}
    task_id = common.task_id("F31", seed, twin, tier, route)
    return Task(task_id, "F31", "strand", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    table = run_counts(work, tools, "ref_")
    forward, reverse = (sum(column(table, i).values()) for i in (1, 2))
    index = 1 if forward >= reverse else 2
    return _result(
        column(table, index),
        f"STAR GeneCounts column {index + 2}: the stranded column with most pairs",
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    table = run_counts(work, tools, "naive_")
    return _result(column(table, 1), "STAR GeneCounts column 3 (first-read strand)")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a command that reads a column of ReadsPerGene.out.tab other than the one the library
# needs. The needed column comes from the twin, not from a table the agent wrote (that one may sit
# in /tmp, which is private to one command, or never be written): the library is forward (column 3)
# when read 1 of most pairs is a piece of a transcript of genes.gtf, reverse (column 4) when its
# reverse complement is. Without genome.fa, genes.gtf and reads_1.fastq nothing can be said and the
# answer is False. Column selectors recognised: awk `$N`, cut `-fN` / `-f1,N`, python-style `[N-1]`
# after a split. Applied per command; a command that uses several columns (compares them) is not
# trapped.
TABLE = re.compile(r"ReadsPerGene")
AWK_COLUMN = re.compile(r"\$([2-4])\b")
CUT_COLUMN = re.compile(r"\bcut\b[^|;&]*?-f\s*(?:1,)?([2-4])\b")
INDEX_COLUMN = re.compile(r"\[\s*([1-3])\s*\]")
SEGMENT = re.compile(r"&&|\|\||;|\n")
SAMPLE = 80  # read-1 records looked at: leaked pairs are 1-3 per gene, a few percent of the file


def _transcripts(work: Path) -> list[str]:
    """The sense sequence of every gene of genes.gtf, cut out of genome.fa (as Gene.transcript)."""
    genome: dict[str, str] = {}
    name = ""
    for line in (work / "genome.fa").read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            genome[name] = ""
        elif name:
            genome[name] += line.strip()
    exons: dict[str, list[tuple[str, int, int, str]]] = {}
    for line in (work / "genes.gtf").read_text().splitlines():
        f = line.split("\t")
        gene = re.search(r'gene_id "([^"]+)"', f[8]) if len(f) > 8 and f[2] == "exon" else None
        if gene:
            exons.setdefault(gene.group(1), []).append((f[0], int(f[3]) - 1, int(f[4]), f[6]))
    out = []
    for rows in exons.values():
        text = "".join(genome[c][a:b] for c, a, b, _ in sorted(rows, key=lambda r: r[1]))
        out.append(text if rows[0][3] == "+" else revcomp(text))
    return out


def _library(work: Path) -> str | None:
    """The library of the workspace, from where read 1 falls: FORWARD, REVERSE or None."""
    try:
        transcripts = _transcripts(work)
        reads = (work / "reads_1.fastq").read_text().splitlines()[1::4][:SAMPLE]
    except (OSError, KeyError, IndexError, ValueError):
        return None
    sense = anti = 0
    for read in reads:
        fits = any(read in t for t in transcripts)
        flips = any(revcomp(read) in t for t in transcripts)
        sense += fits and not flips  # a read in the exon shared by g4 and g5 fits both: skipped
        anti += flips and not fits
    return FORWARD if sense > anti else REVERSE if anti > sense else None


def _needed_column(work: Path) -> int | None:
    return {FORWARD: 3, REVERSE: 4}.get(_library(work) or "")


def _columns(segment: str) -> set[int]:
    found = {int(c) for c in AWK_COLUMN.findall(segment)}
    found |= {int(c) for c in CUT_COLUMN.findall(segment)}
    found |= {int(c) + 1 for c in INDEX_COLUMN.findall(segment)}
    return found


def exposed(commands: list[str], work: Path) -> bool:
    needed = _needed_column(work)
    if needed is None:
        return False
    for command in commands:
        for segment in SEGMENT.split(command):
            if not TABLE.search(segment):
                continue
            columns = _columns(segment)
            if len(columns) == 1 and needed not in columns:
                return True
    return False


SPEC = FamilySpec(
    id="F31",
    slot="strand",
    source="STAR manual (--quantMode GeneCounts: columns 2, 3, 4); GEN-20, GEN-22",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads_1.fastq", "reads_2.fastq", "README.md"}),
    locus="tool_default",
)
