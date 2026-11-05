"""F41 htseq_strand (locus tool_default): `htseq-count` assumes a forward-stranded library.

`htseq-count --help` says `-s {yes,no,reverse}` defaults to `yes`: only reads on the same strand as
the gene are counted. The twins hold the same single-end reads of the same genes (the same
fragments, the same positions). The clean twin is a forward-stranded library (a read maps to the
strand of its gene), where `yes` is right. The faulted twin is a reverse-stranded dUTP library
(even seeds: a read maps to the strand opposite its gene, the Illumina TruSeq Stranded protocol) or
an unstranded one (odd seeds), where the default counts only the 3-6 % leaked sense reads, or about
half of the reads.

Verified on HTSeq 2.1.2 (exit 0, nothing on stderr with `-q`): on a dUTP BAM of 640 reads,
`htseq-count -s yes reads.bam genes.gtf` prints 0 for every gene and `__no_feature 640`; `-s no`
and `-s reverse` give the real counts. The only trace of the problem is the special counter
`__no_feature` at the bottom of the table. HTSeq has no way to infer the library.

This module also holds the simulator shared by the F41-F52 families (`make_layout`, `Frag`,
`sam_genome`, `sam_transcriptome`, `fastq_text`, `gtf_text`, ...): a tiny genome, one-isoform genes
with known exons, fragments sampled from the transcripts with a known library type, and the
resulting alignments written as SAM text. Nothing here needs an aligner.

  explicit  the README states the library type of reads.bam
  implicit  the README only says "RNA-seq reads"; the share of reads on the gene strand is the
            evidence (about 96 % same-strand, about 4 % same-strand, or about 50 %)
  forced    the task sentence dictates `htseq-count`
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
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 50
LEAK = 0.04
TOL = 5.0
FORWARD, REVERSE, UNSTRANDED = "forward", "reverse", "unstranded"
COMPLEMENT = str.maketrans("ACGT", "TGCA")

# ---- the simulator shared by the quantification families ------------------------------------


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Gene:
    id: str  # gene01
    tx: str  # tx01: every gene has exactly one transcript
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]  # 0-based half-open, ascending
    n: int  # fragments sampled from the transcript

    @property
    def length(self) -> int:
        return sum(b - a for a, b in self.exons)

    @property
    def start(self) -> int:
        return self.exons[0][0]

    @property
    def end(self) -> int:
        return self.exons[-1][1]


@dataclass(frozen=True)
class Layout:
    genome: dict[str, str]
    genes: tuple[Gene, ...]

    def sequence(self, gene: Gene) -> str:
        """The transcript in its sense orientation, cut out of the genome."""
        text = "".join(self.genome[gene.chrom][a:b] for a, b in gene.exons)
        return text if gene.strand == "+" else revcomp(text)


def _exons(
    rnd: random.Random, start: int, n_exons: int, lengths: tuple[int, int] = (140, 260)
) -> tuple[tuple[int, int], ...]:
    out, pos = [], start
    for _ in range(n_exons):
        length = rnd.randint(*lengths)
        out.append((pos, pos + length))
        pos += length + rnd.randint(90, 200)
    return tuple(out)


def make_layout(
    key: str,
    n_genes: int,
    *,
    counts: tuple[int, int] = (30, 90),
    antisense: str = "none",
    duplicate: str = "none",
    exon_len: tuple[int, int] = (140, 260),
    n_exons: tuple[int, int] = (3, 4),
) -> Layout:
    """Genes on chr1/chr2/chr3 (gene i on contig i mod 3), plus an optional extra last gene.

    antisense  "none" | "separate" | "overlap": the extra gene sits on the strand opposite to
               gene01 and is either clear of every other gene (chr3, after the last gene) or has
               its first exon overlapping gene01's middle exon by at least 120 bases. Both modes
               give it the same exon lengths and count, and every other draw is identical.
    duplicate  "none" | "same" | "different": the extra gene (chr3) has gene02's exon lengths and
               count; its transcript is an exact copy of gene02's ("same") or unrelated.
    """
    rnd = random.Random(f"{key}-layout")
    contigs = ["chr1", "chr2", "chr3"]
    cursor = {c: rnd.randint(150, 250) for c in contigs}
    flank = {c: rnd.randint(150, 250) for c in contigs}
    genes: list[Gene] = []
    for i in range(n_genes):
        chrom = contigs[i % 3]
        exons = _exons(rnd, cursor[chrom], rnd.randint(*n_exons), exon_len)
        cursor[chrom] = exons[-1][1] + rnd.randint(260, 420)
        n = rnd.randint(*counts)
        genes.append(Gene(f"gene{i + 1:02d}", f"tx{i + 1:02d}", chrom, rnd.choice("+-"), exons, n))
    shift, gap, tail = rnd.randint(40, 80), rnd.randint(90, 140), rnd.randint(140, 200)
    n_extra, strand_extra = rnd.randint(*counts), rnd.choice("+-")
    gene_id, tx_id = f"gene{n_genes + 1:02d}", f"tx{n_genes + 1:02d}"
    if antisense != "none":
        host = genes[0]
        mid = host.exons[1]
        strand = "-" if host.strand == "+" else "+"
        head = (mid[1] + shift) - (mid[0] + 20)  # first exon: from 20 bases into the host's exon
        if antisense == "overlap":
            chrom, first_start = host.chrom, mid[0] + 20
        else:
            chrom, first_start = "chr3", cursor["chr3"] + 100
        exons = ((first_start, first_start + head),)
        exons += ((exons[0][1] + gap, exons[0][1] + gap + tail),)
        genes.append(Gene(gene_id, tx_id, chrom, strand, exons, n_extra))
    elif duplicate != "none":
        src = genes[1]
        pos, built = cursor["chr3"] + 100, []
        for a, b in src.exons:
            built.append((pos, pos + (b - a)))
            pos += (b - a) + rnd.randint(90, 200)
        genes.append(Gene(gene_id, tx_id, "chr3", strand_extra, tuple(built), src.n))
    size = {c: 0 for c in contigs}
    for g in genes:
        size[g.chrom] = max(size[g.chrom], g.end + flank[g.chrom])
    genome = {
        c: "".join(random.Random(f"{key}-genome-{c}").choices("ACGT", k=n))
        for c, n in size.items()
        if n
    }
    layout = Layout(genome, tuple(genes))
    if duplicate == "same":
        return _copy_transcript(layout, from_gene=1, to_gene=len(genes) - 1)
    return layout


def _copy_transcript(layout: Layout, from_gene: int, to_gene: int) -> Layout:
    """Write the transcript of one gene into the exon slots of another (an exact duplicate)."""
    src, dst = layout.genes[from_gene], layout.genes[to_gene]
    sense = layout.sequence(src)
    text = sense if dst.strand == "+" else revcomp(sense)
    chrom = list(layout.genome[dst.chrom])
    pos = 0
    for a, b in dst.exons:
        chrom[a:b] = text[pos : pos + (b - a)]
        pos += b - a
    genome = {**layout.genome, dst.chrom: "".join(chrom)}
    return Layout(genome, layout.genes)


@dataclass(frozen=True)
class Frag:
    name: str
    gene: int  # index into Layout.genes
    a: int  # start in transcript coordinates (sense strand)
    flen: int
    o: str  # "sense": the sequenced strand is the transcript strand; "anti": the opposite


def make_fragments(
    rnd: random.Random,
    layout: Layout,
    library: str,
    *,
    leak: float = 0.0,
    flen: tuple[int, int] = (130, 230),
) -> list[Frag]:
    """Fragments of every gene. forward: read 1 is on the transcript strand; reverse: opposite;
    unstranded: either. `leak` is the share of each gene's fragments that come out with the other
    orientation in a stranded library (a few percent in real libraries): exactly round(leak * n)."""
    frags: list[Frag] = []
    for i, gene in enumerate(layout.genes):
        sense_first = {FORWARD: True, REVERSE: False}.get(library)
        leaked = (
            set(rnd.sample(range(gene.n), round(leak * gene.n)))
            if sense_first is not None
            else set()
        )
        for k in range(gene.n):
            length = rnd.randint(flen[0], min(flen[1], gene.length))
            a = rnd.randint(0, gene.length - length)
            first = rnd.random() < 0.5 if sense_first is None else sense_first != (k in leaked)
            frags.append(
                Frag(f"read{len(frags) + 1:04d}", i, a, length, "sense" if first else "anti")
            )
    return frags


@dataclass(frozen=True)
class Read:
    mate: int
    orient: str  # "sense" | "anti" relative to the transcript
    x: int  # transcript coordinates [x, y), sense strand
    y: int
    seq: str  # as sequenced


def reads_of(layout: Layout, frag: Frag, paired: bool, read_len: int = READ_LEN) -> list[Read]:
    gene = layout.genes[frag.gene]
    text = layout.sequence(gene)[frag.a : frag.a + frag.flen]
    end = frag.a + frag.flen
    if frag.o == "sense":
        first = Read(1, "sense", frag.a, frag.a + read_len, text[:read_len])
        second = Read(2, "anti", end - read_len, end, revcomp(text[-read_len:]))
    else:
        first = Read(1, "anti", end - read_len, end, revcomp(text[-read_len:]))
        second = Read(2, "sense", frag.a, frag.a + read_len, text[:read_len])
    return [first, second] if paired else [first]


def _blocks(gene: Gene, x: int, y: int) -> list[tuple[int, int]]:
    """Genomic intervals (ascending, merged) of the transcript range [x, y)."""
    if gene.strand == "-":
        x, y = gene.length - y, gene.length - x
    out: list[tuple[int, int]] = []
    offset = 0
    for a, b in gene.exons:
        lo, hi = max(x, offset), min(y, offset + b - a)
        if lo < hi:
            piece = (a + lo - offset, a + hi - offset)
            if out and out[-1][1] == piece[0]:
                out[-1] = (out[-1][0], piece[1])
            else:
                out.append(piece)
        offset += b - a
    return out


def _cigar(blocks: list[tuple[int, int]]) -> str:
    parts = []
    for i, (a, b) in enumerate(blocks):
        if i:
            parts.append(f"{a - blocks[i - 1][1]}N")
        parts.append(f"{b - a}M")
    return "".join(parts)


def genome_strand(gene: Gene, orient: str) -> str:
    """The genome strand a read of this orientation aligns to."""
    same = orient == "sense"
    return gene.strand if same else ("-" if gene.strand == "+" else "+")


def sam_genome(
    layout: Layout,
    frags: list[Frag],
    paired: bool,
    *,
    mapq: Any = 60,
    mismatches: Any = None,
    singles: bool = False,
) -> str:
    """Genome-coordinate SAM text. `mapq(frag)` and `mismatches(frag)` (positions to substitute)
    may be callables; `singles` writes only read 1 of a pair as an unpaired record."""
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{len(s)}\n" for c, s in layout.genome.items()
    )
    rows: list[str] = []
    for frag in frags:
        gene = layout.genes[frag.gene]
        quality = mapq(frag) if callable(mapq) else mapq
        subs = mismatches(frag) if callable(mismatches) else ()
        both = paired and not singles
        recs = []
        for read in reads_of(layout, frag, both):
            blocks = _blocks(gene, read.x, read.y)
            bases = "".join(layout.genome[gene.chrom][a:b] for a, b in blocks)
            bases = "".join(
                ("ACGT"[("ACGT".index(c) + 1) % 4] if i in subs else c) for i, c in enumerate(bases)
            )
            minus = genome_strand(gene, read.orient) == "-"
            recs.append((read, blocks, bases, minus))
        for i, (read, blocks, bases, minus) in enumerate(recs):
            pos = blocks[0][0] + 1
            if both:
                other = recs[1 - i]
                flag = (
                    3
                    | (0x10 if minus else 0)
                    | (0x20 if other[3] else 0)
                    | (0x40 if read.mate == 1 else 0x80)
                )
                left = min(r[1][0][0] for r in recs) + 1
                right = max(r[1][-1][1] for r in recs)
                first_left = recs[0][1][0][0] == recs[1][1][0][0]
                tlen = right - left + 1
                tlen = tlen if (pos == left and not (first_left and i == 1)) else -tlen
                mate = ("=", other[1][0][0] + 1, tlen)
            else:
                flag, mate = (0x10 if minus else 0), ("*", 0, 0)
            nm = len(subs) if subs else 0
            rows.append(
                "\t".join(
                    map(
                        str,
                        [frag.name, flag, gene.chrom, pos, quality, _cigar(blocks), *mate, bases]
                        + ["I" * READ_LEN, "NH:i:1", f"NM:i:{nm}"],
                    )
                )
            )
    return header + "\n".join(rows) + "\n"


def sam_transcriptome(layout: Layout, frags: list[Frag], paired: bool) -> str:
    """SAM text against the transcripts (name = Gene.tx), mates adjacent, as RSEM needs it."""
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{g.tx}\tLN:{g.length}\n" for g in layout.genes
    )
    rows: list[str] = []
    for frag in frags:
        gene = layout.genes[frag.gene]
        sense = layout.sequence(gene)
        recs = [(r, sense[r.x : r.y]) for r in reads_of(layout, frag, paired)]
        for i, (read, bases) in enumerate(recs):
            minus = read.orient == "anti"
            if paired:
                other = recs[1 - i][0]
                flag = (
                    3
                    | (0x10 if minus else 0)
                    | (0x20 if other.orient == "anti" else 0)
                    | (0x40 if read.mate == 1 else 0x80)
                )
                left = min(r.x for r, _ in recs) + 1
                span = max(r.y for r, _ in recs) - left + 1
                tlen = (
                    span
                    if (read.x + 1 == left and not (recs[0][0].x == recs[1][0].x and i))
                    else -span
                )
                mate = ("=", other.x + 1, tlen)
            else:
                flag, mate = (0x10 if minus else 0), ("*", 0, 0)
            rows.append(
                "\t".join(
                    map(
                        str,
                        [frag.name, flag, gene.tx, read.x + 1, 60, f"{READ_LEN}M", *mate, bases]
                        + ["I" * READ_LEN, "NH:i:1"],
                    )
                )
            )
    return header + "\n".join(rows) + "\n"


def fastq_text(layout: Layout, frags: list[Frag], paired: bool) -> tuple[str, str]:
    """(mate 1 FASTQ, mate 2 FASTQ) text; the second is empty for single-end reads."""
    mates: dict[int, list[str]] = {1: [], 2: []}
    for frag in frags:
        for read in reads_of(layout, frag, paired):
            mates[read.mate].append(f"@{frag.name}\n{read.seq}\n+\n{'I' * len(read.seq)}\n")
    return "".join(mates[1]), "".join(mates[2])


def gtf_text(layout: Layout) -> str:
    rows = []
    for g in layout.genes:
        attrs = f'gene_id "{g.id}"; transcript_id "{g.tx}";'
        rows.append(
            f'{g.chrom}\tsim\tgene\t{g.start + 1}\t{g.end}\t.\t{g.strand}\t.\tgene_id "{g.id}";'
        )
        rows.append(
            f"{g.chrom}\tsim\ttranscript\t{g.start + 1}\t{g.end}\t.\t{g.strand}\t.\t{attrs}"
        )
        for k, (a, b) in enumerate(g.exons, 1):
            rows.append(
                f'{g.chrom}\tsim\texon\t{a + 1}\t{b}\t.\t{g.strand}\t.\t{attrs} exon_number "{k}";'
            )
    return "\n".join(rows) + "\n"


def transcripts_fasta(layout: Layout) -> str:
    return "".join(f">{g.tx}\n{layout.sequence(g)}\n" for g in layout.genes)


def genome_fasta(layout: Layout) -> str:
    return "".join(f">{c}\n{s}\n" for c, s in layout.genome.items())


def write_transcriptome_bam(work: Path, sam: str, tools: ToolEnv, name: str = "reads_tx") -> None:
    """Unsorted BAM (mates stay adjacent, no @PG line: reproducible bytes)."""
    (work / f"{name}.sam").write_text(sam)
    argv = ["samtools", "view", "--no-PG", "-b", "-o", f"{name}.bam", f"{name}.sam"]
    common.check(tools.run(argv, work), argv)
    (work / f"{name}.sam").unlink()


def counts_by_gene(layout: Layout, frags: list[Frag]) -> dict[str, int]:
    out = {g.id: 0 for g in layout.genes}
    for f in frags:
        out[layout.genes[f.gene].id] += 1
    return out


def wrong_strand_counts(
    layout: Layout, frags: list[Frag], library: str, assumed: str
) -> dict[str, int]:
    """What a count that assumes `assumed` reports for a data set of type `library`."""
    out = {g.id: 0 for g in layout.genes}
    for f in frags:
        # strand of read 1 relative to its gene
        same = f.o == "sense"
        if assumed == UNSTRANDED or same == (assumed == FORWARD):
            out[layout.genes[f.gene].id] += 1
    return out


# ---- F41 -----------------------------------------------------------------------------------

LIBRARY_LINE = {
    FORWARD: "Library: forward-stranded (a read maps to the same strand as its gene).",
    REVERSE: "Library: reverse-stranded, dUTP / fr-firststrand (a read maps to the strand "
    "opposite to its gene).",
    UNSTRANDED: "Library: unstranded (reads map to both strands).",
}
TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "README.md"})


@dataclass(frozen=True)
class Plan:
    layout: Layout
    frags_by_library: dict[str, list[Frag]]
    faulted_library: str

    def library(self, twin: str) -> str:
        return FORWARD if twin == "clean" else self.faulted_library


def make_plan(seed: int) -> Plan:
    """Same genes and fragment positions in both twins; only the orientation draw differs."""
    layout = make_layout(f"f41-{seed}", 5 + seed % 2)
    faulted = REVERSE if seed % 2 == 0 else UNSTRANDED
    frags: dict[str, list[Frag]] = {}
    for library in (FORWARD, faulted):
        frags[library] = make_fragments(
            random.Random(f"f41-{seed}-{library}"), layout, library, leak=LEAK
        )
    return Plan(layout, frags, faulted)


def truth_answers(plan: Plan, twin: str) -> tuple[dict[str, int], dict[str, int]]:
    """(expected, naive): the fragments of each gene, and the counts of `htseq-count -s yes`.

    A correctly stranded count misses the few leaked fragments of a stranded library (at most
    round(LEAK * n) <= 4 per gene), so the answers are compared with a tolerance of TOL."""
    library = plan.library(twin)
    frags = plan.frags_by_library[library]
    return (
        counts_by_gene(plan.layout, frags),
        wrong_strand_counts(plan.layout, frags, library, FORWARD),
    )


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    library = plan.library(twin)
    frags = plan.frags_by_library[library]
    common.write_bam(work, sam_genome(plan.layout, frags, paired=False), tools, "reads")
    (work / "genes.gtf").write_text(gtf_text(plan.layout))
    reads_line = "reads.bam (with reads.bam.bai): single-end RNA-seq reads aligned to the genome."
    if tier == "explicit":
        reads_line += " " + LIBRARY_LINE[library]
    gtf_line = "genes.gtf: gene annotation in GTF format (one transcript per gene)."
    (work / "README.md").write_text(common.readme([reads_line, gtf_line]))
    expected, naive = truth_answers(plan, twin)
    same = sum(wrong_strand_counts(plan.layout, frags, library, FORWARD).values())
    verdict = "fits" if library == FORWARD else "does not fit"
    seen = (
        f"{same} of {len(frags)} reads map to the strand of their gene: a {library} library, "
        f"which {verdict} the default of htseq-count (-s yes)"
    )
    evidence = (seen,)
    if tier == "explicit":
        evidence = (f"README states a {library} library", *evidence)
    what = "the number of reads in reads.bam that come from that gene's transcripts"
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.gtf, {what}",
        f"use htseq-count to report, for each gene in genes.gtf, {what}",
        "a JSON object mapping each gene ID in genes.gtf to its read count (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": TOL}
    task_id = common.task_id("F41", seed, twin, tier, route)
    return Task(task_id, "F41", "strand", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def htseq_counts(work: Path, tools: ToolEnv, *args: str, bam: str = "reads.bam") -> dict[str, int]:
    """Run htseq-count and return {gene: count} (special counters left out)."""
    argv = ["htseq-count", "-q", *args, bam, "genes.gtf"]
    proc = tools.run(argv, work)
    rows = (line.split("\t") for line in proc.stdout.splitlines() if line.strip())
    return {r[0]: int(r[1]) for r in rows if not r[0].startswith("__")}


def infer_library(work: Path, tools: ToolEnv, bam: str = "reads.bam") -> str:
    """The library type from the totals of `-s yes` against `-s reverse` (real tool)."""
    yes = sum(htseq_counts(work, tools, "-s", "yes", bam=bam).values())
    rev = sum(htseq_counts(work, tools, "-s", "reverse", bam=bam).values())
    if yes + rev == 0:
        return UNSTRANDED
    share = yes / (yes + rev)
    return FORWARD if share >= 0.8 else REVERSE if share <= 0.2 else UNSTRANDED


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    library = infer_library(work, tools)
    option = {FORWARD: "yes", REVERSE: "reverse", UNSTRANDED: "no"}[library]
    flags = (
        []
        if library == FORWARD
        else [f"{library} library: htseq-count -s {option}, not the default"]
    )
    return _result(htseq_counts(work, tools, "-s", option), f"htseq-count -s {option}", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(htseq_counts(work, tools), "htseq-count with its defaults")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped when some command runs htseq-count with a strand-specific `-s` (the default, `yes`, or
# `reverse`) that is not the one the library of the workspace implies. The library is read from the
# BAM flags and genes.gtf (the same-strand share: about 96 % forward -> yes, about 4 % -> reverse,
# about half -> no), so `-s reverse` is trapped on a forward or unstranded BAM as well. `-s no` is
# safe on all three: the genes of F41 do not overlap, so counting both strands loses nothing.
HTSEQ = re.compile(r"\bhtseq-count\b")
STRANDED_OPT = re.compile(r"(?<![\w-])(?:-s\s*|--stranded[= ])\s*(yes|no|reverse)\b")


def _same_strand_share(work: Path) -> float | None:
    """Share of the reads overlapping a gene that sit on the gene's strand (stdlib only)."""
    import gzip
    import struct

    try:
        genes = [
            (c[0], int(c[3]) - 1, int(c[4]), c[6])
            for c in (line.split("\t") for line in (work / "genes.gtf").read_text().splitlines())
            if len(c) > 8 and c[2] == "exon"
        ]
        data = gzip.open(work / "reads.bam").read()
        (l_text,) = struct.unpack_from("<i", data, 4)
        offset = 8 + l_text
        (n_ref,) = struct.unpack_from("<i", data, offset)
        offset += 4
        names = []
        for _ in range(n_ref):
            (l_name,) = struct.unpack_from("<i", data, offset)
            names.append(data[offset + 4 : offset + 4 + l_name - 1].decode())
            offset += 4 + l_name + 4
        same = other = 0
        while offset < len(data):
            (size,) = struct.unpack_from("<i", data, offset)
            ref, pos, l_name, _mapq, _bin, n_cigar, flag, l_seq = struct.unpack_from(
                "<iiBBHHHi", data, offset + 4
            )
            cig_at = offset + 4 + 32 + l_name
            span = sum(
                (c >> 4)
                for c in struct.unpack_from(f"<{n_cigar}I", data, cig_at)
                if (c & 15) in (0, 2, 3)
            )
            strand = "-" if flag & 16 else "+"
            hits = {g[3] for g in genes if g[0] == names[ref] and pos < g[2] and pos + span > g[1]}
            if len(hits) == 1:
                if strand in hits:
                    same += 1
                else:
                    other += 1
            offset += 4 + size
    except (OSError, ValueError, struct.error, EOFError):
        return None
    return same / (same + other) if same + other else None


def implied_option(share: float) -> str:
    """The `-s` value that fits a library with this share of reads on the strand of their gene."""
    return "yes" if share >= 0.8 else "reverse" if share <= 0.2 else "no"


def exposed(commands: list[str], work: Path) -> bool:
    share = None
    for command in commands:
        if not HTSEQ.search(command):
            continue
        for part in re.split(r"[|;&\n]", command):
            if not HTSEQ.search(part):
                continue
            opt = STRANDED_OPT.search(part)
            chosen = opt.group(1) if opt else "yes"
            if chosen == "no":
                continue
            if share is None:
                share = _same_strand_share(work)
            if share is not None and chosen != implied_option(share):
                return True
    return False


SPEC = FamilySpec(
    id="F41",
    slot="strand",
    source="GEN-22 (htseq-count --stranded defaults to yes), GEN-20",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
