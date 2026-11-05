"""F16 tss_upstream (strand): the 100 bp upstream of a TSS is left of a + gene, right of a - gene.

The TSS is the gene start (0-based start0) for a + gene and its last base (end0 - 1) for a - gene;
upstream is the 100 bp before it along the gene's strand, TSS base excluded: 0-based [start0-100,
start0) for +, [end0, end0+100) for -. `bedtools flank -l 100 -r 0` is strand-blind unless `-s`
is given (its help says so), and silently builds the + window for every gene.

The VCF and the expected answer are shared by the twins. Each gene owns a slot of the contig; its
upstream window sits at the same place in both twins. In the faulted twin the - genes are mirrored
so that the window is to their right: the gene body lies left of the window, and the strand-blind
window (left of the body) holds its own variants, which are false positives.
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

WINDOW = 100
CONTIGS = {"chr1": 1500, "chr2": 1500}
SLOTS, SLOT_WIDTH, SLOT_START = 3, 470, 30  # three genes per contig
VCF_LINE = "variants.vcf.gz (with variants.vcf.gz.csi): SNVs, VCF 4.2, contigs chr1 and chr2"


@dataclass(frozen=True)
class Gene:
    id: str
    chrom: str
    start0: int
    end0: int
    strand: str


def window0(gene: Gene, strand_aware: bool = True) -> tuple[int, int]:
    """0-based half-open upstream window; strand_aware False is the + window for every gene."""
    if strand_aware and gene.strand == "-":
        return gene.end0, gene.end0 + WINDOW
    return gene.start0 - WINDOW, gene.start0


def hits(variants: list[tuple[str, str, int]], genes: list[Gene], strand_aware: bool) -> list[str]:
    """IDs of the variants (id, chrom, pos1) inside any gene's upstream window."""
    wins = [(g.chrom, *window0(g, strand_aware)) for g in genes]
    return [
        vid for vid, chrom, pos in variants if any(c == chrom and s < pos <= e for c, s, e in wins)
    ]


def _layout(rnd: random.Random) -> tuple[list[tuple[str, int, int]], set[int]]:
    """(chrom, a, length) per gene, where a is the 0-based end of the shared window, and which
    genes are on the - strand in the faulted twin."""
    slots = []
    for chrom in CONTIGS:
        for k in range(SLOTS):
            a = SLOT_START + SLOT_WIDTH * k + 320 + rnd.randint(0, 20)
            slots.append((chrom, a, rnd.randint(80, 120)))
    return slots, set(rnd.sample(range(len(slots)), rnd.randint(2, 4)))


def _genes(slots: list[tuple[str, int, int]], minus: set[int], faulted: bool) -> list[Gene]:
    genes = []
    for i, (chrom, a, length) in enumerate(slots):
        if faulted and i in minus:  # mirrored: the window [a-100, a) is right of the gene
            genes.append(Gene(f"gene{i + 1:02d}", chrom, a - WINDOW - length, a - WINDOW, "-"))
        else:
            genes.append(Gene(f"gene{i + 1:02d}", chrom, a, a + length, "+"))
    return genes


def _positions(
    rnd: random.Random, slots: list[tuple[str, int, int]], minus: set[int]
) -> list[tuple[str, int]]:
    """1-based variant positions: both window edges, the base before and the base after the
    window, interior ones, and for - genes the same around the strand-blind window."""
    chosen: dict[tuple[str, int], None] = {}
    forbidden: set[tuple[str, int]] = set()
    for i, (chrom, a, length) in enumerate(slots):
        lo, hi = a - WINDOW + 1, a  # 1-based first and last base of the true window
        inside = {lo, hi, *rnd.sample(range(lo + 1, hi), rnd.randint(1, 2))}
        chosen.update(((chrom, p), None) for p in inside | {lo - 1, hi + 1})
        forbidden |= {(chrom, p) for p in range(lo, hi + 1)}
        if i in minus:
            start0 = a - WINDOW - length  # strand-blind window: 1-based start0-99 .. start0
            lo, hi = start0 - WINDOW + 1, start0
            inside = {lo, hi, rnd.randint(lo + 1, hi - 1)}
            chosen.update(((chrom, p), None) for p in inside)
            forbidden |= {(chrom, p) for p in range(lo, hi + 1)}
    target = len(chosen) + 8  # eight background variants away from every window
    while len(chosen) < target:
        chrom = rnd.choice(list(CONTIGS))
        pos = rnd.randint(12, CONTIGS[chrom] - 12)
        if (chrom, pos) not in forbidden:
            chosen[(chrom, pos)] = None
    order = {c: i for i, c in enumerate(CONTIGS)}
    return sorted(chosen, key=lambda cp: (order[cp[0]], cp[1]))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rnd = random.Random(f"f16-{seed}")
    slots, minus = _layout(rnd)
    positions = _positions(rnd, slots, minus)
    variants = [(f"v{i:02d}", c, p) for i, (c, p) in enumerate(positions)]
    clean_genes, faulted_genes = _genes(slots, minus, False), _genes(slots, minus, True)
    expected = hits(variants, clean_genes, True)
    if hits(variants, faulted_genes, True) != expected:
        raise RuntimeError("the twins must share the expected answer")
    genes = clean_genes if twin == "clean" else faulted_genes
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in CONTIGS.items()
    )
    body = ""
    for vid, chrom, pos in variants:
        ref = rnd.choice("ACGT")
        alt = rnd.choice([b for b in "ACGT" if b != ref])
        body += f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t.\n"
    common.write_vcf_text(work, header, body, tools)
    bed = "".join(f"{g.chrom}\t{g.start0}\t{g.end0}\t{g.id}\t0\t{g.strand}\n" for g in genes)
    (work / "genes.bed").write_text(bed)
    (work / "genome.sizes").write_text("".join(f"{c}\t{n}\n" for c, n in CONTIGS.items()))
    gene_line = "genes.bed: genes in BED6 format (chrom, start, end, gene_id, score, strand)."
    if tier == "explicit":
        gene_line += (
            " BED intervals: start is 0-based, end is exclusive. The TSS of a + gene is its first"
            " base (start); the TSS of a - gene is its last base (end - 1)."
        )
    readme = [VCF_LINE + ".", gene_line, "genome.sizes: contig lengths, tab-separated."]
    (work / "README.md").write_text(common.readme(readme))
    if twin == "clean":
        evidence = ("genes.bed column 6 is + for every gene",)
    else:
        evidence = (
            "genes.bed column 6 holds - for some genes; the task says upstream along the strand",
            "bedtools flank help: -s defines -l and -r by strand, off by default",
        ) + (("README states which base is the TSS of a - gene",) if tier == "explicit" else ())
    prompt = common.prompt_for(
        route,
        "list the IDs of the variants in variants.vcf.gz that lie in the 100 bp immediately "
        "upstream of the transcription start site (TSS) of any gene in genes.bed, upstream being "
        "taken along the gene's strand and the TSS base itself not included",
        "use `bedtools flank` to build the 100 bp windows immediately upstream of the "
        "transcription start site (TSS) of the genes in genes.bed (upstream along the gene's "
        "strand, the TSS base itself not included) and list the IDs of the variants in "
        "variants.vcf.gz that lie in these windows",
        "a JSON list of variant IDs",
    )
    naive = hits(variants, genes, False)
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F16", seed, twin, tier, route)
    return Task(task_id, "F16", "strand", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _upstream_ids(work: Path, tools: ToolEnv, *strand: str) -> list[str]:
    flank = ["bedtools", "flank", "-i", "genes.bed", "-g", "genome.sizes", "-l", "100", "-r", "0"]
    windows = tools.run([*flank, *strand], work).stdout
    argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "stdin", "-u"]
    lines = tools.run(argv, work, stdin=windows).stdout.splitlines()
    return sorted(line.split("\t")[2] for line in lines if line.strip())


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_upstream_ids(work, tools, "-s"), "bedtools flank -l 100 -r 0 -s, intersect")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_upstream_ids(work, tools), "bedtools flank -l 100 -r 0, intersect")


# Exposure: a strand-blind window built from genes.bed, i.e. `bedtools flank` / `slop` without -s,
# or arithmetic on the start (`$2-100`, `start - 100`) in a command that never looks at the strand.
SEGMENT = re.compile(r"[|;&\n]")
FLANK = re.compile(r"\bbedtools\s+(?:flank|slop)\b")
STRAND_FLAG = re.compile(r"(?<!\S)-s(?!\S)")
START_MINUS = re.compile(r"(?:\$2|\bstart0?|\bs)\s*-\s*100\b")
STRAND_USED = re.compile(r"\$6|strand|['\"]-['\"]|== ?-", re.I)


def _blind(command: str) -> bool:
    if "genes" not in command:
        return False
    if any(FLANK.search(s) and not STRAND_FLAG.search(s) for s in SEGMENT.split(command)):
        return True
    return bool(START_MINUS.search(command)) and not STRAND_USED.search(command)


def exposed(commands: list[str], work: Path) -> bool:
    return any(_blind(c) for c in commands)


SPEC = FamilySpec(
    id="F16",
    slot="strand",
    source="strand-aware promoter windows; bedtools flank ignores strand unless -s",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"genes.bed", "README.md"}),
)
