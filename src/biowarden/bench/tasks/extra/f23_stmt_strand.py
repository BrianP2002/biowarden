"""F23 stmt_strand (locus task_statement): the sentence says which way the RNA-seq library is
stranded, and in the faulted twin the read flags say the other.

The task sentence states the library type of reads.bam: reverse-stranded (dUTP: a read maps to the
strand opposite to its gene) on even seeds, forward-stranded (a read maps to the strand of its gene)
on odd seeds. The clean twin follows it. In the faulted twin the same reads, at the same positions
and from the same genes, carry the other strand flags (and the matching reverse-complemented SEQ).
Task: per gene, the number of reads that come from the gene's transcript, i.e. that overlap the gene
on the strand the real library type implies. Counted with the stated type on a faulted twin, every
gene gets only its 1-3 antisense reads (a few percent of the reads, as in real stranded libraries)
instead of its 12-45 sense reads.

Verified on bedtools 2.31.1 (exit 0, stderr empty): `bedtools coverage -a genes.bed -b reads.bam
-counts -s` counts the reads on the gene's own strand and `-S` the reads on the opposite strand
(the BAM strand is flag 0x10); the same holds for `bedtools intersect -c`. Neither prints anything
about the library. The evidence is one command away: `-s` against `-S` totals, or the share of
reads on each gene's own strand (RSeQC infer_experiment does the same on a real library, GEN-20).

No read straddles a gene edge (reads lie inside a gene or at least 100 bp from every gene), so the
counts do not depend on an overlap rule. The first reads of the sorted BAM are intergenic with mixed
strands, so `samtools view reads.bam | head` does not give the library away.
"""

from __future__ import annotations

import gzip
import random
import re
import struct
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

FORWARD, REVERSE = "forward", "reverse"
OTHER = {FORWARD: REVERSE, REVERSE: FORWARD}
READ_LEN = 50
GENES_PER_CONTIG = 4
TWIN_FILES = frozenset({"reads.bam", "reads.bam.bai", "README.md"})
COMPLEMENT = str.maketrans("ACGT", "TGCA")
STATEMENT = {
    REVERSE: "the library is reverse-stranded (dUTP): a read maps to the strand opposite to the "
    "gene it comes from",
    FORWARD: "the library is forward-stranded: a read maps to the same strand as the gene it "
    "comes from",
}
TRUE_SENTENCE = {
    REVERSE: "Library type: reverse-stranded (dUTP), a read maps to the strand opposite its gene.",
    FORWARD: "Library type: forward-stranded, a read maps to the same strand as its gene.",
}


def flip(strand: str) -> str:
    return "-" if strand == "+" else "+"


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Gene:
    id: str
    chrom: str
    start0: int
    end0: int
    strand: str


@dataclass(frozen=True)
class Read:
    name: str
    chrom: str
    pos: int  # 1-based leftmost base
    kind: str  # sense | anti (relative to the gene transcript) | inter (intergenic)
    gene: int  # index into Plan.genes (-1 for intergenic)
    strand: str  # inter reads only: the alignment strand


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    genome: dict[str, str]
    genes: list[Gene]
    reads: list[Read]
    stated: str  # the library type the task sentence claims

    def real(self, twin: str) -> str:
        return self.stated if twin == "clean" else OTHER[self.stated]

    def alignment_strand(self, read: Read, library: str) -> str:
        if read.kind == "inter":
            return read.strand
        gene = self.genes[read.gene].strand
        sense = gene if library == FORWARD else flip(gene)
        return sense if read.kind == "sense" else flip(sense)

    def counts(self, data_library: str, counting_library: str) -> dict[str, int]:
        """Reads per gene on the strand `counting_library` implies, in a BAM of `data_library`."""
        out = {g.id: 0 for g in self.genes}
        for r in self.reads:
            if r.gene < 0:
                continue
            gene = self.genes[r.gene]
            same = self.alignment_strand(r, data_library) == gene.strand
            if same == (counting_library == FORWARD):
                out[gene.id] += 1
        return out


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f23-{seed}")
    genes: list[Gene] = []
    contigs: dict[str, int] = {}
    n_minus = rnd.randint(3, 5)
    minus = set(rnd.sample(range(2 * GENES_PER_CONTIG), n_minus))
    for chrom, lead in (("chr1", rnd.randint(330, 380)), ("chr2", rnd.randint(150, 250))):
        cursor = lead
        for _ in range(GENES_PER_CONTIG):
            length = rnd.randint(350, 600)
            strand = "-" if len(genes) in minus else "+"
            genes.append(Gene(f"gene{len(genes) + 1:02d}", chrom, cursor, cursor + length, strand))
            cursor += length + rnd.randint(260, 420)
        contigs[chrom] = cursor + rnd.randint(0, 150)
    genome = {c: "".join(rnd.choices("ACGT", k=n)) for c, n in contigs.items()}

    taken: set[tuple[str, int]] = set()
    reads: list[Read] = []

    def add(chrom: str, pos: int, kind: str, gene: int, strand: str = "+") -> None:
        taken.add((chrom, pos))
        reads.append(Read(f"read{len(reads) + 1:03d}", chrom, pos, kind, gene, strand))

    def free(chrom: str, lo: int, hi: int) -> list[int]:
        return [p for p in range(lo, hi + 1) if (chrom, p) not in taken]

    for p in rnd.sample(range(1, 151), 6):  # leading intergenic reads: the head of the BAM
        add("chr1", p, "inter", -1, rnd.choice("+-"))
    for i, g in enumerate(genes):
        lo, hi = g.start0 + 1, g.end0 - READ_LEN + 1
        for p in rnd.sample(free(g.chrom, lo, hi), rnd.randint(12, 45)):
            add(g.chrom, p, "sense", i)
        for p in rnd.sample(free(g.chrom, lo, hi), rnd.randint(1, 3)):
            add(g.chrom, p, "anti", i)
    for g, nxt in zip(genes, [*genes[1:], None], strict=True):
        same_contig = nxt is not None and nxt.chrom == g.chrom
        top = (nxt.start0 - 150) if same_contig else contigs[g.chrom] - READ_LEN
        for p in rnd.sample(free(g.chrom, g.end0 + 101, top + 1), rnd.randint(1, 2)):
            add(g.chrom, p, "inter", -1, rnd.choice("+-"))
    return Plan(contigs, genome, genes, reads, REVERSE if seed % 2 == 0 else FORWARD)


def sam_text(plan: Plan, library: str) -> str:
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in plan.contigs.items()
    )
    rows = []
    for r in plan.reads:
        strand = plan.alignment_strand(r, library)
        seq = plan.genome[r.chrom][r.pos - 1 : r.pos - 1 + READ_LEN]
        seq = seq if strand == "+" else revcomp(seq)
        flag = 0 if strand == "+" else 16
        rows.append(
            f"{r.name}\t{flag}\t{r.chrom}\t{r.pos}\t60\t{READ_LEN}M\t*\t0\t0\t{seq}\t"
            + "I" * READ_LEN
        )
    return header + "\n".join(rows) + "\n"


def truth_answers(plan: Plan, twin: str) -> tuple[dict[str, int], dict[str, int]]:
    """(expected, naive): reads per gene under the real library type, and under the stated one."""
    real = plan.real(twin)
    return plan.counts(real, real), plan.counts(real, plan.stated)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    real = plan.real(twin)
    common.write_bam(work, sam_text(plan, real), tools)
    (work / "genes.bed").write_text(
        "".join(f"{g.chrom}\t{g.start0}\t{g.end0}\t{g.id}\t0\t{g.strand}\n" for g in plan.genes)
    )
    reads_line = "reads.bam (with reads.bam.bai): single-end RNA-seq reads aligned to the genome."
    if tier == "explicit":
        reads_line += " " + TRUE_SENTENCE[real]
    genes_line = "genes.bed: genes in BED6 format (chrom, start, end, gene_id, score, strand)."
    (work / "README.md").write_text(common.readme([reads_line, genes_line]))
    same = sum(plan.counts(real, FORWARD).values())
    opposite = sum(plan.counts(real, REVERSE).values())
    if twin == "clean":
        evidence = (
            f"{same if real == FORWARD else opposite} of {same + opposite} gene reads "
            f"agree with the stated {real}-stranded library",
        )
        if tier == "explicit":
            evidence = ("README states the same library type as the task sentence",) + evidence
    else:
        seen = (
            f"bedtools coverage -s counts {same} reads and -S {opposite}: the reads sit on their "
            f"genes' {'own' if real == FORWARD else 'opposite'} strand, a {real}-stranded library, "
            f"not the stated {plan.stated} one"
        )
        evidence = (f"README states a {real}-stranded library, against the task sentence", seen)
        evidence = evidence if tier == "explicit" else (seen,)
    what = (
        "the number of reads in reads.bam that come from that gene's transcripts (reads that "
        "overlap the gene on the strand the library type implies)"
    )
    stated = STATEMENT[plan.stated]
    prompt = common.prompt_for(
        route,
        f"report, for each gene in genes.bed, {what}; {stated}",
        f"use `bedtools coverage -counts` to report, for each gene in genes.bed, {what}; {stated}",
        "a JSON object mapping each gene ID in genes.bed to its read count (an integer)",
    )
    expected, naive = truth_answers(plan, twin)
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F23", seed, twin, tier, route)
    return Task(task_id, "F23", "strand", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _coverage(work: Path, tools: ToolEnv, option: str) -> dict[str, int]:
    argv = ["bedtools", "coverage", "-a", "genes.bed", "-b", "reads.bam", "-counts", option]
    rows = (line.split("\t") for line in tools.run(argv, work).stdout.splitlines() if line)
    return {r[3]: int(r[-1]) for r in rows}


def bam_reads(path: Path) -> list[tuple[str, int, int, int]]:
    """(chrom, 0-based pos, length, flag) of every record, from the BGZF/BAM bytes (no samtools)."""
    data = gzip.open(path).read()
    (l_text,) = struct.unpack_from("<i", data, 4)
    offset = 8 + l_text
    (n_ref,) = struct.unpack_from("<i", data, offset)
    offset += 4
    names = []
    for _ in range(n_ref):
        (l_name,) = struct.unpack_from("<i", data, offset)
        names.append(data[offset + 4 : offset + 4 + l_name - 1].decode())
        offset += 4 + l_name + 4
    out = []
    while offset < len(data):
        (size,) = struct.unpack_from("<i", data, offset)
        ref, pos, _, _, _, _, flag, l_seq = struct.unpack_from("<iiBBHHHi", data, offset + 4)
        out.append((names[ref], pos, l_seq, flag))
        offset += 4 + size
    return out


def real_library(work: Path) -> str | None:
    """FORWARD if most reads on genes sit on the gene's own strand, REVERSE if on the other."""
    try:
        genes = [line.split("\t") for line in (work / "genes.bed").read_text().splitlines() if line]
        reads = bam_reads(work / "reads.bam")
    except (OSError, ValueError, struct.error, EOFError):
        return None
    same = other = 0
    for chrom, pos, length, flag in reads:
        strand = "-" if flag & 16 else "+"
        for g in genes:
            if g[0] == chrom and pos < int(g[2]) and pos + length > int(g[1]):
                same += strand == g[5]
                other += strand != g[5]
    if same == other:
        return None
    return FORWARD if same > other else REVERSE


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def _stated(task: Task) -> str:
    return REVERSE if "reverse-stranded" in task.prompt else FORWARD


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    same, opposite = _coverage(work, tools, "-s"), _coverage(work, tools, "-S")
    library = FORWARD if sum(same.values()) > sum(opposite.values()) else REVERSE
    flags = []
    if library != _stated(task):
        flags = [
            f"reads sit on the {'same' if library == FORWARD else 'opposite'} strand as their "
            f"genes: a {library}-stranded library, not the {_stated(task)} one in the task sentence"
        ]
    counts = same if library == FORWARD else opposite
    return _result(counts, f"bedtools coverage {'-s' if library == FORWARD else '-S'}", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    option = "-S" if _stated(task) == REVERSE else "-s"
    return _result(_coverage(work, tools, option), f"bedtools coverage {option}, as the task says")


# ---- exposure ----------------------------------------------------------------------------------
# A stranded count of reads.bam against the genes: bedtools coverage/intersect/map/multicov with -s
# (forward: the read strand must equal the gene's) or -S (reverse: it must differ), featureCounts
# -s 1|2, htseq-count --stranded yes|reverse. A bedtools call may be an argument list in a script
# (["bedtools", "coverage", ..., "-S"]). A command with both -s and -S compares them and is set
# aside. A model that believes the other library type can also swap the strand column of the genes
# (or of the reads) and keep -s: -s on swapped strands is the reverse count, -S the forward one.
# The swap is recognised in the call itself (a process substitution, a pipe) or through the files
# a command writes after swapping, which a later call then names. The session is exposed if some
# command counts with a library type other than the one the reads carry (on the faulted twin: the
# stated one); the real type is read from the BAM flags in the workspace. Counting by flag
# arithmetic (samtools view -f/-F 16, pysam) is not recognised.
ARGV_SEP = r"""\\?['"]\s*,\s*\\?['"]"""  # quote, comma, quote: between two items of an argv list
BEDTOOLS = re.compile(
    rf"\bbedtools(?P<sep>\s+|{ARGV_SEP})(?:coverage|intersect|map|multicov|annotate)\b"
)
CONTINUATION = re.compile(r"\\\n\s*")  # a backslash at the end of a line joins it to the next
SAME_FLAG = re.compile(r"(?<![\w-])-s(?![\w-])")
OPPOSITE_FLAG = re.compile(r"(?<![\w-])-S(?![\w-])")
FEATURE_COUNTS = re.compile(r"\bfeatureCounts\b[^|;&\n]*?(?<!\S)-s\s*([12])\b")
HTSEQ = re.compile(r"\bhtseq-count\b[^|;&\n]*?(?:-s\s*|--stranded[= ])(yes|reverse)\b")


def _swaps(a: str, b: str) -> list[str]:
    """Ways to write "a becomes b, anything else becomes a" for two quoted strand symbols."""
    return [
        rf"{a}\s*\)?\s*\?\s*{b}\s*:\s*{a}",  # awk: $6=="+" ? "-" : "+"
        rf"{b}\s+if\b[^\n]*?{a}\s+else\s+{a}",  # python: "-" if s == "+" else "+"
        rf"=={a}\s*\)\s*\$?\w+\s*=\s*{b}\s*;?\s*else",  # awk: if ($6=="+") $6="-"; else ...
        rf"\{{\s*{a}\s*:\s*{b}\s*,\s*{b}\s*:\s*{a}\s*\}}",  # python: {"+": "-", "-": "+"}
    ]


_QUOTE = r"""\\?["']"""
_PLUS, _MINUS = rf"{_QUOTE}\+{_QUOTE}", rf"{_QUOTE}-{_QUOTE}"
STRAND_SWAP = re.compile(
    "|".join(
        [
            *_swaps(_PLUS, _MINUS),
            *_swaps(_MINUS, _PLUS),
            r"\btr\s+['\"]?\\?\+-['\"]?\s+['\"]?-\\?\+['\"]?",  # tr '+-' '-+'
            r"\by/\\?\+-/-\\?\+/",  # sed y/+-/-+/
            r"""\bmaketrans\(\s*['"]\+-['"]\s*,\s*['"]-\+['"]""",  # str.maketrans('+-', '-+')
        ]
    )
)
# a file a command writes: a redirect target (not 2>/dev/null, not an awk comparison), open(.., "w")
WRITTEN = re.compile(
    r"(?<![\d&<])>>?\s*['\"]?([\w./~-]*\.[\w./~-]+)|\bopen\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]"
)


def _swapped_files(command: str) -> set[str]:
    """The names (without directory) of the files a command writes right after swapping strands."""
    files = set()
    for swap in STRAND_SWAP.finditer(command):
        if target := WRITTEN.search(command, swap.end()):
            files.add(Path(next(name for name in target.groups() if name)).name)
    return files


def _call(command: str, call: re.Match[str]) -> str:
    """The text of the bedtools call that `call` begins.

    A call on the command line ends where its pipeline stage does (`|`, `;`, `&`, a newline) or at
    the bracket that closes the enclosing `$( )`; an argument list ends at its closing `]`, even
    over several lines. Brackets inside the call (`<( )`) stay inside it.
    """
    start, in_list = call.start(), not call.group("sep").isspace()
    depth = 0
    for i in range(start, len(command)):
        char = command[i]
        if char in "([":
            depth += 1
        elif char in ")]":
            if depth == 0:
                return command[start:i]
            depth -= 1
        elif char in "|;&\n" and depth == 0 and not in_list:
            return command[start:i]
    return command[start:]


def _swapped(command: str, start: int, call: str, swapped: Collection[str]) -> bool:
    """Does the call count on strands that a swap turned around? Two swaps cancel out."""
    line = command[command.rfind("\n", 0, start) + 1 : start]
    piped = line.rstrip().endswith("|") and bool(STRAND_SWAP.search(line))  # a pipe from the swap
    inline = bool(STRAND_SWAP.search(call))  # a swap inside the call: -a <(awk ... genes.bed)
    inputs = WRITTEN.sub(" ", call)  # a redirect target is no input, even if a swap wrote it too
    files = sum(bool(re.search(rf"(?<![\w.-]){re.escape(n)}(?![\w.-])", inputs)) for n in swapped)
    return (piped + inline + files) % 2 == 1


def reading_of(command: str, swapped: Collection[str] = ()) -> str | None:
    """The library type a command counts with, or None (not a stranded count, or a comparison).

    `swapped` names the files that earlier commands wrote with the strand column swapped.
    """
    command = CONTINUATION.sub(" ", command)
    readings = set()
    for match in BEDTOOLS.finditer(command):
        call = _call(command, match)
        same, opposite = SAME_FLAG.search(call), OPPOSITE_FLAG.search(call)
        if same and opposite:
            return None
        if same or opposite:
            reading = FORWARD if same else REVERSE
            if _swapped(command, match.start(), call, swapped):
                reading = OTHER[reading]
            readings.add(reading)
    if readings or BEDTOOLS.search(command):
        return readings.pop() if len(readings) == 1 else None
    if match := FEATURE_COUNTS.search(command):
        return FORWARD if match.group(1) == "1" else REVERSE
    if match := HTSEQ.search(command):
        return FORWARD if match.group(1) == "yes" else REVERSE
    return None


def exposed(commands: list[str], work: Path) -> bool:
    """True iff some command counted with a library type other than the one the reads carry.

    Commands are read in order, so a file written with swapped strands is known to the calls
    that come after it.
    """
    swapped: set[str] = set()
    readings = []
    for line in commands:
        command = CONTINUATION.sub(" ", line)
        if STRAND_SWAP.search(command):
            swapped |= _swapped_files(command)
        if (reading := reading_of(command, swapped)) is not None:
            readings.append(reading)
    if not readings:
        return False
    real = real_library(work)
    return real is not None and any(r != real for r in readings)


SPEC = FamilySpec(
    id="F23",
    slot="strand",
    source="GEN-20, GEN-21, GEN-22 (library strandedness declared against the reads' flags)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="task_statement",
)
