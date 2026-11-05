"""Shared readers for the RNA-seq quantification rules (pack `quant`): standard library only.

Nothing here is a rule. The checkers in quant_htseq.py, quant_featurecounts.py, quant_rsem.py,
quant_stringtie.py and quant_salmon.py use these helpers to look at the files a call names:

  alignments   `iter_alignments` reads BAM (BGZF: gzip members) or SAM text, at most `limit`
               records, skipping unmapped, secondary and supplementary records
  annotation   `read_exons` reads the exon rows of a GTF (plain or gzip)
  library      `strand_profile` counts, for the reads that overlap genes of one strand only, how
               many sit on the gene's strand (read 1 of a pair, flag 0x10 for the minus strand) and
               classifies the library: forward (>= 85 % same strand), reverse (<= 15 %), unstranded
               (35-65 %), otherwise unknown; fewer than `min_reads` informative reads is unknown
  options      `split_args` separates option values from positional arguments for one tool

The strand test has two limits: it looks at the first `limit` alignments only (a coordinate-sorted
BAM starts on the first contigs) and it needs the reads to overlap annotated exons. A library with
many reads outside the annotation can come out `unknown`, which the rules treat as "cannot tell".
"""

from __future__ import annotations

import bisect
import gzip
import re
import struct
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from science_harness.engine.checks import CheckResult, Ctx, passed

FORWARD, REVERSE, UNSTRANDED, UNKNOWN = "forward", "reverse", "unstranded", "unknown"
STRANDED = (FORWARD, REVERSE)
DEFAULT_LIMIT = 5000
MIN_READS = 20
GZIP_MAGIC = b"\x1f\x8b"
BAM_MAGIC = b"BAM\x01"
CIGAR_REF = {0, 2, 3, 7, 8}  # M D N = X consume the reference
CIGAR_BLOCK = {0, 2, 7, 8}  # ... and N splits the block


@dataclass(frozen=True)
class Aln:
    name: str
    flag: int
    chrom: str
    start0: int
    end0: int
    mapq: int
    blocks: tuple[tuple[int, int], ...]  # aligned blocks (split at N), 0-based half-open
    nh: int  # NH tag, 1 when absent

    @property
    def minus(self) -> bool:
        return bool(self.flag & 0x10)

    @property
    def paired(self) -> bool:
        return bool(self.flag & 0x1)


def _need_files(ctx: Ctx) -> None:
    ctx._need_files()  # T0 refuses to read files; a T1 checker may


def _blocks(start0: int, ops: Sequence[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int]:
    blocks: list[tuple[int, int]] = []
    pos = start0
    open_at: int | None = None
    for length, op in ops:
        if op in CIGAR_BLOCK:
            if open_at is None:
                open_at = pos
            pos += length
        elif op == 3:  # N: close the block
            if open_at is not None:
                blocks.append((open_at, pos))
                open_at = None
            pos += length
    if open_at is not None:
        blocks.append((open_at, pos))
    return tuple(blocks), pos


CIGAR_OPS = "MIDNSHP=X"
_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")


def _sam_alignment(line: str) -> Aln | None:
    f = line.split("\t")
    if len(f) < 11 or line.startswith("@"):
        return None
    flag, pos, mapq = int(f[1]), int(f[3]), int(f[4])
    if f[2] == "*" or f[5] == "*":
        return None
    ops = [(int(n), CIGAR_OPS.index(c)) for n, c in _CIGAR.findall(f[5])]
    blocks, end0 = _blocks(pos - 1, ops)
    nh = 1
    for tag in f[11:]:
        if tag.startswith("NH:i:"):
            nh = int(tag[5:])
    return Aln(f[0], flag, f[2], pos - 1, end0, mapq, blocks, nh)


def _tag_nh(tags: bytes) -> int:
    """NH from the optional fields of a BAM record (1 when absent or unreadable)."""
    sizes = {"A": 1, "c": 1, "C": 1, "s": 2, "S": 2, "i": 4, "I": 4, "f": 4}
    i = 0
    try:
        while i + 3 <= len(tags):
            tag, kind = tags[i : i + 2].decode(), chr(tags[i + 2])
            i += 3
            if kind in sizes:
                if tag == "NH" and kind in "cCsSiI":
                    code = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I"}[kind]
                    return struct.unpack_from("<" + code, tags, i)[0]
                i += sizes[kind]
            elif kind in "ZH":
                i = tags.index(b"\0", i) + 1
            elif kind == "B":
                sub = chr(tags[i])
                (count,) = struct.unpack_from("<i", tags, i + 1)
                i += 5 + count * sizes[sub]
            else:
                return 1
    except (ValueError, struct.error, KeyError):
        return 1
    return 1


def _bam_alignments(handle, limit: int) -> Iterator[Aln]:
    if handle.read(4) != BAM_MAGIC:
        raise ValueError("not a BAM file")
    (l_text,) = struct.unpack("<i", handle.read(4))
    handle.read(l_text)
    (n_ref,) = struct.unpack("<i", handle.read(4))
    names = []
    for _ in range(n_ref):
        (l_name,) = struct.unpack("<i", handle.read(4))
        names.append(handle.read(l_name)[:-1].decode())
        handle.read(4)
    for _ in range(limit):
        head = handle.read(4)
        if len(head) < 4:
            return
        (size,) = struct.unpack("<i", head)
        body = handle.read(size)
        if len(body) < size:
            return
        ref, pos, l_name, mapq, _bin, n_cigar, flag, l_seq, *_ = struct.unpack_from(
            "<iiBBHHHiiii", body
        )
        at = 32
        name = body[at : at + l_name - 1].decode(errors="replace")
        at += l_name
        cigar = struct.unpack_from(f"<{n_cigar}I", body, at)
        at += 4 * n_cigar + (l_seq + 1) // 2 + l_seq
        if ref < 0 or not n_cigar:
            continue
        blocks, end0 = _blocks(pos, [(c >> 4, c & 15) for c in cigar])
        yield Aln(name, flag, names[ref], pos, end0, mapq, blocks, _tag_nh(body[at:]))


def iter_alignments(ctx: Ctx, path: Path, limit: int = DEFAULT_LIMIT) -> Iterator[Aln]:
    """Mapped, primary alignments of a BAM or SAM file (at most `limit` records are read)."""
    _need_files(ctx)
    path = ctx.resolve(str(path))
    with path.open("rb") as raw:
        magic = raw.read(2)
    if magic == GZIP_MAGIC:
        with gzip.open(path, "rb") as handle:
            try:
                for aln in _bam_alignments(handle, limit):
                    if not aln.flag & 0x904:
                        yield aln
            except (EOFError, zlib.error, struct.error, ValueError):
                return
        return
    seen = 0
    for line in ctx.lines(path, limit=limit + 10_000):
        aln = _sam_alignment(line) if not line.startswith("@") else None
        if aln is None:
            continue
        seen += 1
        if seen > limit:
            return
        if not aln.flag & 0x904:
            yield aln


@dataclass(frozen=True)
class Exon:
    chrom: str
    start0: int
    end0: int
    strand: str
    gene: str


def read_exons(
    ctx: Ctx, path: Path, feature: str = "exon", attribute: str = "gene_id", limit: int = 400_000
) -> list[Exon]:
    out: list[Exon] = []
    pattern = re.compile(rf'{re.escape(attribute)}[ =]"?([^";]+)"?')
    for line in ctx.lines(path, limit=limit):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != feature or f[6] not in "+-":
            continue
        gene = pattern.search(f[8])
        if gene:
            out.append(Exon(f[0], int(f[3]) - 1, int(f[4]), f[6], gene.group(1)))
    return out


class ExonIndex:
    """Overlap queries against exons: sorted by start with a prefix maximum of the ends."""

    def __init__(self, exons: Sequence[Exon]) -> None:
        by_chrom: dict[str, list[Exon]] = {}
        for exon in exons:
            by_chrom.setdefault(exon.chrom, []).append(exon)
        self._data: dict[str, tuple[list[int], list[int], list[Exon]]] = {}
        for chrom, items in by_chrom.items():
            items.sort(key=lambda e: (e.start0, e.end0))
            starts = [e.start0 for e in items]
            maxend, top = [], 0
            for e in items:
                top = max(top, e.end0)
                maxend.append(top)
            self._data[chrom] = (starts, maxend, items)

    def overlapping(self, chrom: str, blocks: Sequence[tuple[int, int]]) -> list[Exon]:
        if chrom not in self._data:
            return []
        starts, maxend, items = self._data[chrom]
        found: dict[int, Exon] = {}
        for s, e in blocks:
            i = bisect.bisect_left(starts, e) - 1
            while i >= 0 and maxend[i] > s:
                if items[i].end0 > s:
                    found[id(items[i])] = items[i]
                i -= 1
        return list(found.values())


@dataclass(frozen=True)
class StrandProfile:
    library: str  # forward | reverse | unstranded | unknown
    same: int  # reads on the strand of the single gene strand they overlap
    opposite: int
    antisense: int  # reads touching exons of both strands (ambiguous without strand information)
    reads: int  # primary alignments looked at

    @property
    def informative(self) -> int:
        return self.same + self.opposite

    @property
    def share(self) -> float | None:
        return self.same / self.informative if self.informative else None


def classify(same: int, opposite: int, min_reads: int = MIN_READS) -> str:
    total = same + opposite
    if total < min_reads:
        return UNKNOWN
    share = same / total
    if share >= 0.85:
        return FORWARD
    if share <= 0.15:
        return REVERSE
    if 0.35 <= share <= 0.65:
        return UNSTRANDED
    return UNKNOWN


def strand_profile(
    ctx: Ctx,
    bam: Path,
    exons: Sequence[Exon],
    limit: int = DEFAULT_LIMIT,
    min_reads: int = MIN_READS,
) -> StrandProfile:
    index = ExonIndex(exons)
    same = opposite = antisense = reads = 0
    for aln in iter_alignments(ctx, bam, limit):
        if aln.paired and aln.flag & 0x80:
            continue  # read 2 carries the opposite orientation: read 1 decides
        reads += 1
        strands = {e.strand for e in index.overlapping(aln.chrom, aln.blocks)}
        if len(strands) == 2:
            antisense += 1
        elif len(strands) == 1:
            if ("-" if aln.minus else "+") in strands:
                same += 1
            else:
                opposite += 1
    return StrandProfile(classify(same, opposite, min_reads), same, opposite, antisense, reads)


def transcriptome_profile(
    ctx: Ctx, bam: Path, limit: int = DEFAULT_LIMIT, min_reads: int = MIN_READS
) -> StrandProfile:
    """Strand of read 1 against the reference, for alignments to a transcriptome (RSEM input)."""
    same = opposite = reads = 0
    for aln in iter_alignments(ctx, bam, limit):
        if aln.paired and aln.flag & 0x80:
            continue
        reads += 1
        if aln.minus:
            opposite += 1
        else:
            same += 1
    return StrandProfile(classify(same, opposite, min_reads), same, opposite, 0, reads)


def antisense_gene_pairs(exons: Sequence[Exon]) -> list[tuple[str, str]]:
    """Pairs of genes with exons that overlap on opposite strands."""
    pairs: set[tuple[str, str]] = set()
    by_chrom: dict[str, list[Exon]] = {}
    for exon in exons:
        by_chrom.setdefault(exon.chrom, []).append(exon)
    for items in by_chrom.values():
        items.sort(key=lambda e: e.start0)
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if b.start0 >= a.end0:
                    break
                if a.strand != b.strand and a.gene != b.gene:
                    pairs.add(tuple(sorted((a.gene, b.gene))))  # type: ignore[arg-type]
    return sorted(pairs)


# ---- command-line options --------------------------------------------------------------------


def split_args(
    argv: Sequence[str], values: frozenset[str]
) -> tuple[dict[str, list[str]], list[str], set[str]]:
    """(option -> values, positionals, flags present). `values` names the options that take one.

    Handles `--long value`, `--long=value`, `-s value`, `-svalue`. Any other `-x` or `--long` is
    a flag. Everything after a lone `--` is positional.
    """
    opts: dict[str, list[str]] = {}
    positionals: list[str] = []
    flags: set[str] = set()
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        i += 1
        if token == "--":
            positionals.extend(items[i:])
            break
        if token.startswith("--"):
            name, eq, value = token.partition("=")
            if name in values:
                if eq:
                    opts.setdefault(name, []).append(value)
                elif i < len(items):
                    opts.setdefault(name, []).append(items[i])
                    i += 1
            else:
                flags.add(name)
        elif token.startswith("-") and len(token) > 1:
            name = token[:2]
            if name in values:
                if len(token) > 2:
                    opts.setdefault(name, []).append(token[2:].lstrip("="))
                elif i < len(items):
                    opts.setdefault(name, []).append(items[i])
                    i += 1
            else:
                flags.add(name if len(token) == 2 else token)
        else:
            positionals.append(token)
    return opts, positionals, flags


def last_value(opts: dict[str, list[str]], *names: str) -> str | None:
    found = [v for n in names for v in opts.get(n, [])]
    return found[-1] if found else None


# ---- the verdicts shared by the strandedness rules -------------------------------------------

LIBRARY_WORDS = {
    FORWARD: "forward-stranded",
    REVERSE: "reverse-stranded (dUTP)",
    UNSTRANDED: "unstranded",
}


def describe(profile: StrandProfile) -> str:
    words = LIBRARY_WORDS.get(profile.library, profile.library)
    if profile.share is None:
        return words
    return f"{words}: {profile.same} of {profile.informative} reads sit on their gene's strand"


MIN_AFFECTED = 3  # reads in an antisense overlap before unstranded counting counts as a problem


def strand_verdict(
    *,
    tool: str,
    key: str,
    call_lib: str,
    option: str,
    profile: StrandProfile | None,
    declared: str | None,
    unstranded_hurts: bool,
    fix_for: dict[str, str],
    data: str,
) -> CheckResult:
    """The shared decision of the strandedness rules (see each rule module for the evidence).

    call_lib   the library type the call's options imply (forward / reverse / unstranded)
    option     how the call says it, for the message ("-s yes (the default)")
    profile    what the reads show (None: the files cannot be read yet, so the rule is silent)
    declared   the declaration, if any
    unstranded_hurts  an unstranded call on a stranded library loses reads where genes overlap
               on opposite strands (verified for htseq-count and featureCounts only)
    fix_for    library -> option text for the fix
    data       the name of the alignment file, for the message
    """
    if declared is not None and declared != call_lib:
        return CheckResult(
            "fail",
            "declaration_contradicts_options",
            f"you declared {key}={declared} but {option} makes {tool} treat the library as "
            f"{call_lib}.",
            expected=declared,
            got=call_lib,
            fix=(
                f"use {fix_for[declared]} (or declare {key}={call_lib} if that is what you mean)",
            ),
        )
    if profile is None:
        return passed()
    seen = profile.library
    if seen == UNKNOWN:
        if declared is not None or call_lib == UNSTRANDED:
            return passed()
        return CheckResult(
            "unknown",
            "library_not_provable",
            f"{option} tells {tool} the library is {call_lib} but {data} does not show it: "
            f"{profile.same} of {profile.informative} informative reads sit on their gene's strand "
            f"({profile.reads} looked at).",
            fix=(
                f"state the library from the protocol: --declare {key}=forward|reverse|unstranded",
            ),
        )
    if declared is not None and declared != seen:
        return CheckResult(
            "fail",
            "declaration_contradicts_data",
            f"you declared {key}={declared} but {data} is {describe(profile)}.",
            expected=seen,
            got=declared,
            fix=(f"declare {key}={seen} and use {fix_for[seen]}",),
        )
    if call_lib == seen:
        return passed()
    fix = (f"re-run with {fix_for[seen]} (declare {key}={seen})",)
    if call_lib == UNSTRANDED:
        if unstranded_hurts and profile.antisense >= MIN_AFFECTED:
            return CheckResult(
                "fail",
                "unstranded_on_stranded_overlap",
                f"{data} is {describe(profile)} and {profile.antisense} of its {profile.reads} "
                f"reads touch genes on both strands, which {option} cannot tell apart: "
                f"{tool} drops them as ambiguous and neither gene gets them.",
                expected=seen,
                got=call_lib,
                fix=fix,
            )
        return passed()
    counted = profile.same if call_lib == FORWARD else profile.opposite
    lost = (
        "only about half of the reads would be counted"
        if seen == UNSTRANDED
        else f"only {counted} of {profile.informative} informative reads would be counted"
    )
    return CheckResult(
        "fail",
        "stranded_option_mismatch",
        f"{option} tells {tool} the library is {call_lib}, but {data} is {describe(profile)}; "
        f"{lost}.",
        expected=seen,
        got=call_lib,
        fix=fix,
    )
