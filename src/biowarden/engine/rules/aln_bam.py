"""Stdlib readers for the header and the first records of a SAM or BAM file (rule helpers).

The alignment rules (aln_samtools, aln_star, ...) need the header lines and a sample of records:
which program wrote the file, whether the mates sit next to each other, which tags exist, what the
MAPQ values mean. They read them here without samtools or pysam, so a rule keeps working where
samtools is not on the engine's PATH. Only the first `limit` records are decoded, whatever the
file size (BAM is BGZF, which `gzip` reads as concatenated members). CRAM is not read.

Verified against samtools 1.24 on BAM written by STAR 2.7.11b, bwa 0.7.19, bowtie2 2.5.5, hisat2
2.2.3 and minimap2 2.31 (names, flags, positions, MAPQ and the tags NH, XS, AS, NM, nM, MD, SA, RG
decode to what `samtools view` prints).
"""

from __future__ import annotations

import gzip
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biowarden.engine.checks import Ctx

DEFAULT_LIMIT = 20_000
MAX_DECODED = 64_000_000
CIGAR_OPS = "MIDNSHP=X"
_SCALAR = {"c": "<b", "C": "<B", "s": "<h", "S": "<H", "i": "<i", "I": "<I", "f": "<f"}
_SIZE = {"c": 1, "C": 1, "s": 2, "S": 2, "i": 4, "I": 4, "f": 4}
ALIGNMENT_SUFFIXES = (".bam", ".sam", ".sam.gz", ".cram")


class Unreadable(Exception):
    """The file cannot be decoded here (CRAM, truncated header, not an alignment file)."""


@dataclass
class Rec:
    name: str
    flag: int
    ref: str  # "*" when unplaced
    pos: int  # 1-based, 0 when unplaced
    mapq: int
    cigar: str
    mate_ref: str  # "=" resolved to the contig name, "*" when absent
    mate_pos: int
    tlen: int
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def unmapped(self) -> bool:
        return bool(self.flag & 4)

    @property
    def primary(self) -> bool:
        return not self.flag & 0x900


@dataclass
class Alignment:
    kind: str  # bam | sam
    header: list[str]
    refs: list[str]
    records: list[Rec]
    complete: bool  # every record of the file was decoded (the file ended before `limit`)

    def hd(self) -> dict[str, str]:
        for line in self.header:
            if line.startswith("@HD"):
                return dict(f.split(":", 1) for f in line.split("\t")[1:] if ":" in f)
        return {}

    def program_lines(self) -> list[dict[str, str]]:
        out = []
        for line in self.header:
            if line.startswith("@PG"):
                out.append(dict(f.split(":", 1) for f in line.split("\t")[1:] if ":" in f))
        return out

    def read_groups(self) -> list[dict[str, str]]:
        out = []
        for line in self.header:
            if line.startswith("@RG"):
                out.append(dict(f.split(":", 1) for f in line.split("\t")[1:] if ":" in f))
        return out


def _read(handle: Any, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError
    return data


def _aux(buf: bytes, off: int, end: int) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    while off + 3 <= end:
        tag = buf[off : off + 2].decode("ascii", "replace")
        kind = chr(buf[off + 2])
        off += 3
        if kind == "A":
            value: Any = chr(buf[off])
            off += 1
        elif kind in _SCALAR:
            value = struct.unpack_from(_SCALAR[kind], buf, off)[0]
            off += _SIZE[kind]
        elif kind in "ZH":
            stop = buf.index(b"\0", off)
            value = buf[off:stop].decode("ascii", "replace")
            off = stop + 1
        elif kind == "B":
            sub = chr(buf[off])
            count = struct.unpack_from("<i", buf, off + 1)[0]
            off += 5 + count * _SIZE[sub]
            value = None  # array tags are not needed by any rule
        else:
            break
        tags[tag] = value
    return tags


def _bam_records(handle: Any, refs: list[str], limit: int) -> Iterator[Rec]:
    taken = 0
    for _ in range(limit):
        head = handle.read(4)
        if len(head) < 4:
            return
        (size,) = struct.unpack("<i", head)
        block = _read(handle, size)
        taken += size
        if taken > MAX_DECODED:
            raise EOFError
        ref_id, pos, l_name, mapq, _bin, n_cigar, flag, l_seq, nref_id, npos, tlen = (
            struct.unpack_from("<iiBBHHHiiii", block, 0)
        )
        off = 32
        name = block[off : off + l_name - 1].decode("ascii", "replace")
        off += l_name
        ops = struct.unpack_from(f"<{n_cigar}I", block, off) if n_cigar else ()
        off += 4 * n_cigar
        cigar = "".join(f"{op >> 4}{CIGAR_OPS[op & 15]}" for op in ops) or "*"
        off += (l_seq + 1) // 2 + l_seq
        ref = refs[ref_id] if 0 <= ref_id < len(refs) else "*"
        if nref_id < 0 or nref_id >= len(refs):
            mate = "*"
        else:
            mate = refs[nref_id]
        yield Rec(
            name,
            flag,
            ref,
            pos + 1 if pos >= 0 else 0,
            mapq,
            cigar,
            mate,
            npos + 1 if npos >= 0 else 0,
            tlen,
            _aux(block, off, len(block)),
        )


def _read_bam(path: Path, limit: int) -> Alignment:
    with gzip.open(path, "rb") as handle:
        if handle.read(4) != b"BAM\x01":
            raise Unreadable("not a BAM file")
        (l_text,) = struct.unpack("<i", _read(handle, 4))
        text = _read(handle, l_text).decode("ascii", "replace").rstrip("\0")
        (n_ref,) = struct.unpack("<i", _read(handle, 4))
        refs: list[str] = []
        for _ in range(n_ref):
            (l_name,) = struct.unpack("<i", _read(handle, 4))
            refs.append(_read(handle, l_name)[:-1].decode("ascii", "replace"))
            _read(handle, 4)
        records: list[Rec] = []
        complete = True
        try:
            for rec in _bam_records(handle, refs, limit):
                records.append(rec)
            complete = len(records) < limit
        except (EOFError, zlib.error, struct.error, OSError, ValueError):
            complete = False
    return Alignment("bam", text.splitlines(), refs, records, complete)


def _sam_tag(field_text: str) -> tuple[str, Any]:
    tag, kind, value = field_text.split(":", 2)
    if kind == "i":
        return tag, int(value)
    if kind == "f":
        return tag, float(value)
    return tag, value


def _read_sam(ctx: Ctx, path: Path, limit: int) -> Alignment:
    header: list[str] = []
    records: list[Rec] = []
    complete = True
    for line in ctx.lines(path):
        if line.startswith("@"):
            header.append(line)
            continue
        if len(records) >= limit:
            complete = False
            break
        f = line.split("\t")
        if len(f) < 11:
            raise Unreadable("not a SAM file")
        ref = f[2]
        mate = ref if f[6] == "=" else f[6]
        tags = dict(_sam_tag(x) for x in f[11:] if x.count(":") >= 2)
        records.append(
            Rec(f[0], int(f[1]), ref, int(f[3]), int(f[4]), f[5], mate, int(f[7]), int(f[8]), tags)
        )
    refs = [
        h.split("SN:", 1)[1].split("\t")[0] for h in header if h.startswith("@SQ") and "SN:" in h
    ]
    return Alignment("sam", header, refs, records, complete)


def read_alignment(ctx: Ctx, name: str | Path, limit: int = DEFAULT_LIMIT) -> Alignment:
    """Header and the first `limit` records of a SAM or BAM; raises Unreadable or OSError.

    A T1 operation: the context refuses it for a T0 rule. A BAM that ends early (truncated) gives
    what was decoded with `complete=False`.
    """
    ctx._need_files()
    path = ctx.resolve(str(name))
    with path.open("rb") as raw:
        magic = raw.read(4)
    if magic == b"CRAM":
        raise Unreadable("CRAM is not decoded here")
    if magic[:2] == b"\x1f\x8b":
        try:
            with gzip.open(path, "rb") as probe:
                head = probe.read(4)
        except (EOFError, zlib.error, OSError) as exc:
            raise Unreadable(f"corrupt gzip ({type(exc).__name__})") from exc
        if head == b"BAM\x01":
            try:
                return _read_bam(path, limit)
            except (EOFError, zlib.error, struct.error) as exc:
                raise Unreadable(f"truncated BAM header ({type(exc).__name__})") from exc
    if magic[:1] in (b"@",) or path.name.endswith((".sam", ".sam.gz")):
        try:
            return _read_sam(ctx, path, limit)
        except (ValueError, zlib.error, EOFError) as exc:
            raise Unreadable(f"not a SAM file ({type(exc).__name__})") from exc
    raise Unreadable("not a SAM or BAM file")


def alignment_files(ctx: Ctx, tokens: list[str]) -> list[Path]:
    """Existing SAM/BAM/CRAM files among `tokens`, in order, without repeats."""
    found: list[Path] = []
    for token in tokens:
        if token.endswith(ALIGNMENT_SUFFIXES):
            path = ctx.resolve(token)
            if path.is_file() and path not in found:
                found.append(path)
    return found
