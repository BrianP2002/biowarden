"""Rules for deepTools 3.5.6 (pack iv): bamCoverage, computeMatrix, multiBamSummary.

Every claim below was run on deepTools 3.5.6 (bio-py environment) on tiny synthetic BAM and bigWig
files (50-base reads on a 3-30 kb genome); the evidence is quoted per rule.

iv_deeptools.matrix_missing_data (slot missing_value, CALL rule, tier T1, declaration
`missing_signal` zero|na, optional). computeMatrix writes NaN for every bin that has no interval in
the bigWig (a track written with `bamCoverage --skipNonCoveredRegions`, or from a gapped bedGraph),
and `--missingDataAsZero` turns them into 0. Verified: a 3-gene BED over a sparse bigWig gives a
matrix row `nan nan` for the gene with no entries (`bs.bw`, exit 0, stderr empty) and `0 0` with
`--missingDataAsZero`; on the F73 files (14-16 genes, 20 bins each) the mean of the matrix entries
that are not NaN is 1.7 to 2.3 times the mean over all bins (seeds 0-29; numpy `nanmean` and pandas
`mean` skip NaN, so an average over genes counts only the bins that have reads). The rule reads
the bigWig with a stdlib
decoder (header, chromosome tree, R-tree, zlib sections), builds the window of every BED region the
way computeMatrix does (scale-regions: region plus flanks; reference-point: TSS/TES/center with
-b/-a, strand aware) and counts the bins (size --binSize) without any entry. Fails when such bins
exist and the call has no `--missingDataAsZero`, unless `missing_signal=na` is declared (the
agent wants NaN); `missing_signal=zero` without the flag fails; the flag together with
`missing_signal=na` fails. Dense tracks (bamCoverage default writes zeros) and BED windows that the
track covers fully stay silent.

iv_deeptools.matrix_skip_zeros (slot record_count, CALL rule, tier T1, declaration `zero_regions`
drop|keep, optional). `computeMatrix --skipZeros` removes every region whose values are all zero
(or NaN) from the matrix; the only trace is `Skipping gC, due to being absent in the computeMatrix
output.` on stderr, and `--outFileSortedRegions` lists the kept regions (verified: 3 regions, one
without signal, matrix of 2 rows, the regions file holds gA and gB). A mean over the matrix is then
a mean over the regions that have signal. The rule fires when `--skipZeros` is present and at least
one region has no non-zero entry in any of the bigWigs, unless `zero_regions=drop` is declared;
`zero_regions=keep` with `--skipZeros` fails.

iv_deeptools.bamcoverage_track_units (slot units, CALL rule, tier T0, declaration `track_unit`
reads_per_bin|depth|normalized, REQUIRED). Verified: `bamCoverage -b a.bam -o a.bg -of bedgraph`
(defaults `--normalizeUsing None`, `--binSize 50`) writes for 120 reads of 50 bases the value 32 in
the bin chr1:1000-1050, while the mean per-base depth of the bin (bedtools genomecov) is 26.74: a
bin value is the number of reads that overlap the bin; `--binSize 1` gives the per-base depth (sum
of value x length = 6000 = 120 reads x 50); `--normalizeUsing CPM` gives 266667 = 32 x 1e6/120
(flagged duplicates count in the library size). The options decide what the track means (a
`--scaleFactor` or a `--normalizeUsing` other than None: normalized; `--binSize 1`: depth;
otherwise reads_per_bin) and the declaration must say the same; the rule cannot know which one the
task wants.

iv_deeptools.duplicates_semantics (slot flag_semantics, CALL rule, tier T1, declaration
`duplicates` keep|exclude, optional). Verified on bamCoverage and multiBamSummary: reads flagged
0x400 are COUNTED by default (bin value 31 -> 11 with `--samFlagExclude 1024`); `--ignoreDuplicates`
keeps one read per (start, strand) and does not look at the flag: a stack of 10 unflagged reads
at one start became 1 (multiBamSummary R2 count 10 -> 1), and 5 reads flagged 0x400 at 5 distinct
starts stayed 5 (R3 count 5 -> 5; `--samFlagExclude 1024` gave 0). For paired reads the mate
position joins the key (three pairs starting at 100 with mates at 300, 350 and 300: count 3 -> 2).
Duplicates marked by 5' end (a reverse-strand duplicate trimmed to another length has another
start) therefore survive `--ignoreDuplicates`. On a BAM that has flagged duplicates the rule
groups the first 20000 alignments by (contig, start, strand, mate) the way deepTools does and
compares the read set left by `--ignoreDuplicates` with the unflagged set; a difference fails the
call unless `--samFlagExclude` removes 1024 as well. `duplicates=keep` with an exclusion option, or
`duplicates=exclude` without one, fails.

iv_deeptools.multibamsummary_regions (slot record_order, CALL rule, tier T1, declaration
`region_join` coordinates, optional). Verified: `multiBamSummary BED-file --outRawCounts` writes
`#'chr' 'start' 'end' 'x.bam'` rows ordered by the contig order of the BAM header and then by start
(BAM header chr2, chr10, chr1 with a BED listing chr1, chr10, chr2, chr1: rows chr2, chr10, chr1 50,
chr1 100), with no region names, and drops a region on a contig the BAM does not have without a
message (6 BED rows, `Number of bins found: 5`, exit 0). Matching counts to names by line number
then shifts. The rule fires when a BED row is on a contig absent from the BAM header or the BED is
not in output order; `region_join=coordinates` (the agent joins on chr/start/end) passes.

Limits: only the first 20000 alignments of a BAM are read; the NaN test works on bins of the
declared --binSize in genomic coordinates (a scaled body counts bins of binSize x length/body
length); bigWig files over 16 MB, big-endian files and GTF region files are not judged (unknown or
pass).
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from science_harness.engine.argv import has_option
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed, suggester
from science_harness.engine.rules import aln_bam
from science_harness.engine.spec import Rule

BIGWIG_LIMIT = 16_000_000
BIGWIG_MAGIC = 0x888FFC26
CHROM_TREE_MAGIC = 0x78CA8C91
RTREE_MAGIC = 0x2468ACE0
MAX_BLOCKS = 4000
MAX_REGIONS = 2000
BAM_SAMPLE = 20_000

# ---- command line ----------------------------------------------------------------------------

ALIASES = {
    "computeMatrix": {
        "-S": "--scoreFileName",
        "-R": "--regionsFileName",
        "-b": "--upstream",
        "-a": "--downstream",
        "-m": "--regionBodyLength",
        "-bs": "--binSize",
        "-p": "--numberOfProcessors",
        "-o": "--outFileName",
        "-bl": "--blackListFileName",
    },
    "bamCoverage": {
        "-b": "--bam",
        "-o": "--outFileName",
        "-of": "--outFileFormat",
        "-bs": "--binSize",
        "-p": "--numberOfProcessors",
        "-e": "--extendReads",
        "-r": "--region",
        "-bl": "--blackListFileName",
        "-v": "--verbose",
    },
    "bamCompare": {
        "-b1": "--bamfile1",
        "-b2": "--bamfile2",
        "-o": "--outFileName",
        "-of": "--outFileFormat",
        "-bs": "--binSize",
        "-p": "--numberOfProcessors",
        "-e": "--extendReads",
    },
    "multiBamSummary": {
        "-b": "--bamfiles",
        "-o": "--outFileName",
        "-p": "--numberOfProcessors",
        "-bs": "--binSize",
        "-e": "--extendReads",
        "-r": "--region",
    },
}
MULTI = {
    "--scoreFileName",
    "--regionsFileName",
    "--bamfiles",
    "--BED",
    "--labels",
    "--samplesLabel",
}
FLAGS = {
    "--skipZeros",
    "--missingDataAsZero",
    "--ignoreDuplicates",
    "--skipNonCoveredRegions",
    "--nanAfterEnd",
    "--verbose",
    "--version",
    "--help",
    "-h",
    "--smartLabels",
    "--exactScaling",
    "--MNase",
    "--centerReads",
}
OPTIONAL_VALUE = {"--extendReads"}  # `-e` may stand alone or take a length


def _numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse(tool: str, argv: Sequence[str]) -> tuple[dict[str, list[str]], list[str]]:
    """(canonical option -> values, positionals). A flag has an empty list; multi-value options
    (-S, -R, -b of multiBamSummary, --BED) take every following token up to the next option."""
    alias = ALIASES.get(tool, {})
    opts: dict[str, list[str]] = {}
    positionals: list[str] = []
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        i += 1
        if token == "--":
            positionals.extend(items[i:])
            break
        if not token.startswith("-") or _numeric(token) or token == "-":
            positionals.append(token)
            continue
        name, eq, value = token.partition("=") if token.startswith("--") else (token, "", "")
        name = alias.get(name, name)
        if name in FLAGS:
            opts.setdefault(name, [])
        elif name in MULTI:
            values = [value] if eq else []
            while i < len(items) and (not items[i].startswith("-") or _numeric(items[i])):
                values.append(items[i])
                i += 1
            opts.setdefault(name, []).extend(values)
        elif eq:
            opts.setdefault(name, []).append(value)
        elif name in OPTIONAL_VALUE:
            if i < len(items) and _numeric(items[i]):
                opts.setdefault(name, []).append(items[i])
                i += 1
            else:
                opts.setdefault(name, [])
        elif i < len(items) and (not items[i].startswith("-") or _numeric(items[i])):
            opts.setdefault(name, []).append(items[i])
            i += 1
        else:
            opts.setdefault(name, [])
    return opts, positionals


def last(opts: dict[str, list[str]], name: str) -> str | None:
    return opts[name][-1] if opts.get(name) else None


def _int(text: str | None, default: int) -> int:
    try:
        return int(float(text)) if text is not None else default
    except ValueError:
        return default


def existing(ctx: Ctx, tokens: Sequence[str], suffixes: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for token in tokens:
        if token.lower().endswith(suffixes):
            path = ctx.resolve(token)
            if path.is_file() and path not in found:
                found.append(path)
    return found


# ---- bigWig decoder (standard library) ---------------------------------------------------------


class Unreadable(Exception):
    """A file the decoder cannot read (too large, big-endian, malformed)."""


@dataclass
class BigWig:
    chroms: dict[str, int]
    intervals: dict[str, list[tuple[int, int, float]]] = field(default_factory=dict)

    def covered(self, chrom: str, start: int, end: int) -> list[tuple[int, int]]:
        """Merged covered stretches inside [start, end)."""
        out: list[tuple[int, int]] = []
        for s, e, _ in self.intervals.get(chrom, []):
            if e <= start or s >= end:
                continue
            s, e = max(s, start), min(e, end)
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return out

    def nonzero(self, chrom: str, start: int, end: int) -> bool:
        return any(
            v != 0 and v == v for s, e, v in self.intervals.get(chrom, []) if e > start and s < end
        )


def _chrom_tree(data: bytes, offset: int) -> dict[int, tuple[str, int]]:
    magic, _block, key_size, _val, _items = struct.unpack_from("<IIIIQ", data, offset)
    if magic != CHROM_TREE_MAGIC:
        raise Unreadable("bad chromosome tree")
    out: dict[int, tuple[str, int]] = {}

    def node(at: int) -> None:
        is_leaf, _res, count = struct.unpack_from("<BBH", data, at)
        at += 4
        for _ in range(count):
            key = data[at : at + key_size].split(b"\0", 1)[0].decode("ascii", "replace")
            at += key_size
            if is_leaf:
                chrom_id, size = struct.unpack_from("<II", data, at)
                at += 8
                out[chrom_id] = (key, size)
            else:
                (child,) = struct.unpack_from("<Q", data, at)
                at += 8
                node(child)

    node(offset + 32)
    return out


def _leaves(data: bytes, root: int) -> list[tuple[int, int]]:
    leaves: list[tuple[int, int]] = []

    def node(at: int) -> None:
        is_leaf, _res, count = struct.unpack_from("<BBH", data, at)
        at += 4
        for _ in range(count):
            if is_leaf:
                _sc, _sb, _ec, _eb, offset, size = struct.unpack_from("<IIIIQQ", data, at)
                at += 32
                leaves.append((offset, size))
                if len(leaves) > MAX_BLOCKS:
                    raise Unreadable("too many data blocks")
            else:
                _sc, _sb, _ec, _eb, child = struct.unpack_from("<IIIIQ", data, at)
                at += 24
                node(child)

    node(root)
    return leaves


def _section(block: bytes, names: dict[int, tuple[str, int]], out: dict) -> None:
    at = 0
    while at + 24 <= len(block):
        chrom_id, start, _end, step, span, kind, _res, count = struct.unpack_from(
            "<IIIIIBBH", block, at
        )
        at += 24
        name = names[chrom_id][0]
        rows = out.setdefault(name, [])
        for i in range(count):
            if kind == 1:
                s, e, v = struct.unpack_from("<IIf", block, at)
                at += 12
            elif kind == 2:
                s, v = struct.unpack_from("<If", block, at)
                at += 8
                e = s + span
            elif kind == 3:
                (v,) = struct.unpack_from("<f", block, at)
                at += 4
                s = start + i * step
                e = s + span
            else:
                raise Unreadable("unknown section type")
            rows.append((s, e, v))


def decode_bigwig(data: bytes) -> BigWig:
    """Chromosome sizes and every data interval of a little-endian bigWig."""
    try:
        magic, _ver, _zoom, tree_off, data_off, index_off = struct.unpack_from("<IHHQQQ", data, 0)
        if magic != BIGWIG_MAGIC:
            raise Unreadable("not a (little-endian) bigWig")
        (buf_size,) = struct.unpack_from("<I", data, 52)
        names = _chrom_tree(data, tree_off)
        out: dict[str, list[tuple[int, int, float]]] = {}
        if index_off:
            if struct.unpack_from("<I", data, index_off)[0] != RTREE_MAGIC:
                raise Unreadable("bad R-tree")
            for offset, size in _leaves(data, index_off + 48):
                raw = data[offset : offset + size]
                _section(zlib.decompress(raw) if buf_size else raw, names, out)
        for rows in out.values():
            rows.sort()
        return BigWig({n: s for n, s in names.values()}, out)
    except (struct.error, zlib.error, KeyError, IndexError, RecursionError) as exc:
        raise Unreadable(f"{type(exc).__name__}: {exc}") from exc


def read_bigwig(ctx: Ctx, path: Path) -> BigWig:
    if path.stat().st_size > BIGWIG_LIMIT:
        raise Unreadable("bigWig larger than 16 MB")
    return decode_bigwig(ctx.read_bytes(path, limit=BIGWIG_LIMIT))


# ---- computeMatrix windows -----------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int
    strand: str
    row: int


def read_regions(ctx: Ctx, path: Path) -> list[Region]:
    rows: list[Region] = []
    for line in ctx.lines(path, limit=MAX_REGIONS + 10):
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        if len(f) < 3:
            f = line.split()
        try:
            start, end = int(f[1]), int(f[2])
        except (ValueError, IndexError):
            continue
        strand = f[5] if len(f) > 5 and f[5] in "+-" else "+"
        rows.append(Region(f[0], start, end, strand, len(rows)))
        if len(rows) >= MAX_REGIONS:
            break
    return rows


def window(region: Region, mode: str, up: int, down: int, point: str) -> tuple[int, int]:
    """The genomic stretch computeMatrix reads for one region (strand aware)."""
    left, right = (up, down) if region.strand != "-" else (down, up)
    if mode == "scale-regions":
        return region.start - left, region.end + right
    if point == "center":
        ref = (region.start + region.end) // 2
    elif (point == "TES") == (region.strand != "-"):
        ref = region.end
    else:
        ref = region.start
    return ref - left, ref + right


def matrix_geometry(
    opts: dict[str, list[str]], subcommand: str | None
) -> tuple[str, int, int, int, int]:
    """(mode, upstream, downstream, bin size, scaled body length) of a computeMatrix call."""
    mode = "reference-point" if subcommand == "reference-point" else "scale-regions"
    up = _int(last(opts, "--upstream"), 0)
    down = _int(last(opts, "--downstream"), 0)
    bin_size = max(1, _int(last(opts, "--binSize"), 10))
    body = max(1, _int(last(opts, "--regionBodyLength"), 1000))
    return mode, up, down, bin_size, body


@dataclass
class MatrixScan:
    regions: int = 0
    with_gaps: int = 0
    gap_bins: int = 0
    all_zero: list[str] = field(default_factory=list)
    sample: str = ""
    track: str = ""


def scan_matrix(call: Call, ctx: Ctx) -> MatrixScan | None:
    """Look at the bigWigs and regions of a computeMatrix call; None when nothing can be judged."""
    opts, positionals = parse("computeMatrix", call.argv)
    sub = call.subcommand
    tokens = [*opts.get("--scoreFileName", []), *opts.get("--regionsFileName", []), *positionals]
    bigwigs = existing(ctx, tokens, (".bw", ".bigwig"))
    beds = existing(ctx, tokens, (".bed", ".bed.gz", ".tsv", ".txt", ".narrowpeak"))
    if not bigwigs or not beds:
        return None
    mode, up, down, bin_size, body = matrix_geometry(opts, sub)
    point = last(opts, "--referencePoint") or "TSS"
    tracks = [(p, read_bigwig(ctx, p)) for p in bigwigs]
    scan = MatrixScan()
    for bed in beds:
        for region in read_regions(ctx, bed):
            ws, we = window(region, mode, up, down, point)
            scale = max(1.0, (region.end - region.start) / body) if mode == "scale-regions" else 1
            eff_bin = max(1, round(bin_size * scale))
            seen = False
            gap = 0
            zero_everywhere = True
            for path, track in tracks:
                size = track.chroms.get(region.chrom)
                if size is None:
                    continue
                seen = True
                lo, hi = max(0, ws), min(size, we)
                if hi <= lo:
                    continue
                cov = track.covered(region.chrom, lo, hi)
                edges = [lo, *[x for pair in cov for x in pair], hi]
                holes = [
                    (edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if k + 1 < len(edges)
                ]
                n = sum((b - a) // eff_bin for a, b in holes if b - a >= eff_bin)
                if n and not scan.track:
                    scan.track = path.name
                gap += n
                if track.nonzero(region.chrom, lo, hi):
                    zero_everywhere = False
            if not seen:
                continue
            scan.regions += 1
            if gap:
                scan.with_gaps += 1
                scan.gap_bins += gap
                scan.sample = scan.sample or f"{region.chrom}:{region.start}-{region.end}"
            if zero_everywhere:
                scan.all_zero.append(f"{region.chrom}:{region.start}-{region.end}")
    return scan


@checker("iv_deeptools_matrix_missing_data")
def iv_deeptools_matrix_missing_data(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    if call.subcommand not in ("scale-regions", "reference-point"):
        return passed()
    declared = call.declared.get(key)
    flag = has_option(call.argv, "--missingDataAsZero")
    if flag:
        if declared == "na":
            return CheckResult(
                "fail",
                "declared_na_with_zero_flag",
                f"you declared {key}=na but --missingDataAsZero turns every bin without an entry "
                "into 0, so no NaN will be left.",
                expected="na",
                got="zero",
                fix=("drop --missingDataAsZero to keep NaN for bins without data",),
            )
        return passed()
    if declared == "na":
        return passed()
    try:
        scan = scan_matrix(call, ctx)
    except (Unreadable, OSError, ValueError) as exc:
        return CheckResult(
            "unknown",
            "bigwig_unreadable",
            f"could not read the bigWig to look for bins without data ({exc}).",
            fix=(f"state --declare {key}=zero and add --missingDataAsZero, or {key}=na",),
        )
    if scan is None or scan.with_gaps == 0:
        return passed()
    message = (
        f"{scan.track} has no entry for {scan.gap_bins} bins of {scan.with_gaps} of the "
        f"{scan.regions} regions (for example {scan.sample}): computeMatrix writes NaN for them, "
        "so any mean that skips NaN (numpy nanmean, pandas mean, a plot of the mean profile) "
        "covers only the bins that have signal and overstates it."
    )
    if declared == "zero":
        message = f"you declared {key}=zero but this call has no --missingDataAsZero. " + message
    return CheckResult(
        "fail",
        "missing_bins_not_zero",
        message,
        expected="zero",
        got="na",
        fix=(
            "add --missingDataAsZero (a bin with no entry means no reads) and "
            f"--declare {key}=zero",
            f"or --declare {key}=na if NaN is wanted and the average skips them on purpose",
        ),
    )


@checker("iv_deeptools_matrix_skip_zeros")
def iv_deeptools_matrix_skip_zeros(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    if call.subcommand not in ("scale-regions", "reference-point"):
        return passed()
    declared = call.declared.get(key)
    if declared == "keep":
        return CheckResult(
            "fail",
            "declared_keep_with_skip_zeros",
            f"you declared {key}=keep but --skipZeros removes every region without signal from "
            "the matrix.",
            expected="keep",
            got="drop",
            fix=("drop --skipZeros so every region keeps its row",),
        )
    if declared == "drop":
        return passed()
    try:
        scan = scan_matrix(call, ctx)
    except (Unreadable, OSError, ValueError) as exc:
        return CheckResult(
            "unknown",
            "bigwig_unreadable",
            f"could not read the bigWig to look for regions without signal ({exc}).",
            fix=(f"state --declare {key}=drop if losing regions without signal is intended",),
        )
    if scan is None or not scan.all_zero:
        return passed()
    shown = ", ".join(scan.all_zero[:3])
    return CheckResult(
        "fail",
        "zero_regions_dropped",
        f"{len(scan.all_zero)} of {scan.regions} regions have no signal in the bigWig ({shown}"
        f"{', ...' if len(scan.all_zero) > 3 else ''}) and --skipZeros removes them from the "
        "matrix, so means over the matrix are means over the regions with signal and row numbers "
        "no longer match the BED.",
        expected="keep",
        got="drop",
        fix=(
            "drop --skipZeros (and --missingDataAsZero if empty bins mean no reads) to keep one "
            "row per region",
            f"or --declare {key}=drop and add --outFileSortedRegions to see which regions stayed",
        ),
    )


# ---- bamCoverage track units ---------------------------------------------------------------------

NORMALIZERS = {"None", "RPKM", "CPM", "BPM", "RPGC"}
UNIT_WORDS = {
    "reads_per_bin": "the number of reads that overlap each bin (raw counts)",
    "depth": "the per-base read depth (raw counts, bin size 1)",
    "normalized": "a normalised or scaled value (not comparable to a raw count)",
}
UNIT_FIX = {
    "reads_per_bin": "leave --binSize at its default or above 1 and omit --normalizeUsing",
    "depth": "add --binSize 1 and omit --normalizeUsing and --scaleFactor",
    "normalized": "add --normalizeUsing CPM (or RPKM, BPM, RPGC with --effectiveGenomeSize)",
}


def coverage_unit(argv: Sequence[str]) -> tuple[str | None, str]:
    opts, _ = parse("bamCoverage", argv)
    norm = last(opts, "--normalizeUsing")
    if norm is not None and norm not in NORMALIZERS:
        return None, f"--normalizeUsing {norm} is not a deepTools choice"
    if opts.get("--scaleFactor"):
        return "normalized", "--scaleFactor"
    if norm not in (None, "None"):
        return "normalized", f"--normalizeUsing {norm}"
    bin_size = _int(last(opts, "--binSize"), 50)
    if bin_size == 1:
        return "depth", "--binSize 1"
    return "reads_per_bin", f"--binSize {bin_size}" + (
        "" if opts.get("--binSize") else " (default)"
    )


@checker("iv_deeptools_track_units")
def iv_deeptools_track_units(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    implied, why = coverage_unit(call.argv)
    if implied is None:
        return CheckResult("unknown", "unmapped_value", why)
    declared = call.declared[rule.declare.key]
    if implied == declared:
        return passed()
    return CheckResult(
        "fail",
        "declared_mismatch",
        f"you declared {rule.declare.key}={declared} but this bamCoverage call writes "
        f"{UNIT_WORDS[implied]} ({why}); bamCoverage defaults to --normalizeUsing None and "
        "--binSize 50, and a bin value is the number of reads touching the bin, not the depth.",
        expected=declared,
        got=implied,
        fix=(UNIT_FIX[declared],),
    )


@suggester("iv_deeptools_track_units")
def _track_units_fix(call: Call, rule: Rule) -> tuple[str, ...]:
    key = rule.declare.key if rule.declare else "track_unit"
    implied, why = coverage_unit(call.argv)
    if implied is None:
        return (f"re-run with --declare {key}=<reads_per_bin|depth|normalized>",)
    return (
        f"this call writes {UNIT_WORDS[implied]} ({why}): re-run with --declare {key}={implied}",
        f"or change the options for another unit: {UNIT_FIX['depth']}, or {UNIT_FIX['normalized']}",
    )


# ---- duplicates ----------------------------------------------------------------------------------

DUP_FLAG = 0x400


def exclusion_flags(opts: dict[str, list[str]]) -> int:
    value = last(opts, "--samFlagExclude")
    if value is None:
        return 0
    try:
        return int(value, 0)
    except ValueError:
        return 0


@dataclass
class DupScan:
    reads: int = 0
    flagged: int = 0
    unflagged: int = 0
    kept_by_position: int = 0
    surviving_flagged: int = 0
    collapsed_unflagged: int = 0
    complete: bool = True


def scan_duplicates(ctx: Ctx, path: Path) -> DupScan:
    aln = aln_bam.read_alignment(ctx, path, BAM_SAMPLE)
    groups: dict[tuple, list[bool]] = {}
    scan = DupScan(complete=aln.complete)
    for rec in aln.records:
        if rec.flag & 4:
            continue
        scan.reads += 1
        flagged = bool(rec.flag & DUP_FLAG)
        scan.flagged += flagged
        key = (rec.ref, rec.pos, rec.flag & 16)
        if rec.flag & 1:
            key += (rec.mate_ref, rec.mate_pos)
        groups.setdefault(key, []).append(flagged)
    scan.unflagged = scan.reads - scan.flagged
    scan.kept_by_position = len(groups)
    for members in groups.values():
        if all(members):
            scan.surviving_flagged += 1
        scan.collapsed_unflagged += max(0, members.count(False) - 1)
    return scan


def dedup_effect(opts: dict[str, list[str]]) -> str:
    if exclusion_flags(opts) & DUP_FLAG:
        return "flag"
    if "--ignoreDuplicates" in opts:
        return "position"
    return "none"


@checker("iv_deeptools_duplicates")
def iv_deeptools_duplicates(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    opts, positionals = parse(call.tool, call.argv)
    bams = existing(
        ctx,
        [
            *opts.get("--bamfiles", []),
            *opts.get("--bam", []),
            *opts.get("--bamfile1", []),
            *opts.get("--bamfile2", []),
            *positionals,
        ],
        (".bam", ".sam"),
    )
    if not bams:
        return passed()
    effect = dedup_effect(opts)
    declared = call.declared.get(key)
    if effect == "none" and declared != "exclude":
        return passed()
    try:
        scans = [(b, scan_duplicates(ctx, b)) for b in bams]
    except (aln_bam.Unreadable, OSError, EOFError, zlib.error) as exc:
        return CheckResult("unknown", "bam_unreadable", f"could not read the BAM ({exc}).")
    flagged = [(b, s) for b, s in scans if s.flagged]
    if not flagged:
        return passed()
    name, scan = flagged[0]
    if effect == "none":
        return CheckResult(
            "fail",
            "duplicates_kept",
            f"you declared {key}=exclude but this call has neither --samFlagExclude 1024 nor "
            f"--ignoreDuplicates, and {name.name} has {scan.flagged} reads flagged 0x400: "
            "deepTools counts flagged duplicates by default.",
            expected="exclude",
            got="keep",
            fix=("add --samFlagExclude 1024 to leave out the reads flagged as duplicates",),
        )
    if declared == "keep":
        return CheckResult(
            "fail",
            "declared_keep_with_exclusion",
            f"you declared {key}=keep but this call removes reads: "
            + (
                "--samFlagExclude drops the reads flagged 0x400."
                if effect == "flag"
                else "--ignoreDuplicates keeps one read per start and strand."
            ),
            expected="keep",
            got="exclude",
            fix=("drop the duplicate options to keep every read",),
        )
    if effect == "flag":
        return passed()
    if scan.kept_by_position == scan.unflagged:
        return passed()
    return CheckResult(
        "fail",
        "ignore_duplicates_not_the_flag",
        f"--ignoreDuplicates keeps one read per start and strand and ignores the duplicate flag: "
        f"{name.name} has {scan.flagged} reads flagged 0x400, {scan.surviving_flagged} of them "
        f"start where no unflagged read starts and would stay, while {scan.collapsed_unflagged} "
        f"unflagged reads would be collapsed ({scan.kept_by_position} reads left against "
        f"{scan.unflagged} unflagged).",
        expected=scan.unflagged,
        got=scan.kept_by_position,
        fix=(
            "use --samFlagExclude 1024 to drop exactly the reads flagged as duplicates "
            "(and no --ignoreDuplicates)",
            f"or --declare {key}=keep with no dedup option if duplicates should be counted",
        ),
    )


# ---- multiBamSummary regions ---------------------------------------------------------------------


@checker("iv_deeptools_multibamsummary_regions")
def iv_deeptools_multibamsummary_regions(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.subcommand != "BED-file":
        return passed()
    if call.declared.get(rule.declare.key) == "coordinates":
        return passed()
    opts, positionals = parse("multiBamSummary", call.argv)
    bams = existing(ctx, [*opts.get("--bamfiles", []), *positionals], (".bam",))
    beds = existing(ctx, opts.get("--BED", []), (".bed", ".bed.gz", ".tsv", ".txt"))
    if not bams or not beds:
        return passed()
    try:
        refs = aln_bam.read_alignment(ctx, bams[0], 1).refs
    except (aln_bam.Unreadable, OSError, EOFError, zlib.error) as exc:
        return CheckResult("unknown", "bam_unreadable", f"could not read the BAM header ({exc}).")
    order = {name: i for i, name in enumerate(refs)}
    absent: list[str] = []
    out_of_order = 0
    for bed in beds:
        previous = (-1, -1)
        for region in read_regions(ctx, bed):
            if region.chrom not in order:
                absent.append(f"{region.chrom}:{region.start}-{region.end}")
                continue
            here = (order[region.chrom], region.start)
            if here < previous:
                out_of_order += 1
            previous = max(previous, here)
    if not absent and not out_of_order:
        return passed()
    parts = []
    if absent:
        parts.append(
            f"{len(absent)} region(s) lie on contigs that {bams[0].name} does not have "
            f"({', '.join(absent[:3])}) and are dropped without a message"
        )
    if out_of_order:
        parts.append(
            f"{out_of_order} BED row(s) are out of the order multiBamSummary writes (contig order "
            "of the BAM header, then start)"
        )
    return CheckResult(
        "fail",
        "regions_reordered_or_dropped",
        "multiBamSummary BED-file writes rows with no region names: "
        + "; ".join(parts)
        + ", so counts matched to regions by line number shift.",
        fix=(
            "sort the BED by the contig order of the BAM header and then start "
            "(and keep only contigs the BAM has), or join the counts on chr/start/end",
            f"then --declare {rule.declare.key}=coordinates",
        ),
    )
