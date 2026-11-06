"""bcftools regions/targets conventions: the base of a -R/-T file (GEN-01), what they keep (GEN-02).

Two checkers for engine/specs/bcftools_regions.yaml.

bcftools.regions_file_base (T1, call, declaration optional)
    A -R/-T FILE is read as BED (0-based, half-open) only when its name ends in .bed or .bed.gz
    (case-insensitive); any other name is read as tab-delimited CHROM POS [END], 1-based inclusive.
    The rule is silent when the name already makes bcftools read the file the way it is written.
    It decides only in the risky state: a file read 1-based whose lines look like BED intervals
    (every data line has 3+ tab columns, integer start < end). There it fails without a
    declaration, passes with one_based_closed and fails with zero_based_half_open. Declared
    zero_based_half_open is a stated BED intent, so it fails whatever the content. A file that
    cannot be inspected (missing, pipe, stdin "-", unreadable) gives `unknown` when nothing was
    declared; unknown only warns. When the call is a plain `bcftools view|query|filter|norm|
    annotate|stats` over one VCF/BCF and one -R/-T file, a read-only dry run selects records both
    ways (`view -H -G -t REGION,..`, once with the lines as written, once shifted to BED) and the
    rule passes with a note when the two readings select the same records in the effective overlap
    mode (and, for `query`, which can fall back to POS only, in start-only mode too). Anything the
    dry run cannot do falls back to the static verdict.

bcftools.region_semantics (T0, call, declaration required)
    -R/-r keep records that overlap a region (--regions-overlap, default 1); -T/-t keep records
    whose POS lies in a target (--targets-overlap, default 0). The values 0|pos mean start only,
    1|record and 2|variant mean overlap. The declaration must agree with what the options imply:
    declared overlap fails when any option of the call is start-only; declared start_only fails
    when every option overlaps. Two verified exceptions, where the options do not tell the whole
    story, give `unknown` (a warning, never a block): `bcftools query` with a format that reads no
    INFO/FORMAT field on a text VCF, and `bcftools merge`.

Evidence, bcftools 1.24 (htslib 1.24), scratch directory, synthetic data
  Base. VCF records p99..p102 at POS 99..102, the line "chr1<TAB>100<TAB>101":
    view -R r.bed -> p101; same bytes in r.tsv -> p100 p101 (-T identical). r.BED, r.Bed, a.Bed.Gz
    and a plain-gzip a.bed.gz (or an uncompressed file named *.bed.gz) are BED; r.bed.txt,
    r.bed.tsv, r.bed.bak, rbed, a.bed.bgz, a.bed.gzip, d.bed/regions and "x.bed " (trailing space)
    are 1-based. Plain gzip is read, bgzip is not needed. `-R -` (stdin) is 1-based, `-T ^r.bed`
    excludes with the BED reading (the ^ is dropped before the name test) while `-R ^r.bed` is a
    literal file name (not found).
  Same rule in view, query, filter, annotate, stats, norm, isec, merge, concat -a, gtcheck,
    convert (-g), roh, +fill-tags, `plugin fill-tags`, mpileup and `call -R` (-R/-T); csq and cnv
    list the options in --help but were not run; consensus has no -R/-T; sort/reheader -T are other
    options (temp dir/prefix), which the rule leaves alone. Exception, `call -T`: it reads a
    sites file, only CHROM and POS of each line (END ignored), for any name: on mpileup output
    "chr1 100 105" selects position 100 only, as .bed and as .tsv, while `call -m -R` gives
    101-105 (.bed) and 100-105 (.tsv). The rule gives `unknown` for a BED-like file there.
  A tabix or CSI index next to the file wins over the name: regions.tsv.gz with `tabix -p bed` is
    BED; regions.bed.gz with `tabix -s1 -b2 -e3` is 1-based. The preset sits in the index header:
    .tbi int32 at byte 8, .csi int32 at byte 16 of the decompressed index, flag 0x10000 =
    zero-based. (tabix 1.9 was used only to build those indexes.)
  Loud, so not covered: space-separated lines, header/track lines and a bare contig name fail with
    "Could not parse"; "#" lines are skipped. Content that settles the base: "chr1 100 100"
    (start == end) is a valid 1-based single base but an empty BED interval, a 2-column
    "CHROM POS" file is 1-based positions (named .bed it silently selects nothing), start 0 only
    makes sense as BED. 4+ columns (BED6) are read from the first three.
Semantics. gVCF block A [50,200] with INFO/END, a SNV at 120, deletions at 130/149 (REF ACG/ACGT),
  region = BED chr1 99 150 (1-based 100-150):
    view -R / -r / --regions-overlap 1|record|2|variant -> blk snv del del2
    view -T / -t / --targets-overlap 0|pos -> snv del del2; --targets-overlap 1|record|2|variant ->
    blk snv del del2; --regions-overlap 0|pos -> snv del del2. "overlap" and "3" are rejected
    ("Could not parse"). Mode 2 differs from 1 only for the anchor base of an indel.
    The same option semantics hold in filter, annotate, stats, norm, isec, concat -a and
    +fill-tags. csq, convert, roh, gtcheck, cnv and call list the options in --help but were not
    run on spanning records. annotate, concat and merge have no -T/-t; call and mpileup have no
    --targets-overlap, mpileup has no --regions-overlap either.
  Surprise 1, query: on a text VCF (.vcf, .vcf.gz) `query -f '%ID\\n' -R r.bed` returns snv del
    del2 (no blk) under every overlap mode, and so does -r, -T --targets-overlap 1 and -t; the
    block appears once the format reads INFO or FORMAT ('%INFO/END', '%END', '%INFO',
    '[%GT]', '%INFO/DP') or an -i expression names INFO. '%CHROM %POS %ID %REF %ALT %QUAL %FILTER
    %TYPE %POS0', -e 'QUAL<0', -i 'POS>0' and -s do not. A .bcf input is fine.
  Surprise 2, merge: `merge -R|-r` drops the block under every mode, text or BCF input, but keeps
    a deletion whose REF reaches into the region: INFO/END is never read.
"""

from __future__ import annotations

import gzip
import re
import shlex
import struct
import subprocess
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from biowarden.engine.argv import has_option, option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed, suggester
from biowarden.engine.spec import Rule

# ---- shared ------------------------------------------------------------------------------------

OVERLAP_MODES = {"0": 0, "pos": 0, "1": 1, "record": 1, "2": 2, "variant": 2}


def overlap_mode(argv: Sequence[str], long_option: str, default: int) -> int | None:
    """The mode a --regions-overlap/--targets-overlap option sets (last wins); None if invalid."""
    values = option_values(argv, long_option)
    return OVERLAP_MODES.get(values[-1]) if values else default


def _subcommand_ok(call: Call, match: Match) -> bool:
    sub = call.subcommand
    if sub is None:
        return False
    return sub in match.rule.params.get("subcommands", ()) or sub.startswith("+")


def _with_value(argv: Sequence[str], option: str, value: str, insert_at: int) -> list[str]:
    """argv with `option` set to `value`: replaced where present, else inserted at `insert_at`.

    bcftools reads the last of a repeated option, so that is the occurrence that is replaced.
    """
    items = list(argv)
    last = None
    for i, token in enumerate(items):
        if token == "--":
            break
        if token == option and i + 1 < len(items):
            last = (i, False)
        elif token.startswith(option + "="):
            last = (i, True)
    if last is not None:
        i, attached = last
        if attached:
            items[i] = f"{option}={value}"
        else:
            items[i + 1] = value
        return items
    items[insert_at:insert_at] = [option, value]
    return items


def _command(call: Call, argv: Sequence[str], *declare: str) -> str:
    """The command as text, with the declarations the call already made (except those replaced)
    followed by `declare`: pasting it must not lose a declaration another rule asks for."""
    parts = [call.tool, *argv]
    replaced = {item.partition("=")[0] for item in declare}
    for key, value in call.declared.items():
        if key not in replaced:
            parts += ["--declare", f"{key}={value}"]
    for item in declare:
        parts += ["--declare", item]
    return shlex.join(parts)


# ---- rule 1: the base of a regions/targets file ------------------------------------------------

BED_SUFFIXES = (".bed", ".bed.gz")
TABIX_ZERO_BASED = 0x10000
SCAN_LINES = 1000
DRY_RUN_REGIONS = 500
DRY_RUN_MAX_BYTES = 20_000_000
_DRY_RUN_CACHE: dict[tuple, tuple[bool | None, str]] = {}  # keyed on file stats, so edits count
_DRY_RUN_CACHE_SIZE = 512
DRY_RUN_SUBCOMMANDS = frozenset({"view", "query", "filter", "norm", "annotate", "stats"})
VCF_SUFFIXES = (".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")
# options whose next token is a file that is not one of the VCF inputs
NOT_AN_INPUT = frozenset(
    {"-o", "--output", "-a", "--annotations", "-h", "--header-lines", "-S", "--samples-file"}
)
_DIGITS = re.compile(r"[0-9]+")
_LONG_FILE_OPTIONS = {"regions-file": "-R", "targets-file": "-T"}


@dataclass(frozen=True)
class RegionFile:
    """One -R/-T file named in a call, with enough detail to rewrite that token."""

    option: str  # "-R" or "-T"
    index: int  # position in argv of the token that carries the file name
    prefix: str  # text of that token before the name: "", "-R", "--regions-file="
    name: str  # the name bcftools tests, without a leading ^ (exclusion marker of -T)
    exclude: bool


def region_files(argv: Sequence[str]) -> list[RegionFile]:
    """Every -R/-T file in `argv`: `-R f`, `-Rf`, `--regions-file f`, `--regions-file=f`."""
    found: list[RegionFile] = []
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        if token == "--":
            break
        hit: tuple[str, int, str, str] | None = None
        if token.startswith("--"):
            name, eq, value = token[2:].partition("=")
            option = _LONG_FILE_OPTIONS.get(name)
            if option and eq:
                hit = (option, i, f"--{name}=", value)
            elif option and i + 1 < len(items):
                hit = (option, i + 1, "", items[i + 1])
                i += 1
        elif token in ("-R", "-T"):
            if i + 1 < len(items):
                hit = (token, i + 1, "", items[i + 1])
                i += 1
        elif token[:2] in ("-R", "-T") and len(token) > 2:
            hit = (token[:2], i, token[:2], token[2:])
        if hit:
            option, index, prefix, raw = hit
            exclude = option == "-T" and raw.startswith("^")
            found.append(RegionFile(option, index, prefix, raw[1:] if exclude else raw, exclude))
        i += 1
    return found


def _index_state(path: Path) -> str:
    """'none', 'bed', 'one_based' or 'unreadable' for the tabix/CSI index sitting next to `path`."""
    for suffix in (".tbi", ".csi"):
        index = Path(f"{path}{suffix}")
        if not index.is_file():
            continue
        try:
            with gzip.open(index, "rb") as handle:
                head = handle.read(32)
            if head[:4] == b"TBI\x01" and len(head) >= 12:
                preset = struct.unpack_from("<i", head, 8)[0]
            elif (
                head[:4] == b"CSI\x01"
                and len(head) >= 20
                and struct.unpack_from("<i", head, 12)[0] >= 4
            ):
                preset = struct.unpack_from("<i", head, 16)[0]
            elif head[:4] == b"CSI\x01":
                continue  # a CSI without tabix settings (a BCF index): bcftools ignores it here
            else:
                return "unreadable"
        except (OSError, EOFError, zlib.error, struct.error):
            return "unreadable"
        return "bed" if preset & TABIX_ZERO_BASED else "one_based"
    return "none"


def _reading(path: Path, name: str) -> tuple[str, str]:
    """(how bcftools reads the file, why): 'bed', 'one_based' or 'unknown'."""
    state = _index_state(path)
    if state == "unreadable":
        return "unknown", "its tabix/CSI index cannot be read, and the index decides"
    if state == "bed":
        return "bed", "its tabix index has the BED (zero-based) preset"
    if state == "one_based":
        return "one_based", "its tabix index does not have the BED (zero-based) preset"
    if name.lower().endswith(BED_SUFFIXES):
        return "bed", "its name ends in .bed or .bed.gz"
    return "one_based", "its name does not end in .bed or .bed.gz"


@dataclass(frozen=True)
class Scan:
    kind: str  # bed_like | not_bed | empty | unreadable
    rows: tuple[tuple[str, int, int], ...] = ()
    complete: bool = True
    why: str = ""


def _scan(ctx: Ctx, path: Path) -> Scan:
    """Do the data lines look like BED intervals? Reads at most SCAN_LINES lines."""
    if not path.is_file():
        return Scan("unreadable", why="does not exist (yet), or is a pipe or stdin")
    rows: list[tuple[str, int, int]] = []
    seen = 0
    try:
        for line in ctx.lines(path, limit=SCAN_LINES):
            seen += 1
            text = line.rstrip("\r")
            if not text.strip() or text.startswith("#"):
                continue
            cols = text.split("\t")
            if len(cols) < 3 or not (_DIGITS.fullmatch(cols[1]) and _DIGITS.fullmatch(cols[2])):
                return Scan("not_bed", why="has lines that are not CHROM START END")
            start, end = int(cols[1]), int(cols[2])
            if start >= end:
                return Scan("not_bed", why="has a line with start >= end, which BED does not allow")
            rows.append((cols[0], start, end))
    except (OSError, EOFError, UnicodeError, zlib.error) as exc:
        return Scan("unreadable", why=f"could not be read ({type(exc).__name__})")
    if not rows:
        return Scan("empty", why="has no data lines")
    return Scan("bed_like", tuple(rows), complete=seen < SCAN_LINES)


def _bed_name(name: str) -> str:
    base = Path(name).name
    return base[:-3] + ".bed.gz" if base.lower().endswith(".gz") else base + ".bed"


def _single_input(call: Call, ctx: Ctx, rf: RegionFile) -> Path | None:
    """The one VCF/BCF input of the call, or None when it is not clear-cut."""
    items = list(call.argv)
    start = items.index(call.subcommand) + 1 if call.subcommand in items else 0
    found: list[Path] = []
    for i in range(start, len(items)):
        token = items[i]
        if i == rf.index or token.startswith("-") and token != "-":
            continue
        if token == "-":
            return None
        if not token.lower().endswith(VCF_SUFFIXES) or items[i - 1] in NOT_AN_INPUT:
            continue
        path = ctx.resolve(token)
        if not path.is_file() or path.stat().st_size > DRY_RUN_MAX_BYTES:
            return None
        found.append(path)
    return found[0] if len(found) == 1 else None


def _select(ctx: Ctx, vcf: Path, regions: list[str], mode: int) -> list[str] | None:
    argv = ["view", "-H", "-G", "-t", ",".join(regions), "--targets-overlap", str(mode), str(vcf)]
    try:
        proc = ctx.run_real("bcftools", argv)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sorted(proc.stdout.splitlines()) if proc.returncode == 0 else None


def _readings_agree(call: Call, ctx: Ctx, rf: RegionFile, scan: Scan) -> tuple[bool | None, str]:
    """Dry run: do the 1-based and the BED reading of the file select the same records?

    (True, "") same, (False, detail) different, (None, "") when it cannot be told.
    """
    argv = call.argv
    if call.subcommand not in DRY_RUN_SUBCOMMANDS or rf.exclude or len(region_files(argv)) != 1:
        return None, ""
    if has_option(argv, "-r", "--regions", "-t", "--targets"):
        return None, ""
    if not scan.complete or len(scan.rows) > DRY_RUN_REGIONS:
        return None, ""
    if any(":" in chrom or "," in chrom for chrom, _, _ in scan.rows):
        return None, ""
    vcf = _single_input(call, ctx, rf)
    if vcf is None:
        return None, ""
    if rf.option == "-R":
        effective = overlap_mode(argv, "--regions-overlap", 1)
    else:
        effective = overlap_mode(argv, "--targets-overlap", 0)
    if effective is None:
        return None, ""
    modes = (effective, 0) if call.subcommand == "query" and effective else (effective,)
    key = (*_stat_key(ctx.resolve(rf.name)), *_stat_key(vcf), modes, str(ctx.real_bins))
    cached = _DRY_RUN_CACHE.get(key)
    if cached is not None:
        return cached
    outcome = _compare_readings(ctx, vcf, scan, modes)
    if outcome[0] is not None:  # a failed run is not remembered
        if len(_DRY_RUN_CACHE) >= _DRY_RUN_CACHE_SIZE:
            _DRY_RUN_CACHE.clear()
        _DRY_RUN_CACHE[key] = outcome
    return outcome


def _stat_key(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return str(path), info.st_mtime_ns, info.st_size


def _compare_readings(
    ctx: Ctx, vcf: Path, scan: Scan, modes: tuple[int, ...]
) -> tuple[bool | None, str]:
    as_written = [f"{c}:{s}-{e}" for c, s, e in scan.rows]
    as_bed = [f"{c}:{s + 1}-{e}" for c, s, e in scan.rows]
    for mode in modes:
        one, bed = _select(ctx, vcf, as_written, mode), _select(ctx, vcf, as_bed, mode)
        if one is None or bed is None:
            return None, ""
        if one != bed:
            detail = f"read as 1-based it selects {len(one)} records of {vcf.name}, "
            return False, detail + f"read as BED {len(bed)}"
    return True, ""


def _corrected(call: Call, rf: RegionFile) -> tuple[str, ...]:
    if rf.name == "-":
        return (
            f"write the intervals to a file named regions.bed first and pass it as {rf.option} "
            "regions.bed (bcftools reads '-' as 1-based inclusive)",
            "if they already are 1-based inclusive, re-run with "
            "--declare region_file_base=one_based_closed",
        )
    new = _bed_name(rf.name)
    items = list(call.argv)
    items[rf.index] = rf.prefix + ("^" if rf.exclude else "") + new
    return (
        f"cp {shlex.quote(rf.name)} {shlex.quote(new)} && {_command(call, items)}",
        f"(if {rf.name} already holds 1-based inclusive coordinates, re-run the original command "
        "with --declare region_file_base=one_based_closed instead)",
    )


def _call_sites_file(call: Call, ctx: Ctx, rf: RegionFile) -> CheckResult:
    """`bcftools call -T` takes a sites file (CHROM POS [REF,ALT]); END is ignored."""
    scan = _scan(ctx, ctx.resolve(rf.name))
    if scan.kind != "bed_like":
        return passed(f"{rf.name}: bcftools call -T reads CHROM POS, which is how it is written")
    items = _retarget(call.argv)
    new = rf.name if rf.name.lower().endswith(BED_SUFFIXES) else _bed_name(rf.name)
    items[rf.index] = _retarget([rf.prefix])[0] + ("^" if rf.exclude else "") + new
    copy = "" if new == rf.name else f"cp {shlex.quote(rf.name)} {shlex.quote(new)} && "
    return CheckResult(
        "unknown",
        "call_targets_are_sites",
        f"bcftools call -T reads only CHROM and POS of each line of {rf.name}, whatever its name, "
        "so every BED interval selects just one position, its start value taken as 1-based "
        "(checked on bcftools 1.24)",
        fix=(f"{copy}{_command(call, items)}", "(-R needs an indexed input; -T would stream)"),
    )


def _assess(call: Call, ctx: Ctx, rf: RegionFile, declared: str | None) -> CheckResult:
    if call.subcommand == "call" and rf.option == "-T":
        return _call_sites_file(call, ctx, rf)
    path = ctx.resolve(rf.name)
    reading, why = _reading(path, rf.name)
    if reading == "bed":
        return passed(f"{rf.name}: {why}, which is how BED content is read")
    if reading == "unknown":
        return CheckResult(
            "unknown",
            "index_unreadable",
            f"{rf.name}: {why}",
            fix=_corrected(call, rf)[:1] if rf.name != "-" else (),
        )
    if declared == "one_based_closed":
        return passed(f"{rf.name}: declared one_based_closed, {why}")
    scan = _scan(ctx, path)
    if declared != "zero_based_half_open":
        if scan.kind == "unreadable":
            return CheckResult(
                "unknown",
                "regions_file_unreadable",
                f"{rf.name} {scan.why}, so I cannot check whether it holds BED intervals; bcftools "
                f"reads it as 1-based inclusive because {why}",
                expected="zero_based_half_open",
                got="one_based_closed",
                fix=_corrected(call, rf)[:1],
            )
        if scan.kind != "bed_like":
            return passed(
                f"{rf.name}: not BED-like ({scan.why}), so a 1-based reading is consistent"
            )
    detail = ""
    if scan.kind == "bed_like":
        agree, detail = _readings_agree(call, ctx, rf, scan)
        if agree:
            return passed(f"{rf.name}: both readings select the same records in this dry run")
    count = f"{len(scan.rows)}{'' if scan.complete else '+'} data lines"
    if declared == "zero_based_half_open":
        message = (
            f"you declared region_file_base=zero_based_half_open, but bcftools reads {rf.name} as "
            f"1-based inclusive because {why}, so each interval would start one base early"
        )
        code = "declared_mismatch"
    else:
        message = (
            f"bcftools reads {rf.name} as 1-based inclusive CHROM POS END because {why}, but its "
            f"{count} look like BED intervals (0-based, half-open), so each interval would start "
            "one base early"
        )
        code = "bed_content_read_one_based"
    return CheckResult(
        "fail",
        code,
        message + (f"; {detail}" if detail else ""),
        expected="zero_based_half_open",
        got="one_based_closed",
        fix=_corrected(call, rf),
    )


@checker("bcftools_regions_file_base")
def regions_file_base(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if not _subcommand_ok(call, match):
        return passed("this subcommand does not read a regions file with -R/-T")
    files = region_files(call.argv)
    if not files:
        return passed("no -R/-T file in the call")
    declared = call.declared.get(rule.declare.key)
    unknown: CheckResult | None = None
    notes: list[str] = []
    for rf in files:
        result = _assess(call, ctx, rf, declared)
        if result.verdict == "fail":
            return result
        if result.verdict == "unknown" and unknown is None:
            unknown = result
        if result.note:
            notes.append(result.note)
    return unknown or passed("; ".join(notes))


# ---- rule 2: overlap or start-only selection ---------------------------------------------------

START, OVERLAP = "start_only", "overlap"
SEMANTIC_OF_MODE = {
    "0": START,
    "pos": START,
    "1": OVERLAP,
    "record": OVERLAP,
    "2": OVERLAP,
    "variant": OVERLAP,
}
# (kind, options that give the regions/targets, the overlap option, its default mode)
SELECTIONS = (
    ("regions", ("-r", "--regions", "-R", "--regions-file"), "--regions-overlap", "1"),
    ("targets", ("-t", "--targets", "-T", "--targets-file"), "--targets-overlap", "0"),
)
RETARGET = {"-T": "-R", "-t": "-r", "--targets-file": "--regions-file", "--targets": "--regions"}
NO_TARGETS_OVERLAP = frozenset({"call"})
# what a selection keeps, as the fix of a missing declaration words it
KEEPS = {
    OVERLAP: "every record that overlaps a region",
    START: "only records whose POS lies in a region",
}
# query tokens that need nothing beyond the fixed VCF columns (verified: no INFO/END is parsed)
QUERY_FIXED_COLUMNS = frozenset(
    {"CHROM", "POS", "POS0", "ID", "REF", "ALT", "QUAL", "FILTER", "TYPE"}
)


@dataclass(frozen=True)
class Selection:
    kind: str  # regions | targets
    label: str  # as written: "-T regions.bed"
    overlap_option: str
    mode: str  # as written, or the default
    explicit: bool
    semantic: str

    def describe(self) -> str:
        if self.explicit:
            return f"{self.label} with {self.overlap_option} {self.mode}"
        return f"{self.label} ({self.overlap_option} defaults to {self.mode})"


def _label(argv: Sequence[str], names: Sequence[str]) -> str:
    for name in names:
        values = option_values(argv, name)
        if values:
            return f"{name} {values[0]}"
    return next(n for n in names if has_option(argv, n))


def selections(argv: Sequence[str]) -> tuple[list[Selection], list[str]]:
    """(what the regions/targets options of the call keep, overlap values bcftools would reject)."""
    found: list[Selection] = []
    invalid: list[str] = []
    for kind, names, overlap_option, default in SELECTIONS:
        if not has_option(argv, *names):
            continue
        values = option_values(argv, overlap_option)
        mode = values[-1] if values else default
        semantic = SEMANTIC_OF_MODE.get(mode)
        if semantic is None:
            invalid.append(f"{overlap_option} {mode}")
            continue
        found.append(
            Selection(kind, _label(argv, names), overlap_option, mode, bool(values), semantic)
        )
    return found, invalid


def _retarget(argv: Sequence[str]) -> list[str]:
    """-T -> -R, -t -> -r (and the long forms), for subcommands without --targets-overlap."""
    items: list[str] = []
    for token in argv:
        for old, new in RETARGET.items():
            if (
                token == old
                or token.startswith(old + "=")
                or (len(old) == 2 and token.startswith(old))
            ):
                token = new + token[len(old) :]
                break
        items.append(token)
    return items


def _fix_selection(call: Call, chosen: list[Selection], wanted: str) -> str:
    """The call with each selection in `chosen` set to `wanted` ('0' start only, '1' overlap)."""
    items = list(call.argv)
    at = items.index(call.subcommand) + 1 if call.subcommand in items else 0
    if call.subcommand == "plugin" and at < len(items) and not items[at].startswith("-"):
        at += 1  # bcftools plugin NAME [options] FILE: the name comes before the options
    for sel in chosen:
        if sel.kind == "targets" and wanted == "1" and call.subcommand in NO_TARGETS_OVERLAP:
            items = _retarget(items)
        else:
            items = _with_value(items, sel.overlap_option, wanted, at)
    mapped = {"1": OVERLAP, "0": START}[wanted]
    return _command(call, items, f"region_semantics={mapped}")


def _text_vcf_inputs(argv: Sequence[str]) -> bool:
    names = [t.lower() for t in argv if not t.startswith("-")]
    text = any(n.endswith((".vcf", ".vcf.gz", ".vcf.bgz")) for n in names)
    return text and not any(n.endswith(".bcf") for n in names)


def _query_misses_end(argv: Sequence[str]) -> bool:
    """True when `bcftools query` reads a text VCF without parsing INFO/END (verified, 1.24)."""
    formats = option_values(argv, "-f", "--format")
    if not formats or has_option(argv, "-i", "--include", "-e", "--exclude"):
        return False
    fmt = formats[-1]
    if "[" in fmt or not _text_vcf_inputs(argv):
        return False
    return all(
        token in QUERY_FIXED_COLUMNS for token in re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)", fmt)
    )


def _with_end_column(call: Call) -> str:
    formats = option_values(call.argv, "-f", "--format")
    old = formats[-1]
    tail = "\\n" if old.endswith("\\n") else "\n" if old.endswith("\n") else ""
    new = old[: len(old) - len(tail)] + "\\t%END" + tail
    items = list(call.argv)
    for i, token in enumerate(items):
        if token == old or (token.endswith(old) and token[: -len(old)] in ("-f", "--format=")):
            items[i] = token[: len(token) - len(old)] + new
            break
    return _command(call, items, "region_semantics=overlap")


def _end_caveat(call: Call) -> CheckResult | None:
    sub = call.subcommand
    if sub == "merge":
        return CheckResult(
            "unknown",
            "merge_ignores_end",
            "bcftools merge applies -R/-r to POS and REF length but never reads INFO/END, so a "
            "gVCF reference block that starts before the region is dropped although "
            "--regions-overlap defaults to 1 (checked on bcftools 1.24)",
            fix=(
                "filter each input first with bcftools view -R ... (it reads INFO/END), then merge "
                "the filtered files without -R",
            ),
        )
    if sub == "query" and _query_misses_end(call.argv):
        return CheckResult(
            "unknown",
            "query_ignores_end",
            "bcftools query parses INFO/END only when the format (or an -i/-e expression) reads "
            "INFO or FORMAT fields; with this format the region options select a text VCF by POS "
            "only, so a reference block that starts before the region is missed (bcftools 1.24)",
            fix=(
                _with_end_column(call),
                "(or filter first: bcftools view -R ... -Ou file.vcf.gz | bcftools query ...)",
            ),
        )
    return None


@checker("bcftools_region_semantics")
def region_semantics(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    """T0: the declaration must agree with what the call's own options select."""
    rule = match.rule
    assert rule.declare is not None
    declared = call.declared.get(rule.declare.key)
    if declared is None:  # the engine asks first; a direct caller gets the same answer
        return CheckResult(
            "fail",
            "declaration_missing",
            "state which interval_semantics convention you intend",
            fix=("re-run with --declare region_semantics=<overlap|start_only>",),
        )
    chosen, invalid = selections(call.argv)
    if invalid:
        return CheckResult(
            "unknown",
            "overlap_value",
            f"bcftools does not accept {invalid[0]} (use 0|pos, 1|record, 2|variant)",
        )
    if not chosen:
        return passed("no regions/targets option in the call")
    start = [s for s in chosen if s.semantic == START]
    overlap = [s for s in chosen if s.semantic == OVERLAP]
    if declared == OVERLAP and start:
        first = start[0]
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared region_semantics=overlap, but {first.describe()} keeps only records "
            "whose START (POS) is inside a region, so a record that begins before the region and "
            "runs into it (a gVCF reference block with INFO/END, a long deletion) is silently "
            "dropped",
            expected=OVERLAP,
            got=START,
            fix=(_fix_selection(call, start, "1"),),
        )
    if declared == START and not start:
        first = overlap[0]
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared region_semantics=start_only, but {first.describe()} keeps every record "
            "that overlaps a region, not only the records that start inside one",
            expected=START,
            got=OVERLAP,
            fix=(_fix_selection(call, overlap, "0"),),
        )
    if declared == OVERLAP:
        caveat = _end_caveat(call)
        if caveat is not None:
            return caveat
    return passed()


@suggester("bcftools_region_semantics")
def region_semantics_missing(call: Call, rule: Rule) -> tuple[str, ...]:
    """Two labelled, complete commands for THIS call, overlap first, one per declaration.

    The options of the call decide which of them is the call as written: -R/-r keep the records
    that overlap a region unless --regions-overlap says 0|pos, -T/-t keep the records that start
    in one unless --targets-overlap says 1|record|2|variant. The other command sets the overlap
    option of every selection that differs. A model that sets only the overlap option is blocked
    again, because the declaration is what this rule asks for: the text says so.
    """
    chosen, _ = selections(call.argv)
    if not chosen:
        return ("re-run with --declare region_semantics=<overlap|start_only>",)
    start = [s for s in chosen if s.semantic == START]
    overlap = [s for s in chosen if s.semantic == OVERLAP]
    now = "; ".join(f"{s.describe()} keeps {KEEPS[s.semantic]}" for s in chosen)
    as_written = ", your call as written"
    return (
        f"{now}. Replace the bcftools call with one of these two, exactly as written:",
        f"{_fix_selection(call, start, '1')} (overlap{as_written if not start else ''}: a gVCF "
        "reference block or a deletion that begins before the region is kept)",
        f"or {_fix_selection(call, overlap, '0')} (start_only{as_written if not overlap else ''}: "
        "a record that begins before the region is dropped).",
        "--declare is not a bcftools option: the sandbox checker reads it and removes it before "
        "bcftools runs. Setting --regions-overlap or --targets-overlap alone does not lift this "
        "block.",
    )
