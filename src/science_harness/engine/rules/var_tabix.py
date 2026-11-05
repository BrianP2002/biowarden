"""tabix rules of the variant pack: BED coordinates through tabix (and bgzip).

All three rest on one fact: tabix reads text coordinates as 1-based closed unless it is told that
the file is zero-based, and BED is zero-based and half-open.

Indexing by columns (not shipped as a rule). `tabix -s 1 -b 2 -e 3 FILE` without `-0` indexes a
BED file as 1-based closed, the GFF default: every interval starts one base early for every
later query. Verified here (tabix/bgzip 1.24), BED rows r1 chr1 50 100, r2 99 120, r3 100 150, r4
199 200, r5 200 260: with `-p bed` the query chr1:101-200 returns r2 r3 r4 and chr1:100-100
returns r1 r2; with `-s1 -b2 -e3` (no -0) chr1:101-200 returns r2 r3 r4 r5 (r5 = BED 200-260
starts at 1-based 201) and chr1:100-100 returns r1 r2 r3; with `-0 -s1 -b2 -e3` it matches
-p bed. A `.tsv.gz` file with no option stops ("Failed to parse TBX_GENERIC"); `.bed.gz` is
detected. A rule on the indexing call alone was written and dropped: whether the shift matters
depends on the later queries, so on data without a boundary at a query edge it blocked calls
whose result was right (admission: right calls failed); F61 keeps this route as a trap.

tabix.regions_file_base (T1, call, declaration optional `region_file_base`)
    `tabix FILE -R REGIONS` (and -T) reads REGIONS as BED (0-based, half-open) only when its name
    ends in .bed or .bed.gz; any other name is read as 1-based `chrom from to`.
    Verified here: regions chr1 100 200 against the BED index above: `-R rr.bed` -> r2 r3 r4,
    `-R rr.tsv` (same bytes) -> r1 r2 r3 r4 (r1 = BED 50-100 ends at 1-based 100). The rule is
    the tabix twin of bcftools.regions_file_base. It is silent unless the file is not named .bed
    and its lines look like BED intervals (three or more tab columns, integer start < end, no
    line with start == end). It then runs tabix twice read-only (as written, and with a copy of
    the file named .bed in a temporary directory) and compares what they print: identical output
    passes (the shift changes nothing on this data), different output fails without a
    declaration, passes with `one_based_closed` and fails with `zero_based_half_open`. If the
    dry run cannot be done the verdict rests on the file alone.

tabix.region_strings_bed (T1, call, declaration optional `region_coordinates`)
    A region string `chr:start-end` is always 1-based and closed, also for a zero-based index
    (htslib issue 407). For a BED window [100, 200) the string is chr1:101-200.
    Verified here: with `-p bed`, chr1:100-200 returns r1 r2 r3 r4 (r1 = BED 50-100, which does
    not overlap [100, 200)) and chr1:101-200 returns r2 r3 r4.
    The rule fires when the file's index (.tbi, format flag 0x10000) is zero-based and the call has
    region strings, and a dry run (the call as written against the call with every start moved up
    by one) prints different records: undeclared it fails; `one_based_closed` (the numbers are
    already 1-based) passes; `zero_based_half_open` (the numbers are BED coordinates) fails.
Not seen: .csi indexes, regions given by -R with a BED-named file (those are right), GFF indexes.
"""

from __future__ import annotations

import gzip
import re
import struct
import tempfile
import zlib
from pathlib import Path

from science_harness.engine.argv import has_option, option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.var_common import fail, is_interval_line, unknown

BED_NAMES = (".bed", ".bed.gz", ".bed.bgz")
TBI_ZERO_BASED = 0x10000
REGION_STRING = re.compile(r"^[\w.|-]+:\d+(?:-\d*)?$")
TABIX_VALUE_OPTIONS = ("-s", "-b", "-e", "-m", "-c", "-p", "-S", "-r", "-R", "-T", "-@", "--cache")


def positional_arguments(argv: tuple[str, ...]) -> list[str]:
    """Non-option arguments of a tabix call: the file first, then regions."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in TABIX_VALUE_OPTIONS or token in (
            "--begin",
            "--end",
            "--sequence",
            "--preset",
            "--regions",
            "--targets",
            "--skip-lines",
            "--comment",
            "--min-shift",
            "--reheader",
            "--threads",
            "--verbosity",
        ):
            skip = True
            continue
        if token.startswith("-") and token != "-":
            continue
        out.append(token)
    return out


def tabix_file(call: Call) -> str | None:
    args = positional_arguments(call.argv)
    return args[0] if args else None


def tbi_zero_based(ctx: Ctx, name: str) -> bool | None:
    """Whether the .tbi next to `name` was built for a zero-based file (None: no readable index)."""
    path = ctx.resolve(name + ".tbi")
    if not path.is_file():
        return None
    try:
        data = gzip.decompress(ctx.read_bytes(path, 1_000_000))
        magic, _n_ref, fmt = struct.unpack("<4sii", data[:12])
    except (OSError, EOFError, zlib.error, struct.error):
        return None
    return bool(fmt & TBI_ZERO_BASED) if magic == b"TBI\x01" else None


# ---- tabix.regions_file_base ---------------------------------------------------------------------


def looks_like_bed(ctx: Ctx, path: Path) -> bool | None:
    rows = 0
    try:
        lines = ctx.lines(path, limit=1000)
        try:
            for line in lines:
                if not line.strip() or line.startswith("#"):
                    continue
                f = line.split("\t")
                if len(f) < 3 or not is_interval_line(line) or f[1] == f[2]:
                    return False
                rows += 1
        finally:
            lines.close()  # type: ignore[attr-defined]
    except (OSError, EOFError, zlib.error):
        return None
    return rows > 0


def swap(token: str, old: str, new: str) -> str:
    """`old` replaced by `new` when the token is the value itself or `--option=value`."""
    if token == old:
        return new
    return (
        f"{token[: -len(old)]}{new}"
        if token.startswith("--") and token.endswith("=" + old)
        else token
    )


def dry_run(ctx: Ctx, argv: list[str]) -> list[str] | None:
    """stdout lines of the real tabix on `argv` (None when it fails or is not available)."""
    try:
        proc = ctx.run_real("tabix", argv)
    except (OSError, FileNotFoundError):
        return None
    return proc.stdout.splitlines() if proc.returncode == 0 else None


def readings_differ_for_file(call: Call, ctx: Ctx, names: list[str]) -> bool | None:
    """Do the regions files, read as written and as BED, select different records?"""
    argv = list(call.argv)
    written = dry_run(ctx, argv)
    if written is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(names):
            if name.lower().endswith((".bed", ".bed.gz")):
                continue
            copy = Path(tmp) / f"r{i}.bed"
            try:
                copy.write_text("\n".join(ctx.lines(name, limit=100_000)) + "\n")
            except (OSError, EOFError, zlib.error):
                return None
            argv = [swap(token, name, str(copy)) for token in argv]
        as_bed = dry_run(ctx, argv)
    return None if as_bed is None else sorted(written) != sorted(as_bed)


@checker("tabix_regions_file_base")
def tabix_regions_file_base(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    names = option_values(call.argv, "-R", "--regions", "-T", "--targets")
    if not names:
        return passed()
    declared = call.declared.get(rule.declare.key)
    suspects: list[str] = []
    for name in names:
        like_bed = looks_like_bed(ctx, ctx.resolve(name))
        if like_bed is None:
            return unknown(
                "regions_file_unreadable",
                f"{name} cannot be read, so its coordinate base cannot be checked "
                "(tabix reads it as BED only when the name ends in .bed or .bed.gz).",
                fix=(f"--declare {rule.declare.key}=zero_based_half_open|one_based_closed",),
            )
        bed_name = name.lower().endswith((".bed", ".bed.gz"))
        if like_bed and not bed_name:
            suspects.append(name)
        elif bed_name and like_bed and declared == "one_based_closed":
            return fail(
                "declared_one_based_but_bed_name",
                f"you declared {rule.declare.key}=one_based_closed but tabix reads {name} as "
                "BED (0-based, half-open) because of its name.",
                fix=(f"copy it to a name that does not end in .bed: cp {name} {name}.tsv",),
                expected="one_based_closed",
                got="zero_based_half_open",
            )
    if not suspects or declared == "one_based_closed":
        return passed()
    differ = readings_differ_for_file(call, ctx, names)
    if differ is False:
        return passed(note="reading the file as BED selects the same records here")
    name = suspects[0]
    shown = f"{name}.bed"
    effect = (
        "; run read-only, the two readings print different records"
        if differ
        else " (the two readings could not be compared)"
    )
    return fail(
        "bed_file_read_one_based",
        f"tabix reads {name} as 1-based closed 'chrom from to' because its name does not end in "
        f"'.bed', but its lines look like BED intervals (0-based, half-open): every region "
        f"starts one base early{effect}.",
        fix=(
            f"cp {name} {shown} && use -R {shown}",
            f"or --declare {rule.declare.key}=one_based_closed if the file is 1-based",
        ),
        expected=["zero_based_half_open", "one_based_closed"],
        got=declared,
    )


# ---- tabix.region_strings_bed --------------------------------------------------------------------


def region_strings_differ(call: Call, ctx: Ctx, regions: list[str]) -> bool | None:
    written = dry_run(ctx, list(call.argv))
    if written is None:
        return None
    moved = {}
    for region in regions:
        chrom, _, span = region.rpartition(":")
        start, _, end = span.partition("-")
        moved[region] = f"{chrom}:{int(start) + 1}" + (f"-{end}" if end else "")
    argv = [moved.get(token, token) for token in call.argv]
    other = dry_run(ctx, argv)
    return None if other is None else sorted(written) != sorted(other)


@checker("tabix_region_strings_bed")
def tabix_region_strings_bed(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    args = positional_arguments(call.argv)
    if len(args) < 2 or has_option(call.argv, "-p", "--preset", "-s", "-b", "-e"):
        return passed()
    regions = [a for a in args[1:] if REGION_STRING.match(a)]
    if not regions or tbi_zero_based(ctx, args[0]) is not True:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "one_based_closed":
        return passed()
    differ = region_strings_differ(call, ctx, regions)
    if differ is False:
        return passed(note="moving the starts up by one selects the same records here")
    start = int(re.split(r"[:-]", regions[0])[1])
    chrom_part, _, span = regions[0].rpartition(":")
    end_part = span.partition("-")[2]
    moved = f"{chrom_part}:{start + 1}" + (f"-{end_part}" if end_part else "")
    shown = f"{args[0]} {regions[0]}"
    effect = (
        "; run read-only, chr:S+1-E prints different records"
        if differ
        else " (the two forms could not be compared)"
    )
    if declared == "zero_based_half_open":
        return fail(
            "declared_zero_based_region_string",
            f"you declared {rule.declare.key}=zero_based_half_open but a tabix region string is "
            f"always 1-based and closed: '{regions[0]}' starts at base {start}, so it also returns "
            f"the BED interval that ends at {start} and touches the window{effect}.",
            fix=(f"write the BED window [S, E) as chr:S+1-E: {moved} instead of {regions[0]}",),
            expected="one_based_closed",
            got="zero_based_half_open",
        )
    return fail(
        "bed_region_string_base_undeclared",
        f"{args[0]} is indexed zero-based (BED) but the region string in '{shown}' is read "
        f"1-based and closed: a BED window [S, E) is chr:S+1-E, and chr:S-E also returns the "
        f"interval that ends at S{effect}.",
        fix=(
            f"if '{regions[0]}' is a BED window, query {moved} ({args[0]} is zero-based)",
            f"or --declare {rule.declare.key}=one_based_closed if the numbers are already 1-based",
        ),
        expected=["one_based_closed", "zero_based_half_open"],
    )
