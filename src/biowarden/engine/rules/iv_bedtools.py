"""Rules for the bedtools subcommands of tool pack `iv` (genomic intervals, tracks, coverage).

Every claim below was run on bedtools 2.31.1 (and samtools 1.24 to build the BAMs). The first
lines of each rule section quote the evidence; the engine host has only the standard library, so
the checkers read BED/bedGraph text and BAM through helpers, and dry-run the real bedtools through
`ctx.run_real` where the tool's own arithmetic is the oracle.

bedtools.split_reciprocal   (slot tool_logic, T1 call rule, GEN-40)
    `intersect -split` with a fraction option (`-r -f`, `-F`, `-f -F`) drops a pair that passes the
    test alone when another b record lies inside the span of the same a record, even inside an
    intron (exit 0, stderr empty). Evidence (BED12 transcript A1, exons 1000-1200, 1900-2100,
    2800-3000, and its exact copy P in b): `-split -r -f 0.9 -wa -wb` reports (A1, P); adding D =
    1400-1700 (an intron) removes the pair for `-r -f 0.9`, `-F 0.9`, `-f 0.9 -F 0.9`; D = 3500-3700
    (500 bases past the end) changes nothing; `-split -f 0.9` alone and `-split -f 0.5 -r` keep the
    pair; a D of 20 or 60 bases (under about 10 % of the exon length) does not trigger it, D of
    150 bases and more does. The checker runs the call and one call per b record (in a temporary
    directory) on the files it names and fails when the pair sets differ.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules import quant_common as qc
from biowarden.engine.rules.seq_common import small_file

MAX_B = 150
MAX_LINES = 20_000
INTERSECT_VALUES = frozenset({"-a", "-abam", "-b", "-f", "-F", "-g", "-iobuf", "-names"})
HEADER_PREFIXES = ("#", "track", "browser")


def bt_args(
    argv: list[str] | tuple[str, ...], values: frozenset[str]
) -> tuple[dict[str, list[str]], list[str], set[str]]:
    """(option -> values, positionals, flags) for bedtools: every option is `-word`, and only the
    names in `values` take the next token (`-bed` is not `-b ed`)."""
    opts: dict[str, list[str]] = {}
    positionals: list[str] = []
    flags: set[str] = set()
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        i += 1
        if token in values:
            if i < len(items):
                opts.setdefault(token, []).append(items[i])
                i += 1
        elif token.startswith("-") and len(token) > 1:
            flags.add(token)
        else:
            positionals.append(token)
    return opts, positionals, flags


def data_lines(ctx: Ctx, name: str | Path, limit: int = MAX_LINES) -> list[str]:
    """Non-empty, non-header lines of a text interval file (plain or gzip)."""
    return [
        line
        for line in ctx.lines(name, limit=limit)
        if line.strip() and not line.startswith(HEADER_PREFIXES)
    ]


def rows(ctx: Ctx, name: str | Path, limit: int = MAX_LINES) -> list[list[str]]:
    return [line.split("\t") for line in data_lines(ctx, name, limit)]


def rerunnable(ctx: Ctx, *names: str) -> bool:
    """The inputs are small enough to run bedtools on the whole files again (E-7)."""
    return all(small_file(ctx.resolve(name)) for name in names)


def run_lines(ctx: Ctx, argv: list[str]) -> list[str] | None:
    """Stdout lines of a dry run of bedtools, or None when it failed or ran out of time."""
    try:
        proc = ctx.run_real("bedtools", argv)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line.strip()]


# ---- bedtools.split_reciprocal ---------------------------------------------------------------


def _split_options(opts: dict[str, list[str]], flags: set[str]) -> list[str]:
    out: list[str] = []
    for name in ("-f", "-F"):
        value = qc.last_value(opts, name)
        if value is not None:
            out += [name, value]
    out += [flag for flag in ("-r", "-e", "-s", "-S") if flag in flags]
    return out


@checker("iv_bedtools_split_reciprocal")
def split_reciprocal(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    opts, positionals, flags = bt_args(call.argv[1:], INTERSECT_VALUES)
    if "-split" not in flags:
        return passed()
    if not ({"-r", "-e"} & flags or {"-f", "-F"} & set(opts)):
        return passed()
    a_name, b_name = qc.last_value(opts, "-a"), qc.last_value(opts, "-b")
    if a_name is None or b_name is None or positionals:
        return passed()
    if not ctx.resolve(a_name).is_file() or not ctx.resolve(b_name).is_file():
        return passed()
    if not rerunnable(ctx, a_name, b_name):
        return passed("an input is too large to run bedtools on it again")
    a_rows = rows(ctx, a_name, 1)
    if not a_rows:
        return passed()
    a_width = len(a_rows[0])
    base = ["intersect", "-split", *_split_options(opts, flags), "-wa", "-wb"]
    plain = run_lines(ctx, ["intersect", "-a", a_name, "-b", b_name, "-wa", "-wb"])
    whole = run_lines(ctx, [*base, "-a", a_name, "-b", b_name])
    if plain is None or whole is None:
        return CheckResult("unknown", "dry_run_failed", "the dry run of bedtools failed")
    # the loss needs two b records on the SPAN of one a record (an intron counts: without -split
    # the span is what intersect sees); without such an a record nothing can differ
    per_a: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    for line in plain:
        f = tuple(line.split("\t"))
        per_a.setdefault(f[:a_width], set()).add(f[a_width:])
    crowded = {b for bs in per_a.values() if len(bs) >= 2 for b in bs}
    if not crowded:
        return passed()
    if len(crowded) > MAX_B:
        return CheckResult("unknown", "too_many_b", f"{len(crowded)} b records to test one by one")
    union: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for i, b_fields in enumerate(sorted(crowded)):
            one = Path(tmp) / f"b{i}.bed"
            one.write_text("\t".join(b_fields) + "\n")
            got = run_lines(ctx, [*base, "-a", a_name, "-b", str(one)])
            if got is None:
                return CheckResult("unknown", "dry_run_failed", "the dry run of bedtools failed")
            union.update(got)
    mine = {line for line in whole if tuple(line.split("\t")[a_width:]) in crowded}
    lost, extra = union - mine, mine - union
    if not lost and not extra:
        return passed()
    example = sorted(lost or extra)[0].split("\t")
    pair = f"{example[3]} / {example[a_width + 3]}" if len(example) > a_width + 3 else "?"
    what = (
        f"{len(lost)} pair(s) that pass the overlap test on their own are missing"
        if lost
        else f"{len(extra)} pair(s) that fail the overlap test on their own are reported"
    )
    options = " ".join(_split_options(opts, flags))
    return CheckResult(
        "fail",
        "split_fraction_depends_on_other_records",
        f"bedtools intersect -split with a fraction option gives pairs that depend on the other "
        f"records of {b_name}: {what} (for example {pair}), because a second {b_name} record lies "
        f"inside the span of the same {a_name} record.",
        expected=len(union),
        got=len(mine),
        fix=(
            f"run intersect once per record of {b_name} (for each line: bedtools intersect -a "
            f"{a_name} -b <that one line> -split {options} -wa -wb) and merge the results, or "
            "compute the block overlap yourself (awk/python)",
        ),
    )


# ---- bedtools.map_unweighted -----------------------------------------------------------------
# slot weighting, T1 call rule, optional declaration `signal_weighting`.
# `bedtools map -c 4 -o mean` averages the values of the b LINES that touch each a window; every
# line counts once, whatever its length and however little of it lies inside the window. For a
# bedGraph (one line per run of equal depth) the per-base mean is the length-weighted mean of the
# clipped lines. Evidence (windows of 100 bases, lines 100-110:10, 110-290:2, 290-400:6): map
# gives 6, 4, 6; the per-base means are 2.8, 2.4, 6.0; exit 0, stderr empty. The checker computes
# both for the windows and the track named in the call and fails when they differ by more than
# 0.5 %. Only `mean` is judged (a `sum` of bedGraph lines is meaningful for count tracks).

MAP_VALUES = frozenset({"-a", "-b", "-c", "-o", "-delim", "-prec", "-f", "-F", "-g", "-null"})
BEDGRAPH_NAME = (".bedgraph", ".bedgraph.gz", ".bdg", ".bdg.gz", ".bg", ".bg.gz")


def looks_like_bedgraph(ctx: Ctx, name: str) -> list[list[str]] | None:
    """The rows of a bedGraph (4 columns, numeric 4th), or None when the file is not one."""
    path = ctx.resolve(name)
    if not path.is_file():
        return None
    head = list(ctx.lines(path, limit=3))
    named = name.lower().endswith(BEDGRAPH_NAME)
    tracked = any(line.startswith("track") and "bedGraph" in line for line in head)
    if not (named or tracked):
        return None
    table = rows(ctx, path)
    try:
        ok = bool(table) and all(len(r) == 4 and float(r[3]) is not None for r in table[:200])
    except ValueError:
        ok = False
    return table if ok else None


def window_means(
    windows: list[list[str]], track: list[list[str]]
) -> list[tuple[str, float, float]]:
    """(name, mean of the line values, per-base weighted mean) per window with an overlap."""
    import bisect

    by_chrom: dict[str, list[tuple[int, int, float]]] = {}
    for r in track:
        by_chrom.setdefault(r[0], []).append((int(r[1]), int(r[2]), float(r[3])))
    starts: dict[str, list[int]] = {}
    for chrom, items in by_chrom.items():
        items.sort()
        starts[chrom] = [s for s, _, _ in items]
    out = []
    for w in windows:
        chrom, ws, we = w[0], int(w[1]), int(w[2])
        items = by_chrom.get(chrom, [])
        hits = []
        i = bisect.bisect_left(starts.get(chrom, []), we) - 1
        while i >= 0:
            s, e, v = items[i]
            if e > ws and s < we:
                hits.append((s, e, v))
            i -= 1
            if len(hits) > 5000:
                break
        if not hits:
            continue
        mean_lines = sum(v for _, _, v in hits) / len(hits)
        covered = sum(min(we, e) - max(ws, s) for s, e, _ in hits)
        weighted = sum(v * (min(we, e) - max(ws, s)) for s, e, v in hits) / covered
        out.append((w[3] if len(w) > 3 else f"{chrom}:{ws}-{we}", mean_lines, weighted))
    return out


@checker("iv_bedtools_map_unweighted")
def map_unweighted(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, _, _ = bt_args(call.argv[1:], MAP_VALUES)
    a_name, b_name = qc.last_value(opts, "-a"), qc.last_value(opts, "-b")
    cols = (qc.last_value(opts, "-c") or "").split(",")
    ops = (qc.last_value(opts, "-o") or "").split(",")
    if a_name is None or b_name is None or "mean" not in ops:
        return passed()
    index = ops.index("mean")
    if index >= len(cols) or cols[index] != "4":
        return passed()
    if not ctx.resolve(a_name).is_file():
        return passed()
    track = looks_like_bedgraph(ctx, b_name)
    if track is None:
        return passed()
    windows = rows(ctx, a_name)
    wrong = [
        (n, m, w)
        for n, m, w in window_means(windows, track)
        if abs(m - w) > 0.005 * max(1.0, abs(w))
    ]
    if not wrong:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "per_interval":
        return passed()
    name, mean_lines, weighted = wrong[0]
    return CheckResult(
        "fail",
        "map_mean_is_not_per_base",
        f"bedtools map -o mean averages the values of the {b_name} lines that touch a window, not "
        f"the bases: in {len(wrong)} of {len(windows)} windows that differs from the per-base "
        f"mean (window {name}: {mean_lines:.4g} against {weighted:.4g}), because the lines have "
        "different lengths or stick out of the window.",
        expected=round(weighted, 4),
        got=round(mean_lines, 4),
        fix=(
            f"weight by the overlap: bedtools intersect -a {a_name} -b {b_name} -wo, then sum "
            "depth*overlap per window and divide by the window length (awk/python)",
            f"or declare --declare {rule.declare.key}=per_interval if the mean of the lines is "
            "what you want",
        ),
    )


# ---- bedtools.spliced_without_split ----------------------------------------------------------
# slot split, T1 call rule, optional declaration `spliced_reads`.
# genomecov -ibam, coverage, multicov, intersect and window take an alignment as the interval from
# its first to its last aligned base: without -split a spliced read (CIGAR 20M300N20M) covers the
# 300 bases it skipped. Evidence (reads `101 20M300N20M` and `111 40M` on chr1, region 100-500):
# `genomecov -ibam -d` is positive in 340 positions, with -split in 70; `coverage` gives covered
# bases 340 / fraction 0.85 against 70 / 0.175 with -split, and the read count 2 against 3; the
# bedGraph of `genomecov -bg` shows 150-440:1 for the intron. exit 0, stderr empty throughout.
# The checker reads the first 20000 alignments of the BAM the call names (BGZF through gzip) and
# fails when a region of the call lies in the skipped bases of a spliced read (base-level output)
# or is hit only through them (read counts); `genomecov` has no region, so any spliced read fails.

SPLICE_VALUES = frozenset(
    {"-a", "-abam", "-b", "-bed", "-ibam", "-f", "-F", "-g", "-iobuf", "-names", "-w", "-l", "-r"}
    | {"-q", "-max", "-scale", "-fs", "-strand", "-trackopts", "-bams"}
)
BAM_SUFFIX = (".bam", ".sam")
BAM_LIMIT = 20_000


def _span_overlaps_only_via_gap(reads, regions) -> bool:
    import bisect

    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, s, e in regions:
        by_chrom.setdefault(chrom, []).append((s, e))
    starts = {c: sorted(v) for c, v in by_chrom.items()}
    keys = {c: [s for s, _ in v] for c, v in starts.items()}
    for aln in reads:
        items = starts.get(aln.chrom)
        if not items:
            continue
        i = bisect.bisect_left(keys[aln.chrom], aln.end0) - 1
        low = bisect.bisect_left(keys[aln.chrom], aln.start0 - 100_000)
        while i >= low:
            s, e = items[i]
            if e > aln.start0 and s < aln.end0:
                touched = any(b_e > s and b_s < e for b_s, b_e in aln.blocks)
                if not touched:
                    return True
            i -= 1
    return False


def _gap_overlap(reads, regions) -> bool:
    for aln in reads:
        gaps = [(aln.blocks[i][1], aln.blocks[i + 1][0]) for i in range(len(aln.blocks) - 1)]
        for chrom, s, e in regions:
            if chrom == aln.chrom and any(g_e > s and g_s < e for g_s, g_e in gaps):
                return True
    return False


@checker("iv_bedtools_spliced_without_split")
def spliced_without_split(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    sub = call.argv[0] if call.argv else ""
    opts, _, flags = bt_args(call.argv[1:], SPLICE_VALUES)
    if "-split" in flags:
        return passed()
    bams = [t for t in call.argv[1:] if t.lower().endswith(BAM_SUFFIX)]
    bams = [b for b in bams if ctx.resolve(b).is_file()]
    if not bams:
        return passed()
    spliced = []
    try:
        for bam in bams[:2]:
            spliced += [
                a for a in qc.iter_alignments(ctx, ctx.resolve(bam), BAM_LIMIT) if len(a.blocks) > 1
            ]
    except OSError:
        return passed()
    if not spliced:
        return passed()
    targets = [
        v
        for key in ("-a", "-b", "-bed")
        for v in opts.get(key, [])
        if not v.lower().endswith(BAM_SUFFIX) and ctx.resolve(v).is_file()
    ]
    regions = []
    for target in targets[:1]:
        for r in rows(ctx, target, 5000):
            try:
                regions.append((r[0], int(r[1]), int(r[2])))
            except (ValueError, IndexError):
                return passed()
    counts_only = (
        sub in ("multicov", "intersect", "window") or "-counts" in flags or "-c" in flags
    ) and sub != "genomecov"
    if sub == "genomecov":
        risky = True
        where = "every intron the reads skip is reported as covered"
    elif not regions:
        return passed()
    elif counts_only:
        risky = _span_overlaps_only_via_gap(spliced, regions)
        where = "reads that only skip over a region are counted as hits"
    else:
        risky = _gap_overlap(spliced, regions)
        where = "the bases a spliced read skips inside a region count as covered"
    if not risky:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "span":
        return passed()
    return CheckResult(
        "fail",
        "spliced_reads_without_split",
        f"{bams[0]} holds spliced reads (CIGAR N, e.g. at {spliced[0].chrom}:"
        f"{spliced[0].start0 + 1}) and bedtools {sub} without -split "
        f"treats each read as one interval from its first to its last base: {where}.",
        fix=(
            f"add -split to bedtools {sub} so that only the aligned blocks count",
            f"or declare --declare {rule.declare.key}=span if the whole span of each read is "
            "what you want",
        ),
    )


# ---- bedtools.subtract_trims -----------------------------------------------------------------
# slot overlap_policy, T1 call rule, optional declaration `overlap_policy`.
# `bedtools subtract` removes only the overlapping BASES of an a feature: a feature covered in part
# is written again, shorter, a feature with a b region inside it is written twice, and only a
# feature covered completely disappears; `-A` (or `-N`) removes a feature on any overlap.
# Evidence (peaks p1 100-200, p2 300-400, p3 500-600, p4 700-800; b 180-320, 510-520, 650-900):
# `subtract` writes 100-180 p1, 320-400 p2, 500-510 p3, 520-600 p3; `-A` and `intersect -v` write
# nothing; exit 0, stderr empty. The checker runs the call and the call with -A and fails when the
# outputs differ (some feature is written trimmed or split).

SUBTRACT_VALUES = frozenset({"-a", "-b", "-f", "-F", "-g", "-iobuf"})


@checker("iv_bedtools_subtract_trims")
def subtract_trims(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, _, flags = bt_args(call.argv[1:], SUBTRACT_VALUES)
    if "-A" in flags or "-N" in flags:
        return passed()
    a_name, b_name = qc.last_value(opts, "-a"), qc.last_value(opts, "-b")
    if a_name is None or b_name is None:
        return passed()
    if not ctx.resolve(a_name).is_file() or not ctx.resolve(b_name).is_file():
        return passed()
    if not rerunnable(ctx, a_name, b_name):
        return passed("an input is too large to run bedtools on it again")
    trimmed = run_lines(ctx, list(call.argv))
    removed = run_lines(ctx, [*call.argv, "-A"])
    if trimmed is None or removed is None:
        return CheckResult("unknown", "dry_run_failed", "the dry run of bedtools failed")
    untouched = set(removed)
    changed = [line for line in trimmed if line not in untouched]
    if not changed:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "trim":
        return passed()
    n_a = len({tuple(line.split("\t")[:4]) for line in changed})
    example = changed[0].split("\t")
    label = example[3] if len(example) > 3 else example[0]
    return CheckResult(
        "fail",
        "subtract_trims_not_removes",
        f"bedtools subtract does not drop a feature that overlaps {b_name}, it cuts out the "
        f"overlapping bases: {len(changed)} output line(s) are shortened or split pieces of "
        f"{a_name} features ({label} and others) that a plain overlap test would remove.",
        expected=len(removed),
        got=len(trimmed),
        fix=(
            "add -A to remove every a feature that overlaps b by one base "
            "(or use bedtools intersect -v)",
            f"or declare --declare {rule.declare.key}=trim if the trimmed pieces are what you want",
        ),
        note=f"{n_a} features affected",
    )


# ---- bedtools.window_boundary ----------------------------------------------------------------
# slot distance_unit, T1 call rule, optional declaration `window_boundary`.
# `window -w N` extends each a interval to [start-N, end+N) and tests overlap, so a b interval whose
# gap (bases strictly between the two) is exactly N is NOT reported; N+1 reaches it. Evidence
# (peak 1000-1100, b at 1700-1800, gap 600): `window -w 600` does not report it, `-w 601` does;
# `closest -d` prints gap+1 (book-ended 1, gap 3800 -> 3801). The checker finds a b interval at a
# gap of exactly N from an a interval, on the side(s) the options extend. Whether that interval
# should be in is the user's intent: no declaration warns (unknown), `gap_inclusive` fails,
# `window_exclusive` passes.

WINDOW_VALUES = frozenset({"-a", "-abam", "-b", "-w", "-l", "-r", "-g", "-iobuf"})


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@checker("iv_bedtools_window_boundary")
def window_boundary(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, _, flags = bt_args(call.argv[1:], WINDOW_VALUES)
    w = _int(qc.last_value(opts, "-w"))
    left = _int(qc.last_value(opts, "-l"))
    right = _int(qc.last_value(opts, "-r"))
    left = w if left is None else left
    right = w if right is None else right
    if "-sw" in flags and left != right:
        left = right = max(x for x in (left, right) if x is not None)
    a_name, b_name = qc.last_value(opts, "-a"), qc.last_value(opts, "-b")
    if a_name is None or b_name is None or (left is None and right is None):
        return passed()
    if not ctx.resolve(a_name).is_file() or not ctx.resolve(b_name).is_file():
        return passed()
    b_rows = rows(ctx, b_name)
    a_rows = rows(ctx, a_name)
    try:
        starts = {(r[0], int(r[1])) for r in b_rows}
        ends = {(r[0], int(r[2])) for r in b_rows}
        found = None
        for r in a_rows:
            s, e = int(r[1]), int(r[2])
            if right is not None and (r[0], e + right) in starts:
                found = (r, "after", right)
                break
            if left is not None and (r[0], s - left) in ends:
                found = (r, "before", left)
                break
    except (ValueError, IndexError):
        return passed()
    if found is None:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "window_exclusive":
        return passed()
    row, side, size = found
    label = row[3] if len(row) > 3 else f"{row[0]}:{row[1]}-{row[2]}"
    return CheckResult(
        "unknown" if declared is None else "fail",
        "window_excludes_gap_n" if declared else "window_boundary_undeclared",
        f"bedtools window with a size of {size} does not report a {b_name} interval whose gap to "
        f"an {a_name} interval is exactly {size} bases: {label} has one {side} it at that gap, "
        f"and the window [start-{size}, end+{size}) stops one base short of it.",
        fix=(
            f"use a size of {size + 1} for 'at most {size} bases between the two' "
            "(closest -d prints the gap plus one)",
            f"or declare --declare {rule.declare.key}=window_exclusive if a gap of exactly "
            f"{size} should stay out",
        ),
    )


# ---- bedtools.closest_distance_unit ----------------------------------------------------------
# slot distance_unit, T0 call rule, optional declaration `distance_unit`.
# `closest -d` / `-D` print 0 for overlap, 1 for book-ended intervals, gap+1 otherwise and -1 (with
# `. -1 -1 .` for the b fields) when the chromosome has no b feature. Evidence: peaks 1000-1100 /
# 5000-5100, genes 1100-1200, 4900-4950 give 1 and 3801 for gaps of 0 and 3800; chr3 without
# genes gives -1; ties are reported on several lines (-t all is the default). A threshold on the
# column therefore needs gap+1 and must exclude -1. Decided from the call alone: no declaration
# warns, `distance_unit=bases_between` contradicts what the tool prints.


@checker("iv_bedtools_closest_distance_unit")
def closest_distance_unit(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    declared = call.declared.get(rule.declare.key)
    if declared == "bedtools_distance":
        return passed()
    if declared == "bases_between":
        return CheckResult(
            "fail",
            "distance_is_gap_plus_one",
            "you declared distance_unit=bases_between but bedtools closest -d/-D print the gap "
            "plus one (book-ended intervals 1, a gap of 1000 bases 1001), 0 only for overlap, "
            "and -1 for a chromosome without b features.",
            expected="bedtools_distance",
            got=declared,
            fix=(
                "compare the distance column with limit+1 and exclude -1, then declare "
                f"--declare {rule.declare.key}=bedtools_distance",
            ),
        )
    return CheckResult(
        "unknown",
        "distance_unit_undeclared",
        "bedtools closest -d/-D print the gap plus one (book-ended 1, a gap of 1000 bases 1001), "
        "0 for overlap and -1 when the chromosome has no b feature; ties give several lines.",
        fix=(
            f"state --declare {rule.declare.key}=bedtools_distance and threshold the last "
            "column accordingly (<= limit+1 and >= 0)",
        ),
    )


# ---- bedtools.duplicate_policy ---------------------------------------------------------------
# slot default_param, T1 call rule, optional declaration `duplicate_policy`.
# multicov skips reads flagged 0x400 (and 0x200) unless -D / -F; coverage, intersect, genomecov
# -ibam and window count every record. Evidence in f75_multicov_dups.py: six records on one region
# (flags 0, 1024, 1024, 0, 512, 256/MAPQ 0): coverage -counts 6, intersect -c 6, multicov 3,
# multicov -D -F 6. A BAM with flagged records and a call whose declaration says the other policy
# fails; with no declaration the checker only warns (the intent is the user's).


def _dup_behaviour(sub: str, flags: set[str]) -> str:
    if sub == "multicov":
        return "keep" if "-D" in flags else "exclude"
    return "keep"


@checker("iv_bedtools_duplicate_policy")
def duplicate_policy(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    sub = call.argv[0] if call.argv else ""
    _, _, flags = bt_args(call.argv[1:], SPLICE_VALUES)
    bams = [t for t in call.argv[1:] if t.lower().endswith(BAM_SUFFIX)]
    bams = [b for b in bams if ctx.resolve(b).is_file()]
    if not bams:
        return passed()
    mask = 0x400 | (0x200 if sub == "multicov" and "-F" not in flags else 0)
    flagged = 0
    try:
        for bam in bams[:3]:
            flagged += sum(
                1 for a in qc.iter_alignments(ctx, ctx.resolve(bam), BAM_LIMIT) if a.flag & mask
            )
    except OSError:
        return passed()
    if not flagged:
        return passed()
    behaviour = _dup_behaviour(sub, flags)
    declared = call.declared.get(rule.declare.key)
    facts = f"{bams[0]} has {flagged} read(s) flagged duplicate/QC-fail; bedtools {sub} " + (
        "counts them" if behaviour == "keep" else "skips them (use -D / -F to count them)"
    )
    if declared is None:
        return CheckResult(
            "unknown",
            "duplicate_policy_undeclared",
            facts + "; coverage/intersect/genomecov count them, multicov skips them.",
            fix=(f"state --declare {rule.declare.key}=keep|exclude to match the call",),
        )
    if declared == behaviour:
        return passed()
    fix = (
        "drop the flagged reads first (samtools view -F 1024 -b) or use bedtools multicov"
        if declared == "exclude"
        else "add -D (and -F for QC-fail) to bedtools multicov to keep them"
    )
    return CheckResult(
        "fail",
        "duplicate_policy_contradicts_call",
        f"you declared {rule.declare.key}={declared} but {facts}.",
        expected=declared,
        got=behaviour,
        fix=(fix,),
    )


# ---- bedtools.merge_bookended ----------------------------------------------------------------
# slot interval_semantics, T1 call rule, optional declaration `adjacent_policy`.
# `merge` joins intervals that touch as well as intervals that overlap (-d 0 is the default: "that
# is, overlapping and/or book-ended features are merged"); only a negative -d keeps abutting
# intervals apart, and `-d 1` bridges a gap of one base. Evidence: 10-20, 20-30, 31-40, 45-50: plain
# merge gives 10-30, 31-40, 45-50; -d 1 gives 10-40, 45-50; -d -1 gives 10-20, 20-30, 31-40, 45-50
# (and still joins 10-25 with 20-30). Unsorted input stops loudly. exit 0, stderr empty. The
# checker looks for two intervals of the file with end == start (abutting, not overlapping).
# Whether to merge them is the user's intent: no declaration warns, `adjacent_policy=keep` fails
# a call without a negative -d, `merge` passes.

MERGE_VALUES = frozenset({"-i", "-d", "-c", "-o", "-g", "-prec", "-delim"})


def _abutting(table: list[list[str]], stranded: bool) -> tuple[str, int] | None:
    ends: dict[tuple[str, str], int] = {}
    ordered = sorted(
        ((r[0], r[5] if stranded and len(r) > 5 else ".", int(r[1]), int(r[2])) for r in table),
        key=lambda x: (x[0], x[1], x[2]),
    )
    for chrom, strand, start, end in ordered:
        key = (chrom, strand)
        if key in ends and start == ends[key]:
            return chrom, start
        ends[key] = max(ends.get(key, -1), end)
    return None


@checker("iv_bedtools_merge_bookended")
def merge_bookended(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    opts, _, flags = bt_args(call.argv[1:], MERGE_VALUES)
    name = qc.last_value(opts, "-i")
    distance = _int(qc.last_value(opts, "-d"))
    if name is None or (distance is not None and distance != 0):
        return passed()
    if not ctx.resolve(name).is_file():
        return passed()
    try:
        found = _abutting(rows(ctx, name), "-s" in flags)
    except (ValueError, IndexError):
        return passed()
    if found is None:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "merge":
        return passed()
    chrom, position = found
    return CheckResult(
        "unknown" if declared is None else "fail",
        "merge_joins_abutting" if declared else "merge_adjacent_undeclared",
        f"bedtools merge joins intervals that touch as well as intervals that overlap: {name} has "
        f"book-ended intervals (one ends where the next starts, at {chrom}:{position}), so they "
        "become one region.",
        fix=(
            "add -d -1 to merge only intervals that share at least one base",
            f"or declare --declare {rule.declare.key}=merge if touching intervals belong together",
        ),
    )


# ---- bedtools.groupby_unsorted ---------------------------------------------------------------
# slot record_order, T1 call rule, no declaration.
# `groupby` closes a group when the key changes, so an input that is not grouped by the -g columns
# writes the same key several times, each with a partial summary. Evidence: names g1 g2 g1 g2
# with scores 5 7 3 1: `groupby -g 4 -c 5 -o sum` prints g1 5 / g2 7 / g1 3 / g2 1 (exit 0,
# stderr empty); after `sort -k4,4` g1 8 / g2 8. The checker reads the -i file and fails when a
# key returns after another one.

GROUPBY_VALUES = frozenset({"-i", "-g", "-grp", "-c", "-opCols", "-o", "-ops", "-prec", "-delim"})


@checker("iv_bedtools_groupby_unsorted")
def groupby_unsorted(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    opts, _, flags = bt_args(call.argv[1:], GROUPBY_VALUES)
    name = qc.last_value(opts, "-i")
    cols = qc.last_value(opts, "-g", "-grp")
    if name is None or cols is None or not ctx.resolve(name).is_file():
        return passed()
    try:
        index = [int(c) - 1 for c in cols.split(",")]
    except ValueError:
        return passed()
    lines_ = list(ctx.lines(name, limit=MAX_LINES))
    if {"-header", "-inheader"} & flags:
        lines_ = lines_[1:]
    fold = str.lower if "-ignorecase" in flags else str
    seen: set[tuple[str, ...]] = set()
    previous: tuple[str, ...] | None = None
    for line in lines_:
        if not line.strip():
            continue
        f = line.split("\t")
        try:
            key = tuple(fold(f[i]) for i in index)
        except IndexError:
            return passed()
        if key != previous:
            if key in seen:
                column = ",".join(str(i + 1) for i in index)
                return CheckResult(
                    "fail",
                    "groupby_input_not_grouped",
                    f"bedtools groupby starts a new group whenever the key changes, and {name} is "
                    f"not grouped by column(s) {column}: {'/'.join(key)} comes back after another "
                    "key, so it is reported more than once with partial summaries.",
                    fix=(
                        f"sort first: sort -k{index[0] + 1},{index[0] + 1} {name} | bedtools "
                        f"groupby -i - -g {cols} ... (sort on every grouping column)",
                    ),
                )
            seen.add(key)
            previous = key
    return passed()
