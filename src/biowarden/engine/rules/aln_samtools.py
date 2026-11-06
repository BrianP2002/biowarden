"""samtools rules of the alignment pack: what a MAPQ cutoff keeps, and six state traps.

Rules (engine/specs/aln_samtools.yaml). Every behaviour was run on samtools 1.24 with output of
STAR 2.7.11b, bwa 0.7.19, bowtie2 2.5.5, hisat2 2.2.3 and minimap2 2.31; the numbers are quoted in
each section.

samtools.mapq_unique_cutoff (T1, call, declaration optional; family F29)
    `samtools view -q N` is read as "keep the uniquely mapped reads" because bwa gives multi-mapped
    reads MAPQ 0. Aligners disagree about MAPQ. Measured on 30 reads copied from a repeat that
    occurs twice and 30 unique reads (100 bases each, single-end):
        bwa mem     unique 60, repeat-derived 0                                   -q 1 is unique
        minimap2 -x sr  unique 60, repeat-derived 0                               -q 1 is unique
        bowtie2     unique 42, repeat-derived 1 (30 of 30; 0 or 1 in pairs)       -q 1 keeps multi
        hisat2      unique 60, repeat-derived 1, two records each (flags 0, 256)  -q 1 keeps multi
        STAR        unique 255, 2 loci 3, 3-4 loci 1 (family F29: 19 multi-mapped reads of 92,
                    seed 0, all with MAPQ 3 or 1; 5+ loci would give 0)        -q 1..3 keeps multi
        bowtie2 -k 1   255 for every read (MAPQ is not computed)                  any -q keeps all
    Unique bowtie2 pairs with 4 mismatches have MAPQ 23 (100 pairs: 42 x100, 40 x84, 23 x16), so
    `-q 30` drops unique reads there; STAR unique reads stay at 255 whatever the mismatches.
    The rule samples the first 20000 records of the file named in the call and classifies each
    mapped record: multi-mapped when NH > 1, when the flag has 0x100, or when XS >= AS (a
    bowtie2 tie);
    unique otherwise. It fails when the cutoff keeps a sampled multi-mapped record (unless the call
    declares `quality_only`, a plain quality threshold) or, with `unique_only` declared, when it
    drops a sampled unique record. It stays silent on data without multi-mapped records, on calls
    without -q and on calls that also filter on NH or XS (`-d NH:1`, `-e '[NH]==1'`).

samtools.stats_nm_missing (T1, call, declaration optional; family F32)
    `samtools stats` on STAR output with default attributes (NH HI AS nM): reads with 5-10
    mismatches give `SN mismatches: 0`, `SN error rate: 0.000000e+00`; the same reads aligned with
    `--outSAMattributes NH HI AS nM NM MD` give mismatches 950, error rate 2.38e-02; `stats -r
    ref.fa` leaves both at 0; `samtools calmd -b in.bam ref.fa | samtools stats -` gives 950. hisat2
    output of the same reads (NM present) gave 457 (a different alignment, same tag).

samtools.fixmate_grouping (T1, call, no declaration; family F36)
    `samtools fixmate -m` on a BAM whose mates are not adjacent exits 0 with empty stderr and clears
    the mate fields of every record (properly paired 280 -> 2 of 320 reads); the header only has to
    avoid `SO:coordinate` (a coordinate-sorted header is refused: "Coordinate sorted, require
    grouped/sorted by queryname"). markdup then flags 49 reads instead of 40. markdup on a BAM
    without ms tags stops with "no ms score tag", and on a name-grouped BAM with "not in coordinate
    sorted order": only fixmate is silent.

samtools.merge_combine_rg (T1, call, no declaration; family F37)
    `samtools merge -c` with read groups that share an ID but differ in SM keeps the first file's
    @RG line: 640 reads, all RG:Z:g1 / SM:sampleA. Without -c the second file's group is renamed
    (g1-5B5BFF6A) and both samples survive; `-r` renames the IDs after the file names.

samtools.idxstats_unmapped (T1, call, declaration optional; family F38)
    `samtools idxstats` prints unmapped reads per contig (column 4) and the `*` row only holds reads
    without a coordinate; an unmapped mate sits on its mapped mate's contig. 6 reads, 2 mapped, 4
    unmapped: `chr1 3000 1 1`, `chr2 2000 1 0`, `* 0 0 3`; `samtools view -c -f 4` finds 4.

samtools.split_flag_after_bwa_m (T1, call, no declaration; family F39)
    `bwa mem -M` ("mark shorter split hits as secondary") writes the other parts of a chimeric read
    with flag 0x100 instead of 0x800. 40 chimeric reads of 70+50 bases and 40 ordinary reads:
    default flags 0:80, 2048:40; `-M` flags 0:80, 256:40; `-Y` as default. `samtools view -c -f
    0x800` gives 40 and 0, `-c -f 0x100` gives 0 and 40, `-c -F 0x904 -e 'exists([SA])'` gives 40
    in both (the SA tag stays on the primary); `samtools flagstat` prints "0 + 0 supplementary" and
    "40 + 0 secondary" for the -M file. The rule reads the @PG command line of the input (bwa keeps
    its command there) and fails a `view -f ... 0x800` (or a lone `-F 0x800`) on a file whose
    `bwa mem` line has -M.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.aln_bam import (
    Alignment,
    Unreadable,
    alignment_files,
    read_alignment,
)

# ---- option parsing shared by the rules --------------------------------------------------------

# The options of htslib's shared block (`sam_global_args`) that take a value, in every subcommand:
# `samtools stats --reference genome.fa reads.bam`. They are merged into the table of each
# subcommand. Only --threads has a short form (-@); every other one is keyed by a private
# character that no subcommand uses as an option letter, so its value is skipped without being
# read as another option's: `--reference` is not `-r` (a read group in view, the GC-depth
# reference in stats).
GLOBAL_NO_SHORT = (
    "--reference",
    "--input-fmt",
    "--output-fmt",
    "--input-fmt-option",
    "--output-fmt-option",
    "--verbosity",
)
GLOBAL_LONG = {name: chr(0xE000 + i) for i, name in enumerate(GLOBAL_NO_SHORT)} | {"--threads": "@"}
GLOBAL_ARG_SHORT = "".join(GLOBAL_LONG.values())

VIEW_ARG_SHORT = "oUtTLrRdDNqlmfFGxsOe" + GLOBAL_ARG_SHORT
VIEW_LONG = {
    **GLOBAL_LONG,
    "--min-MQ": "q",
    "--incl-flags": "f",
    "--require-flags": "f",
    "--excl-flags": "F",
    "--exclude-flags": "F",
    "--output": "o",
    "--subsample": "s",
    "--expr": "e",
    "--tag": "d",
    "--region-file": "L",
    "--target-file": "L",
}


def parse_options(
    argv: Sequence[str], arg_short: str, long_map: dict[str, str]
) -> tuple[dict[str, list[str]], list[str]]:
    """(option letter -> values, positionals) of a samtools subcommand's arguments.

    Handles bundled short flags (`-cq1`, `-bh`), attached values (`-q1`), `--name value`,
    `--name=value`, and `--` (everything after is positional). Letters in `arg_short` take a value.
    Long options in `long_map` are folded into their short letter; other long options are kept
    under their full name (with a value only after `=`).
    """
    opts: dict[str, list[str]] = {}
    positionals: list[str] = []
    items = list(argv)
    i = 0
    while i < len(items):
        token = items[i]
        i += 1
        if token == "--":
            positionals += items[i:]
            break
        if token.startswith("--"):
            name, eq, value = token.partition("=")
            letter = long_map.get(name)
            if letter is not None and letter in arg_short:
                if not eq and i < len(items):
                    value, i = items[i], i + 1
                opts.setdefault(letter, []).append(value)
            else:
                opts.setdefault(letter or name, []).append(value)
            continue
        if token.startswith("-") and len(token) > 1:
            for k, letter in enumerate(token[1:], start=1):
                if letter in arg_short:
                    value = token[k + 1 :]
                    if not value and i < len(items):
                        value, i = items[i], i + 1
                    opts.setdefault(letter, []).append(value)
                    break
                opts.setdefault(letter, []).append("")
            continue
        positionals.append(token)
    return opts, positionals


def flag_value(text: str) -> int | None:
    """A samtools flag argument: decimal, 0x hex or 0 octal; None when it is not a number."""
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(text)
        except ValueError:
            return None


def _shell(call: Call, argv: Sequence[str], *declare: str) -> str:
    parts = [call.tool, *argv]
    for item in declare:
        parts += ["--declare", item]
    return shlex.join(parts)


def _programs(aln: Alignment) -> str:
    names = []
    for pg in aln.program_lines():
        name = pg.get("PN") or pg.get("ID", "")
        if name and name not in names and not name.startswith("samtools"):
            names.append(name)
    return "/".join(names) or "an unknown program"


def _unreadable(path: Path, exc: Exception, what: str) -> CheckResult:
    return CheckResult(
        "unknown",
        "alignment_unreadable",
        f"{path.name} could not be sampled ({exc}); {what} cannot be checked against the data.",
    )


# ---- samtools.mapq_unique_cutoff ---------------------------------------------------------------

MAPQ_KEY = "mapq_filter_goal"
SAMPLE = 20_000


@dataclass(frozen=True)
class MapqSample:
    mapped: int
    multi: int
    unique: int
    multi_mapqs: tuple[int, ...]  # sorted distinct MAPQ values of multi-mapped records
    unique_mapqs: tuple[int, ...]  # sorted distinct MAPQ values of unique records

    def multi_max(self) -> int:
        return max(self.multi_mapqs, default=-1)

    def unique_min(self) -> int:
        return min(self.unique_mapqs, default=255)


def is_multi_mapped(rec) -> bool:
    """Tag evidence of a read with another equally good (or secondary) alignment."""
    nh = rec.tags.get("NH")
    if isinstance(nh, int) and nh > 1:
        return True
    if rec.flag & 0x100:
        return True
    xs, as_ = rec.tags.get("XS"), rec.tags.get("AS")
    return isinstance(xs, int) and isinstance(as_, int) and xs >= as_


def sample_mapq(aln: Alignment) -> MapqSample:
    multi_q: set[int] = set()
    unique_q: set[int] = set()
    mapped = multi = unique = 0
    for rec in aln.records:
        if rec.unmapped or rec.flag & 0x800:
            continue
        mapped += 1
        if is_multi_mapped(rec):
            multi += 1
            multi_q.add(rec.mapq)
        else:
            unique += 1
            unique_q.add(rec.mapq)
    return MapqSample(mapped, multi, unique, tuple(sorted(multi_q)), tuple(sorted(unique_q)))


def _mapq_fix(call: Call, aln: Alignment, sample: MapqSample, q: int) -> tuple[str, ...]:
    name = _programs(aln)
    high = sample.multi_max() + 1
    fixes = []
    if any("NH" in rec.tags for rec in aln.records[:50]):
        fixes.append(
            "keep the uniquely mapped reads by their tag instead: add -d NH:1 (and -F 0x100 to "
            f"count reads, not alignments), or raise the cutoff to -q {high}"
        )
    else:
        fixes.append(f"raise the cutoff above the multi-mapped MAPQ values: -q {high} or higher")
    fixes.append(
        f"if -q {q} is only a quality threshold, say so with --declare {MAPQ_KEY}=quality_only "
        f"({name} gives multi-mapped reads MAPQ {', '.join(map(str, sample.multi_mapqs))})"
    )
    return tuple(fixes)


@checker("samtools_mapq_unique_cutoff")
def mapq_unique_cutoff(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "view":
        return passed()
    opts, positionals = parse_options(call.argv[1:], VIEW_ARG_SHORT, VIEW_LONG)
    cutoffs = [flag_value(v) for v in opts.get("q", [])]
    if not cutoffs or cutoffs[-1] is None or cutoffs[-1] <= 0:
        return passed()
    q = cutoffs[-1]
    if any(v.startswith("NH:") for v in opts.get("d", [])) or any(
        re.search(r"\bNH\b|\bXS\b", v) for v in opts.get("e", [])
    ):
        return passed("the call filters on NH or XS as well")
    files = alignment_files(ctx, positionals)
    if not files:
        return passed()
    path = files[0]
    try:
        aln = read_alignment(ctx, path, SAMPLE)
    except (Unreadable, OSError, EOFError) as exc:
        return _unreadable(path, exc, f"what -q {q} keeps")
    sample = sample_mapq(aln)
    declared = call.declared.get(MAPQ_KEY)
    keeps_multi = sample.multi and sample.multi_max() >= q
    drops_unique = sample.unique and sample.unique_min() < q
    if keeps_multi and declared != "quality_only":
        kept = sum(
            1 for rec in aln.records if not rec.unmapped and is_multi_mapped(rec) and rec.mapq >= q
        )
        low = sample.unique_mapqs[0] if sample.unique_mapqs else "?"
        high = sample.unique_mapqs[-1] if sample.unique_mapqs else "?"
        values = ", ".join(map(str, sample.multi_mapqs))
        message = (
            f"-q {q} keeps multi-mapped reads of {path.name}: {kept} of the {sample.mapped} "
            f"sampled alignments are multi-mapped (NH > 1, XS >= AS or flag 0x100) and pass it. "
            f"{_programs(aln)} gives multi-mapped reads MAPQ {values} and unique reads MAPQ "
            f"{low}..{high}; only bwa gives multi-mapped reads MAPQ 0."
        )
        return CheckResult(
            "fail",
            "multimappers_pass_cutoff",
            message,
            expected=f"cutoff above {sample.multi_max()}",
            got=q,
            fix=_mapq_fix(call, aln, sample, q),
        )
    if drops_unique and declared == "unique_only":
        dropped = sum(
            1
            for rec in aln.records
            if not rec.unmapped and not is_multi_mapped(rec) and rec.mapq < q
        )
        return CheckResult(
            "fail",
            "unique_reads_dropped",
            f"-q {q} also drops unique reads of {path.name}: {dropped} of {sample.unique} sampled "
            f"unique alignments written by {_programs(aln)} have MAPQ "
            f"{sample.unique_min()}..{q - 1}.",
            expected=f"cutoff at most {sample.unique_min()}",
            got=q,
            fix=(
                f"use -q {max(sample.multi_max() + 1, 1)} (above the multi-mapped MAPQ values "
                f"{sample.multi_mapqs or 'none sampled'}, at most {sample.unique_min()})",
            ),
        )
    return passed()


# ---- samtools.stats_nm_missing -----------------------------------------------------------------

STATS_KEY = "stats_mismatches"
STATS_ARG_SHORT = "cfFgiIlmPqrSt" + GLOBAL_ARG_SHORT
STATS_LONG = {
    **GLOBAL_LONG,
    "--coverage": "c",
    "--required-flag": "f",
    "--filtering-flag": "F",
    "--GC-depth": "g",
    "--insert-size": "i",
    "--id": "I",
    "--read-length": "l",
    "--most-inserts": "m",
    "--split-prefix": "P",
    "--trim-quality": "q",
    "--ref-seq": "r",
    "--split": "S",
    "--target-regions": "t",
}


def _mismatch_evidence(aln: Alignment) -> int:
    """Sampled primary mapped records that show mismatches without an NM tag."""
    seen = 0
    for rec in aln.records:
        if rec.unmapped or not rec.primary:
            continue
        n_m = rec.tags.get("nM")  # STAR: mismatches per read (pair)
        md = rec.tags.get("MD")
        if (isinstance(n_m, int) and n_m > 0) or (
            isinstance(md, str) and any(c.isalpha() for c in md.replace("^", ""))
        ):
            seen += 1
        elif "X" in rec.cigar:
            seen += 1
    return seen


@checker("samtools_stats_nm_missing")
def stats_nm_missing(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "stats":
        return passed()
    _, positionals = parse_options(call.argv[1:], STATS_ARG_SHORT, STATS_LONG)
    files = alignment_files(ctx, positionals)  # the first one that exists: stats genome.fa in.bam
    if not files:
        return passed()
    path = files[0]
    try:
        aln = read_alignment(ctx, path, SAMPLE)
    except (Unreadable, OSError, EOFError) as exc:
        return _unreadable(path, exc, "whether the mismatches line of stats is meaningful")
    mapped = [r for r in aln.records if not r.unmapped and r.primary]
    if not mapped:
        return passed()
    with_nm = sum("NM" in r.tags for r in mapped)
    if with_nm / len(mapped) >= 0.9:
        return passed()
    if call.declared.get(STATS_KEY) == "not_used":
        return passed()
    evidence = _mismatch_evidence(aln)
    fix = (
        f"add the NM tags first: samtools calmd -b {path.name} <reference.fa> | samtools stats -",
        "or realign with the tags: STAR --outSAMattributes NH HI AS nM NM MD",
        f"if you do not use the mismatches and error rate lines, say so: --declare "
        f"{STATS_KEY}=not_used",
    )
    base = (
        f"{path.name} has no NM tag on {len(mapped) - with_nm} of {len(mapped)} sampled mapped "
        f"alignments (written by {_programs(aln)}), and samtools stats takes its `mismatches` and "
        "`error rate` lines from NM"
    )
    if evidence:
        return CheckResult(
            "fail",
            "nm_missing_mismatches_present",
            f"{base}: it will print mismatches 0 and error rate 0 although {evidence} sampled "
            "alignments carry mismatches (nM or MD). `stats -r reference.fa` does not change it.",
            expected="NM on every alignment",
            got=f"{with_nm} of {len(mapped)}",
            fix=fix,
        )
    return CheckResult(
        "unknown",
        "nm_missing",
        f"{base}: the mismatches and error rate lines will read 0 whether or not the reads have "
        "mismatches.",
        fix=fix,
    )


# ---- samtools.fixmate_grouping -----------------------------------------------------------------

FIXMATE_ARG_SHORT = "Oz" + GLOBAL_ARG_SHORT
FIXMATE_LONG = {**GLOBAL_LONG, "--output-fmt": "O", "--sanitize": "z"}
MIN_PAIRED = 4
MIN_SCATTERED_SHARE = 0.2


def scattered_mates(aln: Alignment) -> tuple[int, int]:
    """(paired primary records whose mate record is somewhere else, paired primary records).

    A record is scattered when its name occurs again in the sample but neither the previous nor the
    next record carries it. Pairs whose mate is absent from the sample are not counted.
    """
    recs = [r for r in aln.records if r.flag & 1 and r.primary]
    names: dict[str, int] = {}
    for rec in recs:
        names[rec.name] = names.get(rec.name, 0) + 1
    scattered = 0
    for i, rec in enumerate(recs):
        if names[rec.name] < 2:
            continue
        near = (i > 0 and recs[i - 1].name == rec.name) or (
            i + 1 < len(recs) and recs[i + 1].name == rec.name
        )
        scattered += not near
    return scattered, len(recs)


@checker("samtools_fixmate_grouping")
def fixmate_grouping(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "fixmate":
        return passed()
    _, positionals = parse_options(call.argv[1:], FIXMATE_ARG_SHORT, FIXMATE_LONG)
    files = alignment_files(ctx, positionals[:1])
    if not files:
        return passed()
    path = files[0]
    try:
        aln = read_alignment(ctx, path, SAMPLE)
    except (Unreadable, OSError, EOFError) as exc:
        return _unreadable(path, exc, "whether the mates are next to each other")
    if aln.hd().get("SO") == "coordinate":
        return passed("samtools itself refuses a coordinate-sorted header")
    scattered, paired = scattered_mates(aln)
    if paired < MIN_PAIRED or scattered / paired < MIN_SCATTERED_SHARE:
        return passed()
    order = f"SO:{aln.hd()['SO']}" if "SO" in aln.hd() else "no SO tag"
    return CheckResult(
        "fail",
        "mates_not_adjacent",
        f"{path.name} is not grouped by read name: in the first {paired} paired records, "
        f"{scattered} have their mate somewhere else (header: {order}). samtools fixmate pairs "
        "mates by looking at consecutive records and only refuses a coordinate-sorted header, so "
        "it will treat those reads as unpaired, clear their mate fields and the proper-pair flag, "
        "and exit 0 without a message.",
        expected="records grouped by read name",
        got=f"{scattered} of {paired} paired records scattered",
        fix=(
            f"samtools sort -n -o names.bam {path.name} && "
            + _shell(call, [a if a != positionals[0] else "names.bam" for a in call.argv]),
            f"(samtools collate -o names.bam {path.name} groups the mates as well)",
        ),
    )


# ---- samtools.merge_combine_rg -----------------------------------------------------------------

MERGE_ARG_SHORT = "tlRhsbLOo" + GLOBAL_ARG_SHORT
MERGE_LONG = {**GLOBAL_LONG, "--output-fmt": "O"}


def _merge_inputs(ctx: Ctx, opts: dict[str, list[str]], positionals: list[str]) -> list[Path]:
    names = positionals if "o" in opts else positionals[1:]
    for listing in opts.get("b", []):
        path = ctx.resolve(listing)
        if path.is_file():
            names += [line.strip() for line in ctx.lines(path) if line.strip()]
    return alignment_files(ctx, names)


@checker("samtools_merge_combine_rg")
def merge_combine_rg(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "merge":
        return passed()
    opts, positionals = parse_options(call.argv[1:], MERGE_ARG_SHORT, MERGE_LONG)
    if "c" not in opts:
        return passed()
    groups: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for path in _merge_inputs(ctx, opts, positionals):
        try:
            aln = read_alignment(ctx, path, 1)
        except (Unreadable, OSError, EOFError) as exc:
            return _unreadable(path, exc, "the read groups that -c would combine")
        for rg in aln.read_groups():
            groups.setdefault(rg.get("ID", ""), []).append((path.name, rg))
    unknown: list[str] = []
    for rg_id, entries in groups.items():
        if len(entries) < 2:
            continue
        samples = {rg.get("SM", "") for _, rg in entries}
        files = ", ".join(f"{name} (SM:{rg.get('SM', '?')})" for name, rg in entries)
        if len(samples) > 1:
            return CheckResult(
                "fail",
                "samples_share_read_group_id",
                f"-c combines @RG lines with the same ID into the first file's line, and read "
                f"group ID {rg_id} is used by different samples in {files}: the reads of the later "
                "files get the first file's SM (and LB, PL), silently, and one sample disappears "
                "from the merged header. Without -c the colliding ID is renamed and every sample "
                "keeps its own group.",
                expected="no -c with colliding read group IDs of different samples",
                got="-c",
                fix=(
                    _shell(call, [a for a in call.argv if a != "-c"]),
                    "(without -c samtools merge renames the colliding ID, for example "
                    f"{rg_id}-5B5BFF6A, and rewrites the RG tags of those reads)",
                ),
            )
        if any(
            {k: v for k, v in rg.items() if k != "ID"}
            != {k: v for k, v in entries[0][1].items() if k != "ID"}
            for _, rg in entries
        ):
            unknown.append(f"{rg_id} ({files})")
    if unknown:
        return CheckResult(
            "unknown",
            "read_group_fields_differ",
            "-c keeps the first file's @RG line for a colliding ID, so the other fields of "
            f"{'; '.join(unknown)} (LB, PL, PU, ...) are lost.",
        )
    return passed()


# ---- samtools.idxstats_unmapped ----------------------------------------------------------------

IDXSTATS_KEY = "idxstats_unmapped"
IDXSTATS_ARG_SHORT = GLOBAL_ARG_SHORT
IDXSTATS_LONG = dict(GLOBAL_LONG)


@checker("samtools_idxstats_unmapped")
def idxstats_unmapped(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "idxstats":
        return passed()
    _, positionals = parse_options(call.argv[1:], IDXSTATS_ARG_SHORT, IDXSTATS_LONG)
    files = alignment_files(ctx, positionals[:1])
    if not files:
        return passed()
    path = files[0]
    try:
        proc = ctx.run_real("samtools", ["idxstats", str(path)])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _unreadable(path, exc, "the unmapped reads on its contig rows")
    if proc.returncode != 0:
        return passed()  # the call fails loudly on its own (no index, not a BAM)
    placed = star = 0
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) != 4 or not f[3].isdigit():
            return passed()
        if f[0] == "*":
            star = int(f[3])
        else:
            placed += int(f[3])
    if placed == 0:
        return passed()
    declared = call.declared.get(IDXSTATS_KEY)
    if declared == "contig_rows_included":
        return passed()
    fix = (
        f"count the unmapped reads over every row: samtools idxstats {path.name} | "
        "awk '{u += $4} END {print u}'",
        f"or ask the flag directly: samtools view -c -f 4 {path.name}",
        f"if you do read the unmapped count of the contig rows too, declare it: --declare "
        f"{IDXSTATS_KEY}=contig_rows_included",
    )
    base = (
        f"{path.name} has {placed} unmapped reads on contig rows (unmapped mates sit beside their "
        f"mapped partner, so idxstats counts them in column 4 of that contig) and {star} on the "
        f"`*` row; the `*` row alone is not the number of unmapped reads (total {placed + star})."
    )
    if declared == "star_row_only":
        return CheckResult(
            "fail",
            "star_row_misses_placed_mates",
            f"you read only the `*` row: {base}",
            expected=placed + star,
            got=star,
            fix=fix,
        )
    return CheckResult("unknown", "unmapped_mates_on_contig_rows", base, fix=fix)


# ---- samtools.split_flag_after_bwa_m -------------------------------------------------------------

SUPPLEMENTARY, SECONDARY = 0x800, 0x100
BWA_M = re.compile(r"(?:^|\s)-[A-Za-z0-9]*M[A-Za-z0-9]*(?=\s|$)")


def _made_with_bwa_m(aln: Alignment) -> str | None:
    """The @PG command line of a `bwa mem` run that used -M, or None."""
    for pg in aln.program_lines():
        line = pg.get("CL", "")
        words = line.split(None, 2)
        if (
            len(words) >= 3
            and words[0].rsplit("/", 1)[-1] == "bwa"
            and words[1] == "mem"
            and BWA_M.search(" " + words[2])
        ):
            return line
    return None


@checker("samtools_split_flag_bwa_m")
def split_flag_after_bwa_m(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    if call.subcommand != "view":
        return passed()
    opts, positionals = parse_options(call.argv[1:], VIEW_ARG_SHORT, VIEW_LONG)
    wanted = [flag_value(v) for v in opts.get("f", [])]
    excluded = [flag_value(v) for v in opts.get("F", [])]
    selects = any(v is not None and v & SUPPLEMENTARY for v in wanted)
    drops_only_supplementary = any(
        v is not None and v & SUPPLEMENTARY and not v & SECONDARY for v in excluded
    )
    if not (selects or drops_only_supplementary):
        return passed()
    files = alignment_files(ctx, positionals)
    if not files:
        return passed()
    path = files[0]
    try:
        aln = read_alignment(ctx, path, 1)
    except (Unreadable, OSError, EOFError) as exc:
        return _unreadable(path, exc, "how the split hits are flagged")
    command = _made_with_bwa_m(aln)
    if command is None:
        return passed()
    effect = (
        "selects nothing for the split parts of chimeric reads"
        if selects
        else "leaves the split parts in (they are 0x100, not 0x800)"
    )
    return CheckResult(
        "fail",
        "split_hits_are_secondary",
        f"{path.name} was made with bwa mem -M ({command.split(None, 2)[2][:50]}...), which "
        f"flags the shorter split hits of a chimeric read 0x100 (secondary) instead of 0x800 "
        f"(supplementary): the flag test on 0x800 {effect}.",
        expected="0x100 for the split parts",
        got="0x800",
        fix=(
            "count chimeric reads by the SA tag on the primary record, which -M leaves in place: "
            f"samtools view -c -F 0x904 -e 'exists([SA])' {path.name}",
            "or test 0x100 as well (secondary) when the file was made with -M",
        ),
    )
