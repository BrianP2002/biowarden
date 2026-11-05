"""Admission cases for the bedtools rules of pack iv (families F65-F69 and F75).

For one seed both twins of the family are built (implicit tier) and every template below is run
with the real bedtools in the workspace; its output is turned into the task's answer and compared
with `task.truth["expected"]` (bench.score.answer_matches), so the label never comes from a rule.
A template is one call plus the small post-processing a solver would write (a column, a count, a
threshold). The same call can therefore be right under one threshold and wrong under another (the
closest -d templates): what tells them apart is the declaration the call carries, which states the
belief the threshold encodes.

Each call is emitted without a declaration and with one. Right calls carry the declaration that
matches what the call really does; wrong calls carry both the honest one (the rule cannot see a
wrong goal) and the one that contradicts the tool (the rule must fail it). Rules without a
declaration are emitted twice, with an empty declaration, so the declared cell is not empty.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

Answer = Callable[[str, Path], Any]


@dataclass(frozen=True)
class Template:
    name: str
    argv: tuple[str, ...]  # bedtools arguments including the subcommand
    answer: Answer  # stdout (and the workspace) -> the task's answer
    right_decl: str | None = None
    wrong_decls: tuple[str, ...] = ()
    setup: Callable[[ToolEnv, Path], None] | None = None  # extra files for this call
    twins: tuple[str, ...] = ("clean", "faulted")


def emit(
    rule_key: str | None,
    family: str,
    seed: int,
    scratch: Path,
    tools: ToolEnv,
    templates: tuple[Template, ...],
    extra: Callable[[dict[str, Any], Path, str], list[Case]] | None = None,
    builder: Callable[[int, str, Path], dict[str, Any]] | None = None,
) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        if builder is not None:
            work.mkdir(parents=True, exist_ok=True)
            truth = builder(seed, twin, work)
        else:
            truth = build_task(family, seed, twin, work, tools, "implicit", "free").truth
        for template in templates:
            if twin not in template.twins:
                continue
            if template.setup is not None:
                template.setup(tools, work)
            proc = tools.run(["bedtools", *template.argv], work)
            label: Label = "wrong"
            if proc.returncode == 0:
                try:
                    ok = answer_matches(truth, template.answer(proc.stdout, work))
                except (ValueError, IndexError, KeyError, ZeroDivisionError):
                    ok = False
                label = "right" if ok else "wrong"
            call = Call("bedtools", template.argv, work)
            out.append(Case(call, label, False, family, seed, twin, template.name))
            if rule_key is None:
                out.append(Case(call, label, True, family, seed, twin, template.name))
                continue
            options = [template.right_decl] if label == "right" else list(template.wrong_decls)
            for value in options:
                decl = {rule_key: value} if value else {}
                call = Call("bedtools", template.argv, work, decl)
                out.append(Case(call, label, True, family, seed, twin, template.name))
        if extra is not None:
            out += extra(truth, work, twin)
    return out


def lines(stdout: str) -> list[list[str]]:
    return [line.split("\t") for line in stdout.splitlines() if line.strip()]


# ---- F65 split_reciprocal ----------------------------------------------------------------------

F = "0.9"


def names_col4(stdout: str, work: Path) -> list[str]:
    return sorted({r[3] for r in lines(stdout)})


def _one_record_files(tools: ToolEnv, work: Path) -> None:
    for i, line in enumerate((work / "tx_b.bed").read_text().splitlines()):
        (work / f"one_{i:03d}.bed").write_text(line + "\n")


def _per_record_answer(tools: ToolEnv, work: Path) -> list[str]:
    names: set[str] = set()
    for path in sorted(work.glob("one_*.bed")):
        proc = tools.run(
            ["bedtools", "intersect", "-a", "tx_a.bed", "-b", path.name]
            + ["-split", "-r", "-f", F, "-wa", "-u"],
            work,
        )
        names |= {r[3] for r in lines(proc.stdout)}
    return sorted(names)


@case_source("bedtools.split_reciprocal")
def split_reciprocal_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    base = ("intersect", "-a", "tx_a.bed", "-b", "tx_b.bed")
    templates = (
        Template("r_f_wa_u", (*base, "-split", "-r", "-f", F, "-wa", "-u"), names_col4),
        Template("r_f_wa_wb", (*base, "-split", "-r", "-f", F, "-wa", "-wb"), names_col4),
        Template("f_F_wa_u", (*base, "-split", "-f", F, "-F", F, "-wa", "-u"), names_col4),
        Template("F_wa_wb", (*base, "-split", "-F", F, "-wa", "-wb"), names_col4),
        Template("f_wa_u", (*base, "-split", "-f", F, "-wa", "-u"), names_col4),
        Template("no_split_r_f", (*base, "-r", "-f", F, "-wa", "-u"), names_col4),
        Template("split_wa_u", (*base, "-split", "-wa", "-u"), names_col4),
    )

    def per_record(truth: dict[str, Any], work: Path, twin: str) -> list[Case]:
        _one_record_files(tools, work)
        right = answer_matches(truth, _per_record_answer(tools, work))
        label: Label = "right" if right else "wrong"
        cases = []
        for path in sorted(work.glob("one_*.bed"))[:8]:
            argv = (*base[:3], path.name, "-split", "-r", "-f", F, "-wa", "-u")
            call = Call("bedtools", argv, work)
            cases.append(Case(call, label, False, "F65", seed, twin, "per_record"))
            cases.append(Case(call, label, True, "F65", seed, twin, "per_record"))
        return cases

    return emit(None, "F65", seed, scratch, tools, templates, per_record)


# ---- F66 map_weighted --------------------------------------------------------------------------


def map_dict(stdout: str, work: Path) -> dict[str, float]:
    return {r[3]: float(r[4]) for r in lines(stdout)}


def wo_weighted(stdout: str, work: Path) -> dict[str, float]:
    total: dict[str, float] = {}
    length: dict[str, int] = {}
    for r in lines(stdout):
        total[r[3]] = total.get(r[3], 0.0) + float(r[7]) * int(r[8])
        length[r[3]] = int(r[2]) - int(r[1])
    return {k: total[k] / length[k] for k in total}


def _binned_track(tools: ToolEnv, work: Path) -> None:
    out = []
    for line in (work / "coverage.bedgraph").read_text().splitlines():
        c, s, e, v = line.split("\t")
        out += [f"{c}\t{x}\t{x + 10}\t{v}\n" for x in range(int(s), int(e), 10)]
    (work / "binned.bedgraph").write_text("".join(out))


@case_source("bedtools.map_unweighted")
def map_unweighted_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    track, win = "coverage.bedgraph", "windows.bed"
    mean = ("map", "-a", win, "-b", track, "-c", "4", "-o", "mean")
    templates = (
        Template("map_mean", mean, map_dict, "per_base", ("per_base", "per_interval")),
        Template(
            "map_mean_prec",
            (*mean, "-prec", "4"),
            map_dict,
            "per_base",
            ("per_base", "per_interval"),
        ),
        Template(
            "map_mean_null",
            (*mean, "-null", "0"),
            map_dict,
            "per_base",
            ("per_base", "per_interval"),
        ),
        Template(
            "map_mean_f", (*mean, "-f", "0.5"), map_dict, "per_base", ("per_base", "per_interval")
        ),
        Template(
            "map_mean_binned",
            ("map", "-a", win, "-b", "binned.bedgraph", "-c", "4", "-o", "mean"),
            map_dict,
            "per_base",
            ("per_base", "per_interval"),
            setup=_binned_track,
        ),
        Template(
            "map_sum",
            ("map", "-a", win, "-b", track, "-c", "4", "-o", "sum"),
            map_dict,
            "per_base",
            ("per_base",),
        ),
        Template(
            "map_max",
            ("map", "-a", win, "-b", track, "-c", "4", "-o", "max"),
            map_dict,
            "per_base",
            ("per_base",),
        ),
        Template(
            "intersect_wo",
            ("intersect", "-a", win, "-b", track, "-wo"),
            wo_weighted,
            "per_base",
            ("per_base",),
        ),
    )
    return emit("signal_weighting", "F66", seed, scratch, tools, templates)


# ---- F67 spliced_coverage ----------------------------------------------------------------------


def sum_col6(stdout: str, work: Path) -> int:
    return sum(int(r[5]) for r in lines(stdout))


def positive_depth_rows(stdout: str, work: Path) -> int:
    return sum(1 for r in lines(stdout) if int(r[-1]) > 0)


def bedgraph_in_introns(stdout: str, work: Path) -> int:
    introns = [r for r in lines((work / "introns.bed").read_text())]
    total = 0
    for r in lines(stdout):
        if float(r[3]) <= 0:
            continue
        for i in introns:
            if i[0] == r[0]:
                total += max(0, min(int(r[2]), int(i[2])) - max(int(r[1]), int(i[1])))
    return total


def per_base_in_introns(stdout: str, work: Path) -> int:
    """genomecov -d rows: (chrom, 1-based position, depth)."""
    introns = lines((work / "introns.bed").read_text())
    total = 0
    for r in lines(stdout):
        if int(r[2]) > 0:
            pos0 = int(r[1]) - 1
            total += any(i[0] == r[0] and int(i[1]) <= pos0 < int(i[2]) for i in introns)
    return total


@case_source("bedtools.spliced_without_split")
def spliced_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    cov = ("coverage", "-a", "introns.bed", "-b", "reads.bam")
    gc = ("genomecov", "-ibam", "reads.bam")
    wrong = ("exons", "span")
    templates = (
        Template("coverage", cov, sum_col6, "exons", wrong),
        Template("coverage_split", (*cov, "-split"), sum_col6, "exons", ("exons",)),
        Template("coverage_d", (*cov, "-d"), positive_depth_rows, "exons", wrong),
        Template(
            "coverage_d_split", (*cov, "-d", "-split"), positive_depth_rows, "exons", ("exons",)
        ),
        Template("genomecov_bg", (*gc, "-bg"), bedgraph_in_introns, "exons", wrong),
        Template(
            "genomecov_bg_split", (*gc, "-bg", "-split"), bedgraph_in_introns, "exons", ("exons",)
        ),
        Template("genomecov_d", (*gc, "-d"), per_base_in_introns, "exons", wrong),
        Template(
            "genomecov_d_split", (*gc, "-d", "-split"), per_base_in_introns, "exons", ("exons",)
        ),
    )
    return emit("spliced_reads", "F67", seed, scratch, tools, templates)


# ---- F68 subtract_trim -------------------------------------------------------------------------


def line_count(stdout: str, work: Path) -> int:
    return len(lines(stdout))


def unique_names(stdout: str, work: Path) -> int:
    return len({r[3] for r in lines(stdout)})


@case_source("bedtools.subtract_trims")
def subtract_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    sub = ("subtract", "-a", "peaks.bed", "-b", "blacklist.bed")
    wrong = ("remove", "trim")
    templates = (
        Template("subtract_lines", sub, line_count, "remove", wrong),
        Template("subtract_names", sub, unique_names, "remove", wrong),
        Template("subtract_A", (*sub, "-A"), line_count, "remove", ("remove",)),
        Template("subtract_N", (*sub, "-N"), line_count, "remove", ("remove",)),
        Template("subtract_A_f", (*sub, "-A", "-f", "0.5"), line_count, "remove", ("remove",)),
        Template("subtract_f", (*sub, "-f", "0.5"), line_count, "remove", wrong),
        Template(
            "intersect_v",
            ("intersect", "-a", "peaks.bed", "-b", "blacklist.bed", "-v"),
            line_count,
            "remove",
            ("remove",),
        ),
    )
    return emit("overlap_policy", "F68", seed, scratch, tools, templates)


# ---- F69 window_gap ----------------------------------------------------------------------------


def line_total(stdout: str, work: Path) -> int:
    return len(lines(stdout))


def window_c_positive(stdout: str, work: Path) -> int:
    return sum(1 for r in lines(stdout) if int(r[4]) > 0)


def closest_at_most(limit: int, signed: bool = False) -> Answer:
    def answer(stdout: str, work: Path) -> int:
        near = set()
        for r in lines(stdout):
            d = int(r[-1])
            d = abs(d) if signed else d
            if 0 <= d <= limit:
                near.add(r[3])
        return len(near)

    return answer


@case_source("bedtools.window_boundary")
def window_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    win = ("window", "-a", "peaks.bed", "-b", "genes.bed")
    wrong = ("gap_inclusive", "window_exclusive")
    templates = (
        Template("w1000_u", (*win, "-w", "1000", "-u"), line_total, "window_exclusive", wrong),
        Template(
            "w1000_c", (*win, "-w", "1000", "-c"), window_c_positive, "window_exclusive", wrong
        ),
        Template(
            "lr1000_u",
            (*win, "-l", "1000", "-r", "1000", "-u"),
            line_total,
            "window_exclusive",
            wrong,
        ),
        Template("w999_u", (*win, "-w", "999", "-u"), line_total, "window_exclusive", wrong),
        Template(
            "w1001_u",
            (*win, "-w", "1001", "-u"),
            line_total,
            "window_exclusive",
            ("window_exclusive",),
        ),
        Template(
            "w1001_c",
            (*win, "-w", "1001", "-c"),
            window_c_positive,
            "window_exclusive",
            ("window_exclusive",),
        ),
        Template(
            "lr1001_u",
            (*win, "-l", "1001", "-r", "1001", "-u"),
            line_total,
            "window_exclusive",
            ("window_exclusive",),
        ),
    )
    return emit("window_boundary", "F69", seed, scratch, tools, templates)


@case_source("bedtools.closest_distance_unit")
def closest_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    cl = ("closest", "-a", "peaks.bed", "-b", "genes.bed")
    wrong = ("bases_between", "bedtools_distance")
    templates = (
        Template("d_le1000", (*cl, "-d"), closest_at_most(1000), "bedtools_distance", wrong),
        Template(
            "d_le1001",
            (*cl, "-d"),
            closest_at_most(1001),
            "bedtools_distance",
            ("bedtools_distance",),
        ),
        Template(
            "d_t_first_le1000",
            (*cl, "-d", "-t", "first"),
            closest_at_most(1000),
            "bedtools_distance",
            wrong,
        ),
        Template(
            "d_t_first_le1001",
            (*cl, "-d", "-t", "first"),
            closest_at_most(1001),
            "bedtools_distance",
            ("bedtools_distance",),
        ),
        Template(
            "D_ref_le1000",
            (*cl, "-D", "ref"),
            closest_at_most(1000, True),
            "bedtools_distance",
            wrong,
        ),
        Template(
            "D_ref_le1001",
            (*cl, "-D", "ref"),
            closest_at_most(1001, True),
            "bedtools_distance",
            ("bedtools_distance",),
        ),
        Template(
            "d_io_le1001", (*cl, "-d", "-io"), closest_at_most(1001), "bedtools_distance", wrong
        ),
    )
    return emit("distance_unit", "F69", seed, scratch, tools, templates)


# ---- F75 multicov_dups -------------------------------------------------------------------------


def counts_col5(stdout: str, work: Path) -> dict[str, int]:
    return {r[3]: int(r[4]) for r in lines(stdout)}


def _nodup_bam(tools: ToolEnv, work: Path) -> None:
    proc = tools.run(
        ["samtools", "view", "-b", "-F", "1024", "-o", "nodup.bam", "sample.bam"], work
    )
    if proc.returncode == 0:
        tools.run(["samtools", "index", "nodup.bam"], work)


@case_source("bedtools.duplicate_policy")
def duplicate_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    cov = ("coverage", "-a", "regions.bed", "-b", "sample.bam", "-counts")
    wrong = ("exclude", "keep")
    templates = (
        Template("coverage_counts", cov, counts_col5, "keep", wrong),
        Template(
            "intersect_c",
            ("intersect", "-a", "regions.bed", "-b", "sample.bam", "-c"),
            counts_col5,
            "keep",
            wrong,
        ),
        Template("coverage_counts_f", (*cov, "-f", "0.000001"), counts_col5, "keep", wrong),
        Template(
            "multicov",
            ("multicov", "-bams", "sample.bam", "-bed", "regions.bed"),
            counts_col5,
            "exclude",
            ("exclude",),
        ),
        Template(
            "multicov_q",
            ("multicov", "-q", "10", "-bams", "sample.bam", "-bed", "regions.bed"),
            counts_col5,
            "exclude",
            ("exclude",),
        ),
        Template(
            "multicov_D",
            ("multicov", "-D", "-bams", "sample.bam", "-bed", "regions.bed"),
            counts_col5,
            "keep",
            wrong,
        ),
        Template(
            "coverage_nodup",
            ("coverage", "-a", "regions.bed", "-b", "nodup.bam", "-counts"),
            counts_col5,
            "exclude",
            ("exclude",),
            setup=_nodup_bam,
        ),
    )
    return emit("duplicate_policy", "F75", seed, scratch, tools, templates)


# ---- bedtools.merge_bookended (a small task of its own) -----------------------------------------


def _merge_task(seed: int, twin: str, work: Path) -> dict[str, Any]:
    """Regions made of intervals that share a base; some regions abut the next one (faulted) or
    sit 20-60 bases from it (clean). Truth: the number of regions."""
    rng = random.Random(f"merge-{seed}")
    rows: list[tuple[str, int, int, str]] = []
    regions = 0
    for chrom in ("chr1", "chr2"):
        cursor = 1000
        for _ in range(rng.randint(8, 11)):
            start = cursor + rng.randint(150, 400)
            for _ in range(rng.randint(1, 3)):
                end = start + rng.randint(80, 200)
                rows.append((chrom, start, end, f"i{len(rows):03d}"))
                start = end - rng.randint(5, 30)
            regions += 1
            last_end = rows[-1][2]
            if rng.random() < 0.4:  # a neighbouring region
                gap = 0 if twin == "faulted" else rng.randint(20, 60)
                start = last_end + gap
                end = start + rng.randint(80, 200)
                rows.append((chrom, start, end, f"i{len(rows):03d}"))
                regions += 1
                last_end = end
            cursor = last_end
    (work / "x.bed").write_text("".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in rows))
    return {"kind": "int", "expected": regions}


def count_lines(stdout: str, work: Path) -> int:
    return len(lines(stdout))


@case_source("bedtools.merge_bookended")
def merge_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    mg = ("merge", "-i", "x.bed")
    wrong = ("keep", "merge")
    templates = (
        Template("merge", mg, count_lines, "keep", wrong),
        Template("merge_d0", (*mg, "-d", "0"), count_lines, "keep", wrong),
        Template("merge_count", (*mg, "-c", "4", "-o", "count"), count_lines, "keep", wrong),
        Template("merge_collapse", (*mg, "-c", "4", "-o", "collapse"), count_lines, "keep", wrong),
        Template("merge_dneg", (*mg, "-d", "-1"), count_lines, "keep", ("keep",)),
        Template("merge_d1", (*mg, "-d", "1"), count_lines, "keep", wrong),
        Template("merge_d50", (*mg, "-d", "50"), count_lines, "keep", wrong),
    )
    return emit("adjacent_policy", "merge", seed, scratch, tools, templates, builder=_merge_task)


# ---- bedtools.groupby_unsorted (a small task of its own) ----------------------------------------


def _groupby_task(seed: int, twin: str, work: Path) -> dict[str, Any]:
    """Scores of records per gene name; the file is grouped by name (clean) or by position."""
    rng = random.Random(f"groupby-{seed}")
    names = [f"gene{k:02d}" for k in range(rng.randint(6, 9))]
    rows = []
    for name in names:
        for _ in range(rng.randint(3, 6)):
            chrom = rng.choice(["chr1", "chr2"])
            start = rng.randint(1000, 90000)
            rows.append((chrom, start, start + rng.randint(50, 400), name, rng.randint(1, 60)))
    totals: dict[str, float] = {}
    for r in rows:
        totals[r[3]] = totals.get(r[3], 0) + r[4]
    key = (lambda r: (r[3], r[0], r[1])) if twin == "clean" else (lambda r: (r[0], r[1]))
    rows.sort(key=key)
    (work / "x.bed").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    return {"kind": "dict_num", "expected": totals, "tol": 0.5}


def _sorted_by_name(tools: ToolEnv, work: Path) -> None:
    rows = sorted(
        (line.split("\t") for line in (work / "x.bed").read_text().splitlines()),
        key=lambda r: (r[3], r[0], int(r[1])),
    )
    (work / "by_name.bed").write_text("".join("\t".join(r) + "\n" for r in rows))


def name_sums(stdout: str, work: Path) -> dict[str, float]:
    return {r[0]: float(r[1]) for r in lines(stdout)}


@case_source("bedtools.groupby_unsorted")
def groupby_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    gb = ("groupby", "-g", "4", "-c", "5", "-o", "sum")
    templates = (
        Template(
            "groupby", ("groupby", "-i", "x.bed", "-g", "4", "-c", "5", "-o", "sum"), name_sums
        ),
        Template(
            "groupby_ops",
            ("groupby", "-i", "x.bed", "-grp", "4", "-opCols", "5", "-ops", "sum"),
            name_sums,
        ),
        Template("groupby_prec", ("groupby", "-i", "x.bed", *gb[1:], "-prec", "2"), name_sums),
        Template(
            "groupby_sorted",
            ("groupby", "-i", "by_name.bed", *gb[1:]),
            name_sums,
            setup=_sorted_by_name,
        ),
        Template(
            "groupby_ignorecase_sorted",
            ("groupby", "-i", "by_name.bed", *gb[1:], "-ignorecase"),
            name_sums,
            setup=_sorted_by_name,
        ),
    )
    return emit(None, "groupby", seed, scratch, tools, templates, builder=_groupby_task)


# ---- bedtools.prec_significant (a small task of its own) ----------------------------------------


def _prec_task(seed: int, twin: str, work: Path) -> dict[str, Any]:
    """Sum of a bedGraph value per window, to 3 decimals; the values are large (1000-9999)."""
    rng = random.Random(f"prec-{seed}")
    windows, lines_out, totals = [], [], {}
    for k in range(rng.randint(6, 9)):
        start = k * 1000
        windows.append(f"chr1\t{start}\t{start + 1000}\tw{k:02d}\n")
        values = [round(rng.uniform(1000, 9999), 5) for _ in range(rng.randint(2, 4))]
        for j, v in enumerate(values):
            lines_out.append(f"chr1\t{start + j * 200}\t{start + j * 200 + 100}\t{v}\n")
        totals[f"w{k:02d}"] = round(sum(values), 3)
    (work / "win.bed").write_text("".join(windows))
    (work / "vals.bedgraph").write_text("".join(lines_out))
    return {"kind": "dict_num", "expected": totals, "tol": 0.0006}


@case_source("bedtools.prec_significant")
def prec_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    mp = ("map", "-a", "win.bed", "-b", "vals.bedgraph", "-c", "4", "-o", "sum")
    wrong = ("decimal_places", "significant_digits")
    templates = (
        Template("map_prec3", (*mp, "-prec", "3"), map_dict, "significant_digits", wrong),
        Template("map_prec4", (*mp, "-prec", "4"), map_dict, "significant_digits", wrong),
        Template("map_prec7", (*mp, "-prec", "7"), map_dict, "significant_digits", wrong),
        Template("map_prec10", (*mp, "-prec", "10"), map_dict, "significant_digits", wrong),
        Template("map_noprec", mp, map_dict, "significant_digits", wrong),
        Template(
            "merge_prec2",
            ("merge", "-i", "vals.bedgraph", "-c", "4", "-o", "sum", "-prec", "2"),
            map_dict,
            "significant_digits",
            wrong,
        ),
    )
    return emit("precision_unit", "prec", seed, scratch, tools, templates, builder=_prec_task)
