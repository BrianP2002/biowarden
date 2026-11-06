"""Admission cases for the deepTools rules (pack iv). Labels come from running the real tools.

deeptools.matrix_missing_data (F73 workspaces, both twins). The task is the mean coverage per bin
over all gene bodies. Each computeMatrix call is run in the workspace and its table is turned into
the answer the way an agent does (mean of the cells that are not NaN, which numpy `nanmean` and
pandas `mean` give); the label compares it with the truth. Without `--missingDataAsZero` the call
is right on the dense twin and wrong on the sparse one; with it, right; `--skipZeros` drops the
silent genes (wrong on both twins).

deeptools.matrix_skip_zeros (F73 workspaces). `--skipZeros` over the full gene list is wrong on
both twins (the silent genes leave the matrix). To have right calls with `--skipZeros`, a second
task is built in the same workspace: the mean over the genes that have reads only, where the BED
has no silent gene (`genes_expressed.bed`) and `--skipZeros` removes nothing.

deeptools.bamcoverage_track_units (F76 faulted workspace). The task is the mean per-base depth of
each region (every read counts, flags ignored); the bedGraph written by bamCoverage is averaged
over each region weighted by length. Only `--binSize 1` without normalisation is right.

deeptools.duplicates_semantics (F76 workspaces, both twins). multiBamSummary BED-file counts per
region against the truth (reads overlapping the region, flag 0x400 left out): `--samFlagExclude
1024` is right on both twins, `--ignoreDuplicates` only on the clean one, a call with no duplicate
option is wrong on both.

deeptools.multibamsummary_regions (F76 faulted workspace). The BED is sorted (right), listed in
reverse (rows out of order) or holds a row on a contig the BAM does not have. A call without
declaration is read the way an agent reads the table without names: rows zipped with the BED names
in file order. A call that declares `region_join=coordinates` joins on chr/start/end, so a shuffled
BED is right and a missing contig leaves the region without a count (wrong).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f73_matrix_missing as f73
from biowarden.bench.tasks.extra import f76_ignore_dups as f76
from biowarden.engine import Call

WORKERS = 2


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    meta: tuple[tuple[str, str], ...] = ()


def _label(ok: bool) -> Label:
    return "right" if ok else "wrong"


def _emit(
    out: list[Case],
    template: Template,
    work: Path,
    label: Label,
    family: str,
    seed: int,
    twin: str,
    declarations: list[dict[str, str]],
) -> None:
    out.append(
        Case(
            Call(template.tool, template.argv, work),
            label,
            False,
            family,
            seed,
            twin,
            template.name,
        )
    )
    for decl in declarations:
        call = Call(template.tool, template.argv, work, decl)
        out.append(Case(call, label, True, family, seed, twin, f"{template.name}:{decl}"))


def _parallel(fn, items):
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, items))


def _run(tools: ToolEnv, work: Path, template: Template):
    proc = tools.run([template.tool, *template.argv], work)
    if proc.returncode != 0:
        raise RuntimeError(f"{template.name}: {proc.stderr}")
    return proc


# ---- computeMatrix -------------------------------------------------------------------


def matrix_template(
    name: str,
    sub: str,
    bed: str = "genes.bed",
    bin_option: str = "--binSize",
    bin_size: str = "50",
    extra: tuple[str, ...] = (),
) -> Template:
    head = ["computeMatrix", sub, "-S", "signal.bw", "-R", bed]
    if sub == "scale-regions":
        geometry = ["-b", "0", "-a", "0", "--regionBodyLength", str(f73.GENE_LEN)]
    else:
        geometry = ["--referencePoint", "TSS", "-b", "0", "-a", str(f73.GENE_LEN)]
    out = ["-p", "1", "-o", f"{name}.gz", "--outFileNameMatrix", f"{name}.tab"]
    argv = [*head[1:], *geometry, bin_option, bin_size, *out, *extra]
    return Template(name, "computeMatrix", tuple(argv))


MATRIX_TEMPLATES = (
    matrix_template("sr_plain", "scale-regions"),
    matrix_template("sr_zero", "scale-regions", extra=("--missingDataAsZero",)),
    matrix_template("rp_plain", "reference-point", bin_option="-bs"),
    matrix_template("rp_zero", "reference-point", bin_option="-bs", extra=("--missingDataAsZero",)),
    matrix_template("sr_skip", "scale-regions", extra=("--skipZeros", "--missingDataAsZero")),
)


def matrix_answer(work: Path, name: str) -> float:
    return f73.mean_skipping_nan(f73.parse_matrix((work / f"{name}.tab").read_text()))


@case_source("deeptools.matrix_missing_data")
def matrix_missing_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F73", seed, twin, work, tools, "implicit", "free")
        _parallel(partial(_run, tools, work), MATRIX_TEMPLATES)
        for template in MATRIX_TEMPLATES:
            ok = answer_matches(task.truth, matrix_answer(work, template.name))
            flag = "--missingDataAsZero" in template.argv
            if ok and flag:
                decls = [{"missing_signal": "zero"}]
            elif ok:
                decls = [{"missing_signal": "zero"}, {"missing_signal": "na"}]
            elif flag:
                decls = [{"missing_signal": "zero"}]
            else:
                decls = [{"missing_signal": "na"}, {"missing_signal": "zero"}]
            _emit(out, template, work, _label(ok), "F73", seed, twin, decls)
    return out


@case_source("deeptools.matrix_skip_zeros")
def matrix_skip_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F73", seed, twin, work, tools, "implicit", "free")
        plan = f73.make_plan(seed)
        kept = [g for g in plan.genes if any(g.bins)]
        (work / "genes_expressed.bed").write_text(
            "".join(
                f"{g.chrom}\t{g.start}\t{g.start + f73.GENE_LEN}\t{g.name}\t0\t{g.strand}\n"
                for g in kept
            )
        )
        cells = [v for g in kept for v in g.bins]
        expressed = {"kind": "float", "expected": sum(cells) / len(cells), "tol": f73.TOL}
        templates = [
            (
                matrix_template(
                    "full_sr", "scale-regions", extra=("--skipZeros", "--missingDataAsZero")
                ),
                task.truth,
            ),
            (
                matrix_template(
                    "full_rp", "reference-point", extra=("--skipZeros", "--missingDataAsZero")
                ),
                task.truth,
            ),
            (
                matrix_template(
                    "kept_sr",
                    "scale-regions",
                    "genes_expressed.bed",
                    extra=("--skipZeros", "--missingDataAsZero"),
                ),
                expressed,
            ),
            (
                matrix_template(
                    "kept_rp",
                    "reference-point",
                    "genes_expressed.bed",
                    extra=("--skipZeros", "--missingDataAsZero"),
                ),
                expressed,
            ),
        ]
        _parallel(partial(_run, tools, work), [t for t, _ in templates])
        for template, truth in templates:
            ok = answer_matches(truth, matrix_answer(work, template.name))
            decls = (
                [{"zero_regions": "drop"}]
                if ok
                else [{"zero_regions": "drop"}, {"zero_regions": "keep"}]
            )
            _emit(out, template, work, _label(ok), "F73", seed, twin, decls)
    return out


# ---- bamCoverage ---------------------------------------------------------------------


def coverage_template(name: str, *extra: str) -> Template:
    argv = ["-b", "reads.bam", "-o", f"{name}.bg", "-of", "bedgraph", "-p", "1", *extra]
    return Template(name, "bamCoverage", tuple(argv))


COVERAGE_TEMPLATES = (
    coverage_template("bs1", "--binSize", "1"),
    coverage_template("bs1_short", "-bs", "1"),
    coverage_template("default"),
    coverage_template("bs10", "--binSize", "10"),
    coverage_template("cpm_bs1", "--binSize", "1", "--normalizeUsing", "CPM"),
    coverage_template("dedup_bs1", "--binSize", "1", "--ignoreDuplicates"),
)


def depth_truth(plan: f76.Plan) -> dict[str, float]:
    out = {}
    for chrom, start, end, name in plan.regions:
        bases = 0
        for read in plan.reads:
            if read.chrom == chrom:
                bases += max(0, min(read.end, end) - max(read.start, start))
        out[name] = bases / (end - start)
    return out


def bedgraph_means(path: Path, plan: f76.Plan) -> dict[str, float]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    out = {}
    for chrom, start, end, name in plan.regions:
        total = 0.0
        for r in rows:
            if r[0] == chrom:
                overlap = min(int(r[2]), end) - max(int(r[1]), start)
                if overlap > 0:
                    total += overlap * float(r[3])
        out[name] = total / (end - start)
    return out


@case_source("deeptools.bamcoverage_track_units")
def track_units_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "faulted"
    build_task("F76", seed, "faulted", work, tools, "implicit", "free")
    plan = f76.make_plan(seed, "faulted")
    truth = {"kind": "dict_num", "expected": depth_truth(plan), "tol": 0.05}
    _parallel(partial(_run, tools, work), COVERAGE_TEMPLATES)
    out: list[Case] = []
    for template in COVERAGE_TEMPLATES:
        ok = answer_matches(truth, bedgraph_means(work / f"{template.name}.bg", plan))
        own = _own_unit(template)
        decls = [{"track_unit": "depth"}] if ok else [{"track_unit": own}, {"track_unit": "depth"}]
        _emit(out, template, work, _label(ok), "F76", seed, "faulted", decls)
    return out


def _own_unit(template: Template) -> str:
    argv = template.argv
    if "--normalizeUsing" in argv:
        return "normalized"
    if "--binSize" in argv and argv[argv.index("--binSize") + 1] == "1":
        return "depth"
    if "-bs" in argv and argv[argv.index("-bs") + 1] == "1":
        return "depth"
    return "reads_per_bin"


# ---- multiBamSummary -----------------------------------------------------------------


def summary_template(name: str, bed: str, *extra: str) -> Template:
    argv = [
        "BED-file",
        "--BED",
        bed,
        "-b",
        "reads.bam",
        "-p",
        "1",
        "-o",
        f"{name}.npz",
        "--outRawCounts",
        f"{name}.tab",
        *extra,
    ]
    return Template(name, "multiBamSummary", tuple(argv))


def table_rows(work: Path, name: str) -> list[tuple[str, int, int, int]]:
    rows = []
    for line in (work / f"{name}.tab").read_text().splitlines():
        if not line.startswith("#"):
            f = line.split("\t")
            rows.append((f[0], int(f[1]), int(f[2]), int(float(f[3]))))
    return rows


DUP_TEMPLATES = (
    summary_template("ignore", "regions.bed", "--ignoreDuplicates"),
    summary_template("flag", "regions.bed", "--samFlagExclude", "1024"),
    summary_template("plain", "regions.bed"),
    summary_template("both", "regions.bed", "--ignoreDuplicates", "--samFlagExclude", "1024"),
)


@case_source("deeptools.duplicates_semantics")
def duplicates_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F76", seed, twin, work, tools, "implicit", "free")
        _parallel(partial(_run, tools, work), DUP_TEMPLATES)
        names = f76.region_names(work)
        for template in DUP_TEMPLATES:
            answer = {names[(c, s, e)]: n for c, s, e, n in table_rows(work, template.name)}
            ok = answer_matches(task.truth, answer)
            decls = (
                [{"duplicates": "exclude"}]
                if ok
                else [{"duplicates": "exclude"}, {"duplicates": "keep"}]
            )
            _emit(out, template, work, _label(ok), "F76", seed, twin, decls)
    return out


@case_source("deeptools.multibamsummary_regions")
def regions_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "faulted"
    task = build_task("F76", seed, "faulted", work, tools, "implicit", "free")
    rows = (work / "regions.bed").read_text().splitlines()
    (work / "regions_reversed.bed").write_text("\n".join(reversed(rows)) + "\n")
    extra_row = "chrX\t100\t300\tregionX"
    (work / "regions_absent.bed").write_text("\n".join([*rows[:2], extra_row, *rows[2:]]) + "\n")
    expected = {**task.truth["expected"], "regionX": 0}
    truths = {
        "regions.bed": task.truth,
        "regions_reversed.bed": task.truth,
        "regions_absent.bed": {**task.truth, "expected": expected},
    }
    templates = [
        summary_template(bed.removesuffix(".bed"), bed, "--samFlagExclude", "1024")
        for bed in truths
    ]
    _parallel(partial(_run, tools, work), templates)
    out: list[Case] = []
    for template, (bed, truth) in zip(templates, truths.items(), strict=True):
        listed = [line.split("\t") for line in (work / bed).read_text().splitlines()]
        table = table_rows(work, template.name)
        by_line = {listed[i][3]: table[i][3] for i in range(min(len(listed), len(table)))}
        by_coords = {}
        for chrom, start, end, count in table:
            for r in listed:
                if (r[0], int(r[1]), int(r[2])) == (chrom, start, end):
                    by_coords[r[3]] = count
        ok_line = answer_matches(truth, by_line)
        ok_coords = answer_matches(truth, by_coords)
        _emit(out, template, work, _label(ok_line), "F76", seed, "faulted", [])
        decl = {"region_join": "coordinates"}
        call = Call(template.tool, template.argv, work, decl)
        out.append(Case(call, _label(ok_coords), True, "F76", seed, "faulted", template.name))
    return out
