"""Admission cases for the MACS3 rules (pack iv). Labels come from running macs3 / bedtools.

Four case sources, each with a small task whose answer the real tool decides:

macs3.paired_end_format   F72 workspaces (paired-end BAM, binding sites 215-250 bases apart in the
    faulted twin, 400-460 in the clean one). The answer is the total width of the called peaks
    in bases, compared with the width `-f BAMPE` gives. The count of peaks (the F72 answer) is
    not used for labels because on the clean twin it survives the single-end mode, while the peak
    boundaries never do (a paired-end BAM read as single-end tags is analysed wrongly even when
    the count happens to agree). Right: BAMPE, with or without the options BAMPE ignores. Wrong:
    AUTO, BAM with --nomodel --extsize 200 / 147, BAM without --nomodel (stops, exit 1).
macs3.effective_genome_size   a 180 kb genome (3 contigs), four strong peaks on chr1 where the
    control has no reads (so lambda is the genome-wide background), treatment and control BAM.
    The answer is the -log10(q) of the strongest peak (tolerance 0.5). Right: the exact total as
    `-g`; wrong: no -g (hs), hs, mm, ce, 2.7e9, 1e6, 360000 (a factor 2, which the rule does not
    flag: a missed call, not a false block).
macs3.extsize_needs_nomodel   1 Mb single-end data with 140 sharp peaks, where the fragment model
    builds. The answer is the total peak width with a fixed fragment length of 150 (`--nomodel
    --extsize 150`). Wrong: `--extsize 150` alone (the predicted length is used), `--nomodel
    --extsize 200`, `--extsize 150 --shift -75`.
macs3.peaks_xls_start   a real MACS run (F72 faulted data, `-f BAMPE`) leaves run_peaks.xls,
    run_peaks.narrowPeak and targets.bed (one-base targets on a peak's first base, just before
    it, inside it, just after it). The answer is the set of overlapping targets according to the
    narrowPeak. Wrong: the xls start fed to bedtools / python as a BED start. Right: start - 1,
    or the narrowPeak.

macs3.control_contigs   the same 180 kb data; a second control BAM has no read on chr1, where all
    four peaks lie. The answer is the number of peaks of the call with the full control. Right:
    `-c control.bam`, `--control control.bam`, no control; wrong: the control without chr1 reads
    (MACS3 writes 0 peaks).

Each call is emitted without a declaration and with one. Rules without a declaration (the genome
size rule) emit every call in both cells.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.tasks import ToolEnv, build_task, common
from science_harness.bench.tasks.extra import f72_macs3_bampe as f72
from science_harness.engine import Call

WORKERS = 2


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    kind: str = ""  # what the call assumes, for the declaration
    inspection: bool = False


def _peaks(tools: ToolEnv, work: Path, argv: tuple[str, ...], tag: str):
    """Rows of the narrowPeak of `macs3 <argv>` run into a scratch directory; None on failure."""
    out = work.parent / f"out-{work.name}-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    proc = tools.run(["macs3", *argv, "--outdir", str(out)], work)
    if proc.returncode != 0:
        return None
    name = argv[argv.index("-n") + 1]
    text = (out / f"{name}_peaks.narrowPeak").read_text()
    return [line.split("\t") for line in text.splitlines() if line.strip()]


def width(rows) -> int | None:
    return None if rows is None else sum(int(r[2]) - int(r[1]) for r in rows)


def top_q(rows) -> float | None:
    return None if rows is None or not rows else max(float(r[8]) for r in rows)


def _run_all(tools: ToolEnv, work: Path, templates, metric):
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(
            pool.map(lambda t: metric(_peaks(tools, work, t.argv, t.name)), templates),
        )


def _emit(
    out: list[Case],
    work: Path,
    template: Template,
    label: Label,
    family: str,
    seed: int,
    twin: str,
    declarations: list[dict[str, str]],
) -> None:
    call = Call(template.tool, template.argv, work)
    out.append(Case(call, label, False, family, seed, twin, template.name))
    for declaration in declarations or [{}]:
        decl = Call(template.tool, template.argv, work, declaration)
        out.append(Case(decl, label, True, family, seed, twin, template.name))


# ---- macs3.paired_end_format -------------------------------------------------------------------


def _pe_templates(size: int) -> tuple[Template, ...]:
    base = ("callpeak", "-t", "reads.bam", "-g", str(size), "-q", f2(), "-n", "x")
    return (
        Template("bampe", "macs3", (*base, "-f", "BAMPE"), "fragments"),
        Template(
            "bampe_ignored_options",
            "macs3",
            (*base, "--format=BAMPE", "--nomodel", "--extsize", "200"),
            "fragments",
        ),
        Template("auto_ext200", "macs3", (*base, "--nomodel", "--extsize", "200"), "single"),
        Template(
            "bam_ext147", "macs3", (*base, "-f", "BAM", "--nomodel", "--extsize", "147"), "single"
        ),
        Template("bam_model", "macs3", (*base, "-f", "BAM"), "single"),
    )


def f2() -> str:
    return f72.QVALUE


@case_source("macs3.paired_end_format")
def paired_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin, keep in (("faulted", None), ("clean", {"bampe", "auto_ext200"})):
        work = scratch / twin
        build_task("F72", seed, twin, work, tools, "implicit", "free")
        size = f72.genome_size(work)
        templates = [t for t in _pe_templates(size) if keep is None or t.name in keep]
        widths = _run_all(tools, work, templates, width)
        truth = widths[[t.name for t in templates].index("bampe")]
        for template, got in zip(templates, widths, strict=True):
            label: Label = "right" if got is not None and got == truth else "wrong"
            wanted = ["fragments"] if label == "right" else ["fragments", "single_end_tags"]
            decls = [{"read_layout": v} for v in wanted]
            _emit(out, work, template, label, "F72", seed, twin, decls)
    return out


# ---- macs3.effective_genome_size ---------------------------------------------------------------

G_CONTIGS = {"chr1": 60_000, "chr2": 60_000, "chr3": 60_000}


def build_gsize(seed: int, work: Path, tools: ToolEnv) -> int:
    """Treatment with four sharp peaks on chr1; the control has no reads near them."""
    rnd = random.Random(f"iv-macs3-g-{seed}")
    header = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in G_CONTIGS.items()
    )

    def sam(rows) -> str:
        order = {c: i for i, c in enumerate(G_CONTIGS)}
        rows = sorted(rows, key=lambda r: (order[r[0]], r[1]))
        return header + "".join(
            f"{n}\t{fl}\t{c}\t{p}\t60\t50M\t*\t0\t0\t{'A' * 50}\t{'I' * 50}\n"
            for c, p, fl, n in rows
        )

    treat: list[tuple[str, int, int, str]] = []
    for centre in sorted(rnd.sample(range(5000, 55000, 9000), 4)):
        for _ in range(rnd.randint(70, 90)):
            start = centre + round(rnd.gauss(0, 30)) - 100
            minus = rnd.random() < 0.5
            treat.append(
                ("chr1", max(1, start + 150 if minus else start), 16 * minus, f"t{len(treat)}")
            )
    for k in range(150):
        treat.append(
            (rnd.choice(["chr2", "chr3"]), rnd.randint(1, 59_900), rnd.choice([0, 16]), f"b{k}")
        )
    control = [
        (rnd.choice(["chr2", "chr3"]), rnd.randint(1, 59_900), rnd.choice([0, 16]), f"c{k}")
        for k in range(400)
    ]
    control += [("chr1", 59_000 + 100 * i, 0, f"x{i}") for i in range(3)]
    common.write_bam(work, sam(treat), tools, "treat")
    common.write_bam(work, sam(control), tools, "control")
    common.write_bam(work, sam([r for r in control if r[0] != "chr1"]), tools, "control_nochr1")
    (work / "genome.sizes").write_text("".join(f"{c}\t{n}\n" for c, n in G_CONTIGS.items()))
    return sum(G_CONTIGS.values())


def _g_templates(size: int) -> tuple[Template, ...]:
    base = (
        "callpeak", "-t", "treat.bam", "-c", "control.bam", "-f", "BAM", "--nomodel",
        "--extsize", "200", "-q", f2(), "-n", "x",
    )  # fmt: skip

    def with_g(name: str, *g: str, kind: str = "wrong") -> Template:
        return Template(name, "macs3", (*base, *g), kind)

    return (
        with_g("g_exact", "-g", str(size), kind="right"),
        with_g("gsize_long", "--gsize", str(size), kind="right"),
        with_g("g_absent"),
        with_g("g_hs", "-g", "hs"),
        with_g("g_mm", "-g", "mm"),
        with_g("g_ce", "-g", "ce"),
        with_g("g_human_number", "-g", "2.7e9"),
        with_g("g_1e6", "-g", "1e6"),
        with_g("g_double", "-g", str(2 * size)),
    )


@case_source("macs3.effective_genome_size")
def gsize_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "g"
    work.mkdir(parents=True, exist_ok=True)
    size = build_gsize(seed, work, tools)
    templates = _g_templates(size)
    scores = _run_all(tools, work, templates, top_q)
    truth = scores[0]
    out: list[Case] = []
    for template, got in zip(templates, scores, strict=True):
        label: Label = "right" if got is not None and truth and abs(got - truth) <= 0.5 else "wrong"
        _emit(out, work, template, label, "iv-macs3-gsize", seed, "genome", [])
    # the rule has no declaration: every call also fills the declared cell
    return out


# ---- macs3.extsize_needs_nomodel ---------------------------------------------------------------


def build_model_data(seed: int, work: Path, tools: ToolEnv) -> int:
    """1 Mb single-end data, 140 sharp peaks of 60 fragments (200-300 bases): the model builds."""
    rnd = random.Random(f"iv-macs3-model-{seed}")
    size = 1_000_000
    fragment = rnd.randint(210, 290)
    rows: list[tuple[int, int, str]] = []
    for centre in sorted(rnd.sample(range(2000, 990_000, 6000), 140)):
        for _ in range(60):
            length = max(80, round(rnd.gauss(fragment, 15)))
            start = centre + round(rnd.gauss(0, 25)) - length // 2
            minus = rnd.random() < 0.5
            rows.append(
                (max(1, start + length - 50 if minus else start), 16 * minus, f"p{len(rows)}")
            )
    rows += [(rnd.randint(1, size - 100), rnd.choice([0, 16]), f"b{k}") for k in range(500)]
    rows.sort()
    text = f"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:{size}\n" + "".join(
        f"{n}\t{fl}\tchr1\t{p}\t60\t50M\t*\t0\t0\t{'A' * 50}\t{'I' * 50}\n" for p, fl, n in rows
    )
    common.write_bam(work, text, tools, "model")
    return size


def _ext_templates(size: int) -> tuple[Template, ...]:
    base = ("callpeak", "-t", "model.bam", "-g", str(size), "-q", f2(), "-n", "x")

    def make(name: str, kind: str, *opts: str) -> Template:
        return Template(name, "macs3", (*base, *opts), kind)

    return (
        make("nomodel_150", "fixed", "-f", "BAM", "--nomodel", "--extsize", "150"),
        make("auto_nomodel_150", "fixed", "--nomodel", "--extsize", "150"),
        make("nomodel_150_equals", "fixed", "--nomodel", "--extsize=150"),
        make("model_ignores_150", "fixed", "-f", "BAM", "--extsize", "150"),
        make("auto_ignores_150", "fixed", "--extsize", "150"),
        make("nomodel_200", "fixed", "--nomodel", "--extsize", "200"),
        make("shift_without_nomodel", "fixed", "--extsize", "150", "--shift", "-75"),
    )


@case_source("macs3.extsize_needs_nomodel")
def extsize_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "m"
    work.mkdir(parents=True, exist_ok=True)
    size = build_model_data(seed, work, tools)
    templates = _ext_templates(size)
    widths = _run_all(tools, work, templates, width)
    truth = widths[0]
    out: list[Case] = []
    for template, got in zip(templates, widths, strict=True):
        label: Label = "right" if got is not None and got == truth else "wrong"
        decls = [{"fragment_length": "fixed"}]
        if label == "wrong" and "--nomodel" not in template.argv:
            decls.append({"fragment_length": "model"})  # accepts the predicted length
        _emit(out, work, template, label, "iv-macs3-extsize", seed, "model", decls)
    return out


# ---- macs3.peaks_xls_start ---------------------------------------------------------------------

STRIP = "grep -v '^#' run_peaks.xls | awk '$2 ~ /^[0-9]+$/'"
INTERSECT = "bedtools intersect -a targets.bed -b - -u | cut -f4"
PY_HEAD = (
    "import sys\n"
    "rows = [l.rstrip('\\n').split('\\t') for l in open('run_peaks.xls')]\n"
    "rows = [r for r in rows if r[0][:1] != '#' and len(r) > 3]\n"
    "rows = [r for r in rows if r[1].isdigit()]\n"
    "targets = [l.split() for l in open('targets.bed')]\n"
)
PY_TAIL = (
    "for t in targets:\n"
    "    if any(r[0] == t[0] and int(t[1]) < int(r[2]) and int(t[2]) > S for r in rows):\n"
    "        print(t[3])\n"
)


def _py(start: str) -> str:
    # S is replaced inline: the loop above uses a per-row start, so build the test explicitly
    body = PY_TAIL.replace("S for r in rows", f"{start} for r in rows")
    return "python3 - <<'EOF'\n" + PY_HEAD + body + "EOF"


def _sh(name: str, command: str, kind: str, inspection: bool = False) -> Template:
    return Template(name, "bash", ("-c", command), kind, inspection)


XLS_TEMPLATES: tuple[Template, ...] = (
    _sh("cut_raw_start", f"{STRIP} | cut -f1-3 | {INTERSECT}", "zero_based"),
    _sh(
        "awk_raw_start",
        "awk 'BEGIN{OFS=\"\\t\"} !/^#/ && $2 ~ /^[0-9]+$/ {print $1,$2,$3}' run_peaks.xls | "
        + INTERSECT,
        "zero_based",
    ),
    _sh("python_raw_start", _py("int(r[1])"), "zero_based"),
    _sh(
        "awk_minus_one",
        f"{STRIP} | awk 'BEGIN{{OFS=\"\\t\"}} {{print $1,$2-1,$3}}' | {INTERSECT}",
        "one_based",
    ),
    _sh(
        "awk_filter_minus_one",
        "awk 'BEGIN{OFS=\"\\t\"} !/^#/ && $2 ~ /^[0-9]+$/ {print $1,$2-1,$3}' run_peaks.xls | "
        + INTERSECT,
        "one_based",
    ),
    _sh("python_minus_one", _py("int(r[1]) - 1"), "one_based"),
    _sh(
        "narrowpeak",
        "bedtools intersect -a targets.bed -b run_peaks.narrowPeak -u | cut -f4",
        "one_based",
    ),
    _sh("head_xls", "grep -v '^#' run_peaks.xls | head -3", "none", True),
    _sh("count_xls", "grep -vc '^#' run_peaks.xls", "none", True),
)


def build_xls(seed: int, work: Path, tools: ToolEnv) -> set[str]:
    """Run MACS3 on the F72 faulted data and write targets.bed; return the overlapping names."""
    build_task("F72", seed, "faulted", work, tools, "implicit", "free")
    size = f72.genome_size(work)
    proc = tools.run(
        ["macs3", "callpeak", "-t", "reads.bam", "-f", "BAMPE", "-g", str(size), "-q", f2()]
        + ["-n", "run", "--outdir", "."],
        work,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-400:])
    peaks = [
        line.split("\t")
        for line in (work / "run_peaks.narrowPeak").read_text().splitlines()
        if line.strip()
    ]
    targets: list[tuple[str, int, int, str]] = []
    expected: set[str] = set()
    for i, row in enumerate(peaks):
        chrom, s, e = row[0], int(row[1]), int(row[2])
        kind = i % 4
        lo, hi = [(s, s + 1), (s - 1, s), ((s + e) // 2, (s + e) // 2 + 1), (e, e + 1)][kind]
        name = f"tg{i:02d}"
        targets.append((chrom, lo, hi, name))
        if kind in (0, 2):
            expected.add(name)
    (work / "targets.bed").write_text("".join(f"{c}\t{a}\t{b}\t{n}\n" for c, a, b, n in targets))
    return expected


@case_source("macs3.peaks_xls_start")
def xls_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "x"
    expected = build_xls(seed, work, tools)
    out: list[Case] = []
    for template in XLS_TEMPLATES:
        proc = tools.run([template.tool, *template.argv], work)
        if template.inspection:
            label: Label = "right"
        else:
            names = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
            label = "right" if proc.returncode == 0 and names == expected else "wrong"
        truth = "one_based"
        wanted = [truth]
        if label == "wrong" and template.kind == "zero_based":
            wanted.append("zero_based")
        _emit(
            out,
            work,
            template,
            label,
            "F72",
            seed,
            "faulted",
            [{"peak_coordinates": v} for v in wanted],
        )
    return out


# ---- macs3.control_contigs ---------------------------------------------------------------------


def count(rows) -> int | None:
    return None if rows is None else len(rows)


def _control_templates(size: int) -> tuple[Template, ...]:
    base = ("callpeak", "-t", "treat.bam", "-f", "BAM", "--nomodel", "--extsize", "200")
    base += ("-g", str(size), "-q", f2(), "-n", "x")
    return (
        Template("control_full", "macs3", (*base, "-c", "control.bam")),
        Template("control_long", "macs3", (*base, "--control", "control.bam")),
        Template("control_equals", "macs3", (*base, "--control=control.bam")),
        Template("no_control", "macs3", base),
        Template("control_without_chr1", "macs3", (*base, "-c", "control_nochr1.bam")),
        Template("control_without_chr1_long", "macs3", (*base, "--control=control_nochr1.bam")),
    )


@case_source("macs3.control_contigs")
def control_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    work = scratch / "c"
    work.mkdir(parents=True, exist_ok=True)
    size = build_gsize(seed, work, tools)
    templates = _control_templates(size)
    counts = _run_all(tools, work, templates, count)
    truth = counts[0]
    out: list[Case] = []
    for template, got in zip(templates, counts, strict=True):
        label: Label = "right" if got is not None and truth and got == truth else "wrong"
        _emit(out, work, template, label, "iv-macs3-control", seed, "genome", [])
    return out
