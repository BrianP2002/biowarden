"""Rules of the deepTools pack: iv_deeptools.yaml / engine/rules/iv_deeptools.py.

Unit tests run on small BAM and bigWig files built here (samtools, and pyBigWig through the
`python-bio` interpreter), the evidence tests run the real deepTools 3.5.6 once per claim, and the
admission tests measure the F73 rules with calls labelled by running the tools
(bench/admission/cases_iv_deeptools.py) on 16 seeds, not the 300-seed calibration. The F76
rules are measured in tests/test_family_iv_deeptools.py.
"""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, common
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import iv_deeptools as dt
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(min(ADMISSION_SEEDS, 16))
MIN_RIGHT = max(1, 200 * len(SEEDS) // 300)
IDS = [
    "deeptools.matrix_missing_data",
    "deeptools.matrix_skip_zeros",
    "deeptools.bamcoverage_track_units",
    "deeptools.duplicates_semantics",
    "deeptools.multibamsummary_regions",
]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    if len(env.real_bins(["bamCoverage", "computeMatrix", "multiBamSummary", "python-bio"])) < 4:
        pytest.skip("deepTools / python-bio are not installed")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("iv_deeptools")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def run(eng: Engine, rule: str, line: str, cwd: Path, declared: dict[str, str] | None = None):
    parts = shlex.split(line)
    decision = eng.evaluate(Call(parts[0], tuple(parts[1:]), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


# ---- the spec -----------------------------------------------------------------------------------


def test_spec_has_five_rules_with_honest_tiers(spec):
    assert [r.id for r in spec.rules] == IDS
    by_id = {r.id: r for r in spec.rules}
    assert by_id[IDS[2]].tier == "T0" and by_id[IDS[2]].declare.required is True
    for rule_id in (IDS[0], IDS[1], IDS[3], IDS[4]):
        rule = by_id[rule_id]
        assert (rule.kind, rule.tier, rule.declare.required) == ("call", "T1", False)
    assert by_id[IDS[1]].applies.options_any == ["--skipZeros"]
    assert by_id[IDS[4]].applies.subcommands == ["BED-file"]
    assert {t for r in spec.rules for t in r.applies.tools} == {
        "computeMatrix",
        "bamCoverage",
        "bamCompare",
        "multiBamSummary",
    }
    engine_for(spec)  # imports the rule modules and checks every checker name
    assert all(r.check in known_checkers() for r in spec.rules)


def test_a_t0_rule_needs_a_declaration(spec):
    data = spec.model_dump(mode="json")
    data["rules"][2]["declare"] = None
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)


# ---- command-line parsing -----------------------------------------------------------------------


def test_parse_handles_aliases_multi_values_and_flags():
    opts, pos = dt.parse(
        "computeMatrix",
        shlex.split(
            "reference-point -S a.bw b.bw -R g.bed h.bed -b 500 -a 0 -bs 25 --skipZeros -o m.gz"
        ),
    )
    assert opts["--scoreFileName"] == ["a.bw", "b.bw"]
    assert opts["--regionsFileName"] == ["g.bed", "h.bed"]
    assert (dt.last(opts, "--upstream"), dt.last(opts, "--downstream")) == ("500", "0")
    assert dt.last(opts, "--binSize") == "25" and "--skipZeros" in opts
    assert pos == ["reference-point"]
    opts, _ = dt.parse(
        "multiBamSummary", shlex.split("BED-file --BED r.bed -b a.bam b.bam --ignoreDuplicates")
    )
    assert opts["--bamfiles"] == ["a.bam", "b.bam"] and opts["--BED"] == ["r.bed"]
    assert opts["--ignoreDuplicates"] == []
    opts, _ = dt.parse(
        "bamCoverage", shlex.split("--bam x.bam --binSize=10 -e --samFlagExclude 0x400")
    )
    assert dt.last(opts, "--binSize") == "10" and opts["--extendReads"] == []
    assert dt.exclusion_flags(opts) == 1024


@pytest.mark.parametrize(
    ("args", "unit"),
    [
        ("", "reads_per_bin"),
        ("--binSize 50", "reads_per_bin"),
        ("--binSize 10 --normalizeUsing None", "reads_per_bin"),
        ("--binSize 1", "depth"),
        ("-bs 1", "depth"),
        ("--binSize 1 --normalizeUsing CPM", "normalized"),
        ("--normalizeUsing RPGC --effectiveGenomeSize 100", "normalized"),
        ("--scaleFactor 0.5", "normalized"),
        ("--binSize 1 --scaleFactor 2", "normalized"),
    ],
)
def test_coverage_unit_of_the_options(args, unit):
    assert dt.coverage_unit(["-b", "x.bam", "-o", "o.bw", *shlex.split(args)])[0] == unit


def test_an_unknown_normaliser_is_unknown(spec, tmp_path):
    decision, result = run(
        engine_for(spec),
        IDS[2],
        "bamCoverage -b x.bam -o o.bw --normalizeUsing cpm",
        tmp_path,
        {"track_unit": "normalized"},
    )
    assert result.verdict == "unknown" and decision.action == "warn"


def test_window_follows_the_strand():
    plus, minus = dt.Region("chr1", 1000, 1200, "+", 0), dt.Region("chr1", 1000, 1200, "-", 1)
    assert dt.window(plus, "scale-regions", 300, 50, "TSS") == (700, 1250)
    assert dt.window(minus, "scale-regions", 300, 50, "TSS") == (950, 1500)
    assert dt.window(plus, "reference-point", 300, 50, "TSS") == (700, 1050)
    assert dt.window(minus, "reference-point", 300, 50, "TSS") == (1150, 1500)
    assert dt.window(plus, "reference-point", 300, 50, "TES") == (900, 1250)
    assert dt.window(minus, "reference-point", 300, 50, "TES") == (950, 1300)
    assert dt.window(plus, "reference-point", 100, 100, "center") == (1000, 1200)


# ---- bigWig decoding and matrix_missing_data / matrix_skip_zeros --------------------------------

WRITER = (
    "import json, sys, pyBigWig\n"
    "s = json.load(sys.stdin)\n"
    "bw = pyBigWig.open(s['out'], 'w')\n"
    "bw.addHeader([tuple(c) for c in s['chroms']], maxZooms=0)\n"
    "bw.addEntries(s['c'], s['s'], ends=s['e'], values=s['v'])\n"
    "bw.close()\n"
)


def write_bw(tools, work: Path, name: str, rows, chroms=(("chr1", 5000), ("chr2", 3000))):
    spec = {
        "out": name,
        "chroms": [list(c) for c in chroms],
        "c": [r[0] for r in rows],
        "s": [r[1] for r in rows],
        "e": [r[2] for r in rows],
        "v": [float(r[3]) for r in rows],
    }
    proc = tools.run(["python-bio", "-c", WRITER], work, stdin=json.dumps(spec))
    assert proc.returncode == 0, proc.stderr
    return work / name


# chr1: signal over 0-1000 and 2000-3000 (a hole at 1000-2000), chr2: 100-900 of zeros only
ROWS = [
    ("chr1", 0, 1000, 1),
    ("chr1", 2000, 3000, 2),
    ("chr2", 100, 900, 0),
]


@pytest.fixture()
def bw_dir(tools, tmp_path):
    write_bw(tools, tmp_path, "sparse.bw", ROWS)
    full = [
        ("chr1", 0, 1000, 1),
        ("chr1", 1000, 2000, 0),
        ("chr1", 2000, 5000, 2),
        ("chr2", 0, 3000, 1),
    ]
    write_bw(tools, tmp_path, "dense.bw", full)
    (tmp_path / "regions.bed").write_text(
        "chr1\t500\t900\tcovered\t0\t+\nchr1\t1100\t1900\tgap\t0\t+\nchr2\t200\t800\tzero\t0\t-\n"
    )
    (tmp_path / "gapfree.bed").write_text(
        "chr1\t500\t900\tcovered\t0\t+\nchr1\t2100\t2900\tc2\t0\t-\n"
    )
    return tmp_path


def test_decoder_reads_intervals_and_chromosome_sizes(bw_dir):
    track = dt.decode_bigwig((bw_dir / "sparse.bw").read_bytes())
    assert track.chroms == {"chr1": 5000, "chr2": 3000}
    assert track.intervals["chr1"] == [(0, 1000, 1.0), (2000, 3000, 2.0)]
    assert track.covered("chr1", 500, 2500) == [(500, 1000), (2000, 2500)]
    assert track.nonzero("chr1", 0, 10) and not track.nonzero("chr2", 0, 3000)
    dense = dt.decode_bigwig((bw_dir / "dense.bw").read_bytes())
    assert dense.covered("chr1", 0, 5000) == [(0, 5000)]


def test_decoder_refuses_what_it_cannot_read():
    with pytest.raises(dt.Unreadable):
        dt.decode_bigwig(b"\x00" * 200)
    with pytest.raises(dt.Unreadable):
        dt.decode_bigwig(b"short")


MATRIX = "computeMatrix scale-regions -S {bw} -R {bed} -b 0 -a 0 --binSize 100 -o m.gz"


def test_sparse_bigwig_fails_until_zero_is_stated_and_names_the_facts(spec, bw_dir):
    eng = engine_for(spec)
    line = MATRIX.format(bw="sparse.bw", bed="regions.bed")
    decision, result = run(eng, IDS[0], line, bw_dir)
    assert decision.action == "block" and result.code == "missing_bins_not_zero"
    text = render_block(decision)
    for needle in ("--missingDataAsZero", "sparse.bw", "NaN", "chr1:1100-1900", "1 of the"):
        assert needle in text, needle
    assert "--declare missing_signal=zero" in text
    assert run(eng, IDS[0], line + " --missingDataAsZero", bw_dir)[1].verdict == "pass"
    assert run(eng, IDS[0], line, bw_dir, {"missing_signal": "na"})[1].verdict == "pass"
    wrong = run(eng, IDS[0], line, bw_dir, {"missing_signal": "zero"})[1]
    assert wrong.verdict == "fail" and "you declared missing_signal=zero" in wrong.message
    both = run(eng, IDS[0], line + " --missingDataAsZero", bw_dir, {"missing_signal": "na"})[1]
    assert both.code == "declared_na_with_zero_flag"


def test_dense_tracks_and_covered_regions_cost_nothing(spec, bw_dir):
    eng = engine_for(spec)
    assert (
        run(eng, IDS[0], MATRIX.format(bw="dense.bw", bed="regions.bed"), bw_dir)[1].verdict
        == "pass"
    )
    assert (
        run(eng, IDS[0], MATRIX.format(bw="sparse.bw", bed="gapfree.bed"), bw_dir)[1].verdict
        == "pass"
    )
    # a hole smaller than one bin leaves no NaN
    line = MATRIX.replace("--binSize 100", "--binSize 1000").format(
        bw="sparse.bw", bed="gapfree.bed"
    )
    assert run(eng, IDS[0], line, bw_dir)[1].verdict == "pass"
    # other computeMatrix subcommands and calls without files are not judged
    assert run(eng, IDS[0], "computeMatrix --help", bw_dir)[1].verdict == "pass"
    assert (
        run(eng, IDS[0], MATRIX.format(bw="absent.bw", bed="regions.bed"), bw_dir)[1].verdict
        == "pass"
    )


def test_reference_point_windows_follow_the_strand(spec, tmp_path, tools):
    write_bw(tools, tmp_path, "left.bw", [("chr1", 0, 3000, 1)])
    (tmp_path / "tss.bed").write_text("chr1\t3000\t3200\tplus\t0\t+\n")
    (tmp_path / "tss_minus.bed").write_text("chr1\t3000\t3200\tminus\t0\t-\n")
    eng = engine_for(spec)
    line = "computeMatrix reference-point -S left.bw -R {bed} -b 400 -a 0 --binSize 100 -o m.gz"
    # plus strand: window 2600-3000 (covered); minus strand: window 3200-3600 (no entry)
    assert run(eng, IDS[0], line.format(bed="tss.bed"), tmp_path)[1].verdict == "pass"
    assert run(eng, IDS[0], line.format(bed="tss_minus.bed"), tmp_path)[1].verdict == "fail"


def test_two_bigwigs_one_bed_and_unreadable_files(spec, bw_dir):
    eng = engine_for(spec)
    line = (
        "computeMatrix scale-regions -S dense.bw sparse.bw -R gapfree.bed regions.bed "
        "-b 0 -a 0 -bs 100 -o m.gz"
    )
    _, result = run(eng, IDS[0], line, bw_dir)
    assert result.verdict == "fail" and "sparse.bw" in result.message
    (bw_dir / "junk.bw").write_bytes(b"not a bigwig" * 10)
    decision, result = run(eng, IDS[0], MATRIX.format(bw="junk.bw", bed="regions.bed"), bw_dir)
    assert result.verdict == "unknown" and result.code == "bigwig_unreadable"
    assert decision.action == "warn"


def test_skip_zeros_fails_when_a_region_has_no_signal(spec, bw_dir):
    eng = engine_for(spec)
    line = MATRIX.format(bw="sparse.bw", bed="regions.bed") + " --skipZeros --missingDataAsZero"
    decision, result = run(eng, IDS[1], line, bw_dir)
    assert result.code == "zero_regions_dropped" and decision.action == "block"
    assert "chr2:200-800" in result.message and "2 of 3 regions" in result.message
    assert "--outFileSortedRegions" in render_block(decision)
    assert run(eng, IDS[1], line, bw_dir, {"zero_regions": "drop"})[1].verdict == "pass"
    keep = run(eng, IDS[1], line, bw_dir, {"zero_regions": "keep"})[1]
    assert keep.code == "declared_keep_with_skip_zeros"
    ok = MATRIX.format(bw="sparse.bw", bed="gapfree.bed") + " --skipZeros"
    assert run(eng, IDS[1], ok, bw_dir)[1].verdict == "pass"  # every region has signal
    # the rule is only matched by calls that have --skipZeros
    assert run(eng, IDS[1], MATRIX.format(bw="sparse.bw", bed="regions.bed"), bw_dir)[1] is None


# ---- bamCoverage units (T0) ---------------------------------------------------------------------


def test_track_units_is_required_and_checked_against_the_options(spec, tmp_path):
    eng = engine_for(spec)
    line = "bamCoverage -b x.bam -o o.bw"
    decision, result = run(eng, IDS[2], line, tmp_path)
    assert decision.action == "block" and result.code == "declaration_missing"
    text = render_block(decision)
    assert "--declare track_unit=reads_per_bin" in text and "--binSize 1" in text
    assert run(eng, IDS[2], line, tmp_path, {"track_unit": "reads_per_bin"})[1].verdict == "pass"
    bad = run(eng, IDS[2], line, tmp_path, {"track_unit": "depth"})[1]
    assert bad.code == "declared_mismatch" and "number of reads touching the bin" in bad.message
    assert "--binSize 1" in render_block(
        run(eng, IDS[2], line, tmp_path, {"track_unit": "depth"})[0]
    )
    assert (
        run(eng, IDS[2], line + " --binSize 1", tmp_path, {"track_unit": "depth"})[1].verdict
        == "pass"
    )
    cpm = line + " --normalizeUsing CPM"
    assert run(eng, IDS[2], cpm, tmp_path, {"track_unit": "normalized"})[1].verdict == "pass"
    assert run(eng, IDS[2], cpm, tmp_path, {"track_unit": "depth"})[1].verdict == "fail"
    assert run(eng, IDS[2], line, tmp_path, {"track_unit": "raw"})[1].code == "declaration_invalid"


def test_t0_and_t1_checkers_keep_their_tier(spec, tmp_path):
    check = known_checkers()
    call = Call(
        "bamCoverage", ("-b", "x.bam", "-o", "o.bw"), tmp_path, {"track_unit": "reads_per_bin"}
    )
    rule = next(r for r in spec.rules if r.id == IDS[2])
    assert check[rule.check](call, Match(rule), Ctx(tmp_path, "T0")).verdict == "pass"
    t1 = next(r for r in spec.rules if r.id == IDS[0])
    matrix = Call(
        "computeMatrix", tuple(shlex.split(MATRIX.format(bw="a.bw", bed="r.bed"))[1:]), tmp_path
    )
    (tmp_path / "a.bw").write_bytes(b"x")
    (tmp_path / "r.bed").write_text("chr1\t1\t2\n")
    with pytest.raises(TierViolation):
        check[t1.check](matrix, Match(t1), Ctx(tmp_path, "T0"))


# ---- duplicates ---------------------------------------------------------------------------------


def make_bam(tools, work: Path, name: str, reads) -> Path:
    """reads: (name, flag, contig, pos1, length[, mate_pos])."""
    header = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr2\tLN:5000\n@SQ\tSN:chr1\tLN:5000\n"
    body = []
    for r in reads:
        n, flag, contig, pos, length = r[:5]
        mate = ("=", str(r[5]), "100") if len(r) > 5 else ("*", "0", "0")
        body.append(
            "\t".join(
                [
                    n,
                    str(flag),
                    contig,
                    str(pos),
                    "60",
                    f"{length}M",
                    *mate,
                    "A" * length,
                    "I" * length,
                ]
            )
        )
    order = {"chr2": 0, "chr1": 1}
    rows = sorted(zip(reads, body, strict=True), key=lambda x: (order[x[0][2]], x[0][3]))
    common.write_bam(work, header + "".join(b + "\n" for _, b in rows), tools, name)
    return work / f"{name}.bam"


COUNT = "multiBamSummary BED-file --BED r.bed -b {bam}.bam -o o.npz {extra}"


def test_duplicates_rule_on_flagged_reads_with_matching_and_other_starts(spec, tools, tmp_path):
    eng = engine_for(spec)
    (tmp_path / "r.bed").write_text("chr1\t90\t400\tR\n")
    # same start, same strand: --ignoreDuplicates leaves exactly the unflagged read
    make_bam(tools, tmp_path, "same", [("a", 0, "chr1", 100, 40), ("b", 1024, "chr1", 100, 40)])
    # a reverse-strand duplicate trimmed after marking starts elsewhere: it survives
    make_bam(tools, tmp_path, "moved", [("a", 16, "chr1", 100, 40), ("b", 1040, "chr1", 105, 35)])
    # no flagged read at all
    make_bam(tools, tmp_path, "nodup", [("a", 0, "chr1", 100, 40), ("b", 0, "chr1", 100, 40)])
    ignore = "--ignoreDuplicates"
    assert run(eng, IDS[3], COUNT.format(bam="same", extra=ignore), tmp_path)[1].verdict == "pass"
    decision, result = run(eng, IDS[3], COUNT.format(bam="moved", extra=ignore), tmp_path)
    assert result.code == "ignore_duplicates_not_the_flag" and decision.action == "block"
    text = render_block(decision)
    for needle in ("1 of them start where no unflagged read starts", "--samFlagExclude 1024"):
        assert needle in text, needle
    flag = COUNT.format(bam="moved", extra="--samFlagExclude 1024")
    assert run(eng, IDS[3], flag, tmp_path)[1].verdict == "pass"
    assert run(eng, IDS[3], flag + " --ignoreDuplicates", tmp_path)[1].verdict == "pass"
    assert run(eng, IDS[3], flag.replace("1024", "0x400"), tmp_path)[1].verdict == "pass"
    # without flagged reads the rule has nothing to say, whatever the options
    assert run(eng, IDS[3], COUNT.format(bam="nodup", extra=ignore), tmp_path)[1].verdict == "pass"
    # no duplicate option: counts flagged duplicates, which only matters if exclusion was meant
    plain = COUNT.format(bam="moved", extra="")
    assert run(eng, IDS[3], plain, tmp_path)[1].verdict == "pass"
    kept = run(eng, IDS[3], plain, tmp_path, {"duplicates": "exclude"})[1]
    assert kept.code == "duplicates_kept" and "--samFlagExclude 1024" in kept.fix[0]
    assert run(eng, IDS[3], plain, tmp_path, {"duplicates": "keep"})[1].verdict == "pass"
    contradiction = run(eng, IDS[3], flag, tmp_path, {"duplicates": "keep"})[1]
    assert contradiction.code == "declared_keep_with_exclusion"


def test_ignore_duplicates_also_collapses_unflagged_reads_that_share_a_start(spec, tools, tmp_path):
    (tmp_path / "r.bed").write_text("chr1\t90\t400\tR\n")
    make_bam(
        tools,
        tmp_path,
        "stack",
        [("a", 0, "chr1", 100, 40), ("b", 0, "chr1", 100, 40), ("c", 1024, "chr1", 100, 40)],
    )
    _, result = run(
        engine_for(spec), IDS[3], COUNT.format(bam="stack", extra="--ignoreDuplicates"), tmp_path
    )
    assert result.verdict == "fail" and "1 unflagged reads would be collapsed" in result.message
    scan = dt.scan_duplicates(Ctx(tmp_path, "T1"), tmp_path / "stack.bam")
    assert (scan.reads, scan.flagged, scan.unflagged, scan.kept_by_position) == (3, 1, 2, 1)


def test_paired_reads_group_by_mate_position_and_other_tools_are_matched(spec, tools, tmp_path):
    eng = engine_for(spec)
    make_bam(
        tools,
        tmp_path,
        "pe",
        [("a", 99, "chr1", 100, 40, 300), ("b", 99 | 1024, "chr1", 100, 40, 350)],
    )
    # same start but another mate: two groups, one unflagged -> the flagged mate-pair survives
    line = "bamCoverage -b pe.bam -o o.bw --binSize 1 --ignoreDuplicates"
    assert run(eng, IDS[3], line, tmp_path, {"track_unit": "depth"})[1].verdict == "fail"
    line = "bamCompare -b1 pe.bam -b2 pe.bam -o o.bw --ignoreDuplicates"
    assert run(eng, IDS[3], line, tmp_path)[1].verdict == "fail"


# ---- multiBamSummary regions --------------------------------------------------------------------


def test_multibamsummary_regions_absent_contigs_and_unsorted_beds(spec, tools, tmp_path):
    eng = engine_for(spec)
    make_bam(tools, tmp_path, "h", [("a", 0, "chr2", 100, 40), ("b", 0, "chr1", 100, 40)])
    lines = {
        "ok.bed": "chr2\t50\t300\tA\nchr2\t400\t500\tB\nchr1\t50\t300\tC\n",
        "natural.bed": "chr1\t50\t300\tC\nchr2\t50\t300\tA\n",
        "late.bed": "chr2\t400\t500\tB\nchr2\t50\t300\tA\n",
        "absent.bed": "chr2\t50\t300\tA\nchrX\t5\t60\tZ\n",
    }
    for name, text in lines.items():
        (tmp_path / name).write_text(text)
    summary = "multiBamSummary BED-file --BED {bed} -b h.bam -o o.npz"
    # output order is the contig order of the BAM header (chr2, chr1), then start
    assert run(eng, IDS[4], summary.format(bed="ok.bed"), tmp_path)[1].verdict == "pass"
    nat = run(eng, IDS[4], summary.format(bed="natural.bed"), tmp_path)
    assert (
        nat[1].code == "regions_reordered_or_dropped"
        and "1 BED row(s) are out of the order" in nat[1].message
    )
    assert run(eng, IDS[4], summary.format(bed="late.bed"), tmp_path)[1].verdict == "fail"
    absent = run(eng, IDS[4], summary.format(bed="absent.bed"), tmp_path)
    assert "chrX:5-60" in absent[1].message and "dropped without a message" in absent[1].message
    assert "chr/start/end" in render_block(absent[0])
    declared = {"region_join": "coordinates"}
    assert (
        run(eng, IDS[4], summary.format(bed="absent.bed"), tmp_path, declared)[1].verdict == "pass"
    )
    # bins mode is not judged
    assert run(eng, IDS[4], "multiBamSummary bins -b h.bam -o o.npz", tmp_path)[1] is None


# ---- evidence: the real tools -------------------------------------------------------------------


def test_real_computematrix_gives_nan_for_missing_bins_and_skipzeros_drops_rows(tools, bw_dir):
    def matrix(bw, bed, *extra):
        argv = [
            "computeMatrix",
            "scale-regions",
            "-S",
            bw,
            "-R",
            bed,
            "-b",
            "0",
            "-a",
            "0",
            "--binSize",
            "100",
            "-p",
            "1",
            "-o",
            "m.gz",
            "--outFileNameMatrix",
            "m.tab",
            *extra,
        ]
        proc = tools.run(argv, bw_dir)
        assert proc.returncode == 0 and proc.stdout == ""
        rows = [
            line.split("\t")
            for line in (bw_dir / "m.tab").read_text().splitlines()
            if not line.startswith(("#", "genes:"))
        ]
        return proc, [[float(x) for x in r] for r in rows]

    proc, rows = matrix("sparse.bw", "regions.bed")
    assert proc.stderr == "" and len(rows) == 3
    assert any(math.isnan(x) for x in rows[1]) and all(math.isnan(x) for x in rows[1][2:8])
    assert not any(
        math.isnan(x) for x in matrix("sparse.bw", "regions.bed", "--missingDataAsZero")[1][1]
    )
    proc, rows = matrix("sparse.bw", "regions.bed", "--skipZeros", "--missingDataAsZero")
    assert len(rows) == 1 and "Skipping" in proc.stderr  # the two regions without signal are gone


def test_real_multibamsummary_reorders_rows_and_drops_absent_contigs(tools, tmp_path):
    make_bam(tools, tmp_path, "h", [("a", 0, "chr2", 100, 40), ("b", 0, "chr1", 100, 40)])
    (tmp_path / "u.bed").write_text("chr1\t50\t300\tC\nchr2\t50\t300\tA\nchrX\t5\t60\tZ\n")
    proc = tools.run(
        [
            "multiBamSummary",
            "BED-file",
            "--BED",
            "u.bed",
            "-b",
            "h.bam",
            "-p",
            "1",
            "-o",
            "o.npz",
            "--outRawCounts",
            "o.tab",
        ],
        tmp_path,
    )
    assert proc.returncode == 0
    rows = [line.split("\t")[:3] for line in (tmp_path / "o.tab").read_text().splitlines()[1:]]
    assert rows == [["chr2", "50", "300"], ["chr1", "50", "300"]]  # BAM header order; chrX dropped
    assert "dropped" not in proc.stderr and "chrX" not in proc.stderr


def test_real_bamcoverage_bin_value_is_reads_per_bin_and_binsize_one_is_depth(tools, tmp_path):
    reads = [(f"r{i}", 0, "chr1", 1000 + i * 5, 50) for i in range(20)]
    make_bam(tools, tmp_path, "a", reads)
    base = ["bamCoverage", "-b", "a.bam", "-of", "bedgraph", "-p", "1"]
    for name, extra in (
        ("d", []),
        ("one", ["--binSize", "1"]),
        ("cpm", ["--normalizeUsing", "CPM"]),
    ):
        assert tools.run([*base, "-o", f"{name}.bg", *extra], tmp_path).returncode == 0
    rows = {
        n: [r.split("\t") for r in (tmp_path / f"{n}.bg").read_text().splitlines()]
        for n in ("d", "one", "cpm")
    }
    default = {int(r[1]): float(r[3]) for r in rows["d"]}
    starts = [999 + i * 5 for i in range(20)]
    touching = sum(1 for a in starts if a < 1050 and a + 50 > 1000)
    depth = sum(max(0, min(a + 50, 1050) - max(a, 1000)) for a in starts) / 50
    assert default[1000] == touching == 11 and depth == pytest.approx(5.68)  # reads, not depth
    assert sum((int(r[2]) - int(r[1])) * float(r[3]) for r in rows["one"]) == 20 * 50  # depth
    assert float({int(r[1]): r for r in rows["cpm"]}[1000][3]) == pytest.approx(touching * 1e6 / 20)


# ---- admission (F73 rules; the F76 rules are measured in test_family_iv_deeptools.py) -----------


@pytest.mark.parametrize("rule_id", IDS[:2])
def test_admission_of_the_matrix_rules(spec, tools, tmp_path, rule_id):
    stats = measure(rule_id, [spec], SEEDS, tools, tmp_path)
    right_declared, right_undeclared = stats.cells[("right", True)], stats.cells[("right", False)]
    assert stats.conditional and stats.admitted()
    assert right_declared.n >= MIN_RIGHT and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT and right_undeclared.failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert all(cell.unknown == 0 for cell in stats.cells.values())
    # an undeclared wrong call over a sparse track or with --skipZeros is caught every time
    assert stats.cells[("wrong", False)].failed >= len(SEEDS) * 2
    assert stats.cells[("wrong", True)].failed > 0
