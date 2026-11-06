"""Rules of the bedtools part of pack iv: spec shape, tier honesty, checker edge cases, admission.

Unit tests run on small files written here (BED, bedGraph, SAM; the checkers read SAM and BAM the
same way) and on F65-F69/F75 workspaces built by the benchmark; the dry runs use the real bedtools
2.31.1. The admission tests measure each rule with calls labelled by running the real tools
(bench/admission/cases_iv_bedtools.py).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

import biowarden.engine.rules  # noqa: F401  (registers the packaged checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules.seq_common import MAX_FILE_BYTES
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "bedtools.split_reciprocal": ("tool_logic", "T1", None),
    "bedtools.map_unweighted": ("weighting", "T1", "signal_weighting"),
    "bedtools.spliced_without_split": ("split", "T1", "spliced_reads"),
    "bedtools.subtract_trims": ("overlap_policy", "T1", "overlap_policy"),
    "bedtools.window_boundary": ("distance_unit", "T1", "window_boundary"),
    "bedtools.closest_distance_unit": ("distance_unit", "T0", "distance_unit"),
    "bedtools.duplicate_policy": ("default_param", "T1", "duplicate_policy"),
    "bedtools.merge_bookended": ("interval_semantics", "T1", "adjacent_policy"),
    "bedtools.groupby_unsorted": ("record_order", "T1", None),
    "bedtools.prec_significant": ("units", "T0", "precision_unit"),
}
REQUIRED = {"bedtools.prec_significant"}  # the built-in option rule: -prec needs a declaration


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "bedtools" not in env.real_bins(["bedtools"]):
        pytest.skip("bedtools is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("iv_bedtools")


def engine_for(spec: Spec, tools: ToolEnv) -> Engine:
    return Engine(
        [spec],
        mode="block",
        real_bins=tools.real_bins(["bedtools"]),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )


def run(eng: Engine, rule_id: str, argv: list[str], cwd: Path, declared=None):
    decision = eng.evaluate(Call("bedtools", tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def verdict(eng, rule_id, argv, cwd, declared=None):
    """The rule's verdict, or None when the rule did not apply."""
    result = run(eng, rule_id, argv, cwd, declared)[1]
    return None if result is None else result.verdict


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_lists_the_rules_with_their_slot_tier_and_declaration(spec):
    assert spec.name == "iv_bedtools" and {r.id for r in spec.rules} == set(RULES)
    for rule in spec.rules:
        slot, tier, key = RULES[rule.id]
        assert (rule.slot, rule.tier, rule.kind) == (slot, tier, "call"), rule.id
        assert rule.applies.tools == ["bedtools"] and rule.applies.subcommands, rule.id
        assert (rule.declare.key if rule.declare else None) == key, rule.id
        if rule.declare:
            assert rule.declare.required is (rule.id in REQUIRED), rule.id
            assert len(rule.declare.values) == 2
        assert rule.check in known_checkers() and rule.message
    assert [r.incidents for r in spec.rules if r.id.endswith("split_reciprocal")] == [["GEN-40"]]


def test_spec_rejects_a_wrong_shape(spec):
    data = spec.model_dump(mode="json")
    data["rules"][5]["declare"] = None  # the T0 rule has no declaration
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][0]["applies"]["tools"] = []
    with pytest.raises(ValidationError, match="needs applies.tools"):
        Spec.model_validate(data)


def test_tier_honesty_t1_checkers_read_files_and_the_t0_checker_does_not(spec, tmp_path):
    (tmp_path / "a.bed").write_text("chr1\t1\t100\tx\n")
    (tmp_path / "b.bed").write_text("chr1\t1100\t1200\ty\n")
    calls = {
        "bedtools.split_reciprocal": ["intersect", "-a", "a.bed", "-b", "b.bed", "-split", "-r"],
        "bedtools.window_boundary": ["window", "-a", "a.bed", "-b", "b.bed", "-w", "1000"],
        "bedtools.subtract_trims": ["subtract", "-a", "a.bed", "-b", "b.bed"],
        "bedtools.merge_bookended": ["merge", "-i", "a.bed"],
        "bedtools.groupby_unsorted": ["groupby", "-i", "a.bed", "-g", "4"],
    }
    for rule in spec.rules:
        check = known_checkers()[rule.check]
        if rule.id == "bedtools.closest_distance_unit":
            call = Call("bedtools", ("closest", "-a", "a.bed", "-b", "b.bed", "-d"), tmp_path)
            assert check(call, Match(rule), Ctx(tmp_path, "T0")).verdict == "unknown"
        elif rule.tier == "T0":
            assert rule.check == "option_implies_declared"
        elif rule.id in calls:
            call = Call("bedtools", tuple(calls[rule.id]), tmp_path)
            with pytest.raises(TierViolation):
                check(call, Match(rule), Ctx(tmp_path, "T0"))


# ---- bedtools.split_reciprocal -----------------------------------------------------------------


def bed12(chrom, exons, name):
    start, end = exons[0][0], exons[-1][1]
    sizes = ",".join(str(b - a) for a, b in exons)
    starts = ",".join(str(a - start) for a, b in exons)
    head = f"{chrom}\t{start}\t{end}\t{name}\t0\t+\t{start}\t{end}\t0"
    return f"{head}\t{len(exons)}\t{sizes}\t{starts}\n"


EXONS = [(1000, 1200), (1900, 2100), (2800, 3000)]


def write_pair(tmp_path, decoy=None):
    (tmp_path / "a.bed").write_text(bed12("chr1", EXONS, "A1"))
    rows = [bed12("chr1", EXONS, "P")]
    if decoy:
        rows.append(bed12("chr1", [decoy], "D"))
    (tmp_path / "b.bed").write_text("".join(rows))


SPLIT = ["intersect", "-a", "a.bed", "-b", "b.bed", "-split"]


@pytest.mark.parametrize(
    "fraction",
    [
        ["-r", "-f", "0.9"],
        ["-F", "0.9"],
        ["-f", "0.9", "-F", "0.9"],
        ["-r", "-f", "0.9", "-wa", "-u"],
    ],
)
def test_a_decoy_in_an_intron_makes_the_fraction_options_fail(spec, tools, tmp_path, fraction):
    write_pair(tmp_path, decoy=(1400, 1700))
    decision, result = run(
        engine_for(spec, tools), "bedtools.split_reciprocal", [*SPLIT, *fraction], tmp_path
    )
    assert result.verdict == "fail" and decision.action == "block"
    assert result.code == "split_fraction_depends_on_other_records"
    text = render_block(decision)
    assert text.startswith("FIX: run intersect once per record of b.bed")
    assert "A1 / P" in text and "-split" in text


@pytest.mark.parametrize("decoy", [None, (3500, 3700), (3000, 3100), (1210, 1250)])
def test_no_decoy_a_far_decoy_a_touching_one_and_a_short_one_pass(spec, tools, tmp_path, decoy):
    write_pair(tmp_path, decoy=decoy)
    eng = engine_for(spec, tools)
    _, result = run(eng, "bedtools.split_reciprocal", [*SPLIT, "-r", "-f", "0.9"], tmp_path)
    assert result.verdict == "pass"


def test_calls_the_rule_does_not_judge_pass_or_stay_silent(spec, tools, tmp_path):
    write_pair(tmp_path, decoy=(1400, 1700))
    eng = engine_for(spec, tools)
    rid = "bedtools.split_reciprocal"
    for argv in (
        ["intersect", "-a", "a.bed", "-b", "b.bed", "-r", "-f", "0.9"],  # no -split
        ["intersect", "-a", "a.bed", "-b", "b.bed", "-split"],  # no fraction option
        ["intersect", "-a", "a.bed", "-b", "absent.bed", "-split", "-r"],  # a file is missing
        ["intersect", "-a", "a.bed", "-b", "b.bed", "c.bed", "-split", "-r"],  # two b files
    ):
        decision, result = run(eng, rid, argv, tmp_path)
        assert decision.action == "allow" and (result is None or result.verdict == "pass"), argv
    # a single b record cannot lose a pair to another record
    (tmp_path / "one.bed").write_text(bed12("chr1", EXONS, "P"))
    _, result = run(
        eng, rid, ["intersect", "-a", "a.bed", "-b", "one.bed", "-split", "-r"], tmp_path
    )
    assert result.verdict == "pass"


def test_option_spellings_and_order_do_not_matter(spec, tools, tmp_path):
    write_pair(tmp_path, decoy=(1400, 1700))
    eng = engine_for(spec, tools)
    argv = ["intersect", "-wa", "-wb", "-split", "-F", "0.9", "-b", "b.bed", "-a", "a.bed"]
    assert run(eng, "bedtools.split_reciprocal", argv, tmp_path)[1].verdict == "fail"


# ---- bedtools.map_unweighted -------------------------------------------------------------------


def write_track(tmp_path, name="cov.bedgraph", lines=None):
    lines = lines or [(100, 110, 10), (110, 290, 2), (290, 400, 6)]
    (tmp_path / name).write_text("".join(f"chr1\t{s}\t{e}\t{v}\n" for s, e, v in lines))
    (tmp_path / "win.bed").write_text(
        "chr1\t100\t200\tw1\nchr1\t200\t300\tw2\nchr1\t300\t400\tw3\n"
    )


MAP = ["map", "-a", "win.bed", "-b", "cov.bedgraph", "-c", "4", "-o", "mean"]


def test_map_mean_over_unequal_lines_fails_and_a_binned_track_passes(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    write_track(tmp_path)
    decision, result = run(eng, "bedtools.map_unweighted", MAP, tmp_path)
    assert result.verdict == "fail" and result.code == "map_mean_is_not_per_base"
    text = render_block(decision)
    assert "2 of 3 windows" in result.message or "windows" in result.message
    assert "bedtools intersect -a win.bed -b cov.bedgraph -wo" in text
    write_track(tmp_path, lines=[(x, x + 10, 5 + (x // 10) % 3) for x in range(100, 400, 10)])
    assert run(eng, "bedtools.map_unweighted", MAP, tmp_path)[1].verdict == "pass"


def test_map_declarations_and_other_operations(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    write_track(tmp_path)
    rid = "bedtools.map_unweighted"
    assert run(eng, rid, MAP, tmp_path, {"signal_weighting": "per_interval"})[1].verdict == "pass"
    assert run(eng, rid, MAP, tmp_path, {"signal_weighting": "per_base"})[1].verdict == "fail"
    for op in ("max", "min", "count", "sum", "median"):  # only the mean is judged
        argv = [*MAP[:-1], op]
        assert run(eng, rid, argv, tmp_path)[1].verdict == "pass", op
    argv = [*MAP[:6], "5", "-o", "mean"]  # another column
    assert run(eng, rid, argv, tmp_path)[1].verdict == "pass"
    # the mean is the second operation of a list
    argv = [*MAP[:6], "4,4", "-o", "count,mean"]
    assert run(eng, rid, argv, tmp_path)[1].verdict == "fail"


def test_map_only_judges_a_bedgraph(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    write_track(tmp_path)
    (tmp_path / "scores.bed").write_text("chr1\t100\t110\tp\t10\nchr1\t110\t290\tq\t2\n")
    assert (
        run(
            eng,
            "bedtools.map_unweighted",
            MAP[:4] + ["scores.bed", "-c", "5", "-o", "mean"],
            tmp_path,
        )[1].verdict
        == "pass"
    )
    (tmp_path / "x.txt").write_text("chr1\t100\t110\t10\n")
    assert (
        run(eng, "bedtools.map_unweighted", MAP[:4] + ["x.txt", "-c", "4", "-o", "mean"], tmp_path)[
            1
        ].verdict
        == "pass"
    )
    (tmp_path / "t.txt").write_text(
        "track type=bedGraph name=x\nchr1\t100\t110\t10\nchr1\t110\t290\t2\nchr1\t290\t400\t6\n"
    )
    assert (
        run(eng, "bedtools.map_unweighted", MAP[:4] + ["t.txt", "-c", "4", "-o", "mean"], tmp_path)[
            1
        ].verdict
        == "fail"
    )


# ---- bedtools.spliced_without_split ------------------------------------------------------------

SAM_HEADER = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:2000\n"


def write_sam(tmp_path, name="reads.sam", records=None):
    rows = records or [("r1", 0, 101, "20M300N20M"), ("r2", 0, 111, "40M")]
    text = SAM_HEADER
    for rid, flag, pos, cigar in rows:
        text += f"{rid}\t{flag}\tchr1\t{pos}\t60\t{cigar}\t*\t0\t0\t{'A' * 40}\t{'I' * 40}\n"
    (tmp_path / name).write_text(text)


def test_spliced_reads_without_split_fail_for_every_coverage_subcommand(spec, tools, tmp_path):
    write_sam(tmp_path)
    (tmp_path / "reg.bed").write_text("chr1\t200\t400\tintron\n")
    eng = engine_for(spec, tools)
    rid = "bedtools.spliced_without_split"
    cases = {
        "coverage": ["coverage", "-a", "reg.bed", "-b", "reads.sam"],
        "coverage_counts": ["coverage", "-a", "reg.bed", "-b", "reads.sam", "-counts"],
        "intersect": ["intersect", "-a", "reg.bed", "-b", "reads.sam", "-c"],
        "multicov": ["multicov", "-bams", "reads.sam", "-bed", "reg.bed"],
        "genomecov": ["genomecov", "-ibam", "reads.sam", "-bg"],
        "window": ["window", "-a", "reg.bed", "-b", "reads.sam", "-w", "10"],
    }
    for name, argv in cases.items():
        decision, result = run(eng, rid, argv, tmp_path)
        assert result.verdict == "fail", name
    text = render_block(run(eng, rid, cases["coverage"], tmp_path)[0])
    assert text.startswith("FIX: add -split to bedtools coverage")


def test_split_a_declared_span_and_unspliced_or_unrelated_regions_pass(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.spliced_without_split"
    write_sam(tmp_path)
    (tmp_path / "reg.bed").write_text("chr1\t200\t400\tintron\n")
    base = ["coverage", "-a", "reg.bed", "-b", "reads.sam"]
    assert run(eng, rid, [*base, "-split"], tmp_path)[1].verdict == "pass"
    assert run(eng, rid, base, tmp_path, {"spliced_reads": "span"})[1].verdict == "pass"
    assert run(eng, rid, base, tmp_path, {"spliced_reads": "exons"})[1].verdict == "fail"
    (tmp_path / "far.bed").write_text("chr1\t1000\t1200\tgene\n")  # no gap lies there
    assert (
        run(eng, rid, ["coverage", "-a", "far.bed", "-b", "reads.sam"], tmp_path)[1].verdict
        == "pass"
    )
    write_sam(tmp_path, "plain.sam", [("r1", 0, 101, "40M"), ("r2", 0, 111, "40M")])
    assert (
        run(eng, rid, ["coverage", "-a", "reg.bed", "-b", "plain.sam"], tmp_path)[1].verdict
        == "pass"
    )
    # a region that contains the whole gap and its flanks is a base-level hit, not a read-level one
    (tmp_path / "gene.bed").write_text("chr1\t90\t500\tgene\n")
    counts = ["coverage", "-a", "gene.bed", "-b", "reads.sam", "-counts"]
    assert run(eng, rid, counts, tmp_path)[1].verdict == "pass"
    assert (
        run(eng, rid, ["coverage", "-a", "gene.bed", "-b", "reads.sam"], tmp_path)[1].verdict
        == "fail"
    )


# ---- bedtools.subtract_trims -------------------------------------------------------------------


def write_peaks(tmp_path, black):
    (tmp_path / "p.bed").write_text("chr1\t100\t200\tp1\nchr1\t300\t400\tp2\nchr1\t500\t600\tp3\n")
    (tmp_path / "bl.bed").write_text("".join(f"chr1\t{s}\t{e}\tb\n" for s, e in black))


def test_subtract_that_leaves_pieces_fails_and_a_or_n_or_full_cover_passes(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.subtract_trims"
    write_peaks(tmp_path, [(180, 320), (510, 520)])
    sub = ["subtract", "-a", "p.bed", "-b", "bl.bed"]
    decision, result = run(eng, rid, sub, tmp_path)
    assert result.verdict == "fail" and result.code == "subtract_trims_not_removes"
    assert render_block(decision).startswith("FIX: add -A to remove every a feature")
    for flag in ("-A", "-N"):
        assert run(eng, rid, [*sub, flag], tmp_path)[1].verdict == "pass"
    assert run(eng, rid, sub, tmp_path, {"overlap_policy": "trim"})[1].verdict == "pass"
    assert run(eng, rid, sub, tmp_path, {"overlap_policy": "remove"})[1].verdict == "fail"
    write_peaks(tmp_path, [(90, 210), (290, 410)])  # covers two peaks completely, touches nothing
    assert run(eng, rid, sub, tmp_path)[1].verdict == "pass"
    write_peaks(tmp_path, [(1000, 1100)])  # no overlap at all
    assert run(eng, rid, sub, tmp_path)[1].verdict == "pass"


# ---- the dry runs stay bounded (E-7) -------------------------------------------------------------

DRY_RUN_CALLS = (
    ("bedtools.split_reciprocal", [*SPLIT, "-r", "-f", "0.9"]),
    ("bedtools.subtract_trims", ["subtract", "-a", "p.bed", "-b", "bl.bed"]),
)


def write_dry_run_inputs(tmp_path):
    write_pair(tmp_path, decoy=(1400, 1700))  # a.bed, b.bed: the split_reciprocal trap
    write_peaks(tmp_path, [(180, 320), (510, 520)])  # p.bed, bl.bed: the subtract trap


def test_a_dry_run_that_times_out_is_unknown_not_a_crash(spec, tools, tmp_path, monkeypatch):
    write_dry_run_inputs(tmp_path)
    eng = engine_for(spec, tools)
    for rule_id, argv in DRY_RUN_CALLS:  # both fail on these inputs when bedtools answers
        assert run(eng, rule_id, argv, tmp_path)[1].verdict == "fail"

    def too_slow(self, tool, argv):
        raise subprocess.TimeoutExpired([tool, *argv], 0.1)

    monkeypatch.setattr(Ctx, "run_real", too_slow)
    for rule_id, argv in DRY_RUN_CALLS:
        decision, result = run(eng, rule_id, argv, tmp_path)
        assert (result.verdict, result.code) == ("unknown", "dry_run_failed"), rule_id
        assert decision.action == "warn"


def test_inputs_too_large_to_rerun_are_not_dry_run(spec, tools, tmp_path, monkeypatch):
    write_dry_run_inputs(tmp_path)
    eng = engine_for(spec, tools)
    for name in ("b.bed", "bl.bed"):  # a sparse file: only the size counts
        os.truncate(tmp_path / name, MAX_FILE_BYTES + 1)

    def refuse(self, tool, argv):
        raise AssertionError(f"{tool} {argv} ran on an input that is too large")

    monkeypatch.setattr(Ctx, "run_real", refuse)
    for rule_id, argv in DRY_RUN_CALLS:
        decision, result = run(eng, rule_id, argv, tmp_path)
        assert result.verdict == "pass" and "too large" in result.note, rule_id
        assert decision.action == "allow"


# ---- bedtools.window_boundary ------------------------------------------------------------------


def write_window(tmp_path, gap, side="after"):
    (tmp_path / "a.bed").write_text("chr1\t5000\t5100\tp1\n")
    b = (5100 + gap, 5200 + gap) if side == "after" else (5000 - gap - 100, 5000 - gap)
    (tmp_path / "b.bed").write_text(f"chr1\t{b[0]}\t{b[1]}\tg1\n")


def test_window_with_a_gap_of_exactly_n_warns_when_undeclared_and_fails_when_gap_inclusive(
    spec, tools, tmp_path
):
    eng = engine_for(spec, tools)
    rid = "bedtools.window_boundary"
    base = ["window", "-a", "a.bed", "-b", "b.bed"]
    for side in ("after", "before"):
        write_window(tmp_path, 1000, side)
        decision, result = run(eng, rid, [*base, "-w", "1000"], tmp_path)
        assert result.verdict == "unknown" and decision.action == "warn", side
        assert "-w 1000" not in render_warning(decision) and "1001" in render_warning(decision)
        assert (
            run(eng, rid, [*base, "-w", "1000"], tmp_path, {"window_boundary": "gap_inclusive"})[
                1
            ].verdict
            == "fail"
        )
        assert (
            run(eng, rid, [*base, "-w", "1000"], tmp_path, {"window_boundary": "window_exclusive"})[
                1
            ].verdict
            == "pass"
        )
    write_window(tmp_path, 1000, "after")
    assert run(eng, rid, [*base, "-r", "1000"], tmp_path)[1].verdict == "unknown"
    assert run(eng, rid, [*base, "-l", "1000"], tmp_path)[1].verdict == "pass"  # the other side
    assert run(eng, rid, [*base, "-l", "1000", "-r", "1000"], tmp_path)[1].verdict == "unknown"
    for size in ("999", "1001", "500"):
        assert run(eng, rid, [*base, "-w", size], tmp_path)[1].verdict == "pass", size


# ---- bedtools.closest_distance_unit ------------------------------------------------------------


def test_closest_distance_unit_is_decided_from_the_call_alone(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.closest_distance_unit"
    for option in (["-d"], ["-D", "ref"], ["-D", "a"]):
        argv = ["closest", "-a", "a.bed", "-b", "b.bed", *option]
        decision, result = run(eng, rid, argv, tmp_path)
        assert result.verdict == "unknown" and decision.action == "warn"
        assert "gap plus one" in render_warning(decision)
        assert (
            run(eng, rid, argv, tmp_path, {"distance_unit": "bedtools_distance"})[1].verdict
            == "pass"
        )
        decision, result = run(eng, rid, argv, tmp_path, {"distance_unit": "bases_between"})
        assert result.verdict == "fail" and decision.action == "block"
        assert render_block(decision).startswith("FIX: compare the distance column with limit+1")
    decision, result = run(eng, rid, ["closest", "-a", "a.bed", "-b", "b.bed"], tmp_path)
    assert result is None and decision.action == "allow"  # no distance column, no rule


# ---- bedtools.duplicate_policy -----------------------------------------------------------------


def test_duplicate_policy_fails_a_declaration_that_contradicts_the_subcommand(
    spec, tools, tmp_path
):
    eng = engine_for(spec, tools)
    rid = "bedtools.duplicate_policy"
    write_sam(
        tmp_path,
        "d.sam",
        [("r1", 0, 101, "40M"), ("r2", 1024, 101, "40M"), ("r3", 512, 120, "40M")],
    )
    (tmp_path / "reg.bed").write_text("chr1\t90\t200\tg\n")
    cov = ["coverage", "-a", "reg.bed", "-b", "d.sam", "-counts"]
    multi = ["multicov", "-bams", "d.sam", "-bed", "reg.bed"]
    assert run(eng, rid, cov, tmp_path)[1].verdict == "unknown"
    assert run(eng, rid, cov, tmp_path, {"duplicate_policy": "keep"})[1].verdict == "pass"
    decision, result = run(eng, rid, cov, tmp_path, {"duplicate_policy": "exclude"})
    assert result.verdict == "fail" and "counts them" in result.message
    assert run(eng, rid, multi, tmp_path, {"duplicate_policy": "exclude"})[1].verdict == "pass"
    assert run(eng, rid, multi, tmp_path, {"duplicate_policy": "keep"})[1].verdict == "fail"
    assert (
        run(eng, rid, [*multi, "-D", "-F"], tmp_path, {"duplicate_policy": "keep"})[1].verdict
        == "pass"
    )
    write_sam(tmp_path, "clean.sam", [("r1", 0, 101, "40M"), ("r2", 0, 111, "40M")])
    clean = ["coverage", "-a", "reg.bed", "-b", "clean.sam", "-counts"]
    assert run(eng, rid, clean, tmp_path, {"duplicate_policy": "exclude"})[1].verdict == "pass"


# ---- bedtools.merge_bookended ------------------------------------------------------------------


def write_intervals(tmp_path, spans, strand=None):
    rows = []
    for i, (s, e) in enumerate(spans):
        rows.append(f"chr1\t{s}\t{e}\tn{i}" + (f"\t0\t{strand[i]}" if strand else "") + "\n")
    (tmp_path / "m.bed").write_text("".join(rows))


def test_merge_with_abutting_intervals_warns_fails_or_passes(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.merge_bookended"
    write_intervals(tmp_path, [(10, 20), (20, 30), (31, 40)])
    base = ["merge", "-i", "m.bed"]
    decision, result = run(eng, rid, base, tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"
    assert "-d -1" in render_warning(decision)
    keep, join = {"adjacent_policy": "keep"}, {"adjacent_policy": "merge"}
    assert verdict(eng, rid, base, tmp_path, keep) == "fail"
    assert verdict(eng, rid, base, tmp_path, join) == "pass"
    assert verdict(eng, rid, [*base, "-d", "0"], tmp_path) == "unknown"
    assert verdict(eng, rid, [*base, "-d", "-1"], tmp_path, keep) == "pass"
    assert verdict(eng, rid, [*base, "-d", "5"], tmp_path) == "pass"
    assert verdict(eng, rid, [*base, "-c", "4", "-o", "count"], tmp_path) == "unknown"
    write_intervals(tmp_path, [(10, 25), (20, 30), (31, 40)])  # overlap, no abutting pair
    assert verdict(eng, rid, base, tmp_path, keep) == "pass"
    write_intervals(tmp_path, [(10, 20), (20, 30)], strand=["+", "-"])  # abut across strands
    assert verdict(eng, rid, [*base, "-s"], tmp_path, keep) == "pass"
    assert verdict(eng, rid, base, tmp_path, keep) == "fail"


# ---- bedtools.groupby_unsorted -----------------------------------------------------------------


def test_groupby_on_input_that_is_not_grouped_fails_and_sorted_input_passes(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.groupby_unsorted"
    (tmp_path / "g.bed").write_text("chr1\t1\t2\tg1\t5\nchr1\t3\t4\tg2\t7\nchr1\t5\t6\tg1\t3\n")
    tail = ["-c", "5", "-o", "sum"]
    decision, result = run(eng, rid, ["groupby", "-i", "g.bed", "-g", "4", *tail], tmp_path)
    assert result.verdict == "fail" and result.code == "groupby_input_not_grouped"
    assert render_block(decision).startswith("FIX: sort first: sort -k4,4 g.bed")
    (tmp_path / "s.bed").write_text("chr1\t1\t2\tg1\t5\nchr1\t5\t6\tg1\t3\nchr1\t3\t4\tg2\t7\n")
    assert verdict(eng, rid, ["groupby", "-i", "s.bed", "-g", "4", *tail], tmp_path) == "pass"
    assert verdict(eng, rid, ["groupby", "-grp", "1", "-i", "g.bed", *tail], tmp_path) == "pass"
    assert verdict(eng, rid, ["groupby", "-i", "g.bed", "-g", "1,4", *tail], tmp_path) == "fail"
    assert verdict(eng, rid, ["groupby", "-g", "4", *tail], tmp_path) == "pass"  # stdin
    (tmp_path / "c.bed").write_text("chr1\t1\t2\tG1\t5\nchr1\t5\t6\tg2\t3\nchr1\t6\t7\tg1\t3\n")
    case = ["groupby", "-i", "c.bed", "-g", "4", *tail]
    assert verdict(eng, rid, case, tmp_path) == "pass"  # G1 and g1 are different groups
    assert verdict(eng, rid, [*case, "-ignorecase"], tmp_path) == "fail"


# ---- bedtools.prec_significant -----------------------------------------------------------------


def test_prec_needs_a_declaration_and_decimal_places_contradict_the_tool(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    rid = "bedtools.prec_significant"
    base = ["map", "-a", "a.bed", "-b", "b.bed", "-c", "4", "-o", "sum", "-prec", "4"]
    decision, result = run(eng, rid, base, tmp_path)
    assert result.code == "declaration_missing" and decision.action == "block"
    sig, dec = {"precision_unit": "significant_digits"}, {"precision_unit": "decimal_places"}
    assert verdict(eng, rid, base, tmp_path, sig) == "pass"
    decision, result = run(eng, rid, base, tmp_path, dec)
    assert result.verdict == "fail" and "significant_digits" in result.message
    assert render_block(decision).startswith("FIX: use no -prec at all")
    assert verdict(eng, rid, base[:-2], tmp_path) is None  # no -prec, no rule
    assert (
        verdict(eng, rid, ["groupby", "-i", "a.bed", "-g", "4", "-prec", "2"], tmp_path, sig)
        == "pass"
    )


# ---- family workspaces -------------------------------------------------------------------------


def test_the_family_naive_calls_fail_on_the_faulted_twin_and_the_references_do_not(
    spec, tools, tmp_path
):
    eng = engine_for(spec, tools)
    cases = [
        (
            "F65",
            "bedtools.split_reciprocal",
            "intersect -a tx_a.bed -b tx_b.bed -split -r -f 0.9 -wa -u",
            None,
        ),
        (
            "F66",
            "bedtools.map_unweighted",
            "map -a windows.bed -b coverage.bedgraph -c 4 -o mean",
            "intersect -a windows.bed -b coverage.bedgraph -wo",
        ),
        (
            "F67",
            "bedtools.spliced_without_split",
            "coverage -a introns.bed -b reads.bam",
            "coverage -a introns.bed -b reads.bam -split",
        ),
        (
            "F68",
            "bedtools.subtract_trims",
            "subtract -a peaks.bed -b blacklist.bed",
            "subtract -a peaks.bed -b blacklist.bed -A",
        ),
        ("F69", "bedtools.window_boundary", "window -a peaks.bed -b genes.bed -w 1000 -u", None),
        (
            "F75",
            "bedtools.duplicate_policy",
            "coverage -a regions.bed -b sample.bam -counts",
            "multicov -bams sample.bam -bed regions.bed",
        ),
    ]
    for family, rid, naive, reference in cases:
        for twin in ("clean", "faulted"):
            work = tmp_path / f"{family}-{twin}"
            build_task(family, 6, twin, work, tools, "implicit")
            decision, result = run(eng, rid, naive.split(), work)
            expected_fail = twin == "faulted" and family not in ("F69", "F75")
            assert (result is not None and result.verdict == "fail") == expected_fail, (
                family,
                twin,
            )
            if family in ("F69", "F75") and twin == "faulted":
                assert result.verdict == "unknown"  # an intent-dependent rule only warns
            if reference:
                declared = {"duplicate_policy": "exclude"} if family == "F75" else None
                _, ref = run(eng, rid, reference.split(), work, declared)
                assert ref is None or ref.verdict == "pass", (family, twin)


# ---- admission ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_admission(spec, tools, tmp_path, rule_id):
    seeds = range(20) if rule_id == "bedtools.split_reciprocal" else range(ADMISSION_SEEDS)
    stats = measure(rule_id, [spec], seeds, tools, tmp_path)
    right_declared, right_undeclared = stats.cells[("right", True)], stats.cells[("right", False)]
    assert stats.admitted() and stats.conditional is (rule_id not in REQUIRED)
    assert right_declared.n >= MIN_RIGHT_CALLS * len(seeds) // ADMISSION_SEEDS
    assert right_declared.failed == 0
    if rule_id in REQUIRED:  # a required declaration: omitting it is a failure by design
        assert (
            0 < right_undeclared.failed < right_undeclared.n
        )  # calls without -prec are not judged
    else:
        assert right_undeclared.failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    if len(seeds) == ADMISSION_SEEDS:
        assert promotion_is_consistent(stats)
