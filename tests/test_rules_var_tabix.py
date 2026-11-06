"""Rules of the variant pack for tabix: unit tests, evidence against tabix/bgzip 1.24, admission.

tabix.regions_file_base (-R/-T file named other than .bed holding BED windows) and
tabix.region_strings_bed (region strings against a zero-based index), both family F61. The
unit tests build small BED files; the evidence tests run the installed tabix and bgzip; the
admission tests measure the rules over ADMISSION_SEEDS seeds with calls labelled by running tabix
(bench/admission/cases_var_tabix.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import var_tabix
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_tabix")
RULES = {rule.id: rule for rule in SPEC.rules}
BASE = "tabix.regions_file_base"
STRINGS = "tabix.region_strings_bed"
BASE_KEY = "region_file_base"
COORD_KEY = "region_coordinates"
BED = "".join(
    f"chr1\t{s}\t{e}\t{n}\n"
    for s, e, n in (
        (50, 100, "r1"),
        (99, 120, "r2"),
        (100, 150, "r3"),
        (199, 200, "r4"),
        (200, 260, "r5"),
    )
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or {"tabix", "bgzip"} - set(env.real_bins(["tabix", "bgzip"])):
        pytest.skip("samtools, bcftools and tabix are not available")
    return env


def engine(tools):
    return Engine(
        [SPEC],
        mode="block",
        retry_cap=10**9,
        real_bins=tools.real_bins(["tabix", "bgzip"]),
        real_env=tools.process_env(),
    )


def verdict(tools, rule_id, argv, cwd, declared=None):
    decision = engine(tools).evaluate(Call("tabix", tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


@pytest.fixture()
def indexed(tmp_path, tools):
    (tmp_path / "iv.bed").write_text(BED)
    for argv in (["bgzip", "-f", "iv.bed"], ["tabix", "-f", "-p", "bed", "iv.bed.gz"]):
        assert tools.run(argv, tmp_path).returncode == 0
    return tmp_path


def run_tabix(tools, work, *argv):
    proc = tools.run(["tabix", *argv], work)
    assert proc.returncode == 0
    return [line.split("\t")[3] for line in proc.stdout.splitlines()]


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_declares_the_two_rules_with_the_expected_shape():
    assert set(RULES) == {BASE, STRINGS}
    for rule in RULES.values():
        assert (rule.kind, rule.tier, rule.slot) == ("call", "T1", "index_base")
        assert rule.applies.tools == ["tabix"] and rule.check in known_checkers()
        assert rule.declare is not None and not rule.declare.required
    assert RULES[BASE].incidents == ["GEN-01"] and RULES[STRINGS].incidents == ["GEN-04"]
    assert RULES[BASE].declare.key == BASE_KEY and RULES[STRINGS].declare.key == COORD_KEY


# ---- tabix.regions_file_base -------------------------------------------------------------------


def test_tabix_reads_a_regions_file_as_bed_only_by_its_name(indexed, tools):
    (indexed / "w.bed").write_text("chr1\t100\t200\n")
    shutil.copy(indexed / "w.bed", indexed / "w.tsv")
    assert run_tabix(tools, indexed, "iv.bed.gz", "-R", "w.bed") == ["r2", "r3", "r4"]
    assert run_tabix(tools, indexed, "iv.bed.gz", "-R", "w.tsv") == ["r1", "r2", "r3", "r4"]
    assert run_tabix(tools, indexed, "iv.bed.gz", "-T", "w.tsv") == ["r1", "r2", "r3", "r4"]


def test_a_bed_window_under_another_name_fails_with_the_dry_run_and_a_fix(indexed, tools):
    (indexed / "w.tsv").write_text("chr1\t100\t200\n")
    decision, result = verdict(tools, BASE, ["iv.bed.gz", "-R", "w.tsv"], indexed)
    assert decision.action == "block" and result.code == "bed_file_read_one_based"
    text = render_block(decision)
    assert text.startswith("FIX: cp w.tsv w.tsv.bed && use -R w.tsv.bed")
    assert "two readings print different records" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["iv.bed.gz", "-R", "w.tsv"],
        ["iv.bed.gz", "--regions", "w.tsv"],
        ["iv.bed.gz", "-T", "w.tsv"],
        ["-R", "w.tsv", "iv.bed.gz"],
        ["iv.bed.gz", "--regions=w.tsv"],
    ],
)
def test_spellings_of_the_regions_option_are_seen(indexed, tools, argv):
    (indexed / "w.tsv").write_text("chr1\t100\t200\n")
    assert verdict(tools, BASE, argv, indexed)[1].verdict == "fail"


def test_a_bed_name_one_based_content_and_equal_readings_are_silent(indexed, tools):
    (indexed / "w.bed").write_text("chr1\t100\t200\n")
    assert verdict(tools, BASE, ["iv.bed.gz", "-R", "w.bed"], indexed)[1].verdict == "pass"
    # two columns (chrom, pos): not BED-like
    (indexed / "p.txt").write_text("chr1\t100\n")
    assert verdict(tools, BASE, ["iv.bed.gz", "-R", "p.txt"], indexed)[1].verdict == "pass"
    # start == end is a valid 1-based position, not a BED interval
    (indexed / "s.txt").write_text("chr1\t100\t100\n")
    assert verdict(tools, BASE, ["iv.bed.gz", "-R", "s.txt"], indexed)[1].verdict == "pass"
    # a window no interval touches: both readings print the same records
    (indexed / "far.tsv").write_text("chr1\t300\t400\n")
    result = verdict(tools, BASE, ["iv.bed.gz", "-R", "far.tsv"], indexed)[1]
    assert result.verdict == "pass" and "same records" in result.note


def test_regions_file_declarations(indexed, tools):
    (indexed / "w.tsv").write_text("chr1\t100\t200\n")
    argv = ["iv.bed.gz", "-R", "w.tsv"]
    assert verdict(tools, BASE, argv, indexed, {BASE_KEY: "one_based_closed"})[1].verdict == "pass"
    result = verdict(tools, BASE, argv, indexed, {BASE_KEY: "zero_based_half_open"})[1]
    assert result.verdict == "fail"
    (indexed / "w.bed").write_text("chr1\t100\t200\n")
    argv = ["iv.bed.gz", "-R", "w.bed"]
    result = verdict(tools, BASE, argv, indexed, {BASE_KEY: "one_based_closed"})[1]
    assert result.code == "declared_one_based_but_bed_name"


def test_unreadable_regions_file_is_unknown(indexed, tools):
    decision, result = verdict(tools, BASE, ["iv.bed.gz", "-R", "absent.tsv"], indexed)
    assert result.verdict == "unknown" and decision.action == "warn"


# ---- tabix.region_strings_bed ------------------------------------------------------------------


def test_tabix_region_strings_are_one_based_for_a_bed_index(indexed, tools):
    assert run_tabix(tools, indexed, "iv.bed.gz", "chr1:100-200") == ["r1", "r2", "r3", "r4"]
    assert run_tabix(tools, indexed, "iv.bed.gz", "chr1:101-200") == ["r2", "r3", "r4"]


def test_a_raw_bed_window_string_fails_and_the_converted_one_does_not(indexed, tools):
    decision, result = verdict(tools, STRINGS, ["iv.bed.gz", "chr1:100-200"], indexed)
    assert decision.action == "block" and result.code == "bed_region_string_base_undeclared"
    text = render_block(decision)
    assert "chr1:101-200" in text and "iv.bed.gz chr1:100-200" in text
    assert "prints different records" in text
    # a string that is already converted: moving it up once more changes nothing on this data, so
    # the rule is silent even undeclared; it fails when the shift would change the output
    assert verdict(tools, STRINGS, ["iv.bed.gz", "chr1:101-200"], indexed)[1].verdict == "pass"
    assert verdict(tools, STRINGS, ["iv.bed.gz", "chr1:130-190"], indexed)[1].verdict == "pass"
    assert verdict(tools, STRINGS, ["iv.bed.gz", "chr1:200-250"], indexed)[1].verdict == "fail"


def test_region_string_declarations(indexed, tools):
    argv = ["iv.bed.gz", "chr1:101-200"]
    assert verdict(tools, STRINGS, argv, indexed, {COORD_KEY: "one_based_closed"})[1].verdict == (
        "pass"
    )
    argv = ["iv.bed.gz", "chr1:100-200"]
    decision, result = verdict(tools, STRINGS, argv, indexed, {COORD_KEY: "zero_based_half_open"})
    assert result.code == "declared_zero_based_region_string" and decision.action == "block"
    assert "you declared region_coordinates=zero_based_half_open" in render_block(decision)


def test_other_tabix_calls_are_not_decided(indexed, tools):
    (indexed / "t.vcf").write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    for argv in (
        ["-p", "bed", "iv.bed.gz"],
        ["-l", "iv.bed.gz"],
        ["iv.bed.gz", "chr1"],
        ["iv.bed.gz"],
        ["-p", "bed", "iv.bed.gz", "chr1:100-200"],
    ):
        result = verdict(tools, STRINGS, argv, indexed)[1]
        assert result is None or result.verdict == "pass", argv


def test_a_vcf_index_is_one_based_and_silent(tmp_path, tools):
    rows = "".join(f"chr1\t{p}\t.\tA\tC\t50\tPASS\t.\n" for p in (100, 150, 200))
    (tmp_path / "v.vcf").write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + rows
    )
    for argv in (["bgzip", "-f", "v.vcf"], ["tabix", "-f", "-p", "vcf", "v.vcf.gz"]):
        assert tools.run(argv, tmp_path).returncode == 0
    assert (
        var_tabix.tbi_zero_based(
            __import__("biowarden.engine.checks", fromlist=["Ctx"]).Ctx(tmp_path, "T1"),
            "v.vcf.gz",
        )
        is False
    )
    assert verdict(tools, STRINGS, ["v.vcf.gz", "chr1:100-200"], tmp_path)[1].verdict == "pass"


# ---- F61 workspaces and admission --------------------------------------------------------------


def test_the_rules_on_the_f61_workspaces(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F61", 4, twin, work, tools, "implicit")
        for argv in (
            ["bgzip", "-k", "-f", "intervals.bed"],
            ["tabix", "-f", "-p", "bed", "intervals.bed.gz"],
        ):
            assert tools.run(argv, work).returncode == 0
        got = verdict(tools, BASE, ["intervals.bed.gz", "-R", "windows.tsv"], work)[1]
        assert got.verdict == expected
        rows = [r.split("\t") for r in (work / "windows.tsv").read_text().splitlines()]
        strings = [f"{c}:{s}-{e}" for c, s, e in rows]
        assert verdict(tools, STRINGS, ["intervals.bed.gz", *strings], work)[1].verdict == expected
        moved = [f"{c}:{int(s) + 1}-{e}" for c, s, e in rows]
        assert (
            verdict(
                tools, STRINGS, ["intervals.bed.gz", *moved], work, {COORD_KEY: "one_based_closed"}
            )[1].verdict
            == "pass"
        )


def check_stats(stats, wrong_undeclared_min):
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= wrong_undeclared_min
    assert promotion_is_consistent(stats)


def test_admission_regions_file_base(tools, tmp_path):
    stats = measure(BASE, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    check_stats(stats, ADMISSION_SEEDS * 4)


def test_admission_region_strings_bed(tools, tmp_path):
    stats = measure(STRINGS, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    check_stats(stats, ADMISSION_SEEDS * 3)
