"""bcftools regions rules: the base of a -R/-T file (GEN-01) and overlap versus start-only (GEN-02).

Unit tests of both checkers on real argv, spec and tier checks, evidence tests that run the
installed bcftools (skipped without it) to tie the checkers' tables to the tool, and the admission
measurement over 300 seeds for each rule.
"""

from __future__ import annotations

import gzip
import os
import re
import shlex
import struct
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.common import write_bam, write_fasta, write_vcf_text
from biowarden.engine import Call, Engine, Spec, known_checkers
from biowarden.engine.checks import Ctx, Match, TierViolation
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules import bcftools_regions as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
BASE = "bcftools.regions_file_base"
SEMANTICS = "bcftools.region_semantics"
SPEC = load_packaged("bcftools_regions")
RULES = {r.id: r for r in SPEC.rules}
BED_TEXT = "chr1\t100\t150\nchr1\t300\t350\n"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def only(rule_id: str) -> Spec:
    data = SPEC.model_dump(mode="json")
    data["rules"] = [r for r in data["rules"] if r["id"] == rule_id]
    return Spec.model_validate(data)


def engine(rule_id: str | None = None, tools: ToolEnv | None = None, mode="block") -> Engine:
    spec = only(rule_id) if rule_id else SPEC
    bins = tools.real_bins() if tools else {}
    env = tools.process_env() if tools else None
    return Engine([spec], mode=mode, real_bins=bins, real_env=env, retry_cap=10**9)


def outcome(rule_id, argv, cwd, declared=None, tools=None):
    eng = engine(rule_id, tools)
    decision = eng.evaluate(Call("bcftools", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0] if decision.outcomes else None)


def gz(text: str) -> bytes:
    return gzip.compress(text.encode(), mtime=0)


# ---- the spec ------------------------------------------------------------------------------


def test_spec_is_valid_and_registers_both_checkers():
    assert SPEC.name == "bcftools_regions" and set(RULES) == {BASE, SEMANTICS}
    assert {"bcftools_regions_file_base", "bcftools_region_semantics"} <= set(known_checkers())
    Engine([SPEC])  # every checker name resolves


def test_slots_families_and_tiers_are_as_designed():
    base, semantics = RULES[BASE], RULES[SEMANTICS]
    assert (base.slot, base.kind, base.tier, base.incidents) == (
        "index_base",
        "call",
        "T1",
        ["GEN-01"],
    )
    assert (semantics.slot, semantics.kind, semantics.tier, semantics.incidents) == (
        "interval_semantics",
        "call",
        "T0",
        ["GEN-02"],
    )
    assert base.declare is not None and base.declare.required is False
    assert base.declare.values == ["zero_based_half_open", "one_based_closed"]
    assert semantics.declare is not None and semantics.declare.required is True
    assert semantics.declare.values == ["overlap", "start_only"]


def test_only_the_semantics_declaration_is_required_in_the_schema():
    schema = SPEC.declaration_schema()
    assert schema["required"] == ["region_semantics"]
    assert schema["properties"]["region_file_base"]["enum"] == [
        "zero_based_half_open",
        "one_based_closed",
    ]


def test_applies_lists_cover_the_subcommands_that_take_regions_and_leave_the_others():
    semantics = RULES[SEMANTICS].applies
    assert {"view", "query", "filter", "annotate", "stats", "norm", "concat", "call"} <= set(
        semantics.subcommands
    )
    assert "mpileup" not in semantics.subcommands and "sort" not in semantics.subcommands
    assert "mpileup" in RULES[BASE].params["subcommands"]
    assert set(RULES[BASE].applies.options_any) == {"-R", "--regions-file", "-T", "--targets-file"}


# ---- tier honesty ----------------------------------------------------------------------------


def test_the_semantics_checker_is_t0_and_touches_no_file():
    ctx = Ctx(Path("/nonexistent-workspace"), "T0")
    call = Call(
        "bcftools", ("view", "-T", "r.bed", "in.vcf.gz"), ctx.cwd, {"region_semantics": "overlap"}
    )
    result = known_checkers()["bcftools_region_semantics"](call, Match(RULES[SEMANTICS]), ctx)
    assert result.verdict == "fail" and result.code == "declared_mismatch"
    with pytest.raises(TierViolation):
        list(ctx.lines("r.bed"))


def test_the_base_checker_needs_t1_to_read_the_file(tmp_path):
    (tmp_path / "r.tsv").write_text(BED_TEXT)
    call = Call("bcftools", ("view", "-R", "r.tsv", "in.vcf.gz"), tmp_path, {})
    with pytest.raises(TierViolation):
        known_checkers()["bcftools_regions_file_base"](
            call, Match(RULES[BASE]), Ctx(tmp_path, "T0")
        )


def test_a_name_that_already_decides_the_reading_is_decided_without_reading(tmp_path):
    call = Call("bcftools", ("view", "-R", "r.bed", "in.vcf.gz"), tmp_path, {})
    ctx = Ctx(tmp_path, "T0")  # would raise if the checker tried to read
    assert (
        known_checkers()["bcftools_regions_file_base"](call, Match(RULES[BASE]), ctx).verdict
        == "pass"
    )


# ---- rule 1: parsing the file options -------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["view", "-R", "a.bed"], [("-R", "a.bed", False)]),
        (["view", "-Ra.bed"], [("-R", "a.bed", False)]),
        (["view", "--regions-file", "a.bed"], [("-R", "a.bed", False)]),
        (["view", "--regions-file=a.bed"], [("-R", "a.bed", False)]),
        (["view", "-T", "t.tsv"], [("-T", "t.tsv", False)]),
        (["view", "--targets-file=t.tsv"], [("-T", "t.tsv", False)]),
        (["view", "-T", "^t.tsv"], [("-T", "t.tsv", True)]),
        (["view", "-R", "a", "-T", "b"], [("-R", "a", False), ("-T", "b", False)]),
        (["view", "-R"], []),
        (["view", "-r", "chr1:1-5", "-t", "chr1"], []),
        (["+fill-tags", "in.vcf", "-R", "a", "--", "-t", "AF"], [("-R", "a", False)]),
        (["view", "--", "-R", "a"], []),
    ],
)
def test_region_files_reads_every_spelling(argv, expected):
    got = [(f.option, f.name, f.exclude) for f in rules.region_files(argv)]
    assert got == expected


def test_a_caret_is_an_exclusion_marker_only_for_targets():
    assert rules.region_files(["view", "-R", "^a.bed"])[0].name == "^a.bed"
    assert rules.region_files(["view", "-T", "^a.bed"])[0].name == "a.bed"


# ---- rule 1: the safe state costs nothing -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["regions.bed", "REGIONS.BED", "r.Bed", "regions.bed.gz", "r.BED.GZ", "r.Bed.Gz", "sub/x.bed"],
)
@pytest.mark.parametrize(
    "declared",
    [{}, {"region_file_base": "zero_based_half_open"}, {"region_file_base": "one_based_closed"}],
)
def test_a_bed_name_passes_with_or_without_a_declaration_and_is_never_read(
    tmp_path, name, declared
):
    # the file does not even exist: the name alone decides, so nothing is opened
    _, out = outcome(BASE, ["view", "-H", "-R", name, "v.vcf.gz"], tmp_path, declared)
    assert out.result.verdict == "pass" and out.effective == "none"
    assert "ends in .bed" in out.result.note


@pytest.mark.parametrize("option", [["-T"], ["--targets-file"], ["--regions-file"]])
def test_every_spelling_of_the_option_is_covered(tmp_path, option):
    (tmp_path / "r.tsv").write_text(BED_TEXT)
    _, out = outcome(BASE, ["view", *option, "r.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "fail"


def test_other_tools_and_other_options_are_not_matched(tmp_path):
    eng = engine(BASE)
    assert eng.evaluate(Call("samtools", ("view", "-R", "r.tsv"), tmp_path)).outcomes == ()
    assert eng.evaluate(Call("bcftools", ("view", "-r", "chr1:5-9", "v"), tmp_path)).outcomes == ()


@pytest.mark.parametrize(
    "argv",
    [["sort", "-T", "/tmp/x", "v.vcf.gz"], ["reheader", "-T", "p", "-h", "h.txt", "v.bcf"]],
)
def test_the_t_option_of_other_subcommands_is_left_alone(tmp_path, argv):
    _, out = outcome(BASE, argv, tmp_path)
    assert out.result.verdict == "pass" and "does not read a regions file" in out.result.note


# ---- rule 1: the risky state --------------------------------------------------------------------


def risky(tmp_path, name="regions.tsv", text=BED_TEXT):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_undeclared_bed_content_under_a_tsv_name_fails_with_a_corrected_command(tmp_path):
    risky(tmp_path)
    decision, out = outcome(BASE, ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"], tmp_path)
    assert (out.result.verdict, out.result.code) == ("fail", "bed_content_read_one_based")
    assert "1-based inclusive" in out.result.message and "BED intervals" in out.result.message
    assert "regions.tsv" in out.result.message and "2 data lines" in out.result.message
    text = render_block(decision)
    assert text.splitlines()[0].startswith(
        "FIX: cp regions.tsv regions.tsv.bed && bcftools view -H -R regions.tsv.bed variants.vcf.gz"
    )
    assert "--declare region_file_base=one_based_closed" in text.splitlines()[0]


def test_declaring_one_based_closed_passes_in_the_risky_state(tmp_path):
    risky(tmp_path)
    declared = {"region_file_base": "one_based_closed"}
    _, out = outcome(BASE, ["view", "-R", "regions.tsv", "v.vcf.gz"], tmp_path, declared)
    assert out.result.verdict == "pass"


def test_declaring_zero_based_half_open_fails_with_the_rename_fix(tmp_path):
    risky(tmp_path)
    declared = {"region_file_base": "zero_based_half_open"}
    decision, out = outcome(
        BASE, ["query", "-f", "%ID\\n", "-T", "regions.tsv", "v.gz"], tmp_path, declared
    )
    assert (out.result.verdict, out.result.code) == ("fail", "declared_mismatch")
    assert out.result.message.startswith("you declared region_file_base=zero_based_half_open")
    first = render_block(decision).splitlines()[0]
    assert first.startswith(
        "FIX: cp regions.tsv regions.tsv.bed && bcftools query -f '%ID\\n' -T regions.tsv.bed v.gz"
    )


def test_the_fix_keeps_the_spelling_of_the_option_and_the_exclusion_marker(tmp_path):
    risky(tmp_path)
    cases = {
        ("view", "--regions-file=regions.tsv"): "bcftools view --regions-file=regions.tsv.bed",
        ("view", "-Rregions.tsv"): "bcftools view -Rregions.tsv.bed",
        ("view", "--targets-file", "regions.tsv"): "bcftools view --targets-file regions.tsv.bed",
        ("view", "-T", "^regions.tsv"): "bcftools view -T '^regions.tsv.bed'",
    }
    for argv, expected in cases.items():
        _, out = outcome(BASE, [*argv, "v.vcf.gz"], tmp_path)
        assert out.result.verdict == "fail", argv
        assert expected in out.result.fix[0], argv


def test_a_gzip_file_gets_a_bed_gz_copy_name(tmp_path):
    (tmp_path / "regions.tsv.gz").write_bytes(gz(BED_TEXT))
    _, out = outcome(BASE, ["view", "-R", "regions.tsv.gz", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "fail"
    assert out.result.fix[0].startswith("cp regions.tsv.gz regions.tsv.bed.gz && ")


def test_a_path_in_another_directory_is_copied_into_the_working_directory(tmp_path):
    risky(tmp_path, "sub/regions.tsv")
    _, out = outcome(BASE, ["view", "-R", "sub/regions.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.fix[0].startswith(
        "cp sub/regions.tsv regions.tsv.bed && bcftools view -R regions.tsv.bed"
    )


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("chr1\t100\t200\n", "fail"),
        ("chr1\t100\t200\tname\t0\t+\n", "fail"),  # BED6: the first three columns decide
        ("# header\nchr1\t100\t200\n", "fail"),
        ("chr1\t100\t200\r\nchr1\t300\t400\r\n", "fail"),
        ("chr1\t0\t200\n", "fail"),  # a start of 0 only exists in BED
        ("chr1\t100\t100\n", "pass"),  # start == end: a 1-based single base, an empty BED interval
        ("chr1\t100\t200\nchr1\t300\t300\n", "pass"),
        ("chr1\t100\n", "pass"),  # CHROM POS positions
        ("chr1\t100\trs1\tA\tC\n", "pass"),  # a sites file
        ("chr1\t200\t100\n", "pass"),
        ("chrom\tstart\tend\nchr1\t100\t200\n", "pass"),  # bcftools stops with "Could not parse"
        ("chr1 100 200\n", "pass"),  # space separated: bcftools stops with "Could not parse"
        ("", "pass"),
        ("# only a comment\n", "pass"),
    ],
)
def test_content_decides_whether_a_non_bed_name_is_risky(tmp_path, text, verdict):
    risky(tmp_path, text=text)
    _, out = outcome(BASE, ["view", "-R", "regions.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == verdict


def test_a_file_with_no_inspectable_content_is_unknown_and_only_warns(tmp_path):
    decision, out = outcome(BASE, ["view", "-R", "later.tsv", "v.vcf.gz"], tmp_path)
    assert (out.result.verdict, out.result.code) == ("unknown", "regions_file_unreadable")
    assert "does not exist (yet)" in out.result.message and out.effective == "warn"
    assert decision.action == "warn" and render_warning(decision).startswith("UNVERIFIED")


def test_stdin_a_directory_and_a_fifo_are_unknown_and_never_opened(tmp_path):
    os.mkfifo(tmp_path / "pipe.tsv")  # opening it for reading would block forever
    (tmp_path / "dir.tsv").mkdir()
    for name in ("-", "dir.tsv", "pipe.tsv", "/dev/fd/63"):
        _, out = outcome(BASE, ["view", "-R", name, "v.vcf.gz"], tmp_path)
        assert out.result.verdict == "unknown", name


def test_a_stated_bed_intent_fails_even_when_the_file_cannot_be_read(tmp_path):
    declared = {"region_file_base": "zero_based_half_open"}
    for name in ("later.tsv", "-"):
        _, out = outcome(BASE, ["view", "-R", name, "v.vcf.gz"], tmp_path, declared)
        assert out.result.verdict == "fail", name
    _, out = outcome(BASE, ["view", "-R", "-", "v.vcf.gz"], tmp_path, declared)
    assert "named regions.bed" in out.result.fix[0]


def test_a_broken_gzip_file_is_unknown_not_a_crash(tmp_path):
    (tmp_path / "broken.tsv.gz").write_bytes(b"\x1f\x8b\x08\x00 not really gzip")
    _, out = outcome(BASE, ["view", "-R", "broken.tsv.gz", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "unknown" and "could not be read" in out.result.message


def test_one_failing_file_decides_even_when_another_cannot_be_read(tmp_path):
    risky(tmp_path, "bad.tsv")
    risky(tmp_path, "ok.bed")
    _, out = outcome(BASE, ["view", "-R", "missing.tsv", "-T", "bad.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "fail" and "bad.tsv" in out.result.message
    _, out = outcome(BASE, ["view", "-R", "ok.bed", "-T", "missing.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "unknown"


def test_an_invalid_declaration_value_is_rejected_by_the_engine(tmp_path):
    _, out = outcome(BASE, ["view", "-R", "r.bed", "v"], tmp_path, {"region_file_base": "zero"})
    assert out.result.code == "declaration_invalid"


def test_plugins_and_other_subcommands_that_take_regions_are_covered(tmp_path):
    risky(tmp_path)
    for argv in (
        ["+fill-tags", "-R", "regions.tsv", "v.vcf.gz", "--", "-t", "AF"],
        ["plugin", "fill-tags", "-T", "regions.tsv", "v.vcf.gz"],
        ["mpileup", "-f", "ref.fa", "-R", "regions.tsv", "a.bam"],
        ["stats", "-R", "regions.tsv", "v.vcf.gz"],
        ["norm", "-m-any", "-T", "regions.tsv", "v.vcf.gz"],
        ["annotate", "-R", "regions.tsv", "v.vcf.gz"],
        ["call", "-mv", "-R", "regions.tsv", "v.vcf.gz"],
        ["isec", "-R", "regions.tsv", "a.vcf.gz", "b.vcf.gz"],
    ):
        _, out = outcome(BASE, argv, tmp_path)
        assert out.result.verdict == "fail", argv


def test_call_targets_is_a_sites_file_so_the_bed_rule_does_not_apply(tmp_path):
    risky(tmp_path)
    _, out = outcome(BASE, ["call", "-mv", "-T", "regions.tsv", "v.vcf.gz"], tmp_path)
    assert (out.result.verdict, out.result.code) == ("unknown", "call_targets_are_sites")
    assert out.result.fix[0] == (
        "cp regions.tsv regions.tsv.bed && bcftools call -mv -R regions.tsv.bed v.vcf.gz"
    )
    risky(tmp_path, "regions.bed")
    _, out = outcome(BASE, ["call", "-mv", "--targets-file=regions.bed", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "unknown"
    assert out.result.fix[0] == "bcftools call -mv --regions-file=regions.bed v.vcf.gz"
    risky(tmp_path, "sites.tsv", "chr1\t100\tA,C\n")  # a real sites file: CHROM POS REF,ALT
    _, out = outcome(
        BASE, ["call", "-mv", "-C", "alleles", "-T", "sites.tsv", "v.vcf.gz"], tmp_path
    )
    assert out.result.verdict == "pass"
    _, out = outcome(BASE, ["call", "-mv", "-T", "missing.tsv", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "pass"  # nothing to inspect: no claim is made


# ---- rule 1: a tabix or CSI index next to the file decides, not the name ----------------------


def tbi(preset: int) -> bytes:
    head = struct.pack("<4s8i", b"TBI\x01", 0, preset, 1, 2, 3, ord("#"), 0, 0)
    return gzip.compress(head, mtime=0)


def csi(preset: int, l_aux: int = 28) -> bytes:
    head = struct.pack("<4s3i", b"CSI\x01", 14, 6, l_aux)
    aux = struct.pack("<7i", preset, 1, 2, 3, ord("#"), 0, 0)[:l_aux]
    return gzip.compress(head + aux + struct.pack("<i", 0), mtime=0)


@pytest.mark.parametrize("make", [tbi, csi])
def test_a_bed_preset_index_makes_a_tsv_name_safe(tmp_path, make):
    (tmp_path / "regions.tsv.gz").write_bytes(gz(BED_TEXT))
    suffix = ".tbi" if make is tbi else ".csi"
    (tmp_path / f"regions.tsv.gz{suffix}").write_bytes(make(0x10000))
    _, out = outcome(BASE, ["view", "-R", "regions.tsv.gz", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "pass" and "BED (zero-based) preset" in out.result.note


@pytest.mark.parametrize("make", [tbi, csi])
def test_a_one_based_preset_index_makes_a_bed_name_risky(tmp_path, make):
    (tmp_path / "regions.bed.gz").write_bytes(gz(BED_TEXT))
    suffix = ".tbi" if make is tbi else ".csi"
    (tmp_path / f"regions.bed.gz{suffix}").write_bytes(make(0))
    _, out = outcome(BASE, ["view", "-R", "regions.bed.gz", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "fail"
    assert "tabix index does not have the BED" in out.result.message


def test_a_csi_without_tabix_settings_is_ignored_and_a_garbled_index_is_unknown(tmp_path):
    (tmp_path / "regions.bed").write_text(BED_TEXT)
    (tmp_path / "regions.bed.csi").write_bytes(csi(0, l_aux=0))
    _, out = outcome(BASE, ["view", "-R", "regions.bed", "v.vcf.gz"], tmp_path)
    assert out.result.verdict == "pass" and "ends in .bed" in out.result.note
    (tmp_path / "regions.bed.tbi").write_bytes(b"garbage")
    _, out = outcome(BASE, ["view", "-R", "regions.bed", "v.vcf.gz"], tmp_path)
    assert (out.result.verdict, out.result.code) == ("unknown", "index_unreadable")


# ---- rule 1: the dry run -----------------------------------------------------------------------


def vcf_workspace(tools, tmp_path, positions, regions=BED_TEXT, name="regions.tsv", ref="A"):
    header = "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
    body = "".join(f"chr1\t{p}\tv{p}\t{ref}\tC\t.\t.\t.\n" for p in positions)
    for old in tmp_path.glob("variants.vcf.gz*"):
        old.unlink()
    write_vcf_text(tmp_path, header, body, tools)
    (tmp_path / name).write_text(regions)


def test_the_dry_run_passes_when_both_readings_select_the_same_records(tools, tmp_path):
    vcf_workspace(tools, tmp_path, [125, 320, 700])  # none on an edge: 1-based vs BED agree
    decision, out = outcome(
        BASE, ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"], tmp_path, tools=tools
    )
    assert out.result.verdict == "pass" and "same records" in out.result.note
    assert decision.action == "allow"


def test_the_dry_run_fails_with_the_record_counts_when_the_readings_differ(tools, tmp_path):
    vcf_workspace(tools, tmp_path, [100, 125, 320, 700])  # 100 is only inside the 1-based reading
    decision, out = outcome(
        BASE, ["view", "-H", "-T", "regions.tsv", "variants.vcf.gz"], tmp_path, tools=tools
    )
    assert out.result.verdict == "fail"
    assert (
        "read as 1-based it selects 3 records of variants.vcf.gz, read as BED 2"
        in out.result.message
    )
    assert decision.action == "block"


def test_the_dry_run_covers_the_declared_zero_based_case_too(tools, tmp_path):
    vcf_workspace(tools, tmp_path, [125, 320])
    declared = {"region_file_base": "zero_based_half_open"}
    _, out = outcome(
        BASE, ["view", "-R", "regions.tsv", "variants.vcf.gz"], tmp_path, declared, tools
    )
    assert out.result.verdict == "pass" and "same records" in out.result.note


def test_the_dry_run_follows_the_overlap_mode_of_the_call(tools, tmp_path):
    # a deletion at POS 99 (REF 3 bases) reaches into the 1-based region from the left
    vcf_workspace(tools, tmp_path, [99], ref="ACG")
    argv = ["view", "-H", "regions.tsv", "variants.vcf.gz"]
    _, overlapping = outcome(BASE, [*argv[:2], "-R", *argv[2:]], tmp_path, tools=tools)
    _, start_only = outcome(BASE, [*argv[:2], "-T", *argv[2:]], tmp_path, tools=tools)
    assert overlapping.result.verdict == "pass"  # -R overlaps: the record is in both readings
    assert start_only.result.verdict == "pass"  # -T: POS 99 is outside both readings
    vcf_workspace(tools, tmp_path, [100], ref="ACG")
    _, start_only = outcome(BASE, [*argv[:2], "-T", *argv[2:]], tmp_path, tools=tools)
    assert start_only.result.verdict == "fail"  # POS 100: only the 1-based reading keeps it


def test_without_a_usable_dry_run_the_static_verdict_stands(tools, tmp_path):
    vcf_workspace(tools, tmp_path, [125, 320])  # the readings would agree
    cases = [
        ["view", "-H", "-r", "chr1", "-R", "regions.tsv", "variants.vcf.gz"],  # -r mixed in
        ["view", "-H", "-T", "^regions.tsv", "variants.vcf.gz"],  # exclusion
        ["view", "-H", "-R", "regions.tsv", "missing.vcf.gz"],  # input not there
        ["view", "-H", "-R", "regions.tsv", "-"],  # stdin
        ["isec", "-R", "regions.tsv", "variants.vcf.gz", "variants.vcf.gz"],  # several inputs
    ]
    for argv in cases:
        _, out = outcome(BASE, argv, tmp_path, tools=tools)
        assert out.result.verdict == "fail", argv
    # and with no real binary known at all
    _, out = outcome(BASE, ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"], tmp_path)
    assert out.result.verdict == "fail"


def test_a_long_region_file_skips_the_dry_run(tools, tmp_path):
    vcf_workspace(tools, tmp_path, [125])
    many = "".join(f"chr1\t{i}\t{i + 5}\n" for i in range(1000, 1700, 1))
    (tmp_path / "regions.tsv").write_text(many)
    _, out = outcome(
        BASE, ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"], tmp_path, tools=tools
    )
    assert out.result.verdict == "fail" and "selects" not in out.result.message


# ---- rule 2: what the options select ----------------------------------------------------------

OV, ST = {"region_semantics": "overlap"}, {"region_semantics": "start_only"}
DEC_OV, DEC_ST = "--declare region_semantics=overlap", "--declare region_semantics=start_only"


def semantic_verdict(argv, declared):
    _, out = outcome(SEMANTICS, argv, Path("/nonexistent-workspace"), declared)
    return out.result.verdict, out.result.code


@pytest.mark.parametrize(
    ("argv", "declared", "expected"),
    [
        (["view", "-H", "-R", "r.bed", "v.vcf.gz"], OV, ("pass", "ok")),
        (["view", "-H", "-R", "r.bed", "v.vcf.gz"], ST, ("fail", "declared_mismatch")),
        (["view", "-H", "-T", "r.bed", "v.vcf.gz"], OV, ("fail", "declared_mismatch")),
        (["view", "-H", "-T", "r.bed", "v.vcf.gz"], ST, ("pass", "ok")),
        (["view", "-H", "-Rr.bed", "v.vcf.gz"], OV, ("pass", "ok")),
        (["view", "-H", "-Tr.bed", "v.vcf.gz"], OV, ("fail", "declared_mismatch")),
        (["view", "-H", "--regions-file", "r.bed", "v"], OV, ("pass", "ok")),
        (["view", "-H", "--regions-file=r.bed", "v"], ST, ("fail", "declared_mismatch")),
        (["view", "-H", "--targets-file", "r.bed", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-H", "--targets-file=r.bed", "v"], ST, ("pass", "ok")),
        (["view", "-H", "-r", "chr1:1-10", "v"], OV, ("pass", "ok")),
        (["view", "-H", "--regions", "chr1:1-10", "v"], ST, ("fail", "declared_mismatch")),
        (["view", "-H", "-t", "chr1:1-10", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-H", "--targets=chr1:1-10", "v"], ST, ("pass", "ok")),
        (["view", "-T", "r.bed", "--targets-overlap", "1", "v"], OV, ("pass", "ok")),
        (["view", "-T", "r.bed", "--targets-overlap=record", "v"], OV, ("pass", "ok")),
        (["view", "-T", "r.bed", "--targets-overlap", "2", "v"], OV, ("pass", "ok")),
        (["view", "-T", "r.bed", "--targets-overlap=variant", "v"], OV, ("pass", "ok")),
        (["view", "-T", "r.bed", "--targets-overlap", "1", "v"], ST, ("fail", "declared_mismatch")),
        (["view", "-T", "r.bed", "--targets-overlap", "0", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-T", "r.bed", "--targets-overlap=pos", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-R", "r.bed", "--regions-overlap", "0", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-R", "r.bed", "--regions-overlap=pos", "v"], ST, ("pass", "ok")),
        (["view", "-R", "r.bed", "--regions-overlap", "2", "v"], OV, ("pass", "ok")),
        (["view", "-r", "chr1:1-9", "--regions-overlap=variant", "v"], OV, ("pass", "ok")),
        # a -T overlap option does not change what -R keeps, and the other way round
        (["view", "-R", "r.bed", "--targets-overlap", "0", "v"], OV, ("pass", "ok")),
        (["view", "-T", "r.bed", "--regions-overlap", "1", "v"], OV, ("fail", "declared_mismatch")),
        # both kinds at once: a start-only part makes "overlap" untrue, "start_only" holds
        (["view", "-r", "chr1", "-T", "r.bed", "v"], OV, ("fail", "declared_mismatch")),
        (["view", "-r", "chr1", "-T", "r.bed", "v"], ST, ("pass", "ok")),
        (["view", "-R", "a.bed", "-t", "chr1:1-9", "v"], ST, ("pass", "ok")),
        (
            ["query", "-f", "%ID\\t%END\\n", "-T", "r.bed", "v.vcf.gz"],
            OV,
            ("fail", "declared_mismatch"),
        ),
        (["filter", "-i", "POS>0", "-R", "r.bed", "v.vcf.gz"], OV, ("pass", "ok")),
        (["stats", "-T", "r.bed", "v.vcf.gz"], ST, ("pass", "ok")),
        (["norm", "-m-any", "-T", "r.bed", "v.vcf.gz"], OV, ("fail", "declared_mismatch")),
        (["isec", "-r", "chr1", "a.vcf.gz", "b.vcf.gz"], ST, ("fail", "declared_mismatch")),
        (["plugin", "fill-tags", "-T", "r.bed", "v.vcf.gz"], OV, ("fail", "declared_mismatch")),
        (["view", "-T", "r.bed", "--targets-overlap", "3", "v"], OV, ("unknown", "overlap_value")),
        (
            ["view", "-T", "r.bed", "--targets-overlap=overlap", "v"],
            ST,
            ("unknown", "overlap_value"),
        ),
    ],
)
def test_the_declaration_must_agree_with_what_the_options_select(argv, declared, expected):
    assert semantic_verdict(argv, declared) == expected


def test_the_last_overlap_option_wins_as_in_bcftools():
    argv = ["view", "-T", "r.bed", "--targets-overlap", "0", "--targets-overlap", "1", "v"]
    assert semantic_verdict(argv, OV) == ("pass", "ok")


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "v.vcf.gz"],
        ["view", "--targets-overlap", "1", "v.vcf.gz"],  # an overlap option alone selects nothing
        ["mpileup", "-f", "ref.fa", "-R", "r.bed", "a.bam"],  # pileup positions, no overlap modes
        ["sort", "-T", "/tmp/x", "v.vcf.gz"],
        ["reheader", "-T", "p", "v.bcf"],
        ["index", "-t", "v.vcf.gz"],
    ],
)
def test_calls_without_a_selection_are_not_matched(argv):
    assert (
        engine(SEMANTICS).evaluate(Call("bcftools", tuple(argv), Path("/tmp"), {})).outcomes == ()
    )


def test_plugins_are_matched_through_the_glob_pattern():
    decision = engine(SEMANTICS).evaluate(
        Call("bcftools", ("+fill-tags", "-T", "r.bed", "v.vcf.gz"), Path("/tmp"))
    )
    (fired,) = decision.firing()
    assert fired.result.code == "declaration_missing"


def test_a_missing_declaration_is_a_failure_with_two_complete_commands_as_the_fix():
    decision = engine(SEMANTICS).evaluate(
        Call("bcftools", ("view", "-T", "r.bed", "v"), Path("/tmp"))
    )
    (fired,) = decision.firing()
    assert fired.result.code == "declaration_missing"
    first = render_block(decision).splitlines()[0]
    # the suggester words the fix for THIS call (E-2): what the call keeps now, then two labelled
    # complete commands, overlap first, and what --declare is
    assert first.startswith(
        "FIX: -T r.bed (--targets-overlap defaults to 0) keeps only records whose POS lies in a "
        "region. Replace the bcftools call with one of these two, exactly as written: "
    )
    overlap = "bcftools view --targets-overlap 1 -T r.bed v --declare region_semantics=overlap"
    start_only = "bcftools view -T r.bed v --declare region_semantics=start_only"
    assert f"{overlap} (overlap: " in first and f"or {start_only} (start_only, your call" in first
    assert first.index(overlap) < first.index(start_only)
    assert first.endswith(
        "--declare is not a bcftools option: the sandbox checker reads it and removes it before "
        "bcftools runs. Setting --regions-overlap or --targets-overlap alone does not lift this "
        "block."
    )
    assert "only records whose POS lies in one" in fired.result.message


def test_the_checker_called_directly_without_a_declaration_fails_the_same_way():
    call = Call("bcftools", ("view", "-R", "r.bed", "v"), Path("/tmp"), {})
    result = known_checkers()["bcftools_region_semantics"](
        call, Match(RULES[SEMANTICS]), Ctx(Path("/tmp"), "T0")
    )
    assert (result.verdict, result.code) == ("fail", "declaration_missing")


def test_a_start_only_call_declared_overlap_gets_the_corrected_command_first():
    argv = ("view", "-H", "-T", "regions.bed", "variants.vcf.gz")
    decision = engine(SEMANTICS, mode="block").evaluate(Call("bcftools", argv, Path("/tmp"), OV))
    text = render_block(decision)
    assert text.splitlines()[0] == (
        "FIX: bcftools view --targets-overlap 1 -H -T regions.bed variants.vcf.gz "
        "--declare region_semantics=overlap"
    )
    (fired,) = decision.firing()
    assert fired.result.message.startswith(
        "you declared region_semantics=overlap, but -T regions.bed "
        "(--targets-overlap defaults to 0) keeps only records whose START (POS) is inside a region"
    )


@pytest.mark.parametrize(
    ("argv", "declared", "fix"),
    [
        (
            ("view", "-T", "r.bed", "--targets-overlap", "0", "v.vcf.gz"),
            OV,
            f"bcftools view -T r.bed --targets-overlap 1 v.vcf.gz {DEC_OV}",
        ),
        (
            ("view", "-R", "r.bed", "--regions-overlap=pos", "v.vcf.gz"),
            OV,
            f"bcftools view -R r.bed --regions-overlap=1 v.vcf.gz {DEC_OV}",
        ),
        (
            ("view", "-t", "chr1:1-9", "v.vcf.gz"),
            OV,
            f"bcftools view --targets-overlap 1 -t chr1:1-9 v.vcf.gz {DEC_OV}",
        ),
        (
            ("call", "-mv", "-T", "r.bed", "v.vcf.gz"),
            OV,
            "bcftools call -mv -R r.bed v.vcf.gz --declare region_semantics=overlap",
        ),
        (
            ("view", "-R", "r.bed", "v.vcf.gz"),
            ST,
            f"bcftools view --regions-overlap 0 -R r.bed v.vcf.gz {DEC_ST}",
        ),
        (
            ("view", "-r", "chr1", "-T", "r.bed", "--targets-overlap", "2", "v.vcf.gz"),
            ST,
            "bcftools view --regions-overlap 0 -r chr1 -T r.bed --targets-overlap 0 v.vcf.gz "
            "--declare region_semantics=start_only",
        ),
    ],
)
def test_the_fix_is_a_complete_corrected_command(argv, declared, fix):
    _, out = outcome(SEMANTICS, argv, Path("/tmp"), declared)
    assert out.result.verdict == "fail" and out.result.fix[0] == fix


def test_the_corrected_command_passes_its_own_check():
    argv = ("view", "-H", "-T", "regions.bed", "variants.vcf.gz")
    _, out = outcome(SEMANTICS, argv, Path("/tmp"), OV)
    fixed = out.result.fix[0].split()
    assert fixed[0] == "bcftools"
    call = Call.from_argv("bcftools", fixed[1:], Path("/tmp"), env={})
    assert semantic_verdict(list(call.argv), dict(call.declared)) == ("pass", "ok")


# ---- rule 2: the two verified cases the options do not describe -------------------------------


def test_query_without_info_in_the_format_only_warns_and_suggests_the_end_column():
    argv = ("query", "-f", "%ID\\n", "-R", "regions.bed", "variants.vcf.gz")
    decision = engine(SEMANTICS, mode="block").evaluate(Call("bcftools", argv, Path("/tmp"), OV))
    (fired,) = decision.firing()
    assert (fired.result.verdict, fired.result.code) == ("unknown", "query_ignores_end")
    assert decision.action == "warn"  # unknown never blocks
    assert fired.result.fix[0] == (
        "bcftools query -f '%ID\\t%END\\n' -R regions.bed variants.vcf.gz "
        "--declare region_semantics=overlap"
    )
    assert render_warning(decision).startswith("UNVERIFIED bcftools.region_semantics")


@pytest.mark.parametrize(
    ("argv", "verdict"),
    [
        (["query", "-f", "%ID\\n", "-R", "r.bed", "v.vcf.gz"], "unknown"),
        (
            [
                "query",
                "-f",
                "%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\t%TYPE\\n",
                "-r",
                "chr1",
                "v.vcf",
            ],
            "unknown",
        ),
        (["query", "-f%ID\\n", "-T", "r.bed", "--targets-overlap", "1", "v.vcf.gz"], "unknown"),
        (["query", "--format", "%POS0\\t%ID\\n", "-R", "r.bed", "v.vcf.gz"], "unknown"),
        (["query", "-f", "%ID\\t%END\\n", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\t%INFO/END\\n", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\t%INFO\\n", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\t[%GT]\\n", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\t%AC\\n", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\n", "-i", "INFO/DP>3", "-R", "r.bed", "v.vcf.gz"], "pass"),
        (["query", "-f", "%ID\\n", "-R", "r.bed", "v.bcf"], "pass"),  # BCF carries the end
        (["query", "-f", "%ID\\n", "-R", "r.bed"], "pass"),  # input unknown: stdin
    ],
)
def test_query_warns_only_when_the_format_cannot_reveal_info_end(argv, verdict):
    assert semantic_verdict(argv, OV)[0] == verdict


def test_the_query_warning_is_for_the_overlap_declaration_only():
    argv = ["query", "-f", "%ID\\n", "-T", "r.bed", "v.vcf.gz"]
    assert semantic_verdict(argv, ST) == ("pass", "ok")
    assert semantic_verdict(argv, OV) == ("fail", "declared_mismatch")  # the options win


def test_merge_with_a_region_option_warns_that_info_end_is_ignored():
    argv = ["merge", "-R", "r.bed", "a.vcf.gz", "b.vcf.gz"]
    assert semantic_verdict(argv, OV) == ("unknown", "merge_ignores_end")
    assert semantic_verdict(argv, ST) == ("fail", "declared_mismatch")


# ---- evidence: the checkers' tables against the installed bcftools -----------------------------

VCF_LINES = [("p99", 99), ("p100", 100), ("p101", 101), ("p102", 102)]


def small_vcf(tools, work: Path) -> None:
    header = "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
    write_vcf_text(
        work, header, "".join(f"chr1\t{p}\t{i}\tA\tC\t.\t.\t.\n" for i, p in VCF_LINES), tools
    )


def ids_of(tools, work, argv, col=2):
    proc = tools.run(["bcftools", *argv], work)
    assert proc.returncode == 0, proc.stderr
    return [line.split("\t")[col] for line in proc.stdout.splitlines() if line and line[0] != "#"]


@pytest.mark.parametrize(
    "name",
    [
        "r.bed",
        "r.BED",
        "r.Bed",
        "r.tsv",
        "r.txt",
        "rbed",
        "r.bed.txt",
        "r.bed.bak",
        "x.bed ",
        "r.bed.bgz",
    ]
    + ["r.bed.gz", "r.BED.GZ", "r.Bed.Gz", "r.bed.gzip", "d.bed/regions"],
)
def test_the_name_rule_matches_what_the_installed_bcftools_does(tools, tmp_path, name):
    small_vcf(tools, tmp_path)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "chr1\t100\t101\n"  # BED: 1-based position 101 only; read 1-based: 100 and 101
    path.write_bytes(
        gz(text) if name.lower().rstrip().endswith((".gz", ".gzip", ".bgz")) else text.encode()
    )
    for option in ("-R", "-T"):
        got = ids_of(tools, tmp_path, ["view", "-H", option, name, "variants.vcf.gz"])
        reading, _ = rules._reading(path, name)
        assert got == (["p101"] if reading == "bed" else ["p100", "p101"]), (name, option)


def test_stdin_and_the_exclusion_marker_behave_as_documented(tools, tmp_path):
    small_vcf(tools, tmp_path)
    (tmp_path / "r.bed").write_text("chr1\t100\t101\n")
    (tmp_path / "r.tsv").write_text("chr1\t100\t101\n")
    proc = tools.run(
        ["bcftools", "view", "-H", "-R", "-", "variants.vcf.gz"], tmp_path, "chr1\t100\t101\n"
    )
    assert [ln.split("\t")[2] for ln in proc.stdout.splitlines()] == ["p100", "p101"]
    assert ids_of(tools, tmp_path, ["view", "-H", "-T", "^r.bed", "variants.vcf.gz"]) == [
        "p99",
        "p100",
        "p102",
    ]
    assert ids_of(tools, tmp_path, ["view", "-H", "-T", "^r.tsv", "variants.vcf.gz"]) == [
        "p99",
        "p102",
    ]
    failed = tools.run(["bcftools", "view", "-H", "-R", "^r.bed", "variants.vcf.gz"], tmp_path)
    assert failed.returncode != 0  # for -R the caret is part of the file name


GVCF = """##fileformat=VCFv4.2
##INFO=<ID=END,Number=1,Type=Integer,Description="end">
##ALT=<ID=NON_REF,Description="block">
##contig=<ID=chr1,length=1000>
"""
GVCF_BODY = (
    "chr1\t50\tblk\tA\t<NON_REF>\t.\t.\tEND=200\n"
    "chr1\t120\tsnv\tA\tC\t.\t.\t.\n"
    "chr1\t300\tfar\tA\tC\t.\t.\t.\n"
)


def gvcf_workspace(tools, work: Path) -> None:
    write_vcf_text(work, GVCF, GVCF_BODY, tools)
    (work / "r.bed").write_text("chr1\t99\t150\n")  # 1-based 100-150: the block starts before it


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-R", "r.bed"],
        ["view", "-H", "--regions-file", "r.bed", "--regions-overlap", "1"],
        ["view", "-H", "-R", "r.bed", "--regions-overlap=record"],
        ["view", "-H", "-R", "r.bed", "--regions-overlap", "2"],
        ["view", "-H", "-r", "chr1:100-150"],
        ["view", "-H", "-R", "r.bed", "--regions-overlap", "0"],
        ["view", "-H", "-R", "r.bed", "--regions-overlap=pos"],
        ["view", "-H", "-T", "r.bed"],
        ["view", "-H", "--targets-file=r.bed"],
        ["view", "-H", "-T", "r.bed", "--targets-overlap", "0"],
        ["view", "-H", "-T", "r.bed", "--targets-overlap", "1"],
        ["view", "-H", "-T", "r.bed", "--targets-overlap=record"],
        ["view", "-H", "-T", "r.bed", "--targets-overlap", "2"],
        ["view", "-H", "-t", "chr1:100-150"],
        ["view", "-H", "-t", "chr1:100-150", "--targets-overlap", "1"],
        ["filter", "-i", "POS>0", "-R", "r.bed"],
        ["filter", "-i", "POS>0", "-T", "r.bed"],
        ["filter", "-i", "POS>0", "-T", "r.bed", "--targets-overlap", "1"],
        ["annotate", "-R", "r.bed"],
        ["annotate", "-R", "r.bed", "--regions-overlap", "0"],
        ["norm", "-m-any", "-R", "r.bed"],
        ["norm", "-m-any", "-T", "r.bed"],
        ["+fill-tags", "-R", "r.bed"],
        ["+fill-tags", "-T", "r.bed"],
        ["+fill-tags", "-T", "r.bed", "--targets-overlap", "1"],
        ["query", "-f", "%CHROM\\t%POS\\t%ID\\t%END\\n", "-R", "r.bed"],
        ["query", "-f", "%CHROM\\t%POS\\t%ID\\t%END\\n", "-T", "r.bed"],
        ["query", "-f", "%CHROM\\t%POS\\t%ID\\t%INFO/END\\n", "-T", "r.bed", "--targets-overlap=1"],
    ],
)
def test_the_options_select_what_the_semantics_table_says(tools, tmp_path, argv):
    gvcf_workspace(tools, tmp_path)
    got = ids_of(tools, tmp_path, [*argv, "variants.vcf.gz"])
    selections, invalid = rules.selections(argv)
    assert selections and not invalid
    kept_block = "blk" in got
    assert kept_block == all(s.semantic == "overlap" for s in selections), argv
    assert "snv" in got and "far" not in got


def test_query_on_a_text_vcf_misses_the_block_unless_the_format_reads_info(tools, tmp_path):
    gvcf_workspace(tools, tmp_path)
    bcf = ["view", "-Ob", "-o", "variants.bcf", "variants.vcf.gz"]
    assert tools.run(["bcftools", *bcf], tmp_path).returncode == 0
    assert tools.run(["bcftools", "index", "variants.bcf"], tmp_path).returncode == 0
    blind = {
        "%ID\\n": 0,
        "%CHROM\\t%POS\\t%ID\\n": 2,
        "%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\t%TYPE\\n": 0,
    }
    for fmt, col in blind.items():
        argv = ["query", "-f", fmt, "-R", "r.bed", "variants.vcf.gz"]
        assert "blk" not in ids_of(tools, tmp_path, argv, col) and rules._query_misses_end(argv), (
            fmt
        )
        # a BCF input keeps the block, and the checker does not claim otherwise
        bcf_argv = ["query", "-f", fmt, "-R", "r.bed", "variants.bcf"]
        assert not rules._query_misses_end(bcf_argv)
        assert "blk" in ids_of(tools, tmp_path, bcf_argv, col)
    for fmt in ("%ID\\t%END\\n", "%ID\\t%INFO/END\\n", "%ID\\t%INFO\\n"):
        argv = ["query", "-f", fmt, "-R", "r.bed", "variants.vcf.gz"]
        assert "blk" in ids_of(tools, tmp_path, argv, 0) and not rules._query_misses_end(argv), fmt
    start_only = ["query", "-f", "%ID\\n", "-T", "r.bed", "--targets-overlap", "1"]
    start_only.append("variants.vcf.gz")
    assert "blk" not in ids_of(tools, tmp_path, start_only, 0)
    assert rules._query_misses_end(start_only)


def test_merge_ignores_info_end_with_a_region_option(tools, tmp_path):
    gvcf_workspace(tools, tmp_path)
    argv = ["merge", "-R", "r.bed", "variants.vcf.gz", "variants.vcf.gz"]
    ids = ids_of(tools, tmp_path, argv)
    assert "snv" in ids and "blk" not in ids  # view -R keeps the block: see the table above


def test_call_targets_reads_only_chrom_and_pos_while_call_regions_follows_the_name(tools, tmp_path):
    seq = "ACGT" * 50
    write_fasta(tmp_path, {"chr1": seq}, tools)
    reads = "".join(
        f"r{i}\t0\tchr1\t81\t60\t50M\t*\t0\t0\t{seq[80:130]}\t{'I' * 50}\n" for i in range(3)
    )
    write_bam(tmp_path, "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:200\n" + reads, tools)
    pile = ["bcftools", "mpileup", "-f", "ref.fa", "-Oz", "-o", "pile.vcf.gz", "reads.bam"]
    assert tools.run(pile, tmp_path).returncode == 0
    assert tools.run(["bcftools", "index", "pile.vcf.gz"], tmp_path).returncode == 0
    for name in ("w.bed", "w.tsv"):
        (tmp_path / name).write_text("chr1\t100\t105\n")

    def positions(option, name):
        proc = tools.run(["bcftools", "call", "-m", option, name, "pile.vcf.gz"], tmp_path)
        return [ln.split("\t")[1] for ln in proc.stdout.splitlines() if ln and ln[0] != "#"]

    assert positions("-T", "w.bed") == positions("-T", "w.tsv") == ["100"]
    assert positions("-R", "w.bed") == ["101", "102", "103", "104", "105"]
    assert positions("-R", "w.tsv") == ["100", "101", "102", "103", "104", "105"]


# ---- admission over 300 seeds ------------------------------------------------------------------


def test_regions_file_base_is_admitted_with_zero_friction(tools, tmp_path):
    stats = measure(BASE, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared, right_undeclared = stats.right_declared, stats.cells[("right", False)]
    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    wrong_undeclared = stats.cells[("wrong", False)]
    assert wrong_undeclared.n > 0 and wrong_undeclared.failed == wrong_undeclared.n
    assert promotion_is_consistent(stats)


def test_region_semantics_is_admitted(tools, tmp_path):
    stats = measure(SEMANTICS, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert not stats.conditional and stats.admitted() and stats.killed() > 0
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert promotion_is_consistent(stats)


# ---- the two offered commands are valid when pasted (E-2, audit of the stage-2 review) ---------


def offered_commands(argv, declared=None):
    """The two commands the missing-declaration fix offers: (overlap, start_only) as call parts."""
    _, out = outcome(SEMANTICS, argv, Path("/tmp"), declared or {})
    assert out.result.code == "declaration_missing", out.result
    commands = []
    for line in out.result.fix[1:3]:
        text = re.match(r"(?:or )?(.+?) \((?:overlap|start_only)", line).group(1)
        commands.append(Call.from_argv("bcftools", shlex.split(text)[1:], Path("/tmp"), env={}))
    return commands


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("view", "-H", "-R", "r.bed", "v.vcf.gz"), id="plain"),
        pytest.param(
            ("view", "-T", "r.bed", "--targets-overlap", "0", "--targets-overlap", "1", "v.vcf.gz"),
            id="a repeated option: the last one counts",
        ),
        pytest.param(("plugin", "fill-tags", "-R", "r.bed", "v.vcf.gz"), id="the plugin form"),
        pytest.param(("+fill-tags", "-R", "r.bed", "v.vcf.gz"), id="the +name form"),
    ],
)
def test_each_offered_command_passes_its_own_check(argv):
    overlap, start_only = offered_commands(argv)

    for call in (overlap, start_only):
        assert semantic_verdict(list(call.argv), dict(call.declared)) == ("pass", "ok"), call.argv
    assert overlap.declared["region_semantics"] == "overlap"
    assert start_only.declared["region_semantics"] == "start_only"


def test_the_plugin_name_stays_before_the_option_that_was_inserted():
    overlap, start_only = offered_commands(("plugin", "fill-tags", "-R", "r.bed", "v.vcf.gz"))

    assert start_only.argv[:3] == ("plugin", "fill-tags", "--regions-overlap")
    assert overlap.argv == ("plugin", "fill-tags", "-R", "r.bed", "v.vcf.gz")


def test_a_repeated_overlap_option_is_changed_where_bcftools_reads_it():
    argv = ("view", "-T", "r.bed", "--targets-overlap", "0", "--targets-overlap", "1", "v.vcf.gz")
    overlap, start_only = offered_commands(argv)

    assert overlap.argv == argv  # the call as written already keeps overlapping records
    assert start_only.argv == (*argv[:5], "--targets-overlap", "0", "v.vcf.gz")


def test_declarations_the_call_already_made_survive_in_both_commands():
    declared = {"region_file_base": "one_based_closed"}
    overlap, start_only = offered_commands(("view", "-H", "-R", "regions.tsv", "v.vcf"), declared)

    for call in (overlap, start_only):
        assert call.declared["region_file_base"] == "one_based_closed"
