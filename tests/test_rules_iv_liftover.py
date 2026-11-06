"""Rules liftover.stale_columns, liftover.unmapped_discarded, liftover.chain_contigs (pack iv).

Unit tests on a small hand-built chain and BED files written here, the real liftOver on the traps
the rules name, the F74 workspaces, and the admission measurement (calls labelled by running the
real tool, bench/admission/cases_iv_liftover.py).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
STALE_ID = "liftover.stale_columns"
UNMAPPED_ID = "liftover.unmapped_discarded"
CONTIGS_ID = "liftover.chain_contigs"

# chr1 2000 -> 2050: a 100-base insertion in the new assembly at old 500, old 1300-1350 deleted
CHAIN = "chain 1000 chr1 2000 + 0 2000 chr1 2050 + 0 2050 1\n500\t0\t100\n800\t50\t0\n650\n"
PEAK_STALE = "chr1\t450\t550\tpk_stale\t500\t.\t5.0\t10.0\t8.0\t80"  # summit 530 -> 630
PEAK_OK = "chr1\t100\t200\tpk_ok\t500\t.\t5.0\t10.0\t8.0\t50"
PEAK_LATE = "chr1\t1000\t1100\tpk_late\t500\t.\t5.0\t10.0\t8.0\t50"  # all after the indels
PEAK_DROPPED = "chr1\t1200\t1400\tpk_drop\t500\t.\t5.0\t10.0\t8.0\t100"  # 50 of 200 deleted


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "liftOver" not in env.real_bins(["liftOver"]):
        pytest.skip("liftOver is not installed")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("iv_liftover")


def engine_for(spec: Spec, tools: ToolEnv | None = None) -> Engine:
    if tools is None:
        return Engine([spec], mode="block", retry_cap=10**9)
    return Engine(
        [spec],
        mode="block",
        real_bins=tools.real_bins(["liftOver"]),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )


def outcome(engine: Engine, rule_id: str, argv, cwd: Path, declared=None):
    decision = engine.evaluate(Call("liftOver", tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def workspace(tmp_path: Path, peaks: list[str], chain: str = CHAIN) -> Path:
    (tmp_path / "old2new.chain").write_text(chain)
    (tmp_path / "peaks.narrowPeak").write_text("".join(p + "\n" for p in peaks))
    return tmp_path


def lift(*extra: str, source="peaks.narrowPeak", unmapped="peaks.unmapped.bed"):
    return [*extra, source, "old2new.chain", "peaks.lifted.bed", unmapped]


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_has_the_three_call_rules(spec):
    assert spec.name == "iv_liftover"
    rules = {r.id: r for r in spec.rules}
    assert set(rules) == {STALE_ID, UNMAPPED_ID, CONTIGS_ID}
    engine_for(spec)  # imports the rule modules
    for rule in rules.values():
        assert (rule.kind, rule.tier, rule.applies.tools) == ("call", "T1", ["liftOver"])
        assert rule.check in known_checkers()
    assert rules[STALE_ID].declare.key == "summit_offset"
    assert rules[STALE_ID].declare.values == ["not_used", "relifted"]
    assert rules[UNMAPPED_ID].declare.key == "dropped_records"
    assert rules[CONTIGS_ID].declare is None
    assert all(r.declare is None or r.declare.required is False for r in rules.values())
    assert spec.declaration_schema()["required"] == []


def test_spec_rejects_a_fact_rule_without_file_patterns_and_a_t0_without_declaration(spec):
    data = spec.model_dump(mode="json")
    data["rules"][0]["kind"] = "fact"
    with pytest.raises(ValidationError, match="file_patterns"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][2]["tier"] = "T0"  # no declaration: a T0 rule is decided from the call alone
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)


def test_the_checkers_read_files_so_a_t0_context_refuses(spec, tmp_path):
    workspace(tmp_path, [PEAK_STALE])
    call = Call("liftOver", tuple(lift("-bedPlus=6", unmapped="/dev/null")), tmp_path)
    for rule in spec.rules:
        with pytest.raises(TierViolation):
            known_checkers()[rule.check](call, Match(rule), Ctx(tmp_path, "T0"))


# ---- liftover.stale_columns --------------------------------------------------------------------


def test_a_peak_whose_summit_lies_after_an_insertion_fails_with_the_summit_fix(
    spec, tools, tmp_path
):
    work = workspace(tmp_path, [PEAK_STALE, PEAK_OK])
    eng = engine_for(spec, tools)
    decision, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work)
    assert result.verdict == "fail" and result.code == "summit_offset_stale"
    assert decision.action == "block"
    text = render_block(decision)
    assert text.splitlines()[0].startswith("FIX: lift the summit bases as intervals of their own")
    for needle in ("pk_stale", "1 of 2", "530", "630", "$2+$10", "summit_offset=not_used"):
        assert needle in text, needle


def test_peaks_that_keep_their_offset_pass_and_so_do_declarations(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_OK, PEAK_LATE])
    decision, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work)
    assert decision.action == "allow" and result.verdict == "pass"  # one shift for the whole peak
    work = workspace(tmp_path, [PEAK_STALE])
    for value in ("not_used", "relifted"):
        _, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work, {"summit_offset": value})
        assert result.verdict == "pass", value
    _, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work, {"summit_offset": "later"})
    assert result.code == "declaration_invalid"


def test_a_small_deletion_before_the_summit_shifts_it_and_a_dropped_peak_is_not_judged(
    spec, tools, tmp_path
):
    eng = engine_for(spec, tools)
    # old 1290-1400: 50 of 110 bases deleted, dropped by default; a deletion of 3 bases is lifted
    chain = "chain 1000 chr1 2000 + 0 2000 chr1 1997 + 0 1997\n500\t3\t0\n1497\n"
    work = workspace(tmp_path, ["chr1\t480\t620\tp\t5\t.\t1\t1\t1\t100"], chain)
    _, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work)
    assert result.verdict == "fail" and "p:" in result.message
    (tmp_path / "d").mkdir()
    work = workspace(tmp_path / "d", [PEAK_DROPPED])
    _, result = outcome(eng, STALE_ID, lift("-bedPlus=6"), work)
    assert result.verdict == "pass"  # nothing lifted: the other rule talks about that


def test_inputs_the_rule_does_not_look_at_pass(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_STALE])
    bed4 = "chr1\t450\t550\tb4\n"
    (work / "bed4.bed").write_text(bed4)
    assert outcome(eng, STALE_ID, lift(source="bed4.bed"), work)[1].verdict == "pass"
    # ten columns without -bedPlus: liftOver stops by itself (loud), the rule stays out of it
    assert outcome(eng, STALE_ID, lift(), work)[1].verdict == "pass"
    # other formats and -multiple change the columns
    assert outcome(eng, STALE_ID, lift("-bedPlus=6", "-multiple"), work)[1].verdict == "pass"
    assert outcome(eng, STALE_ID, lift("-gff"), work)[1].verdict == "pass"
    # missing files and too few arguments
    assert (
        outcome(eng, STALE_ID, lift("-bedPlus=6", source="absent.bed"), work)[1].verdict == "pass"
    )
    assert outcome(eng, STALE_ID, ["-bedPlus=6", "peaks.narrowPeak"], work)[1].verdict == "pass"
    (work / "words.txt").write_text("a\tb\tc\td\te\tf\tg\th\ti\tj\n")
    assert outcome(eng, STALE_ID, lift("-bedPlus=6", source="words.txt"), work)[1].verdict == "pass"


def test_without_a_real_liftover_the_rule_cannot_dry_run_and_passes(spec, tmp_path):
    work = workspace(tmp_path, [PEAK_STALE])
    decision, result = outcome(engine_for(spec), STALE_ID, lift("-bedPlus=6"), work)
    assert result.verdict == "pass" and decision.action == "allow"


# ---- liftover.unmapped_discarded ---------------------------------------------------------------


def test_dropped_intervals_sent_to_dev_null_fail_and_a_real_file_does_not(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_OK, PEAK_DROPPED])
    decision, result = outcome(eng, UNMAPPED_ID, lift("-bedPlus=6", unmapped="/dev/null"), work)
    assert result.verdict == "fail" and result.code == "dropped_records_discarded"
    text = render_block(decision)
    assert text.splitlines()[0].startswith("FIX: write the fourth argument to a file")
    for needle in (
        "1 interval(s)",
        "pk_drop",
        "Partially deleted in new",
        "dropped_records=accepted",
    ):
        assert needle in text, needle
    assert outcome(eng, UNMAPPED_ID, lift("-bedPlus=6"), work)[1].verdict == "pass"
    declared = {"dropped_records": "accepted"}
    args = lift("-bedPlus=6", unmapped="/dev/null")
    assert outcome(eng, UNMAPPED_ID, args, work, declared)[1].verdict == "pass"


def test_nothing_dropped_or_a_lower_min_match_that_keeps_everything_passes(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_OK, PEAK_STALE])
    assert (
        outcome(eng, UNMAPPED_ID, lift("-bedPlus=6", unmapped="/dev/null"), work)[1].verdict
        == "pass"
    )
    work = workspace(tmp_path, [PEAK_OK, PEAK_DROPPED])
    args = lift("-bedPlus=6", "-minMatch=0.5", unmapped="/dev/null")
    assert outcome(eng, UNMAPPED_ID, args, work)[1].verdict == "pass"  # lifted shrunk, not dropped
    args = lift("-bedPlus=6", "-minMatch=0.99", unmapped="/dev/null")
    assert outcome(eng, UNMAPPED_ID, args, work)[1].verdict == "fail"


def test_a_ten_column_file_without_bedplus_stops_liftover_and_the_rule_stays_out(
    spec, tools, tmp_path
):
    work = workspace(tmp_path, [PEAK_OK, PEAK_DROPPED])
    proc = tools.run(["liftOver", *lift(unmapped="/dev/null")], work)
    assert proc.returncode != 0 and "Expecting integer field 7" in proc.stderr
    eng = engine_for(spec, tools)
    assert outcome(eng, UNMAPPED_ID, lift(unmapped="/dev/null"), work)[1].verdict == "pass"


# ---- liftover.chain_contigs --------------------------------------------------------------------


def test_contigs_that_are_not_in_the_chain_fail_with_a_renaming_fix(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_OK])
    (work / "nochr.bed").write_text("1\t100\t200\ta\n1\t5\t9\tb\n")
    decision, result = outcome(eng, CONTIGS_ID, lift(source="nochr.bed"), work)
    assert result.verdict == "fail" and result.code == "no_contig_in_chain"
    text = render_block(decision)
    assert text.splitlines()[0].startswith("FIX: awk 'BEGIN{OFS=")
    assert (
        '$1="chr"$1' in text and "nochr.bed" in text and "chr1" in text and "unmapped file" in text
    )
    # the other way round: the input has the prefix, the chain has not
    (work / "plain.chain").write_text(CHAIN.replace("chr1", "1"))
    (work / "chr.bed").write_text("chr1\t100\t200\ta\n")
    args = ["chr.bed", "plain.chain", "o.bed", "u.bed"]
    result = outcome(eng, CONTIGS_ID, args, work)[1]
    assert result.verdict == "fail" and result.fix[0].startswith("sed 's/^chr//'")
    (work / "odd.bed").write_text("scaffold9\t1\t2\ta\n")
    assert (
        outcome(eng, CONTIGS_ID, lift(source="odd.bed"), work)[1]
        .fix[0]
        .startswith("rename the contigs")
    )


def test_matching_partly_matching_and_unreadable_chains_pass(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    work = workspace(tmp_path, [PEAK_OK])
    assert outcome(eng, CONTIGS_ID, lift("-bedPlus=6"), work)[1].verdict == "pass"
    (work / "mixed.bed").write_text("chr1\t100\t200\ta\nchrUn_gl000220\t1\t5\tb\n")
    assert outcome(eng, CONTIGS_ID, lift(source="mixed.bed"), work)[1].verdict == "pass"
    with gzip.open(work / "g.chain.gz", "wt") as handle:
        handle.write(CHAIN)
    (work / "nochr.bed").write_text("1\t100\t200\ta\n")
    args = ["nochr.bed", "g.chain.gz", "o.bed", "u.bed"]
    assert outcome(eng, CONTIGS_ID, args, work)[1].verdict == "fail"  # a gzip chain is read
    big = "chain 1 chr7 99 + 0 99 chr7 99 + 0 99 9\n99\n" + "\n".join(["99\t0\t0"] * 400_000) + "\n"
    (work / "big.chain").write_text(big)
    args = ["nochr.bed", "big.chain", "o.bed", "u.bed"]
    assert (
        outcome(eng, CONTIGS_ID, args, work)[1].verdict == "pass"
    )  # too large to read: no verdict
    assert (
        outcome(eng, CONTIGS_ID, ["nochr.bed", "absent.chain", "o", "u"], work)[1].verdict == "pass"
    )
    assert outcome(eng, CONTIGS_ID, lift("-gff", source="nochr.bed"), work)[1].verdict == "pass"


# ---- the traps on the real liftOver ------------------------------------------------------------


def test_liftover_keeps_the_offset_drops_partial_deletions_and_is_silent_about_both(
    tools, tmp_path
):
    work = workspace(tmp_path, [PEAK_STALE, PEAK_DROPPED])
    proc = tools.run(["liftOver", "-bedPlus=6", *lift()], work)
    assert proc.returncode == 0
    assert proc.stderr.splitlines() == ["Reading liftover chains", "Mapping coordinates"]
    lifted = (work / "peaks.lifted.bed").read_text().splitlines()
    assert lifted == ["chr1\t450\t650\tpk_stale\t500\t.\t5.0\t10.0\t8.0\t80"]  # grown, offset kept
    unmapped = (work / "peaks.unmapped.bed").read_text().splitlines()
    assert unmapped[0] == "#Partially deleted in new" and unmapped[1].split("\t")[3] == "pk_drop"
    (work / "summit.bed").write_text("chr1\t530\t531\ts\n")
    tools.run(["liftOver", "summit.bed", "old2new.chain", "s.out", "s.unm"], work)
    assert (work / "s.out").read_text() == "chr1\t630\t631\ts\n"  # the summit base itself
    tools.run(["liftOver", "-bedPlus=6", "-minMatch=0.5", *lift()], work)
    assert "pk_drop" in (work / "peaks.lifted.bed").read_text()  # lifted as 1300-1450, shrunk


def test_a_contig_mismatch_unmaps_everything_silently(tools, tmp_path):
    work = workspace(tmp_path, [PEAK_OK])
    (work / "nochr.bed").write_text("1\t100\t200\ta\n")
    proc = tools.run(["liftOver", "nochr.bed", "old2new.chain", "o.bed", "u.bed"], work)
    assert proc.returncode == 0 and proc.stderr.splitlines() == [
        "Reading liftover chains",
        "Mapping coordinates",
    ]
    assert (work / "o.bed").read_text() == ""
    assert (work / "u.bed").read_text().splitlines() == ["#Deleted in new", "1\t100\t200\ta"]


# ---- the F74 workspaces ------------------------------------------------------------------------


def test_the_family_commands_through_the_rules(spec, tools, tmp_path):
    eng = engine_for(spec, tools)
    faulted, clean = tmp_path / "f", tmp_path / "c"
    build_task("F74", 4, "faulted", faulted, tools, "implicit")
    build_task("F74", 4, "clean", clean, tools, "implicit")
    naive = ["-bedPlus=6", "peaks_old.narrowPeak", "old_to_new.chain", "lifted.bed", "unmapped.bed"]
    decision, result = outcome(eng, STALE_ID, naive, faulted)
    assert result.verdict == "fail" and decision.action == "block"
    assert outcome(eng, STALE_ID, naive, clean)[1].verdict == "pass"
    devnull = [*naive[:-1], "/dev/null"]
    assert outcome(eng, UNMAPPED_ID, devnull, faulted)[1].verdict == "fail"
    assert outcome(eng, UNMAPPED_ID, devnull, clean)[1].verdict == "pass"
    assert outcome(eng, CONTIGS_ID, naive, faulted)[1].verdict == "pass"
    rows = [
        line.split("\t") for line in (faulted / "peaks_old.narrowPeak").read_text().splitlines()
    ]
    summits = "".join(
        f"{r[0]}\t{int(r[1]) + int(r[9])}\t{int(r[1]) + int(r[9]) + 1}\t{r[3]}\n" for r in rows
    )
    (faulted / "summits.bed").write_text(summits)
    fixed = ["summits.bed", "old_to_new.chain", "summits.lifted.bed", "/dev/null"]
    for rule_id in (STALE_ID, UNMAPPED_ID, CONTIGS_ID):
        assert outcome(eng, rule_id, fixed, faulted)[1].verdict == "pass", rule_id


# ---- admission ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_id", "min_wrong_failed"), [(STALE_ID, 4), (UNMAPPED_ID, 4), (CONTIGS_ID, 5)]
)
def test_admission(spec, tools, tmp_path, rule_id, min_wrong_failed):
    alone = spec.model_copy(update={"rules": [r for r in spec.rules if r.id == rule_id]})
    stats = measure(rule_id, [alone], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    right_undeclared = stats.cells[("right", False)]
    assert stats.conditional
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert stats.killed() >= ADMISSION_SEEDS * min_wrong_failed
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert promotion_is_consistent(stats)
