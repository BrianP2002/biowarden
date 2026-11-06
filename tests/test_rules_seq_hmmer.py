"""Rule hmmer.split_database (F88): the checker on real argv and files, the HMMER facts it relies
on (run with the real programs), and admission over calls labelled by running hmmsearch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f88_hmmer_split as f88
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules.seq_hmmer import siblings
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_hmmer")
(RULE,) = SPEC.rules
KEY = "evalue_database"
NEEDS = ["hmmsearch", "hmmbuild", "hmmscan"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not set(env.real_bins(NEEDS)) >= set(NEEDS):
        pytest.skip("HMMER is not available")
    return env


@pytest.fixture(scope="module")
def sets(tools, tmp_path_factory):
    out = {}
    for twin in ("clean", "faulted"):
        work = tmp_path_factory.mktemp(f"f88{twin}")
        task = build_task("F88", 4, twin, work, tools, "implicit")
        out[twin] = (task, work)
    work = out["faulted"][1]
    names = sorted(p.name for p in work.glob("proteome_part*.fasta"))
    (work / "proteome_all.fasta").write_text("".join((work / n).read_text() for n in names))
    return out


def evaluate(tool, argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def search(*options, target="proteome_part1.fasta"):
    return ["--tblout", "t.tbl", "-o", "/dev/null", *options, "fam.hmm", target]


def test_spec_is_a_conditional_t1_call_rule_on_the_hmmer_programs():
    assert RULE.id == "hmmer.split_database" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["hmmsearch", "phmmer", "nhmmer", "hmmscan"]
    assert RULE.declare.key == KEY and not RULE.declare.required
    assert set(RULE.declare.values) == {"whole_collection", "this_file"}
    assert RULE.check in known_checkers()


def test_the_checker_counts_the_files_so_it_needs_tier_t1(sets):
    _, work = sets["faulted"]
    call = Call("hmmsearch", tuple(search("-E", "1e-5")), work)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(work, "T0"))


@pytest.mark.parametrize(
    ("options", "verdict"),
    [
        ([], "fail"),
        (["-E", "1e-5"], "fail"),
        (["--incE", "1e-5"], "fail"),
        (["--domE", "1e-5"], "fail"),
        (["-E", "1e-5", "--cpu", "1"], "fail"),
        (["-Z", "200"], "pass"),
        (["-E", "1e-5", "-Z", "200"], "pass"),
        (["-E", "1e-5", "-Z200"], "pass"),
        (["-T", "30"], "pass"),  # a bit-score threshold does not depend on the database size
        (["--incT", "30", "--incdomT", "30"], "pass"),
        (["--cut_ga"], "pass"),
        (["-T", "30", "-E", "1e-5"], "fail"),
    ],
)
def test_naming_z_or_a_score_threshold_decides_on_a_database_in_parts(sets, options, verdict):
    _, work = sets["faulted"]
    decision, result = evaluate("hmmsearch", search(*options), work)
    assert result.verdict == verdict, (options, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_a_single_file_the_merged_file_and_other_inputs_stay_silent(sets):
    _, clean = sets["clean"]
    for options in ([], ["-E", "1e-5"]):
        argv = search(*options, target="proteome.fasta")
        assert evaluate("hmmsearch", argv, clean)[1].verdict == "pass"
    _, work = sets["faulted"]
    merged = search("-E", "1e-5", target="proteome_all.fasta")
    assert evaluate("hmmsearch", merged, work)[1].verdict == "pass"
    for argv in (
        search("-E", "1e-5", target="absent_part1.fasta"),
        ["fam.hmm"],
        ["-h"],
        search("-E", "1e-5", target="-"),
    ):
        assert evaluate("hmmsearch", argv, work)[1].verdict == "pass", argv


def test_other_programs_and_profile_files_are_covered(sets, tmp_path):
    _, work = sets["faulted"]
    assert (
        evaluate("phmmer", ["-E", "1e-5", "fam.hmm", "proteome_part2.fasta"], work)[1].verdict
        == "fail"
    )
    assert (
        evaluate("nhmmer", ["-E", "1e-5", "fam.hmm", "proteome_part2.fasta"], work)[1].verdict
        == "fail"
    )
    model = (work / "fam.hmm").read_text()
    for i in (1, 2, 3):
        (tmp_path / f"fam_{i}.hmm").write_text(model)
    (tmp_path / "queries.fasta").write_text(">q\nMKT\n")
    call = ["--tblout", "t.tbl", "-E", "1e-5", "fam_1.hmm", "queries.fasta"]
    decision, result = evaluate("hmmscan", call, tmp_path)
    assert result.verdict == "fail" and "3 profiles together" in result.message
    assert evaluate("hmmscan", ["-Z", "3", *call], tmp_path)[1].verdict == "pass"
    assert siblings(tmp_path / "fam_1.hmm") == [tmp_path / f"fam_{i}.hmm" for i in (1, 2, 3)]
    assert siblings(tmp_path / "queries.fasta") == [tmp_path / "queries.fasta"]


def test_the_message_has_the_fix_first_and_the_instance_facts(sets):
    task, work = sets["faulted"]
    decision, result = evaluate("hmmsearch", search("-E", "1e-5"), work)
    total = sum((work / f"proteome_part{i}.fasta").read_text().count(">") for i in range(1, 9))
    per_part = total // 8
    text = render_block(decision)
    assert text.startswith(f"FIX: add -Z {total} (the number of sequences in all 8 files)")
    assert result.code == "evalue_from_one_part" and (result.expected, result.got) == (
        total,
        per_part,
    )
    assert f"from the {per_part} sequences of proteome_part1.fasta" in result.message
    assert "8.0 times smaller than in a search of the whole collection" in result.message


def test_declarations_of_the_evalue_database(sets):
    _, work = sets["faulted"]
    call = search("-E", "1e-5")
    assert evaluate("hmmsearch", call, work, {KEY: "this_file"})[1].verdict == "pass"
    assert evaluate("hmmsearch", call, work, {KEY: "whole_collection"})[1].verdict == "fail"
    done = search("-E", "1e-5", "-Z", "200")
    assert evaluate("hmmsearch", done, work, {KEY: "whole_collection"})[1].verdict == "pass"
    assert evaluate("hmmsearch", call, work, {KEY: "x"})[0].action == "block"


# ---- HMMER facts -----------------------------------------------------------------------------


def hits(tools, work, target, *options):
    argv = ["hmmsearch", "--cpu", "1", "--tblout", "x.tbl", "-o", "/dev/null", *options]
    proc = tools.run([*argv, "fam.hmm", target], work)
    assert proc.returncode == 0 and proc.stderr == ""
    rows = [r.split() for r in (work / "x.tbl").read_text().splitlines() if not r.startswith("#")]
    return {r[0]: r[4] for r in rows}


def test_evalues_scale_with_the_number_of_sequences_in_the_file_and_z_restores_them(tools, sets):
    task, work = sets["faulted"]
    plan = f88.make_proteome(tools, 4)
    whole = hits(tools, work, "proteome_all.fasta", "-E", "10")
    part = hits(tools, work, "proteome_part1.fasta", "-E", "10")
    assert part and set(part) <= set(whole)
    for name, text in part.items():
        ratio = float(whole[name]) / float(text)
        assert abs(ratio - 8) / 8 < 0.15, (name, whole[name], text)  # E has 2 digits in the table
    total = len(plan["records"])
    same = hits(tools, work, "proteome_part1.fasta", "-E", "10", "-Z", str(total))
    assert same == {k: whole[k] for k in same}
    # the hits that pass on the parts but not on the whole file are the planted borderline ones
    naive = {k for k, e in plan["parts"].items() if e <= f88.THRESHOLD}
    exact = {k for k, e in plan["whole"].items() if e <= f88.THRESHOLD}
    assert sorted(naive) == task.truth["naive"] and sorted(exact) == task.truth["expected"]
    assert len(naive - exact) >= 3


def test_cut_ga_on_a_model_without_ga_is_loud(tools, sets):
    _, work = sets["faulted"]
    proc = tools.run(["hmmsearch", "--cut_ga", "fam.hmm", "proteome_all.fasta"], work)
    assert proc.returncode != 0 and "GA bit thresholds unavailable" in proc.stderr + proc.stdout


def test_family_workspaces_fail_on_the_faulted_twin_only(sets):
    _, clean = sets["clean"]
    _, faulted = sets["faulted"]
    assert (
        evaluate("hmmsearch", search("-E", "1e-5", target="proteome.fasta"), clean)[1].verdict
        == "pass"
    )
    assert evaluate("hmmsearch", search("-E", "1e-5"), faulted)[1].verdict == "fail"
    # all eight parts are covered, not only the first
    for k in range(1, 9):
        target = f"proteome_part{k}.fasta"
        assert (
            evaluate("hmmsearch", search("-E", "1e-5", target=target), faulted)[1].verdict == "fail"
        )


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_hmmsearch(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print(
        "ADMISSION",
        RULE.id,
        "seeds",
        ADMISSION_SEEDS,
        cells,
        "recall",
        round(stats.recall() or 0, 3),
    )
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= MIN_RIGHT_CALLS
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
