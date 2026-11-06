"""Rules diamond.hit_cap (F79) and diamond.sensitivity_mode (F80): the checkers on real argv and
files, the DIAMOND facts they rely on (run with the real program), and admission over calls
labelled by running diamond.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f79_diamond_cap as f79
from biowarden.bench.tasks.extra import f80_diamond_sens as f80
from biowarden.bench.tasks.extra.f77_blastn_task import fasta_text, identity_exact, rand_prot
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from biowarden.engine.tools import call_rule_tools
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_diamond")
CAP, MODE = SPEC.rules
AA = "ACDEFGHIKLMNPQRSTVWY"
QUIET = ["--threads", "1", "--quiet"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "diamond" not in env.real_bins(["diamond"]):
        pytest.skip("diamond is not available")
    return env


def only(rule):
    return SPEC.model_copy(update={"rules": [rule]})


def evaluate(rule, argv, cwd, tools=None, declared=None):
    spec = only(rule)
    real = tools.real_bins(call_rule_tools([spec])) if tools else {}
    env = tools.process_env() if tools else None
    eng = Engine([spec], mode="block", real_bins=real, real_env=env, retry_cap=10**9)
    decision = eng.evaluate(Call("diamond", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def make_db(tools, work):
    assert (
        tools.run(["diamond", "makedb", "--in", "db.fasta", "-d", "db", *QUIET], work).returncode
        == 0
    )


# ---- specs ------------------------------------------------------------------------------------


def test_specs_are_conditional_t1_call_rules_on_blastp_and_blastx():
    assert [r.id for r in SPEC.rules] == ["diamond.hit_cap", "diamond.sensitivity_mode"]
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier) == ("call", "T1")
        assert rule.applies.tools == ["diamond"] and rule.applies.subcommands == [
            "blastp",
            "blastx",
        ]
        assert rule.declare is not None and not rule.declare.required
        assert rule.check in known_checkers()
    assert CAP.declare.key == "hit_limit" and set(CAP.declare.values) == {"all_hits", "top_hits"}
    assert MODE.declare.key == "homology_range" and set(MODE.declare.values) == {"close", "remote"}


def test_the_checkers_rerun_the_tool_so_they_need_tier_t1(tmp_path):
    (tmp_path / "q.fa").write_text(">q\nMKT\n")
    (tmp_path / "db.dmnd").write_bytes(b"x")
    call = Call("diamond", ("blastp", "-q", "q.fa", "-d", "db"), tmp_path)
    for rule in (CAP, MODE):
        with pytest.raises(TierViolation):
            known_checkers()[rule.check](call, Match(rule), Ctx(tmp_path, "T0"))


# ---- diamond.hit_cap ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cap_sets(tools, tmp_path_factory):
    out = {}
    for twin in ("clean", "faulted"):
        work = tmp_path_factory.mktemp(f"f79{twin}")
        build_task("F79", 2, twin, work, tools, "implicit")
        make_db(tools, work)
        out[twin] = work
    return out


CALL = ["blastp", "-q", "query.fasta", "-d", "db", "-o", "out.tsv", "-e", "1e-5"]


@pytest.mark.parametrize(
    ("extra", "verdict"),
    [
        ([], "fail"),
        (["-k", "25"], "fail"),
        (["--max-target-seqs", "30"], "fail"),
        (["-k", "5"], "fail"),
        (["--sensitive"], "fail"),
        (["-k", "0"], "pass"),
        (["--max-target-seqs", "0"], "pass"),
        (["-k0"], "pass"),
        (["-k", "100"], "pass"),
        (["--max-target-seqs=500"], "pass"),
        (["--top", "10"], "pass"),
    ],
)
def test_the_cap_in_force_decides_on_a_family_larger_than_25(tools, cap_sets, extra, verdict):
    decision, result = evaluate(CAP, [*CALL, *extra], cap_sets["faulted"], tools)
    assert result.verdict == verdict, (extra, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_a_family_below_the_cap_stays_silent_without_a_declaration(tools, cap_sets):
    for extra in ([], ["-k", "25"], ["-k", "30"]):
        assert evaluate(CAP, [*CALL, *extra], cap_sets["clean"], tools)[1].verdict == "pass"
    assert evaluate(CAP, [*CALL, "-k", "5"], cap_sets["clean"], tools)[1].verdict == "fail"


def test_the_message_names_the_cap_the_counts_and_the_fix(tools, cap_sets):
    decision, result = evaluate(CAP, CALL, cap_sets["faulted"], tools)
    text = render_block(decision)
    assert text.startswith("FIX: add --max-target-seqs 0 (all hits)")
    assert (
        result.code == "hit_cap_binds" and "at most 25 target sequences per query" in result.message
    )
    assert "1 of 1 queries in query.fasta" in result.message


def test_declarations_of_the_hit_limit(tools, cap_sets):
    work = cap_sets["faulted"]
    assert evaluate(CAP, CALL, work, tools, {"hit_limit": "top_hits"})[1].verdict == "pass"
    assert evaluate(CAP, CALL, work, tools, {"hit_limit": "all_hits"})[1].verdict == "fail"
    top = [*CALL, "--top", "5"]
    assert evaluate(CAP, top, work, tools, {"hit_limit": "all_hits"})[1].code == "declared_mismatch"
    assert evaluate(CAP, top, work, tools, {"hit_limit": "top_hits"})[1].verdict == "pass"
    assert evaluate(CAP, CALL, work, tools, {"hit_limit": "x"})[0].action == "block"


def test_missing_inputs_other_subcommands_and_stdin_are_silent(tools, cap_sets):
    work = cap_sets["faulted"]
    for argv in (
        ["blastp", "-q", "absent.fasta", "-d", "db"],
        ["blastp", "-q", "query.fasta", "-d", "absent"],
        ["blastp", "-q", "-", "-d", "db"],
        ["blastp", "--help"],
    ):
        assert evaluate(CAP, argv, work, tools)[1].verdict == "pass", argv
    for argv in (["makedb", "--in", "db.fasta", "-d", "db"], ["dbinfo", "-d", "db"]):
        assert evaluate(CAP, argv, work, tools)[0].outcomes == ()  # not a search: no rule


def test_diamond_reports_25_subjects_by_default_without_warning_and_all_with_k_zero(
    tools, cap_sets
):
    work = cap_sets["faulted"]
    n_family = f79.make_plan(2, "faulted")["n_family"]
    base = ["diamond", "blastp", "-q", "query.fasta", "-d", "db", "-e", "1e-5", *QUIET]
    default = tools.run([*base, "-f", "6", "sseqid"], work)
    full = tools.run([*base, "-k", "0", "-f", "6", "sseqid"], work)
    assert default.returncode == 0 and "warn" not in default.stderr.lower()
    assert len(set(default.stdout.split())) == 25 and len(set(full.stdout.split())) == n_family > 25
    # --top replaces the cap
    top = tools.run([*base, "--top", "100", "-f", "6", "sseqid"], work)
    assert len(set(top.stdout.split())) > 25


# ---- diamond.sensitivity_mode ------------------------------------------------------------------


@pytest.fixture(scope="module")
def mode_sets(tools, tmp_path_factory):
    out = {}
    for twin in ("clean", "faulted"):
        work = tmp_path_factory.mktemp(f"f80{twin}")
        build_task("F80", 5, twin, work, tools, "implicit")
        make_db(tools, work)
        out[twin] = work
    return out


MODE_CALL = ["blastp", "-q", "queries.fasta", "-d", "db", "-o", "out.tsv"]


@pytest.mark.parametrize(
    ("extra", "verdict"),
    [
        ([], "fail"),
        (["--fast"], "fail"),
        (["--mid-sensitive"], "fail"),
        (["-k", "0", "-e", "0.001"], "fail"),
        (["--ultra-sensitive"], "pass"),
    ],
)
def test_the_mode_decides_on_remote_homologs(tools, mode_sets, extra, verdict):
    decision, result = evaluate(MODE, [*MODE_CALL, *extra], mode_sets["faulted"], tools)
    assert result.verdict == verdict, (extra, result)


@pytest.mark.parametrize("flag", ["--sensitive", "--more-sensitive", "--very-sensitive"])
def test_the_intermediate_modes_fail_exactly_when_ultra_sensitive_finds_more(
    tools, mode_sets, flag
):
    work = mode_sets["faulted"]
    base = ["diamond", "blastp", "-q", "queries.fasta", "-d", "db", "-f", "6", "qseqid", *QUIET]
    mine = set(tools.run([*base, flag], work).stdout.split())
    ultra = set(tools.run([*base, "--ultra-sensitive"], work).stdout.split())
    result = evaluate(MODE, [*MODE_CALL, flag], work, tools)[1]
    assert (result.verdict == "fail") == bool(ultra - mine), (flag, ultra - mine)


def test_close_homologs_stay_silent_in_the_default_mode(tools, mode_sets):
    for extra in ([], ["--fast"], ["--sensitive"]):
        assert evaluate(MODE, [*MODE_CALL, *extra], mode_sets["clean"], tools)[1].verdict == "pass"


def test_the_message_names_the_queries_and_the_fix(tools, mode_sets):
    decision, result = evaluate(MODE, MODE_CALL, mode_sets["faulted"], tools)
    text = render_block(decision)
    assert text.startswith("FIX: add --ultra-sensitive")
    assert result.code == "mode_misses_hits" and "in the default mode" in result.message


def test_declarations_of_the_homology_range(tools, mode_sets):
    work = mode_sets["faulted"]
    assert evaluate(MODE, MODE_CALL, work, tools, {"homology_range": "close"})[1].verdict == "pass"
    assert evaluate(MODE, MODE_CALL, work, tools, {"homology_range": "remote"})[1].verdict == "fail"
    done = [*MODE_CALL, "--ultra-sensitive"]
    assert evaluate(MODE, done, work, tools, {"homology_range": "remote"})[1].verdict == "pass"


def test_default_mode_misses_remote_partners_silently_and_ultra_sensitive_finds_them(
    tools, mode_sets
):
    work = mode_sets["faulted"]
    base = ["diamond", "blastp", "-q", "queries.fasta", "-d", "db", "-f", "6", "qseqid", *QUIET]
    fast = tools.run(base, work)
    ultra = tools.run([*base, "--ultra-sensitive"], work)
    assert fast.returncode == ultra.returncode == 0 and "warn" not in fast.stderr.lower()
    found = sorted(set(ultra.stdout.split()))
    task = build_task("F80", 5, "faulted", work.parent / "again", tools, "implicit")
    assert found == task.truth["expected"] and sorted(set(fast.stdout.split())) != found


def test_the_hit_cap_and_mode_facts_on_hand_made_sets(tools, tmp_path):
    rng = random.Random(8)
    root = rand_prot(rng, 190)
    recs = [(f"fam{i:02d}", identity_exact(rng, root, 0.9, AA)) for i in range(30)]
    recs += [(f"oth{i}", rand_prot(rng, 190)) for i in range(15)]
    (tmp_path / "db.fasta").write_text(fasta_text(recs))
    (tmp_path / "query.fasta").write_text(fasta_text([("q", root)]))
    make_db(tools, tmp_path)
    assert evaluate(CAP, CALL, tmp_path, tools)[1].verdict == "fail"
    assert evaluate(CAP, [*CALL, "-k", "0"], tmp_path, tools)[1].verdict == "pass"
    assert f79.CAP == 25 and f80.TWIN_FILES


# ---- admission -------------------------------------------------------------------------------


@pytest.mark.parametrize(("rule", "seeds"), [(CAP, 24), (MODE, 8)])
def test_admission_on_calls_labelled_by_running_diamond(tools, tmp_path, rule, seeds):
    stats = measure(rule.id, [only(rule)], range(seeds), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", rule.id, "seeds", seeds, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * seeds // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.cells[("right", False)].n > 0 and stats.killed() > 0
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
