"""Rules blastn.megablast_default (F77) and blast.stale_database (F78): the checkers on real argv
and files, the BLAST+ facts they rely on (run with the real programs), and admission over calls
labelled by running blastn / blastp.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f77_blastn_task as f77
from biowarden.bench.tasks.extra.f77_blastn_task import fasta_text, rand_dna, rand_prot
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from biowarden.engine.tools import call_rule_tools
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_blast")
MEGA, STALE = SPEC.rules
SEEDS = 10
NEEDS = ["blastn", "blastp", "makeblastdb"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not set(env.real_bins(NEEDS)) >= set(NEEDS):
        pytest.skip("BLAST+ is not available")
    return env


def only(rule):
    return SPEC.model_copy(update={"rules": [rule]})


def evaluate(rule, tool, argv, cwd, tools=None, declared=None):
    spec = only(rule)
    real = tools.real_bins(call_rule_tools([spec])) if tools else {}
    env = tools.process_env() if tools else None
    eng = Engine([spec], mode="block", real_bins=real, real_env=env, retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


# ---- specs ------------------------------------------------------------------------------------


def test_specs_are_conditional_t1_call_rules_with_registered_checkers():
    assert [r.id for r in SPEC.rules] == ["blastn.megablast_default", "blast.stale_database"]
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier) == ("call", "T1")
        assert rule.declare is not None and not rule.declare.required
        assert rule.check in known_checkers()
    assert MEGA.applies.tools == ["blastn"] and MEGA.declare.key == "blast_sensitivity"
    assert set(MEGA.declare.values) == {"near_identical", "divergent"}
    assert "blastp" in STALE.applies.tools and STALE.declare.key == "database_currency"
    assert set(STALE.declare.values) == {"current", "frozen"}


def test_both_checkers_read_files_so_they_need_tier_t1(tmp_path):
    (tmp_path / "q.fa").write_text(">q\nACGT\n")
    (tmp_path / "r.fa").write_text(">r\nACGT\n")
    call = Call("blastn", ("-query", "q.fa", "-subject", "r.fa"), tmp_path)
    assert call.cwd == tmp_path
    stale = Call("blastp", ("-query", "q.fa", "-db", "db"), tmp_path)
    meta = {"number-of-sequences": 3, "number-of-letters": 12, "description": "r.fa"}
    (tmp_path / "db.pjs").write_text(json.dumps(meta))
    with pytest.raises(TierViolation):
        known_checkers()[STALE.check](stale, Match(STALE), Ctx(tmp_path, "T0"))
    # the megablast checker only reaches the dry run through ctx.run_real (T1) or ctx.lines
    with pytest.raises(TierViolation):
        known_checkers()[MEGA.check](call, Match(MEGA), Ctx(tmp_path, "T0"))


# ---- blastn.megablast_default: argv and declarations -------------------------------------------


@pytest.fixture(scope="module")
def faulted(tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("f77f")
    task = build_task("F77", 3, "faulted", work, tools, "implicit")
    tools.run(["makeblastdb", "-in", "ref.fasta", "-dbtype", "nucl", "-out", "refdb"], work)
    return task, work


@pytest.fixture(scope="module")
def clean(tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("f77c")
    task = build_task("F77", 3, "clean", work, tools, "implicit")
    return task, work


BASE = ["-query", "queries.fasta", "-subject", "ref.fasta", "-evalue", "1e-5", "-outfmt", "6"]


@pytest.mark.parametrize(
    ("extra", "verdict"),
    [
        ([], "fail"),
        (["-task", "megablast"], "fail"),
        (["-word_size", "28"], "fail"),
        (["-word_size", "20"], "fail"),
        (["-perc_identity", "80"], "fail"),
        (["-max_target_seqs", "5"], "fail"),
        (["-task", "blastn"], "pass"),
        (["-task", "blastn-short"], "pass"),
        (["-task", "dc-megablast"], "pass"),
        (["-word_size", "11"], "pass"),
        (["-evalue", "1e-300"], "pass"),  # no hit survives the E-value limit of the rerun
    ],
)
def test_the_task_decides_on_diverged_queries(tools, faulted, extra, verdict):
    _, work = faulted
    decision, result = evaluate(MEGA, "blastn", [*BASE, *extra], work, tools)
    assert result.verdict == verdict, (extra, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_the_block_message_has_the_fix_first_and_the_counts(tools, faulted):
    _, work = faulted
    decision, result = evaluate(MEGA, "blastn", BASE, work, tools)
    text = render_block(decision)
    assert text.startswith("FIX: add -task blastn (word size 11")
    assert result.code == "megablast_misses_hits" and "have no hit with the default task" in text
    assert "of 12 queries in queries.fasta" in result.message


def test_a_database_given_by_name_is_searched_too(tools, faulted):
    _, work = faulted
    argv = ["-query", "queries.fasta", "-db", "refdb", "-evalue", "1e-5"]
    assert evaluate(MEGA, "blastn", argv, work, tools)[1].verdict == "fail"
    assert evaluate(MEGA, "blastn", [*argv, "-task", "blastn"], work, tools)[1].verdict == "pass"


def test_near_identical_queries_stay_silent_without_any_declaration(tools, clean):
    _, work = clean
    for extra in ([], ["-task", "megablast"], ["-task", "blastn"], ["-word_size", "20"]):
        assert evaluate(MEGA, "blastn", [*BASE, *extra], work, tools)[1].verdict == "pass", extra


def test_declarations_of_the_blast_sensitivity(tools, faulted):
    _, work = faulted
    near = {"blast_sensitivity": "near_identical"}
    far = {"blast_sensitivity": "divergent"}
    assert evaluate(MEGA, "blastn", BASE, work, tools, near)[1].verdict == "pass"
    decision, result = evaluate(MEGA, "blastn", BASE, work, tools, far)
    assert result.code == "declared_mismatch" and decision.action == "block"
    assert "you declared blast_sensitivity=divergent" in render_block(decision)
    assert (
        evaluate(MEGA, "blastn", [*BASE, "-task", "blastn"], work, tools, far)[1].verdict == "pass"
    )
    # a megablast call with a small word size is not the default: the rerun decides
    small = [*BASE, "-word_size", "11"]
    assert evaluate(MEGA, "blastn", small, work, tools, far)[1].verdict == "pass"
    assert (
        evaluate(MEGA, "blastn", BASE, work, tools, {"blast_sensitivity": "x"})[0].action == "block"
    )


def test_missing_inputs_standard_input_and_search_strategies_are_silent(tools, faulted):
    _, work = faulted
    for argv in (
        ["-query", "absent.fasta", "-subject", "ref.fasta"],
        ["-query", "queries.fasta", "-subject", "absent.fasta"],
        ["-query", "-", "-subject", "ref.fasta"],
        ["-query", "queries.fasta"],
        ["-import_search_strategy", "s.txt", "-outfmt", "6"],
        ["-version"],
    ):
        assert evaluate(MEGA, "blastn", argv, work, tools)[1].verdict == "pass", argv


def test_short_exact_queries_are_lost_by_megablast_and_found_by_the_short_task(tools, tmp_path):
    rng = random.Random(4)
    ref = rand_dna(rng, 3000)
    (tmp_path / "ref.fasta").write_text(fasta_text([("chr1", ref)]))
    oligos = [(f"oligo{i}", ref[200 * i + 7 : 200 * i + 7 + 22]) for i in range(1, 8)]
    (tmp_path / "queries.fasta").write_text(fasta_text(oligos))
    argv = ["-query", "queries.fasta", "-subject", "ref.fasta", "-evalue", "1e-3"]
    assert evaluate(MEGA, "blastn", argv, tmp_path, tools)[1].verdict == "fail"
    assert (
        evaluate(MEGA, "blastn", [*argv, "-task", "blastn-short"], tmp_path, tools)[1].verdict
        == "pass"
    )
    assert (
        evaluate(MEGA, "blastn", [*argv, "-task", "blastn"], tmp_path, tools)[1].verdict == "pass"
    )


# ---- blastn.megablast_default: the BLAST+ facts ------------------------------------------------


def hit_ids(tools, work, *options):
    argv = ["blastn", "-query", "queries.fasta", "-subject", "ref.fasta", "-evalue", "1e-5"]
    proc = tools.run([*argv, *options, "-outfmt", "6 qseqid", "-num_threads", "1"], work)
    assert proc.returncode == 0
    return sorted({line for line in proc.stdout.split() if line}), proc


def test_megablast_misses_diverged_copies_silently_and_blastn_finds_them(tools, faulted):
    task, work = faulted
    found, proc = hit_ids(tools, work)
    assert proc.stderr == "" and found == task.truth["naive"]
    assert len(found) < len(task.truth["expected"])
    assert hit_ids(tools, work, "-task", "blastn")[0] == task.truth["expected"]
    assert hit_ids(tools, work, "-task", "dc-megablast")[0] == task.truth["expected"]


def test_megablast_seeds_need_28_exact_bases(tools, tmp_path):
    rng = random.Random(9)
    ref = rand_dna(rng, 2000)
    (tmp_path / "ref.fasta").write_text(fasta_text([("chr1", ref)]))
    region = ref[300:420]
    queries = []
    for name, run in (("run20", 20), ("run27", 27), ("run28", 28), ("run40", 40)):
        copy = list(region)
        for i in range(run, len(copy), run + 1):  # a substitution after every `run` exact bases
            copy[i] = {"A": "C", "C": "G", "G": "T", "T": "A"}[copy[i]]
        queries.append((name, "".join(copy)))
    (tmp_path / "queries.fasta").write_text(fasta_text(queries))
    default, _ = hit_ids(tools, tmp_path)
    assert "run40" in default and "run20" not in default and "run27" not in default
    assert hit_ids(tools, tmp_path, "-task", "blastn")[0] == ["run20", "run27", "run28", "run40"]


# ---- blast.stale_database ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def stale(tools, tmp_path_factory):
    out = {}
    for twin in ("clean", "faulted"):
        work = tmp_path_factory.mktemp(f"f78{twin}")
        out[twin] = (build_task("F78", 4, twin, work, tools, "implicit"), work)
    return out


DB_CALL = ["-query", "queries.fasta", "-db", "proteins.fasta", "-outfmt", "6"]


def test_a_database_older_than_its_fasta_fails_until_rebuilt_or_declared(stale):
    _, work = stale["faulted"]
    decision, result = evaluate(STALE, "blastp", DB_CALL, work)
    assert decision.action == "block" and result.code == "database_older_than_fasta"
    text = render_block(decision)
    assert text.startswith("FIX: rebuild it first: makeblastdb -in proteins.fasta -dbtype prot")
    n_fasta = (work / "proteins.fasta").read_text().count(">")
    assert f"now holds {n_fasta} sequences" in result.message and "holds 4" in result.message
    assert (
        evaluate(STALE, "blastp", DB_CALL, work, declared={"database_currency": "frozen"})[
            1
        ].verdict
        == "pass"
    )
    mismatch = evaluate(STALE, "blastp", DB_CALL, work, declared={"database_currency": "current"})
    assert mismatch[1].verdict == "fail"
    assert evaluate(STALE, "blastp", [*DB_CALL, "-evalue", "1e-3"], work)[1].verdict == "fail"


def test_a_current_database_a_subject_file_and_other_programs_pass(stale, tools):
    _, work = stale["clean"]
    assert evaluate(STALE, "blastp", DB_CALL, work)[1].verdict == "pass"
    _, work = stale["faulted"]
    subject = ["-query", "queries.fasta", "-subject", "proteins.fasta"]
    assert evaluate(STALE, "blastp", subject, work)[1].verdict == "pass"
    assert (
        evaluate(STALE, "blastp", ["-query", "queries.fasta", "-db", "absent"], work)[1].verdict
        == "pass"
    )
    assert evaluate(STALE, "blastp", ["-version"], work)[1].verdict == "pass"


def test_rebuilding_in_place_or_under_a_new_name_clears_the_rule(stale, tools, tmp_path):
    _, work = stale["faulted"]
    for name in ("proteins.fasta", "queries.fasta"):
        (tmp_path / name).write_bytes((work / name).read_bytes())
    build = ["makeblastdb", "-in", "proteins.fasta", "-dbtype", "prot"]
    assert tools.run(build, tmp_path).returncode == 0
    assert evaluate(STALE, "blastp", DB_CALL, tmp_path)[1].verdict == "pass"
    assert tools.run([*build, "-out", "fresh"], tmp_path).returncode == 0
    assert (
        evaluate(STALE, "blastp", ["-query", "queries.fasta", "-db", "fresh"], tmp_path)[1].verdict
        == "pass"
    )


def test_the_letters_catch_a_changed_sequence_with_the_same_count(tools, tmp_path):
    rng = random.Random(1)
    records = [(f"p{i}", rand_prot(rng, 80)) for i in range(6)]
    (tmp_path / "db.fasta").write_text(fasta_text(records))
    assert (
        tools.run(["makeblastdb", "-in", "db.fasta", "-dbtype", "prot"], tmp_path).returncode == 0
    )
    call = ["-query", "q.fasta", "-db", "db.fasta"]
    assert evaluate(STALE, "blastp", call, tmp_path)[1].verdict == "pass"
    records[2] = ("p2", rand_prot(rng, 95))  # same number of sequences, a longer one
    (tmp_path / "db.fasta").write_text(fasta_text(records))
    result = evaluate(STALE, "blastp", call, tmp_path)[1]
    assert result.verdict == "fail" and "letters" in result.message


def test_a_fasta_too_big_to_read_in_full_is_not_compared(tmp_path):
    rng = random.Random(2)
    records = [(f"p{i}", rand_prot(rng, 300)) for i in range(7500)]  # about 2.3 MB of sequence
    (tmp_path / "big.fasta").write_text(fasta_text(records))
    meta = {"number-of-sequences": 100, "number-of-letters": 30000, "description": "big.fasta"}
    (tmp_path / "big.fasta.pjs").write_text(json.dumps(meta))
    # the checker would read 4 MB at most: a count from a partial read must not be reported
    assert (
        evaluate(STALE, "blastp", ["-query", "q.fa", "-db", "big.fasta"], tmp_path)[1].verdict
        == "pass"
    )


def test_blast_database_metadata_matches_the_fasta_and_the_search_ignores_new_proteins(
    stale, tools
):
    task, work = stale["faulted"]
    info = tools.run(["blastdbcmd", "-db", "proteins.fasta", "-info"], work)
    meta = json.loads((work / "proteins.fasta.pjs").read_text())
    n_fasta = (work / "proteins.fasta").read_text().count(">")
    assert meta["description"] == "proteins.fasta" and meta["number-of-sequences"] < n_fasta
    assert f"{meta['number-of-sequences']} sequences" in info.stdout
    proc = tools.run(["blastp", "-query", "queries.fasta", "-db", "proteins.fasta",
                      "-outfmt", "6 qseqid sseqid bitscore"], work)  # fmt: skip
    assert proc.returncode == 0 and proc.stderr == ""
    from biowarden.bench.tasks.extra.f78_blast_stale_db import best_hits

    assert (
        best_hits(tools, work, ["-db", "proteins.fasta"])
        == task.truth["naive"]
        != task.truth["expected"]
    )


# ---- admission -------------------------------------------------------------------------------


@pytest.mark.parametrize(("rule", "seeds"), [(MEGA, 10), (STALE, 20)])
def test_admission_on_calls_labelled_by_running_blast(tools, tmp_path, rule, seeds):
    spec = only(rule)
    stats = measure(rule.id, [spec], range(seeds), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", rule.id, "seeds", seeds, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * seeds // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.cells[("right", False)].n > 0 and stats.killed() > 0
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # an undeclared wrong call is always caught here: every one of them is a megablast / stale call
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n


def test_family_f77_is_the_workspace_the_cases_use():
    assert f77.SPEC.id == "F77" and f77.SPEC.locus == "tool_default"
