"""Rule msa.sequence_orientation (F86): the checker on real argv and FASTA files, the MAFFT / MUSCLE
facts it relies on (run with the real programs), and admission over calls labelled by running them.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f86_mafft_orient as f86
from biowarden.bench.tasks.extra.f77_blastn_task import fasta_text, rand_prot
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules.seq_msa import reversed_members
from biowarden.engine.spec import load_packaged
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_msa")
(RULE,) = SPEC.rules
KEY = "sequence_orientation"
SEEDS = 10
NEEDS = ["mafft", "muscle"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not set(env.real_bins(NEEDS)) >= set(NEEDS):
        pytest.skip("mafft and muscle are not available")
    return env


def write_set(path: Path, seed=1, twin="faulted") -> list[str]:
    plan = f86.make_plan(seed)
    path.write_text(fasta_text(f86.written(plan, twin)))
    return [plan["members"][i][0] for i in sorted(plan["flipped"])]


def evaluate(tool, argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_conditional_t1_call_rule_on_mafft_and_muscle():
    assert RULE.id == "msa.sequence_orientation" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["mafft", "muscle"] and RULE.declare.key == KEY
    assert not RULE.declare.required and set(RULE.declare.values) == {
        "same_strand",
        "mixed_strands",
    }
    assert RULE.check in known_checkers()


def test_the_checker_reads_the_sequences_so_it_needs_tier_t1(tmp_path):
    write_set(tmp_path / "seqs.fasta")
    call = Call("mafft", ("--quiet", "seqs.fasta"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("tool", "args", "verdict"),
    [
        ("mafft", ["--quiet", "seqs.fasta"], "fail"),
        ("mafft", ["--auto", "seqs.fasta"], "fail"),
        ("mafft", ["--localpair", "--maxiterate", "100", "--thread", "1", "seqs.fasta"], "fail"),
        ("mafft", ["--reorder", "seqs.fasta"], "fail"),
        ("mafft", ["--adjustdirection", "seqs.fasta"], "pass"),
        ("mafft", ["--adjustdirectionaccurately", "seqs.fasta"], "pass"),
        ("mafft", ["--adjustdirection", "--auto", "seqs.fasta"], "pass"),
        ("mafft", ["--quiet", "absent.fasta"], "pass"),
        ("mafft", ["--version"], "pass"),
        ("muscle", ["-align", "seqs.fasta", "-output", "o.afa"], "fail"),
        ("muscle", ["-super5", "seqs.fasta", "-output", "o.afa"], "fail"),
        ("muscle", ["-align", "seqs.fasta", "-output", "o.afa", "-threads", "1"], "fail"),
        ("muscle", ["-version"], "pass"),
    ],
)
def test_the_option_decides_when_the_input_holds_reversed_members(tmp_path, tool, args, verdict):
    write_set(tmp_path / "seqs.fasta")
    decision, result = evaluate(tool, args, tmp_path)
    assert result.verdict == verdict, (tool, args, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_sequences_on_one_strand_proteins_and_tiny_files_stay_silent(tmp_path):
    write_set(tmp_path / "same.fasta", twin="clean")
    assert evaluate("mafft", ["same.fasta"], tmp_path)[1].verdict == "pass"
    assert (
        evaluate("muscle", ["-align", "same.fasta", "-output", "o"], tmp_path)[1].verdict == "pass"
    )
    rng = random.Random(1)
    (tmp_path / "prot.fasta").write_text(
        fasta_text([(f"p{i}", rand_prot(rng, 80)) for i in range(6)])
    )
    assert evaluate("mafft", ["prot.fasta"], tmp_path)[1].verdict == "pass"
    write_set(tmp_path / "all.fasta")
    two = ">a\n" + "ACGTTGCAAGGCTTAACCGGTT" * 3 + "\n>b\n" + "AACCGGTTAAGCCTTGCAACGT" * 3 + "\n"
    (tmp_path / "two.fasta").write_text(two)
    assert evaluate("mafft", ["two.fasta"], tmp_path)[1].verdict == "pass"  # fewer than 3 records


def test_the_detector_names_the_reversed_members_and_nothing_else(tmp_path):
    names = write_set(tmp_path / "seqs.fasta", seed=4)
    plan = f86.make_plan(4)
    records = f86.written(plan, "faulted")
    assert sorted(reversed_members(records)) == sorted(names)
    assert reversed_members(f86.written(plan, "clean")) == []
    _, result = evaluate("mafft", ["seqs.fasta"], tmp_path)
    assert result.code == "reverse_complemented_input" and names[0] in result.message


def test_the_message_has_the_fix_first_and_the_instance_facts(tmp_path):
    write_set(tmp_path / "seqs.fasta")
    decision, result = evaluate("mafft", ["--quiet", "seqs.fasta"], tmp_path)
    text = render_block(decision)
    assert text.startswith("FIX: add --adjustdirectionaccurately (flipped sequences are renamed")
    assert (
        "of 8 sequences in seqs.fasta" in result.message
        or " sequences in seqs.fasta" in result.message
    )
    decision, result = evaluate("muscle", ["-align", "seqs.fasta", "-output", "o.afa"], tmp_path)
    assert render_block(decision).startswith("FIX: orient the reversed sequences first")
    assert "muscle -align has no option that flips sequences" in result.message


def test_declarations_of_the_sequence_orientation(tmp_path):
    write_set(tmp_path / "seqs.fasta")
    call = ["--quiet", "seqs.fasta"]
    decision, result = evaluate("mafft", call, tmp_path, {KEY: "same_strand"})
    assert (
        result.code == "declared_mismatch"
        and "you declared sequence_orientation=same_strand" in result.message
    )
    decision, result = evaluate("mafft", call, tmp_path, {KEY: "mixed_strands"})
    assert result.code == "declared_mismatch" and "does not flip sequences" in result.message
    flipped = ["--adjustdirection", "seqs.fasta"]
    assert evaluate("mafft", flipped, tmp_path, {KEY: "mixed_strands"})[1].verdict == "pass"
    write_set(tmp_path / "same.fasta", twin="clean")
    assert evaluate("mafft", ["same.fasta"], tmp_path, {KEY: "same_strand"})[1].verdict == "pass"
    assert evaluate("mafft", call, tmp_path, {KEY: "x"})[0].action == "block"


# ---- MAFFT and MUSCLE facts ------------------------------------------------------------------


def alignment(text):
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            rows[name] = ""
        elif rows:
            rows[name] += line.strip()
    return rows


def test_mafft_aligns_as_written_unless_told_to_flip_and_then_renames(tools, tmp_path):
    names = write_set(tmp_path / "seqs.fasta", seed=2)
    task = build_task("F86", 2, "faulted", tmp_path / "ws", tools, "implicit")
    base = ["mafft", "--quiet", "--thread", "1"]
    plain = tools.run([*base, "seqs.fasta"], tmp_path)
    flipped = tools.run([*base, "--adjustdirection", "seqs.fasta"], tmp_path)
    accurate = tools.run([*base, "--adjustdirectionaccurately", "seqs.fasta"], tmp_path)
    assert plain.returncode == flipped.returncode == 0 and plain.stderr == ""
    assert f86.conserved_columns(plain.stdout) < task.truth["expected"] - 50
    assert f86.conserved_columns(flipped.stdout) == f86.conserved_columns(accurate.stdout)
    assert f86.conserved_columns(accurate.stdout) == task.truth["expected"]
    assert sorted(n for n in alignment(flipped.stdout) if n.startswith("_R_")) == sorted(
        f"_R_{n}" for n in names
    )
    assert list(alignment(plain.stdout)) == [n for n, _ in f86.written(f86.make_plan(2), "faulted")]
    reordered = tools.run([*base, "--reorder", "seqs.fasta"], tmp_path)
    assert list(alignment(reordered.stdout)) != list(alignment(plain.stdout))


def test_muscle_aligns_the_reversed_member_as_unrelated_and_writes_rows_in_tree_order(
    tools, tmp_path
):
    write_set(tmp_path / "seqs.fasta", seed=2)
    task = build_task("F86", 2, "faulted", tmp_path / "ws", tools, "implicit")
    proc = tools.run(
        ["muscle", "-align", "seqs.fasta", "-output", "o.afa", "-threads", "1"], tmp_path
    )
    assert proc.returncode == 0
    text = (tmp_path / "o.afa").read_text()
    assert f86.conserved_columns(text) < task.truth["expected"] - 50
    given = [n for n, _ in f86.written(f86.make_plan(2), "faulted")]
    assert sorted(alignment(text)) == sorted(given) and list(alignment(text)) != given
    argv = ["muscle", "-align", "seqs.fasta", "-output", "p.afa", "-stable", "-threads", "1"]
    stable = tools.run(argv, tmp_path)
    assert stable.returncode != 0 and "stable" in stable.stderr + stable.stdout


def test_family_workspaces_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F86", 3, twin, work, tools, "implicit")
        verdict = evaluate("mafft", ["--quiet", "seqs.fasta"], work)[1].verdict
        assert verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate("mafft", ["--adjustdirection", "seqs.fasta"], work)[1].verdict == "pass"


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_mafft_and_muscle(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(SEEDS), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", RULE.id, "seeds", SEEDS, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * SEEDS // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.cells[("wrong", True)].failed == stats.cells[("wrong", True)].n  # recall 1.0
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
