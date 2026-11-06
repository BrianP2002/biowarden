"""Rule trimmomatic.minlen_order (F85): the checker on real argv and FASTQ files, the Trimmomatic
facts it relies on (run with the real wrapper, -Xmx512m), and admission over calls labelled by
running Trimmomatic.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f77_blastn_task import fastq_text, phred, rand_dna
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from biowarden.engine.tools import call_rule_tools
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_trimmomatic")
(RULE,) = SPEC.rules
KEY = "minlen_stage"
SEEDS = 10
HEADS = [10, 20, 30, 40, 50, 60, 24, 58, 12, 46]  # good bases of 60-base reads
STANDARD = ["LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36"]
ALPHABETICAL = ["LEADING:3", "MINLEN:36", "SLIDINGWINDOW:4:15", "TRAILING:3"]
OPTIONS = ["-Xmx512m", "-threads", "1", "-phred33"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "trimmomatic" not in env.real_bins(["trimmomatic"]):
        pytest.skip("trimmomatic is not available")
    return env


def reads(heads, length=60, seed=1, mate=""):
    rng = random.Random(seed)
    out = []
    for i, head in enumerate(heads):
        quals = [rng.randint(32, 40) for _ in range(head)]
        quals += [rng.randint(2, 8) for _ in range(length - head)]
        out.append((f"r{i}{mate}", rand_dna(rng, length), phred(quals)))
    return fastq_text(out)


def evaluate(argv, cwd, tools=None, declared=None):
    real = tools.real_bins(call_rule_tools([SPEC])) if tools else {}
    env = tools.process_env() if tools else None
    eng = Engine([SPEC], mode="block", real_bins=real, real_env=env, retry_cap=10**9)
    decision = eng.evaluate(Call("trimmomatic", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def se(steps, inp="r.fq", out="o.fq"):
    return ["SE", *OPTIONS, inp, out, *steps]


def test_spec_is_a_conditional_t1_call_rule_on_trimmomatic():
    assert RULE.id == "trimmomatic.minlen_order" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["trimmomatic"] and RULE.declare.key == KEY
    assert not RULE.declare.required and set(RULE.declare.values) == {
        "after_trimming",
        "before_trimming",
    }
    assert RULE.check in known_checkers()


def test_the_rerun_needs_tier_t1(tmp_path):
    (tmp_path / "r.fq").write_text(reads(HEADS))
    call = Call("trimmomatic", tuple(se(ALPHABETICAL)), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0", {}))


@pytest.mark.parametrize(
    ("steps", "verdict"),
    [
        (STANDARD, "pass"),
        (["SLIDINGWINDOW:4:15", "MINLEN:36"], "pass"),
        (["MINLEN:36"], "pass"),
        (["LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15"], "pass"),
        (ALPHABETICAL, "fail"),
        (["MINLEN:36", "SLIDINGWINDOW:4:15"], "fail"),
        (["MINLEN:36", "LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15"], "fail"),
        (["LEADING:3", "TRAILING:3", "MINLEN:36", "SLIDINGWINDOW:4:15"], "fail"),
        (["MINLEN:36", "HEADCROP:30"], "fail"),  # 60 - 30 < 36 for every read
        (["MINLEN:36", "HEADCROP:2"], "pass"),  # no read falls under 36 by it
        (["MINLEN:36", "AVGQUAL:20"], "pass"),  # AVGQUAL does not shorten reads
        (["TRAILING:3", "MINLEN:36", "AVGQUAL:20"], "pass"),
    ],
)
def test_the_order_and_the_reads_decide(tools, tmp_path, steps, verdict):
    (tmp_path / "r.fq").write_text(reads(HEADS))
    decision, result = evaluate(se(steps), tmp_path, tools)
    assert result.verdict == verdict, (steps, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_reads_the_order_cannot_matter_for_stay_silent_without_a_declaration(tools, tmp_path):
    (tmp_path / "long.fq").write_text(reads([50, 55, 60, 48, 60]))  # no read ends under 36
    decision, result = evaluate(se(ALPHABETICAL, inp="long.fq"), tmp_path, tools)
    assert result.verdict == "pass" and decision.action == "allow"
    (tmp_path / "short.fq").write_text(reads([10, 12, 20], length=30))  # every read is under 36
    assert evaluate(se(ALPHABETICAL, inp="short.fq"), tmp_path, tools)[1].verdict == "pass"


def test_the_message_has_the_fix_first_and_the_instance_facts(tools, tmp_path):
    (tmp_path / "r.fq").write_text(reads(HEADS))
    decision, result = evaluate(se(ALPHABETICAL), tmp_path, tools)
    text = render_block(decision)
    fix = "FIX: put MINLEN after the trimming steps"
    assert text.startswith(f"{fix} (step order: LEADING:3 SLIDINGWINDOW:4:15 TRAILING:3 MINLEN:36)")
    assert result.code == "minlen_before_trimming"
    assert "comes before SLIDINGWINDOW, TRAILING" in result.message
    assert "keeps 10 reads, with MINLEN last 5" in result.message  # heads 40 50 60 58 46 reach 36
    assert (result.expected, result.got) == (5, 10)


def test_declarations_of_the_minlen_stage(tools, tmp_path):
    (tmp_path / "r.fq").write_text(reads(HEADS))
    assert (
        evaluate(se(ALPHABETICAL), tmp_path, tools, {KEY: "before_trimming"})[1].verdict == "pass"
    )
    decision, result = evaluate(se(ALPHABETICAL), tmp_path, None, {KEY: "after_trimming"})
    assert result.code == "declared_mismatch" and decision.action == "block"  # needs no file
    assert evaluate(se(STANDARD), tmp_path, tools, {KEY: "after_trimming"})[1].verdict == "pass"
    assert evaluate(se(ALPHABETICAL), tmp_path, tools, {KEY: "x"})[0].action == "block"


def test_paired_input_is_rerun_with_both_mates(tools, tmp_path):
    (tmp_path / "a_1.fq").write_text(reads(HEADS, mate="/1"))
    (tmp_path / "a_2.fq").write_text(reads(HEADS, seed=2, mate="/2"))
    pe = ["PE", *OPTIONS, "a_1.fq", "a_2.fq", "p1.fq", "u1.fq", "p2.fq", "u2.fq"]
    assert evaluate([*pe, *ALPHABETICAL], tmp_path, tools)[1].verdict == "fail"
    assert evaluate([*pe, *STANDARD], tmp_path, tools)[1].verdict == "pass"


def test_unreadable_situations_warn_but_never_block(tools, tmp_path):
    base = se(ALPHABETICAL)
    decision, result = evaluate(base, tmp_path, tools)  # r.fq does not exist
    assert result.verdict == "unknown" and decision.action == "warn"
    (tmp_path / "r.fq").write_text(reads(HEADS))
    decision, result = evaluate(["SE", "-baseout", "b.fq", "r.fq", *ALPHABETICAL], tmp_path, tools)
    assert result.verdict == "unknown" and decision.action == "warn"
    (tmp_path / "bad.fq").write_text("not a fastq\n")
    result = evaluate(se(ALPHABETICAL, inp="bad.fq"), tmp_path, tools)[1]
    assert result.verdict == "unknown"
    assert evaluate(["-version"], tmp_path, tools)[1].verdict == "pass"


# ---- Trimmomatic facts -----------------------------------------------------------------------


def survivors(tools, work, steps):
    proc = tools.run(["trimmomatic", *se(steps, out="out.fq")], work)
    assert proc.returncode == 0, proc.stderr
    lines = (work / "out.fq").read_text().splitlines()
    return len(lines) // 4, min(len(s) for s in lines[1::4])


def test_minlen_before_the_trimming_steps_keeps_reads_that_the_later_steps_cut_short(
    tools, tmp_path
):
    (tmp_path / "r.fq").write_text(reads(HEADS))
    kept, shortest = survivors(tools, tmp_path, STANDARD)
    assert (kept, shortest) == (5, 40) or kept == sum(h >= 40 for h in HEADS) and shortest >= 36
    assert survivors(tools, tmp_path, ["MINLEN:36", "SLIDINGWINDOW:4:15"])[0] == len(HEADS)
    assert survivors(tools, tmp_path, ["SLIDINGWINDOW:4:15", "MINLEN:36"])[0] == kept
    assert survivors(tools, tmp_path, ["MINLEN:36", *STANDARD[:3]])[0] == len(HEADS)
    assert survivors(tools, tmp_path, ALPHABETICAL)[0] == len(HEADS) > kept


def test_family_f85_workspaces_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        task = build_task("F85", 3, twin, work, tools, "implicit")
        call = se(ALPHABETICAL, inp="reads.fastq")
        verdict = evaluate(call, work, tools)[1].verdict
        assert verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate(se(STANDARD, inp="reads.fastq"), work, tools)[1].verdict == "pass"
        n_naive = survivors_listed(tools, work)
        assert n_naive == task.truth["naive"]


def survivors_listed(tools, work):
    argv = ["trimmomatic", *se(ALPHABETICAL, inp="reads.fastq", out="o_listed.fq")]
    assert tools.run(argv, work).returncode == 0
    return len((work / "o_listed.fq").read_text().splitlines()) // 4


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_trimmomatic(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(SEEDS), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", RULE.id, "seeds", SEEDS, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * SEEDS // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
