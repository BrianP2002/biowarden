"""Rule seqtk.trimfq_floor (F81): the checker on real argv and FASTQ files, the seqtk facts it
relies on (run with the real program), and admission over calls labelled by running seqtk.
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
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_seqtk")
(RULE,) = SPEC.rules
KEY = "trim_floor"
HEADS = [25, 33, 40, 23, 17, 18, 19, 29, 57, 44, 38, 20]  # good bases of the 60-base reads


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "seqtk" not in env.real_bins(["seqtk"]):
        pytest.skip("seqtk is not available")
    return env


def reads(heads, length=60, seed=1):
    rng = random.Random(seed)
    out = []
    for i, head in enumerate(heads):
        quals = [rng.randint(32, 40) for _ in range(head)]
        quals += [rng.randint(2, 8) for _ in range(length - head)]
        out.append((f"r{i}", rand_dna(rng, length), phred(quals)))
    return out


def write(path: Path, heads, length=60):
    path.write_text(fastq_text(reads(heads, length)))
    return path


def evaluate(argv, cwd, tools, declared=None):
    eng = Engine(
        [SPEC],
        mode="block",
        real_bins=tools.real_bins(call_rule_tools([SPEC])),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )
    decision = eng.evaluate(Call("seqtk", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_conditional_t1_call_rule_on_trimfq():
    assert RULE.id == "seqtk.trimfq_floor" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["seqtk"] and RULE.applies.subcommands == ["trimfq"]
    assert RULE.declare.key == KEY and not RULE.declare.required
    assert set(RULE.declare.values) == {"keep_minimum", "no_minimum"}
    assert RULE.check in known_checkers()


def test_the_checker_reruns_seqtk_so_it_needs_tier_t1(tmp_path):
    write(tmp_path / "r.fq", HEADS)
    call = Call("seqtk", ("trimfq", "-q", "0.01", "r.fq"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("args", "verdict"),
    [
        (["-q", "0.01"], "fail"),
        ([], "fail"),
        (["-q", "0.05"], "fail"),
        (["-q", "0.01", "-l", "30"], "fail"),
        (["-q", "0.01", "-l", "20"], "fail"),
        (["-q", "0.01", "-l30"], "fail"),
        (["-Q", "-q", "0.01"], "fail"),
        (["-q", "0.01", "-l", "1"], "pass"),
        (["-q", "0.01", "-l", "0"], "pass"),
        (["-q", "0.01", "-l1"], "pass"),
        (["-l", "10", "-q", "0.001"], "pass"),  # no read has fewer than 17 good bases
        (["-l", "18", "-q", "0.001"], "fail"),  # the reads with 17 good bases keep 18
        (["-q", "0.01", "-l", "2"], "pass"),  # no read has fewer than 17 good bases here
        (["-b", "5"], "pass"),  # a fixed trim switches quality trimming off
        (["-e", "5"], "pass"),
        (["-L", "50"], "pass"),
    ],
)
def test_the_floor_decides_on_reads_with_short_good_heads(tools, tmp_path, args, verdict):
    write(tmp_path / "r.fq", HEADS)
    decision, result = evaluate(["trimfq", *args, "r.fq"], tmp_path, tools)
    assert result.verdict == verdict, (args, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_long_good_heads_stay_silent_and_so_do_other_inputs(tools, tmp_path):
    write(tmp_path / "long.fq", [40, 45, 60, 52])
    assert evaluate(["trimfq", "-q", "0.01", "long.fq"], tmp_path, tools)[1].verdict == "pass"
    (tmp_path / "reads.fa").write_text(">a\nACGTACGTAC\n")  # FASTA: no qualities
    for argv in (
        ["trimfq", "absent.fq"],
        ["trimfq", "-q", "0.01", "-"],
        ["trimfq", "-q", "0.01"],
        ["trimfq", "-q", "0.01", "reads.fa"],
    ):
        assert evaluate(argv, tmp_path, tools)[1].verdict == "pass", argv
    assert evaluate(["seq", "-A", "long.fq"], tmp_path, tools)[0].outcomes == ()


def test_the_message_has_the_fix_first_and_the_instance_facts(tools, tmp_path):
    write(tmp_path / "r.fq", HEADS)
    decision, result = evaluate(["trimfq", "-q", "0.01", "r.fq"], tmp_path, tools)
    text = render_block(decision)
    assert text.startswith("FIX: add -l 1 (trim every low-quality tail")
    assert (
        result.code == "trim_floor_binds" and "never cuts a read below 30 bases" in result.message
    )
    assert "of 12 reads in r.fq" in result.message and "the default -l 30" in result.message
    assert "r0: 25 good bases, 30 kept" in result.message


def test_declarations_of_the_trim_floor(tools, tmp_path):
    write(tmp_path / "r.fq", HEADS)
    call = ["trimfq", "-q", "0.01", "r.fq"]
    assert evaluate(call, tmp_path, tools, {KEY: "keep_minimum"})[1].verdict == "pass"
    assert evaluate(call, tmp_path, tools, {KEY: "no_minimum"})[1].verdict == "fail"
    done = ["trimfq", "-q", "0.01", "-l", "1", "r.fq"]
    assert evaluate(done, tmp_path, tools, {KEY: "no_minimum"})[1].verdict == "pass"
    assert evaluate(call, tmp_path, tools, {KEY: "bad"})[0].action == "block"
    write(tmp_path / "ok.fq", [40, 50])
    ok = ["trimfq", "-q", "0.01", "ok.fq"]
    assert evaluate(ok, tmp_path, tools, {KEY: "no_minimum"})[1].verdict == "pass"


# ---- seqtk facts -----------------------------------------------------------------------------


def trimmed_lengths(tools, work, *options):
    proc = tools.run(["seqtk", "trimfq", *options, "r.fq"], work)
    assert proc.returncode == 0 and proc.stderr == ""
    return [len(line) for line in proc.stdout.splitlines()[1::4]]


def test_trimfq_keeps_30_bases_by_default_and_exactly_the_good_head_with_l_1(tools, tmp_path):
    write(tmp_path / "r.fq", HEADS)
    expected = [h if h < 60 else 60 for h in HEADS]
    default = trimmed_lengths(tools, tmp_path, "-q", "0.01")
    assert default == [max(h, 30) if h < 60 else 60 for h in HEADS]
    assert trimmed_lengths(tools, tmp_path, "-q", "0.01", "-l", "1") == expected
    assert trimmed_lengths(tools, tmp_path, "-q", "0.01", "-l", "0") == expected
    # the threshold is an error rate: Q20 as "20" trims nothing
    assert trimmed_lengths(tools, tmp_path, "-q", "20") == [60] * len(HEADS)


def test_the_family_sets_show_the_floor_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        task = build_task("F81", 7, twin, work, tools, "implicit")
        decision, result = evaluate(["trimfq", "-q", "0.01", "reads.fastq"], work, tools)
        assert result.verdict == ("pass" if twin == "clean" else "fail")
        assert task.truth["naive"] == (
            task.truth["expected"] if twin == "clean" else task.truth["naive"]
        )
        done = evaluate(["trimfq", "-q", "0.01", "-l", "1", "reads.fastq"], work, tools)[1]
        assert done.verdict == "pass"


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_seqtk(tools, tmp_path):
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
