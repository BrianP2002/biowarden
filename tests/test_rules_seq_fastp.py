"""Rules fastp.quality_trimming (F83) and fastp.adapter_detection (F87): the checkers on real argv
and FASTQ files, the fastp facts they rely on (run with the real program), and admission over
calls labelled by running fastp.
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
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_fastp")
QUALITY, ADAPTER = SPEC.rules
TRUSEQ = "AGATCGGAAGAGC"
FULL = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"  # the full TruSeq read-1 adapter


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "fastp" not in env.real_bins(["fastp"]):
        pytest.skip("fastp is not available")
    return env


def evaluate(rule, argv, cwd, declared=None):
    eng = Engine([SPEC.model_copy(update={"rules": [rule]})], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("fastp", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def tailed(path: Path, n=20, tail=12, share=0.5, seed=1) -> Path:
    rng = random.Random(seed)
    reads = []
    for i in range(n):
        has_tail = rng.random() < share
        head = 60 if has_tail else 80
        quals = [rng.randint(38, 40) for _ in range(head)]
        quals += [rng.randint(14, 18) for _ in range(20 if has_tail else 0)]
        reads.append((f"r{i}", rand_dna(rng, len(quals)), phred(quals)))
    path.write_text(fastq_text(reads))
    return path


def adapter_reads(path: Path, n=300, share=0.6, adapter=TRUSEQ, seed=2) -> Path:
    rng = random.Random(seed)
    reads = []
    for i in range(n):
        seq = rand_dna(rng, 100)
        if rng.random() < share:
            insert = rng.randint(30, 70)
            seq = (rand_dna(rng, insert) + adapter + rand_dna(rng, 100))[:100]
        reads.append((f"r{i}", seq, phred([37] * 100)))
    path.write_text(fastq_text(reads))
    return path


def test_specs_are_conditional_t1_call_rules_on_fastp():
    assert [r.id for r in SPEC.rules] == ["fastp.quality_trimming", "fastp.adapter_detection"]
    for rule in SPEC.rules:
        assert (rule.kind, rule.tier) == ("call", "T1") and rule.applies.tools == ["fastp"]
        assert rule.declare is not None and not rule.declare.required
        assert rule.check in known_checkers()
    assert QUALITY.declare.key == "low_quality_goal"
    assert set(QUALITY.declare.values) == {"trim_tails", "filter_reads"}
    assert ADAPTER.declare.key == "adapter_trimming"
    assert set(ADAPTER.declare.values) == {"adapter_given", "none_needed"}


def test_both_checkers_read_the_fastq_so_they_need_tier_t1(tmp_path):
    tailed(tmp_path / "r.fq")
    adapter_reads(tmp_path / "a.fq")
    with pytest.raises(TierViolation):
        known_checkers()[QUALITY.check](
            Call("fastp", ("-i", "r.fq", "-o", "o.fq", "-q", "20"), tmp_path),
            Match(QUALITY),
            Ctx(tmp_path, "T0"),
        )
    with pytest.raises(TierViolation):
        known_checkers()[ADAPTER.check](
            Call("fastp", ("-i", "a.fq", "-o", "o.fq"), tmp_path),
            Match(ADAPTER),
            Ctx(tmp_path, "T0"),
        )


# ---- fastp.quality_trimming --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "verdict"),
    [
        (["-q", "20"], "fail"),
        (["-q", "15"], "fail"),
        (["-u", "20"], "fail"),
        (["-e", "25"], "fail"),
        (["--qualified_quality_phred", "20"], "fail"),
        (["--unqualified_percent_limit", "30"], "fail"),
        (["-q", "20", "-l", "30"], "fail"),
        (["--cut_right"], "pass"),
        (["-q", "20", "--cut_right"], "pass"),
        (["-q", "20", "-r", "-W", "4", "-M", "20"], "pass"),
        (["-q", "20", "--cut_tail"], "pass"),
        (["-q", "20", "-3"], "pass"),
        (["-q", "20", "--cut_front"], "pass"),
        (["-q", "20", "-5"], "pass"),
        ([], "pass"),  # no quality option: not an attempt at quality handling
        (["-l", "30", "-A"], "pass"),
    ],
)
def test_the_options_decide_on_reads_with_low_quality_tails(tmp_path, extra, verdict):
    tailed(tmp_path / "r.fq")
    decision, result = evaluate(QUALITY, ["-i", "r.fq", "-o", "o.fq", *extra], tmp_path)
    assert result.verdict == verdict, (extra, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_reads_without_tails_and_other_inputs_stay_silent(tmp_path):
    tailed(tmp_path / "clean.fq", share=0.0)
    assert (
        evaluate(QUALITY, ["-i", "clean.fq", "-o", "o.fq", "-q", "20"], tmp_path)[1].verdict
        == "pass"
    )
    for argv in (
        ["-i", "absent.fq", "-q", "20"],
        ["-q", "20"],
        ["--in1", "absent.fq", "-q", "20"],
    ):
        assert evaluate(QUALITY, argv, tmp_path)[1].verdict == "pass", argv
    tailed(tmp_path / "few.fq", n=60, share=0.02, seed=5)  # under 3% of the reads
    assert evaluate(QUALITY, ["-i", "few.fq", "-q", "20"], tmp_path)[1].verdict == "pass"


def test_the_threshold_is_the_q_value_of_the_call(tmp_path):
    tailed(tmp_path / "r.fq")  # tails of Q14-18
    low = evaluate(QUALITY, ["-i", "r.fq", "-o", "o.fq", "-q", "10"], tmp_path)[1]
    assert low.verdict == "pass"  # no 4-base window is below Q10
    assert (
        evaluate(QUALITY, ["-i", "r.fq", "-o", "o.fq", "-q", "20"], tmp_path)[1].verdict == "fail"
    )


def test_quality_message_has_the_fix_first_and_the_facts(tmp_path):
    tailed(tmp_path / "r.fq")
    decision, result = evaluate(QUALITY, ["-i", "r.fq", "-o", "o.fq", "-q", "20"], tmp_path)
    text = render_block(decision)
    assert text.startswith(
        "FIX: add --cut_right --cut_right_window_size 4 --cut_right_mean_quality 20"
    )
    assert result.code == "filter_not_trimmer"
    assert "fastp -q is a read filter, not a trimmer" in result.message
    assert "of the 20 reads sampled from r.fq" in result.message


def test_declarations_of_the_low_quality_goal(tmp_path):
    tailed(tmp_path / "r.fq")
    call = ["-i", "r.fq", "-o", "o.fq", "-q", "20"]
    assert (
        evaluate(QUALITY, call, tmp_path, {"low_quality_goal": "filter_reads"})[1].verdict == "pass"
    )
    decision, result = evaluate(QUALITY, call, tmp_path, {"low_quality_goal": "trim_tails"})
    assert result.code == "declared_mismatch" and decision.action == "block"
    done = [*call, "--cut_right"]
    assert (
        evaluate(QUALITY, done, tmp_path, {"low_quality_goal": "trim_tails"})[1].verdict == "pass"
    )
    assert evaluate(QUALITY, call, tmp_path, {"low_quality_goal": "x"})[0].action == "block"


def run_fastp(tools, work, *options):
    argv = ["fastp", "-i", "r.fq", "-o", "o.fq", "-j", "o.json", "-h", "o.html", "--thread", "1"]
    proc = tools.run([*argv, *options], work)
    assert proc.returncode == 0 and "warn" not in proc.stderr.lower()
    lines = (work / "o.fq").read_text().splitlines()
    return len(lines) // 4, sum(len(s) for s in lines[1::4])


def test_fastp_q_filters_reads_and_leaves_tails_while_cut_right_trims_them(tools, tmp_path):
    rng = random.Random(4)
    reads = []
    for i in range(50):
        head = rng.randint(22, 70) if i % 2 else 80
        quals = [rng.randint(38, 40) for _ in range(head)] + [
            rng.randint(14, 18) for _ in range(80 - head)
        ]
        reads.append((f"r{i}", rand_dna(rng, 80), phred(quals), head))
    (tmp_path / "r.fq").write_text(fastq_text([(n, s, q) for n, s, q, _ in reads]))
    heads = sum(h for *_, h in reads)
    assert run_fastp(tools, tmp_path, "--cut_right", "-W", "4", "-M", "20") == (50, heads)
    n_q, bases_q = run_fastp(tools, tmp_path, "-q", "20")
    assert n_q < 50 and bases_q != heads  # reads dropped, the tails of the others kept
    assert (
        run_fastp(tools, tmp_path, "-q", "20", "-u", "100")[0] == 50
    )  # nothing dropped, nothing cut
    assert run_fastp(tools, tmp_path, "-q", "20", "-u", "100")[1] == 80 * 50


def test_family_f83_workspaces_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F83", 8, twin, work, tools, "implicit")
        call = ["-i", "reads.fastq", "-o", "o.fq", "-q", "20"]
        assert evaluate(QUALITY, call, work)[1].verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate(QUALITY, [*call, "--cut_right"], work)[1].verdict == "pass"


# ---- fastp.adapter_detection -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "verdict"),
    [
        ([], "fail"),
        (["-l", "30"], "fail"),
        (["--detect_adapter_for_pe"], "fail"),  # single-end: no effect
        (["-a", TRUSEQ], "pass"),
        (["--adapter_sequence", TRUSEQ], "pass"),
        (["--adapter_sequence=" + TRUSEQ], "pass"),
        (["-a", "AGATCGGAAGAGCACACGTC", "-q", "15"], "pass"),
        (["--adapter_fasta", "adapters.fa"], "pass"),
        (["-A"], "pass"),
        (["--disable_adapter_trimming"], "pass"),
        (["-I", "r2.fq", "-O", "o2.fq"], "pass"),  # paired-end: overlap analysis, no detection
    ],
)
def test_naming_the_adapter_decides_on_a_small_file_with_adapters(tmp_path, extra, verdict):
    adapter_reads(tmp_path / "r.fq")
    decision, result = evaluate(ADAPTER, ["-i", "r.fq", "-o", "o.fq", *extra], tmp_path)
    assert result.verdict == verdict, (extra, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_other_known_adapters_are_found_and_unrelated_reads_stay_silent(tmp_path):
    for name, adapter in (("nextera", "CTGTCTCTTATACACATCT"), ("small", "TGGAATTCTCGGGTGCCAAGG")):
        adapter_reads(tmp_path / f"{name}.fq", adapter=adapter)
        result = evaluate(ADAPTER, ["-i", f"{name}.fq", "-o", "o.fq"], tmp_path)[1]
        assert result.verdict == "fail", name
    adapter_reads(tmp_path / "none.fq", share=0.0)
    assert evaluate(ADAPTER, ["-i", "none.fq", "-o", "o.fq"], tmp_path)[1].verdict == "pass"
    adapter_reads(tmp_path / "few.fq", share=0.01, seed=9)  # under 5% of the reads
    assert evaluate(ADAPTER, ["-i", "few.fq", "-o", "o.fq"], tmp_path)[1].verdict == "pass"
    for argv in (["-i", "absent.fq"], ["-o", "o.fq"]):
        assert evaluate(ADAPTER, argv, tmp_path)[1].verdict == "pass"


def test_ten_thousand_reads_are_detected_by_fastp_itself_and_the_rule_stays_silent(tmp_path):
    adapter_reads(tmp_path / "big.fq", n=10_000, share=0.75)
    assert evaluate(ADAPTER, ["-i", "big.fq", "-o", "o.fq"], tmp_path)[1].verdict == "pass"
    adapter_reads(tmp_path / "just_under.fq", n=9_999, share=0.75)
    assert evaluate(ADAPTER, ["-i", "just_under.fq", "-o", "o.fq"], tmp_path)[1].verdict == "fail"


def test_a_file_cut_short_by_the_byte_limit_is_not_taken_for_a_small_one(tmp_path):
    rng = random.Random(3)
    reads = []
    for i in range(9_500):  # 2 x 200 bases per record: the 4 MB limit stops the read before 10,000
        seq = (rand_dna(rng, 60) + FULL + rand_dna(rng, 300))[:200]
        reads.append((f"r{i}", seq, phred([36] * 200)))
    (tmp_path / "wide.fq").write_text(fastq_text(reads))
    assert evaluate(ADAPTER, ["-i", "wide.fq", "-o", "o.fq"], tmp_path)[1].verdict == "pass"


def test_adapter_message_and_declarations(tmp_path):
    adapter_reads(tmp_path / "r.fq")
    call = ["-i", "r.fq", "-o", "o.fq"]
    decision, result = evaluate(ADAPTER, call, tmp_path)
    text = render_block(decision)
    assert text.startswith("FIX: add -a AGATCGGAAGAGC (the adapter sequence)")
    assert (
        result.code == "adapter_not_detected_in_small_file"
        and "TruSeq / universal" in result.message
    )
    assert (
        evaluate(ADAPTER, call, tmp_path, {"adapter_trimming": "none_needed"})[1].verdict == "pass"
    )
    assert (
        evaluate(ADAPTER, call, tmp_path, {"adapter_trimming": "adapter_given"})[1].verdict
        == "fail"
    )
    assert (
        evaluate(ADAPTER, [*call, "-a", TRUSEQ], tmp_path, {"adapter_trimming": "adapter_given"})[
            1
        ].verdict
        == "pass"
    )
    assert evaluate(ADAPTER, call, tmp_path, {"adapter_trimming": "x"})[0].action == "block"


def reads_with_adapter(tools, work, *options):
    proc = tools.run(
        ["fastp", "-i", "r.fq", "-o", "o.fq", "-j", "o.json", "-h", "o.html", *options], work
    )
    assert proc.returncode == 0
    seqs = (work / "o.fq").read_text().splitlines()[1::4]
    return sum(TRUSEQ[:12] in s for s in seqs), proc.stderr


def test_fastp_detects_single_end_adapters_only_from_10000_reads_on(tools, tmp_path):
    adapter_reads(tmp_path / "r.fq", n=9_999, share=0.75, adapter=FULL)
    left, err = reads_with_adapter(tools, tmp_path)
    assert left > 6000 and "No adapter detected" in err
    adapter_reads(tmp_path / "r.fq", n=10_000, share=0.75, adapter=FULL)
    left, err = reads_with_adapter(tools, tmp_path)
    assert left == 0 and "No adapter detected" not in err
    adapter_reads(tmp_path / "r.fq", n=500, share=0.75, adapter=FULL)
    assert reads_with_adapter(tools, tmp_path)[0] > 300
    assert reads_with_adapter(tools, tmp_path, "-a", TRUSEQ)[0] == 0


def test_family_f87_workspaces_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F87", 9, twin, work, tools, "implicit")
        call = ["-i", "reads.fastq", "-o", "o.fq"]
        assert evaluate(ADAPTER, call, work)[1].verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate(ADAPTER, [*call, "-a", TRUSEQ], work)[1].verdict == "pass"


# ---- admission -------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [QUALITY, ADAPTER])
def test_admission_on_calls_labelled_by_running_fastp(tools, tmp_path, rule):
    seeds = 14
    stats = measure(
        rule.id, [SPEC.model_copy(update={"rules": [rule]})], range(seeds), tools, tmp_path
    )
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", rule.id, "seeds", seeds, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * seeds // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
