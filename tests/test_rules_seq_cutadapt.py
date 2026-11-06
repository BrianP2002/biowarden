"""Rule cutadapt.adapters_per_read (F84): the checker on real argv and FASTQ files, the cutadapt
facts it relies on (run with the real program), and admission over calls labelled by running it.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import biowarden.engine.rules  # noqa: F401  (registers the checkers)
from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f77_blastn_task import fastq_text, phred, rand_dna
from biowarden.bench.tasks.extra.f84_cutadapt_adapters import ADAPTER, PRIMER
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.spec import load_packaged
from helpers import promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("seq_cutadapt")
(RULE,) = SPEC.rules
KEY = "adapters_per_read"
SEEDS = 16
OTHER = "CTGTCTCTTATACACATCTG"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "cutadapt" not in env.real_bins(["cutadapt"]):
        pytest.skip("cutadapt is not available")
    return env


def amplicons(path: Path, n=12, adapter=ADAPTER, both=True, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        seq = PRIMER + rand_dna(rng, rng.randint(20, 35)) + (adapter if both else "")
        seq += rand_dna(rng, 100)
        out.append((f"r{i}", seq[:80], phred([36] * 80)))
    path.write_text(fastq_text(out))
    return path


def evaluate(argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("cutadapt", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_conditional_t1_call_rule_on_cutadapt():
    assert RULE.id == "cutadapt.adapters_per_read" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["cutadapt"] and RULE.declare.key == KEY
    assert not RULE.declare.required and set(RULE.declare.values) == {"one", "all"}
    assert RULE.check in known_checkers()


def test_the_checker_samples_the_reads_so_it_needs_tier_t1(tmp_path):
    amplicons(tmp_path / "r.fq")
    call = Call("cutadapt", ("-g", PRIMER, "-a", ADAPTER, "r.fq"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("args", "verdict"),
    [
        (["-g", PRIMER, "-a", ADAPTER], "fail"),
        (["-a", ADAPTER, "-g", PRIMER], "fail"),
        (["-g", PRIMER, "-a", ADAPTER, "-n", "1"], "fail"),
        (["-g", PRIMER, "-a", ADAPTER, "--times", "1"], "fail"),
        ([f"-g{PRIMER}", f"-a{ADAPTER}"], "fail"),
        (["--front", PRIMER, "--adapter", ADAPTER], "fail"),
        (["-b", PRIMER, "-b", ADAPTER], "fail"),
        (["-g", PRIMER, "-a", ADAPTER, "-m", "10", "-q", "20"], "fail"),
        (["-g", PRIMER, "-a", ADAPTER, "-n", "2"], "pass"),
        (["-g", PRIMER, "-a", ADAPTER, "--times=2"], "pass"),
        (["-g", PRIMER, "-a", ADAPTER, "-n", "3"], "pass"),
        (["-g", f"{PRIMER}...{ADAPTER}"], "pass"),  # a linked adapter removes both
        (["-a", f"{PRIMER}...{ADAPTER}"], "pass"),
        (["-g", PRIMER], "pass"),  # one adapter: nothing to lose
        (["-a", ADAPTER], "pass"),
        (["-g", PRIMER, "-a", OTHER], "pass"),  # no read carries both listed adapters
        (["-g", f"^{PRIMER}", "-a", f"{ADAPTER}$"], "fail"),  # anchors do not hide the sequences
        (["-g", f"{PRIMER};e=0.2", "-a", ADAPTER], "fail"),
    ],
)
def test_the_times_option_and_the_reads_decide(tmp_path, args, verdict):
    amplicons(tmp_path / "r.fq")
    decision, result = evaluate([*args, "-o", "out.fq", "r.fq"], tmp_path)
    assert result.verdict == verdict, (args, result)
    assert decision.action == ("block" if verdict == "fail" else "allow")


def test_reads_with_only_one_of_the_adapters_stay_silent(tmp_path):
    amplicons(tmp_path / "primer_only.fq", both=False)
    call = ["-g", PRIMER, "-a", ADAPTER, "-o", "out.fq", "primer_only.fq"]
    assert evaluate(call, tmp_path)[1].verdict == "pass"
    for argv in (
        [*call[:4], "-o", "out.fq", "absent.fq"],
        [*call[:4], "-o", "out.fq"],
        [*call[:4], "-o", "out.fq", "-"],
        ["--version"],
    ):
        assert evaluate(argv, tmp_path)[1].verdict == "pass", argv


def test_paired_files_are_checked_per_side(tmp_path):
    amplicons(tmp_path / "r1.fq", both=False)
    amplicons(tmp_path / "r2.fq", both=True, seed=2)
    pe = ["-g", PRIMER, "-A", PRIMER, "-G", ADAPTER, "-o", "o1.fq", "-p", "o2.fq", "r1.fq", "r2.fq"]
    decision, result = evaluate(pe, tmp_path)
    assert result.verdict == "fail" and "R2" in result.message
    assert evaluate([*pe[:-2], "-n", "2", "r1.fq", "r2.fq"], tmp_path)[1].verdict == "pass"
    one_each = ["-a", ADAPTER, "-A", ADAPTER, "-o", "o1.fq", "-p", "o2.fq", "r1.fq", "r2.fq"]
    assert evaluate(one_each, tmp_path)[1].verdict == "pass"


def test_adapters_in_a_fasta_file_are_read(tmp_path):
    amplicons(tmp_path / "r.fq")
    (tmp_path / "ad.fasta").write_text(f">fwd\n{PRIMER}\n>rev\n{ADAPTER}\n")
    call = ["-a", "file:ad.fasta", "-o", "out.fq", "r.fq"]
    assert evaluate(call, tmp_path)[1].verdict == "fail"
    assert evaluate([*call, "-n", "2"], tmp_path)[1].verdict == "pass"


def test_the_message_has_the_fix_first_and_the_instance_facts(tmp_path):
    amplicons(tmp_path / "r.fq")
    decision, result = evaluate(["-g", PRIMER, "-a", ADAPTER, "-o", "out.fq", "r.fq"], tmp_path)
    text = render_block(decision)
    assert text.startswith("FIX: add -n 2 (remove every listed adapter from each read)")
    assert result.code == "one_adapter_per_read"
    assert "lists 2 adapters for R1 but removes only 1 per read" in result.message
    assert "12 of the 12 reads sampled from r.fq" in result.message


def test_declarations_of_the_adapters_per_read(tmp_path):
    amplicons(tmp_path / "r.fq")
    call = ["-g", PRIMER, "-a", ADAPTER, "-o", "out.fq", "r.fq"]
    assert evaluate(call, tmp_path, {KEY: "one"})[1].verdict == "pass"
    assert evaluate(call, tmp_path, {KEY: "all"})[1].verdict == "fail"
    assert evaluate([*call[:4], "-n", "2", *call[4:]], tmp_path, {KEY: "all"})[1].verdict == "pass"
    assert evaluate(call, tmp_path, {KEY: "x"})[0].action == "block"


# ---- cutadapt facts --------------------------------------------------------------------------


def insert_lengths(tools, work, *options):
    argv = ["cutadapt", "-g", PRIMER, "-a", ADAPTER, *options, "-o", "out.fq", "r.fq"]
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and proc.stderr == ""
    return [len(line) for line in (work / "out.fq").read_text().splitlines()[1::4]]


def test_cutadapt_removes_one_adapter_per_read_unless_n_or_a_linked_adapter_says_more(
    tools, tmp_path
):
    rng = random.Random(2)
    inserts = [rand_dna(rng, rng.randint(20, 35)) for _ in range(30)]
    reads = [(f"r{i}", (PRIMER + s + ADAPTER + rand_dna(rng, 60))[:90], phred([36] * 90))
             for i, s in enumerate(inserts)]  # fmt: skip
    (tmp_path / "r.fq").write_text(fastq_text(reads))
    want = [len(s) for s in inserts]
    assert insert_lengths(tools, tmp_path) != want
    assert insert_lengths(tools, tmp_path, "-n", "2") == want
    assert insert_lengths(tools, tmp_path, "--times", "2") == want
    linked = tools.run(
        ["cutadapt", "-g", f"{PRIMER}...{ADAPTER}", "-o", "out.fq", "r.fq"], tmp_path
    )
    assert [len(x) for x in (tmp_path / "out.fq").read_text().splitlines()[1::4]] == want
    assert linked.returncode == 0 and linked.stderr == ""
    for single in (["-g", PRIMER], ["-a", ADAPTER]):
        tools.run(["cutadapt", *single, "-o", "one.fq", "r.fq"], tmp_path)
        lengths = [len(x) for x in (tmp_path / "one.fq").read_text().splitlines()[1::4]]
        assert all(a > b for a, b in zip(lengths, want, strict=True))


def test_the_family_sets_fail_on_the_faulted_twin_only(tools, tmp_path):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F84", 6, twin, work, tools, "implicit")
        call = ["-g", PRIMER, "-a", ADAPTER, "-o", "out.fastq", "reads.fastq"]
        assert evaluate(call, work)[1].verdict == ("pass" if twin == "clean" else "fail")
        assert evaluate([*call[:4], "-n", "2", *call[4:]], work)[1].verdict == "pass"


# ---- admission -------------------------------------------------------------------------------


def test_admission_on_calls_labelled_by_running_cutadapt(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(SEEDS), tools, tmp_path)
    cells = {f"{k[0]}_{k[1]}": [c.n, c.failed, c.unknown] for k, c in stats.cells.items()}
    print("ADMISSION", RULE.id, "seeds", SEEDS, cells, "recall", round(stats.recall() or 0, 3))
    assert stats.admitted() and stats.conditional
    assert stats.right_declared.n >= 200 * SEEDS // 300
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.3
    assert all(c.crashed == 0 and c.unknown == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
