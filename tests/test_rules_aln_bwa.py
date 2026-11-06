"""Rule bwa.mem_min_score (family F30): unit tests, evidence against bwa, admission.

The unit tests write small FASTQ/FASTA files; the family tests use F30 workspaces built by the
benchmark; the evidence test runs the installed bwa 0.7.19 on the family data to tie the rule's
claim (reads under -T are written as unmapped) to the tool; the admission test measures the rule
over ADMISSION_SEEDS seeds with calls labelled by running bwa (bench/admission/cases_aln_bwa.py).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f30_bwa_short import mapped_per_contig
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import aln_bwa  # noqa: F401  (registers the checker)
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_bwa")
(RULE,) = SPEC.rules
KEY = "short_read_policy"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "bwa" not in env.real_bins(["bwa"]):
        pytest.skip("samtools and bwa are not available")
    return env


def run(argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("bwa", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def write_fastq(path: Path, lengths: list[int], gz: bool = False) -> Path:
    text = "".join(f"@r{i}\n{'A' * n}\n+\n{'I' * n}\n" for i, n in enumerate(lengths))
    if gz:
        with gzip.open(path, "wb") as handle:
            handle.write(text.encode())
    else:
        path.write_text(text)
    return path


def test_spec_and_tier():
    assert RULE.id == "bwa.mem_min_score" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["bwa"] and RULE.applies.subcommands == ["mem"]
    assert RULE.declare is not None and RULE.declare.key == KEY and not RULE.declare.required
    assert RULE.declare.values == ["drop_short", "keep_all"]
    assert RULE.check in known_checkers()


def test_reading_the_reads_needs_t1(tmp_path):
    write_fastq(tmp_path / "r.fq", [20, 40])
    call = Call("bwa", ("mem", "ref.fa", "r.fq"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


def test_reads_under_the_default_minimum_score_fail_with_the_lowered_t_as_the_fix(tmp_path):
    write_fastq(tmp_path / "r.fq", [60, 28, 22, 45, 20])
    decision, result = run(["mem", "genome.fa", "r.fq"], tmp_path)
    assert result.verdict == "fail" and result.code == "reads_below_min_score"
    assert decision.action == "block"
    assert result.fix[0].startswith("add -T 20 ")
    assert "3 of 5" in result.message and "(default)" in result.message
    assert render_block(decision).startswith("FIX: add -T 20")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["mem", "g.fa", "r.fq"], "fail"),
        (["mem", "-t", "4", "g.fa", "r.fq"], "fail"),
        (["mem", "-T", "30", "g.fa", "r.fq"], "fail"),
        (["mem", "-T30", "g.fa", "r.fq"], "fail"),
        (["mem", "-T", "25", "g.fa", "r.fq"], "fail"),
        (["mem", "-T", "20", "g.fa", "r.fq"], "pass"),
        (["mem", "-T", "15", "-t", "2", "g.fa", "r.fq"], "pass"),
        (["mem", "-T15", "g.fa", "r.fq"], "pass"),
        (["mem", "-A", "2", "g.fa", "r.fq"], "pass"),  # 20 bases x 2 = 40 >= 30
        (["mem", "-A", "2", "-T", "50", "g.fa", "r.fq"], "fail"),
        (["mem", "-R", "@RG\\tID:a", "-M", "-T", "10", "g.fa", "r.fq"], "pass"),
        (["mem", "-O", "6,6", "-E", "1,1", "g.fa", "r.fq"], "fail"),
        (["mem", "-p", "g.fa", "r.fq"], "fail"),
        (["mem", "-T", "40", "-T", "15", "g.fa", "r.fq"], "pass"),  # the last -T wins
    ],
)
def test_options_decide_the_minimum_score(tmp_path, argv, expected):
    write_fastq(tmp_path / "r.fq", [20, 35, 60])
    assert run(argv, tmp_path)[1].verdict == expected


def test_no_short_read_no_missing_files_and_other_subcommands_are_silent(tmp_path):
    write_fastq(tmp_path / "long.fq", [31, 50, 100])
    assert run(["mem", "g.fa", "long.fq"], tmp_path)[1].verdict == "pass"
    assert run(["mem", "g.fa", "absent.fq"], tmp_path)[1].verdict == "pass"
    assert run(["mem", "g.fa"], tmp_path)[1].verdict == "pass"
    assert run(["mem", "g.fa", "-"], tmp_path)[1].verdict == "pass"
    write_fastq(tmp_path / "short.fq", [20])
    assert run(["aln", "g.fa", "short.fq"], tmp_path)[1] is None  # bwa aln has no -T filter
    assert run(["index", "short.fq"], tmp_path)[1] is None


def test_paired_files_gzip_and_fasta_are_read(tmp_path):
    write_fastq(tmp_path / "a_1.fq", [50, 50])
    write_fastq(tmp_path / "a_2.fq.gz", [50, 24], gz=True)
    assert run(["mem", "g.fa", "a_1.fq", "a_2.fq.gz"], tmp_path)[1].verdict == "fail"
    (tmp_path / "r.fa").write_text(">x\nACGTACGTAC\nACGTACGTAC\n>y\n" + "ACGT" * 20 + "\n")
    result = run(["mem", "g.fa", "r.fa"], tmp_path)[1]
    assert result.verdict == "fail" and "1 of 2" in result.message  # the 20-base read
    (tmp_path / "ok.fa").write_text(">y\n" + "ACGT" * 20 + "\n")
    assert run(["mem", "g.fa", "ok.fa"], tmp_path)[1].verdict == "pass"


def test_reads_shorter_than_the_seed_length_point_to_bwa_aln(tmp_path):
    write_fastq(tmp_path / "r.fq", [15, 40])
    result = run(["mem", "g.fa", "r.fq"], tmp_path)[1]
    assert "bwa aln" in result.message


def test_declarations(tmp_path):
    write_fastq(tmp_path / "r.fq", [20, 40])
    argv = ["mem", "g.fa", "r.fq"]
    assert run(argv, tmp_path, {KEY: "drop_short"})[1].verdict == "pass"
    assert run(argv, tmp_path, {KEY: "keep_all"})[1].verdict == "fail"
    assert "you declared" in run(argv, tmp_path, {KEY: "keep_all"})[1].message
    assert (
        run(["mem", "-T", "15", "g.fa", "r.fq"], tmp_path, {KEY: "keep_all"})[1].verdict == "pass"
    )
    assert (
        run(["mem", "-T", "15", "g.fa", "r.fq"], tmp_path, {KEY: "drop_short"})[1].verdict == "pass"
    )
    decision, result = run(argv, tmp_path, {KEY: "oops"})
    assert result.code == "declaration_invalid" and decision.action == "block"


def test_an_unreadable_gzip_is_unknown_when_nothing_else_decides(tmp_path):
    (tmp_path / "bad.fq.gz").write_bytes(b"\x1f\x8b\x08\x00broken")
    decision, result = run(["mem", "g.fa", "bad.fq.gz"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_the_sample_is_bounded(tmp_path):
    write_fastq(tmp_path / "r.fq", [60] * 5100 + [20])  # the short read is past the 5000 sampled
    assert run(["mem", "g.fa", "r.fq"], tmp_path)[1].verdict == "pass"


# ---- F30 workspaces and the tool -------------------------------------------------------------


@pytest.fixture(scope="module")
def f30(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f30")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F30", 8, twin, work, tools, "implicit"), work)
    return out


def test_family_calls_fail_on_the_faulted_twin_and_pass_on_the_clean_one(f30):
    naive = ["mem", "-t", "1", "genome.fa", "reads.fastq"]
    safe = ["mem", "-t", "1", "-T", "15", "genome.fa", "reads.fastq"]
    _, clean = f30["clean"]
    _, faulted = f30["faulted"]
    assert run(naive, clean)[1].verdict == "pass"
    assert run(safe, clean)[1].verdict == "pass"
    assert run(naive, faulted)[1].verdict == "fail"
    assert run(safe, faulted)[1].verdict == "pass"
    assert (
        run(["mem", "-T", "15", "genome.fa", "reads.fastq"], faulted, {KEY: "keep_all"})[1].verdict
        == "pass"
    )


def test_bwa_mem_leaves_reads_under_the_minimum_score_unmapped_and_t_15_maps_them(f30, tools):
    task, work = f30["faulted"]
    contigs = list(task.truth["expected"])
    default = tools.run(["bwa", "mem", "-t", "1", "genome.fa", "reads.fastq"], work)
    lowered = tools.run(["bwa", "mem", "-t", "1", "-T", "15", "genome.fa", "reads.fastq"], work)
    assert default.returncode == 0 == lowered.returncode
    assert "[main] CMD: bwa mem" in default.stderr and "error" not in default.stderr.lower()
    assert not any("short" in x.lower() or "skip" in x.lower() for x in default.stderr.splitlines())
    assert mapped_per_contig(default.stdout, contigs) == task.truth["naive"]
    assert mapped_per_contig(lowered.stdout, contigs) == task.truth["expected"]
    short = [
        line.split("\t")
        for line in default.stdout.splitlines()
        if not line.startswith("@") and len(line.split("\t")[9]) < 30
    ]
    assert short and all(int(f[1]) & 4 for f in short)  # every read under 30 bases is flag 4


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.admitted() and stats.conditional
    assert stats.cells[("right", True)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", True)].failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # every undeclared default-threshold call on the faulted twin is caught: 10 templates per seed
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8
