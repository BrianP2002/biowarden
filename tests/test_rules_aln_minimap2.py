"""Rule minimap2.short_reads_preset (family F35): unit tests, evidence against minimap2, admission.

The unit tests write small FASTQ files; the family tests use F35 workspaces; the evidence test runs
minimap2 2.31 on the family data to tie the module docstring's table (reads under about 60 bases are
lost without -x sr) to the tool; the admission test measures the rule over ADMISSION_SEEDS seeds
with calls labelled by running minimap2 (bench/admission/cases_aln_minimap2.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f35_mm2_short import mapped_per_contig
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.rules import aln_minimap2 as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_minimap2")
(RULE,) = SPEC.rules
KEY = "short_read_policy"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "minimap2" not in env.real_bins(["minimap2"]):
        pytest.skip("samtools and minimap2 are not available")
    return env


def run(argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("minimap2", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def write_fastq(path: Path, lengths: list[int]) -> Path:
    path.write_text("".join(f"@r{i}\n{'A' * n}\n+\n{'I' * n}\n" for i, n in enumerate(lengths)))
    return path


def test_spec_is_a_conditional_t1_call_rule():
    assert RULE.id == "minimap2.short_reads_preset" and (RULE.kind, RULE.tier) == ("call", "T1")
    assert RULE.applies.tools == ["minimap2"] and RULE.declare.key == KEY
    assert not RULE.declare.required and RULE.declare.values == ["drop_short", "keep_all"]
    assert RULE.check in known_checkers()


def test_reading_the_reads_needs_t1(tmp_path):
    write_fastq(tmp_path / "r.fq", [40])
    call = Call("minimap2", ("-a", "ref.fa", "r.fq"), tmp_path)
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, Match(RULE), Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-a", "ref.fa", "r.fq"], "fail"),
        (["-ax", "map-ont", "ref.fa", "r.fq"], "fail"),
        (["-a", "-x", "map-ont", "ref.fa", "r.fq"], "fail"),
        (["-a", "-x", "splice", "ref.fa", "r.fq"], "fail"),
        (["-a", "-t", "4", "ref.fa", "r.fq"], "fail"),
        (["-ax", "sr", "ref.fa", "r.fq"], "pass"),
        (["-a", "-x", "sr", "ref.fa", "r.fq"], "pass"),
        (["-a", "-xsr", "ref.fa", "r.fq"], "pass"),
        (["-a", "--preset", "sr", "ref.fa", "r.fq"], "pass"),
        (["-a", "-k", "12", "-w", "5", "ref.fa", "r.fq"], "pass"),  # the caller tunes the seeds
        (["-a", "-s", "40", "ref.fa", "r.fq"], "pass"),
        (["-a", "-o", "out.sam", "ref.fa", "r.fq"], "fail"),
        (["-a", "-R", "@RG\\tID:a", "ref.fa", "r.fq"], "fail"),
    ],
)
def test_the_preset_decides_for_short_reads(tmp_path, argv, expected):
    write_fastq(tmp_path / "r.fq", [50, 52, 48, 100])
    assert run(argv, tmp_path)[1].verdict == expected


def test_long_reads_missing_files_and_stdin_are_silent(tmp_path):
    write_fastq(tmp_path / "long.fq", [100, 250, 2000, 60])
    assert run(["-a", "ref.fa", "long.fq"], tmp_path)[1].verdict == "pass"
    assert run(["-a", "ref.fa", "absent.fq"], tmp_path)[1].verdict == "pass"
    assert run(["-a", "ref.fa"], tmp_path)[1].verdict == "pass"
    assert run(["-a", "ref.fa", "-"], tmp_path)[1].verdict == "pass"


def test_five_percent_of_short_reads_is_the_threshold(tmp_path):
    write_fastq(tmp_path / "few.fq", [100] * 96 + [50] * 4)
    write_fastq(tmp_path / "some.fq", [100] * 94 + [50] * 6)
    assert run(["-a", "ref.fa", "few.fq"], tmp_path)[1].verdict == "pass"
    assert run(["-a", "ref.fa", "some.fq"], tmp_path)[1].verdict == "fail"


def test_the_fail_message_has_the_fix_first_and_the_numbers(tmp_path):
    write_fastq(tmp_path / "r.fq", [47, 50, 52, 100])
    decision, result = run(["-a", "ref.fa", "r.fq"], tmp_path)
    assert result.code == "short_reads_without_sr_preset" and decision.action == "block"
    assert result.fix[0] == "add -x sr (short-read preset) to the minimap2 call"
    assert "3 of 4" in result.message and "the shortest has 47" in result.message


def test_declarations(tmp_path):
    write_fastq(tmp_path / "r.fq", [50, 50])
    argv = ["-a", "ref.fa", "r.fq"]
    assert run(argv, tmp_path, {KEY: "drop_short"})[1].verdict == "pass"
    assert run(argv, tmp_path, {KEY: "keep_all"})[1].verdict == "fail"
    assert run(["-ax", "sr", "ref.fa", "r.fq"], tmp_path, {KEY: "keep_all"})[1].verdict == "pass"
    assert run(argv, tmp_path, {KEY: "x"})[0].action == "block"


def test_two_read_files_and_an_unreadable_gzip(tmp_path):
    write_fastq(tmp_path / "a_1.fq", [100, 100])
    write_fastq(tmp_path / "a_2.fq", [50, 50])
    assert run(["-a", "-x", "map-ont", "ref.fa", "a_1.fq", "a_2.fq"], tmp_path)[1].verdict == "fail"
    (tmp_path / "bad.fq.gz").write_bytes(b"\x1f\x8b\x08\x00broken")
    decision, result = run(["-a", "ref.fa", "bad.fq.gz"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_parsing_of_options_with_values_keeps_the_positionals(tmp_path):
    write_fastq(tmp_path / "r.fq", [50])
    # -o, -R, -t, -K take values: the reference is the first positional, the reads the next
    opts, pos = rules.parse_options(
        ["-a", "-t", "2", "-K", "500M", "-R", "@RG", "ref.fa", "r.fq"],
        rules.MM2_ARG_SHORT,
        rules.MM2_LONG,
    )
    assert pos == ["ref.fa", "r.fq"] and opts["t"] == ["2"]


# ---- F35 workspaces and the tool -------------------------------------------------------------


@pytest.fixture(scope="module")
def f35(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f35")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F35", 9, twin, work, tools, "implicit"), work)
    return out


def test_family_calls_fail_on_the_faulted_twin_and_pass_on_the_clean_one(f35):
    plain = ["-a", "-t", "1", "genome.fa", "reads.fastq"]
    sr = ["-a", "-x", "sr", "-t", "1", "genome.fa", "reads.fastq"]
    assert run(plain, f35["clean"][1])[1].verdict == "pass"
    assert run(sr, f35["clean"][1])[1].verdict == "pass"
    assert run(plain, f35["faulted"][1])[1].verdict == "fail"
    assert run(sr, f35["faulted"][1])[1].verdict == "pass"


def test_minimap2_without_the_preset_loses_short_reads_silently_and_sr_maps_them(f35, tools):
    task, work = f35["faulted"]
    contigs = list(task.truth["expected"])
    plain = tools.run(["minimap2", "-a", "-t", "1", "genome.fa", "reads.fastq"], work)
    ont = tools.run(
        ["minimap2", "-a", "-x", "map-ont", "-t", "1", "genome.fa", "reads.fastq"], work
    )
    sr = tools.run(["minimap2", "-a", "-x", "sr", "-t", "1", "genome.fa", "reads.fastq"], work)
    assert plain.returncode == ont.returncode == sr.returncode == 0
    assert "warning" not in plain.stderr.lower() and "too short" not in plain.stderr.lower()
    assert mapped_per_contig(plain.stdout, contigs) == task.truth["naive"]
    assert sum(task.truth["naive"].values()) < sum(task.truth["expected"].values())
    assert mapped_per_contig(sr.stdout, contigs) == task.truth["expected"]
    assert mapped_per_contig(ont.stdout, contigs) != task.truth["expected"]
    clean_task, clean = f35["clean"]
    again = tools.run(["minimap2", "-a", "-t", "1", "genome.fa", "reads.fastq"], clean)
    assert mapped_per_contig(again.stdout, contigs) == clean_task.truth["expected"]


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.admitted() and stats.conditional
    assert stats.cells[("right", True)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", True)].failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # every undeclared call without the preset on the faulted twin is caught: 10 templates per seed
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 9
