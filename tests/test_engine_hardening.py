"""Engine hardening: what an agent-chosen path, a huge command, a probe of the socket or a slow dry
run can do to the checking layer (final review E-1, E-4, E-6, E-7, E-8, E-9, E-10, FD-8).

Normal calls must decide as before; everything here is about the odd ones: the layer fails open and
says so in the ledger, never raises into the trial, and never reads or runs what lies outside it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import pytest
from fastmcp import Client

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, CheckResult, Engine, Promotion, Spec, checker
from biowarden.engine.call import has_declare_marker
from biowarden.engine.checks import (
    OUTSIDE,
    Ctx,
    LayerTimeout,
    inside,
    is_regular_file,
)
from biowarden.engine.decide import EVALUATE_SECONDS, tool_names
from biowarden.engine.mcp_server import build_mcp
from biowarden.engine.precheck import MAX_CHECKED_COMMANDS, precheck
from biowarden.engine.render import (
    CRASHED_FIX,
    STOP_TEXT,
    STOP_TEXT_BARE,
    render_block,
    render_warning,
)
from biowarden.engine.server import (
    MAX_LOGGED,
    EngineServer,
    PathMap,
    check_request,
    serve_in_thread,
)
from biowarden.engine.spec import load_packaged, packaged_specs

ROOT = Path(__file__).resolve().parents[1]
ROOT_USER = os.geteuid() == 0


@contextlib.contextmanager
def watchdog(seconds: float = 8.0):
    """Fail the test instead of hanging the suite when a read blocks (a FIFO) or never ends."""

    def fire(*_: object) -> None:
        raise TimeoutError("hung")

    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# -- rules and checkers of the tests ---------------------------------------------------------------


def spec_of(*rules: dict) -> Spec:
    return Spec.model_validate({"name": "hardening", "rules": list(rules)})


def fact(check: str, tools: list[str] | None = None, tier: str = "T1", suffix: str = "x") -> dict:
    return {
        "id": f"hardening.{suffix}",
        "slot": "units",
        "kind": "fact",
        "tier": tier,
        "applies": {"tools": tools or [], "file_patterns": [r"(?<![\w./-])[\w./-]+\.dat"]},
        "check": check,
        "message": "m",
    }


RISKY = {
    "id": "hardening.risky",
    "slot": "units",
    "kind": "call",
    "tier": "T0",
    "applies": {"tools": ["faketool"], "options_any": ["--risky"]},
    "declare": {"key": "units", "values": ["a", "b"]},
    "check": "option_implies_declared",
    "params": {"option_map": {"--risky": "a"}, "fix_option": {}},
    "message": "faketool --risky means units a",
}


@checker("hardening_flags")
def _flags(call, match, ctx):
    return CheckResult("fail", "risky", "the data is risky")


@checker("hardening_crashes")
def _crashes(call, match, ctx):
    raise RuntimeError("boom")


@checker("hardening_sleeps")
def _sleeps(call, match, ctx):
    time.sleep(0.4)
    return CheckResult("fail", "slow", "slow")


@checker("hardening_spins")
def _spins(call, match, ctx):
    while True:  # a checker that never ends, but asks the context for the time
        time.sleep(0.01)
        ctx.check_time()


@checker("hardening_times_out")
def _times_out(call, match, ctx):
    raise subprocess.TimeoutExpired("a dry run", 30)


@checker("hardening_denied")
def _denied(call, match, ctx):
    raise PermissionError(13, "Permission denied", "/root/x.dat")


@checker("hardening_too_long")
def _too_long(call, match, ctx):
    raise OSError(36, "File name too long", "a" * 300)


@checker("hardening_missing")
def _missing(call, match, ctx):
    raise FileNotFoundError(2, "No such file or directory", "x.dat")


@checker("hardening_reads")
def _reads(call, match, ctx):
    return CheckResult("fail", "read", next(ctx.lines(match.files[0]), "<empty>"))


def verdicts(decision) -> list[tuple[str, str, str]]:
    return [(o.rule.id, o.result.code, o.effective) for o in decision.outcomes]


def faults(result) -> list[str]:
    """Rules of a pre-check that crashed or could not be checked."""
    codes = ("checker_crashed", "layer_error")
    return [o.rule.id for d in result.decisions for o in d.outcomes if o.result.code in codes]


# -- a path the host cannot stat, or should not (E-1, E-4) -----------------------------------------


@pytest.mark.skipif(ROOT_USER, reason="root can search every directory")
def test_a_call_naming_a_file_in_a_directory_the_host_cannot_search_is_evaluated(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "x.dat").write_text("x\n")
    locked.chmod(0)
    engine = Engine([spec_of(fact("hardening_flags"))], mode="block")
    try:
        decision = engine.evaluate(Call("awk", (f"{locked}/x.dat",), tmp_path))
    finally:
        locked.chmod(0o700)

    assert decision.action == "allow" and decision.outcomes == ()


@pytest.mark.parametrize(
    "name",
    ["/root/x.dat", "a" * 300 + ".dat", "bad\x00name.dat", "../" * 40 + "etc/x.dat"],
    ids=["root-home", "300-characters", "nul", "far-above"],
)
def test_the_precheck_survives_file_names_the_host_stat_would_raise_on(tmp_path, name):
    engine = Engine(packaged_specs() + [spec_of(fact("hardening_flags"))], mode="block")
    command = f"python3 -c \"open('{name}')\""

    result = precheck(command, tmp_path, engine)

    assert result.action in ("allow", "warn") and faults(result) == []  # a plain miss, no fault


def test_the_real_packaged_rules_survive_every_odd_name_on_one_command(tmp_path):
    engine = Engine(packaged_specs(), mode="block")
    odd = " ".join(
        f"/root/x{ext} {'a' * 300}{ext}" for ext in (".tsv", ".vcf", ".bam", ".fq", ".cif")
    )

    result = precheck(f"cat {odd} | python3 -c 'print(1)'", tmp_path, engine)

    assert result.action in ("allow", "warn") and faults(result) == []


def test_the_precheck_survives_special_files_named_like_data(tmp_path):
    os.mkfifo(tmp_path / "pipe.tsv")
    os.mkfifo(tmp_path / "reads.fastq")
    (tmp_path / "zero.vcf").symlink_to("/dev/zero")
    (tmp_path / "huge.fastq").symlink_to("/dev/zero")
    (tmp_path / "loop.tsv").symlink_to("loop.tsv")
    engine = Engine(packaged_specs(), mode="block", real_bins={}, retry_cap=10**9)
    commands = [
        "cat pipe.tsv | awk '{print $1}'",
        "bcftools view zero.vcf | head",
        "python3 -c \"print(open('huge.fastq').read(10))\"",
        "awk 'NR%4==0' reads.fastq | wc -l",
        "cut -f1 loop.tsv",
        "samtools view -H /dev/zero",
    ]

    for command in commands:
        with watchdog(20):
            result = precheck(command, tmp_path, engine)
        assert result.action in ("allow", "warn", "block"), command
        assert faults(result) == [], command


def test_a_fault_inside_a_rule_costs_that_rule_only(tmp_path, monkeypatch):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_flags"))], mode="block")

    def broken(path, root=None):
        raise RuntimeError("the host fell over")

    monkeypatch.setattr("biowarden.engine.decide.is_regular_file", broken)
    decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))

    assert verdicts(decision) == [("hardening.x", "layer_error", "warn")]
    assert decision.action == "warn"  # unknown never blocks, even in block mode
    (outcome,) = decision.outcomes
    assert outcome.result.verdict == "unknown" and "RuntimeError" in outcome.result.note
    assert render_warning(decision) == ""  # the agent is told nothing, the ledger has the cause
    assert render_warning(decision, style="generic") == ""
    quiet = precheck("awk 1 a.dat", tmp_path, engine)
    assert quiet.action == "allow" and quiet.text == "" and faults(quiet) == ["hardening.x"]


def test_a_fault_leaves_the_other_rules_their_verdicts(tmp_path, monkeypatch):
    (tmp_path / "a.dat").write_text("x\n")
    other = {**RISKY, "id": "hardening.other"}
    engine = Engine([spec_of(fact("hardening_flags"), other)], mode="block")
    monkeypatch.setattr("biowarden.engine.decide.is_regular_file", lambda *a: 1 / 0)

    decision = engine.evaluate(Call("faketool", ("--risky", "a.dat"), tmp_path))

    assert verdicts(decision) == [
        ("hardening.x", "layer_error", "warn"),
        ("hardening.other", "declaration_missing", "block"),
    ]


def test_a_crashing_checker_still_fails_closed_but_a_timeout_does_not(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    run = Call("awk", ("a.dat",), tmp_path)

    crash = Engine([spec_of(fact("hardening_crashes"))], mode="block").evaluate(run)
    with watchdog():
        slow = Engine([spec_of(fact("hardening_spins"))], mode="block", max_seconds=0.2).evaluate(
            run
        )

    assert verdicts(crash) == [("hardening.x", "checker_crashed", "block")]
    assert verdicts(slow) == [("hardening.x", "layer_error", "warn")]
    assert slow.action == "warn"


@pytest.mark.parametrize(
    "check,code,effective",
    [
        ("hardening_denied", "layer_error", "warn"),
        ("hardening_too_long", "layer_error", "warn"),
        (
            "hardening_missing",
            "checker_crashed",
            "block",
        ),  # a rule that forgot a case still fails closed
    ],
)
def test_a_rule_that_stats_a_name_of_the_call_itself_and_is_denied_is_a_layer_error(
    tmp_path, check, code, effective
):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact(check))], mode="block")

    decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))

    assert verdicts(decision) == [("hardening.x", code, effective)]


@pytest.mark.skipif(ROOT_USER, reason="root can search every directory")
def test_a_command_naming_a_real_and_an_unreadable_single_cell_file_is_not_blocked(tmp_path):
    work = tmp_path / "work"
    locked = tmp_path / "locked"
    work.mkdir()
    locked.mkdir()
    (work / "real.h5ad").write_bytes(b"\x89HDF\r\n\x1a\n" + b"0" * 64)
    locked.chmod(0)
    engine = Engine(packaged_specs(), mode="block")
    command = (
        "python3 -c \"import scanpy as sc; a = sc.read_h5ad('real.h5ad'); "
        f"b = sc.read_h5ad('{locked}/x.h5ad')\""
    )
    try:
        result = precheck(command, Path("/work"), engine, PathMap(work.resolve()))
    finally:
        locked.chmod(0o700)

    codes = {o.result.code for d in result.decisions for o in d.outcomes}
    assert result.action != "block" and "checker_crashed" not in codes


def test_a_dry_run_that_timed_out_inside_a_checker_is_a_layer_error_not_a_crash(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_times_out"))], mode="block")

    decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))

    assert verdicts(decision) == [("hardening.x", "layer_error", "warn")]
    assert "TimeoutExpired" in decision.outcomes[0].result.note


# -- time and size (E-7) ---------------------------------------------------------------------------


def test_the_watchdog_stops_an_evaluation_that_runs_out_of_time(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    rules = [fact("hardening_sleeps", suffix=str(i)) for i in range(5)]
    engine = Engine([spec_of(*rules)], mode="block", max_seconds=0.5)
    started = time.monotonic()

    decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))

    assert time.monotonic() - started < 1.5  # two rules ran, the other three were skipped
    codes = [(o.rule.id, o.result.code) for o in decision.outcomes]
    assert codes[:2] == [("hardening.0", "slow"), ("hardening.1", "slow")]
    assert codes[-1][1] == "layer_error" and len(codes) == 3
    assert "3 rules not run" in decision.outcomes[-1].result.note
    assert engine.max_seconds == 0.5 and EVALUATE_SECONDS == 20.0


def test_a_checker_that_asks_for_the_time_is_stopped_inside_a_single_rule(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_spins"))], mode="promoted", max_seconds=0.3)
    started = time.monotonic()

    with watchdog():
        decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))

    assert time.monotonic() - started < 2.0
    assert verdicts(decision) == [("hardening.x", "layer_error", "warn")]


def test_only_the_first_200_kb_of_a_call_is_searched_for_file_names(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_reads"))], mode="block")
    filler = "x" * 250_000

    early = engine.evaluate(Call("awk", (f"a.dat {filler}",), tmp_path))
    late = engine.evaluate(Call("awk", (f"{filler} a.dat",), tmp_path))
    in_context = engine.evaluate(Call("awk", ("1",), tmp_path, {}, (filler, "a.dat")))

    assert early.action == "block" and late.outcomes == () and in_context.outcomes == ()


def test_a_call_that_stays_under_the_scan_limit_is_matched_in_argv_and_context(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_reads"))], mode="block")

    by_argv = engine.evaluate(Call("awk", ("1", "a.dat"), tmp_path))
    by_context = engine.evaluate(Call("awk", ("1",), tmp_path, {}, ("cat", "a.dat")))

    assert by_argv.action == by_context.action == "block"


def test_a_string_of_hundreds_of_commands_is_checked_up_to_a_cap_and_then_warns(tmp_path):
    engine = Engine(packaged_specs(), mode="warn")
    command = "; ".join(["echo a"] * (MAX_CHECKED_COMMANDS + 100))
    started = time.monotonic()

    result = precheck(command, tmp_path, engine)

    assert len(result.decisions) == MAX_CHECKED_COMMANDS
    assert result.action == "warn" and "too many commands" in result.text
    assert str(MAX_CHECKED_COMMANDS) in result.text
    assert time.monotonic() - started < 10


def test_a_string_at_the_cap_gets_no_warning(tmp_path):
    engine = Engine(packaged_specs(), mode="warn")
    command = "; ".join(["echo a"] * MAX_CHECKED_COMMANDS)

    result = precheck(command, tmp_path, engine)

    assert result.action == "allow" and len(result.decisions) == MAX_CHECKED_COMMANDS


# -- versioned interpreters and interpreters by path (FD-8) ----------------------------------------


@pytest.mark.parametrize(
    "tool,names",
    [
        ("python3", ("python3",)),
        ("python3.10", ("python3.10", "python3")),
        ("/projects/x/envs/bio-struct/bin/python3.10", ("python3.10", "python3")),
        ("python3.12", ("python3.12", "python3")),
        ("python2.7", ("python2.7", "python2")),
        ("python", ("python",)),
        ("python3.10-config", ("python3.10-config",)),
        ("python-struct", ("python-struct",)),
        ("bcftools", ("bcftools",)),
    ],
)
def test_tool_names_adds_the_plain_name_of_a_versioned_interpreter(tool, names):
    assert tool_names(tool) == names


@pytest.mark.parametrize(
    "tool",
    ["python3", "python3.10", "python3.12", "/projects/x/envs/bio-struct/bin/python3.10"],
)
def test_a_versioned_or_path_called_interpreter_meets_the_same_fact_rule(tmp_path, tool):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_flags", ["python3", "python"]))], mode="block")

    decision = engine.evaluate(Call(tool, ("-c", "open('a.dat')"), tmp_path))

    assert verdicts(decision) == [("hardening.x", "risky", "block")]


def test_an_interpreter_the_rule_does_not_list_stays_unmatched(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_flags", ["python3", "python"]))], mode="block")

    for tool in ("python2.7", "ruby", "python3x", "mypython3.10"):
        assert engine.evaluate(Call(tool, ("a.dat",), tmp_path)).outcomes == (), tool


def test_the_structure_rules_of_the_packaged_specs_apply_to_python3_10(tmp_path):
    (tmp_path / "c.cif").write_text("data_x\n#\n")
    engine = Engine([load_packaged("struct_prody")], mode="block")
    code = ("-c", "import prody; prody.parseMMCIF('c.cif')")

    seen = {
        t: verdicts(engine.evaluate(Call(t, code, tmp_path))) for t in ("python3", "python3.10")
    }
    path = precheck(f'/projects/p/envs/bio-struct/bin/python3.10 -c "{code[1]}"', tmp_path, engine)

    assert seen["python3"] and seen["python3.10"] == seen["python3"]
    assert {o.rule.id for d in path.decisions for o in d.outcomes} >= {"prody.chain_scheme"}


# -- Ctx: what a checker may open (E-4, E-8) -------------------------------------------------------


def test_resolve_leads_outside_the_trial_to_a_path_that_does_not_exist(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "in.dat").write_text("x\n")
    (tmp_path / "secret.dat").write_text("secret\n")
    (work / "out.dat").symlink_to("../secret.dat")
    (work / "up").symlink_to(tmp_path)
    ctx = Ctx(work, "T1", root=work.resolve())

    assert ctx.resolve("in.dat") == work / "in.dat"
    assert ctx.resolve(str(work / "in.dat")) == work / "in.dat"
    assert ctx.resolve("missing.dat") == work / "missing.dat"  # absent, but inside
    for name in ("/etc/passwd", "../secret.dat", "out.dat", "up/secret.dat", "a/../../secret.dat"):
        outside = ctx.resolve(name)
        assert outside.parent == OUTSIDE and not outside.exists(), name
        assert outside.name == Path(name).name
    assert Ctx(work, "T1").resolve("../secret.dat") == work / "../secret.dat"  # no root: as before


def test_a_checker_reading_a_name_that_leads_outside_sees_a_missing_file(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.dat").write_text("secret\n")
    ctx = Ctx(work, "T1", root=work.resolve())

    with pytest.raises(FileNotFoundError):
        list(ctx.lines("../secret.dat"))
    with pytest.raises(FileNotFoundError):
        ctx.read_bytes("/etc/passwd")
    assert list(Ctx(work, "T1").lines("../secret.dat")) == ["secret"]


def test_inside_follows_links_and_dotdot_and_needs_no_file(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "to-parent").symlink_to("..")

    assert inside(work / "a" / "b.dat", work)
    assert inside(work, work)
    assert not inside(tmp_path, work)
    assert not inside(work / "to-parent" / "x", work)
    assert not inside(work / ".." / "x", work)
    assert not inside("/etc/passwd", work)
    assert not inside(work / "bad\x00name", work)
    assert inside(work / ("a" * 5000), work)  # too long to exist, but it is not an error


def test_is_regular_file_is_false_instead_of_raising(tmp_path):
    (tmp_path / "f.dat").write_text("x\n")
    (tmp_path / "d.dat").mkdir()
    os.mkfifo(tmp_path / "p.dat")
    (tmp_path / "zero.dat").symlink_to("/dev/zero")
    (tmp_path / "loop.dat").symlink_to("loop.dat")
    (tmp_path / "gone.dat").symlink_to("nowhere.dat")

    assert is_regular_file(tmp_path / "f.dat")
    for name in ("d.dat", "p.dat", "zero.dat", "loop.dat", "gone.dat", "missing.dat"):
        assert is_regular_file(tmp_path / name) is False, name
    assert not is_regular_file(tmp_path / ("a" * 300))
    assert not is_regular_file(Path("/proc/1/cmdline/x"))
    assert not is_regular_file(Path("bad\x00name"))
    assert is_regular_file(tmp_path / "f.dat", tmp_path)
    assert not is_regular_file(tmp_path / "f.dat", tmp_path / "d.dat")


def test_the_precheck_with_a_path_map_does_not_follow_a_link_out_of_the_trial(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.dat").write_text("secret\n")
    (work / "link.dat").symlink_to("../secret.dat")
    (work / "own.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_reads"))], mode="block")
    paths = PathMap(work.resolve())

    confined = precheck("awk 1 link.dat", Path("/work"), engine, paths)
    own = precheck("awk 1 own.dat", Path("/work"), engine, paths)
    unmapped = precheck("awk 1 link.dat", work.resolve(), engine)

    assert confined.action == "allow" and own.action == "block" and unmapped.action == "block"


def test_a_symlink_that_leaves_the_trial_is_not_a_match(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.dat").write_text("secret\n")
    (work / "link.dat").symlink_to("../secret.dat")
    engine = Engine([spec_of(fact("hardening_reads"))], mode="block")

    confined = engine.evaluate(Call("awk", ("link.dat",), work.resolve(), root=work.resolve()))
    loose = engine.evaluate(Call("awk", ("link.dat",), work.resolve()))

    assert confined.outcomes == () and loose.action == "block"


def test_an_absolute_path_of_the_sandbox_is_no_match_for_a_confined_call(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "other.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_reads"))], mode="block")
    command = ("-c", f"open('{tmp_path}/other.dat')")

    confined = engine.evaluate(Call("python3", command, work.resolve(), root=work.resolve()))
    loose = engine.evaluate(Call("python3", command, work.resolve()))

    assert confined.outcomes == () and loose.action == "block"


def test_lines_of_a_file_without_newlines_are_bounded_by_max_bytes(tmp_path):
    (tmp_path / "huge.dat").write_bytes(b"x" * 20_000_000)
    ctx = Ctx(tmp_path, "T1", max_bytes=1000)
    tracemalloc.start()
    try:
        got = list(ctx.lines("huge.dat"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert got == [] and peak < 2_000_000


def test_lines_keeps_its_meaning_for_ordinary_files(tmp_path):
    import gzip

    (tmp_path / "a.dat").write_text("one\ntwo\r\nthree\nfour")
    with gzip.open(tmp_path / "b.dat", "wt") as handle:
        handle.write("g1\ng2\n")
    (tmp_path / "empty.dat").write_text("")
    ctx = Ctx(tmp_path, "T1")

    assert list(ctx.lines("a.dat")) == ["one", "two", "three", "four"]
    assert list(ctx.lines("a.dat", limit=2)) == ["one", "two"]
    assert list(ctx.lines("a.dat", limit=0)) == []
    assert list(ctx.lines("b.dat")) == ["g1", "g2"]
    assert list(ctx.lines("empty.dat")) == []
    # the budget counts the decoded text, newline included: "one\n" and "two\n" fit in 9
    assert list(Ctx(tmp_path, "T1", max_bytes=9).lines("a.dat")) == ["one", "two"]
    assert list(Ctx(tmp_path, "T1", max_bytes=8).lines("a.dat")) == ["one", "two"]
    assert list(Ctx(tmp_path, "T1", max_bytes=7).lines("a.dat")) == ["one"]


@pytest.mark.parametrize("kind", ["fifo", "zero", "directory", "null"])
def test_a_checker_cannot_be_hung_or_flooded_by_a_special_file(tmp_path, kind):
    path = tmp_path / "x.dat"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "zero":
        path.symlink_to("/dev/zero")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to("/dev/null")
    ctx = Ctx(tmp_path, "T1")
    # a directory raises what open() always raised; anything else is "not there"
    error = IsADirectoryError if kind == "directory" else FileNotFoundError

    with watchdog(), pytest.raises(error):
        list(ctx.lines("x.dat"))
    with watchdog(), pytest.raises(error):
        ctx.read_bytes("x.dat", 10)


def test_a_t0_checker_still_may_not_read_and_a_t1_checker_sees_the_time(tmp_path):
    (tmp_path / "a.dat").write_text("x\n" * 10)

    with pytest.raises(LayerTimeout):
        list(Ctx(tmp_path, "T1", deadline=time.monotonic() - 1).lines("a.dat"))
    with pytest.raises(LayerTimeout):
        Ctx(tmp_path, "T1", deadline=time.monotonic() - 1).read_bytes("a.dat")
    assert list(Ctx(tmp_path, "T1", deadline=time.monotonic() + 60).lines("a.dat", 2)) == ["x", "x"]
    assert issubclass(LayerTimeout, OSError)  # a rule that catches OSError reads it as unreadable


# -- run_real: memory, time, stdin, process group (E-8) --------------------------------------------


def script(path: Path, body: str, shebang: str = "/bin/sh") -> Path:
    path.write_text(f"#!{shebang}\n{body}")
    path.chmod(0o755)
    return path


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


@pytest.mark.skipif(not hasattr(resource, "prlimit"), reason="needs resource.prlimit (Linux)")
def test_a_dry_run_has_a_limit_on_its_address_space(tmp_path):
    hog = script(
        tmp_path / "hog",
        "import time\ntime.sleep(0.3)\n"
        "try:\n    bytearray(1 << 30)\n"
        "except MemoryError:\n    print('limited')\n    raise SystemExit(3)\n"
        "print('allocated')\n",
        sys.executable,
    )
    ctx = Ctx(tmp_path, "T1", {"hog": hog}, memory_limit=256 * 1024 * 1024)

    proc = ctx.run_real("hog", [])

    assert (proc.returncode, proc.stdout.strip()) == (3, "limited")


def test_an_ordinary_dry_run_returns_what_subprocess_run_returned(tmp_path):
    tool = script(tmp_path / "tool", 'echo "out $1"; echo "err" >&2; exit 4\n')
    ctx = Ctx(tmp_path, "T1", {"tool": tool})

    proc = ctx.run_real("tool", ["x"])

    assert (proc.returncode, proc.stdout, proc.stderr) == (4, "out x\n", "err\n")
    assert proc.args == [str(tool), "x"]
    with pytest.raises(FileNotFoundError):
        ctx.run_real("unknown", [])
    script(tmp_path / "noexec", "true\n").chmod(0o644)
    with pytest.raises(PermissionError):
        Ctx(tmp_path, "T1", {"n": tmp_path / "noexec"}).run_real("n", [])
    with pytest.raises(FileNotFoundError):
        Ctx(tmp_path, "T1", {"n": tmp_path / "nowhere"}).run_real("n", [])


def test_a_dry_run_that_prints_binary_garbage_does_not_crash_the_checker(tmp_path):
    tool = script(tmp_path / "tool", "printf '\\377\\376ok'\n")

    proc = Ctx(tmp_path, "T1", {"tool": tool}).run_real("tool", [])

    assert proc.stdout.endswith("ok")


def test_a_dry_run_that_outlives_its_time_is_killed_with_everything_it_started(tmp_path):
    pidfile = tmp_path / "child.pid"
    tool = script(tmp_path / "tool", f"sleep 30 &\necho $! > {pidfile}\nwait\n")
    ctx = Ctx(tmp_path, "T1", {"tool": tool}, run_timeout=0.5)
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        ctx.run_real("tool", [])

    assert time.monotonic() - started < 5
    child = int(pidfile.read_text())
    for _ in range(40):
        if not alive(child):
            break
        time.sleep(0.05)
    assert not alive(child)


def test_a_dry_run_starts_no_later_than_the_evaluation_allows(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    tool = script(tmp_path / "tool", "true\n")

    with pytest.raises(subprocess.TimeoutExpired):
        Ctx(tmp_path, "T1", {"tool": tool}, deadline=time.monotonic() - 1).run_real("tool", [])

    assert started == []


def test_the_time_left_of_the_evaluation_cuts_a_longer_run_timeout(tmp_path):
    tool = script(tmp_path / "tool", "sleep 30\n")
    ctx = Ctx(tmp_path, "T1", {"tool": tool}, run_timeout=60, deadline=time.monotonic() + 0.5)
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        ctx.run_real("tool", [])

    assert time.monotonic() - started < 5


def test_a_dry_run_reads_no_stdin(tmp_path):
    tool = script(tmp_path / "tool", "cat\necho done\n")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from biowarden.engine.checks import Ctx\n"
        f"bins = {{'tool': Path({str(tool)!r})}}\n"
        f"ctx = Ctx(Path({str(tmp_path)!r}), 'T1', bins, run_timeout=20)\n"
        "print(ctx.run_real('tool', []).stdout, end='')\n"
    )
    # the runner's stdin may be a pipe that never closes (communicate() would close it): the tool
    # must not wait on it
    with subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as child:
        try:
            child.wait(timeout=10)
            out = child.stdout.read()
        finally:
            child.kill()

    assert out == "done\n"


def test_a_dry_run_does_not_start_in_a_directory_outside_the_trial(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    marker = tmp_path / "ran"
    tool = script(tmp_path / "tool", f"touch {marker}\n")

    with pytest.raises(FileNotFoundError):
        Ctx(tmp_path, "T1", {"tool": tool}, root=work).run_real("tool", [])

    assert not marker.exists()
    assert Ctx(work, "T1", {"tool": tool}, root=work).run_real("tool", []).returncode == 0


# -- the socket (E-6) ------------------------------------------------------------------------------


@contextlib.contextmanager
def serving(spec: Spec, **engine_options):
    base = Path(tempfile.mkdtemp(prefix="biowarden-h-"))
    work = base / "work"
    work.mkdir()
    engine = Engine([spec], mode="block", **engine_options)
    server = EngineServer(base / "e.sock", engine, PathMap(work))
    serve_in_thread(server)
    try:
        yield server, work, base / "e.sock"
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def served():
    with serving(spec_of(RISKY)) as rig:
        yield rig


def talk(sock: Path, payload: bytes, close_write: bool = False) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(10)
        conn.connect(str(sock))
        conn.sendall(payload)
        if close_write:
            conn.shutdown(socket.SHUT_WR)
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def request(**fields) -> bytes:
    body = {"tool": "faketool", "argv": ["--risky"], "cwd": "/work", **fields}
    return (json.dumps({k: v for k, v in body.items() if v is not None}) + "\n").encode()


ALLOWED = {"action": "allow", "text": ""}


@pytest.mark.parametrize(
    "payload",
    [
        b"not json at all\n",
        b"\xff\xfe\x00\n",
        b"[]\n",
        b'"a string"\n',
        b"null\n",
        b"42\n",
        b"{}\n",
        b'{"tool": "faketool"}\n',
        request(argv=None),
        request(argv="--risky"),
        request(argv=["--risky", 3]),
        request(argv=[None]),
        request(cwd=None),
        request(cwd=""),
        request(cwd=["/work"]),
        request(tool=None),
        request(tool=5),
        request(declare="units=a"),
        request(declare=[1]),
        request(env_declare=7),
    ],
    ids=lambda p: repr(p[:30]),
)
def test_a_malformed_request_is_allowed_without_text_and_logged(served, payload):
    server, _, sock = served

    reply = talk(sock, payload)

    assert reply == ALLOWED
    assert len(server.errors) == 1 and server.errors[0].startswith("malformed request")


def test_an_empty_connection_is_a_malformed_request_too(served):
    server, _, sock = served

    assert talk(sock, b"", close_write=True) == ALLOWED
    assert len(server.errors) == 1


def test_the_shim_clients_own_requests_are_well_formed():
    check_request(
        {"tool": "t", "argv": ["a"], "declare": ["k=v"], "env_declare": "k=v", "cwd": "/work"}
    )
    check_request({"tool": "t", "argv": [], "cwd": "/work"})  # declare and env_declare optional
    with pytest.raises(ValueError):
        check_request({"tool": "t", "cwd": "/work"})


def test_a_valid_request_is_still_answered_with_the_fix_and_counted(served):
    server, work, sock = served

    reply = talk(sock, request())

    assert reply["action"] == "block" and reply["text"].startswith("FIX:")
    assert "BLOCKED, the command did not run" in reply["text"] and str(work) not in reply["text"]
    assert server.errors == [] and server.calls == 1
    assert talk(sock, request(argv=["--risky"], declare=["units=a"]))["action"] == "allow"


def test_the_host_path_of_the_shim_is_also_a_working_directory_inside_the_trial(served):
    server, work, sock = served

    for cwd in (str(work), str(work / "sub"), "/work/sub"):
        assert talk(sock, request(cwd=cwd))["action"] == "block", cwd


@pytest.mark.parametrize("cwd", ["/tmp", "/etc", "/workshop", "/tmp/work", "/", "/work/../etc"])
def test_a_working_directory_outside_the_trial_is_judged_from_the_work_directory(served, cwd):
    server, _, sock = served

    reply = talk(sock, request(cwd=cwd))

    assert reply["action"] == "block" and reply["text"].startswith("FIX:"), cwd
    assert server.errors == []  # a `cd /tmp` is no way round the layer, and no fault of it


def test_a_working_directory_beside_the_trial_is_judged_from_the_work_directory_too(served):
    server, work, sock = served

    assert talk(sock, request(cwd=str(work.parent)))["action"] == "block"
    assert talk(sock, request(cwd=str(work) + "-other"))["action"] == "block"


def test_the_log_of_a_probing_agent_is_capped(served):
    server, _, sock = served

    for _ in range(MAX_LOGGED + 20):
        talk(sock, b"junk\n")

    assert len(server.errors) == MAX_LOGGED


def test_the_server_does_not_follow_a_link_out_of_the_trial():
    with serving(spec_of(fact("hardening_reads"))) as (server, work, sock):
        (work.parent / "secret.dat").write_text("secret\n")
        (work / "link.dat").symlink_to("../secret.dat")
        (work / "own.dat").write_text("own\n")

        escaped = talk(sock, request(tool="awk", argv=["link.dat"]))
        own = talk(sock, request(tool="awk", argv=["own.dat"]))
        absolute = talk(sock, request(tool="awk", argv=[str(work.parent / "secret.dat")]))

    assert escaped == ALLOWED and absolute == ALLOWED
    assert own["action"] == "block" and "own" in own["text"]


@pytest.mark.parametrize("contract", ["full", "bare"])
def test_the_server_words_stop_by_the_contract_of_its_engine(contract):
    with serving(spec_of(RISKY), retry_cap=1, contract=contract) as (_, _, sock):
        first, second = (talk(sock, request())["text"] for _ in range(2))

    assert "STOP" not in first and "STOP" in second
    assert ("needs_input" in second) is (contract == "full")
    assert ('result.json with {"answer": null}' in second) is (contract == "bare")


def test_a_client_that_leaves_before_the_answer_does_not_crash_the_server(served):
    server, _, sock = served
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(str(sock))
        conn.sendall(request())
    time.sleep(0.2)

    assert talk(sock, request())["action"] == "block"


def test_a_rule_that_could_not_be_checked_says_nothing_to_the_shim(monkeypatch):
    monkeypatch.setattr("biowarden.engine.decide.is_regular_file", lambda *a: 1 / 0)
    with serving(spec_of(fact("hardening_reads"))) as (server, work, sock):
        (work / "a.dat").write_text("x\n")

        reply = talk(sock, request(tool="awk", argv=["a.dat"]))

    assert reply == {"action": "warn", "text": ""}  # the shim prints nothing and runs the tool
    assert server.errors == []  # an outcome of the ledger, not a fault of the handler


def test_a_genuine_fault_of_the_engine_still_warns_and_is_logged(served, monkeypatch):
    server, _, sock = served

    def broken(call, source="shim"):
        raise RuntimeError("engine fell over")

    monkeypatch.setattr(server.engine, "evaluate", broken)
    reply = talk(sock, request())

    assert reply["action"] == "warn" and "failed" in reply["text"]
    assert server.errors == ["RuntimeError: engine fell over"]


def test_pathmap_maps_a_working_directory_by_its_start_only(tmp_path):
    paths = PathMap(tmp_path / "w")

    assert paths.cwd_to_host("/work") == str(tmp_path / "w")
    assert paths.cwd_to_host("/work/a/b") == f"{tmp_path}/w/a/b"
    unrelated = (
        "/workshop",
        "/tmp/work",
        "/hom" + "e/u/work",
        "/work2",
        "/",
        "/tmp",
    )  # split literal
    for other in unrelated:
        assert paths.cwd_to_host(other) == other, other
    assert paths.to_host("a /tmp/work") != "a /tmp/work"  # free text is still replaced in place


def test_pathmap_confines_only_when_a_sandbox_sits_between(tmp_path):
    assert PathMap(tmp_path / "w").root == tmp_path / "w"
    assert PathMap(tmp_path, str(tmp_path)).root is None  # biowarden: the same files, no sandbox


# -- the pre-check and the declaration plumbing (E-9) ----------------------------------------------


@pytest.fixture
def declared_engine():
    def build(mode="block", **kwargs):
        return Engine([spec_of(RISKY)], mode=mode, retry_cap=10**9, **kwargs)

    return build


def test_a_malformed_declaration_on_a_watched_tool_blocks_before_anything_runs(
    tmp_path, declared_engine
):
    engine = declared_engine()

    for command in (
        "touch ran; faketool --risky --declare",
        "touch ran; faketool --risky --declare units",
        "touch ran; faketool --risky --declare units=",
        "touch ran; faketool --risky --declare=",
        "touch ran; BIOWARDEN_DECLARE=nonsense faketool --risky",
    ):
        result = precheck(command, tmp_path, engine)
        assert result.action == "block", command
        assert result.text.startswith("BLOCKED, the command did not run. "), command
        assert "key=value" in result.text, command


def test_a_malformed_declaration_on_a_program_nobody_watches_is_left_alone(
    tmp_path, declared_engine
):
    engine = declared_engine()

    for command in ("echo --declare", "grep -- --declare notes.txt", "ls --declare units"):
        assert precheck(command, tmp_path, engine).action == "allow", command


def test_an_exported_declaration_reaches_the_commands_after_it(tmp_path, declared_engine):
    engine = declared_engine()

    undeclared = precheck("faketool --risky", tmp_path, engine)
    exported = precheck("export BIOWARDEN_DECLARE=units=a; faketool --risky", tmp_path, engine)
    prefixed = precheck("BIOWARDEN_DECLARE=units=a faketool --risky", tmp_path, engine)
    later = precheck("faketool --risky; export BIOWARDEN_DECLARE=units=a", tmp_path, engine)
    mixed = precheck(
        "export OTHER=1 BIOWARDEN_DECLARE=units=a; echo x; faketool --risky", tmp_path, engine
    )
    quoted = precheck("export BIOWARDEN_DECLARE='units=a'; faketool --risky", tmp_path, engine)

    assert undeclared.action == later.action == "block"
    assert exported.action == prefixed.action == mixed.action == quoted.action == "allow"


def test_an_exported_declaration_is_checked_against_the_call(tmp_path, declared_engine):
    result = precheck(
        "export BIOWARDEN_DECLARE=units=b; faketool --risky", tmp_path, declared_engine()
    )

    assert result.action == "block" and "you declared units=b" in result.text


def test_a_prefix_assignment_beats_an_earlier_export(tmp_path, declared_engine):
    engine = declared_engine()

    result = precheck(
        "export BIOWARDEN_DECLARE=units=b; BIOWARDEN_DECLARE=units=a faketool --risky",
        tmp_path,
        engine,
    )

    assert result.action == "allow"


def test_a_tool_called_by_its_path_is_told_to_be_called_by_name(tmp_path, declared_engine):
    engine = declared_engine()

    for command in (
        "/usr/bin/faketool --risky",  # the pre-check asks for a declaration the tool cannot take
        "/usr/bin/faketool --risky --declare units=a",
        "./faketool --risky --declare units=a",
        "env X=1 /opt/x/faketool --risky",
    ):
        result = precheck(command, tmp_path, engine)
        assert result.action == "block", command
        assert result.text.startswith("FIX: call faketool by name, not as "), command
        assert "BLOCKED, the command did not run." in result.text, command


def test_a_declaration_rule_that_only_warns_leaves_the_path_call_running(tmp_path):
    unpromoted = Engine([spec_of(RISKY)], mode="promoted", promotion=Promotion())
    promoted = Engine(
        [spec_of(RISKY)],
        mode="promoted",
        promotion=Promotion({"hardening.risky": {"mode": "block"}}),
    )

    soft = precheck("/usr/bin/faketool --risky", tmp_path, unpromoted)
    hard = precheck("/usr/bin/faketool --risky", tmp_path, promoted)
    marked = precheck("/usr/bin/faketool --risky --declare units=a", tmp_path, unpromoted)

    assert soft.action == "warn" and "call faketool by name" in soft.text
    assert "BLOCKED" not in soft.text
    assert hard.action == "block" and hard.text.startswith("FIX: call faketool by name")
    assert marked.action == "block"  # the tool would be handed --declare and fail


def test_calling_by_name_after_that_message_works(tmp_path, declared_engine):
    engine = declared_engine()

    assert precheck("faketool --risky --declare units=a", tmp_path, engine).action == "allow"


def test_a_path_call_that_needs_no_declaration_is_left_alone(tmp_path, declared_engine):
    engine = declared_engine()

    for command in ("/usr/bin/faketool --fine", "/usr/bin/echo --risky", "/usr/bin/python3 x.py"):
        assert precheck(command, tmp_path, engine).action == "allow", command


def test_the_shim_directory_is_not_a_path_call(tmp_path, declared_engine):
    engine = declared_engine()

    result = precheck("/opt/biowarden/bin/faketool --risky --declare units=a", tmp_path, engine)

    assert result.action == "allow"  # the shim strips the marker


def test_in_warn_mode_a_path_call_with_a_declaration_warns_and_still_runs(
    tmp_path, declared_engine
):
    engine = declared_engine("warn")

    marked = precheck("/usr/bin/faketool --risky --declare units=a", tmp_path, engine)
    bare = precheck("/usr/bin/faketool --risky", tmp_path, engine)

    for result in (marked, bare):
        assert result.action == "warn" and "call faketool by name" in result.text
        assert "BLOCKED" not in result.text


def test_the_generic_style_never_mentions_declarations(tmp_path, declared_engine):
    engine = declared_engine(style="generic")

    result = precheck("/usr/bin/faketool --risky", tmp_path, engine)

    assert "declare" not in result.text and "by name" not in result.text


def test_has_declare_marker():
    assert has_declare_marker(["--declare", "a=b"]) and has_declare_marker(["--declare=a=b"])
    assert has_declare_marker(["x", "--declare"])
    assert not has_declare_marker(["--declared", "x", "--declare-all"])


# -- the words of STOP and of a crash (E-10) -------------------------------------------------------


def stopped(contract: str):
    engine = Engine([spec_of(RISKY)], mode="block", retry_cap=1, contract=contract)
    call = Call("faketool", ("--risky",), Path("/tmp"), {"units": "b"})
    decisions = [engine.evaluate(call) for _ in range(2)]
    return engine, decisions


def test_stop_names_needs_input_only_under_the_full_contract():
    engine, (first, second) = stopped("full")
    assert not first.stop and second.stop

    full = render_block(second, engine.retry_cap, engine.style, engine.contract)
    bare = render_block(second, engine.retry_cap, engine.style, "bare")

    assert "needs_input" in full and "needs_input" not in bare
    assert bare.splitlines()[0].startswith("STOP: this exact call was already blocked 1 times.")
    assert 'result.json with {"answer": null}' in bare and "FIX: declare units=a" in bare


def test_the_engine_carries_its_contract_to_every_front_end(tmp_path):
    full = Engine([spec_of(RISKY)], mode="block", retry_cap=1)
    bare = Engine([spec_of(RISKY)], mode="block", retry_cap=1, contract="bare")
    assert full.contract == "full" and bare.contract == "bare"

    for engine, expected, absent in (
        (full, "needs_input", "result.json"),
        (bare, "result.json", "needs_input"),
    ):
        texts = [precheck("faketool --risky", tmp_path, engine).text for _ in range(2)]
        assert "STOP" not in texts[0] and "STOP" in texts[1]
        assert expected in texts[1] and absent not in texts[1]


def test_a_block_that_is_not_a_stop_reads_the_same_under_both_contracts():
    engine, (first, _) = stopped("full")

    assert render_block(first, 1, "full", "bare") == render_block(first, 1, "full", "full")
    assert render_block(first, 1) == render_block(first, 1, "full", "full")


def test_the_stop_texts_are_the_pinned_ones():
    assert STOP_TEXT.format(n=2).startswith("STOP: this exact call was already blocked 2 times.")
    assert "needs_input" in STOP_TEXT
    assert STOP_TEXT_BARE.format(n=2).endswith('result.json with {"answer": null}.')


def test_a_crashed_checker_block_carries_a_fix_line(tmp_path):
    (tmp_path / "a.dat").write_text("x\n")
    engine = Engine([spec_of(fact("hardening_crashes"))], mode="block")

    decision = engine.evaluate(Call("awk", ("a.dat",), tmp_path))
    text = render_block(decision)

    assert decision.action == "block" and text.splitlines()[0] == f"FIX: {CRASHED_FIX}"
    assert "BLOCKED, the command did not run. hardening.x [units/T1]: RuntimeError: boom" in text
    warn = Engine([spec_of(fact("hardening_crashes"))], mode="warn").evaluate(
        Call("awk", ("a.dat",), tmp_path)
    )
    assert f"FIX: {CRASHED_FIX}" in render_warning(warn)


def test_a_block_with_its_own_fix_is_not_given_the_crash_fix(tmp_path):
    engine = Engine([spec_of(RISKY)], mode="block")

    text = render_block(engine.evaluate(Call("faketool", ("--risky",), tmp_path)))

    assert CRASHED_FIX not in text and text.startswith("FIX: re-run with --declare units=<a|b>")


# -- MCP (E-9) -------------------------------------------------------------------------------------


def ask(server, arguments):
    async def go():
        async with Client(server) as client:
            return await client.call_tool_mcp("run_tool", arguments)

    return asyncio.run(go())


def mcp_server(ran, mode="block"):
    engine = Engine(packaged_specs(), mode=mode, contract="bare", retry_cap=1)

    def runner(tool, args, cwd):
        ran.append((tool, list(args)))
        return 0, "out\n", ""

    return build_mcp(engine, runner, Path("/tmp"), ["samtools", "bcftools", "bedtools"])


@pytest.mark.parametrize(
    "args",
    [
        ["view", "-R", "r.bed", "--declare", "region_semantics=overlap", "v.vcf.gz"],
        ["view", "-R", "r.bed", "--declare=region_semantics=overlap"],
        ["view", "--declare"],
    ],
)
def test_a_declare_marker_in_args_is_rejected_and_the_tool_does_not_run(args):
    ran = []

    result = ask(mcp_server(ran), {"tool": "bcftools", "args": args})

    assert result.is_error is True and ran == []
    text = result.content[0].text
    assert text.startswith("Error: --declare is not a tool argument") and "`declared`" in text
    assert result.structured_content == {"status": "rejected"}


def test_a_declaration_in_declared_still_works_through_mcp():
    ran = []

    result = ask(
        mcp_server(ran),
        {
            "tool": "bcftools",
            "args": ["view", "-R", "r.bed", "v.vcf.gz"],
            "declared": {"region_semantics": "overlap"},
        },
    )

    assert result.is_error is False and ran == [("bcftools", ["view", "-R", "r.bed", "v.vcf.gz"])]


def test_a_fault_of_the_layer_adds_no_blank_line_to_an_mcp_result(monkeypatch):
    monkeypatch.setattr("biowarden.engine.decide.is_regular_file", lambda *a: 1 / 0)
    ran = []
    engine = Engine([spec_of(fact("hardening_reads", ["bcftools"]))], mode="block")
    server = build_mcp(
        engine, lambda t, a, c: ran.append(t) or (0, "out\n", ""), Path("/tmp"), ["bcftools"]
    )

    result = ask(server, {"tool": "bcftools", "args": ["view", "a.dat"]})

    assert result.is_error is False and ran == ["bcftools"]
    assert result.content[0].text.startswith("[exit code 0]")


def test_mcp_words_stop_by_the_contract_of_its_engine():
    ran = []
    server = mcp_server(ran)
    call = {"tool": "bcftools", "args": ["view", "-R", "r.bed", "v.vcf.gz"]}

    texts = [ask(server, call).content[0].text for _ in range(2)]

    assert "STOP" not in texts[0] and "STOP" in texts[1]
    assert "needs_input" not in texts[1] and "result.json" in texts[1]


# -- normal calls decide as before -----------------------------------------------------------------


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def test_a_real_workspace_is_decided_the_same_with_and_without_the_confinement(tmp_path, tools):
    work = tmp_path / "work"
    build_task("F1", 1000, "faulted", work, tools, "implicit", "forced")
    engine = Engine(
        packaged_specs(),
        mode="block",
        real_bins=tools.real_bins(["bcftools", "samtools", "bedtools"]),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )
    argv = ("view", "-H", "-R", "regions.tsv", "variants.vcf.gz")
    home = work.resolve()

    loose = engine.evaluate(Call("bcftools", argv, home))
    tight = engine.evaluate(Call("bcftools", argv, home, root=home))
    blind = engine.evaluate(Call("bcftools", argv, home, root=tmp_path / "elsewhere"))

    assert verdicts(loose) == verdicts(tight) and loose.action == "block"
    assert render_block(loose) == render_block(tight)
    assert ("bcftools.regions_file_base", "regions_file_unreadable", "warn") in verdicts(blind)


def test_the_precheck_of_a_real_workspace_is_unchanged_by_a_path_map(tmp_path, tools):
    work = tmp_path / "work"
    build_task("F1", 1000, "faulted", work, tools, "implicit", "forced")
    engine = Engine(
        packaged_specs(),
        mode="block",
        real_bins=tools.real_bins(["bcftools", "samtools", "bedtools"]),
        real_env=tools.process_env(),
        retry_cap=10**9,
    )
    command = "cd /work && bcftools view -H -R regions.tsv variants.vcf.gz | wc -l"

    mapped = precheck(command, Path("/work"), engine, PathMap(work.resolve()))
    plain = precheck(command.replace("/work", str(work.resolve())), work.resolve(), engine)

    assert mapped.action == plain.action == "block"
    assert mapped.text == plain.text.replace(str(work.resolve()), "/work")


def test_a_pathological_json_line_gets_the_silent_reply_not_a_failure_notice(served):
    server, _, sock = served

    reply = talk(sock, b"[" * 200_000 + b"\n")

    assert reply == ALLOWED and any("RecursionError" in line for line in server.errors)


def test_files_named_in_code_must_lie_inside_the_trial(tmp_path):
    import re

    from biowarden.engine.rules.sc_common import find_files

    work, other = tmp_path / "work", tmp_path / "other"
    work.mkdir()
    other.mkdir()
    (work / "a.h5ad").write_text("x")
    (other / "secret.h5ad").write_text("x")
    (work / "link.h5ad").symlink_to(other / "secret.h5ad")
    text = f"{work}/a.h5ad {other}/secret.h5ad link.h5ad ../other/secret.h5ad"
    pattern = re.compile(r"[\w./-]+\.h5ad")

    assert [p.name for p in find_files(text, work, pattern, work)] == ["a.h5ad"]
    assert len(find_files(text, work, pattern)) == 4  # without a root nothing is confined
