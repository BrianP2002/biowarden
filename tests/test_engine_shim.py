"""The shim client, the engine server behind it, and the sandbox wiring (a fake tool, real F1)."""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from biowarden.bench.runner import sandbox
from biowarden.bench.runner.layer import ARMS, start_layer
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Engine, Spec
from biowarden.engine.install import install_shims
from biowarden.engine.server import EngineServer, PathMap, serve_in_thread

ROOT = Path(__file__).resolve().parents[1]

RULE = {
    "id": "shimtest.mark",
    "slot": "units",
    "kind": "call",
    "tier": "T0",
    "applies": {"tools": ["faketool"], "options_any": ["--risky"]},
    "declare": {"key": "units", "values": ["a", "b"]},
    "check": "option_implies_declared",
    "params": {"option_map": {"--risky": "a"}, "fix_option": {}},
    "message": "faketool --risky means units a",
}


def spec() -> Spec:
    return Spec.model_validate({"name": "shimtest", "rules": [RULE]})


@pytest.fixture
def rig(tmp_path):
    """A shim dir with one fake tool behind it, an engine server, and a way to run the shim."""
    real = tmp_path / "real"
    real.mkdir()
    marker = tmp_path / "ran.txt"
    fake = real / "faketool"
    fake.write_text(f'#!/bin/sh\necho "$0 $@" >> {marker}\necho tool-output\n')
    fake.chmod(0o755)
    shims = tmp_path / "shims"
    install_shims(shims, [real], names=("faketool",))
    (shims / "missing").symlink_to("biowarden-shim")  # a watched name with no real tool behind it
    work = tmp_path / "work"
    work.mkdir()
    sock = Path(f"/tmp/biowarden-test-{os.getpid()}-{tmp_path.name[-6:]}.sock")
    holder = {}

    def start(mode="block"):
        engine = Engine([spec()], mode=mode)
        server = EngineServer(sock, engine, PathMap(work))
        serve_in_thread(server)
        holder["server"] = server
        return server

    def run(*args, tool="faketool", sock_path=sock, env_extra=None):
        env = {
            "PATH": "/usr/bin:/bin",
            "BIOWARDEN_SOCK": str(sock_path),
            "BIOWARDEN_REAL_PATH": str(real),
            **(env_extra or {}),
        }
        return subprocess.run(
            [str(shims / tool), *args], cwd=work, env=env, capture_output=True, text=True
        )

    yield start, run, marker, holder
    if "server" in holder:
        holder["server"].shutdown()
        holder["server"].server_close()
    sock.unlink(missing_ok=True)


def test_an_allowed_call_runs_the_real_tool_with_the_real_path_as_argv0(rig):
    start, run, marker, _ = rig
    start()

    out = run("--fine")

    assert out.returncode == 0 and out.stdout.strip() == "tool-output" and out.stderr == ""
    assert marker.read_text().split()[0].endswith("/real/faketool")


def test_a_blocked_call_never_runs_the_tool_and_exits_97(rig):
    start, run, marker, _ = rig
    start("block")

    out = run("--risky", "x")

    assert out.returncode == 97
    assert not marker.exists()
    assert out.stderr.startswith("FIX:") and "BLOCKED, the command did not run" in out.stderr


def test_declarations_are_stripped_before_the_tool_sees_the_arguments(rig):
    start, run, marker, _ = rig
    start("block")

    out = run("--risky", "--declare", "units=a", "x")

    assert out.returncode == 0
    assert marker.read_text().split()[1:] == ["--risky", "x"]


def test_a_warning_is_printed_and_the_tool_still_runs(rig):
    start, run, marker, _ = rig
    start("warn")

    out = run("--risky", "--declare", "units=b")

    assert out.returncode == 0 and "WARNING" in out.stderr and marker.exists()


def test_the_environment_declaration_channel_works_for_every_call(rig):
    start, run, marker, _ = rig
    start("block")

    out = run("--risky", env_extra={"BIOWARDEN_DECLARE": "units=a"})

    assert out.returncode == 0 and marker.exists()


def test_an_unreachable_engine_lets_the_call_run_and_says_so(rig):
    _, run, marker, _ = rig

    out = run("--risky", sock_path=Path("/tmp/does-not-exist.sock"))

    assert out.returncode == 0 and marker.exists()
    assert "could not be reached" in out.stderr


def test_a_tool_that_is_not_installed_behaves_like_command_not_found(rig):
    start, run, _, _ = rig
    start()

    out = run(tool="missing")

    assert out.returncode == 127 and "command not found" in out.stderr


def test_server_maps_sandbox_paths_in_both_directions(tmp_path):
    paths = PathMap(tmp_path / "w")

    assert paths.to_host("cat /work/a.txt /workshop/b") == f"cat {tmp_path}/w/a.txt /workshop/b"
    assert paths.to_sandbox(f"file {tmp_path}/w/a.txt") == "file /work/a.txt"


def test_concurrent_calls_are_all_answered(rig):
    start, run, _, _ = rig
    start("block")
    results = []

    def one() -> None:
        results.append(run("--risky").returncode)

    threads = [threading.Thread(target=one) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert results == [97] * 8


def test_arm_table_covers_the_planned_arms():
    assert set(ARMS) == {
        "bare",
        "note",
        "warn",
        "warn-nonote",
        "block-nonote",
        "warn-generic",
        "block-generic",
        "careful",
        "block",
        "block-shim",
        "block-bypass",
        "block-gateway",
        "blockall",
    }
    assert ARMS["bare"] is None and ARMS["block-bypass"].hide_real is False
    assert ARMS["block-shim"].precheck is False and ARMS["block"].precheck is True
    gateway_arm = ARMS["block-gateway"]
    assert gateway_arm.gateway and not gateway_arm.shims


# -- the real sandbox, the real F1 task and the packaged rules ---------------------------------


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not sandbox.available():
        pytest.skip("real tools or bubblewrap are not available")
    return env


def f1_layer(tmp_path, tools, arm):
    work = tmp_path / "trial" / "work"
    task = build_task("F1", 0, "faulted", work, tools, "implicit", "forced")
    rig = start_layer(arm, tmp_path / "trial", work, tools)
    return work, task, rig


def test_the_precheck_refuses_the_whole_command_before_anything_runs(tmp_path, tools):
    work, _, rig = f1_layer(tmp_path, tools, "blockall")
    try:
        out = sandbox.run_command(
            work,
            tools,
            "echo started > ran.txt; bcftools view -H -R regions.tsv variants.vcf.gz | wc -l",
            layer=rig.layer,
        )
        summary = rig.summary()
    finally:
        rig.close()

    assert out.exit_code == 97 and "BLOCKED, the command did not run" in out.stderr
    assert "/gpfs" not in out.stderr and out.stdout == ""
    assert not (work / "ran.txt").exists()
    assert summary["by_source"]["precheck"]["block"] == 1 and summary["errors"] == []


def test_the_shim_alone_still_blocks_the_tool_but_not_the_rest_of_the_pipeline(tmp_path, tools):
    work, _, rig = f1_layer(tmp_path, tools, "blockall")
    shim_only = replace(rig.layer, precheck=None)
    try:
        out = sandbox.run_command(
            work,
            tools,
            "bcftools view -H -R regions.tsv variants.vcf.gz | wc -l",
            layer=shim_only,
        )
        summary = rig.summary()
    finally:
        rig.close()

    assert "BLOCKED, the command did not run" in out.stderr and out.stdout.strip() == "0"
    assert summary["by_source"]["shim"]["block"] == 1


def test_a_pipeline_stage_that_names_no_file_is_caught_only_with_the_precheck(tmp_path, tools):
    work = tmp_path / "trial" / "work"
    build_task("F13", 0, "faulted", work, tools, "implicit", "forced")
    command = (
        "cat reads.fastq | awk 'NR%4==0{for(i=1;i<=length($0);i++)"
        'if(substr($0,i,1)>="?")n++}END{print n}\''
    )
    for use_precheck, expect_block in ((True, True), (False, False)):
        rig = start_layer("blockall", tmp_path / f"t{use_precheck}", work, tools)
        layer = rig.layer if use_precheck else replace(rig.layer, precheck=None)
        try:
            out = sandbox.run_command(work, tools, command, layer=layer)
        finally:
            rig.close()
        assert ("BLOCKED" in out.stderr) is expect_block


def test_the_declaration_unlocks_the_call_and_the_ledger_records_it(tmp_path, tools):
    work, _, rig = f1_layer(tmp_path, tools, "blockall")
    try:
        out = sandbox.run_command(
            work,
            tools,
            "bcftools --declare region_file_base=one_based_closed "
            "--declare region_semantics=overlap view -H -R regions.tsv variants.vcf.gz | wc -l",
            layer=rig.layer,
        )
        summary = rig.summary()
    finally:
        rig.close()

    assert int(out.stdout.strip()) > 0 and "BLOCKED" not in out.stderr
    assert summary["actions"]["block"] == 0


def test_hidden_real_tools_are_not_on_path_but_bypass_arm_keeps_them(tmp_path, tools):
    # the sandbox always has /usr/bin on PATH, so tools installed there (a distribution's
    # packages, as on CI) cannot be hidden: only the order of PATH and the shim can be checked
    hideable = str(tools.bin_dirs[0]) != "/usr/bin"
    for arm, on_path in (("blockall", False), ("block-bypass", True)):
        work, _, rig = f1_layer(tmp_path / arm, tools, arm)
        try:
            out = sandbox.run_command(work, tools, "echo $PATH; which -a bcftools", layer=rig.layer)
        finally:
            rig.close()
        lines = out.stdout.split()
        assert lines[0].startswith("/opt/biowarden/bin:")
        if hideable:
            assert (str(tools.bin_dirs[0]) in lines[0]) is on_path
        assert "/opt/biowarden/bin/bcftools" in out.stdout


def test_python3_inside_the_sandbox_goes_through_the_shim_without_breaking(tmp_path, tools):
    work, _, rig = f1_layer(tmp_path, tools, "blockall")
    try:
        out = sandbox.run_command(work, tools, "python3 -c 'print(6*7)'", layer=rig.layer)
    finally:
        rig.close()

    assert out.stdout.strip() == "42" and out.exit_code == 0


def test_the_gateway_tool_blocks_a_trap_and_runs_a_declared_call(tmp_path, tools):
    work = tmp_path / "trial" / "work"
    build_task("F1", 0, "faulted", work, tools, "implicit", "forced")
    table = tmp_path / "promotion.json"
    table.write_text(
        '{"bcftools.regions_file_base": {"mode": "block"},'
        ' "bcftools.region_semantics": {"mode": "block"}}'
    )
    # a two-rule table: start_layer would refuse it unless the missing entries are only recorded
    rig = start_layer(
        "block-gateway", tmp_path / "trial", work, tools, str(table), allow_unpromoted=None
    )
    assert rig is not None and rig.layer is None and rig.gateway is not None
    try:
        bad = rig.gateway.run(
            {"tool": "bcftools", "args": ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"]}
        )
        good = rig.gateway.run(
            {
                "tool": "bcftools",
                "args": ["view", "-H", "-R", "regions.tsv", "variants.vcf.gz"],
                "declared": {
                    "region_file_base": "one_based_closed",
                    "region_semantics": "overlap",
                },
            }
        )
        junk = rig.gateway.run({"tool": "rm", "args": ["-rf", "x"]})
        summary = rig.summary()
    finally:
        rig.close()

    assert bad[1] is True and "BLOCKED" in bad[0] and "/gpfs" not in bad[0]
    assert good[1] is False and "[exit code 0]" in good[0]
    assert junk[1] is True and "Error" in junk[0]
    assert summary["by_source"]["gateway"]["block"] == 1


def test_the_gateway_schema_lists_the_declarations_as_enums(tmp_path, tools):
    work = tmp_path / "trial" / "work"
    build_task("F1", 0, "faulted", work, tools, "implicit", "forced")
    rig = start_layer("block-gateway", tmp_path / "trial", work, tools, None)
    try:
        schema = rig.gateway.anthropic_schema()
    finally:
        rig.close()

    declared = schema["input_schema"]["properties"]["declared"]["properties"]
    assert declared["region_semantics"]["enum"] == ["overlap", "start_only"]
    assert "fastq_quality_offset" in declared and schema["name"] == "run_tool"


def test_opencode_sandbox_gets_the_shims_behind_its_own_binary(tmp_path):
    from biowarden.bench.runner.opencode import OpenCodeRig, sandbox_argv
    from biowarden.bench.runner.sandbox import SandboxLayer

    class FakeProxy:
        socket_path = tmp_path / "proxy.sock"

    env = ToolEnv((tmp_path / "bin",), ())
    layer = SandboxLayer(tmp_path / "shims", tmp_path / "e.sock", hide_real=True)
    for path in (
        layer.shim_dir,
        tmp_path / "bin",
        tmp_path / "oc",
        tmp_path / "work",
        tmp_path / "oc",
    ):
        path.mkdir(exist_ok=True)
    rig = OpenCodeRig(tmp_path / "ocbin", FakeProxy())

    plain = sandbox_argv(rig, env, tmp_path / "work", tmp_path / "oc", "m/x", "t", "hi", 60)
    shimmed = sandbox_argv(
        rig, env, tmp_path / "work", tmp_path / "oc", "m/x", "t", "hi", 60, layer
    )

    def setenv(argv, name):
        i = max(i for i, a in enumerate(argv) if a == "--setenv" and argv[i + 1] == name)
        return argv[i + 2]

    assert "/opt/biowarden/bin" not in " ".join(plain) and "BIOWARDEN_SOCK" not in plain
    assert setenv(shimmed, "PATH") == "/opt/opencode:/opt/biowarden/bin:/usr/bin"
    assert setenv(shimmed, "BIOWARDEN_SOCK") == "/opt/biowarden/engine.sock"
    assert setenv(shimmed, "BIOWARDEN_REAL_PATH").startswith(str(tmp_path / "bin"))
    assert ["--ro-bind", str(layer.shim_dir.resolve()), "/opt/biowarden/bin"] == shimmed[
        shimmed.index("/opt/biowarden/bin") - 2 : shimmed.index("/opt/biowarden/bin") + 1
    ]


def test_a_program_no_rule_watches_passes_straight_through_without_the_engine(rig):
    start, run, marker, _ = rig

    out = run(
        "--risky",
        env_extra={"BIOWARDEN_WATCH": "somethingelse"},
        sock_path=Path("/tmp/no-engine.sock"),
    )

    assert out.returncode == 0 and marker.exists()
    assert "could not be reached" not in out.stderr  # it never tried


def test_tail_directories_come_after_usr_bin_and_are_bound(tmp_path):
    first, tail = tmp_path / "bin", tmp_path / "env" / "bin"
    first.mkdir()
    tail.mkdir(parents=True)
    (tail / "bwa").write_text("#!/bin/sh\n")
    (tail / "bwa").chmod(0o755)
    env = ToolEnv((first,), (), (tail,))

    assert env.path_var() == f"{first}:/usr/bin:{tail}"
    assert tail.parent in env.ro_binds()
    assert env.real_bins(["bwa", "nothere"]) == {"bwa": tail / "bwa"}
    assert env.all_executables() == ["bwa"]
