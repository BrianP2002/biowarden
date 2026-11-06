"""biowarden: run a command with the shims on PATH, no sandbox."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def guard(tools, work, *command, mode="block"):
    env = {**os.environ, **tools.process_env(), "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "biowarden.engine.guard", "--mode", mode, "--", *command],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )


def test_a_risky_call_is_refused_and_the_status_is_the_commands(tmp_path, tools):
    work = tmp_path / "w"
    build_task("F1", 1000, "faulted", work, tools, "implicit", "forced")

    out = guard(tools, work, "bash", "-c", "bcftools view -H -R regions.tsv variants.vcf.gz")

    assert out.returncode == 97
    assert "BLOCKED, the command did not run" in out.stderr and str(tmp_path) not in out.stderr[:20]


def test_the_declared_call_runs_and_ordinary_commands_are_untouched(tmp_path, tools):
    work = tmp_path / "w"
    build_task("F1", 1000, "faulted", work, tools, "implicit", "forced")
    declared = (
        "bcftools --declare region_file_base=one_based_closed --declare region_semantics=overlap "
        "view -H -R regions.tsv variants.vcf.gz | wc -l"
    )

    ok = guard(tools, work, "bash", "-c", declared)
    plain = guard(tools, work, "bash", "-c", "echo hello; python3 -c 'print(6*7)'")

    assert ok.returncode == 0 and int(ok.stdout.strip()) > 0
    assert plain.returncode == 0 and plain.stdout.split() == ["hello", "42"]


def test_warn_mode_lets_the_call_run(tmp_path, tools):
    work = tmp_path / "w"
    build_task("F1", 1000, "faulted", work, tools, "implicit", "forced")

    out = guard(
        tools, work, "bash", "-c", "bcftools view -H -R regions.tsv variants.vcf.gz", mode="warn"
    )

    assert out.returncode == 0 and "WARNING" in out.stderr


def test_a_missing_command_is_reported(tmp_path, tools):
    out = guard(tools, tmp_path, "no-such-command-xyz")

    assert out.returncode == 127 and "command not found" in out.stderr
