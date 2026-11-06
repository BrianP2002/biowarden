"""examples/bed-base: the README demo is what the guard prints, and its two counts are real."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv
from helpers import console_blocks

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "bed-base"
COMMAND = "bcftools view -H -R regions.tsv variants.vcf.gz"
AS_BED = COMMAND.replace("regions.tsv", "regions.tsv.bed")


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    tools = ToolEnv.discover(ROOT)
    if tools is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    env = {**os.environ, **tools.process_env(), "PYTHONPATH": str(ROOT / "src")}
    try:
        probe = subprocess.run(["bcftools", "--version"], env=env, capture_output=True)
    except OSError:
        probe = None
    if probe is None or probe.returncode != 0:
        pytest.skip("bcftools does not run here (a missing shared library?)")
    return env


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A copy of the example: the BED reading below writes a file next to the inputs."""
    shutil.copytree(EXAMPLE, tmp_path / "bed-base")
    return tmp_path / "bed-base"


def run(env: dict[str, str], cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)


def test_the_demo_in_the_readme_is_what_the_guard_prints(env, work):
    demo = console_blocks((ROOT / "README.md").read_text())[0]
    assert demo[0] == f"$ cd examples/bed-base && biowarden -- {COMMAND}"

    guard = [sys.executable, "-m", "biowarden.engine.guard", "--", *COMMAND.split()]
    out = run(env, work, *guard)

    # the first two lines word for word, the third as far as the README prints it
    assert out.returncode == 97 and out.stdout == ""
    printed = out.stderr.splitlines()
    assert [re.sub(r"\s+", " ", line) for line in printed[:2]] == demo[1:3]
    assert printed[2].startswith(demo[3].removesuffix(" …"))
    assert "Also: bcftools.region_semantics" in (EXAMPLE / "README.md").read_text()


def test_the_two_readings_of_the_regions_file_select_different_records(env, work):
    """17 records read as 1-based, 14 read as BED: the numbers in the demo and the example."""
    one_based = run(env, work, "bash", "-c", f"{COMMAND} | wc -l")
    shutil.copy(work / "regions.tsv", work / "regions.tsv.bed")
    bed = run(env, work, "bash", "-c", f"{AS_BED} | wc -l")

    assert (int(one_based.stdout), int(bed.stdout)) == (17, 14)
    demo = " ".join(console_blocks((ROOT / "README.md").read_text())[0])
    assert "selects 17 records of variants.vcf.gz, read as BED 14" in demo
    example = (EXAMPLE / "README.md").read_text()
    assert re.search(r"^17$", example, re.M) and re.search(r"^14$", example, re.M)


def test_the_commands_in_the_readme_are_installed_scripts():
    scripts = (ROOT / "pyproject.toml").read_text()
    assert 'biowarden = "biowarden.engine.guard:main"' in scripts
    assert 'biowarden-mcp = "biowarden.engine.mcp_server:main"' in scripts


def test_following_the_fix_and_declaring_prints_the_14_records(env, work):
    """The bash block of the example: the fix meets the second rule, a declaration passes it."""
    text = (EXAMPLE / "README.md").read_text()
    copy, call = re.findall(r"```bash\n(.*?)```", text, flags=re.S)[0].splitlines()
    argv = shlex.split(call.split("  #")[0])
    assert copy == "cp regions.tsv regions.tsv.bed" and argv[:2] == ["biowarden", "--"]
    guard = [sys.executable, "-m", "biowarden.engine.guard", *argv[1:]]
    shutil.copy(work / "regions.tsv", work / "regions.tsv.bed")

    declared = run(env, work, *guard)
    undeclared = run(env, work, *guard[: guard.index("--declare")])

    assert declared.returncode == 0 and len(declared.stdout.splitlines()) == 14
    assert undeclared.returncode == 97 and "bcftools.region_semantics" in undeclared.stderr
