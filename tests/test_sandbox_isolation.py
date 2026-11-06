"""The sandbox exposes the tools and the trial's work directory, and nothing of the project.

Regression for the stage-2 review (R-1/FC-1): a tool directory that is not named `bin` used to be
bound through its parent, and a wrapper directory inside the project's `.local` therefore put the
whole `.local` (rule catalogue, notes, scripts) inside every agent command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.runner import sandbox
from biowarden.bench.tasks.tools import ToolEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not sandbox.available():
        pytest.skip("needs samtools, bcftools, bedtools and bubblewrap")
    return env


@pytest.fixture
def work(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


def project_like(tmp_path):
    """A directory tree shaped like the project: wrapper links beside a private file."""
    project = tmp_path / "proj"
    wrappers = project / ".local" / "pyenvs"
    wrappers.mkdir(parents=True)
    (wrappers / "python-x").write_text("#!/bin/sh\n")
    (project / ".local" / "secret.txt").write_text("TRAP-CATALOGUE")
    firsts = tmp_path / "soft" / "tool" / "bin"
    firsts.mkdir(parents=True)
    return project, wrappers, firsts


# --- what gets bound -----------------------------------------------------------------------


def test_a_bin_directory_is_bound_through_its_install_prefix_and_others_are_bound_themselves(
    tmp_path,
):
    project, wrappers, firsts = project_like(tmp_path)
    conda = tmp_path / "envs" / "bio" / "bin"
    conda.mkdir(parents=True)
    env = ToolEnv((firsts,), (tmp_path / "lib",), (wrappers, conda))

    binds = env.ro_binds()

    assert binds == [firsts.parent, wrappers, conda.parent, tmp_path / "lib"]
    assert project / ".local" not in binds and project not in binds


def test_two_tool_directories_with_one_prefix_are_bound_once(tmp_path):
    prefix = tmp_path / "soft" / "tool"
    (prefix / "bin").mkdir(parents=True)
    env = ToolEnv((prefix / "bin",), (), (prefix / "bin",))

    assert env.ro_binds() == [prefix]


# --- the guard -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "where",
    [
        pytest.param(lambda: ROOT / "some" / "bin", id="inside the project"),
        pytest.param(lambda: ROOT / ".local" / "pyenvs", id="a wrapper dir in the project"),
        pytest.param(lambda: ROOT, id="the project itself"),
        pytest.param(lambda: ROOT.parent, id="an ancestor of the project"),
        pytest.param(lambda: Path.home(), id="the home directory"),
    ],
)
def test_a_bind_that_would_expose_the_project_or_home_is_refused(where):
    with pytest.raises(ValueError, match="refusing to bind"):
        sandbox.tool_binds(ToolEnv((Path("/usr/bin"),), (), (where(),)))


def test_the_guard_applies_to_the_bwrap_command_line(tmp_path):
    wrappers = ROOT / ".local" / "pyenvs"
    with pytest.raises(ValueError):
        sandbox.bwrap_argv(tmp_path, ToolEnv((Path("/usr/bin"),), (), (wrappers,)), "true", 5)


def test_tool_directories_outside_the_project_pass(tmp_path):
    _project, wrappers, firsts = project_like(tmp_path)

    assert sandbox.tool_binds(ToolEnv((firsts,), (), (wrappers,))) == [firsts.parent, wrappers]


# --- inside real bubblewrap ----------------------------------------------------------------


def test_a_private_file_beside_the_wrapper_directory_is_not_readable(tmp_path, work):
    project, wrappers, firsts = project_like(tmp_path)
    tools = ToolEnv((firsts,), (), (wrappers,))
    if not sandbox.available():
        pytest.skip("needs bubblewrap")

    peek = sandbox.run_command(work, tools, f"cat {project}/.local/secret.txt")
    parent = sandbox.run_command(work, tools, f"ls -A {project}/.local")
    listing = sandbox.run_command(work, tools, f"ls {wrappers}")

    assert peek.exit_code != 0 and "TRAP-CATALOGUE" not in peek.stdout + peek.stderr
    assert parent.stdout.split() == ["pyenvs"]  # only the bound directory, not its neighbours
    assert listing.exit_code == 0 and "python-x" in listing.stdout


def test_the_project_is_invisible_with_the_real_tool_configuration(real_tools, work):
    for tail in real_tools.tail_dirs:
        assert ROOT not in (tail, *tail.parents), f"{tail} lies inside the project"

    out = sandbox.run_command(
        work,
        real_tools,
        f"ls -A {ROOT}/.local; cat {ROOT}/.local/tools.json; ls {ROOT}/src; ls {ROOT}/runs; "
        f"ls {ROOT}/docs; echo PATH=$PATH; echo done",
    )

    text = out.stdout + out.stderr
    assert "done" in out.stdout
    for private in ("tools.json", "calib_rules", "new_families", "scripts", "scratch", "bin_dirs"):
        assert private not in out.stdout, private
    assert str(ROOT) not in out.stdout.split("PATH=")[1]
    assert text.count("No such file or directory") >= 5


def test_the_tools_still_run_with_the_real_tool_configuration(real_tools, work):
    out = sandbox.run_command(
        work,
        real_tools,
        "samtools --version | head -1; bcftools --version | head -1; bedtools --version",
    )

    assert out.exit_code == 0
    assert "samtools" in out.stdout and "bcftools" in out.stdout and "bedtools" in out.stdout


def test_every_python_wrapper_in_the_tool_configuration_still_runs(real_tools, work):
    names = sorted(
        p.name
        for d in real_tools.tail_dirs
        if d.is_dir()
        for p in d.iterdir()
        if p.name.startswith("python-")
    )
    if not names:
        pytest.skip("no python-* wrappers configured")

    for name in names:
        out = sandbox.run_command(
            work, real_tools, f'{name} -c "import sys; print(sys.version_info[0])"'
        )
        assert out.exit_code == 0 and out.stdout.strip() == "3", (name, out.stderr)
