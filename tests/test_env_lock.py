"""tools/env_lock.py: sanitised package lists of the environments, and the check against them."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "/hom" + "e/someone"  # a path that must never reach a lock file


@pytest.fixture(scope="module")
def el():
    spec = importlib.util.spec_from_file_location("env_lock", ROOT / "tools" / "env_lock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["env_lock"] = module
    spec.loader.exec_module(module)
    return module


def package(prefix: Path, name: str, version: str = "1.0", build: str = "h0_0", **more) -> None:
    folder = prefix / "conda-meta"
    folder.mkdir(parents=True, exist_ok=True)
    fn = f"{name}-{version}-{build}.conda"
    record = {
        "name": name,
        "version": version,
        "build": build,
        "build_number": 0,
        "channel": "https://conda.anaconda.org/conda-forge/linux-64",
        "fn": fn,
        "url": f"https://conda.anaconda.org/conda-forge/linux-64/{fn}",
        "sha256": hashlib.sha256(fn.encode()).hexdigest(),
        "md5": "0" * 32,
        # the records of conda-meta also say where things are; none of it may reach a lock file
        "extracted_package_dir": f"{prefix}/pkgs/{name}",
        "link": {"source": f"{prefix}/pkgs/{name}", "type": 1},
        "files": [f"lib/{name}.so"],
    }
    stem = f"{name}-{version}-{build}".replace("/", "_")
    (folder / f"{stem}.json").write_text(json.dumps(record | more))


def make_tools(tmp_path: Path, envs=("bio-a", "bio-b")) -> Path:
    """A tools.json with real-looking directories: tools, a library, two environments."""
    bins, libs = tmp_path / "bin", tmp_path / "lib"
    bins.mkdir()
    libs.mkdir()
    for name, line in (
        ("samtools", "samtools 1.24\\nUsing htslib 1.24"),
        ("bcftools", "bcftools 1.24"),
        ("bedtools", "bedtools v2.31.1"),
    ):
        exe = bins / name
        exe.write_text(f"#!/bin/sh\nprintf '{line}\\n'\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (libs / "libbz2.so.1.0.8").write_bytes(b"the library")
    (libs / "libbz2.so.1").symlink_to("libbz2.so.1.0.8")  # a link has no content of its own
    tails = [tmp_path / "wrappers"]
    for env in envs:
        package(tmp_path / "conda" / env, "zlib", "1.3", "h1_0")
        package(tmp_path / "conda" / env, "bzip2", "1.0.8", "h2_0")
        (tmp_path / "conda" / env / "bin").mkdir()
        tails.append(tmp_path / "conda" / env / "bin")
    tools = {
        "bin_dirs": [str(bins)],
        "lib_dirs": [str(libs)],
        "tail_dirs": [str(p) for p in tails],
    }
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(tools))
    return path


@pytest.fixture
def no_pip(el, monkeypatch):
    monkeypatch.setattr(el, "pip_freeze", lambda python: "pytest==9.0.0\n")
    monkeypatch.setattr(el, "find_bwrap", lambda root: None)


# -- urls and rows --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"url": "https://conda.anaconda.org/c/linux-64/a-1.conda"}, "https://conda.anaconda.org/c/linux-64/a-1.conda"),
        ({"url": "https://user:secret" + "@host.test/c/a-1.conda"}, "https://host.test/c/a-1.conda"),
        ({"url": "https://conda.anaconda.org/t/tk-123abc/private/linux-64/a-1.conda"}, "https://conda.anaconda.org/private/linux-64/a-1.conda"),
        ({"url": f"file://{PRIVATE}/pkgs/a-1.conda", "fn": "a-1.conda"}, "local:a-1.conda"),
        ({"url": "/projects/x/pkgs/a-1.conda", "fn": "a-1.conda"}, "local:a-1.conda"),
        ({"channel": "https://conda.anaconda.org/c/linux-64/", "fn": "a-1.conda"}, "https://conda.anaconda.org/c/linux-64/a-1.conda"),
        ({"channel": "/some/dir", "fn": "a-1.conda"}, "local:a-1.conda"),
        ({"name": "a"}, "local:a"),
    ],
)  # fmt: skip
def test_urls_lose_credentials_tokens_and_local_paths(el, info, expected):
    assert el.clean_url(info) == expected


def test_package_rows_are_sorted_and_hold_no_path(el, tmp_path):
    prefix = tmp_path / "env"
    package(prefix, "zlib", "1.3", "h1_0")
    package(prefix, "bzip2", "1.0.8", "h2_0")
    package(prefix, "bzip2", "1.0.6", "h9_0")  # two versions are listed, the older one first
    package(prefix, "old", "1", "0", sha256="", md5="a" * 32)  # no sha256: the md5 stands in

    rows = el.package_rows(prefix)

    assert [r[:3] for r in rows] == [
        ("bzip2", "1.0.6", "h9_0"),
        ("bzip2", "1.0.8", "h2_0"),
        ("old", "1", "0"),
        ("zlib", "1.3", "h1_0"),
    ]
    assert rows[2][4] == "md5:" + "a" * 32
    text = el.tsv(rows)
    assert text.splitlines()[0] == "name\tversion\tbuild\turl\tsha256"
    assert str(tmp_path) not in text and not el.PATH_LIKE.search(text)


def test_a_record_without_a_checksum_or_that_is_not_a_package_is_an_error(el, tmp_path):
    prefix = tmp_path / "env"
    package(prefix, "a", sha256="", md5="")
    with pytest.raises(el.LockError, match="no sha256 or md5"):
        el.package_rows(prefix)
    (prefix / "conda-meta" / "a-1.0-h0_0.json").write_text("{}")
    with pytest.raises(el.LockError, match="not a conda package record"):
        el.package_rows(prefix)
    (prefix / "conda-meta" / "a-1.0-h0_0.json").write_text("not json")
    with pytest.raises(el.LockError, match="not a conda package record"):
        el.package_rows(prefix)


def test_environments_are_the_conda_prefixes_among_the_tail_directories(el, tmp_path):
    tools = json.loads(make_tools(tmp_path).read_text())

    envs = el.conda_envs(tools)

    assert sorted(envs) == ["bio-a", "bio-b"]  # the wrapper directory is not one
    assert envs["bio-a"] == tmp_path / "conda" / "bio-a"
    bad = tmp_path / "conda" / "bad name"
    package(bad, "x")
    (bad / "bin").mkdir()
    with pytest.raises(el.LockError, match="not usable as a file name"):
        el.conda_envs({"tail_dirs": [str(bad / "bin")]})
    other = tmp_path / "elsewhere" / "bio-a"
    package(other, "x")
    (other / "bin").mkdir()
    both = {"tail_dirs": [*tools["tail_dirs"], str(other / "bin")]}
    with pytest.raises(el.LockError, match="two environments are called 'bio-a'"):
        el.conda_envs(both)


# -- pip freeze -----------------------------------------------------------------------------------


def test_pip_freeze_loses_editable_installs_and_local_paths(el):
    raw = (
        "anthropic==2.3.0\n"
        "# Editable Git install with no remote (some-project==0.0.1)\n"
        f"-e {PRIVATE}/work/some-project\n"
        f"local-thing @ file://{PRIVATE}/wheels/local_thing-1.0-py3-none-any.whl\n"
        "other @ /opt/wheels/other.whl\n"
        "openai==3.40.0\n"
        "-e git+https://example.invalid/x.git#egg=x\n"
        "\n"
    )

    text = el.sanitise_freeze(raw)

    assert text.splitlines() == [
        "anthropic==2.3.0",
        el.EDITABLE_NOTE,
        "# local-thing is installed from a local path (not pinned here)",
        "# other is installed from a local path (not pinned here)",
        "openai==3.40.0",
    ]
    assert not el.PATH_LIKE.search(text)


# -- tools ----------------------------------------------------------------------------------------


def test_tool_versions_have_the_version_line_and_the_hash_of_each_program(el, tmp_path, no_pip):
    tools = json.loads(make_tools(tmp_path).read_text())

    text = el.tool_versions(tools, tmp_path)

    lines = text.splitlines()
    assert lines[:5] == [
        "samtools: samtools 1.24",
        "bcftools: bcftools 1.24",
        "bedtools: bedtools v2.31.1",
        "bwrap: not found",
        f"python: Python {sys.version.split()[0]}",
    ]
    exe = Path(tools["bin_dirs"][0]) / "samtools"
    assert f"sha256 samtools: {hashlib.sha256(exe.read_bytes()).hexdigest()}" in lines
    assert f"sha256 libbz2.so.1.0.8: {hashlib.sha256(b'the library').hexdigest()}" in lines
    assert not any("libbz2.so.1:" in line for line in lines)  # the link is not a second library
    assert not el.PATH_LIKE.search(text)


def test_bwrap_is_found_through_the_environment_variable(el, tmp_path, monkeypatch):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\necho 'bubblewrap 9.9.9'\n")
    bwrap.chmod(0o755)
    monkeypatch.setenv(el.BWRAP_ENV_VAR, str(bwrap))
    tools = json.loads(make_tools(tmp_path).read_text())

    assert "bwrap: bubblewrap 9.9.9" in el.tool_versions(tools, tmp_path).splitlines()


def test_a_tool_that_is_missing_or_silent_is_an_error(el, tmp_path):
    tools = json.loads(make_tools(tmp_path).read_text())
    (Path(tools["bin_dirs"][0]) / "bedtools").unlink()
    with pytest.raises(el.LockError, match="bedtools is in none of the bin_dirs"):
        el.tool_versions(tools, tmp_path)
    quiet = Path(tools["bin_dirs"][0]) / "bedtools"
    quiet.write_text("#!/bin/sh\nexit 0\n")
    quiet.chmod(0o755)
    with pytest.raises(el.LockError, match="printed nothing"):
        el.tool_versions(tools, tmp_path)


# -- the files and the check ----------------------------------------------------------------------


def test_generate_makes_one_file_per_environment_and_the_two_others(el, tmp_path, no_pip):
    files = el.generate(make_tools(tmp_path), root=tmp_path)

    assert sorted(files) == ["bio-a.tsv", "bio-b.tsv", "tool-versions.txt", "venv-pip-freeze.txt"]
    assert files["bio-a.tsv"] == files["bio-b.tsv"]
    assert files["venv-pip-freeze.txt"] == "pytest==9.0.0\n"
    assert all(not el.PATH_LIKE.search(text) for text in files.values())


def test_nothing_with_a_filesystem_path_is_written(el, tmp_path, no_pip):
    tools = make_tools(tmp_path)
    package(tmp_path / "conda" / "bio-a", "odd", build=f"{PRIVATE}/build")

    with pytest.raises(el.LockError, match="would hold a filesystem path"):
        el.generate(tools, root=tmp_path)


def test_a_tools_json_without_conda_environments_is_an_error(el, tmp_path, no_pip):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"bin_dirs": [], "tail_dirs": [str(tmp_path / "wrappers")]}))
    with pytest.raises(el.LockError, match="names no conda environment"):
        el.generate(path, root=tmp_path)
    with pytest.raises(el.LockError, match="tools.json"):
        el.generate(tmp_path / "missing" / "tools.json", root=tmp_path)


def test_main_writes_the_files_and_check_finds_every_kind_of_drift(
    el, tmp_path, no_pip, capsys, monkeypatch
):
    tools, out = make_tools(tmp_path), tmp_path / "locks"

    assert el.main(["--tools-json", str(tools), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "bio-a.tsv", "bio-b.tsv", "tool-versions.txt", "venv-pip-freeze.txt",
    ]  # fmt: skip
    capsys.readouterr()
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 0
    assert "unchanged" in capsys.readouterr().out

    package(tmp_path / "conda" / "bio-a", "numpy", "2.0", "h3_0")  # a package is installed
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 1
    err = capsys.readouterr().err
    assert "bio-a.tsv: differs (1 lines new, 0 gone)" in err and "numpy" in err
    (tmp_path / "conda" / "bio-a" / "conda-meta" / "numpy-2.0-h3_0.json").unlink()
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 0

    monkeypatch.setattr(el, "pip_freeze", lambda python: "pytest==9.0.0\nrequests==2.0\n")
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 1
    assert "venv-pip-freeze.txt: differs" in capsys.readouterr().err
    monkeypatch.setattr(el, "pip_freeze", lambda python: "pytest==9.0.0\n")

    (out / "bio-b.tsv").unlink()
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 1
    assert "bio-b.tsv: not in locks/" in capsys.readouterr().err
    (out / "bio-b.tsv").write_text((out / "bio-a.tsv").read_text())
    (out / "bio-c.tsv").write_text("stale\n")
    assert el.main(["--tools-json", str(tools), "--out", str(out), "--check"]) == 1
    assert "bio-c.tsv: no longer produced" in capsys.readouterr().err
    assert el.main(["--tools-json", str(tmp_path / "none.json"), "--out", str(out)]) == 2
    assert el.main(["--tools-json", str(tools), "--out", str(tmp_path / "nodir"), "--check"]) == 1


# -- the files in the repository ------------------------------------------------------------------


LOCKS = ROOT / "envs" / "locks"


def test_the_committed_lock_files_are_complete_and_hold_no_private_path(el):
    names = sorted(p.name for p in LOCKS.iterdir())
    assert {"venv-pip-freeze.txt", "tool-versions.txt"} <= set(names)
    assert len([n for n in names if n.endswith(".tsv")]) >= 4  # the four pack environments

    for path in LOCKS.iterdir():
        text = path.read_text()
        assert not el.PATH_LIKE.search(text), path.name
        assert "/gpfs" not in text and "/projects/" not in text, path.name
        if path.suffix == ".tsv":
            lines = text.splitlines()
            assert lines[0] == "\t".join(el.HEADER)
            assert lines[1:] == sorted(lines[1:]) and len(lines) > 100
            for line in lines[1:]:
                name, version, build, url, digest = line.split("\t")
                assert url.startswith("https://"), line
                assert re.fullmatch(r"[0-9a-f]{64}|md5:[0-9a-f]{32}", digest), line


def test_the_locked_sdk_versions_satisfy_the_bench_extra_of_pyproject():
    packaging = pytest.importorskip("packaging.requirements")
    version = pytest.importorskip("packaging.version")
    tomllib = pytest.importorskip("tomllib")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    locked = dict(
        line.split("==") for line in (LOCKS / "venv-pip-freeze.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    )  # fmt: skip

    for extra in ("bench", "dev"):
        for text in extras[extra]:
            req = packaging.Requirement(text)
            if req.name in ("anthropic", "openai"):
                assert req.specifier.contains(version.Version(locked[req.name])), (extra, text)
    assert set(extras["bench"]) <= set(extras["dev"])  # the tests import the runner
