"""Write sanitised, explicit package lists of the environments the benchmark runs in.

    python tools/env_lock.py [--tools-json FILE] [--out envs/locks] [--check]

For every conda environment named in tools.json (a tail directory `<env>/bin` with `conda-meta`
beside it) the file `<env>.tsv` has one row per installed package: name, version, build, url and
sha256, read from conda-meta (what `conda list --explicit` gives, without paths). Besides:

    venv-pip-freeze.txt   `pip freeze --all` of the interpreter that runs this script; an
                          editable install of this project is reduced to a comment
    tool-versions.txt     `--version` line and sha256 of samtools, bcftools and bedtools, sha256
                          of libbz2, the versions of bwrap and Python

No file holds a filesystem path (the script refuses to write one), so the lists can be published.
`--check` writes nothing and exits 1 when the environments differ from the files in --out: a job
runs it before its first trial, so a package that moved after the lock was taken stops the job.
Packages that were pip-installed inside a conda environment are not in conda-meta and not listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ENV_VAR = "BIOWARDEN_TOOLS"
BWRAP_ENV_VAR = "BIOWARDEN_BWRAP"
HEADER = ("name", "version", "build", "url", "sha256")
TOOLS = ("samtools", "bcftools", "bedtools")
ENV_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
LOADER_NOISE = re.compile(r"no version information available")
# a path in a file that is going to be published: '/' after a space, '=' or the start of a line,
# followed by a name and another '/'
PATH_LIKE = re.compile(r"(?:^|[\s=\"'(])/(?!/)[^\s/]+/")
EDITABLE_NOTE = "# the project itself is installed in editable mode (not pinned here)"
TOKEN = re.compile(r"/t/[^/]+/")  # a private conda channel carries its token in the url
CREDENTIALS = re.compile(r"//[^/@]*@")


class LockError(Exception):
    """An environment cannot be listed, or the output would not be safe to publish."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- conda environments --------------------------------------------------------------------------


def conda_envs(tools: dict) -> dict[str, Path]:
    """name -> prefix of every conda environment among the tail directories of tools.json."""
    envs: dict[str, Path] = {}
    for entry in tools.get("tail_dirs", ()):
        path = Path(entry)
        if path.name != "bin" or not (path.parent / "conda-meta").is_dir():
            continue
        name = path.parent.name
        if not ENV_NAME.fullmatch(name):
            raise LockError(f"environment name {name!r} is not usable as a file name")
        if name in envs and envs[name] != path.parent:
            raise LockError(f"two environments are called {name!r}")
        envs[name] = path.parent
    return envs


def clean_url(info: dict) -> str:
    """The url a package came from, without credentials or a local path."""
    url = str(info.get("url") or "")
    if not url:
        channel, fn = str(info.get("channel") or ""), str(info.get("fn") or "")
        url = f"{channel.rstrip('/')}/{fn}" if channel.startswith("http") and fn else ""
    if url.startswith(("http://", "https://")):
        return TOKEN.sub("/", CREDENTIALS.sub("//", url))
    fn = str(info.get("fn") or info.get("name") or "unknown")
    return f"local:{fn}"  # file:// channels and bare paths say where it was, not what it is


def package_rows(prefix: Path) -> list[tuple[str, ...]]:
    rows = []
    for meta in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            info = json.loads(meta.read_text())
            name, version, build = info["name"], info["version"], info["build"]
        except (OSError, ValueError, KeyError) as exc:
            raise LockError(f"{meta.name}: not a conda package record ({exc!r})") from exc
        digest = info.get("sha256") or (f"md5:{info['md5']}" if info.get("md5") else "")
        if not digest:
            raise LockError(f"{meta.name}: no sha256 or md5")
        rows.append((str(name), str(version), str(build), clean_url(info), str(digest)))
    return sorted(rows)


def tsv(rows: Sequence[Sequence[str]]) -> str:
    return "\n".join("\t".join(row) for row in [HEADER, *rows]) + "\n"


# -- pip ------------------------------------------------------------------------------------------


def sanitise_freeze(text: str) -> str:
    """`pip freeze --all` without paths: a local or editable install becomes one comment."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue  # pip's own remarks about editable installs name a path or the project
        if line.startswith(("-e ", "--editable")):
            note = EDITABLE_NOTE
        elif " @ file:" in line or " @ /" in line:
            note = f"# {line.split(' @ ', 1)[0]} is installed from a local path (not pinned here)"
        else:
            out.append(line)
            continue
        if note not in out:
            out.append(note)
    return "\n".join(out) + "\n"


def pip_freeze(python: str) -> str:
    proc = subprocess.run(
        [python, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise LockError(f"pip freeze failed: {proc.stderr.strip()[:200]}")
    return sanitise_freeze(proc.stdout)


# -- tools ----------------------------------------------------------------------------------------


def version_line(argv: list[str], env: dict[str, str]) -> str:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LockError(f"{Path(argv[0]).name} --version failed: {exc!r}") from exc
    for stream in (proc.stdout, proc.stderr):
        for line in stream.splitlines():
            if line.strip() and not LOADER_NOISE.search(line):
                return line.strip()
    raise LockError(f"{Path(argv[0]).name} --version printed nothing")


def find_tool(name: str, dirs: Sequence[Path]) -> Path:
    for directory in dirs:
        if (directory / name).is_file():
            return directory / name
    raise LockError(f"{name} is in none of the bin_dirs of tools.json")


def find_bwrap(root: Path) -> Path | None:
    for candidate in (os.environ.get(BWRAP_ENV_VAR), root / ".local" / "bin" / "bwrap"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    found = shutil.which("bwrap")
    return Path(found) if found else None


def tool_versions(tools: dict, root: Path = ROOT) -> str:
    bins = [Path(p) for p in tools.get("bin_dirs", ())]
    libs = [Path(p) for p in tools.get("lib_dirs", ())]
    env = {"PATH": os.environ.get("PATH", "/usr/bin"), "LC_ALL": "C"}
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(map(str, libs))
    lines, hashes = [], []
    for name in TOOLS:
        exe = find_tool(name, bins)
        lines.append(f"{name}: {version_line([str(exe), '--version'], env)}")
        hashes.append(f"sha256 {name}: {sha256_file(exe)}")
    bwrap = find_bwrap(root)
    lines.append(f"bwrap: {version_line([str(bwrap), '--version'], env) if bwrap else 'not found'}")
    lines.append(f"python: Python {platform.python_version()}")
    for directory in libs:
        for lib in sorted(directory.glob("libbz2.so*")):
            if lib.is_file() and not lib.is_symlink():
                hashes.append(f"sha256 {lib.name}: {sha256_file(lib)}")
    return "\n".join(lines + hashes) + "\n"


# -- the lock files -------------------------------------------------------------------------------


def generate(tools_json: Path, python: str | None = None, root: Path = ROOT) -> dict[str, str]:
    """{file name: content} of every lock file for the environments in `tools_json`."""
    try:
        tools = json.loads(tools_json.read_text())
    except (OSError, ValueError) as exc:
        raise LockError(f"{tools_json}: {exc}") from exc
    envs = conda_envs(tools)
    if not envs:
        raise LockError(f"{tools_json} names no conda environment among its tail_dirs")
    files = {f"{name}.tsv": tsv(package_rows(prefix)) for name, prefix in sorted(envs.items())}
    files["venv-pip-freeze.txt"] = pip_freeze(python or sys.executable)
    files["tool-versions.txt"] = tool_versions(tools, root)
    for name, text in files.items():
        hit = next((m.group(0).strip() for m in map(PATH_LIKE.search, text.splitlines()) if m), "")
        if hit:
            raise LockError(f"{name} would hold a filesystem path ({hit}...); not written")
    return files


def differences(files: dict[str, str], out: Path) -> list[str]:
    problems = []
    for name, text in files.items():
        path = out / name
        if not path.is_file():
            problems.append(f"{name}: not in {out.name}/ (run tools/env_lock.py)")
        elif path.read_text() != text:
            old = set(path.read_text().splitlines())
            now = set(text.splitlines())
            problems.append(
                f"{name}: differs ({len(now - old)} lines new, {len(old - now)} gone), "
                f"first: {next(iter(sorted(now - old) or sorted(old - now)), '')[:120]}"
            )
    stale = sorted(p.name for p in out.glob("*") if p.is_file() and p.name not in files)
    return problems + [f"{name}: no longer produced by tools.json" for name in stale]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="env_lock.py", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    default_tools = os.environ.get(TOOLS_ENV_VAR) or str(ROOT / ".local" / "tools.json")
    p.add_argument(
        "--tools-json", default=default_tools, help=f"default: ${TOOLS_ENV_VAR} or .local"
    )
    p.add_argument("--out", default=str(ROOT / "envs" / "locks"), help="default: envs/locks")
    p.add_argument("--check", action="store_true", help="compare, write nothing, exit 1 on a diff")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = Path(args.out)
    try:
        files = generate(Path(args.tools_json))
    except LockError as exc:
        print(f"env_lock: {exc}", file=sys.stderr)
        return 2
    if args.check:
        problems = differences(files, out) if out.is_dir() else [f"{out}: not a directory"]
        for problem in problems:
            print(f"env_lock: {problem}", file=sys.stderr)
        print(
            f"env_lock: {'DIFFERS' if problems else 'unchanged'}: {len(files)} files vs {out.name}/"
        )
        return 1 if problems else 0
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"{name:<22}{len(text.splitlines()):>5} lines  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
