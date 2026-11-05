"""API-key discovery and hygiene checks.

A key is looked up in the process environment, then in the first dotenv file that defines it:
``$SCIENCE_HARNESS_ENV``, ``<project>/.env``, ``~/.config/science-harness/keys.env``. Values leave
this module only through ``get_key``; the audit and the CLI report names, sources and file modes,
never values. Keys are never exported to ``os.environ``: Claude Code prefers an exported
``ANTHROPIC_API_KEY`` over the subscription login, so an export would silently move its billing to
the API.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ENV_FILE_VAR = "SCIENCE_HARNESS_ENV"
BARE_ANTHROPIC = "ANTHROPIC_API_KEY"

# Variable names per provider, preferred first. The prefixed name is the one to put in .env.
PROVIDER_VARS = {
    "anthropic": ("HARNESS_ANTHROPIC_API_KEY", BARE_ANTHROPIC),
    "deepseek": ("HARNESS_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    "zai": ("HARNESS_ZAI_API_KEY", "ZAI_API_KEY"),
    "openrouter": ("HARNESS_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
}


@dataclass(frozen=True)
class KeyStatus:
    provider: str
    variable: str | None  # which variable supplied the key; None when unset
    source: str | None  # "environment" or a file path


@dataclass(frozen=True)
class FileStatus:
    path: Path
    mode: int | None  # permission bits; None when the file does not exist
    git_ignored: bool | None  # None: not in a git work tree, or git is unavailable
    variables: tuple[str, ...] = ()  # names defined in the file


def project_root(start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").exists() or (directory / ".git").exists():
            return directory
    return None


def env_files(start: Path | None = None, environ: Mapping[str, str] | None = None) -> list[Path]:
    environ = os.environ if environ is None else environ
    files: list[Path] = []
    if explicit := environ.get(ENV_FILE_VAR):
        files.append(Path(explicit).expanduser())
    if (root := project_root(start)) is not None:
        files.append(root / ".env")
    files.append(Path.home() / ".config" / "science-harness" / "keys.env")
    return files


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[name] = value
    return values


def read_env_file(path: Path) -> dict[str, str]:
    try:
        return parse_dotenv(path.read_text())
    except (OSError, UnicodeDecodeError):
        return {}


def _lookup(
    provider: str, environ: Mapping[str, str], files: Sequence[Path]
) -> tuple[str, str, str] | None:
    """(value, variable, source) of the first non-empty match, or None."""
    names = PROVIDER_VARS[provider]
    for name in names:
        if environ.get(name):
            return environ[name], name, "environment"
    for path in files:
        values = read_env_file(path)
        for name in names:
            if values.get(name):
                return values[name], name, str(path)
    return None


def get_key(
    provider: str, *, environ: Mapping[str, str] | None = None, files: Sequence[Path] | None = None
) -> str | None:
    environ = os.environ if environ is None else environ
    files = env_files() if files is None else files
    found = _lookup(provider, environ, files)
    return found[0] if found else None


def _git_ignored(path: Path) -> bool | None:
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return {0: True, 1: False}.get(proc.returncode)  # 128: not a git work tree


def audit(
    *, environ: Mapping[str, str] | None = None, files: Sequence[Path] | None = None
) -> tuple[list[KeyStatus], list[FileStatus]]:
    environ = os.environ if environ is None else environ
    files = env_files() if files is None else files
    keys = []
    for provider in PROVIDER_VARS:
        found = _lookup(provider, environ, files)
        if found is None:
            keys.append(KeyStatus(provider, None, None))
        else:
            _, variable, source = found  # the value is dropped here on purpose
            keys.append(KeyStatus(provider, variable, source))
    statuses = []
    for path in files:
        if not path.is_file():
            statuses.append(FileStatus(path, None, None))
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        names = tuple(read_env_file(path))
        statuses.append(FileStatus(path, mode, _git_ignored(path), names))
    return keys, statuses


def problems(files: Sequence[FileStatus], environ: Mapping[str, str] | None = None) -> list[str]:
    environ = os.environ if environ is None else environ
    found = []
    for item in files:
        if item.mode is not None and item.mode & 0o077:
            found.append(f"{item.path}: mode {item.mode:o} is open to group/other; chmod 600")
        if item.git_ignored is False:
            found.append(f"{item.path}: not ignored by git; add it to .gitignore")
        if BARE_ANTHROPIC in item.variables:
            found.append(f"{item.path}: defines {BARE_ANTHROPIC}; use HARNESS_ANTHROPIC_API_KEY")
    if BARE_ANTHROPIC in environ:
        found.append(f"{BARE_ANTHROPIC} is exported: Claude Code would bill the API, not the plan")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="science-harness-keys",
        description="Show where API keys come from. Prints names and sources, never values.",
    )
    parser.add_argument("--env-file", type=Path, help="dotenv file to search before the defaults")
    args = parser.parse_args(argv)
    files = env_files()
    if args.env_file is not None:
        files.insert(0, args.env_file)
    keys, statuses = audit(files=files)

    print("keys")
    for key in keys:
        state = f"set  {key.variable}  {key.source}" if key.variable else "unset"
        print(f"  {key.provider:<11}{state}")
    print("files")
    for item in statuses:
        if item.mode is None:
            print(f"  {item.path}  missing")
            continue
        ignored = {True: "git-ignored", False: "NOT git-ignored", None: "not in a git repo"}
        print(f"  {item.path}  mode {item.mode:o}  {ignored[item.git_ignored]}")
    issues = problems(statuses)
    print("problems: none" if not issues else "problems:")
    for issue in issues:
        print(f"  {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
