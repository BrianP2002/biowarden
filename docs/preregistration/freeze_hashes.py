"""Print SHA-256 digests of the files a run depends on; freeze and verify a stage-2 snapshot.

Stage 1 (legacy mode; the algorithm is unchanged, so `freeze.txt` and `freeze-rerun.txt` still
reproduce from the snapshot they were taken from)

    python docs/preregistration/freeze_hashes.py [ROOT]

Stage 2 (docs/preregistration/stage2-draft.md)

    python docs/preregistration/freeze_hashes.py --stage2 --code-root runs/stage2-code \\
        --repo-root . [--sets FILE ...] [--write FILE | --verify FILE]

`--code-root` is the snapshot the jobs import (`<code-root>/src/biowarden`) and `--repo-root`
the checkout that is committed and tagged. There is no default for either, and the two `src/` trees
must be the same files. The manifest has one `sha256  path` line for every file of the package
(`__pycache__` and `.pyc` excluded), `pyproject.toml`, `tools/stage2_job.sbatch` and the two
programs it runs (`tools/preflight.py`, `tools/env_lock.py`), every file of `envs/locks` and
every `--sets` file (freeze 2: the sets file with its seed registry, the plan
manifest, the draft; a path is relative to `--repo-root`). It also has one digest per group and a
root digest over all the lines. `code` equals the `code_digest` the runner stamps on each record.

Exit status 2: an unknown flag, a missing root, a root without `src/biowarden`, a group
without files (printed as MISSING, never as the digest of empty input), a group below its expected
size, a `src/` that differs from the snapshot, or a mismatch under `--verify`. Nothing is written
then, and `--write` does not replace a different file unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

# -- stage 1: the digests of the published freeze files; do not change -------------------------

LEGACY_GROUPS = {
    "scorer": ["src/biowarden/bench/score.py"],
    "analysis": ["src/biowarden/bench/analysis.py"],
    "stage1": ["src/biowarden/bench/stage1.py"],
    "engine": ["src/biowarden/engine/*.py", "src/biowarden/engine/rules/*.py"],
    "specs": ["src/biowarden/engine/specs/*.yaml"],
    "promotion": ["src/biowarden/engine/specs/promotion.json"],
    "tasks": [
        "src/biowarden/bench/tasks/families.py",
        "src/biowarden/bench/tasks/common.py",
        "src/biowarden/bench/tasks/instance.py",
        "src/biowarden/bench/tasks/extra/f07_regions_vs_targets.py",
        "src/biowarden/bench/tasks/extra/f10_indel_bed.py",
        "src/biowarden/bench/tasks/extra/f13_phred_offset.py",
        "src/biowarden/bench/tasks/extra/f15_log_scale.py",
    ],
    "runner": ["src/biowarden/bench/runner/*.py"],
}


def legacy_digest(root: Path, patterns: list[str]) -> str:
    h = hashlib.sha256()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def legacy_main(root: Path) -> int:
    for name, patterns in LEGACY_GROUPS.items():
        print(f"{name:<10}{legacy_digest(root, patterns)}")
    return 0


# -- stage 2 ------------------------------------------------------------------------------------

PACKAGE = "biowarden"
SRC = f"src/{PACKAGE}"
PYPROJECT = "pyproject.toml"
JOB_TEMPLATE = "tools/stage2_job.sbatch"
# the template, and the programs it runs to cut a job into calls, to verify it and to check the
# environments: they decide what is run, so they are frozen with it
CONFIG_FILES = (PYPROJECT, JOB_TEMPLATE, "tools/preflight.py", "tools/env_lock.py")
LOCKS = "envs/locks"
FORMAT = 1
HEADER = (
    f"# stage-2 freeze (freeze_hashes.py, format {FORMAT}); a group digest is the sha256 of its "
    "'sha256  path' lines, sorted by path and joined by newlines"
)
SEPARATOR = "--"
GROUPS = (
    "scorer", "analysis", "engine", "specs", "promotion", "tasks", "runner",
    "support", "config", "env", "sets",
)  # fmt: skip
# Fewest files a group may have (the tree at the time of freeze 1). `tasks` is the 110 builders
# and helpers of bench/tasks plus `family_rules.json` (also in REQUIRED), 111 files: one builder
# fewer must stop the freeze. `sets` has no floor: it is exactly the files named with --sets, and
# each of them must exist.
EXPECTED = {
    "scorer": 1,
    "analysis": 4,
    "engine": 80,
    "specs": 51,
    "promotion": 1,
    "tasks": 111,
    "runner": 14,
    "support": 1,
    "config": len(CONFIG_FILES),
    "env": 6,
}
ANALYSIS_FILES = (
    "bench/analysis.py",
    "bench/pilot.py",
    "bench/stage2.py",
    "bench/admission/replay.py",
)
REQUIRED = tuple(
    f"{SRC}/{name}"
    for name in (
        "bench/score.py",
        *ANALYSIS_FILES,
        "bench/tasks/tools.py",
        "bench/tasks/family_rules.json",
        "engine/specs/promotion.json",
    )
) + CONFIG_FILES  # fmt: skip
SHOWN = 8  # differing files listed in a mismatch report


class Refused(Exception):
    """The freeze cannot be written or verified; `problems` says why (one line each)."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_of(path: str, digest: str) -> str:
    return f"{digest}  {path}"


def lines_digest(files: Iterable[tuple[str, str]]) -> str:
    """sha256 of the 'sha256  path' lines of `files` (path, sha256), sorted by path."""
    return sha256("\n".join(line_of(path, digest) for path, digest in sorted(files)).encode())


def group_of(path: str) -> str:
    """The group a manifest path belongs to (every file of the package is in exactly one)."""
    if path.startswith(f"{SRC}/"):
        rel = path[len(SRC) + 1 :]
        if rel == "bench/score.py":
            return "scorer"
        if rel in ANALYSIS_FILES:
            return "analysis"
        if rel == "engine/specs/promotion.json":
            return "promotion"
        if rel.startswith("engine/specs/"):
            return "specs"
        if rel.startswith(("engine/", "ledger/", "contracts/")):
            return "engine"
        if rel.startswith("bench/tasks/"):
            return "tasks"
        if rel.startswith("bench/runner/"):
            return "runner"
        return "support"
    if path in CONFIG_FILES:
        return "config"
    if path.startswith(f"{LOCKS}/"):
        return "env"
    return "sets"


def tree_digests(top: Path, prefix: str) -> dict[str, str]:
    """{`prefix/relative path`: sha256} of every file under `top`, caches excluded.

    A symbolic link anywhere in a frozen tree is refused: the bytes it stands for are not under
    the directory that is being frozen.
    """
    out: dict[str, str] = {}
    for here, dirs, files in os.walk(top):
        base = Path(here)
        for name in (*dirs, *files):
            if (base / name).is_symlink():
                raise Refused(
                    [f"{(base / name).relative_to(top).as_posix()} in {top}: symbolic link"]
                )
        dirs[:] = [d for d in sorted(dirs) if d != "__pycache__"]
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            path = base / name
            rel = path.relative_to(top).as_posix()
            out[f"{prefix}/{rel}"] = sha256(path.read_bytes())
    return out


def one_file(root: Path, rel: str) -> dict[str, str]:
    path = root / rel
    if path.is_symlink():
        raise Refused([f"{rel}: symbolic link"])
    return {rel: sha256(path.read_bytes())} if path.is_file() else {}


def extra_file(repo: Path, name: str) -> tuple[str, str]:
    """(manifest path, sha256) of a --sets file: inside the repository, not empty."""
    path = Path(name) if Path(name).is_absolute() else repo / name
    if not path.is_file():
        raise Refused([f"sets: MISSING {name}"])
    if path.is_symlink():
        raise Refused([f"sets: {name} is a symbolic link"])
    try:
        rel = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        raise Refused([f"sets: {name} is not inside --repo-root"]) from None
    data = path.read_bytes()
    if not data.strip():
        raise Refused([f"sets: {rel} is empty"])
    return rel, sha256(data)


def differences(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """One line per file that was changed, is gone, or is new (old = recorded or snapshot)."""
    out = [f"changed: {p}" for p in sorted(old.keys() & new.keys()) if old[p] != new[p]]
    out += [f"missing: {p}" for p in sorted(old.keys() - new.keys())]
    out += [f"new: {p}" for p in sorted(new.keys() - old.keys())]
    return out


def shorten(lines: list[str], what: str) -> list[str]:
    if len(lines) <= SHOWN:
        return [f"{what}: {item}" for item in lines]
    return [f"{what}: {item}" for item in lines[:SHOWN]] + [
        f"{what}: ... and {len(lines) - SHOWN} more"
    ]


class Freeze:
    """The manifest of a snapshot and everything derived from it."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(sorted(files.items()))
        self.members: dict[str, dict[str, str]] = {g: {} for g in GROUPS}
        for path, digest in self.files.items():
            self.members[group_of(path)][path] = digest
        self.groups: dict[str, str | None] = {
            g: lines_digest(self.members[g].items()) if self.members[g] else None
            for g in GROUPS
            if self.members[g] or g in EXPECTED
        }
        package = {p[len(SRC) + 1 :]: d for p, d in self.files.items() if p.startswith(f"{SRC}/")}
        self.code = lines_digest(package.items())
        self.root = lines_digest(self.files.items())

    def problems(self) -> list[str]:
        out = []
        for name, digest in self.groups.items():
            found = len(self.members[name])
            if digest is None:
                out.append(f"{name}: MISSING (no files)")
            elif found < EXPECTED.get(name, 1):
                out.append(f"{name}: {found} files, expected at least {EXPECTED[name]}")
        out += [f"{p}: MISSING (required)" for p in REQUIRED if p not in self.files]
        return out

    def table(self) -> list[str]:
        rows = []
        for name, digest in self.groups.items():
            shown = digest if digest is not None else "MISSING"
            rows.append(f"{name:<10}{shown}  ({len(self.members[name])} files)")
        return rows

    def render(self) -> str:
        head = [HEADER]
        head += [f"{name:<10}{digest}" for name, digest in self.groups.items()]
        head += [f"{'code':<10}{self.code}", f"{'root':<10}{self.root}"]
        head += [f"{'files':<10}{len(self.files)}", SEPARATOR]
        return "\n".join(head + [line_of(p, d) for p, d in self.files.items()]) + "\n"


def parse(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """(group lines, files) of a freeze file written by `Freeze.render`."""
    heads: dict[str, str] = {}
    files: dict[str, str] = {}
    in_files = False
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        if raw == SEPARATOR:
            in_files = True
        elif in_files:
            digest, sep, path = raw.partition("  ")
            if not sep or not path:
                raise Refused([f"unreadable line in the freeze file: {raw[:60]!r}"])
            files[path] = digest
        else:
            name, _, value = raw.partition(" ")
            heads[name] = value.strip()
    return heads, files


def build(code_root: Path, repo_root: Path, sets: Sequence[str]) -> Freeze:
    """The manifest of the snapshot, after checking it is what `repo_root/src` holds."""
    bad = []
    for flag, root in (("--code-root", code_root), ("--repo-root", repo_root)):
        if not root.is_dir():
            bad.append(f"{flag} {root}: not a directory")
        elif not (root / SRC).is_dir():
            bad.append(f"{flag} {root}: no {SRC}")
        else:
            # a link on the way to the package would make "the snapshot" the live tree itself
            for here in (root / "src", root / SRC):
                if here.is_symlink():
                    bad.append(f"{flag} {root}: {here.relative_to(root).as_posix()} is a link")
    if bad:
        raise Refused(bad)
    snapshot = tree_digests(code_root / SRC, SRC)
    live = tree_digests(repo_root / SRC, SRC)
    if snapshot != live:
        raise Refused(
            [f"{SRC} of --repo-root differs from the snapshot in --code-root"]
            + shorten(differences(snapshot, live), "src")
        )
    files = dict(snapshot)
    for rel in CONFIG_FILES:
        files.update(one_file(repo_root, rel))
    locks = repo_root / LOCKS
    if locks.is_dir():
        files.update(tree_digests(locks, LOCKS))
    for name in sets:
        path, digest = extra_file(repo_root, name)
        if group_of(path) != "sets":
            raise Refused([f"sets: {path} is already part of the manifest as {group_of(path)}"])
        files[path] = digest
    return Freeze(files)


def mismatch(heads: dict[str, str], files: dict[str, str], now: Freeze, name: str) -> list[str]:
    """Why `now` is not the freeze recorded in `name` (empty: it is)."""
    expected = {g: d for g, d in now.groups.items()}
    expected |= {"code": now.code, "root": now.root, "files": str(len(now.files))}
    if heads == expected and files == now.files:
        return []
    out = [f"verify: the files no longer match {name}"]
    out += shorten(differences(files, now.files), "file")
    for key, digest in expected.items():
        if heads.get(key) != digest:
            out.append(f"{key}: recorded {heads.get(key)}, now {digest}")
    return out


def stage2_main(args: argparse.Namespace) -> int:
    code_root, repo_root = Path(args.code_root), Path(args.repo_root)
    sets = list(args.sets)
    recorded_heads: dict[str, str] = {}
    recorded_files: dict[str, str] = {}
    try:
        if args.verify:
            try:
                text = Path(args.verify).read_text()
            except OSError as exc:
                raise Refused([f"--verify {args.verify}: {exc.strerror or exc}"]) from exc
            recorded_heads, recorded_files = parse(text)
            # the sets files are the ones the freeze lists, plus any named again
            sets += [p for p in recorded_files if group_of(p) == "sets" and p not in sets]
        freeze = build(code_root, repo_root, sets)
    except Refused as exc:
        print("\n".join(exc.problems), file=sys.stderr)
        return 2
    problems = freeze.problems()
    if args.verify and not problems:
        problems = mismatch(recorded_heads, recorded_files, freeze, str(args.verify))
    if problems:
        print("\n".join(freeze.table()))
        print("\n".join(problems), file=sys.stderr)
        return 2
    if args.verify:
        print(f"verified {len(freeze.files)} files, root {freeze.root}")
        return 0
    if not args.write:
        print(freeze.render(), end="")
        return 0
    target = Path(args.write)
    if target.exists() and target.read_text() != freeze.render() and not args.overwrite:
        print(f"{target} exists and differs; pass --overwrite to replace it", file=sys.stderr)
        return 2
    target.write_text(freeze.render())
    print("\n".join(freeze.table()))
    print(f"wrote {target}: {len(freeze.files)} files, root {freeze.root}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="freeze_hashes.py", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    p.add_argument(
        "root", nargs="?", help="stage 1 only: the snapshot to hash (default: this repo)"
    )
    p.add_argument("--stage2", action="store_true", help="freeze (or verify) a stage-2 snapshot")
    p.add_argument("--code-root", help="the snapshot the jobs import (holds src/)")
    p.add_argument("--repo-root", help="the checkout that is committed (holds src/ and the rest)")
    p.add_argument(
        "--sets",
        action="append",
        default=[],
        metavar="FILE",
        help="freeze 2: a file to add (sets file, plan manifest, draft); repeatable",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--write", metavar="FILE", help="write the manifest here (stdout if omitted)")
    mode.add_argument("--verify", metavar="FILE", help="check the files against this manifest")
    p.add_argument("--overwrite", action="store_true", help="replace a different --write target")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not args.stage2:
        stray = [
            flag
            for flag, value in (
                ("--code-root", args.code_root),
                ("--repo-root", args.repo_root),
                ("--sets", args.sets),
                ("--write", args.write),
                ("--verify", args.verify),
                ("--overwrite", args.overwrite),
            )
            if value
        ]
        if stray:
            p.error(f"{', '.join(stray)} need --stage2")
        root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
        return legacy_main(root)
    if args.root:
        p.error("--stage2 takes --code-root and --repo-root, not a positional root")
    if not args.code_root or not args.repo_root:
        p.error("--stage2 needs --code-root and --repo-root (there is no default)")
    return stage2_main(args)


if __name__ == "__main__":
    sys.exit(main())
