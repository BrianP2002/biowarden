"""Where the real command-line tools live, for direct runs (task setup, oracles) and the sandbox."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TOOL_NAMES = ("samtools", "bcftools", "bedtools")
# Dynamic-loader chatter some cluster installs print on every call; it says nothing about the data.
LOADER_NOISE = re.compile(r"no version information available")
TOOLS_ENV_VAR = "SCIENCE_HARNESS_TOOLS"


@dataclass(frozen=True)
class ToolEnv:
    bin_dirs: tuple[Path, ...]
    lib_dirs: tuple[Path, ...] = ()
    # directories searched AFTER /usr/bin: more tools (for example a conda environment) that must
    # never shadow a system program such as python3
    tail_dirs: tuple[Path, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> ToolEnv:
        data = json.loads(Path(path).read_text())
        return cls(
            tuple(Path(p) for p in data["bin_dirs"]),
            tuple(Path(p) for p in data.get("lib_dirs", ())),
            tuple(Path(p) for p in data.get("tail_dirs", ())),
        )

    @classmethod
    def from_path(cls, names: Sequence[str] = TOOL_NAMES) -> ToolEnv:
        found = [shutil.which(n) for n in names]
        if not all(found):
            missing = [n for n, f in zip(names, found, strict=True) if not f]
            raise FileNotFoundError(f"not on PATH: {', '.join(missing)}")
        return cls(tuple(dict.fromkeys(Path(f).resolve().parent for f in found if f)))

    @classmethod
    def discover(cls, root: Path | None = None) -> ToolEnv | None:
        """$SCIENCE_HARNESS_TOOLS, then <root>/.local/tools.json, then the tools on PATH."""
        explicit = os.environ.get(TOOLS_ENV_VAR)
        local = (root or Path.cwd()) / ".local" / "tools.json"
        for candidate in (Path(explicit) if explicit else None, local):
            if candidate is not None and candidate.is_file():
                return cls.from_json(candidate)
        try:
            return cls.from_path()
        except FileNotFoundError:
            return None

    def real_bins(self, names: Sequence[str] = TOOL_NAMES) -> dict[str, Path]:
        """tool name -> executable, for the engine's read-only dry runs."""
        found: dict[str, Path] = {}
        for name in names:
            for directory in (*self.bin_dirs, *self.tail_dirs):
                if (directory / name).is_file():
                    found[name] = directory / name
                    break
        return found

    def ro_binds(self) -> list[Path]:
        """Directories a sandbox must expose read-only for the tools to run.

        A `bin` directory is bound through its parent, the install prefix with the libraries and
        data files the programs need. Any other tool directory (for example a directory of wrapper
        links) is bound itself: its parent can be anything, including the project.
        """
        binds = [d.parent if d.name == "bin" else d for d in (*self.bin_dirs, *self.tail_dirs)]
        return [*dict.fromkeys(binds), *self.lib_dirs]

    def path_var(self, tail: str = "/usr/bin") -> str:
        return ":".join([*map(str, self.bin_dirs), tail, *map(str, self.tail_dirs)])

    def all_executables(self) -> list[str]:
        """Names of every executable in the tool directories (for pass-through shims)."""
        names: set[str] = set()
        for directory in (*self.bin_dirs, *self.tail_dirs):
            if directory.is_dir():
                names |= {
                    p.name for p in directory.iterdir() if p.is_file() and os.access(p, os.X_OK)
                }
        return sorted(names)

    def lib_var(self) -> str:
        return ":".join(map(str, self.lib_dirs))

    def process_env(self) -> dict[str, str]:
        env = {
            "PATH": self.path_var(os.environ.get("PATH", "/usr/bin")),
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",  # ~/.local packages must not leak into a build
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        if self.lib_dirs:
            env["LD_LIBRARY_PATH"] = self.lib_var()
        return env

    def run(
        self, argv: Sequence[str], cwd: Path, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run a tool directly (outside the sandbox) with loader noise removed from stderr."""
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            env=self.process_env(),
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
        proc.stderr = strip_loader_noise(proc.stderr)
        return proc


def strip_loader_noise(text: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if not LOADER_NOISE.search(line))
