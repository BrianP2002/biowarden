"""Checker registry and the context a checker runs in.

A checker is a pure function `(call, match, ctx) -> CheckResult`. The context enforces the rule's
tier: a T0 checker may not touch the filesystem, a T1 checker may read files and run the real tool
read-only, and a T2 rule can never block (it only produces notes).

The context also keeps a checker within bounds: files only below the trial directory (`root`; a
name that leads out of it is a file that does not exist), only regular files, at most `max_bytes`
of text, a dry run without stdin and with a limit on its memory and time, and everything within the
time the evaluation has left (`deadline`).
"""

from __future__ import annotations

import errno
import gzip
import io
import os
import resource
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.spec import Rule

Verdict = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class CheckResult:
    verdict: Verdict
    code: str
    message: str = ""
    expected: Any = None
    got: Any = None
    fix: tuple[str, ...] = ()
    note: str = ""


def passed(note: str = "") -> CheckResult:
    return CheckResult("pass", "ok", note=note)


@dataclass(frozen=True)
class Match:
    rule: Rule
    files: tuple[Path, ...] = ()


class TierViolation(RuntimeError):
    """A checker used a capability its tier does not grant."""


GZIP_MAGIC = b"\x1f\x8b"
# the parent of every name that leads out of the trial directory: a directory that does not exist
OUTSIDE = Path("/.tbm-outside-the-trial")
DRY_RUN_MEMORY = 6 * 1024**3  # bytes of address space, what the sandbox gives the agent's commands


class LayerTimeout(TimeoutError):
    """The evaluation used up its time. The rule it stopped says nothing about the call."""


def inside(path: str | Path, root: str | Path) -> bool:
    """True when `path` lies in `root` or below it once symlinks are followed (it need not exist).

    A path the host cannot resolve is outside.
    """
    try:
        base, real = os.path.realpath(root), os.path.realpath(path)
    except (OSError, ValueError, RuntimeError):
        return False
    return real == base or real.startswith(base.rstrip(os.sep) + os.sep)


def is_regular_file(path: Path, root: Path | None = None) -> bool:
    """True for an existing regular file; never raises.

    A name the host cannot stat (too long, no permission, a loop) is not a file, and neither is a
    directory, a FIFO or a device. With `root`, a file whose real place is outside it is not one.
    """
    try:
        return stat.S_ISREG(path.stat().st_mode) and (root is None or inside(path, root))
    except (OSError, ValueError, RuntimeError):
        return False


def _open_regular(path: Path) -> io.BufferedReader:
    """Open a regular file for reading. A FIFO would block the open for ever and a device never
    ends, so anything else is refused as if the file were not there."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        mode = os.fstat(fd).st_mode
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
        if not stat.S_ISREG(mode):
            raise FileNotFoundError(errno.ENOENT, "not a regular file", str(path))
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # the tool and whatever it started
    except OSError:
        proc.kill()


@dataclass
class Ctx:
    cwd: Path
    tier: str
    real_bins: Mapping[str, Path] = field(default_factory=dict)
    real_env: Mapping[str, str] | None = None
    max_bytes: int = 4_000_000
    run_timeout: float = 30.0
    root: Path | None = None  # as Call.root: a name that leads outside it resolves to nothing
    deadline: float | None = None  # time.monotonic() after which the checker is stopped
    memory_limit: int | None = DRY_RUN_MEMORY  # address space of a dry run, in bytes

    def _need_files(self) -> None:
        if self.tier != "T1":
            raise TierViolation(f"tier {self.tier} may not read files")

    def check_time(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise LayerTimeout("the time for this evaluation is used up")

    def resolve(self, name: str) -> Path:
        """The host path of a file name in the call. A name that leads out of the trial directory
        (an absolute path of the sandbox, `..`, a symlink out) gives a path that does not exist, so
        the checker sees a missing file and no other file of the host is opened."""
        path = Path(name)
        path = path if path.is_absolute() else self.cwd.absolute() / path
        if self.root is not None and not inside(path, self.root):
            return OUTSIDE / path.name.replace("\0", "")
        return path

    def lines(self, name: str | Path, limit: int | None = None) -> Iterator[str]:
        """Text lines of a plain or (b)gzip file, at most `max_bytes` of decoded text."""
        self._need_files()
        path = self.resolve(str(name))
        with _open_regular(path) as raw:
            gz = raw.read(2) == GZIP_MAGIC
            raw.seek(0)
            handle = io.TextIOWrapper(gzip.GzipFile(fileobj=raw) if gz else raw, errors="replace")
            seen = count = 0
            while limit is None or count < limit:
                self.check_time()
                line = handle.readline(self.max_bytes - seen + 1)  # a line without end is bounded
                seen += len(line)
                if not line or seen > self.max_bytes:
                    return
                count += 1
                yield line.rstrip("\n")

    def read_bytes(self, name: str | Path, limit: int | None = None) -> bytes:
        """Raw bytes of a file, at most `limit` (default `max_bytes`); for binary formats (BAM)."""
        self._need_files()
        self.check_time()
        with _open_regular(self.resolve(str(name))) as handle:
            return handle.read(self.max_bytes if limit is None else limit)

    def _limit_memory(self, pid: int) -> None:
        if self.memory_limit:
            try:
                resource.prlimit(pid, resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
            except (AttributeError, OSError, ValueError):
                pass  # no prlimit here, or the tool is already gone

    def run_real(self, tool: str, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run the real tool for a dry run; the caller keeps the arguments free of side effects.

        The tool gets no stdin and a limit on its address space. When `run_timeout` (or the time
        left of the evaluation) runs out, the tool and anything it started are killed and the caller
        gets `subprocess.TimeoutExpired`, as before.
        """
        self._need_files()
        exe = self.real_bins.get(tool)
        if exe is None:
            raise FileNotFoundError(f"no real binary known for {tool}")
        if self.root is not None and not inside(self.cwd, self.root):
            raise FileNotFoundError("the working directory is outside the trial")
        command = [str(exe), *argv]
        timeout = self.run_timeout
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise subprocess.TimeoutExpired(command, 0)
        with subprocess.Popen(
            command,
            cwd=self.cwd,
            env=None if self.real_env is None else dict(self.real_env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        ) as proc:
            self._limit_memory(proc.pid)
            try:
                out, err = proc.communicate(timeout=timeout)
            except BaseException:
                _kill_group(proc)
                raise
        return subprocess.CompletedProcess(command, proc.returncode, out, err)


Checker = Callable[[Call, Match, Ctx], CheckResult]
_REGISTRY: dict[str, Checker] = {}


def checker(name: str) -> Callable[[Checker], Checker]:
    def register(fn: Checker) -> Checker:
        if name in _REGISTRY:
            raise ValueError(f"checker {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return register


def known_checkers() -> dict[str, Checker]:
    return dict(_REGISTRY)


Suggester = Callable[[Call, Rule], tuple[str, ...]]
_SUGGESTERS: dict[str, Suggester] = {}


def suggester(check_name: str) -> Callable[[Suggester], Suggester]:
    """Register the function that words the fix when a required declaration is missing.

    It receives the call and the rule and returns fix lines that fit THIS call (for example the
    exact `--declare` that matches the options the call already uses).
    """

    def register(fn: Suggester) -> Suggester:
        _SUGGESTERS[check_name] = fn
        return fn

    return register


def suggestion_for(check_name: str) -> Suggester | None:
    return _SUGGESTERS.get(check_name)


@checker("option_implies_declared")
def option_implies_declared(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    """T0 built-in: each option names the convention it implies; it must equal the declaration.

    params: option_map {option: value} (the option present in the call decides `actual`), and
    `fix_option` {value: option} used to suggest the option that matches the declaration.
    """
    rule = match.rule
    assert rule.declare is not None
    option_map: dict[str, str] = rule.params["option_map"]
    fix_option: dict[str, str] = rule.params.get("fix_option", {})
    attached = rule.params.get("attached", True)
    implied = {
        value
        for option, value in option_map.items()
        if option_values(call.argv, option, attached=attached) or option in call.argv
    }
    if len(implied) != 1:
        return CheckResult("unknown", "ambiguous_call", f"{len(implied)} conventions implied")
    (actual,) = implied
    declared = call.declared.get(rule.declare.key)
    if declared is None:  # an optional declaration that was not made: nothing to compare
        return passed("no declaration")
    if actual == declared:
        return passed()
    wanted = fix_option.get(declared)
    return CheckResult(
        "fail",
        "declared_mismatch",
        f"you declared {rule.declare.key}={declared} but this call implies {actual}",
        expected=declared,
        got=actual,
        fix=(f"use {wanted} instead" if wanted else f"declare {rule.declare.key}={actual}",),
    )


@checker("option_value_implies_declared")
def option_value_implies_declared(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    """T0 built-in: the VALUE of an option names the convention; it must equal the declaration.

    params: option_values {option: {value: convention}}, for example
    {"--rna-strandness": {"F": "forward", "R": "reverse"}}, and `fix_value`
    {convention: "option value"}.
    An option present with a value outside the map, or two options that disagree, is `unknown`.
    """
    rule = match.rule
    assert rule.declare is not None
    table: dict[str, dict[str, str]] = rule.params["option_values"]
    fix: dict[str, str] = rule.params.get("fix_value", {})
    implied: set[str] = set()
    for option, values in table.items():
        for value in option_values(call.argv, option, attached=rule.params.get("attached", True)):
            if value not in values:
                return CheckResult(
                    "unknown", "unmapped_value", f"{option} {value}: value not in the rule"
                )
            implied.add(values[value])
    if len(implied) != 1:
        return CheckResult("unknown", "ambiguous_call", f"{len(implied)} conventions implied")
    (actual,) = implied
    declared = call.declared.get(rule.declare.key)
    if declared is None:
        return passed("no declaration")
    if actual == declared:
        return passed()
    wanted = fix.get(declared)
    return CheckResult(
        "fail",
        "declared_mismatch",
        f"you declared {rule.declare.key}={declared} but this call implies {actual}",
        expected=declared,
        got=actual,
        fix=(f"use {wanted} instead" if wanted else f"declare {rule.declare.key}={actual}",),
    )
