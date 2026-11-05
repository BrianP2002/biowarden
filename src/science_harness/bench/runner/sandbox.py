"""Run an agent's shell command in a bubblewrap sandbox that exposes one trial's work directory.

The sandbox root is built from scratch: read-only /usr, the tool installs, a tmpfs /tmp, and the
work directory bound read-write at /work. Home directories, the project (and its .env, notes and
scripts), the truth files and the network are not there: ``tool_binds`` refuses any tool directory
whose bind would expose them. bwrap 0.4
has no --clearenv, so the environment is emptied with ``env -i`` and rebuilt with --setenv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks.tools import ToolEnv, strip_loader_noise

BWRAP = "/usr/bin/bwrap"
WORKDIR = (
    "/work"  # where the trial directory appears inside the sandbox; the host path stays hidden
)
# cpu seconds, address space (KiB), file size (KiB), set inside the sandbox. There is deliberately
# no process limit (ulimit -u): RLIMIT_NPROC counts every process of the user on the node, so under
# many concurrent trials it made forks fail ("Resource temporarily unavailable") in almost every
# trial of the first open-weight round. Runaway forks are bounded by the wall-clock timeout, the
# pid namespace, and the scheduler job's cgroup.
LIMITS = "ulimit -t 120 -v 6291456 -f 262144 2>/dev/null"
# one thread per numeric library: OpenBLAS reserves address space per core and dies under the
# sandbox's memory limit otherwise, and results should not depend on the machine's core count
THREADS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MMSEQS_NUM_THREADS": "1",
}
SYSTEM_RO = ("/usr", "/etc/ld.so.cache", "/etc/alternatives")
SYMLINKS = (
    ("usr/bin", "/bin"),
    ("usr/lib", "/lib"),
    ("usr/lib64", "/lib64"),
    ("usr/sbin", "/sbin"),
)


SHIM_MOUNT = "/opt/tbm/bin"
SOCK_MOUNT = "/opt/tbm/engine.sock"


@dataclass(frozen=True)
class SandboxLayer:
    """The checking layer as the sandbox sees it: shims first on PATH, a socket to the engine.

    hide_real: the real tool directories are not on PATH (only the shim can find them through
    TBM_REAL_PATH); otherwise they stay on PATH behind the shims and a call can bypass the layer.
    """

    shim_dir: Path
    sock: Path
    hide_real: bool = True
    watch: tuple[str, ...] | None = None  # programs the rules watch; None: every shim asks
    # checks a whole command string before it runs (pipeline-aware); None: shims only
    precheck: Callable[[str], Any] | None = None


@dataclass(frozen=True)
class CommandResult:
    exit_code: int | None  # None: the command timed out
    stdout: str
    stderr: str
    timeout_s: int
    truncated: bool = False

    @property
    def timed_out(self) -> bool:
        return self.exit_code is None

    def render(self) -> str:
        """The text the agent sees."""
        head = (
            f"[timed out after {self.timeout_s}s]"
            if self.timed_out
            else f"[exit code {self.exit_code}]"
        )
        parts = [head]
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout.rstrip()}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr.rstrip()}")
        if not (self.stdout or self.stderr):
            parts.append("(no output)")
        return "\n".join(parts)


def bwrap_path() -> str | None:
    """$SCIENCE_HARNESS_BWRAP, the system bwrap, a project-local copy (compute nodes lack bwrap)."""
    candidates = [
        os.environ.get("SCIENCE_HARNESS_BWRAP"),
        BWRAP,
        str(Path.cwd() / ".local/bin/bwrap"),
        shutil.which("bwrap"),
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def available() -> bool:
    return bwrap_path() is not None


def tool_binds(tools: ToolEnv) -> list[Path]:
    """The read-only tool directories of a sandbox, refusing any that would expose the project.

    A bind source is refused when it is, contains or lies inside the project (the directory the
    jobs run from and the checkout this code is imported from) or when it is, or contains, the home
    directory. The project holds the truth files, the rule catalogue and the design notes; the
    sandbox must not be able to read them.
    """
    here = Path(__file__).resolve()
    projects = {Path.cwd().resolve()}
    if here.parents[3].name == "src":
        projects.add(here.parents[4])
    home = Path.home().resolve()
    binds = tools.ro_binds()
    for bind in binds:
        real = bind.resolve()
        exposes_project = any(real == p or real in p.parents or p in real.parents for p in projects)
        if exposes_project or real == home or real in home.parents:
            raise ValueError(
                f"refusing to bind {bind} into the sandbox: it is, contains or lies inside the "
                "project or the home directory; keep tool directories outside both"
            )
    return binds


def bwrap_argv(
    work: Path,
    tools: ToolEnv,
    command: str,
    timeout: int,
    layer: SandboxLayer | None = None,
) -> list[str]:
    argv = ["timeout", "-k", "3", str(timeout), "env", "-i", bwrap_path() or BWRAP]
    argv += ["--unshare-all", "--die-with-parent", "--new-session"]
    for path in SYSTEM_RO:
        argv += ["--ro-bind", path, path]
    for target, link in SYMLINKS:
        argv += ["--symlink", target, link]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in tool_binds(tools):
        argv += ["--ro-bind", str(path), str(path)]
    argv += ["--bind", str(work.resolve()), WORKDIR, "--chdir", WORKDIR]
    env = {"PATH": tools.path_var(), "HOME": WORKDIR, "LC_ALL": "C", "TMPDIR": "/tmp", **THREADS}
    if layer is not None:
        argv += ["--ro-bind", str(layer.shim_dir.resolve()), SHIM_MOUNT]
        argv += ["--bind", str(layer.sock), SOCK_MOUNT]
        env["TBM_SOCK"] = SOCK_MOUNT
        env["TBM_REAL_PATH"] = tools.path_var()
        if layer.watch is not None:
            env["TBM_WATCH"] = ",".join(layer.watch)
        env["PATH"] = f"{SHIM_MOUNT}:{'/usr/bin' if layer.hide_real else tools.path_var()}"
    if tools.lib_dirs:
        env["LD_LIBRARY_PATH"] = tools.lib_var()
    for name, value in env.items():
        argv += ["--setenv", name, value]
    # the command travels as a positional argument, so no quoting survives to be mangled
    argv += ["--", "bash", "-c", f'{LIMITS}; exec bash -c "$1"', "_", command]
    return argv


def clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}", True


def run_command(
    work: Path,
    tools: ToolEnv,
    command: str,
    timeout: int = 60,
    max_chars: int = 6000,
    layer: SandboxLayer | None = None,
) -> CommandResult:
    warning = ""
    if layer is not None and layer.precheck is not None:
        checked = layer.precheck(command)
        if checked.action == "block":
            return CommandResult(97, "", checked.text, timeout)
        warning = checked.text if checked.action == "warn" else ""
    proc = subprocess.run(
        bwrap_argv(work, tools, command, timeout, layer),
        capture_output=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=timeout + 15,
    )
    out, cut_out = clip(proc.stdout.decode("utf-8", "replace"), max_chars)
    err, cut_err = clip(strip_loader_noise(proc.stderr.decode("utf-8", "replace")), max_chars)
    timed_out = proc.returncode in (124, 137)  # timeout(1): 124, or 137 after the -k kill
    if warning:
        err = f"{warning}\n{err}".rstrip("\n")
    return CommandResult(
        None if timed_out else proc.returncode, out, err, timeout, cut_out or cut_err
    )
