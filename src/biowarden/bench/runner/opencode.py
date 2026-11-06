"""Run one trial with OpenCode as the harness, inside the same kind of sandbox as the minimal loop.

OpenCode keeps its own system prompt, tools (bash, read, edit, write, grep, glob, ...), permission
system, context management and retry logic, which is the point: it tells us whether a result seen
with our minimal bash loop survives a real harness. It runs headless (``opencode run``) with:

  no network       the sandbox has only loopback; a relay forwards 127.0.0.1:PORT to the benchmark
                   proxy's unix socket, and the proxy (outside) holds the real API key
  isolated state   HOME and the XDG directories live in a per-trial directory
  no side effects  autoupdate, model-list fetching, LSP downloads, plugins and Claude Code
                   compatibility reading are switched off
  same task        the contract text and the task sentence form the first user message

The commands the model ran, the tokens and the USD come from the proxy, so exposure accounting and
cost work exactly as for the minimal loop.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.runner import fwd as _fwd
from biowarden.bench.runner.agent import CONTRACTS, INTRO, Trajectory
from biowarden.bench.runner.proxy import ProxyServer, TrialStats
from biowarden.bench.runner.sandbox import (
    BWRAP,
    SHIM_MOUNT,
    SOCK_MOUNT,
    SYMLINKS,
    SYSTEM_RO,
    THREADS,
    WORKDIR,
    SandboxLayer,
    bwrap_path,
    tool_binds,
)
from biowarden.bench.tasks.tools import ToolEnv

PORT = 18080
OC_DIR = "/oc"
OPENCODE_FLAGS = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_PURE": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_PRUNE": "1",
}


@dataclass(frozen=True)
class OpenCodeRig:
    """Where OpenCode lives and the proxy it talks to (the proxy outlives every trial)."""

    binary_dir: Path  # directory holding the `opencode` executable
    proxy: ProxyServer

    @property
    def fwd_script(self) -> Path:
        return Path(_fwd.__file__)


def find_opencode_dir(root: Path | None = None) -> Path | None:
    """$BIOWARDEN_OPENCODE_DIR, else the project-local install, else None."""
    explicit = os.environ.get("BIOWARDEN_OPENCODE_DIR")
    local = (root or Path.cwd()) / ".local/opencode/node_modules/opencode-linux-x64/bin"
    for candidate in (Path(explicit) if explicit else None, local):
        if candidate is not None and (candidate / "opencode").is_file():
            return candidate
    return None


def opencode_config(model: str, trial: str, max_output: int = 16000) -> dict[str, Any]:
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        "lsp": False,
        "mcp": {},
        "provider": {
            "openrouter": {
                "options": {
                    "baseURL": f"http://127.0.0.1:{PORT}/t/{trial}/api/v1",
                    "apiKey": "dummy",
                },
                "models": {
                    model: {
                        "name": model,
                        "tool_call": True,
                        "reasoning": True,
                        "limit": {"context": 1_000_000, "output": max_output},
                    }
                },
            }
        },
        "model": f"openrouter/{model}",
    }


def first_message(task_prompt: str, contract: str) -> str:
    """The contract text that the minimal loop puts in its system prompt, plus the task."""
    return "\n".join([INTRO, "", *CONTRACTS[contract], "", task_prompt])


def sandbox_argv(
    rig: OpenCodeRig,
    tools: ToolEnv,
    work: Path,
    oc_dir: Path,
    model: str,
    trial: str,
    message: str,
    timeout: int,
    layer: SandboxLayer | None = None,
) -> list[str]:
    argv = ["timeout", "-k", "10", str(timeout), "env", "-i", bwrap_path() or BWRAP]
    argv += ["--unshare-all", "--die-with-parent", "--new-session"]
    for path in (*SYSTEM_RO, "/etc/passwd", "/etc/group"):
        argv += ["--ro-bind", path, path]
    for target, link in SYMLINKS:
        argv += ["--symlink", target, link]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in tool_binds(tools):
        argv += ["--ro-bind", str(path), str(path)]
    argv += ["--ro-bind", str(rig.binary_dir.resolve()), "/opt/opencode"]
    argv += ["--ro-bind", str(rig.fwd_script), "/opt/fwd.py"]
    argv += ["--bind", str(work.resolve()), WORKDIR, "--bind", str(oc_dir.resolve()), OC_DIR]
    argv += ["--bind", str(rig.proxy.socket_path.resolve()), "/tmp/proxy.sock", "--chdir", WORKDIR]
    real_path = ":".join(["/opt/opencode", *map(str, tools.bin_dirs), "/usr/bin"])
    path = real_path
    if layer is not None:  # shims only: OpenCode runs its own bash, so there is no string pre-check
        argv += ["--ro-bind", str(layer.shim_dir.resolve()), SHIM_MOUNT]
        argv += ["--bind", str(layer.sock), SOCK_MOUNT]
        behind = (
            "/usr/bin" if layer.hide_real else ":".join([*map(str, tools.bin_dirs), "/usr/bin"])
        )
        path = f"/opt/opencode:{SHIM_MOUNT}:{behind}"
    home = f"{OC_DIR}/home"
    env = {
        "PATH": path,
        "HOME": home,
        "XDG_CONFIG_HOME": f"{home}/.config",
        "XDG_DATA_HOME": f"{home}/.local/share",
        "XDG_CACHE_HOME": f"{home}/.cache",
        "XDG_STATE_HOME": f"{home}/.local/state",
        "LC_ALL": "C",
        "TMPDIR": "/tmp",
        **THREADS,
        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config(model, trial)),
        "OC_MODEL": f"openrouter/{model}",
        "OC_MESSAGE": message,
        **OPENCODE_FLAGS,
    }
    if layer is not None:
        env["BIOWARDEN_SOCK"] = SOCK_MOUNT
        env["BIOWARDEN_REAL_PATH"] = tools.path_var()
        if layer.watch is not None:
            env["BIOWARDEN_WATCH"] = ",".join(layer.watch)
    if tools.lib_dirs:
        env["LD_LIBRARY_PATH"] = tools.lib_var()
    for name, value in env.items():
        argv += ["--setenv", name, value]
    script = (
        f"python3.12 /opt/fwd.py {PORT} /tmp/proxy.sock & sleep 0.5; "
        f'exec opencode run --format json --auto --dir {WORKDIR} --model "$OC_MODEL" "$OC_MESSAGE" '
        f"> {OC_DIR}/events.jsonl 2> {OC_DIR}/stderr.txt"
    )
    return [*argv, "--", "bash", "-c", script]


def run_opencode(
    rig: OpenCodeRig,
    tools: ToolEnv,
    work: Path,
    trial_dir: Path,
    model: str,
    trial: str,
    task_prompt: str,
    contract: str,
    timeout: int = 900,
    layer: SandboxLayer | None = None,
) -> tuple[TrialStats, dict[str, Any]]:
    """Run OpenCode on the task; returns the proxy's stats for the trial and process details."""
    oc_dir = trial_dir / "oc"
    (oc_dir / "home").mkdir(parents=True, exist_ok=True)
    message = first_message(task_prompt, contract)
    argv = sandbox_argv(rig, tools, work, oc_dir, model, trial, message, timeout, layer)
    proc = subprocess.run(
        argv, capture_output=True, check=False, stdin=subprocess.DEVNULL, timeout=timeout + 60
    )
    stats = rig.proxy.pop_stats(trial)
    log = oc_dir / "stderr.txt"
    inside = log.read_text(errors="replace") if log.exists() else ""
    info = {
        "returncode": proc.returncode,
        "timed_out": proc.returncode in (124, 137),
        # bwrap's own complaints (namespaces, mounts) arrive on proc.stderr, OpenCode's in the log
        "stderr_tail": (proc.stderr.decode("utf-8", "replace") + inside)[-600:],
    }
    return stats, info


def transcript_from_commands(commands: list[str]) -> list[dict[str, Any]]:
    """A transcript in the Anthropic layout, so exposure accounting reads it unchanged."""
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "bash", "input": {"command": c}}],
        }
        for c in commands
    ]


def opencode_trajectory(
    rig: OpenCodeRig,
    tools: ToolEnv,
    work: Path,
    trial_dir: Path,
    model: str,
    trial: str,
    task_prompt: str,
    contract: str,
    timeout: int = 900,
    layer: SandboxLayer | None = None,
) -> Trajectory:
    """Run OpenCode and describe the run in the same shape as the minimal loop's trajectory."""
    stats, info = run_opencode(
        rig, tools, work, trial_dir, model, trial, task_prompt, contract, timeout, layer
    )
    stop = (
        "end_turn"
        if info["returncode"] == 0
        else "timeout"
        if info["timed_out"]
        else "harness_error"
    )
    error = None
    if stats.requests == 0 or stats.http_errors >= stats.requests:  # the model was never reached
        stop, error = "api_error", info["stderr_tail"] or "no successful model request"
    provider = stats.providers.most_common(1)[0][0] if stats.providers else None
    return Trajectory(
        stop=stop,
        turns=stats.requests,
        commands=len(stats.commands),
        tool_arg_errors=stats.tool_arg_errors,
        usd=stats.usd,
        tokens=Counter(stats.tokens),
        served_by=f"{model}@{provider}" if provider else model,
        error=error,
        messages=transcript_from_commands(stats.commands),
    )
