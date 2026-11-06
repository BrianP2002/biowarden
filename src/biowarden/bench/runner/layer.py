"""Per-trial checking layer: an engine server on a unix socket plus a shim directory.

Arms (what a trial's agent works with):
  bare          no layer at all (the baseline of every earlier run)
  note          no layer, but the system prompt carries the layer's one sentence (control for it)
  *-nonote      post-hoc variants of warn and block with NO sentence about the layer in the prompt
  warn          shims; every failing rule only warns, the call still runs; real tools hidden
  block         shims plus a pipeline-aware pre-check of the command string; rules the promotion
                table promotes block, the others warn; real tools hidden
  block-shim    like block without the command pre-check (what a shim alone can see)
  block-bypass  like block, but the real tools stay on PATH behind the shims (a call can bypass)
  block-gateway no shims: a run_tool function checked by the engine next to an unchecked bash
  blockall      development arm: every failing rule blocks, whatever the promotion table says

What the sandbox binds never names the trial: the shim directory and the engine socket live in
directories with random names under $BIOWARDEN_NEUTRAL_ROOT (default: the system temp dir), and are
removed when the rig closes. The ledger and the block counts stay in the trial directory, on the
host only. A fault inside the layer costs the check, never the command or the trial: it is recorded
in the summary and the command runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from biowarden.bench.runner.gateway_tool import GatewayTool, make_gateway
from biowarden.bench.runner.sandbox import WORKDIR, SandboxLayer
from biowarden.bench.tasks.tools import ToolEnv
from biowarden.engine import Engine, Promotion
from biowarden.engine.decide import PACKAGED_PROMOTION, BlockState, Mode
from biowarden.engine.install import install_shims
from biowarden.engine.precheck import Precheck, precheck
from biowarden.engine.server import EngineServer, PathMap, serve_in_thread
from biowarden.engine.spec import Spec, packaged_specs
from biowarden.engine.tools import watched_tools
from biowarden.ledger import Ledger

NEUTRAL_ROOT_VAR = "BIOWARDEN_NEUTRAL_ROOT"
SOCKET_NAME = "e.sock"
SOCKET_PATH_MAX = 107  # bytes of an AF_UNIX path
# rules the promotion table may leave without a current entry (they only warn) until recalibrated:
# none, every rule of the packaged specs is promoted for its current spec hash
UNPROMOTED_ALLOWED: tuple[str, ...] = ()
# a command that names the layer: its mounts, its shim, its variables or its socket
TOUCHES_LAYER = re.compile(r"/opt/biowarden|biowarden-shim|BIOWARDEN_|engine\.sock")
CRASH_CODES = frozenset({"checker_crashed", "layer_error"})


@dataclass(frozen=True)
class ArmSpec:
    mode: Mode
    shims: bool = True  # PATH shims in the sandbox
    hide_real: bool = True  # real tools off PATH (only the shims can reach them)
    precheck: bool = True  # pipeline-aware check of the whole command string
    gateway: bool = False  # a run_tool function next to bash
    note: bool = True  # the system prompt carries the layer's one sentence
    style: Literal["full", "generic"] = "full"  # message content: rule and fix, or detection only


ARMS: dict[str, ArmSpec | None] = {
    "bare": None,
    "note": None,  # post-hoc control: the layer's system-prompt sentence, no layer
    "warn": ArmSpec("warn"),
    "warn-nonote": ArmSpec("warn", note=False),  # post-hoc: the layer without the prompt sentence
    "block": ArmSpec("promoted"),
    "block-nonote": ArmSpec("promoted", note=False),
    # post-hoc: the layer detects the risk but says only "check formats and biological meaning"
    "warn-generic": ArmSpec("warn", note=False, style="generic"),
    "block-generic": ArmSpec("promoted", note=False, style="generic"),
    "careful": None,  # post-hoc baseline: no layer, a generic "check formats and meaning" sentence
    "block-shim": ArmSpec("promoted", precheck=False),
    "block-bypass": ArmSpec("promoted", hide_real=False),
    "block-gateway": ArmSpec("promoted", shims=False, precheck=False, gateway=True),
    "blockall": ArmSpec("block"),
}


def neutral_root() -> Path:
    """Where the directories a sandbox binds are made: paths there say nothing about a trial."""
    return Path(os.environ.get(NEUTRAL_ROOT_VAR) or tempfile.gettempdir()).resolve()


def opaque_dir(root: Path) -> Path:
    """A new private directory under `root`, named by 16 random hex digits (it spells no word)."""
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / secrets.token_hex(8)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return path


def unpromoted_rules(specs: Sequence[Spec], promotion: Promotion) -> list[str]:
    """Rules the table cannot promote: no entry, or an entry measured on another version of the
    spec. `Promotion.mode_of` quietly makes both warn-only; here they are listed."""
    found = []
    for spec in specs:
        digest = spec.content_hash()
        for rule in spec.rules:
            entry = promotion.entries.get(rule.id)
            if entry is None or entry.get("spec_hash") not in (None, digest):
                found.append(rule.id)
    return sorted(found)


class PromotionMismatch(ValueError):
    """The promotion table does not cover the specs the way the run allows: a fault of the job."""


@dataclass(frozen=True)
class PromotionCheck:
    promotion: Promotion
    sha: str  # sha256 of the table file
    unpromoted: list[str]  # rules with no current entry, sorted


def check_promotion(
    promotion_path: str | None = None,
    allow_unpromoted: Collection[str] | None = UNPROMOTED_ALLOWED,
    specs: Sequence[Spec] | None = None,
) -> PromotionCheck:
    """Read the promotion table and compare it with the packaged specs.

    Raises PromotionMismatch unless the rules it cannot promote (no entry, or an entry measured on
    another version of the spec) are exactly `allow_unpromoted`; None only records them. A job
    calls this once before its workers start, and `start_layer` again for every trial.
    """
    specs = packaged_specs() if specs is None else specs
    if promotion_path is None and not PACKAGED_PROMOTION.is_file():
        raw = b""
        promotion = Promotion()
    else:
        raw = Path(promotion_path or PACKAGED_PROMOTION).read_bytes()
        promotion = Promotion(json.loads(raw))
    unpromoted = unpromoted_rules(specs, promotion)
    if allow_unpromoted is not None and set(unpromoted) != set(allow_unpromoted):
        missing = sorted(set(unpromoted) - set(allow_unpromoted))
        promoted = sorted(set(allow_unpromoted) - set(unpromoted))
        found = []
        if missing:
            found.append(f"rules with no current entry, which would only warn: {missing}")
        if promoted:
            found.append(f"rules allowed to lack an entry that have one: {promoted}")
        raise PromotionMismatch(
            f"promotion table {promotion_path or PACKAGED_PROMOTION} does not match the specs: "
            f"{'; '.join(found)}. Re-run the admission for them, or change allow_unpromoted."
        )
    return PromotionCheck(promotion, hashlib.sha256(raw).hexdigest(), unpromoted)


def _tally(record: dict[str, Any]) -> tuple[str, str, bool, list[tuple[str, str, str]]]:
    """(action, source, stop, [(rule, code, effective)]) of a decision; raises if malformed."""
    action = record["action"]
    if action not in ("allow", "warn", "block"):
        raise ValueError(action)
    outcomes = [(o["rule"], o["code"], o["effective"]) for o in record["outcomes"]]
    return action, record.get("source", "shim"), bool(record.get("stop")), outcomes


@dataclass
class LayerRig:
    layer: SandboxLayer | None
    gateway: GatewayTool | None
    server: EngineServer
    ledger_path: Path
    tmp: Path  # the engine socket's directory (random name, under the neutral root)
    shim_dir: Path | None = None  # likewise for the shims; None when the arm has none
    spec_hashes: dict[str, str] = field(default_factory=dict)
    promotion_sha: str = ""
    unpromoted: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)  # commands that named the layer

    def summary(self) -> dict[str, Any]:
        """What the layer did in this trial, from its own ledger."""
        actions = {"allow": 0, "warn": 0, "block": 0}
        by_source: dict[str, dict[str, int]] = {}
        fired: dict[str, int] = {}
        stops = crashed = ledger_errors = 0
        try:
            lines = self.ledger_path.read_text(errors="replace").split("\n")
        except FileNotFoundError:
            lines = []
        except OSError:
            lines, ledger_errors = [], 1
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("event") != "engine_decision":
                    continue
                action, source, stop, outcomes = _tally(record)
            except (ValueError, KeyError, TypeError, AttributeError):
                ledger_errors += 1
                continue
            actions[action] += 1
            by_source.setdefault(source, {"allow": 0, "warn": 0, "block": 0})[action] += 1
            stops += stop
            for rule, code, effective in outcomes:
                crashed += code in CRASH_CODES
                if effective != "none":
                    key = f"{rule}:{code}:{effective}"
                    fired[key] = fired.get(key, 0) + 1
        return {
            "calls": self.server.calls,
            "actions": actions,
            "by_source": by_source,
            "stops": stops,
            "fired": sorted(fired),  # rule:code:effective, each once; the counts are below
            "fired_counts": fired,
            "crashed": crashed,  # checker_crashed and layer_error outcomes
            "errors": list(self.server.errors),  # faults of the shim handler and of the pre-check
            "ledger_errors": ledger_errors,
            "touched_layer": len(self.touched),
            "spec_hashes": dict(self.spec_hashes),
            "promotion_sha": self.promotion_sha,
            "unpromoted": list(self.unpromoted),
        }

    def close(self) -> None:
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            for path in (self.tmp, self.shim_dir):
                if path is not None:
                    shutil.rmtree(path, ignore_errors=True)


def start_layer(
    arm: str,
    trial_dir: Path,
    work: Path,
    tools: ToolEnv,
    promotion_path: str | None = None,
    timeout: int = 60,
    max_chars: int = 6000,
    *,
    allow_unpromoted: Collection[str] | None = UNPROMOTED_ALLOWED,
    contract: str = "full",
) -> LayerRig | None:
    """Start the layer of `arm` for one trial; None for an arm without a layer.

    `contract` is the result.json the agent was asked for (full or bare): it words the STOP text.

    The launch fails (PromotionMismatch, see `check_promotion`) unless the rules the promotion
    table cannot promote are exactly `allow_unpromoted`; None only records them. They are in the
    summary either way, with the spec and table digests.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    setting = ARMS[arm]
    if setting is None:
        return None
    specs = packaged_specs()
    checked = check_promotion(promotion_path, allow_unpromoted, specs)
    root = neutral_root()
    if len(os.fsencode(root / ("0" * 16) / SOCKET_NAME)) > SOCKET_PATH_MAX:
        raise ValueError(f"{NEUTRAL_ROOT_VAR}={root} is too long for a unix socket path")
    state_dir = trial_dir / "biowarden"  # host only: the ledger and the block counts
    state_dir.mkdir(parents=True, exist_ok=True)
    real_dirs = [*map(str, tools.bin_dirs), "/usr/bin", *map(str, tools.tail_dirs)]
    watched = watched_tools(specs)
    touched: list[str] = []  # commands that named the layer, filled by the pre-check
    shim_dir = opaque_dir(root) if setting.shims else None
    sock_dir = opaque_dir(root)
    server, serving = None, False
    try:
        if shim_dir is not None:
            # a shim for every tool directory entry: hidden real tools stay reachable, and
            # programs no rule watches pass straight through the shim without asking the engine
            names = sorted({*watched, *tools.all_executables()})
            install_shims(shim_dir, real_dirs, names)
            if not os.access(shim_dir / "biowarden-shim", os.X_OK):  # False on a noexec mount
                raise OSError(
                    f"the shim cannot be executed from {root}; set {NEUTRAL_ROOT_VAR} to a "
                    "directory on a filesystem that allows executables"
                )
        ledger_path = state_dir / "ledger.jsonl"
        engine = Engine(
            specs,
            mode=setting.mode,
            promotion=checked.promotion,
            real_bins=tools.real_bins(watched),
            real_env=tools.process_env(),
            state=BlockState(state_dir / "blocks.json"),
            ledger=Ledger(ledger_path),
            style=setting.style,
            contract="bare" if contract == "bare" else "full",
        )
        paths = PathMap(work.resolve())
        server = EngineServer(sock_dir / SOCKET_NAME, engine, paths)
        serve_in_thread(server)
        serving = True

        def note(command: str) -> None:
            if TOUCHES_LAYER.search(command):
                touched.append(command[:200])

        def check(command: str) -> Precheck:
            note(command)
            try:
                return precheck(command, Path(WORKDIR), engine, paths, server.reported)
            except Exception as exc:  # a fault in the layer costs the check, never the command
                server.errors.append(f"precheck: {type(exc).__name__}: {str(exc)[:200]}")
                return Precheck("allow", "")

        def observe(command: str) -> Precheck:  # arms without a pre-check still log the touches
            note(command)
            return Precheck("allow", "")

        layer = None
        if shim_dir is not None:
            layer = SandboxLayer(
                shim_dir,
                server.socket_path,
                setting.hide_real,
                tuple(watched),
                check if setting.precheck else observe,
            )
        gateway = (
            make_gateway(engine, paths, work, tools, timeout, max_chars)
            if setting.gateway
            else None
        )
    except BaseException:
        if server is not None:
            if serving:
                server.shutdown()
            server.server_close()
        for path in (sock_dir, shim_dir):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)
        raise
    return LayerRig(
        layer,
        gateway,
        server,
        ledger_path,
        sock_dir,
        shim_dir,
        spec_hashes={spec.name: spec.content_hash() for spec in specs},
        promotion_sha=checked.sha,
        unpromoted=checked.unpromoted,
        touched=touched,
    )
