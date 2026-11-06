"""One trial = build a workspace, let the agent work in the sandbox, score result.json.

A trial directory holds everything needed to audit it: task.json (with the truth, outside the
sandbox), transcript.json, score.json and the work directory. The agent works in a directory with
a random name under $BIOWARDEN_NEUTRAL_ROOT, so nothing the sandbox can read (its mounts, /proc)
names the trial; the scoring reads it there, and when the sandbox processes are finished it is
moved to trial_dir/work, which task.json records. score.json is written last and atomically: its
presence means the trial is complete.

A finished trial is never re-run, so an interrupted run resumes where it stopped. Every trial that
starts leaves exactly one record: a builder failure or any unexpected exception is persisted as an
infra_error with its traceback, and the trial is run again until it has had MAX_ATTEMPTS tries.
A trial the budget skips leaves no score.json (a record would block the resume after a top-up) but
a line in unrun.jsonl.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biowarden.bench.runner.agent import INFRA_STOPS, AgentConfig, run_agent
from biowarden.bench.runner.agent_openai import run_agent_openai
from biowarden.bench.runner.exposure import trial_exposure
from biowarden.bench.runner.layer import ARMS, PromotionMismatch, start_layer
from biowarden.bench.runner.opencode import opencode_trajectory
from biowarden.bench.score import SCORER_VERSION, Score, read_result, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.locus import locus_of

NEUTRAL_ROOT_VAR = "BIOWARDEN_NEUTRAL_ROOT"  # where the work, shim and socket directories are made
MAX_ATTEMPTS = 3  # a trial that ended in infra_error is run again until it has had this many tries
TRACEBACK_CHARS = 6000
# Words that name a trial, its arm or its model. The sandbox can read the host path of what it
# binds (in /proc), so a neutral root that spells one of them gives the trial away again.
IDENTIFYING = re.compile(
    r"faulted|clean|seed|tier|route|explicit|implicit|forced|bare|note|warn|block|careful"
    r"|haiku|claude|sonnet|opus|deepseek|gpt|openai|qwen|mimo|glm|(?<![A-Za-z0-9])F[0-9]+(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_UNRUN_LOCK = threading.Lock()


@dataclass
class Clients:
    """One API client per provider; a model id with a slash goes through OpenRouter."""

    anthropic: Any = None
    openrouter: Any = None
    opencode: Any = None  # an OpenCodeRig


@dataclass(frozen=True)
class RunInfo:
    """What a job stamps on every record it writes (the CLI computes it once per job)."""

    code_digest: str  # of the biowarden package the job imports
    run_config_sha: str  # of the run configuration in run.json (the same for every arm)
    tool_manifest_sha: str  # of the tool manifest in run.json (versions, locks, SDKs)
    job_id: str


def is_openrouter(model: str) -> bool:
    return "/" in model


@dataclass(frozen=True)
class TrialSpec:
    model: str
    family: str
    seed: int
    twin: str
    rep: int = 0
    tier: str = "explicit"
    route: str = "free"
    contract: str = "full"
    harness: str = "minimal"
    arm: str = "bare"

    @property
    def trial_id(self) -> str:
        extra = "".join(
            f"-{value}"
            for value, default in (
                (self.tier, "explicit"),
                (self.route, "free"),
                (self.contract, "full"),
                (self.harness, "minimal"),
                (self.arm, "bare"),
            )
            if value != default
        )
        model = self.model.replace("/", "__")  # keep the trial directory two levels deep
        return f"{model}/{self.family}-s{self.seed:02d}-{self.twin}{extra}-r{self.rep}"


class Budget:
    """Shared spend counter; the runner stops launching trials once it is exhausted."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    def charge(self, usd: float) -> None:
        with self._lock:
            self._spent += usd

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def exhausted(self) -> bool:
        return self._spent >= self.max_usd


# -- files: atomic writes, deleting what an agent made, moving the work directory ---------------


def utc_stamp(when: float | None = None) -> str:
    moment = datetime.fromtimestamp(time.time() if when is None else when, timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, data: Any, indent: int | None = 1) -> None:
    """Write JSON so that a reader, or a crash, sees the old file or the whole new one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w") as handle:
            json.dump(data, handle, indent=indent)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_writable(path: Path) -> None:
    """chmod -R u+rwX, never through a symbolic link: an agent can leave directories nobody can
    enter and files nobody can delete."""

    def fix(name: str) -> None:
        if os.path.islink(name):
            return
        try:
            mode = os.lstat(name).st_mode
            more = stat.S_IRUSR | stat.S_IWUSR
            if stat.S_ISDIR(mode) or mode & 0o111:
                more |= stat.S_IXUSR
            os.chmod(name, mode | more)
        except OSError:
            pass

    fix(str(path))
    for dirpath, dirnames, filenames in os.walk(path):  # a directory is fixed before it is listed
        for name in (*dirnames, *filenames):
            fix(os.path.join(dirpath, name))


def remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    make_writable(path)
    shutil.rmtree(path)


def _copy_regular(src: str, dst: str) -> None:
    if stat.S_ISREG(os.lstat(src).st_mode):  # a FIFO would block the copy, a device is no data
        shutil.copy2(src, dst)


def move_tree(src: Path, dst: Path) -> None:
    """Rename `src` to `dst`; across filesystems copy it (links stay links, special files are
    left out) and then remove it. If the copy fails `src` is untouched."""
    try:
        os.rename(src, dst)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    make_writable(src)
    try:
        shutil.copytree(src, dst, symlinks=True, copy_function=_copy_regular)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    remove_tree(src)


def neutral_root() -> Path:
    """Where the sandbox's directories are made (the same variable layer.py uses for its own)."""
    return Path(os.environ.get(NEUTRAL_ROOT_VAR) or tempfile.gettempdir()).resolve()


def opaque_work_dir() -> Path:
    """A new private directory named by 16 random hex digits (it spells no word) under the
    neutral root. The sandbox binds it as /work, and /proc shows the host path of that bind."""
    root = neutral_root()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / secrets.token_hex(8)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return path


def check_neutral_root() -> Path:
    """Make a work directory and see that a program can be run from it; returns the root.

    The agent runs the scripts it writes in /work. On a noexec mount (some nodes mount /tmp so)
    they would fail, which no earlier run saw, so a job refuses to start there. It refuses a root
    whose path names a trial, an arm or a model too, since the sandbox can read it.
    """
    named = IDENTIFYING.search(str(neutral_root()))
    if named:
        raise OSError(
            f"{NEUTRAL_ROOT_VAR}={neutral_root()} contains {named.group(0)!r}, and the sandbox "
            "can read that path: choose a directory whose path names no trial, arm or model"
        )
    path = opaque_work_dir()
    try:
        probe = path / "probe.sh"
        probe.write_text("#!/bin/sh\n")
        probe.chmod(0o700)
        if not os.access(probe, os.X_OK):
            raise OSError(
                f"programs cannot be run from {neutral_root()} (a noexec mount?); set "
                f"{NEUTRAL_ROOT_VAR} to a directory on a filesystem that allows executables"
            )
    finally:
        shutil.rmtree(path, ignore_errors=True)
    return neutral_root()


# -- records ----------------------------------------------------------------------------------


def attempts_of(record: dict[str, Any] | None) -> int:
    """How many times the trial has been tried; a record from before the counter had one try."""
    return 0 if record is None else int(record.get("attempt") or 1)


def _load_previous(root: Path, done: Path, trial_id: str) -> dict[str, Any] | None:
    """The record of an earlier run. A file that does not parse (a run killed while writing it
    before writes were atomic) is moved aside, outside the trial directory, and counts as none."""
    try:
        text = done.read_text()
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
        if isinstance(record, dict) and "outcome" in record:
            return record
    except ValueError:
        pass
    aside = root / "quarantine"
    aside.mkdir(parents=True, exist_ok=True)
    stem = trial_id.replace("/", "__")
    number = 0
    while (target := aside / f"{stem}.score.json.{number}").exists():
        number += 1
    os.replace(done, target)
    return None


def note_unrun(root: Path, spec: TrialSpec, reason: str, info: RunInfo | None = None) -> None:
    """Append a planned trial that was not run to unrun.jsonl (history: later runs may do it)."""
    line = {
        "trial": spec.trial_id,
        "reason": reason,
        "at": utc_stamp(),
        "job_id": info.job_id if info else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    with _UNRUN_LOCK, open(root / "unrun.jsonl", "a") as handle:
        handle.write(json.dumps(line) + "\n")


def _provenance(attempt: int, stamp: str, info: RunInfo | None) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "started": stamp,
        "job_id": info.job_id if info else None,
        "code_digest": info.code_digest if info else None,
        "run_config_sha": info.run_config_sha if info else None,
        "tool_manifest_sha": info.tool_manifest_sha if info else None,
    }


def _failure(
    spec: TrialSpec,
    reason: str,
    attempt: int,
    stamp: str,
    info: RunInfo | None,
    usd: float = 0.0,
    exc: BaseException | None = None,
    stop: str = "crashed",
) -> dict[str, Any]:
    """The record of a trial that could not be run or scored: an infra_error, rerun later."""
    text = "".join(traceback.format_exception(exc))[-TRACEBACK_CHARS:] if exc else None
    return {
        "trial": spec.trial_id,
        **asdict(spec),
        **Score("infra_error", False, False, False, False, reason=reason).to_json(),
        "scorer": SCORER_VERSION,
        "stop": stop,
        "usd": round(usd, 5),
        "traceback": text,
        **_provenance(attempt, stamp, info),
    }


def record_crash(
    spec: TrialSpec, root: Path, exc: BaseException, info: RunInfo | None = None
) -> dict[str, Any]:
    """Persist an infra_error for a trial whose run_trial raised: the last resort of a worker.
    A finished record is never replaced."""
    trial_dir = root.resolve() / "trials" / spec.trial_id
    done = trial_dir / "score.json"
    try:
        previous = json.loads(done.read_text())
    except (OSError, ValueError):
        previous = None
    if isinstance(previous, dict) and previous.get("outcome") not in (None, "infra_error"):
        return previous
    attempt = attempts_of(previous if isinstance(previous, dict) else None) + 1
    reason = f"crashed: {type(exc).__name__}: {str(exc)[:200]}"
    record = _failure(spec, reason, attempt, utc_stamp(), info, exc=exc)
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(done, record)
    return record


# -- one trial ---------------------------------------------------------------------------------


def run_trial(
    spec: TrialSpec,
    root: Path,
    client: Clients | Any,
    tools: ToolEnv,
    cfg: AgentConfig,
    budget: Budget,
    info: RunInfo | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    trial_dir = root / "trials" / spec.trial_id  # builders may run tools with cwd=work
    done = trial_dir / "score.json"
    previous = _load_previous(root, done, spec.trial_id)
    if previous is not None and (
        previous.get("outcome") != "infra_error" or attempts_of(previous) >= MAX_ATTEMPTS
    ):
        return previous  # finished (an infra_error is final once it has had its attempts)
    if budget.exhausted:
        note_unrun(root, spec, "skipped_budget", info)
        return {"trial": spec.trial_id, **asdict(spec), "outcome": "skipped_budget", "usd": 0.0}
    attempt = attempts_of(previous) + 1
    if trial_dir.exists():
        remove_tree(trial_dir)
    trial_dir.mkdir(parents=True)
    stamp = utc_stamp()
    spent = [0.0]  # what the agent cost, known to the failure record even if a later step raises
    work: Path | None = None
    settled = False
    try:
        try:
            work = opaque_work_dir()
            record = _run(
                spec, trial_dir, work, client, tools, cfg, budget, spent, attempt, stamp, info
            )
        except PromotionMismatch:
            raise  # the specs or the table changed under the job: a fault of the job, not the trial
        except Exception as exc:  # whatever went wrong costs this trial, never the run
            reason = f"crashed: {type(exc).__name__}: {str(exc)[:200]}"
            record = _failure(spec, reason, attempt, stamp, info, spent[0], exc)
        settled = True
        if work is not None:
            record = _keep_work(record, work, trial_dir, spec, attempt, stamp, info, spent[0])
        write_json_atomic(done, record)
        return record
    except PromotionMismatch:
        shutil.rmtree(trial_dir, ignore_errors=True)
        raise
    finally:
        if work is not None and not settled:  # an interrupt or a refused launch: leave no orphan
            make_writable(work)
            shutil.rmtree(work, ignore_errors=True)


def _run(
    spec: TrialSpec,
    trial_dir: Path,
    work: Path,
    client: Clients | Any,
    tools: ToolEnv,
    cfg: AgentConfig,
    budget: Budget,
    spent: list[float],
    attempt: int,
    stamp: str,
    info: RunInfo | None,
) -> dict[str, Any]:
    """Build, play and score one trial; the work directory is still where the agent used it."""
    try:
        task = build_task(spec.family, spec.seed, spec.twin, work, tools, spec.tier, spec.route)
    except Exception as exc:  # a broken builder must cost one trial, not the whole run
        reason = f"build failed: {type(exc).__name__}: {str(exc)[:200]}"
        return _failure(spec, reason, attempt, stamp, info, exc=exc, stop="build_failed")
    where = {"opaque": str(work), "final": str(trial_dir / "work")}  # the trial id is not in it
    write_json_atomic(
        trial_dir / "task.json", {**asdict(task), "spec": asdict(spec), "work_dir": where}
    )
    started = time.time()
    rig = None
    if spec.arm != "bare":
        if spec.harness != "minimal" and spec.arm != "block-shim":
            raise ValueError(
                "OpenCode runs only the bare and block-shim arms (shims, no pre-check)"
            )
        rig = start_layer(
            spec.arm,
            trial_dir,
            work,
            tools,
            cfg.promotion_path,
            cfg.command_timeout,
            cfg.max_output_chars,
            contract=spec.contract,
        )
    cfg = replace(
        cfg,
        contract=spec.contract,
        layer=rig.layer if rig else None,
        gateway=rig.gateway if rig else None,
        layer_note=spec.arm == "note" or (rig is not None and ARMS[spec.arm].note),
        careful_note=spec.arm == "careful",
    )
    layer_summary = None
    try:
        traj = _drive(spec, client, tools, cfg, task, work, trial_dir)
    finally:
        if rig is not None:
            try:
                layer_summary = rig.summary()
            finally:
                rig.close()
    budget.charge(traj.usd)
    spent[0] = traj.usd
    write_json_atomic(trial_dir / "transcript.json", traj.messages, indent=None)
    pin = (cfg.provider or {}).get("order")
    return _finish(spec, traj, task, work, started, layer_summary, pin, attempt, stamp, info)


def _drive(
    spec: TrialSpec,
    client: Clients | Any,
    tools: ToolEnv,
    cfg: AgentConfig,
    task: Any,
    work: Path,
    trial_dir: Path,
) -> Any:
    if spec.harness == "opencode":
        key = spec.trial_id.replace("/", "--")  # url-safe id the proxy files its record under
        traj = opencode_trajectory(
            client.opencode,
            tools,
            work,
            trial_dir,
            spec.model,
            key,
            task.prompt,
            spec.contract,
            layer=cfg.layer,
        )
    elif is_openrouter(spec.model):
        chosen = client.openrouter if isinstance(client, Clients) else client
        traj = run_agent_openai(chosen, cfg, task.prompt, work, tools)
    else:
        chosen = client.anthropic if isinstance(client, Clients) else client
        traj = run_agent(chosen, cfg, task.prompt, work, tools)
    return traj


def _finish(
    spec: TrialSpec,
    traj: Any,
    task: Any,
    work: Path,
    started: float,
    layer_summary: dict[str, Any] | None,
    provider_pin: list[str] | None,
    attempt: int,
    stamp: str,
    info: RunInfo | None,
) -> dict[str, Any]:
    """Score the trial and describe it in one record (read from `work`, nothing is written)."""
    if traj.stop == "refusal":
        score = Score("refused", False, False, False, False, reason=traj.refusal_category or "")
    elif traj.stop in INFRA_STOPS:
        score = Score("infra_error", False, False, False, False, reason=traj.error or "")
    else:
        result, problem = read_result(work)
        score = score_result(task.truth, result, problem or f"stopped: {traj.stop}", spec.contract)
    return {
        "trial": spec.trial_id,
        **asdict(spec),
        "slot": task.slot,
        "locus": locus_of(task.family),
        **score.to_json(),
        **trial_exposure(spec.family, traj.messages, work),
        "scorer": SCORER_VERSION,
        "stop": traj.stop,
        "turns": traj.turns,
        "commands": traj.commands,
        "tool_arg_errors": traj.tool_arg_errors,
        "usd": round(traj.usd, 5),
        "tokens": dict(traj.tokens),
        "served_by": traj.served_by,
        "providers": sorted(getattr(traj, "providers", [])),
        "provider_pin": provider_pin,
        "pin_match": getattr(traj, "pin_match", None),
        "finish_reasons": dict(getattr(traj, "finish_reasons", None) or {}),
        "refusal_category": traj.refusal_category,
        "layer": layer_summary,
        "error": traj.error,
        "secs": round(time.time() - started, 1),
        **_provenance(attempt, stamp, info),
    }


def _keep_work(
    record: dict[str, Any],
    work: Path,
    trial_dir: Path,
    spec: TrialSpec,
    attempt: int,
    stamp: str,
    info: RunInfo | None,
    usd: float,
) -> dict[str, Any]:
    """Move the work directory to trial_dir/work, the sandbox processes being finished.

    If it cannot be moved the data stay where they are (the record says so) and a trial that
    cannot be audited is an infra_error, whatever it scored.
    """
    try:
        move_tree(work, trial_dir / "work")
    except Exception as exc:
        problem = f"work directory kept at {work}: {type(exc).__name__}: {str(exc)[:200]}"
        if record["outcome"] == "infra_error":
            return {**record, "reason": f"{record.get('reason', '')}; {problem}"[:600]}
        return _failure(spec, problem, attempt, stamp, info, usd, exc)
    return record
