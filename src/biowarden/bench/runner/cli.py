"""Command line of the runner: ``run`` launches trials, ``report`` summarises them.

python -I -m biowarden.bench.runner.cli run --out runs/pilot --models haiku --seeds 0-1
python -I -m biowarden.bench.runner.cli report --out runs/pilot

A job that must run a frozen copy of the code uses plain ``python`` with PYTHONPATH set to the
copy's ``src`` (``-I`` ignores PYTHONPATH) and passes ``--snapshot`` so that this is checked.

A run leaves, beside trials/, in --out:
  plan.json    every planned trial id with its spec (all the runs that shared the directory)
  run.json     the configuration the records were made under; a later run must match it
  unrun.jsonl  trials the budget skipped (they have no score.json)
  jobs.jsonl   one line per run: planned, scored by outcome, infra, skipped, build failed, absent

One run at a time per directory; runs with different arms, seeds or families may follow one
another in it. Exit status: 0 every planned trial has a final record; 2 refused to start; 3 the run
ended with a trial absent or an infra_error still to be rerun (run the same command again:
finished trials are skipped, an infra_error is tried up to three times); 4 the run stopped early:
exhausted credit, an API outage (circuit breaker) or a promotion table that no longer matches the
specs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic
import openai

import biowarden
from biowarden.bench import analysis
from biowarden.bench import score as scoring
from biowarden.bench.runner import sandbox
from biowarden.bench.runner.agent import (
    ABORT_STOPS,
    CAREFUL_NOTE,
    LAYER_NOTE,
    AgentConfig,
    system_prompt,
)
from biowarden.bench.runner.agent_openai import (
    OPENROUTER_URL,
    PINNED_PROVIDERS,
    provider_routing,
)
from biowarden.bench.runner.exposure import trial_exposure
from biowarden.bench.runner.layer import (
    ARMS,
    SOCKET_NAME,
    SOCKET_PATH_MAX,
    UNPROMOTED_ALLOWED,
    PromotionCheck,
    PromotionMismatch,
    check_promotion,
)
from biowarden.bench.runner.opencode import OpenCodeRig, find_opencode_dir
from biowarden.bench.runner.pricing import ALIASES
from biowarden.bench.runner.proxy import ProxyServer, serve_in_thread
from biowarden.bench.runner.resources import (
    MINIMAL_WORKER_MB,
    OPENCODE_WORKER_MB,
    on_scheduler_job,
    plan_workers,
)
from biowarden.bench.runner.trial import (
    MAX_ATTEMPTS,
    Budget,
    Clients,
    RunInfo,
    TrialSpec,
    attempts_of,
    check_neutral_root,
    is_openrouter,
    record_crash,
    run_trial,
    utc_stamp,
    write_json_atomic,
)
from biowarden.bench.score import CONTRACTS
from biowarden.bench.tasks import ROUTES, TIERS, TWINS, ToolEnv, make_instance
from biowarden.bench.tasks.families import all_families, f4_detect_tokens
from biowarden.bench.tasks.tools import TOOLS_ENV_VAR
from biowarden.engine.decide import PACKAGED_PROMOTION
from biowarden.keys import get_key

ORDER_SEED = "tbm-family-order"  # fixed: the same plan is always run in the same order
ENV_LOCKS_DIR = Path(".local") / "env-locks"  # explicit package lists, hashed into the manifest

EXIT_REFUSED, EXIT_INCOMPLETE, EXIT_ABORTED = 2, 3, 4


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def plan(
    models: Sequence[str],
    families: Sequence[str],
    seeds: Sequence[int],
    reps: int,
    tier: str = "explicit",
    route: str = "free",
    contract: str = "full",
    harness: str = "minimal",
    arm: str = "bare",
) -> Iterator[TrialSpec]:
    """The planned trials: the one source of trial ids (the analysis regenerates them from here).

    Seed-major order, so a run cut short by the budget still covers every model evenly; inside a
    seed the families come in a fixed pseudo-random order, so a cut does not always drop the same
    ones. A model, family or seed that is listed twice is planned once (two workers would
    otherwise run one directory).

    F4 is a must-detect task with one fixed wording: it needs the warning channel (full contract)
    and ignores tier and route, so it is only planned for the default cell.
    """
    models = list(dict.fromkeys(models))
    seeds = list(dict.fromkeys(seeds))
    for rep in range(reps):
        for seed in seeds:
            order = sorted(set(families))
            random.Random(f"{ORDER_SEED}/{rep}/{seed}").shuffle(order)
            for family in order:
                if family == "F4" and (contract, tier, route) != ("full", "explicit", "free"):
                    continue
                for twin in TWINS:
                    for model in models:
                        yield TrialSpec(
                            model, family, seed, twin, rep, tier, route, contract, harness, arm
                        )


def load_records(root: Path, unreadable: list[Path] | None = None) -> list[dict[str, Any]]:
    """Every score.json under root/trials. One that vanished while the directory was read (a
    rerun removes the old trial) is skipped; one that does not parse is skipped and listed in
    `unreadable`."""
    found: list[dict[str, Any]] = []
    for path in sorted((root / "trials").glob("*/*/score.json")):
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or "outcome" not in record:
                raise ValueError("not a trial record")
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            if unreadable is not None:
                unreadable.append(path)
            continue
        found.append(record)
    return found


def cell_of(record: dict[str, Any]) -> str:
    """contract/tier/route, with the harness and the arm when they are not the default ones."""
    return analysis.cell_of(record)


def _pct(rate: float | None) -> str:
    return "   -" if rate is None else f"{100 * rate:4.1f}"


def _ci(interval: tuple[float, float] | None) -> str:
    return "" if interval is None else f"[{100 * interval[0]:.1f}-{100 * interval[1]:.1f}]"


def read_plan(root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((root / "plan.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("specs"), dict) else None


def unrun_ids(root: Path) -> set[str]:
    """The trials the budget skipped at some point (a later run may have done them since)."""
    found: set[str] = set()
    try:
        lines = (root / "unrun.jsonl").read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            entry = json.loads(line)
            if entry.get("reason") == "skipped_budget":
                found.add(entry["trial"])
        except (ValueError, KeyError, AttributeError):
            continue
    return found


def report(root: Path, out=sys.stdout) -> dict[str, Any]:
    skipped = ("infra_error", "skipped_budget")
    unreadable: list[Path] = []
    everything = load_records(root, unreadable)
    rows = [r for r in everything if r["outcome"] not in skipped]
    plan_json = read_plan(root)
    planned: Counter[tuple[str, str, str]] = Counter()  # trials planned per group
    for spec in (plan_json or {}).get("specs", {}).values():
        planned[(cell_of(spec), spec["twin"], spec["model"])] += 1
        planned[(cell_of(spec), spec["twin"], "ALL")] += 1
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for key in planned:
        groups.setdefault(key, [])  # a group with no record yet still has its row
    for r in rows:
        groups[(cell_of(r), r["twin"], r["model"])].append(r)
        groups[(cell_of(r), r["twin"], "ALL")].append(r)
    summary = {}
    for key, g in sorted(groups.items()):
        s = scoring.summarize(g)
        s["planned"] = planned[key] if plan_json else None
        s["silent"] = s["counts"].get("silent_wrong", 0)
        s["silent_planned"] = s["silent"] / s["planned"] if s["planned"] else None
        summary["|".join(key)] = s
    print("cell: contract/tier/route[+arm]", file=out)
    head = "  twin     model              n  plan  deliv  expo  SWR% [exact 95% CI]"
    print(head + "   SWR_exposed% [CI]         proto%  flag%  sil/plan%    usd", file=out)
    last = None
    for key, s in summary.items():
        cell, twin, model = key.split("|")
        if cell != last:
            print(cell, file=out)
            last = cell
        plan_n = "   -" if s["planned"] is None else f"{s['planned']:>4}"
        print(
            f"  {twin:<8} {model:<18} {s['n']:>3} {plan_n}  {s['delivered']:>4}  "
            f"{s['exposed']:>4}  {_pct(s['swr'])} {_ci(s['swr_ci']):<16} {_pct(s['swr_exposed'])} "
            f"{_ci(s['swr_exposed_ci']):<16} {_pct(s['protocol_fail_rate'])}  "
            f"{_pct(s['flag_rate'])}  {_pct(s['silent_planned']):>9}  {s['usd']:6.2f}",
            file=out,
        )
    by_family: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_family[(cell_of(r), r["family"], r["twin"])].append(r)
    print("\ncell/family/twin: outcomes (all models pooled), exposed", file=out)
    for (cell, family, twin), g in sorted(by_family.items()):
        counts = scoring.summarize(g)
        print(
            f"  {cell:<22} {family} {twin:<8} {counts['counts']} exposed {counts['exposed']}",
            file=out,
        )
    infra = sum(1 for r in everything if r["outcome"] == "infra_error")
    total = sum(r.get("usd", 0.0) for r in everything)
    line = f"\ntrials: {len(rows)} scored, {infra} infra errors"
    if plan_json:
        absent = len(set(plan_json["specs"]) - {r.get("trial") for r in everything})
        line += f", {absent} absent of {len(plan_json['specs'])} planned"
    if unreadable:
        line += f", {len(unreadable)} unreadable records"
    print(f"{line}; spend ${total:.2f}", file=out)
    return summary


def rescore(root: Path) -> int:
    """Re-score finished trials with the current rules; returns how many records changed.

    Exposure is judged as the runner judged it, on the commands that ran: a command the layer
    refused is not exposure (`exposed_issued` has the issued ones).
    """
    changed = 0
    for done in sorted((root / "trials").glob("*/*/score.json")):
        try:
            record = json.loads(done.read_text())
        except (OSError, ValueError):
            print(f"rescore: skipped {done.parent.name}: unreadable score.json", file=sys.stderr)
            continue
        if record.get("outcome") in ("refused", "infra_error", "skipped_budget"):
            continue  # decided by the runner, not by the scorer
        trial_dir = done.parent
        if not (trial_dir / "work").is_dir():  # nothing to score: not a protocol failure
            print(f"rescore: skipped {trial_dir.name}: no work directory", file=sys.stderr)
            continue
        try:
            task = json.loads((trial_dir / "task.json").read_text())
            transcript = json.loads((trial_dir / "transcript.json").read_text())
        except (OSError, ValueError) as exc:
            print(f"rescore: skipped {trial_dir.name}: {type(exc).__name__}", file=sys.stderr)
            continue
        contract = task.get("spec", {}).get("contract", "full")
        truth = task["truth"]
        if task["family"] == "F4" and "detect_tokens" not in truth:  # trials from before tokens
            truth = {**truth, "detect_tokens": f4_detect_tokens(make_instance(task["seed"]))}
        result, problem = scoring.read_result(trial_dir / "work")
        score = scoring.score_result(
            truth, result, problem or f"stopped: {record['stop']}", contract
        )
        found = trial_exposure(record["family"], transcript, trial_dir / "work")
        updated = {**record, **score.to_json(), **found, "scorer": scoring.SCORER_VERSION}
        if updated != record:
            write_json_atomic(done, updated)
            changed += 1
    return changed


# -- what a job writes down before it starts ----------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(obj: Any) -> str:
    return _sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def code_digest(package: Path | None = None) -> str:
    """sha256 of the lines `sha256  path` of every file of the package (caches excluded): the
    code a job imports, whatever it was installed from."""
    package = package or Path(biowarden.__file__).resolve().parent
    lines = []
    for path in sorted(package.rglob("*"), key=lambda p: p.relative_to(package).as_posix()):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or not path.is_file():
            continue
        lines.append(f"{_sha256(path.read_bytes())}  {path.relative_to(package).as_posix()}")
    return _sha256("\n".join(lines).encode())


def snapshot_problem(snapshot: str | None) -> str | None:
    """Why the imported package is not the declared snapshot; None if it is, or none is declared."""
    if snapshot is None:
        return None
    package = Path(biowarden.__file__).resolve().parent
    base = Path(snapshot).resolve()
    if package.is_relative_to(base):
        return None
    return (
        f"biowarden is imported from {package}, not from the snapshot {base}; run plain "
        "python with PYTHONPATH=<snapshot>/src (python -I ignores PYTHONPATH)"
    )


def tools_json_path(root: Path) -> Path | None:
    """The tools.json that ToolEnv.discover reads, if the tools came from one."""
    explicit = os.environ.get(TOOLS_ENV_VAR)
    for candidate in (Path(explicit) if explicit else None, root / ".local" / "tools.json"):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _first_line(argv: list[str], env: dict[str, str] | None = None) -> str | None:
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
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0] if lines else None


def tool_manifest(tools: ToolEnv, env_locks: Path | None, json_file: Path | None) -> dict[str, Any]:
    """Which tools a job runs: directories, versions, lock files, SDKs, sandbox. Its sha256 is
    stamped on every record; a run that resumes a directory must see the same manifest."""
    env = tools.process_env()
    bwrap = sandbox.bwrap_path()
    locks = None
    if env_locks is not None:
        locks = {
            p.name: _sha256(p.read_bytes()) for p in sorted(env_locks.iterdir()) if p.is_file()
        }
    return {
        "bin_dirs": [str(p) for p in tools.bin_dirs],
        "lib_dirs": [str(p) for p in tools.lib_dirs],
        "tail_dirs": [str(p) for p in tools.tail_dirs],
        "versions": {
            name: _first_line([str(path), "--version"], env)
            for name, path in sorted(tools.real_bins().items())
        },
        "tools_json_sha256": _sha256(json_file.read_bytes()) if json_file else None,
        "env_locks": locks,
        "bwrap_version": _first_line([bwrap, "--version"]) if bwrap else None,
        "sdk": {"anthropic": anthropic.__version__, "openai": openai.__version__},
        "python": platform.python_version(),
    }


def prompt_hashes() -> dict[str, str]:
    """sha256 of every piece of system prompt a run can use (the arms differ by one sentence)."""
    hashes = {name: _sha256(system_prompt(name).encode()) for name in CONTRACTS}
    hashes["layer_note"] = _sha256(LAYER_NOTE.encode())
    hashes["careful_note"] = _sha256(CAREFUL_NOTE.encode())
    return hashes


def run_config(
    args: argparse.Namespace, manifest: dict[str, Any], promotion: PromotionCheck
) -> dict[str, Any]:
    """Everything that decides what a record means and that every run of a directory shares.

    What differs between the runs of one directory (arm, seeds, families, workers, budget) is not.
    """
    agent = AgentConfig(
        "-", effort=args.effort, max_turns=args.max_turns, max_usd=args.max_trial_usd
    )
    return {
        "agent": {
            "effort": agent.effort,
            "max_turns": agent.max_turns,
            "max_tokens": agent.max_tokens,
            "command_timeout": agent.command_timeout,
            "max_output_chars": agent.max_output_chars,
            "max_trial_usd": agent.max_usd,
        },
        "pin": {
            "enabled": not args.no_pin,
            "routes": {m: provider_routing(m) for m in sorted(PINNED_PROVIDERS)},
        },
        "promotion": {"sha256": promotion.sha, "unpromoted": promotion.unpromoted},
        "prompts": prompt_hashes(),
        "scorer": scoring.SCORER_VERSION,
        "tools": manifest,
        "code": {"digest": code_digest()},
    }


_ABSENT = object()


def config_differences(old: Any, new: Any, prefix: str = "") -> list[str]:
    """The keys (dotted) whose values differ between two configurations."""
    if isinstance(old, dict) and isinstance(new, dict):
        found: list[str] = []
        for key in sorted(set(old) | set(new)):
            found += config_differences(
                old.get(key, _ABSENT), new.get(key, _ABSENT), f"{prefix}{key}."
            )
        return found

    def short(value: Any) -> str:
        return "<absent>" if value is _ABSENT else repr(value)[:60]

    return [] if old == new else [f"{prefix.rstrip('.')}: {short(old)} != {short(new)}"]


def settle_run_json(root: Path, config: dict[str, Any], job: dict[str, Any]) -> list[str]:
    """Write run.json for a new directory; for one that has it, check that this run matches it.

    Returns the problems that forbid the run (empty: go ahead). A directory with records and no
    run.json was written by an older runner and nothing says under which configuration.
    """
    path = root / "run.json"
    if not path.exists():
        if next((root / "trials").glob("*/*/score.json"), None) is not None:
            return [
                "the directory has records but no run.json (written by an older runner): "
                "use a fresh --out"
            ]
        write_json_atomic(
            path, {"config": config, "run_config_sha": canonical_sha(config), "jobs": [job]}
        )
        return []
    try:
        existing = json.loads(path.read_text())
        old = existing["config"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"run.json is unreadable ({type(exc).__name__}): use a fresh --out"]
    differences = config_differences(old, config)
    if differences:
        return [
            "run.json differs from this run's configuration (use a fresh --out, or the same "
            "settings as before):",
            *differences,
        ]
    write_json_atomic(path, {**existing, "jobs": [*existing.get("jobs", []), job]})
    return []


def update_plan(root: Path, specs: Sequence[TrialSpec], job: dict[str, Any]) -> None:
    """Add this run's trials to plan.json: every planned id, in the order the runs list them."""
    plan_json = read_plan(root) or {"format": 1, "ids": [], "specs": {}, "jobs": []}
    for spec in specs:
        if spec.trial_id not in plan_json["specs"]:
            plan_json["ids"].append(spec.trial_id)
            plan_json["specs"][spec.trial_id] = asdict(spec)
    plan_json["jobs"].append({**job, "n": len(specs)})
    write_json_atomic(root / "plan.json", plan_json)


def job_ledger(
    root: Path, specs: Sequence[TrialSpec], job_id: str, usd: float, aborted: str | None
) -> dict[str, Any]:
    """What the run came to, read from the records on disk: every planned trial is scored, an
    infra_error, skipped by the budget, or absent (no record at all)."""
    outcomes: Counter[str] = Counter()
    infra = to_rerun = build_failed = unreadable = 0
    absent: list[str] = []
    for spec in specs:
        path = root / "trials" / spec.trial_id / "score.json"
        try:
            record = json.loads(path.read_text())
            outcome = record["outcome"]
        except FileNotFoundError:
            absent.append(spec.trial_id)
            continue
        except (OSError, ValueError, KeyError, TypeError):
            unreadable += 1
            continue
        if outcome != "infra_error":
            outcomes[outcome] += 1
            continue
        infra += 1
        to_rerun += attempts_of(record) < MAX_ATTEMPTS
        build_failed += record.get("stop") == "build_failed"
    skipped = unrun_ids(root)
    return {
        "job_id": job_id,
        "ended": utc_stamp(),
        "planned": len(specs),
        "scored": dict(sorted(outcomes.items())),
        "infra": infra,
        "infra_to_rerun": to_rerun,
        "build_failed": build_failed,
        "skipped_budget": sum(1 for trial in absent if trial in skipped),
        "absent": len(absent),
        "unreadable": unreadable,
        "aborted": aborted,
        "usd": round(usd, 4),
    }


def exit_status(ledger: dict[str, Any]) -> int:
    if ledger["aborted"]:
        return EXIT_ABORTED
    if ledger["absent"] or ledger["infra_to_rerun"] or ledger["unreadable"]:
        return EXIT_INCOMPLETE
    return 0


def new_job_id() -> str:
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        return f"slurm-{slurm}" + (f"_{task}" if task else "")
    return f"local-{os.getpid()}-{int(time.time())}"


# -- run ----------------------------------------------------------------------------------------


class Refused(Exception):
    """The run cannot start; the message says why."""


def _mtime(path: Path) -> int | None:
    """When a file was last written (None: it is not there); tells a record just written from one
    that was already on disk."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def preflight(
    args: argparse.Namespace, tools: ToolEnv, root: Path, specs: Sequence[TrialSpec]
) -> RunInfo:
    """Check what the job depends on, then write run.json and plan.json (before any worker).

    Raises Refused when it cannot go on: a work directory that cannot run programs, a promotion
    table that does not match the specs, code that is not the declared snapshot, or a directory
    whose records were made under another configuration.
    """
    try:
        neutral = check_neutral_root()
        layered = ARMS[args.arm] is not None
        if layered and len(os.fsencode(neutral / ("0" * 16) / SOCKET_NAME)) > SOCKET_PATH_MAX:
            raise ValueError(f"{neutral} is too long for the engine's unix socket path")
        # a layer arm needs every rule promoted (bar the allow-list); the others only record it
        promotion = check_promotion(
            args.promotion, allow_unpromoted=UNPROMOTED_ALLOWED if layered else None
        )
        env_locks = Path(args.env_locks) if args.env_locks else Path.cwd() / ENV_LOCKS_DIR
        if args.env_locks and not env_locks.is_dir():
            raise FileNotFoundError(f"--env-locks {env_locks} is not a directory")
        manifest = tool_manifest(
            tools, env_locks if env_locks.is_dir() else None, tools_json_path(Path.cwd())
        )
        if manifest["env_locks"] is None:
            print(
                f"warning: no environment lock files in {env_locks}: run.json will not pin the "
                "packages of the conda environments (see --env-locks)",
                file=sys.stderr,
            )
    except (OSError, ValueError) as exc:  # PromotionMismatch is a ValueError
        raise Refused(str(exc)) from exc
    problem = snapshot_problem(args.snapshot)
    if problem:
        raise Refused(problem)
    config = run_config(args, manifest, promotion)
    info = RunInfo(
        config["code"]["digest"], canonical_sha(config), canonical_sha(manifest), new_job_id()
    )
    job = {
        "job_id": info.job_id,
        "started": utc_stamp(),
        "args": {k: v for k, v in vars(args).items() if k != "cmd"},
        "code_digest": info.code_digest,
        "package": str(Path(biowarden.__file__).resolve().parent),
        "snapshot": args.snapshot,
        "promotion_path": str(args.promotion or PACKAGED_PROMOTION),
        "neutral_root": str(neutral),
    }
    problems = settle_run_json(root, config, job)
    if problems:
        raise Refused("\n  ".join(problems))
    update_plan(root, specs, {k: job[k] for k in ("job_id", "started", "args", "code_digest")})
    return info


def cmd_run(args: argparse.Namespace) -> int:
    tools = ToolEnv.discover(Path.cwd())
    if tools is None:
        print("samtools, bcftools and bedtools not found (see .local/tools.json)", file=sys.stderr)
        return EXIT_REFUSED
    if not sandbox.available():
        print(
            "bubblewrap (bwrap) not found; refusing to run agent commands unsandboxed",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    models = [ALIASES.get(m, m) for m in args.models]
    seeds = parse_seeds(args.seeds)
    families = [f for f in dict.fromkeys(args.families) if f not in set(args.exclude_families)]
    specs = list(
        plan(
            models,
            families,
            seeds,
            args.reps,
            args.tier,
            args.route,
            args.contract,
            args.harness,
            args.arm,
        )
    )
    if args.limit:
        specs = specs[: args.limit]
    ids = [spec.trial_id for spec in specs]
    if not specs or len(set(ids)) != len(ids):
        print(
            "the plan is empty" if not specs else "the plan has trial ids that appear twice",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    clients = Clients()
    if any(not is_openrouter(m) for m in models):
        key = get_key("anthropic")
        if not key:
            print("no Anthropic key found (python -I -m biowarden.keys)", file=sys.stderr)
            return EXIT_REFUSED
        clients.anthropic = anthropic.Anthropic(api_key=key, max_retries=4)
    if any(is_openrouter(m) for m in models):
        key = get_key("openrouter")
        if not key:
            print("no OpenRouter key found (python -I -m biowarden.keys)", file=sys.stderr)
            return EXIT_REFUSED
        clients.openrouter = openai.OpenAI(
            api_key=key,
            base_url=OPENROUTER_URL,
            max_retries=4,
            timeout=600,
            default_headers={"X-Title": "biowarden"},
        )
    per_worker_mb = OPENCODE_WORKER_MB if args.harness == "opencode" else MINIMAL_WORKER_MB
    if args.harness == "opencode" and not (on_scheduler_job() or args.allow_login_node):
        print(
            "refusing to run OpenCode outside a scheduler job: login nodes cap each user at a few "
            "GiB and kill the session when it is exceeded. Submit a job (see .local/scripts) or "
            "pass --allow-login-node to run a few workers anyway.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    workers = plan_workers(args.workers, per_worker_mb)
    if workers < args.workers:
        print(
            f"workers reduced from {args.workers} to {workers} to fit the node: outside a "
            "scheduler job at most 2 run, and the cgroup memory limit applies",
            flush=True,
        )
    if args.harness == "opencode" and args.arm not in ("bare", "block-shim"):
        print("OpenCode runs only the bare and block-shim arms", file=sys.stderr)
        return EXIT_REFUSED
    if args.harness == "opencode" and not all(is_openrouter(m) for m in models):
        print("the opencode harness is wired to OpenRouter models only", file=sys.stderr)
        return EXIT_REFUSED
    try:
        info = preflight(args, tools, root, specs)
    except Refused as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    budget = Budget(args.max_usd)
    lock = threading.Lock()
    abort = threading.Event()  # set when the rest of the plan cannot be run
    stopped: list[str] = []

    def stop(reason: str) -> None:
        with lock:
            if not stopped:
                stopped.append(reason)
                print(f"STOP: {reason}; trials not started are left for the next run", flush=True)
        abort.set()

    def work(spec: TrialSpec) -> dict[str, Any] | None:
        if abort.is_set():
            return None  # not started: absent from the records, counted in the ledger
        cfg = AgentConfig(
            spec.model,
            effort=args.effort,
            max_turns=args.max_turns,
            max_usd=args.max_trial_usd,
            promotion_path=args.promotion,
            provider=provider_routing(spec.model, not args.no_pin),
        )
        done = root / "trials" / spec.trial_id / "score.json"
        before = _mtime(done)
        try:
            record = run_trial(spec, root, clients, tools, cfg, budget, info)
        except PromotionMismatch as exc:  # the specs or the table changed under the job
            stop(f"the promotion table no longer matches the specs: {exc}")
            return None
        except Exception as exc:  # last resort: one trial can never stop the pool
            try:
                record = record_crash(spec, root, exc, info)
            except Exception as inner:
                print(f"{spec.trial_id}: lost ({exc!r}; {inner!r})", file=sys.stderr, flush=True)
                return None
        try:  # the progress line is a courtesy: it must never cost the run its ledger
            why = (record.get("reason") or "")[:100] if record["outcome"] == "infra_error" else ""
            line = (
                f"{record['trial']:<46} {record['outcome']:<13} "
                f"turns={record.get('turns', '-')!s:<3} ${record.get('usd', 0):.3f}  "
                f"total ${budget.spent:.2f}  {why}".rstrip()
            )
            with lock:
                print(line.encode("ascii", "backslashreplace").decode(), flush=True)
        except Exception as exc:
            print(f"{spec.trial_id}: progress line failed ({exc!r})", file=sys.stderr, flush=True)
        # only a record this call wrote can stop the run: an earlier run may have left a final one
        if _mtime(done) != before and record.get("stop") in ABORT_STOPS:
            stop(f"{record['stop']}: {record.get('reason') or record.get('error') or ''}"[:300])
        return record

    proxy = None
    if args.harness == "opencode":
        key = get_key("openrouter")
        if not key:
            print("no OpenRouter key found (python -I -m biowarden.keys)", file=sys.stderr)
            return EXIT_REFUSED
        binary_dir = find_opencode_dir()
        if binary_dir is None:
            print("opencode not installed (see .local/opencode)", file=sys.stderr)
            return EXIT_REFUSED
        proxy = ProxyServer(
            root / f"proxy-{os.getpid()}.sock",
            key,
            set(models),
            root / "proxylogs",
            args.max_trial_usd,
        )
        serve_in_thread(proxy)
        clients.opencode = OpenCodeRig(binary_dir, proxy)

    print(
        f"{len(specs)} trials, models {models}, budget ${args.max_usd}, job {info.job_id}",
        flush=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, specs))
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
            proxy.socket_path.unlink(missing_ok=True)
    ledger = job_ledger(root, specs, info.job_id, budget.spent, stopped[0] if stopped else None)
    print("ledger: " + json.dumps(ledger, sort_keys=True), flush=True)
    with open(root / "jobs.jsonl", "a") as handle:
        handle.write(json.dumps(ledger, sort_keys=True) + "\n")
    try:
        report(root)
    except Exception as exc:  # the summary is a convenience; the ledger above is the verdict
        print(f"report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = exit_status(ledger)
    if status == EXIT_INCOMPLETE:
        print(
            f"INCOMPLETE: {ledger['absent']} planned trials have no record "
            f"({ledger['skipped_budget']} skipped by the budget) and {ledger['infra_to_rerun']} "
            "ended in infra_error with attempts left; run the same command again",
            file=sys.stderr,
        )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biowarden-bench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run pilot trials (calls the Claude API; spends credits)")
    run.add_argument("--out", required=True)
    run.add_argument("--models", nargs="+", default=["haiku"], help="haiku sonnet opus or full ids")
    run.add_argument(
        "--families", nargs="+", default=list(all_families()), choices=list(all_families())
    )
    run.add_argument(
        "--exclude-families",
        nargs="+",
        default=[],
        choices=list(all_families()),
        metavar="FAMILY",
        help="families to leave out of --families (the default is all of them)",
    )
    run.add_argument("--seeds", default="0-1", help="e.g. 0-9 or 0,3,5-7")
    run.add_argument("--reps", type=int, default=1)
    run.add_argument("--tier", default="explicit", choices=list(TIERS))
    run.add_argument(
        "--route", default="free", choices=list(ROUTES), help="forced: dictate the tool"
    )
    run.add_argument("--contract", default="full", choices=list(CONTRACTS))
    run.add_argument("--harness", default="minimal", choices=["minimal", "opencode"])
    run.add_argument(
        "--arm",
        default="bare",
        choices=sorted(ARMS),
        help="checking layer: bare (none), warn, block, block-bypass, blockall (development)",
    )
    run.add_argument("--promotion", default=None, help="promotion table for the block arms")
    run.add_argument(
        "--no-pin",
        action="store_true",
        help="let OpenRouter pick the provider (default: pin each model to one provider)",
    )
    run.add_argument(
        "--allow-login-node", action="store_true", help="run OpenCode outside a scheduler job"
    )
    run.add_argument(
        "--snapshot",
        default=None,
        help="the code snapshot this run must import from (refuses to start otherwise)",
    )
    run.add_argument(
        "--env-locks",
        default=None,
        help=f"directory of environment lock files hashed into run.json (default: {ENV_LOCKS_DIR})",
    )
    run.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"]
    )
    run.add_argument("--max-turns", type=int, default=30)
    run.add_argument(
        "--max-usd", type=float, default=5.0, help="stop launching trials past this total"
    )
    run.add_argument("--max-trial-usd", type=float, default=0.75)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--limit", type=int, default=0, help="only the first N trials of the plan")
    rep = sub.add_parser("report", help="summarise finished trials")
    rep.add_argument("--out", required=True)
    res = sub.add_parser("rescore", help="re-score finished trials with the current rules")
    res.add_argument("--out", required=True)
    ana = sub.add_parser("analyze", help="clean-versus-faulted comparison per cell and model")
    ana.add_argument("--out", required=True, nargs="+", help="one or more run directories")
    args = parser.parse_args(argv)
    if args.cmd == "report":
        report(Path(args.out))
        return 0
    if args.cmd == "analyze":
        records = analysis.load_records(args.out)
        print(analysis.render(analysis.compare(records)))
        print("\nEfficiency (twins pooled)")
        print(analysis.render_efficiency(analysis.efficiency(records)))
        return 0
    if args.cmd == "rescore":
        print(f"{rescore(Path(args.out))} records changed")
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
