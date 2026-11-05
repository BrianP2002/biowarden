"""Checks to run before a stage-2 job, and the schedule of a job.

Run it with the code the jobs run: plain `python` with PYTHONPATH=<snapshot>/src (`python -I`
ignores PYTHONPATH) and `--snapshot <snapshot>`, which refuses to go on when the imported package
lies anywhere else.

    preflight.py endpoints [--models dsflash gptoss]    OpenRouter: the pin tags exist, with tools
                                                        and reasoning support (public GET)
    preflight.py credits --model dsflash --need 4       OpenRouter balance covers the job (GET)
    preflight.py smoke --model dsflash [--n 20]         20 pinned trials: one provider name, usd >
                                                        0, reasoning tokens > 0 (spends cents)
    preflight.py probe --model dsflash [--n 40]         40 trials at the planned worker count with
                                                        max_retries=0: the share of 429/5xx
    preflight.py blocks|chunks ...                      what tools/stage2_job.sbatch runs
    preflight.py plan-manifest --sets F --out F         every planned trial id, by digest, for
                                                        freeze 2
    preflight.py verify --out D --model M --cell C ...  the job's records against the frozen plan

Exit status: 0 all good, 1 a check failed, 2 it could not run (usage, missing key, snapshot).
Only endpoints, credits, smoke and probe call an API; keys are read through
science_harness.keys and are never printed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "science_harness"
OPENROUTER = "https://openrouter.ai/api/v1"
SMOKE_FAMILIES = ("F1", "F2", "F3", "F5", "F6", "F7", "F8", "F9", "F10", "F11")  # x 2 twins = 20
PROBE_FAMILIES = ("F1", "F2", "F3", *(f"F{n}" for n in range(5, 22)))  # 20 x 2 twins = 40
MAX_ATTEMPTS = 3  # the runner's: an infra_error with this many tries is final
MAX_ERROR_SHARE = 0.05  # probe: more 429/5xx than this among the requests fails the check
Fetch = Callable[..., Any]


class Refused(Exception):
    """The check cannot run (exit 2), as opposed to running and failing (exit 1)."""


def say(text: str = "") -> None:
    print(text, flush=True)


# -- which package is imported ----------------------------------------------------------------


def check_snapshot(snapshot: str | None) -> None:
    if snapshot is None:
        print("note: not checking which copy of the package is imported", file=sys.stderr)
        return
    package = __import__(PACKAGE)
    found = Path(package.__file__).resolve().parent
    base = Path(snapshot).resolve()
    if not found.is_relative_to(base):
        raise Refused(
            f"{PACKAGE} is imported from {found}, not from the snapshot {base}; run plain "
            "python with PYTHONPATH=<snapshot>/src (python -I ignores it)"
        )


def model_id(name: str) -> str:
    from science_harness.bench.runner.pricing import ALIASES

    return ALIASES.get(name, name)


# -- OpenRouter -------------------------------------------------------------------------------


def fetch_json(url: str, key: str | None = None, timeout: float = 30.0, opener: Any = None) -> Any:
    """GET `url` and parse the JSON body (the key, if any, goes in the Authorization header)."""
    headers = {"Accept": "application/json", "User-Agent": "stage2-preflight"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Refused(f"GET {url}: {exc}") from exc


def endpoint_report(model: str, tag: str, payload: Any) -> tuple[str, list[str]]:
    """(one line, problems) for the pin `tag` of `model` in the /models/<id>/endpoints reply."""
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = (data or {}).get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        return f"{model:<32}{tag:<14}", [f"{model}: no endpoint list in the reply"]
    tags = sorted({str(e.get("tag")) for e in endpoints if isinstance(e, dict)})
    found = [e for e in endpoints if isinstance(e, dict) and e.get("tag") == tag]
    if not found:
        return f"{model:<32}{tag:<14}MISSING", [f"{model}: no endpoint tagged {tag!r}; has {tags}"]
    problems = []
    for endpoint in found:
        params = set(endpoint.get("supported_parameters") or [])
        if "tools" not in params:
            problems.append(f"{model} {tag}: no tool support")
        if not params & {"reasoning", "include_reasoning"}:
            problems.append(f"{model} {tag}: no reasoning support")
    shown = found[0]
    detail = f"quantization={shown.get('quantization')} uptime={shown.get('uptime_last_30m')}"
    return f"{model:<32}{tag:<14}{'FAIL' if problems else 'ok  '}  {detail}", problems


def pinned_models(names: Sequence[str] | None) -> dict[str, str]:
    from science_harness.bench.runner.agent_openai import PINNED_PROVIDERS

    if not names:
        return dict(PINNED_PROVIDERS)
    wanted = [model_id(n) for n in names]
    unknown = [m for m in wanted if m not in PINNED_PROVIDERS]
    if unknown:
        raise Refused(f"no provider pin for {', '.join(unknown)}")
    return {m: PINNED_PROVIDERS[m] for m in wanted}


def cmd_endpoints(args: argparse.Namespace, fetch: Fetch = fetch_json) -> int:
    check_snapshot(args.snapshot)
    from science_harness.keys import get_key

    key = get_key("openrouter")  # the listing is public; a key does no harm if it asks for one
    failed = False
    for model, tag in pinned_models(args.models).items():
        reply = fetch(f"{OPENROUTER}/models/{model}/endpoints", key)
        line, problems = endpoint_report(model, tag, reply)
        say(line)
        for problem in problems:
            say(f"  {problem}")
        failed |= bool(problems)
    return 1 if failed else 0


def remaining_credit(payload: Any) -> float:
    try:
        data = payload["data"]
        return float(data["total_credits"]) - float(data["total_usage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused(f"unexpected reply from /credits: {str(payload)[:120]}") from exc


def cmd_credits(args: argparse.Namespace, fetch: Fetch = fetch_json) -> int:
    check_snapshot(args.snapshot)
    model = model_id(args.model)
    if "/" not in model:
        say(f"credits: {model} is billed by Anthropic, which has no balance endpoint; not checked")
        return 0
    from science_harness.keys import get_key

    key = get_key("openrouter")
    if not key:
        raise Refused("no OpenRouter key found (python -I -m science_harness.keys)")
    left = remaining_credit(fetch(f"{OPENROUTER}/credits", key))
    enough = left >= args.need
    say(f"credits: ${left:.2f} left, ${args.need:.2f} needed: {'ok' if enough else 'NOT ENOUGH'}")
    return 0 if enough else 1


# -- trials: the smoke test and the probe -----------------------------------------------------


def fresh_directory(out: Path) -> Path:
    """`out`, which must hold no trial yet: finished trials are skipped, so a second run into
    it would judge the records of the first."""
    if (out / "trials").exists():
        raise Refused(f"{out} already holds trials; use a fresh --out")
    return out


def read_records(out: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((out / "trials").glob("*/*/score.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return records


def smoke_problems(model: str, records: list[dict[str, Any]], wanted: int) -> list[str]:
    """Why the pinned smoke test of `model` failed (empty: it passed)."""
    problems = []
    if len(records) < wanted:
        problems.append(f"only {len(records)} of {wanted} trials left a record")
    failed = [r for r in records if r.get("outcome") == "infra_error"]
    if failed:
        reason = str(failed[0].get("reason") or failed[0].get("error") or "")[:120]
        problems.append(f"{len(failed)} infra_error trials, first: {reason}")
    free = [r for r in records if int(r.get("turns") or 0) > 0 and not float(r.get("usd") or 0) > 0]
    if free:
        problems.append(f"{len(free)} trials answered but cost usd = 0")
    if "/" not in model:
        return problems  # Anthropic: no provider names, no reasoning token count
    providers = sorted({p for r in records for p in r.get("providers") or []})
    if len(providers) != 1:
        problems.append(f"served by {providers or 'no named provider'}, expected exactly one")
    if any(r.get("pin_match") is False for r in records):
        problems.append("a trial was served by a provider other than the pin (pin_match false)")
    reasoning = sum(int((r.get("tokens") or {}).get("reasoning_tokens") or 0) for r in records)
    if reasoning <= 0:
        problems.append("no reasoning tokens in any reply")
    return problems


def run_cli(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


def cmd_smoke(args: argparse.Namespace, runner: Callable[[list[str]], int] = run_cli) -> int:
    check_snapshot(args.snapshot)
    model = model_id(args.model)
    out = fresh_directory(Path(args.out) / "smoke" / model.replace("/", "__"))
    argv = [
        sys.executable, "-m", f"{PACKAGE}.bench.runner.cli", "run", "--out", str(out),
        "--models", model, "--families", *SMOKE_FAMILIES, "--seeds", "0",
        "--tier", "implicit", "--route", "forced", "--contract", "bare", "--arm", "bare",
        "--limit", str(args.n), "--workers", str(args.workers), "--max-usd", str(args.max_usd),
        "--env-locks", str(args.env_locks),
    ]  # fmt: skip
    if args.snapshot:
        argv += ["--snapshot", args.snapshot]
    say(f"smoke: {args.n} pinned trials of {model} into {out}")
    status = runner(argv)
    records = read_records(out)
    problems = smoke_problems(model, records, args.n)
    served = sorted({p for r in records for p in r.get("providers") or []})
    spent = sum(float(r.get("usd") or 0) for r in records)
    say(f"smoke: runner status {status}, {len(records)} records, providers {served}, ${spent:.4f}")
    for problem in problems:
        say(f"  FAIL {problem}")
    say(f"smoke: {model}: {'FAILED' if problems else 'passed'}")
    return 1 if problems else 0


def status_kind(text: str) -> str:
    """What an api_error was: 429, 5xx, timeout, connection, or other."""
    low = text.lower()
    if re.search(r"(?:code|status)\D{0,3}429|ratelimit|rate limit|too many requests", low):
        return "429"
    five = r"(?:code|status)\D{0,3}5\d\d|internalserver|service unavailable|bad gateway|overloaded"
    if re.search(five, low):
        return "5xx"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    return "connection" if "connection" in low else "other"


def probe_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The failed requests of a probe run with max_retries=0 (one per failed trial), by kind."""
    failed = [
        r
        for r in records
        if r.get("outcome") == "infra_error"
        and r.get("stop") in ("api_error", "credit_exhausted", "circuit_open")
    ]
    kinds = collections.Counter(
        status_kind(f"{r.get('reason') or ''} {r.get('error') or ''}") for r in failed
    )
    requests = sum(int(r.get("turns") or 0) for r in records) + len(failed)
    throttled = kinds["429"] + kinds["5xx"]
    return {
        "trials": len(records),
        "requests": requests,
        "failed": len(failed),
        "kinds": dict(sorted(kinds.items())),
        "share": throttled / requests if requests else math.nan,
    }


def probe_runner(model: str, out: Path, max_usd: float) -> Callable[[Any], dict[str, Any]]:
    """run_trial for `model` with no retries and no breaker, so every failed request shows."""
    import anthropic
    import openai

    from science_harness.bench.runner.agent import AgentConfig, CircuitBreaker
    from science_harness.bench.runner.agent_openai import OPENROUTER_URL, provider_routing
    from science_harness.bench.runner.trial import Budget, Clients, run_trial
    from science_harness.bench.tasks.tools import ToolEnv
    from science_harness.keys import get_key

    clients = Clients()
    if "/" in model:
        key = get_key("openrouter")
        if not key:
            raise Refused("no OpenRouter key found (python -I -m science_harness.keys)")
        clients.openrouter = openai.OpenAI(
            api_key=key, base_url=OPENROUTER_URL, max_retries=0, timeout=600
        )
    else:
        key = get_key("anthropic")
        if not key:
            raise Refused("no Anthropic key found (python -I -m science_harness.keys)")
        clients.anthropic = anthropic.Anthropic(api_key=key, max_retries=0)
    tools = ToolEnv.discover(Path.cwd())
    if tools is None:
        raise Refused("samtools, bcftools and bedtools not found (see .local/tools.json)")
    cfg = AgentConfig(
        model, max_retries=0, breaker=CircuitBreaker(0), provider=provider_routing(model)
    )
    budget = Budget(max_usd)
    return lambda spec: run_trial(spec, out, clients, tools, cfg, budget, None)


def cmd_probe(args: argparse.Namespace, make_runner: Callable[..., Any] = probe_runner) -> int:
    check_snapshot(args.snapshot)
    from science_harness.bench.runner.cli import plan

    model = model_id(args.model)
    out = fresh_directory(Path(args.out) / "probe" / model.replace("/", "__"))
    out.mkdir(parents=True, exist_ok=True)
    run = make_runner(model, out, args.max_usd)
    specs = list(plan([model], PROBE_FAMILIES, [1], 1, "implicit", "forced", "bare", "minimal"))
    specs = specs[: args.n]
    say(f"probe: {len(specs)} trials of {model}, {args.workers} workers, max_retries=0")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, specs))
    summary = probe_summary(read_records(out))
    share = summary["share"]
    say(
        f"probe: {summary['trials']} trials, {summary['requests']} requests, "
        f"{summary['failed']} failed {summary['kinds']}, "
        f"429/5xx share {share:.1%} (limit {args.max_share:.0%})"
    )
    ok = summary["trials"] > 0 and share <= args.max_share
    say(f"probe: {model}: {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


# -- the schedule of a job --------------------------------------------------------------------


class Block(NamedTuple):
    """One runner invocation: `families` None means all of them."""

    contract: str
    tier: str
    route: str
    arm: str
    reps: int
    seeds: str
    families: list[str] | None

    def line(self) -> str:
        fams = " ".join(self.families) if self.families is not None else "-"
        fields = (self.contract, self.tier, self.route, self.arm, str(self.reps), self.seeds, fams)
        return "\t".join(fields)


def cell_of(cell: str) -> tuple[str, str, str]:
    from science_harness.bench import stage2

    if cell in ("primary", "pilot"):
        return stage2.PRIMARY_CELL
    if cell == "default":
        return stage2.DEFAULT_CELL
    raise Refused(f"unknown cell {cell!r} (pilot, primary or default)")


def load_sets(path: str | None, cell: str) -> dict[str, Any] | None:
    if cell == "pilot":
        return None
    if not path:
        raise Refused(f"--sets is needed for the {cell} cell")
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"--sets {path}: {exc}") from exc


def blocks(cell: str, arm: str, chunk: int, size: int, sets: dict[str, Any] | None) -> list[Block]:
    """The invocations of `arm` for seed chunk number `chunk` (`size` seeds per chunk).

    The runner takes one seed list per call and the registry has two seed blocks, so the families
    of an arm are grouped by block. Arm `aa` is the bare replicate on P (repetition 1, primary
    cell): `--reps 2` resumes repetition 0 and runs repetition 1.
    """
    from science_harness.bench import stage2

    contract, tier, route = cell_of(cell)
    if cell == "pilot":  # one call: bare, seeds 0-4, every family
        one = [Block(contract, tier, route, "bare", 1, "0-4", None)]
        return one if arm == "bare" and chunk == 0 else []
    reg = stage2.Registry.from_sets(sets or {})
    cell_tuple = (contract, tier, route)
    if arm == "aa":
        families, runner_arm, reps = (reg.primary if cell == "primary" else ()), "bare", 2
    else:
        families, runner_arm, reps = reg.families_for(arm, cell_tuple), arm, 1
    seeds = reg.seeds_for(cell_tuple)
    by_seeds: dict[tuple[int, ...], list[str]] = collections.defaultdict(list)
    for family in families:
        by_seeds[tuple(seeds[family])].append(family)
    out = []
    for seed_block, members in sorted(by_seeds.items()):
        part = seed_block[chunk * size : (chunk + 1) * size]
        if part:
            seed_text = ",".join(map(str, part))
            out.append(Block(contract, tier, route, runner_arm, reps, seed_text, members))
    return out


def chunk_count(cell: str, size: int, sets: dict[str, Any] | None) -> int:
    if cell == "pilot":
        return 1
    from science_harness.bench import stage2

    reg = stage2.Registry.from_sets(sets or {})
    longest = max((len(s) for s in reg.seeds_for(cell_of(cell)).values()), default=0)
    return math.ceil(longest / size)


def cmd_blocks(args: argparse.Namespace) -> int:
    check_snapshot(args.snapshot)
    for block in blocks(
        args.cell, args.arm, args.chunk, args.size, load_sets(args.sets, args.cell)
    ):
        say(block.line())
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    check_snapshot(args.snapshot)
    say(str(chunk_count(args.cell, args.size, load_sets(args.sets, args.cell))))
    return 0


def plan_manifest(sets: dict[str, Any], sets_bytes: bytes) -> dict[str, Any]:
    """Every planned (model, arm, cell): how many trial ids, and the sha256 of the sorted ids."""
    from science_harness.bench import stage2

    reg = stage2.Registry.from_sets(sets)
    entries = []
    for (model, arm, cell), ids in stage2.campaign_ids(reg).items():
        digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        entries.append(
            {"model": model, "arm": arm, "cell": "/".join(cell), "n": len(ids), "sha256": digest}
        )
    return {
        "format": 1,
        "sets_sha256": hashlib.sha256(sets_bytes).hexdigest(),
        "entries": entries,
        "total": sum(e["n"] for e in entries),
    }


def cmd_plan_manifest(args: argparse.Namespace) -> int:
    check_snapshot(args.snapshot)
    try:
        raw = Path(args.sets).read_bytes()
        sets = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise Refused(f"--sets {args.sets}: {exc}") from exc
    manifest = plan_manifest(sets, raw)
    Path(args.out).write_text(json.dumps(manifest, indent=1) + "\n")
    n, total = len(manifest["entries"]), manifest["total"]
    say(f"plan manifest: {n} entries, {total} trials -> {args.out}")
    return 0


def expected_ids(
    model: str, cell: str, sets: dict[str, Any] | None, aa: bool, exclude: Sequence[str]
) -> dict[str, list[str]]:
    """arm -> the trial ids a finished job of (model, cell) must have records for."""
    from science_harness.bench import stage2
    from science_harness.bench.runner.cli import plan
    from science_harness.bench.tasks.families import all_families

    contract, tier, route = cell_of(cell)
    if cell == "pilot":
        families = [f for f in all_families() if f not in set(exclude)]
        specs = plan([model], families, range(5), 1, tier, route, contract, "minimal", "bare")
        return {"bare": [s.trial_id for s in specs]}
    reg = stage2.Registry.from_sets(sets or {})
    cell_tuple = (contract, tier, route)
    arms = (stage2.BASE_ARM, stage2.PRIMARY_ARM) + ((stage2.WARN_ARM,) if cell == "primary" else ())
    wanted = {arm: stage2.planned(reg, model, arm, cell_tuple) for arm in arms}
    if aa and cell == "primary":  # repetition 1 of bare on P, in every seed chunk
        replicate = []
        for chunk in range(chunk_count(cell, 1, sets)):
            for b in blocks(cell, "aa", chunk, 1, sets):
                seeds = [int(x) for x in b.seeds.split(",")]
                specs = plan([model], b.families or [], seeds, b.reps, tier, route, contract)
                replicate += [s.trial_id for s in specs if s.rep == 1]
        wanted["aa"] = sorted(replicate)
    return wanted


def verify_job(out: Path, wanted: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """(table lines, problems): `wanted` ids against plan.json and the score.json files in `out`."""
    try:
        listed = set(json.loads((out / "plan.json").read_text()).get("ids") or [])
    except (OSError, ValueError, AttributeError):
        return [], [f"{out}/plan.json is missing or unreadable"]
    rows, problems = [], []
    for arm, ids in wanted.items():
        found = collections.Counter()
        for trial in ids:
            if trial not in listed:
                found["not in plan.json"] += 1
                continue
            try:
                record = json.loads((out / "trials" / trial / "score.json").read_text())
                outcome, attempt = record["outcome"], int(record.get("attempt") or 1)
            except FileNotFoundError:
                found["absent"] += 1
            except (OSError, ValueError, KeyError, TypeError):
                found["unreadable"] += 1
            else:
                open_ = outcome == "infra_error" and attempt < MAX_ATTEMPTS
                found["infra_error, attempts left" if open_ else "final"] += 1
        bad = {k: v for k, v in found.items() if k != "final"}
        rows.append(f"{arm:<14}{len(ids):>6} planned {found['final']:>6} final  {bad or ''}")
        problems += [f"{arm}: {n} {what}" for what, n in bad.items()]
    if not any(wanted.values()):  # an empty plan is not a finished one
        problems.append("the frozen plan has no trial for this job")
    return rows, problems


def cmd_verify(args: argparse.Namespace) -> int:
    check_snapshot(args.snapshot)
    model = model_id(args.model)
    sets = load_sets(args.sets, args.cell)
    wanted = expected_ids(model, args.cell, sets, args.aa, args.exclude)
    rows, problems = verify_job(Path(args.out), wanted)
    for row in rows:
        say(row)
    for problem in problems:
        say(f"  FAIL {problem}")
    verdict = "INCOMPLETE" if problems else "every planned trial has a final record"
    say(f"verify: {model} {args.cell}: {verdict}")
    return 1 if problems else 0


# -- command line -----------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preflight.py", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)

    def command(name: str, help_: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help_, allow_abbrev=False)
        c.add_argument("--snapshot", help="the code snapshot that must be imported")
        return c

    c = command("endpoints", "the pin tags exist and support tools and reasoning")
    c.add_argument("--models", nargs="+", help="aliases or ids (default: every pinned model)")
    c = command("credits", "the OpenRouter balance covers the job")
    c.add_argument("--model", required=True)
    c.add_argument("--need", type=float, required=True, help="USD the job may spend")
    c = command("smoke", "20 pinned trials: one provider, usd > 0, reasoning tokens > 0")
    c.add_argument("--model", required=True)
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--max-usd", type=float, default=1.0)
    c.add_argument("--out", default="runs/preflight")
    c.add_argument("--env-locks", default=str(ROOT / "envs" / "locks"))
    c = command("probe", "40 trials with max_retries=0: the share of 429 and 5xx")
    c.add_argument("--model", required=True)
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--workers", type=int, default=8, help="the planned worker count")
    c.add_argument("--max-usd", type=float, default=2.0)
    c.add_argument("--max-share", type=float, default=MAX_ERROR_SHARE)
    c.add_argument("--out", default="runs/preflight")
    c = command("blocks", "runner invocations of one arm and seed chunk (tab separated)")
    c.add_argument("--cell", required=True, choices=["pilot", "primary", "default"])
    c.add_argument(
        "--arm",
        required=True,
        choices=["bare", "block-nonote", "warn-nonote", "aa"],
        help="aa: the bare replicate on P (repetition 1)",
    )
    c.add_argument("--chunk", type=int, required=True)
    c.add_argument("--size", type=int, default=1, help="seeds per chunk")
    c.add_argument("--sets", help="the frozen sets file (not for the pilot)")
    c = command("chunks", "how many seed chunks a cell has")
    c.add_argument("--cell", required=True, choices=["pilot", "primary", "default"])
    c.add_argument("--size", type=int, default=1)
    c.add_argument("--sets")
    c = command("verify", "every id the frozen plan has for this job has a final record")
    c.add_argument("--out", required=True, help="the job's output directory")
    c.add_argument("--model", required=True)
    c.add_argument("--cell", required=True, choices=["pilot", "primary", "default"])
    c.add_argument("--sets")
    c.add_argument("--aa", action="store_true", help="the bare replicate on P was run too")
    c.add_argument("--exclude", nargs="*", default=[], help="pilot: families left out")
    c = command("plan-manifest", "the planned trial ids of the whole campaign, by digest")
    c.add_argument("--sets", required=True)
    c.add_argument("--out", required=True)
    return p


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "endpoints": cmd_endpoints,
    "credits": cmd_credits,
    "smoke": cmd_smoke,
    "probe": cmd_probe,
    "blocks": cmd_blocks,
    "chunks": cmd_chunks,
    "plan-manifest": cmd_plan_manifest,
    "verify": cmd_verify,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except Refused as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
