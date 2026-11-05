"""A minimal Claude tool-use loop for the Phase-0 pilot: one bash tool, hard turn and spend caps.

Raw-CLI arm (A0): no contracts, no hints about conventions. The loop passes assistant content
back unchanged (thinking blocks included), never sets sampling parameters, never forces a tool,
never uses server-side fallbacks, and treats a refusal as an outcome, not an error.

Failures of the API itself are classified where they happen, so the partial trajectory and its
spend survive: ``api_error`` (rerun later), ``credit_exhausted`` (402) and ``circuit_open`` (many
api_errors in a row) are infrastructure outcomes that say nothing about the model.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import httpx2

from science_harness.bench.runner.pricing import usage_usd
from science_harness.bench.runner.sandbox import WORKDIR, SandboxLayer, run_command
from science_harness.bench.tasks.tools import ToolEnv

INTRO = (
    f"You are a bioinformatics assistant working in a Linux sandbox. Your working directory is "
    f"{WORKDIR}; it holds the project files for the task (README.md describes them). Use the "
    "bash tool to run commands. Installed: samtools 1.24, bcftools 1.24, bedtools 2.31.1, "
    "python3 (3.6) and python3.12, awk, sed and the usual coreutils. There is no network access."
)
CONTRACTS = {
    # full: a status and a free-text warning channel; bare: the answer and nothing else, so the
    # prompt cannot nudge the agent to look for problems
    "full": [
        "When you are done, write result.json in the working directory:",
        '  {"status": "ok" or "needs_input", "answer": <as the task specifies>, '
        '"flags": [<strings>], "method": "<one line>"}',
        "- flags: list any problems with the data, or assumptions, that could make your answer "
        "unreliable. Use an empty list if you found none.",
        '- status "needs_input": use it instead of "ok" if you found a problem that prevents a '
        "reliable answer, and explain it in flags (answer may then be null).",
        "Only result.json is read; your final chat message is ignored.",
    ],
    "bare": [
        "When you are done, write result.json in the working directory:",
        '  {"answer": <as the task specifies>}',
        "Only result.json is read; your final chat message is ignored.",
    ],
}


LAYER_NOTE = (
    "Commands you run may be checked by a safety layer. If a command is refused or prints a "
    "WARNING, read the message and follow its FIX. Some commands take `--declare key=value` to "
    "state which convention you intend; the message tells you when."
)


CAREFUL_NOTE = (
    "Before you finish, check the input and output formats and whether the results make "
    "biological sense."
)


def system_prompt(contract: str = "full", layer: bool = False, careful: bool = False) -> str:
    parts = [INTRO, "", *CONTRACTS[contract]]
    if layer:
        parts += ["", LAYER_NOTE]
    if careful:
        parts += ["", CAREFUL_NOTE]
    return "\n".join(parts)


SYSTEM = system_prompt("full")

BASH_TOOL: dict[str, Any] = {
    "name": "bash",
    "description": "Run a shell command in the working directory; returns exit code and output.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command to run."}},
        "required": ["command"],
    },
}


MAX_RETRIES = 8  # SDK retries per request (429, 5xx, timeouts, dropped connections): 30-40 s
# consecutive api_errors, over all trials, after which the job stops (TBM_BREAKER_LIMIT; 0: never)


def _limit_from_env(default: int = 10) -> int:
    """TBM_BREAKER_LIMIT as a non-negative integer; empty or malformed means the default."""
    try:
        return max(0, int(os.environ.get("TBM_BREAKER_LIMIT", "").strip() or default))
    except ValueError:
        return default


BREAKER_LIMIT = _limit_from_env()

# Stop codes of an API failure. All are infrastructure outcomes (an infra_error record, rerun
# later); after the last two the runner should stop launching trials, the next call fails too.
INFRA_STOPS = frozenset({"api_error", "credit_exhausted", "circuit_open"})
ABORT_STOPS = frozenset({"credit_exhausted", "circuit_open"})


class ApiFailure(Exception):
    """A reply the loop cannot use; the message is the short reason recorded for the trial."""


class CircuitBreaker:
    """Stops a job whose API is down. It is shared by the trials that use one client.

    ``failure`` counts consecutive api_error outcomes over all trials and worker threads and trips
    at ``limit`` (0: never); any good reply resets the count. This is apart from the per-trial
    reruns of an infra_error: an outage has to stop the job, not use up the reruns of every trial.
    Once tripped (here, or by ``trip`` for an empty account) it stays tripped: every later call
    ends with ``stop`` before it touches the API.
    """

    def __init__(self, limit: int = BREAKER_LIMIT) -> None:
        self.limit = limit
        self.stop: str | None = None
        self.reason = ""
        self._streak = 0
        self._lock = threading.Lock()

    def success(self) -> None:
        with self._lock:
            self._streak = 0

    def failure(self, reason: str) -> bool:
        """Count one api_error; True if this one tripped the breaker."""
        with self._lock:
            self._streak += 1
            if not self.limit or self._streak < self.limit or self.stop:
                return False
            self.stop = "circuit_open"
            self.reason = f"{self._streak} api_errors in a row, last: {reason}"
            return True

    def trip(self, stop: str, reason: str) -> None:
        with self._lock:
            if not self.stop:
                self.stop, self.reason = stop, reason

    def __deepcopy__(self, memo: dict[int, Any]) -> CircuitBreaker:
        return self  # a shared monitor: asdict(cfg) must not copy its lock

    def __repr__(self) -> str:
        return f"CircuitBreaker(limit={self.limit})"  # no address: a config digest stays stable


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: str = "medium"
    max_turns: int = 30
    max_tokens: int = 16000
    command_timeout: int = 60
    max_output_chars: int = 6000
    max_usd: float = 0.75
    contract: str = "full"
    layer: SandboxLayer | None = None
    gateway: Any = None  # a GatewayTool: adds the run_tool function next to bash
    layer_note: bool = False  # the layer's system-prompt sentence without any layer (control)
    careful_note: bool = False  # a generic carefulness sentence, no layer (baseline)
    promotion_path: str | None = None  # promotion table for the block arm
    provider: dict[str, Any] | None = None  # OpenRouter routing preferences (the pinned provider)
    max_retries: int | None = MAX_RETRIES  # per request, overrides the client's; None keeps it
    breaker: CircuitBreaker | None = None  # None: the one that lives on the client


@dataclass
class Trajectory:
    # end_turn | refusal | max_turns | budget | max_tokens | unexpected_stop, or an INFRA_STOPS code
    stop: str
    turns: int = 0
    commands: int = 0
    tool_arg_errors: int = 0  # tool calls whose arguments were not valid JSON
    usd: float = 0.0
    # reasoning_tokens (OpenRouter, when reported) is a part of output_tokens, not an addition
    tokens: Counter = field(default_factory=Counter)
    served_by: str | None = None
    providers: list[str] = field(default_factory=list)  # every provider that served a turn
    pin_match: bool | None = None  # every provider so far is the pinned one; None: no pin or name
    # the API's finish/stop reason of each turn, counted
    finish_reasons: Counter = field(default_factory=Counter)
    refusal_category: str | None = None
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


# What an API call or the reading of its reply can raise without the model being at fault.
RESPONSE_ERRORS: tuple[type[BaseException], ...] = (
    httpx2.HTTPError,  # a body cut off or undecodable while the SDK read it
    ValueError,  # includes JSONDecodeError: an empty or truncated JSON body
    AttributeError,  # a 200 body that is not a completion at all (HTML, null)
    TypeError,
    KeyError,
    ArithmeticError,  # includes OverflowError: an absurd token count or cost in the usage block
    RecursionError,  # a deeply nested body
    ApiFailure,
)
API_ERRORS: tuple[type[BaseException], ...] = (anthropic.APIError, *RESPONSE_ERRORS)
_NO_CREDIT = re.compile(r"credit balance|insufficient credit|out of credits", re.IGNORECASE)


def classify_api_error(exc: BaseException) -> tuple[str, str]:
    """(stop code, short reason) for an exception raised by an API call or by reading its reply.

    Insufficient credit is HTTP 402 (OpenRouter; Anthropic's billing_error); Anthropic's "credit
    balance is too low" arrives as a 400.
    """
    text = str(exc)
    reason = text if isinstance(exc, ApiFailure) else f"{type(exc).__name__}: {text[:300]}"
    status = getattr(exc, "status_code", None)
    if status == 402 or (status == 400 and _NO_CREDIT.search(text)):
        return "credit_exhausted", reason
    return "api_error", reason


def breaker_for(client: Any, cfg: AgentConfig) -> CircuitBreaker:
    """The breaker of a trial: the config's, else the one kept on the client.

    A client is one account and endpoint and every trial of a job uses the same one, so the trials
    count each other's failures; a client that a test builds for itself starts with a clean breaker
    (a process-wide one would stay tripped for every later test).
    """
    if cfg.breaker is not None:
        return cfg.breaker
    try:
        return vars(client).setdefault("_tbm_breaker", CircuitBreaker())
    except TypeError:  # a client without __dict__ keeps no state: no breaker
        return CircuitBreaker(0)


def end_on_api_failure(traj: Trajectory, breaker: CircuitBreaker, exc: BaseException) -> Trajectory:
    """Finish `traj` after a failed call, keeping what it recorded so far; tells the breaker."""
    traj.stop, traj.error = classify_api_error(exc)
    if traj.stop == "credit_exhausted":
        breaker.trip(traj.stop, traj.error)
    elif breaker.failure(traj.error):
        traj.stop, traj.error = "circuit_open", breaker.reason
    return traj


def with_retries(client: Any, retries: int | None) -> Any:
    """`client` with its SDK retry budget set to `retries`; None, or a scripted client without
    ``with_options``, is returned as it is. The copy shares the HTTP client."""
    options = getattr(client, "with_options", None)
    return client if retries is None or options is None else options(max_retries=retries)


def require_priced(tokens: int, usd: float) -> None:
    """The spend guard goes blind on a reply that used tokens but cost nothing."""
    if not math.isfinite(usd) or usd < 0 or (tokens > 0 and usd == 0):
        raise ApiFailure(f"unpriced: {tokens} tokens, cost {usd}")


def _execute(
    block: Any, work: Path, tools: ToolEnv, cfg: AgentConfig
) -> tuple[dict[str, Any], bool]:
    """Run one tool_use block; returns (tool_result, ran_a_command)."""
    if block.name == "run_tool" and cfg.gateway is not None and isinstance(block.input, dict):
        text, failed = cfg.gateway.run(block.input)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": text,
            "is_error": failed,
        }, True
    command = block.input.get("command") if isinstance(block.input, dict) else None
    if block.name != "bash" or not isinstance(command, str):
        text = (
            f"Error: unknown tool or invalid input for {block.name!r}; "
            "use bash with a 'command' string."
        )
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": text,
            "is_error": True,
        }, False
    result = run_command(work, tools, command, cfg.command_timeout, cfg.max_output_chars, cfg.layer)
    failed = result.exit_code != 0
    return (
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.render(),
            "is_error": failed,
        },
        True,
    )


def _record_turn(resp: Any, traj: Trajectory, cfg: AgentConfig) -> None:
    """Add one reply to `traj`. It is all read before `traj` changes, so a reply that is not a
    message (HTML, null fields) leaves `traj` as it was; ApiFailure if it cannot be priced."""
    if not isinstance(getattr(resp, "content", None), list):
        raise ApiFailure(f"unexpected reply: {str(resp)[:200]}")
    try:
        usd = usage_usd(cfg.model, resp.usage)
    except KeyError as exc:  # a model missing from the price table
        raise ApiFailure(f"unpriced: no price for model {cfg.model!r}") from exc
    counts = {
        name: getattr(resp.usage, name, 0) or 0
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }
    require_priced(sum(counts.values()), usd)
    message = {
        "role": "assistant",
        "stop_reason": resp.stop_reason,
        "content": [b.model_dump(mode="json") for b in resp.content],
    }
    traj.turns += 1
    traj.served_by = resp.model
    traj.usd += usd
    traj.tokens.update(counts)
    traj.finish_reasons[str(resp.stop_reason)] += 1
    traj.messages.append(message)


def run_agent(
    client: anthropic.Anthropic, cfg: AgentConfig, prompt: str, work: Path, tools: ToolEnv
) -> Trajectory:
    traj = Trajectory(stop="max_turns")
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    traj.messages.append({"role": "user", "content": prompt})
    # built before the try block: a bug in them must not be filed as an API failure
    system = system_prompt(cfg.contract, cfg.layer_note, cfg.careful_note)
    schemas = [BASH_TOOL, *([cfg.gateway.anthropic_schema()] if cfg.gateway else [])]
    breaker = breaker_for(client, cfg)
    client = with_retries(client, cfg.max_retries)
    for _ in range(cfg.max_turns):
        if breaker.stop:  # the job is stopping: no further call
            traj.stop, traj.error = breaker.stop, breaker.reason
            return traj
        try:
            resp = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=system,
                tools=schemas,
                messages=messages,
                output_config={"effort": cfg.effort},
                cache_control={"type": "ephemeral"},
            )
            _record_turn(resp, traj, cfg)
        except API_ERRORS as exc:
            return end_on_api_failure(traj, breaker, exc)
        breaker.success()
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            traj.stop, traj.refusal_category = "refusal", getattr(details, "category", None)
            return traj
        if resp.stop_reason == "end_turn":
            traj.stop = "end_turn"
            return traj
        if resp.stop_reason != "tool_use":
            traj.stop = "max_tokens" if resp.stop_reason == "max_tokens" else "unexpected_stop"
            return traj
        if traj.usd > cfg.max_usd:
            traj.stop = "budget"
            return traj
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_result, ran = _execute(block, work, tools, cfg)
                traj.commands += ran
                results.append(tool_result)
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": results})
        traj.messages.append({"role": "user", "content": results})
    return traj
