"""The same tool-use loop over an OpenAI-compatible chat API (OpenRouter and other gateways).

Open-weight models are reached through serving layers whose tool-call parsers can drop calls or
mangle arguments, so the loop records those events instead of hiding them: a call whose arguments
are not valid JSON gets an error result (the agent may retry) and is counted in
``tool_arg_errors``; the provider that served each turn is kept in ``served_by``.

A gateway also fails in ways the SDK passes through: an HTML page or an empty body with status 200,
a reply without choices, ``finish_reason`` "error". The API call and every access to its reply sit
in one try block, so each of these ends the trial as ``api_error`` (402: ``credit_exhausted``) with
the turns and spend recorded so far, instead of an exception that takes the whole job down.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import openai

from science_harness.bench.runner.agent import (
    RESPONSE_ERRORS,
    AgentConfig,
    ApiFailure,
    Trajectory,
    breaker_for,
    end_on_api_failure,
    require_priced,
    system_prompt,
    with_retries,
)
from science_harness.bench.runner.sandbox import run_command
from science_harness.bench.tasks.tools import ToolEnv

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# One serving stack per model for the pre-registered runs: OpenRouter otherwise spreads a model
# over a dozen providers with different quantisations. With `allow_fallbacks` off, an outage of
# the pinned provider becomes an infra error (rerun later), never a silent switch.
PINNED_PROVIDERS: dict[str, str] = {
    "deepseek/deepseek-v4.1-flash": "wafer/us",
    "openai/gpt-oss-120b": "near-ai/fp4",
}


def provider_routing(model: str, pin: bool = True) -> dict[str, Any] | None:
    """OpenRouter `provider` preferences that pin `model` to its provider, or None."""
    tag = PINNED_PROVIDERS.get(model) if pin else None
    return {"order": [tag], "allow_fallbacks": False} if tag else None


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def pin_matches(pin: dict[str, Any] | None, providers: list[str]) -> bool | None:
    """Whether every provider that served a turn is a pinned one; None without a pin or a name.

    The pin is an endpoint tag ("wafer/us"), the reply names the provider ("Wafer"), so only the
    provider part of the tag is compared, ignoring case and punctuation ("near-ai" is "Near AI").
    """
    order = (pin or {}).get("order") or []
    if not order or not providers:
        return None
    wanted = {_squash(tag.split("/")[0]) for tag in order}
    return all(_squash(name) in wanted for name in providers)


# OpenRouter maps every provider's reason onto these four, plus "error" for a failure inside a
# reply with status 200; anything else is not an answer of the model.
FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "content_filter"})
API_ERRORS: tuple[type[BaseException], ...] = (openai.OpenAIError, *RESPONSE_ERRORS)


BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the working directory; returns exit code, output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def _error_result(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": f"Error: {text}"}


def _run_call(
    call: Any, work: Path, tools: ToolEnv, cfg: AgentConfig
) -> tuple[dict[str, Any], str]:
    """(tool message, kind) where kind is 'ran', 'bad_args' or 'unknown_tool'."""
    if call.function.name == "run_tool" and cfg.gateway is not None:
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise TypeError("arguments must be an object")
        except (ValueError, TypeError) as exc:
            return _error_result(call.id, f"invalid tool arguments ({exc})"), "bad_args"
        text, _ = cfg.gateway.run(args)
        return {"role": "tool", "tool_call_id": call.id, "content": text}, "ran"
    if call.function.name != "bash":
        return _error_result(
            call.id, f"unknown tool {call.function.name!r}; use bash"
        ), "unknown_tool"
    try:
        args = json.loads(call.function.arguments or "{}")
        command = args["command"]
        if not isinstance(command, str):
            raise TypeError("command must be a string")
    except (ValueError, KeyError, TypeError) as exc:
        return _error_result(
            call.id, f'invalid tool arguments ({exc}); send {{"command": ...}}'
        ), "bad_args"
    result = run_command(work, tools, command, cfg.command_timeout, cfg.max_output_chars, cfg.layer)
    return {"role": "tool", "tool_call_id": call.id, "content": result.render()}, "ran"


def _cost(usage: Any) -> float:
    extra = getattr(usage, "model_extra", None) or {}
    cost = extra.get("cost", getattr(usage, "cost", 0.0))
    return float(cost or 0.0)


def _usage(usage: Any) -> tuple[int, int, int | None, float]:
    """(prompt tokens, completion tokens, reasoning tokens if reported, USD) of one reply."""
    details = getattr(usage, "completion_tokens_details", None)
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
    else:
        reasoning = getattr(details, "reasoning_tokens", None)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        None if reasoning is None else int(reasoning),
        _cost(usage),
    )


def _describe(resp: Any) -> str:
    """What a reply without choices says, for the trial's error text."""
    error = (getattr(resp, "model_extra", None) or {}).get("error")
    return (f"gateway error {error}" if error else str(resp))[:200]


def _read_turn(
    resp: Any, traj: Trajectory, cfg: AgentConfig
) -> tuple[Any, dict[str, Any], list[Any]]:
    """Add one reply to `traj`; returns (choice, the message to send back, its tool calls).

    A reply that cannot be read raises before `traj` changes. One that was read but cannot be used
    (a finish_reason outside FINISH_REASONS, no price) is recorded first and then raises ApiFailure,
    so its turn and spend are not lost.
    """
    choices = getattr(resp, "choices", None)
    if not choices:  # a gateway error body that parsed as a response, or a body that is no JSON
        raise ApiFailure(f"empty response: {_describe(resp)}")
    choice = choices[0]
    message = choice.message.model_dump(exclude_none=True)
    calls = list(choice.message.tool_calls or [])
    for call in calls:  # what _run_call reads: a call without its function fails here, not later
        _ = call.id, call.function.name, call.function.arguments
    provider = (getattr(resp, "model_extra", None) or {}).get("provider")
    served_by = f"{resp.model}@{provider}" if provider else resp.model
    prompt_tokens, completion_tokens, reasoning, usd = _usage(resp.usage)

    traj.turns += 1
    traj.served_by = served_by
    if provider and provider not in traj.providers:
        traj.providers.append(provider)
    traj.pin_match = pin_matches(cfg.provider, traj.providers)
    traj.finish_reasons[str(choice.finish_reason)] += 1
    if math.isfinite(usd) and usd >= 0:  # a NaN or negative cost must not blind the spend guard
        traj.usd += usd
    traj.tokens["input_tokens"] += prompt_tokens
    traj.tokens["output_tokens"] += completion_tokens
    if reasoning is not None:
        traj.tokens["reasoning_tokens"] += reasoning
    traj.messages.append({"role": "assistant", "finish_reason": choice.finish_reason, **message})

    if choice.finish_reason not in FINISH_REASONS:
        error = (getattr(choice, "model_extra", None) or {}).get("error")
        raise ApiFailure(f"finish_reason {choice.finish_reason!r}: {str(error or message)[:200]}")
    if choice.finish_reason != "content_filter":  # a blocked reply may well be free
        if resp.usage is None:
            raise ApiFailure("unpriced: the reply carries no usage")
        require_priced(prompt_tokens + completion_tokens, usd)
    return choice, message, calls


def run_agent_openai(
    client: openai.OpenAI, cfg: AgentConfig, prompt: str, work: Path, tools: ToolEnv
) -> Trajectory:
    traj = Trajectory(stop="max_turns")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(cfg.contract, cfg.layer_note, cfg.careful_note),
        },
        {"role": "user", "content": prompt},
    ]
    traj.messages.extend(messages)
    extra_body: dict[str, Any] = {"usage": {"include": True}, "reasoning": {"effort": cfg.effort}}
    if cfg.provider:
        extra_body["provider"] = cfg.provider
    # built before the try block: a bug in it must not be filed as an API failure
    schemas = [BASH_TOOL, *([cfg.gateway.openai_schema()] if cfg.gateway else [])]
    breaker = breaker_for(client, cfg)
    client = with_retries(client, cfg.max_retries)
    for _ in range(cfg.max_turns):
        if breaker.stop:  # the job is stopping: no further call
            traj.stop, traj.error = breaker.stop, breaker.reason
            return traj
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=schemas,
                max_tokens=cfg.max_tokens,
                extra_body=extra_body,
            )
            choice, message, calls = _read_turn(resp, traj, cfg)
        except API_ERRORS as exc:
            return end_on_api_failure(traj, breaker, exc)
        breaker.success()
        if not calls:
            reason = choice.finish_reason
            if reason == "content_filter":
                traj.stop, traj.refusal_category = "refusal", "content_filter"
            elif reason == "length":
                traj.stop = "max_tokens"
            else:
                traj.stop = "end_turn"
            return traj
        if traj.usd > cfg.max_usd:
            traj.stop = "budget"
            return traj
        messages.append(message)
        for call in calls:
            tool_message, kind = _run_call(call, work, tools, cfg)
            traj.commands += kind == "ran"
            traj.tool_arg_errors += kind == "bad_args"
            messages.append(tool_message)
            traj.messages.append(tool_message)
    return traj
