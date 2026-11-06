"""The API clients of the two tool loops, driven through the real SDKs over a mock HTTP transport.

Nothing here calls an API or needs the sandbox: the transport answers, `run_command` is replaced and
the SDKs' backoff sleeps are recorded instead of waited for.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

from biowarden.bench.runner import agent, agent_openai
from biowarden.bench.runner.agent import (
    ABORT_STOPS,
    INFRA_STOPS,
    MAX_RETRIES,
    AgentConfig,
    CircuitBreaker,
    breaker_for,
    run_agent,
    system_prompt,
    with_retries,
)
from biowarden.bench.runner.agent_openai import (
    pin_matches,
    provider_routing,
    run_agent_openai,
)
from biowarden.bench.runner.pricing import usage_usd

OPEN_MODEL = "deepseek/deepseek-v4.1-flash"
CLAUDE = "claude-haiku-5-5"
PIN = {"order": ["wafer/us"], "allow_fallbacks": False}


# --- the mock transport ----------------------------------------------------------------------


def fake_command(work, tools, command, timeout, max_chars, layer):
    return SimpleNamespace(exit_code=0, render=lambda: "[exit code 0]")


@pytest.fixture
def sleeps(monkeypatch):
    """The SDKs back off with time.sleep; record the delays instead of waiting for them."""
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


@pytest.fixture(autouse=True)
def isolated(sleeps, monkeypatch):
    monkeypatch.setattr(agent, "run_command", fake_command)
    monkeypatch.setattr(agent_openai, "run_command", fake_command)


def http(status=200, body=None, *, text=None, raw=None, headers=None):
    """A reply for the Server: JSON, an HTML page, or raw bytes under a JSON content type."""

    def make():
        if text is not None:
            return httpx2.Response(status, text=text, headers={"content-type": "text/html"})
        if raw is not None:
            return httpx2.Response(
                status, content=raw, headers={"content-type": "application/json"}
            )
        return httpx2.Response(status, json=body, headers=headers)

    return make


class CutOff(httpx2.SyncByteStream):
    def __iter__(self):
        yield b'{"id": "gen-1", "choices": [{"index": 0, "mes'
        raise httpx2.ReadError("peer closed connection without sending complete message body")


def cut_off():
    return httpx2.Response(200, stream=CutOff(), headers={"content-type": "application/json"})


class Server:
    """Replays the given replies in order, the last one repeating, and keeps the requests."""

    def __init__(self, *replies):
        self.replies = replies
        self.requests: list[httpx2.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        return self.replies[min(len(self.requests), len(self.replies)) - 1]()

    def sent(self, n=0):
        return json.loads(self.requests[n].content)


def openrouter(server, retries=0):
    return openai.OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        max_retries=retries,
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )


def anthropic_client(server, retries=0):
    return anthropic.Anthropic(
        api_key="test",
        base_url="https://anthropic.test",
        max_retries=retries,
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )


def completion(message=None, finish="stop", cost=0.001, provider="Wafer", reasoning=None):
    """An OpenRouter chat completion; cost=None leaves the price out of the usage."""
    usage = {"prompt_tokens": 900, "completion_tokens": 100}
    if cost is not None:
        usage["cost"] = cost
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": OPEN_MODEL,
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": message or {"role": "assistant", "content": "done"},
            }
        ],
        "usage": usage,
    }


def bash(command="echo hi", call_id="call_1"):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def run_open(server, work, retries=0, client=None, **cfg):
    cfg.setdefault("provider", PIN)
    client = client or openrouter(server, retries)
    return run_agent_openai(client, AgentConfig(OPEN_MODEL, **cfg), "task", work, None)


def claude_message(content, stop="end_turn"):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": CLAUDE,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }


def tool_use(command="echo hi"):
    return [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": command}}]


def claude_error(kind, message="x"):
    return {"type": "error", "error": {"type": kind, "message": message}}


def run_claude(server, work, retries=0, model=CLAUDE, client=None, **cfg):
    client = client or anthropic_client(server, retries)
    return run_agent(client, AgentConfig(model, **cfg), "task", work, None)


NO_ENDPOINT = {
    "error": {"message": "No endpoints found for deepseek/deepseek-v4.1-flash", "code": 404}
}


# --- the OpenAI-compatible loop: a normal reply -------------------------------------------------


def test_a_reply_is_recorded_with_cost_reasoning_tokens_finish_reasons_and_providers(tmp_path):
    server = Server(
        http(body=completion(bash(), "tool_calls", cost=0.002, reasoning=40)),
        http(body=completion(finish="stop", cost=0.001, reasoning=10)),
    )

    traj = run_open(server, tmp_path)

    assert traj.stop == "end_turn" and traj.turns == 2 and traj.commands == 1
    assert traj.usd == pytest.approx(0.003)
    # reasoning_tokens is a part of output_tokens, kept next to it
    assert traj.tokens == {"input_tokens": 1800, "output_tokens": 200, "reasoning_tokens": 50}
    assert traj.finish_reasons == {"tool_calls": 1, "stop": 1}
    assert traj.providers == ["Wafer"] and traj.served_by == f"{OPEN_MODEL}@Wafer"
    assert traj.pin_match is True
    assert server.sent()["provider"] == PIN  # every request carries the pin
    assert server.sent(1)["messages"][-1]["role"] == "tool"


def test_reasoning_tokens_are_left_out_when_the_provider_does_not_report_them(tmp_path):
    traj = run_open(Server(http(body=completion())), tmp_path)

    assert traj.tokens == {"input_tokens": 900, "output_tokens": 100}


def test_the_pin_check_compares_the_provider_part_of_the_tag_with_the_returned_name():
    assert pin_matches(PIN, ["Wafer"]) is True
    assert pin_matches(provider_routing("openai/gpt-oss-120b"), ["Near AI"]) is True
    assert pin_matches(PIN, ["Wafer", "NextBit"]) is False  # a switch inside a trial
    assert pin_matches(None, ["Wafer"]) is None  # no pin
    assert pin_matches(PIN, []) is None  # the reply named no provider


def test_another_provider_is_recorded_not_failed(tmp_path):
    server = Server(http(body=completion(provider="NextBit")))

    pinned = run_open(server, tmp_path)
    free = run_open(Server(http(body=completion(provider="NextBit"))), tmp_path, provider=None)

    assert pinned.stop == "end_turn" and pinned.pin_match is False
    assert free.stop == "end_turn" and free.pin_match is None


# --- the OpenAI-compatible loop: replies and errors the loop cannot use ---------------------------


def gateway_error_in_a_200():
    return {"error": {"message": "Provider returned error", "code": 502}}


def finish_error():
    body = completion(finish="error", cost=0.0005)
    body["choices"][0]["error"] = {"code": 502, "message": "upstream died"}
    return body


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (http(text="<html>502 Bad Gateway</html>"), "empty response: <html>502 Bad Gateway"),
        (http(raw=b""), "JSONDecodeError"),
        (http(raw=b'{"id": "gen-1", "choices": [{"index": 0, "mes'), "JSONDecodeError"),
        (http(raw=b"null"), "empty response: None"),
        (http(body=[1, 2, 3]), "empty response: [1, 2, 3]"),
        (http(body={"id": "gen-1", "choices": []}), "empty response"),
        (http(body={"id": "gen-1", "choices": None}), "empty response"),
        (http(body=gateway_error_in_a_200()), "gateway error {'message': 'Provider returned"),
        (http(body=finish_error()), "finish_reason 'error': {'code': 502, 'message': 'upstream"),
        (http(body=completion(finish="function_call")), "finish_reason 'function_call'"),
        (http(body=completion(finish=None)), "finish_reason None"),
    ],
    ids=[
        "html-200",
        "empty-json",
        "truncated-json",
        "null-body",
        "json-list",
        "no-choices",
        "choices-null",
        "gateway-error-in-200",
        "finish-reason-error",
        "finish-reason-unknown",
        "finish-reason-null",
    ],
)
def test_replies_the_loop_cannot_use_end_the_trial_as_api_error(tmp_path, reply, expected):
    server = Server(reply)

    traj = run_open(server, tmp_path)

    assert traj.stop == "api_error" and expected in traj.error
    assert len(server.requests) == 1 and len(traj.error) <= 400


def test_a_body_cut_off_in_transit_is_retried_and_then_an_api_error(tmp_path, sleeps):
    server = Server(cut_off)

    traj = run_open(server, tmp_path)

    assert traj.stop == "api_error" and "APIConnectionError" in traj.error
    assert len(server.requests) == MAX_RETRIES + 1


@pytest.mark.parametrize(
    "exc",
    [
        httpx2.DecodingError("bad gzip"),
        ValueError("Expecting value"),
        AttributeError("'str' object has no attribute 'choices'"),
        TypeError("'NoneType' object is not iterable"),
        KeyError("choices"),
        openai.APIConnectionError(request=httpx2.Request("POST", "https://openrouter.test")),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_exceptions_raised_by_the_call_or_the_reading_of_its_reply_are_classified(tmp_path, exc):
    def create(**kwargs):
        raise exc

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    traj = run_agent_openai(client, AgentConfig(OPEN_MODEL), "task", tmp_path, None)

    assert traj.stop == "api_error" and traj.error.startswith(type(exc).__name__)


def test_a_pinned_endpoint_that_does_not_exist_is_an_api_error_that_says_so(tmp_path):
    server = Server(http(404, NO_ENDPOINT))

    traj = run_open(server, tmp_path)

    assert traj.stop == "api_error" and traj.turns == 0
    assert "NotFoundError" in traj.error and "No endpoints found" in traj.error
    pin = server.sent()["provider"]  # allow_fallbacks stays off: no silent switch to another host
    assert pin == PIN


def test_a_reply_that_cannot_be_priced_ends_the_trial_as_unpriced(tmp_path):
    no_usage = completion()
    del no_usage["usage"]
    for body in (completion(cost=None), completion(cost=0), no_usage):
        traj = run_open(Server(http(body=body)), tmp_path)

        assert traj.stop == "api_error" and traj.error.startswith("unpriced")


def test_a_blocked_reply_is_a_refusal_even_when_it_cost_nothing(tmp_path):
    blocked = completion({"role": "assistant", "content": None}, "content_filter", cost=0)

    traj = run_open(Server(http(body=blocked)), tmp_path)

    assert traj.stop == "refusal" and traj.refusal_category == "content_filter"


def test_a_failure_on_a_later_turn_keeps_the_trajectory_and_the_spend_so_far(tmp_path):
    outage = Server(http(body=completion(bash(), "tool_calls", cost=0.002)), http(503, {}))

    traj = run_open(outage, tmp_path)

    assert traj.stop == "api_error" and traj.turns == 1 and traj.commands == 1
    assert traj.usd == pytest.approx(0.002)
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant", "tool"]


def test_the_turn_whose_finish_reason_is_an_error_is_recorded_with_its_spend(tmp_path):
    server = Server(
        http(body=completion(bash(), "tool_calls", cost=0.002)), http(body=finish_error())
    )

    traj = run_open(server, tmp_path)

    assert traj.stop == "api_error" and traj.turns == 2 and traj.commands == 1
    assert traj.usd == pytest.approx(0.0025)
    assert traj.finish_reasons == {"tool_calls": 1, "error": 1}
    last = traj.messages[-1]
    assert last["role"] == "assistant" and last["finish_reason"] == "error"


# --- credit, retries and the circuit breaker -----------------------------------------------------


def test_insufficient_credit_is_credit_exhausted_and_no_later_trial_calls_the_api(tmp_path):
    server = Server(http(402, {"error": {"message": "Insufficient credits", "code": 402}}))
    client = openrouter(server)

    first = run_open(server, tmp_path, client=client)
    second = run_open(server, tmp_path, client=client)

    assert first.stop == "credit_exhausted" and "Insufficient credits" in first.error
    assert second.stop == "credit_exhausted" and second.turns == 0
    assert len(server.requests) == 1  # the second trial never reached the transport
    assert first.stop in ABORT_STOPS <= INFRA_STOPS


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limits_and_outages_are_retried_for_about_half_a_minute(tmp_path, sleeps, status):
    server = Server(http(status, {"error": {"message": "busy", "code": status}}))

    traj = run_open(server, tmp_path, retries=4)  # the client cli.py builds still has 4

    assert traj.stop == "api_error"
    assert len(server.requests) == MAX_RETRIES + 1 and len(sleeps) == MAX_RETRIES
    for n, delay in enumerate(sleeps):  # 0.5 s doubling to 8 s, each cut by up to a quarter
        base = min(0.5 * 2**n, 8.0)
        assert 0.75 * base <= delay <= base
    assert 29 < sum(sleeps) <= 39.5


def test_a_transient_failure_is_retried_and_the_trial_goes_on(tmp_path, sleeps):
    busy = http(429, {"error": {"message": "busy"}}, headers={"retry-after": "3"})
    server = Server(http(503, {}), busy, http(body=completion()))

    traj = run_open(server, tmp_path)

    assert traj.stop == "end_turn" and traj.turns == 1 and len(server.requests) == 3
    assert 0.375 <= sleeps[0] <= 0.5 and sleeps[1] == 3.0  # Retry-After is honoured


def test_the_retry_budget_is_set_per_trial_and_can_be_switched_off_or_left_to_the_client(tmp_path):
    down = http(503, {})
    off, own, default = Server(down), Server(down), Server(down)

    run_open(off, tmp_path, retries=4, max_retries=0)  # the preflight probe: every 429 shows up
    run_open(own, tmp_path, retries=4, max_retries=None)
    run_open(default, tmp_path, retries=4)

    assert (len(off.requests), len(own.requests), len(default.requests)) == (1, 5, 9)


def test_a_client_without_with_options_is_used_as_it_is():
    scripted = SimpleNamespace(chat=None)

    assert with_retries(scripted, 8) is scripted
    assert with_retries(scripted, None) is scripted


def test_the_breaker_stops_the_job_after_n_api_errors_in_a_row(tmp_path):
    breaker = CircuitBreaker(limit=3)
    server = Server(http(404, NO_ENDPOINT))

    runs = [run_open(server, tmp_path, breaker=breaker) for _ in range(5)]

    assert [t.stop for t in runs] == ["api_error"] * 2 + ["circuit_open"] * 3
    assert len(server.requests) == 3  # the last two did not call
    assert runs[2].error.startswith("3 api_errors in a row, last: NotFoundError")
    assert runs[4].error == runs[2].error and runs[2].stop in ABORT_STOPS


def test_a_good_reply_resets_the_count_of_api_errors_in_a_row(tmp_path):
    breaker = CircuitBreaker(limit=3)
    down, up = http(404, NO_ENDPOINT), http(body=completion())
    server = Server(down, down, up, down, down)

    stops = [run_open(server, tmp_path, breaker=breaker).stop for _ in range(5)]

    assert stops == ["api_error", "api_error", "end_turn", "api_error", "api_error"]
    assert breaker.stop is None


def test_trials_that_share_a_client_share_a_breaker_and_a_new_client_starts_clean(tmp_path):
    server = Server(http(404, NO_ENDPOINT))
    shared = openrouter(server)
    n = CircuitBreaker().limit  # the default limit

    stops = [run_open(server, tmp_path, client=shared).stop for _ in range(n + 1)]

    assert stops == ["api_error"] * (n - 1) + ["circuit_open"] * 2
    assert len(server.requests) == n  # the last trial did not call
    assert run_open(server, tmp_path).stop == "api_error"  # another client, clean breaker


def test_a_client_without_a_dict_gets_no_breaker():
    class Slotted:
        __slots__ = ("chat",)

    assert breaker_for(Slotted(), AgentConfig(OPEN_MODEL)).limit == 0


def test_a_limit_of_zero_never_trips(tmp_path):
    breaker = CircuitBreaker(limit=0)
    server = Server(http(404, NO_ENDPOINT))

    stops = {run_open(server, tmp_path, breaker=breaker).stop for _ in range(25)}

    assert stops == {"api_error"} and breaker.stop is None


def test_the_default_limit_comes_from_the_environment():
    code = "from biowarden.bench.runner.agent import BREAKER_LIMIT; print(BREAKER_LIMIT)"
    env = {**os.environ, "BIOWARDEN_BREAKER_LIMIT": "7"}
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert done.stdout.strip() == "7", done.stderr


def test_the_breaker_trips_exactly_once_however_many_threads_report_failures():
    breaker = CircuitBreaker(limit=100)
    tripped: list[bool] = []

    def worker():
        tripped.extend(breaker.failure("down") for _ in range(25))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tripped.count(True) == 1 and breaker.stop == "circuit_open"


def test_a_config_holding_a_breaker_can_be_copied_and_digested():
    breaker = CircuitBreaker(limit=5)
    cfg = AgentConfig(OPEN_MODEL, breaker=breaker)

    assert asdict(cfg)["breaker"] is breaker  # asdict must not try to copy the lock
    assert "CircuitBreaker(limit=5)" in repr(cfg) and "0x" not in repr(
        cfg
    )  # no address in a digest


# --- the Claude loop -----------------------------------------------------------------------------


def test_a_claude_reply_is_priced_and_its_stop_reason_counted(tmp_path):
    server = Server(
        http(body=claude_message(tool_use(), "tool_use")), http(body=claude_message([]))
    )

    traj = run_claude(server, tmp_path)

    assert traj.stop == "end_turn" and traj.turns == 2 and traj.commands == 1
    assert traj.finish_reasons == {"tool_use": 1, "end_turn": 1}
    assert traj.tokens["input_tokens"] == 2000 and traj.tokens["output_tokens"] == 400
    price = usage_usd(CLAUDE, SimpleNamespace(input_tokens=2000, output_tokens=400))
    assert traj.usd == pytest.approx(price) and price > 0


@pytest.mark.parametrize(
    ("reply", "model", "expected"),
    [
        (http(text="<html>502</html>"), CLAUDE, "unexpected reply: <html>502"),
        (http(raw=b""), CLAUDE, "JSONDecodeError"),
        (http(body={"type": "message"}), CLAUDE, "unexpected reply"),
        (http(body=claude_message([])), "claude-mystery-1", "unpriced: no price for model"),
        (http(529, claude_error("overloaded_error")), CLAUDE, "OverloadedError"),
        (http(400, claude_error("invalid_request_error", "too long")), CLAUDE, "BadRequestError"),
    ],
    ids=["html-200", "empty-json", "no-content", "unpriced-model", "overloaded", "bad-request"],
)
def test_claude_replies_the_loop_cannot_use_end_the_trial_as_api_error(
    tmp_path, reply, model, expected
):
    traj = run_claude(Server(reply), tmp_path, model=model)

    assert traj.stop == "api_error" and expected in traj.error


LOW_CREDIT = "Your credit balance is too low to access the Anthropic API. Go to Plans & Billing."


@pytest.mark.parametrize(
    "reply",
    [
        http(400, claude_error("invalid_request_error", LOW_CREDIT)),
        http(402, claude_error("billing_error")),
    ],
    ids=["low-credit-400", "billing-402"],
)
def test_claude_credit_errors_are_credit_exhausted(tmp_path, reply):
    server = Server(reply)
    client = anthropic_client(server)

    first = run_claude(server, tmp_path, client=client)
    second = run_claude(server, tmp_path, client=client)

    assert first.stop == "credit_exhausted" and second.stop == "credit_exhausted"
    assert len(server.requests) == 1


def test_claude_overload_is_retried_for_about_half_a_minute(tmp_path, sleeps):
    server = Server(http(529, claude_error("overloaded_error", "busy")))

    traj = run_claude(server, tmp_path, retries=4)

    assert traj.stop == "api_error" and len(server.requests) == MAX_RETRIES + 1
    assert 29 < sum(sleeps) <= 39.5


def test_a_claude_failure_on_a_later_turn_keeps_the_trajectory_and_the_spend(tmp_path):
    server = Server(http(body=claude_message(tool_use(), "tool_use")), http(503, {}))

    traj = run_claude(server, tmp_path)

    assert traj.stop == "api_error" and traj.turns == 1 and traj.commands == 1
    assert traj.usd > 0 and [m["role"] for m in traj.messages] == ["user", "assistant", "user"]


def test_a_tripped_breaker_ends_a_claude_trial_before_its_call(tmp_path):
    breaker = CircuitBreaker(limit=1)
    breaker.failure("another trial's outage")
    server = Server(http(body=claude_message([])))

    traj = run_claude(server, tmp_path, breaker=breaker)

    assert traj.stop == "circuit_open" and not server.requests


# --- what the stage-1 comparison relies on -----------------------------------------------------


def test_the_system_prompts_of_stage_1_are_unchanged():
    """sha256 prefixes of the system messages stored in the stage-1 transcripts."""
    seen = {
        ("full", False, False): "40f88097471e",
        ("full", True, False): "93736b6d3165",
        ("bare", False, False): "d01f97a63200",
        ("bare", True, False): "f793ae77414b",
        ("bare", False, True): "8aea1548439c",
    }

    for args, digest in seen.items():
        assert hashlib.sha256(system_prompt(*args).encode()).hexdigest()[:12] == digest, args


def test_the_pinned_providers_and_their_routing_are_unchanged():
    assert provider_routing(OPEN_MODEL) == PIN
    assert provider_routing("openai/gpt-oss-120b") == {
        "order": ["near-ai/fp4"],
        "allow_fallbacks": False,
    }


# ---- audit of the stage-2 fixes: spend guard, absurd replies, malformed environment ------------


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -0.5], ids=["nan", "inf", "negative"])
def test_a_reply_with_a_nan_infinite_or_negative_cost_is_unpriced_and_never_added_to_the_spend(
    tmp_path, cost
):
    traj = run_open(Server(http(text=json.dumps(completion(cost=cost)))), tmp_path)

    assert traj.stop == "api_error" and traj.error.startswith("unpriced")
    assert traj.usd == 0.0  # the spend guard stays usable


@pytest.mark.parametrize(
    "text",
    [json.dumps(completion(cost=10**400)), "[" * 100_000],
    ids=["a cost no float can hold", "a body nested too deep"],
)
def test_absurd_replies_are_api_errors_not_crashes(tmp_path, text):
    traj = run_open(Server(http(text=text)), tmp_path)

    assert traj.stop == "api_error"


@pytest.mark.parametrize("value", ["", " ", "ten", "-3", "1e3"])
def test_a_malformed_breaker_limit_falls_back_instead_of_breaking_the_import(monkeypatch, value):
    from biowarden.bench.runner import agent as agent_module

    monkeypatch.setenv("BIOWARDEN_BREAKER_LIMIT", value)

    assert agent_module._limit_from_env(10) == (0 if value == "-3" else 10)
    monkeypatch.setenv("BIOWARDEN_BREAKER_LIMIT", "4")
    assert agent_module._limit_from_env(10) == 4
