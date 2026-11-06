"""A logging reverse proxy between a sandboxed harness and OpenRouter.

The harness runs in a sandbox with no network; it reaches this proxy through a unix socket. The
proxy holds the real API key (the harness gets a dummy one, so a model that prints its environment
leaks nothing), forwards chat-completion requests, relays streamed responses unchanged, and keeps a
per-trial record of what the model actually did: the bash commands it asked for, the tokens and USD
OpenRouter reports, which upstream provider served each turn, and malformed tool-call arguments.
That record is harness-agnostic, so exposure accounting and cost work for any harness.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socketserver
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

UPSTREAM_URL = "https://openrouter.ai"
PATH_RE = re.compile(r"^/t/([A-Za-z0-9_.\-]+)(/api/v1/(?:chat/completions|models))(?:\?.*)?$")
LOG_CAP = 8_000_000  # bytes of one response kept for parsing


@dataclass
class TrialStats:
    requests: int = 0
    usd: float = 0.0
    tokens: Counter = field(default_factory=Counter)
    providers: Counter = field(default_factory=Counter)
    finish_reasons: Counter = field(default_factory=Counter)
    commands: list[str] = field(default_factory=list)
    tool_arg_errors: int = 0
    http_errors: int = 0
    refused: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "usd": round(self.usd, 6),
            "tokens": dict(self.tokens),
            "providers": dict(self.providers),
            "finish_reasons": dict(self.finish_reasons),
            "tool_arg_errors": self.tool_arg_errors,
            "http_errors": self.http_errors,
        }


def parse_completion(text: str, streamed: bool) -> dict[str, Any]:
    """Reduce a chat-completion response (JSON or SSE text) to what the benchmark needs."""
    chunks: list[dict[str, Any]] = []
    if streamed:
        for line in text.splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    chunks.append(json.loads(line[5:]))
                except ValueError:
                    continue
    else:
        try:
            chunks = [json.loads(text)]
        except ValueError:
            chunks = []
    out: dict[str, Any] = {"content": "", "tool_calls": [], "finish_reason": None, "usage": None}
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        out["provider"] = chunk.get("provider", out.get("provider"))
        out["model"] = chunk.get("model", out.get("model"))
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            out["finish_reason"] = choice.get("finish_reason") or out["finish_reason"]
            part = choice.get("delta") if streamed else choice.get("message")
            part = part or {}
            out["content"] += part.get("content") or ""
            for call in part.get("tool_calls") or []:
                slot = calls.setdefault(
                    call.get("index", len(calls)), {"name": "", "arguments": ""}
                )
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    out["tool_calls"] = [calls[i] for i in sorted(calls)]
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited bodies keep streaming trivial
    server: ProxyServer

    def address_string(self) -> str:
        return "unix"

    def log_message(self, *args: Any) -> None:
        return

    def _reply(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def _forward(self, method: str) -> None:
        match = PATH_RE.match(self.path)
        if not match:
            return self._reply(403, "path not allowed by the benchmark proxy")
        trial, upstream_path = match.groups()
        stats = self.server.stats_for(trial)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        streamed = False
        if method == "POST":
            try:
                payload = json.loads(raw)
            except ValueError:
                return self._reply(400, "body is not JSON")
            if payload.get("model") not in self.server.allowed_models:
                return self._reply(403, f"model {payload.get('model')!r} not allowed")
            if stats.usd >= self.server.max_usd_per_trial:
                stats.refused = True
                return self._reply(402, "per-trial budget exhausted")
            streamed = bool(payload.get("stream"))
            payload["usage"] = {"include": True}
            if streamed:
                payload["stream_options"] = {"include_usage": True}
            for key in ("max_tokens", "max_completion_tokens"):
                if key in payload:
                    payload[key] = min(int(payload[key]), self.server.max_output_tokens)
            raw = json.dumps(payload).encode()
        started = time.time()
        conn = self.server.connect_upstream()
        headers = {
            "Authorization": f"Bearer {self.server.api_key}",
            "Content-Type": "application/json",
            "X-Title": "biowarden",
        }
        try:
            conn.request(method, upstream_path, body=raw or None, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            stats.http_errors += 1
            return self._reply(502, f"upstream error: {type(exc).__name__}")
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Connection", "close")
        self.end_headers()
        kept = bytearray()
        try:
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                if len(kept) < LOG_CAP:
                    kept += chunk
        except (BrokenPipeError, ConnectionResetError):
            pass  # the harness gave up on this request
        finally:
            conn.close()
        self.server.record(
            trial, stats, resp.status, bytes(kept), streamed, time.time() - started, raw
        )


class ProxyServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 256

    def __init__(
        self,
        socket_path: Path,
        api_key: str,
        allowed_models: set[str],
        log_dir: Path,
        max_usd_per_trial: float = 1.0,
        max_output_tokens: int = 16000,
        upstream_url: str = UPSTREAM_URL,
    ) -> None:
        if socket_path.exists():
            socket_path.unlink()
        super().__init__(str(socket_path), _Handler)
        os.chmod(socket_path, 0o600)
        self.socket_path = socket_path
        self.api_key = api_key
        self.allowed_models = allowed_models
        self.log_dir = log_dir
        self.max_usd_per_trial = max_usd_per_trial
        self.max_output_tokens = max_output_tokens
        self.upstream = urlparse(upstream_url)
        self._stats: dict[str, TrialStats] = {}
        self._lock = threading.Lock()
        log_dir.mkdir(parents=True, exist_ok=True)

    def connect_upstream(self) -> http.client.HTTPConnection:
        host, port = self.upstream.hostname or "", self.upstream.port
        if self.upstream.scheme == "http":
            return http.client.HTTPConnection(host, port or 80, timeout=900)
        return http.client.HTTPSConnection(host, port or 443, timeout=900)

    def stats_for(self, trial: str) -> TrialStats:
        with self._lock:
            return self._stats.setdefault(trial, TrialStats())

    def pop_stats(self, trial: str) -> TrialStats:
        with self._lock:
            return self._stats.pop(trial, TrialStats())

    def record(
        self,
        trial: str,
        stats: TrialStats,
        status: int,
        body: bytes,
        streamed: bool,
        secs: float,
        request: bytes,
    ) -> None:
        text = body.decode("utf-8", "replace")
        info = parse_completion(text, streamed) if status == 200 else {}
        with self._lock:
            stats.requests += 1
            if status != 200:
                stats.http_errors += 1
            usage = info.get("usage") or {}
            stats.usd += float(usage.get("cost") or 0.0)
            stats.tokens["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            stats.tokens["output_tokens"] += int(usage.get("completion_tokens") or 0)
            if info.get("provider"):
                stats.providers[info["provider"]] += 1
            if info.get("finish_reason"):
                stats.finish_reasons[info["finish_reason"]] += 1
            for call in info.get("tool_calls", []):
                try:
                    args = json.loads(call["arguments"] or "{}")
                except ValueError:
                    stats.tool_arg_errors += 1
                    continue
                command = args.get("command") if isinstance(args, dict) else None
                if call["name"] == "bash" and isinstance(command, str):
                    stats.commands.append(command)
        entry = {
            "ts": round(time.time(), 2),
            "status": status,
            "secs": round(secs, 2),
            "model": info.get("model"),
            "provider": info.get("provider"),
            "finish": info.get("finish_reason"),
            "usage": info.get("usage"),
            "content": (info.get("content") or "")[:600],
            "tool_calls": info.get("tool_calls", []),
            "request_bytes": len(request),
            "error": text[:300] if status != 200 else None,
        }
        with self._lock, (self.log_dir / f"{trial}.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


def serve_in_thread(server: ProxyServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="bench-proxy")
    thread.start()
    return thread
