"""The engine behind a unix socket, one per trial, for shims that run inside a sandbox.

The sandbox holds only the stdlib shim client; the engine, the specs and the real tools stay on the
host. Paths are translated both ways (the sandbox sees /work, the host sees the trial directory) so
checkers read real files and the agent never sees a host path. Only the work directory is shared:
a call that runs somewhere else is not checked, and a file name that leads out of the work directory
does not exist for the checkers. A line on the socket that is not a shim request is answered with
"allow" and no text, so a probe learns nothing, and is logged in `errors`.
"""

from __future__ import annotations

import json
import re
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.engine.call import Call, DeclarationError
from science_harness.engine.checks import inside
from science_harness.engine.decide import Engine
from science_harness.engine.render import render_block, render_warning

SANDBOX_WORK = "/work"
MAX_LOGGED = 100  # faults and malformed requests kept in `errors`: an agent can probe in a loop
ALLOW = {"action": "allow", "text": ""}


class MalformedRequest(ValueError):
    """A line on the socket that is not a request the shim client would send."""


def check_request(request: Any) -> None:
    """Raise MalformedRequest unless `request` has the shape the shim client sends."""
    if not isinstance(request, dict):
        raise MalformedRequest("not an object")
    for key in ("tool", "cwd"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise MalformedRequest(f"{key} must be a non-empty string")
    if not isinstance(request.get("env_declare", ""), str):
        raise MalformedRequest("env_declare must be a string")
    for key in ("argv", "declare"):
        items = request.get(key, [] if key == "declare" else None)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise MalformedRequest(f"{key} must be a list of strings")


@dataclass(frozen=True)
class PathMap:
    host: Path
    sandbox: str = SANDBOX_WORK

    @property
    def root(self) -> Path | None:
        """Where the host may follow the agent's paths: the work directory, whenever a sandbox
        sits between the two (the sandbox's /tmp is not the host's). None: no sandbox, no limit."""
        return self.host if self.sandbox != str(self.host) else None

    def to_host(self, text: str) -> str:
        pattern = re.escape(self.sandbox) + r"(?![\w.-])"
        return re.sub(pattern, lambda _: str(self.host), text)

    def cwd_to_host(self, cwd: str) -> str:
        """A working directory is one path, so only its start can be the sandbox's; any other
        path (the sandbox's /tmp, a directory called .../work) stays as it is."""
        if cwd == self.sandbox or cwd.startswith(self.sandbox.rstrip("/") + "/"):
            return str(self.host) + cwd[len(self.sandbox) :]
        return cwd

    def to_sandbox(self, text: str) -> str:
        return text.replace(str(self.host), self.sandbox)


class _Handler(socketserver.StreamRequestHandler):
    server: EngineServer

    def handle(self) -> None:
        line = self.rfile.readline(8_000_000)
        try:
            try:
                request = json.loads(line)
            except (ValueError, RecursionError) as exc:  # bad JSON or UTF-8, a line of brackets
                raise MalformedRequest(f"not JSON ({type(exc).__name__})") from None
            reply = self.server.answer(request)
        except MalformedRequest as exc:  # a probe of the socket, not a call: it gets no text
            self.server.log(f"malformed request: {exc}")
            reply = dict(ALLOW)
        except Exception as exc:  # never leave the shim hanging; the call then runs unchecked
            self.server.log(f"{type(exc).__name__}: {exc}")
            reply = {"action": "warn", "text": "[tbm] the checking layer failed on this call"}
        try:
            self.wfile.write((json.dumps(reply) + "\n").encode())
        except OSError:  # the client has gone
            pass


class EngineServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, socket_path: Path, engine: Engine, paths: PathMap) -> None:
        if socket_path.exists():
            socket_path.unlink()
        super().__init__(str(socket_path), _Handler)
        socket_path.chmod(0o600)
        self.socket_path = socket_path
        self.engine = engine
        self.paths = paths
        self.errors: list[str] = []
        # (tool, argv) -> rule ids the command pre-check already warned about
        self.reported: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        self.calls = 0
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        if len(self.errors) < MAX_LOGGED:
            self.errors.append(message[:300])

    def answer(self, request: Any) -> dict[str, str]:
        check_request(request)
        argv = [self.paths.to_host(a) for a in request["argv"]]
        markers = [m for pair in request.get("declare", []) for m in ("--declare", pair)]
        cwd = Path(self.paths.cwd_to_host(request["cwd"]))
        env = {"TBM_DECLARE": request.get("env_declare", "")}
        root = self.paths.root
        with self._lock:
            self.calls += 1
            if root is not None and not inside(cwd, root):
                # the agent stands where the host sees other files than the sandbox does (for
                # example /tmp): judge the call from the work directory, so that a `cd /tmp` is no
                # way round the layer; the paths it names under /work still resolve
                cwd = root
            try:
                call = Call.from_argv(request["tool"], [*argv, *markers], cwd, env, root)
            except DeclarationError as exc:
                return {"action": "block", "text": f"BLOCKED, the command did not run. {exc}"}
            decision = self.engine.evaluate(call)
        if decision.action == "block":
            text = render_block(
                decision, self.engine.retry_cap, self.engine.style, self.engine.contract
            )
        elif decision.action == "warn":
            skip = self.reported.get((call.tool, call.argv), set())
            text = render_warning(decision, skip, self.engine.style)
        else:
            text = ""
        return {"action": decision.action, "text": self.paths.to_sandbox(text)}


def serve_in_thread(server: EngineServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="tbm-engine")
    thread.start()
    return thread
