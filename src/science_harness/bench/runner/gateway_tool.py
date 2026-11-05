"""The gateway arm: a second tool, `run_tool`, next to bash.

It is the in-loop twin of the MCP front end (engine/mcp_server.py would put the same function behind
a FastMCP server): the agent hands over the tool name, its arguments and the conventions it
declares; the engine decides; an allowed call runs inside the same sandbox as bash would, never on
the host. Bash stays available, so a model can route around this tool; how often it does is the
bypass rate the experiment reports.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.runner.exposure import BLOCKED_MARK
from science_harness.bench.runner.sandbox import WORKDIR, run_command
from science_harness.bench.tasks.tools import TOOL_NAMES, ToolEnv
from science_harness.engine import Call, Engine
from science_harness.engine.render import render_block, render_warning
from science_harness.engine.server import PathMap

DESCRIPTION = (
    "Run samtools, bcftools or bedtools in the working directory with a safety check. `args` are "
    "the command-line arguments after the tool name. `declared` states the conventions you "
    'intend (for example {"region_semantics": "overlap"}); an error message tells you when a '
    "declaration is needed. Returns the exit code and output like bash does."
)


@dataclass(frozen=True)
class GatewayTool:
    run: Callable[[dict[str, Any]], tuple[str, bool]]  # args -> (text, is_error)
    declared_schema: dict[str, Any]

    def _input_schema(self) -> dict[str, Any]:
        declared = {**self.declared_schema, "required": []}
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(TOOL_NAMES)},
                "args": {"type": "array", "items": {"type": "string"}},
                "declared": declared,
            },
            "required": ["tool", "args"],
        }

    def anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": "run_tool",
            "description": DESCRIPTION,
            "input_schema": self._input_schema(),
        }

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_tool",
                "description": DESCRIPTION,
                "parameters": self._input_schema(),
            },
        }


NOT_RUN = BLOCKED_MARK  # the text the exposure count reads as 'this command did not run'


def make_gateway(
    engine: Engine, paths: PathMap, work: Path, tools: ToolEnv, timeout: int, max_chars: int
) -> GatewayTool:
    host_work = work.resolve()

    def run(args: dict[str, Any]) -> tuple[str, bool]:
        tool, argv, declared = args.get("tool"), args.get("args"), args.get("declared") or {}
        # a refusal below never ran anything: the mark keeps it out of the executed commands
        if tool not in TOOL_NAMES or not isinstance(argv, list):
            return (
                f"Error: send {{tool: samtools|bcftools|bedtools, args: [strings]}} ({NOT_RUN})",
                True,
            )
        if not all(isinstance(a, str) for a in argv) or not isinstance(declared, dict):
            return (
                "Error: args must be a list of strings and declared an object of strings "
                f"({NOT_RUN})",
                True,
            )
        if any(a == "--declare" or a.startswith("--declare=") for a in argv):
            # no shim strips it here, so the engine would never read it and the tool would fail
            return (
                "Error: --declare is not a tool argument in run_tool; state the conventions in "
                f'the `declared` object, for example {{"region_semantics": "overlap"}} ({NOT_RUN})',
                True,
            )
        call = Call(
            tool,
            tuple(paths.to_host(a) for a in argv),
            host_work,
            {str(k): str(v) for k, v in declared.items()},
        )
        decision = engine.evaluate(call, source="gateway")
        if decision.action == "block":
            block = render_block(decision, engine.retry_cap, engine.style, engine.contract)
            return paths.to_sandbox(block), True
        result = run_command(host_work, tools, shlex.join([tool, *argv]), timeout, max_chars)
        text = result.render()
        if decision.action == "warn":
            text = f"{paths.to_sandbox(render_warning(decision, style=engine.style))}\n{text}"
        return text, result.exit_code != 0

    schema = _declared_schema(engine)
    return GatewayTool(run, schema)


def _declared_schema(engine: Engine) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for spec in engine.specs:
        properties.update(spec.declaration_schema()["properties"])
    return {"type": "object", "properties": properties, "additionalProperties": False}


__all__ = ["WORKDIR", "GatewayTool", "make_gateway"]
