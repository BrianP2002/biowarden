"""MCP front end: the engine behind one `run_tool` function, for hosts that speak MCP.

    biowarden-mcp --cwd DIR [--ledger FILE] [--mode promoted|warn|block]

The host's model hands over the tool name, its arguments and the conventions it declares. The engine
decides; an allowed call runs the real tool in `cwd`. A refused call is an MCP error result whose
FIRST text block is the fix (hosts may drop structured content on errors, so nothing depends on it).
This server does not sandbox: run it where the model could run the same commands anyway.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from biowarden.engine.call import Call, has_declare_marker
from biowarden.engine.decide import Engine, Mode, Promotion
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.spec import collect_specs
from biowarden.engine.tools import (
    call_rule_tools,
    find_binaries,
    import_python_rules,
    path_dirs,
)
from biowarden.ledger import Ledger

Runner = Callable[[str, Sequence[str], Path], tuple[int, str, str]]

INSTRUCTIONS = (
    "Run a command-line tool through a checked entry point. Pass the arguments that "
    "follow the tool name in `args`. Pass the conventions you intend in `declared` (for example "
    '{"region_semantics": "overlap"}). A refused call is an error whose first line is the fix: '
    "apply it instead of repeating the same call."
)


def subprocess_runner(
    real_bins: Mapping[str, Path], env: Mapping[str, str] | None = None, timeout: float = 120.0
) -> Runner:
    def run(tool: str, args: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        exe = real_bins.get(tool)
        if exe is None:
            return 127, "", f"{tool}: command not found"
        try:
            proc = subprocess.run(
                [str(exe), *args],
                cwd=cwd,
                env=None if env is None else dict(env),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"{tool}: timed out after {timeout:.0f}s"
        return proc.returncode, proc.stdout, proc.stderr

    return run


def declaration_help(engine: Engine) -> str:
    lines = []
    for spec in engine.specs:
        for rule in spec.rules:
            if rule.declare is not None:
                values = "|".join(rule.declare.values)
                lines.append(f"{rule.declare.key} = {values}: {rule.message}")
    return "\n".join(lines)


def build_mcp(engine: Engine, runner: Runner, cwd: Path, tools: Sequence[str]) -> FastMCP:
    mcp = FastMCP("biowarden", instructions=INSTRUCTIONS, strict_input_validation=True)
    keys = declaration_help(engine)

    def run_tool(
        tool: str,
        args: list[str],
        declared: dict[str, str] | None = None,
    ) -> ToolResult:
        if has_declare_marker(args):
            # nothing here strips it, so the engine would never read it and the tool would fail
            text = (
                "Error: --declare is not a tool argument in run_tool; state the conventions in "
                'the `declared` object, for example {"region_semantics": "overlap"}'
            )
            rejected = {"status": "rejected"}
            return ToolResult(content=text, structured_content=rejected, is_error=True)
        call = Call(tool, tuple(args), cwd, dict(declared or {}))
        decision = engine.evaluate(call, source="mcp")
        if decision.action == "block":
            text = render_block(decision, engine.retry_cap, engine.style, engine.contract)
            return ToolResult(content=text, structured_content={"status": "blocked"}, is_error=True)
        code, out, err = runner(tool, args, cwd)
        parts = [f"[exit code {code}]"]
        warning = render_warning(decision, style=engine.style) if decision.action == "warn" else ""
        if warning:  # a fault of the layer is a warn decision with nothing to say
            parts.insert(0, warning)
        if out:
            parts.append(f"--- stdout ---\n{out.rstrip()}")
        if err:
            parts.append(f"--- stderr ---\n{err.rstrip()}")
        return ToolResult(
            content="\n".join(parts),
            structured_content={"status": decision.action, "exit_code": code},
            is_error=code != 0,
        )

    # the tool name is an enum with one value per served program (built at run time)
    run_tool.__annotations__["tool"] = Literal[tuple(tools)]  # type: ignore[valid-type]
    mcp.tool(description=f"{INSTRUCTIONS}\n\nDeclarations:\n{keys}")(run_tool)
    return mcp


def main(argv: Sequence[str] | None = None) -> int:
    from biowarden.bench.tasks.tools import ToolEnv

    parser = argparse.ArgumentParser(prog="biowarden-mcp", description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--mode", default="promoted", choices=["off", "warn", "block", "promoted"])
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--specs", type=Path, action="append", default=[], help="extra rule specs")
    parser.add_argument("--no-packaged", action="store_true", help="do not load the shipped rules")
    parser.add_argument("--watch", action="append", default=[], help="also serve this program")
    parser.add_argument("--trust-python", action="store_true", help="import *.py checkers")
    args = parser.parse_args(argv)
    discovered = ToolEnv.discover(args.cwd)
    dirs = [*(map(str, discovered.bin_dirs) if discovered else []), *path_dirs()]
    if args.trust_python:
        import_python_rules(args.specs)
    specs = collect_specs(args.specs, packaged=not args.no_packaged)
    real = find_binaries([*call_rule_tools(specs), *args.watch], dirs)
    if not real:
        parser.error("none of the programs named by the rules or --watch were found on PATH")
    mode: Mode = args.mode
    engine = Engine(
        specs,
        mode=mode,
        promotion=Promotion.load(args.promotion) if args.promotion else Promotion.packaged(),
        real_bins=real,
        real_env=discovered.process_env() if discovered else None,
        ledger=Ledger(args.ledger) if args.ledger else None,
    )
    runner = subprocess_runner(real, discovered.process_env() if discovered else None)
    build_mcp(engine, runner, args.cwd.resolve(), sorted(real)).run(show_banner=False)
    return 0


__all__: list[Any] = ["build_mcp", "main", "subprocess_runner"]
