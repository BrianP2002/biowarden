"""Check a whole bash command string before it runs (the pipeline-aware front end).

A shim sees one process at a time, so a pipeline stage that reads from stdin (`bcftools query ... |
awk ...`, `cat reads.fastq | python3 -c ...`, a here-document script) never names the data file the
rule cares about. Here the command string is split into simple commands first; each one is evaluated
with the words of the earlier ones (and any here-document body) as context, so fact rules can still
find the file. A block stops the whole command before anything runs. Shims keep running for what a
string cannot show (variables, globs, scripts), and skip warnings this pre-check already gave.

The shim is where `--declare` is read and removed, so the pre-check follows what the shim would do:
a malformed declaration on a watched program blocks here instead of after earlier commands ran, an
`export TBM_DECLARE=...` reaches the commands after it, and a watched program called by its path
(which the shim never sees) is told to be called by name when a declaration is involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from science_harness.engine.call import DECLARE_ENV, Call, DeclarationError, has_declare_marker
from science_harness.engine.decide import Action, Decision, Engine, tool_names
from science_harness.engine.render import render_block, render_warning
from science_harness.engine.server import PathMap
from science_harness.engine.shellsplit import SimpleCommand, split_command
from science_harness.engine.tools import watched_tools

_RANK = {"allow": 0, "warn": 1, "block": 2}
MAX_CHECKED_COMMANDS = 200  # simple commands of one string that are checked; the rest only warn
TOO_MANY = "WARNING: too many commands in one call; only the first {n} were checked."
SHIM_DIR = "/opt/tbm/bin/"  # where the sandbox mounts the shims; a program there is a shim
DECLARATION_CODES = ("declaration_missing", "declaration_invalid")


@dataclass(frozen=True)
class Precheck:
    action: Action
    text: str
    decisions: tuple[Decision, ...] = ()


def _exported(argv: Sequence[str], current: str) -> str:
    """TBM_DECLARE after `export NAME=value ...`: the shims read it from their environment."""
    for word in argv:
        name, sep, value = word.partition("=")
        if sep and name == DECLARE_ENV:
            current = value
    return current


def _by_path(item: SimpleCommand, watched: set[str]) -> bool:
    """A watched program called by a path never meets the shim, which is what strips --declare."""
    return (
        "/" in item.program
        and not item.program.startswith(SHIM_DIR)
        and any(name in watched for name in tool_names(item.tool))
    )


def _by_name(item: SimpleCommand, blocking: bool) -> str:
    headline = "--declare does not work on a tool called by its path."
    fix = f"call {item.tool} by name, not as {item.program}; --declare is read only then."
    if blocking:
        return f"FIX: {fix}\nBLOCKED, the command did not run. {headline}"
    return f"WARNING {headline} FIX: {fix}"


def precheck(
    command: str,
    cwd: Path,
    engine: Engine,
    paths: PathMap | None = None,
    reported: dict[tuple[str, tuple[str, ...]], set[str]] | None = None,
) -> Precheck:
    """Evaluate every simple command in `command`; `cwd` is where it will start running."""
    if engine.mode == "off":
        return Precheck("allow", "")
    host_text = paths.to_host(command) if paths else command
    host_cwd = Path(paths.to_host(str(cwd))) if paths else cwd
    root = paths.root if paths else None
    parsed = split_command(host_text, host_cwd)
    watched = set(watched_tools(engine.specs))
    exported = ""  # TBM_DECLARE as an earlier `export` in this string left it
    context: list[str] = []
    decisions: list[Decision] = []
    warnings: list[str] = []

    def blocked(text: str) -> Precheck:
        return Precheck("block", paths.to_sandbox(text) if paths else text, tuple(decisions))

    for number, item in enumerate(parsed.commands):
        if number >= MAX_CHECKED_COMMANDS:
            warnings.append(TOO_MANY.format(n=MAX_CHECKED_COMMANDS))
            break
        if item.tool == "export":
            exported = _exported(item.argv, exported)
        env = {DECLARE_ENV: dict(item.env).get(DECLARE_ENV, exported)}
        try:
            call = Call.from_argv(item.tool, item.argv, item.cwd or host_cwd, env, root)
        except DeclarationError as exc:
            if any(name in watched for name in tool_names(item.tool)):
                return blocked(f"BLOCKED, the command did not run. {exc}")  # the shim would too
            context.extend(item.argv)
            continue
        seen = (*context, item.heredoc) if item.heredoc else tuple(context)
        call = replace(call, context=tuple(seen))
        decision = engine.evaluate(call, source="precheck")
        decisions.append(decision)
        if engine.style == "full" and _by_path(item, watched):
            asked = [
                o.effective
                for o in decision.outcomes
                if o.result.code in DECLARATION_CODES and o.effective != "none"
            ]
            marked = has_declare_marker(item.argv)
            if marked or asked:  # either way the agent is about to give --declare to the tool
                if (marked and engine.mode != "warn") or "block" in asked:
                    return blocked(_by_name(item, True))
                warnings.append(_by_name(item, False))
        if decision.action == "block":
            return blocked(render_block(decision, engine.retry_cap, engine.style, engine.contract))
        if decision.action == "warn":
            warning = render_warning(decision, style=engine.style)
            if warning:
                warnings.append(warning)
            if reported is not None:
                reported.setdefault((call.tool, call.argv), set()).update(
                    o.rule.id for o in decision.outcomes if o.effective == "warn"
                )
        context.extend(item.argv)
    if not warnings:
        return Precheck("allow", "", tuple(decisions))
    text = "\n".join(warnings)
    return Precheck("warn", paths.to_sandbox(text) if paths else text, tuple(decisions))
