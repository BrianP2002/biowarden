"""Did the agent actually run the call whose default behaviour is the trap?

A faulted trial only tests a convention if the trapped route is taken. Agents that read the README
and then use bedtools or their own awk never touch the trap, so a zero silent-error rate over such
trials says nothing. Exposure is computed offline from the command ledger with patterns fixed
before any confirmatory run; it is a property of the trial, not of the answer.

  F1  a bcftools region-file option (-R, -T, --regions-file, --targets-file) was executed
  F2  samtools faidx was executed on the region strings of regions.txt
  F3  a region tool was executed on the original regions.bed
  F4  ref.fa was read (always the case: the task needs it)

Every family check sees the commands after `drop_discovery`: looking a tool up (`which x`,
`type x`, `command -v x`, `whereis x`, `man x`, a call with `--help` or `--version`) never takes its
trapped route. `statements` splits a command, or the program inside it, for families that judge each
trapped call by its own statement; `trial_exposure` is the one call a runner makes per trial.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BCFTOOLS_REGION_FILE = re.compile(r"(?:^|\s)(?:-R|-T|--regions-file|--targets-file)(?:\s+|=)\S")
REGION_TOOLS = re.compile(r"\b(?:bedtools|bcftools|samtools|tabix)\b")
REGIONS_BED_ARG = re.compile(
    r"(?:-a|-b|-R|-T|-L|-i|--regions-file|--targets-file)(?:\s+|=)['\"]?(?:\./)?regions\.bed\b"
)


def commands_of(transcript: list[dict[str, Any]]) -> list[str]:
    """The bash commands the agent issued, in order (Anthropic and OpenAI-style transcripts)."""
    found: list[str] = []
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "tool_use" and block.get("name") == "bash":
                command = (block.get("input") or {}).get("command")
                if isinstance(command, str):
                    found.append(command)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "bash":
                continue
            try:
                command = json.loads(function.get("arguments") or "{}").get("command")
            except (ValueError, AttributeError):
                continue
            if isinstance(command, str):
                found.append(command)
    return found


BLOCKED_MARK = "BLOCKED, the command did not run"


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _as_command(name: str, args: Any) -> str | None:
    """The shell-like command a tool call stands for (bash, or the gateway's run_tool)."""
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    if name == "bash":
        command = args.get("command")
        return command if isinstance(command, str) else None
    if name == "run_tool" and isinstance(args.get("args"), list):
        return shlex.join([str(args.get("tool", "")), *map(str, args["args"])])
    return None


def _calls(transcript: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], set[str]]:
    """Every (call id, command) the agent issued, in order, and the ids the layer refused."""
    issued: list[tuple[str, str]] = []
    blocked: set[str] = set()
    for message in transcript:
        content = message.get("content")
        if message.get("role") == "assistant":
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use":
                    command = _as_command(block.get("name", ""), block.get("input"))
                    if command is not None:
                        issued.append((block.get("id", ""), command))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                command = _as_command(function.get("name", ""), function.get("arguments"))
                if command is not None:
                    issued.append((call.get("id", ""), command))
        elif message.get("role") == "tool" and BLOCKED_MARK in _text_of(content):
            blocked.add(message.get("tool_call_id", ""))
        elif message.get("role") == "user":
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_result" and BLOCKED_MARK in _text_of(
                    block.get("content")
                ):
                    blocked.add(block.get("tool_use_id", ""))
    return issued, blocked


def issued_commands_of(transcript: list[dict[str, Any]]) -> list[str]:
    """Every bash and run_tool command the agent issued, the ones the layer refused included."""
    return [command for _, command in _calls(transcript)[0]]


def executed_commands_of(transcript: list[dict[str, Any]]) -> list[str]:
    """Like commands_of, but only the commands that actually ran.

    A command the checking layer refused never executed the trapped route, so it must not count
    as exposure. The refusal is recognised by its message in the matching tool result.
    """
    issued, blocked = _calls(transcript)
    return [command for call_id, command in issued if call_id not in blocked]


# -- tool lookups and statements -------------------------------------------------------------

LOOKUPS = frozenset({"which", "type", "whereis", "man"})
# never a bare -h: it is fastp's --html and samtools view's "keep the header"
HELP_FLAGS = frozenset({"--help", "--version", "-help", "-version"})
SHELL_WORDS = frozenset(
    {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "(", "env", "time"}
)
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^']*)'|\"([^\"]*)\"|\\?([A-Za-z_]\w*))")


def statements(command: str) -> list[str]:
    """The statements of a command, in order: split at a newline or `;` outside brackets.

    For families that judge each trapped call by its own statement, so a safe variant elsewhere in
    the same command (`b.raw = None` beside the default call) does not set the whole command aside,
    and a handling statement can be told from one that comes before or after the trapped call.
    Quotes are ignored on purpose: the same split reads the shell line and the program inside it
    (`python -c "..."`, a here-document). A bracket that a string literal leaves open only makes
    the statements after it merge. A backslash-newline joins lines; empty statements are dropped.
    """
    text = re.sub(r"\\\r?\n", "", command)
    found: list[str] = []
    depth = start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in ";\n" and depth == 0:
            found.append(text[start:i])
            start = i + 1
    found.append(text[start:])
    return [s.strip() for s in found if s.strip()]


def _skip_heredocs(text: str, pos: int, pending: list[tuple[str, bool]]) -> int:
    """Index of the newline that ends the last terminator line of the pending here-documents."""
    for delimiter, strip_tabs in pending:
        while pos < len(text):
            end = text.find("\n", pos)
            end = len(text) if end < 0 else end
            line = text[pos:end]
            pos = end + 1
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
    return min(pos - 1, len(text))


def _segments(text: str) -> list[tuple[str, str]]:
    """The simple commands of a shell line as (operator before it, text) pairs.

    Split at `;`, a newline, `&&`, `||`, `|` and a lone ` & `. Quoted text, backticks, brackets,
    comments and here-document bodies stay inside the command around them.
    """
    parts: list[tuple[str, str]] = []
    operator, start, depth, quote, i = "", 0, 0, "", 0
    pending: list[tuple[str, bool]] = []  # here-document terminators awaiting their body
    while i < len(text):
        ch = text[i]
        if quote:
            i += 2 if ch == "\\" and quote != "'" else 1
            quote = "" if ch == quote else quote
            continue
        if ch == "\\":
            i += 2
            continue
        if ch in "'\"`":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1] in " \t\n;&|("):
            i = text.find("\n", i)  # a comment runs to the end of the line
            if i < 0:
                break
            continue
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            found = ""
            if ch == "<" and text.startswith("<<", i):
                heredoc = None if text.startswith("<<<", i) else HEREDOC.match(text, i)
                if heredoc:
                    word = heredoc.group(2) or heredoc.group(3) or heredoc.group(4)
                    pending.append((word, bool(heredoc.group(1))))
                i = heredoc.end() if heredoc else i + (3 if text.startswith("<<<", i) else 2)
                continue
            if ch == "\n":
                if pending:
                    i = _skip_heredocs(text, i + 1, pending)
                    pending = []
                    continue
                found = ch
            elif pending:
                pass  # the rest of a here-document's header line stays with its command
            elif ch == ";":
                found = ch
            elif ch == "|" and text[i - 1 : i] != ">":  # not the `>|` of a redirection
                found = "||" if text.startswith("||", i) else "|"
            elif ch == "&":
                if text.startswith("&&", i):
                    found = "&&"
                elif text[i - 1 : i] in " \t" and text[i + 1 : i + 2] in ("", " ", "\t", "\n"):
                    found = "&"  # a lone ` & `, not the `2>&1` or `&>` of a redirection
            if found:
                parts.append((operator, text[start:i]))
                operator, start = found, i + len(found)
                i = start
                continue
        i += 1
    parts.append((operator, text[start:]))
    return parts


def _is_lookup(segment: str) -> bool:
    """Does this simple command look a tool up instead of running it on data?

    Only its first line is read: a quoted program or a here-document body below it is not a
    command line, and a line that does not parse (a quote left open, a continuation) is left alone.
    """
    try:
        words = shlex.split(segment.split("\n", 1)[0])
    except ValueError:
        return False
    while words and (words[0] in SHELL_WORDS or ASSIGNMENT.match(words[0])):
        words = words[1:]
    if not words:
        return False
    tool = words[0].lstrip("({").rsplit("/", 1)[-1]
    if tool in LOOKUPS:
        return True
    if tool == "command":  # `command -v x` and `command -V x`
        return len(words) > 1 and re.fullmatch(r"-p?[vV]p?", words[1]) is not None
    return any(word in HELP_FLAGS for word in words[1:])


def strip_discovery(command: str) -> str:
    """`command` without its tool lookups, or '' when nothing else is left.

    A lookup is a segment (see `_segments`) that starts with `which`, `type`, `whereis`, `man` or
    `command -v`, or that has a `--help`, `--version`, `-help` or `-version` argument; what a pipe
    feeds from it (`| head`) goes with it. A command without one comes back unchanged, character
    for character.
    """
    parts = [(op, seg) for op, seg in _segments(command) if seg.strip()]
    kept: list[tuple[str, str]] = []
    gone = False
    for op, seg in parts:
        gone = _is_lookup(seg) or (gone and op == "|")
        if not gone:
            kept.append((op, seg))
    if len(kept) == len(parts):
        return command
    return "".join((op if k else "") + seg for k, (op, seg) in enumerate(kept)).strip()


def drop_discovery(commands: Iterable[str]) -> list[str]:
    """The commands with their tool lookups removed; a command that was only a lookup is gone.

    Run before every family check: an agent that types `which fastp` or `fastp --version` has not
    run fastp on the data, so it must not count as exposure to a trapped default.
    """
    return [kept for kept in map(strip_discovery, commands) if kept.strip()]


def _f1(commands: list[str], work: Path) -> bool:
    for command in commands:
        at = command.find("bcftools")
        if at >= 0 and BCFTOOLS_REGION_FILE.search(command[at:]):
            return True
    return False


def _f2(commands: list[str], work: Path) -> bool:
    try:
        regions = [line for line in (work / "regions.txt").read_text().split() if line]
    except OSError:
        regions = []
    for command in commands:
        if "faidx" not in command:
            continue
        if "regions.txt" in command or any(region in command for region in regions):
            return True
    return False


def _f3(commands: list[str], work: Path) -> bool:
    return any(REGION_TOOLS.search(c) and REGIONS_BED_ARG.search(c) for c in commands)


def _f4(commands: list[str], work: Path) -> bool:
    return any("ref.fa" in c for c in commands)


CHECKS = {"F1": _f1, "F2": _f2, "F3": _f3, "F4": _f4}


def _check_of(family: str) -> Any:
    if family in CHECKS:
        return CHECKS[family]
    from biowarden.bench.tasks.extra import load

    return load()[family].exposed


def exposure(family: str, commands: list[str], work: Path) -> dict[str, Any]:
    """Did these commands take the family's trapped route? Tool lookups do not count."""
    check = _check_of(family)
    return {"exposed": check(drop_discovery(commands), work), "n_commands": len(commands)}


def trial_exposure(family: str, messages: list[dict[str, Any]], work: Path) -> dict[str, Any]:
    """The exposure fields of one finished trial, from its transcript.

    exposed         the route was taken by a command that ran (one the layer refused is not)
    exposed_issued  the route was in a command the agent issued, refused or not
    n_commands      how many commands ran
    """
    ran, issued = executed_commands_of(messages), issued_commands_of(messages)
    found = exposure(family, ran, work)
    tried = found["exposed"] if issued == ran else exposure(family, issued, work)["exposed"]
    return {**found, "exposed_issued": bool(tried)}
