"""Text the agent reads when a call is blocked or flagged. The fix comes first."""

from __future__ import annotations

from collections.abc import Collection

from biowarden.engine.decide import Decision, RuleOutcome

GENERIC_BLOCK = (
    "BLOCKED, the command did not run. A convention in this command may be misread. Check the "
    "input and output formats and what the values mean biologically, then run the command again "
    "if you are sure."
)
GENERIC_WARNING = (
    "WARNING: a convention in this command may be misread. Check the input and output formats "
    "and what the values mean biologically."
)
STOP_TEXT = (
    "STOP: this exact call was already blocked {n} times. Do not retry it. Report the problem to "
    "the user with status needs_input and explain which convention could not be confirmed."
)
# the bare contract has no status and no flags: a null answer is all the agent can hand in
STOP_TEXT_BARE = (
    "STOP: this exact call was already blocked {n} times. Do not retry it. Stop here and write "
    'result.json with {{"answer": null}}.'
)
# a checker that crashed leaves no fix of its own; the agent can only change the command
CRASHED_FIX = (
    "the check of this rule failed on this command, which says nothing about the command itself. "
    "Run the step another way (one command at a time, or with other tools or options)."
)


def _headline(outcome: RuleOutcome) -> str:
    return f"{outcome.rule.id} [{outcome.rule.slot}/{outcome.rule.tier}]: {outcome.result.message}"


def _shown(outcome: RuleOutcome) -> bool:
    """A fault of the layer is for the ledger: the agent is told nothing about it."""
    return outcome.result.code != "layer_error"


def _fix(outcome: RuleOutcome) -> str:
    if outcome.result.fix:
        return " ".join(outcome.result.fix)
    return CRASHED_FIX if outcome.result.code == "checker_crashed" else ""


def render_block(
    decision: Decision, retry_cap: int = 2, style: str = "full", contract: str = "full"
) -> str:
    """The block message; `contract` is the result.json the agent was asked for (it words STOP)."""
    if style == "generic":  # detection only: no rule, no tool behaviour, no fix
        return GENERIC_BLOCK
    blocking = [o for o in decision.outcomes if o.effective == "block"]
    first = blocking[0]
    lines: list[str] = []
    if decision.stop:
        lines.append((STOP_TEXT_BARE if contract == "bare" else STOP_TEXT).format(n=retry_cap))
    lines.append(f"BLOCKED, the command did not run. {_headline(first)}")
    fix = _fix(first)
    if fix:
        lines.insert(1 if decision.stop else 0, f"FIX: {fix}")
    for extra in blocking[1:3]:
        lines.append(f"Also: {_headline(extra)}")
    return "\n".join(lines)


def render_warning(decision: Decision, skip: Collection[str] = (), style: str = "full") -> str:
    """Warnings for the call; `skip` lists rule ids already reported to the agent for this call."""
    if style == "generic":
        fired = [
            o
            for o in decision.outcomes
            if o.effective == "warn" and o.rule.id not in skip and _shown(o)
        ]
        return GENERIC_WARNING if fired else ""
    lines = []
    for outcome in decision.outcomes:
        if outcome.effective != "warn" or outcome.rule.id in skip or not _shown(outcome):
            continue
        prefix = "WARNING" if outcome.result.verdict == "fail" else "UNVERIFIED"
        fix = _fix(outcome)
        lines.append(f"{prefix} {_headline(outcome)}" + (f" FIX: {fix}" if fix else ""))
    return "\n".join(lines)
