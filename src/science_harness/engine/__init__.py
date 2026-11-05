"""Rule engine v1: check a tool call, or the data it touches, against a tool spec before it runs.

A spec lists rules. A *call* rule looks at the arguments of one tool (F1, F7, F10 style); a *fact*
rule looks at files named anywhere in a command and fires only when the data is in a risky state
(F13, F15 style). Every rule returns a three-valued verdict; the engine turns verdicts into
allow / warn / block according to the mode and the promotion table, and never rewrites a call.
"""

from science_harness.engine.call import Call
from science_harness.engine.checks import (
    CheckResult,
    Ctx,
    Match,
    checker,
    known_checkers,
    suggester,
)
from science_harness.engine.decide import Decision, Engine, Promotion, RuleOutcome
from science_harness.engine.spec import Applies, Declare, Rule, Spec, load_spec

__all__ = [
    "Applies",
    "Call",
    "CheckResult",
    "Ctx",
    "Declare",
    "Decision",
    "Engine",
    "Match",
    "Promotion",
    "Rule",
    "RuleOutcome",
    "Spec",
    "checker",
    "known_checkers",
    "suggester",
    "load_spec",
]
