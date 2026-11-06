"""The Task record and the vocabularies shared by every task family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TWINS = ("clean", "faulted")
TIERS = ("explicit", "implicit")
ROUTES = ("free", "forced")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    slot: str
    twin: str
    seed: int
    prompt: str
    truth: dict[str, Any]  # JSON-serialisable; kept out of the workspace
    evidence: tuple[str, ...]  # where the agent can see the trap before first use
    tier: str = "explicit"
    route: str = "free"
