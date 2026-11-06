"""Call payloads shared by the validators, oracles, ledger and gateway."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Entity(BaseModel):
    """One molecular entity of a complex, in the caller's words."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(default="", description="Tool-specific role code; ignored by some tools.")
    label: str = Field(description="Caller's name for the entity; unique within a request.")
    sequence: str = Field(description="One-letter amino-acid sequence.")


class Request(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[Entity]

    def content_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def to_tool_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
