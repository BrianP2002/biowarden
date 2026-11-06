"""Contract schema v0: what a tool promises about its conventions, and how to check it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from biowarden.contracts.request import canonical_json, sha256_hex


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolIdentity(_Frozen):
    name: str
    backend: str
    version: str = Field(description="Exact version pin, checked against `<tool> --version`.")


class ChainOrderSlot(_Frozen):
    """How chains are ordered and named on the way into and out of the tool.

    ``tool_policy`` is what the tool does. ``consumer_policy`` is what downstream code is promised
    (output chain k holds request entity k). When they differ the pre-call validator demands that
    the request already follows the tool's order, and the oracles verify the promise on the output.
    """

    tool_policy: Literal["input_order", "declared_by_tool"]
    chain_ids: Literal["positional", "key"] = "positional"
    consumer_policy: Literal["input_order"] = "input_order"
    key: str | None = Field(default=None, description="Entity attribute the tool orders by.")
    order: list[str] = Field(default_factory=list, description="Fixed order of `key` values.")
    required: list[str] = Field(default_factory=list)
    exactly_one_of: list[list[str]] = Field(
        default_factory=list,
        description="Groups of `key` values; exactly one group must be present, and whole.",
    )

    @model_validator(mode="after")
    def _check(self) -> ChainOrderSlot:
        if self.tool_policy == "input_order":
            if self.key or self.order or self.required or self.exactly_one_of:
                raise ValueError("input_order takes no key/order/required/exactly_one_of")
            if self.chain_ids == "key":
                raise ValueError("chain_ids=key needs a declared key")
            return self
        if not self.key or not self.order:
            raise ValueError("declared_by_tool needs key and order")
        if len(set(self.order)) != len(self.order):
            raise ValueError("order has duplicates")
        known = set(self.order)
        members = {value for group in self.exactly_one_of for value in group}
        if not (set(self.required) | members) <= known:
            raise ValueError("required/exactly_one_of values must appear in order")
        return self


class ScaleSpec(_Frozen):
    min: float
    max: float


class RecordCountSlot(_Frozen):
    chains: Literal["equals_entities"] = "equals_entities"


class Slots(_Frozen):
    chain_order: ChainOrderSlot | None = None
    metric_scale: dict[str, ScaleSpec] = Field(default_factory=dict)
    record_count: RecordCountSlot | None = None


class Invariant(_Frozen):
    id: str
    slot: Literal["chain_order", "metric_scale", "record_count"]
    oracle: str = Field(description="Name in the oracle registry.")


class Probe(_Frozen):
    id: str
    slot: Literal["chain_order"]
    kind: Literal["permute_entities"]


class ToolContract(_Frozen):
    schema_version: Literal["0"] = "0"
    tool: ToolIdentity
    slots: Slots
    invariants: list[Invariant] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)

    def content_hash(self) -> str:
        """Stable across YAML key order and formatting."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))
