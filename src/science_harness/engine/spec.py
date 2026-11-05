"""Tool spec v1: the rules a tool (or the data it reads) is checked against."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_harness.contracts.request import canonical_json, sha256_hex

SPEC_DIR = Path(__file__).parent / "specs"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Declare(_Frozen):
    """The convention the agent must state ("required_explicit"): no default, omission is an error.

    `required` applies to call rules. A fact rule inspects the data first and asks for the
    declaration only when the data is in a risky state, so its checker decides when it is needed.
    """

    key: str
    values: list[str] = Field(min_length=1)
    required: bool = True
    description: str = ""


class Applies(_Frozen):
    """When a rule is relevant to a call.

    Call rules: `tools` (basenames), optional `subcommands` (glob patterns, so `+*` matches every
    bcftools plugin), optional `options_any` (at least one present). Fact rules: `file_patterns`
    are regexes searched in every argument, including inside code strings, and the rule applies
    when a matched name is an existing file in the working directory; `tools` limits the
    interpreters it watches (empty means any tool).
    """

    tools: list[str] = Field(default_factory=list)
    subcommands: list[str] = Field(default_factory=list)
    options_any: list[str] = Field(default_factory=list)
    file_patterns: list[str] = Field(default_factory=list)


class Rule(_Frozen):
    id: str
    slot: str = Field(description="Convention slot, e.g. index_base, interval_semantics.")
    kind: Literal["call", "fact"]
    tier: Literal["T0", "T1", "T2"]
    incidents: list[str] = Field(default_factory=list)
    applies: Applies
    declare: Declare | None = None
    check: str = Field(description="Name in the checker registry.")
    params: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(description="One sentence: what goes wrong without this rule.")

    @model_validator(mode="after")
    def _shape(self) -> Rule:
        if self.kind == "call" and not self.applies.tools:
            raise ValueError("a call rule needs applies.tools")
        if self.kind == "fact" and not self.applies.file_patterns:
            raise ValueError("a fact rule needs applies.file_patterns")
        if self.kind == "call" and self.applies.file_patterns:
            raise ValueError("file_patterns belong to fact rules")
        if self.tier == "T0" and self.kind == "fact":
            raise ValueError("a fact rule reads files, so it cannot be tier T0")
        if self.tier == "T0" and self.declare is None:
            raise ValueError("a T0 rule is decided from the call alone, so it needs a declaration")
        return self


class Spec(_Frozen):
    spec_version: Literal["1"] = "1"
    name: str
    version_note: str = Field(default="", description="Tool versions the rules were checked on.")
    rules: list[Rule]

    @model_validator(mode="after")
    def _unique(self) -> Spec:
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate rule id")
        return self

    def content_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def declaration_schema(self) -> dict[str, Any]:
        """JSON Schema of the `declared` object (what an MCP front end advertises)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for rule in self.rules:
            if rule.declare is None:
                continue
            properties[rule.declare.key] = {
                "enum": rule.declare.values,
                "description": rule.declare.description or rule.message,
            }
            if rule.kind == "call" and rule.declare.required:
                required.append(rule.declare.key)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def load_spec(path: str | Path) -> Spec:
    return Spec.model_validate(yaml.safe_load(Path(path).read_text()))


def load_packaged(name: str) -> Spec:
    return load_spec(SPEC_DIR / f"{name}.yaml")


def packaged_specs() -> list[Spec]:
    return [load_spec(p) for p in sorted(SPEC_DIR.glob("*.yaml"))]


def load_spec_dir(path: str | Path) -> list[Spec]:
    """Every `*.yaml` / `*.yml` spec in a directory (not recursive), in name order."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"spec directory not found: {directory}")
    files = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    return [load_spec(p) for p in files if p.name != "promotion.json"]


def collect_specs(
    dirs: Sequence[str | Path] = (),
    packaged: bool = True,
    env: Mapping[str, str] | None = None,
) -> list[Spec]:
    """Shipped specs (optional) plus those in `dirs` and in $TBM_SPECS (os.pathsep-separated)."""
    from_env = [
        p
        for p in (env if env is not None else os.environ).get("TBM_SPECS", "").split(os.pathsep)
        if p
    ]
    specs = packaged_specs() if packaged else []
    for directory in [*dirs, *from_env]:
        specs += load_spec_dir(directory)
    return specs
