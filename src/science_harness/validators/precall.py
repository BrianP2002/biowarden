"""Pre-call validation: check a request against a contract before the tool runs.

Normalisation never happens here. When a request can be repaired, the verdict carries a
``proposed_fix``; whoever applies it must do so explicitly and record it (the gateway does).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from science_harness.contracts.request import Request
from science_harness.contracts.schema import ChainOrderSlot, ToolContract

_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot: str
    code: str
    expected: Any
    got: Any
    message: str


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    proposed_fix: Request | None = None


def _check_entities(request: Request) -> list[Violation]:
    found: list[Violation] = []
    if not request.entities:
        return [
            Violation(
                slot="entities",
                code="no_entities",
                expected=">=1",
                got=0,
                message="request has no entities",
            )
        ]
    for label, count in Counter(e.label for e in request.entities).items():
        if count > 1:
            found.append(
                Violation(
                    slot="entities",
                    code="duplicate_label",
                    expected="unique labels",
                    got=label,
                    message=f"label {label!r} is used {count} times",
                )
            )
    for entity in request.entities:
        bad = sorted(set(entity.sequence) - _RESIDUES) if entity.sequence else ["<empty>"]
        if bad:
            found.append(
                Violation(
                    slot="entities",
                    code="invalid_sequence",
                    expected="ACDEFGHIKLMNPQRSTVWY",
                    got=bad,
                    message=f"entity {entity.label!r} has invalid residues {bad}",
                )
            )
    return found


def _check_declared_by_tool(
    contract: ToolContract, slot: ChainOrderSlot, request: Request
) -> tuple[list[Violation], Violation | None]:
    assert slot.key is not None
    tool = contract.tool
    keys = [getattr(e, slot.key) for e in request.entities]
    found: list[Violation] = []

    unknown = sorted({k for k in keys if k not in slot.order})
    if unknown:
        found.append(
            Violation(
                slot="chain_order",
                code="unknown_key",
                expected=slot.order,
                got=unknown,
                message=f"{slot.key} value(s) {unknown} are not valid for {tool.backend}",
            )
        )
    repeated = sorted(k for k, n in Counter(keys).items() if n > 1)
    if repeated:
        found.append(
            Violation(
                slot="chain_order",
                code="duplicate_key",
                expected="each once",
                got=repeated,
                message=f"{slot.key} value(s) {repeated} are given more than once",
            )
        )
    missing = [k for k in slot.required if k not in keys]
    if missing:
        found.append(
            Violation(
                slot="chain_order",
                code="required_missing",
                expected=slot.required,
                got=keys,
                message=f"required {slot.key} value(s) {missing} are missing",
            )
        )
    if slot.exactly_one_of:
        present = [g for g in slot.exactly_one_of if any(k in keys for k in g)]
        if len(present) != 1:
            found.append(
                Violation(
                    slot="chain_order",
                    code="group_conflict",
                    expected=f"exactly one of {slot.exactly_one_of}",
                    got=present,
                    message=f"{len(present)} of the groups {slot.exactly_one_of} are present",
                )
            )
        for group in present:
            if not all(k in keys for k in group):
                found.append(
                    Violation(
                        slot="chain_order",
                        code="group_incomplete",
                        expected=group,
                        got=keys,
                        message=f"group {group} must be given whole",
                    )
                )
    if found:
        return found, None

    expected = sorted(keys, key=slot.order.index)
    if keys == expected:
        return found, None
    order = Violation(
        slot="chain_order",
        code="order_mismatch",
        expected=expected,
        got=keys,
        message=(
            f"{tool.backend} ({tool.name} {tool.version}) writes chains in the fixed {slot.key} "
            f"order {','.join(expected)} whatever the request order, but consumers pair output "
            f"chain k with request entity k; the request lists {','.join(keys)}"
        ),
    )
    return found, order


def validate_request(contract: ToolContract, request: Request) -> Verdict:
    violations = _check_entities(request)
    slot = contract.slots.chain_order
    order_violation = None
    if slot is not None and slot.tool_policy == "declared_by_tool" and not violations:
        structural, order_violation = _check_declared_by_tool(contract, slot, request)
        violations.extend(structural)
        if order_violation is not None:
            violations.append(order_violation)

    fix = None
    if order_violation is not None and len(violations) == 1:
        assert slot is not None and slot.key is not None
        key = slot.key
        candidate = Request(
            entities=sorted(request.entities, key=lambda e: slot.order.index(getattr(e, key)))
        )
        if validate_request(contract, candidate).ok:
            fix = candidate
    return Verdict(ok=not violations, violations=violations, proposed_fix=fix)
