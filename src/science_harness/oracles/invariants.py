"""Post-call oracles: check what the tool actually wrote against the contract."""

from __future__ import annotations

import math
import string
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from science_harness.contracts.request import Request, sha256_hex
from science_harness.contracts.schema import ToolContract
from science_harness.oracles.structure import OutputError, ParsedOutput, parse_output


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class OracleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    invariant_id: str
    oracle: str
    slot: str
    ok: bool
    findings: list[Finding] = Field(default_factory=list)


@dataclass(frozen=True)
class OracleContext:
    contract: ToolContract
    request: Request
    output: ParsedOutput


OracleFn = Callable[[OracleContext], list[Finding]]
_REGISTRY: dict[str, OracleFn] = {}


def oracle(name: str) -> Callable[[OracleFn], OracleFn]:
    def register(fn: OracleFn) -> OracleFn:
        _REGISTRY[name] = fn
        return fn

    return register


def known_oracles() -> frozenset[str]:
    return frozenset(_REGISTRY)


def short_sha(sequence: str) -> str:
    return sha256_hex(sequence)[:12]


@oracle("chain_sequence_roundtrip")
def chain_sequence_roundtrip(ctx: OracleContext) -> list[Finding]:
    """Output chain k must hold request entity k, under the ID scheme the contract declares."""
    slot = ctx.contract.slots.chain_order
    if slot is None:
        return []
    entities, chains = ctx.request.entities, ctx.output.chains
    label_by_sha = {short_sha(e.sequence): e.label for e in entities}
    findings: list[Finding] = []

    for k, entity in enumerate(entities):
        if k >= len(chains):
            findings.append(
                Finding(
                    code="missing_chain",
                    message=f"no output chain at position {k} for entity {entity.label!r}",
                    evidence={"position": k, "entity": entity.label},
                )
            )
            continue
        chain = chains[k]
        if slot.chain_ids == "key":
            expected_id = str(getattr(entity, slot.key or ""))
        else:
            expected_id = string.ascii_uppercase[k % 26]
        if chain.chain_id != expected_id:
            findings.append(
                Finding(
                    code="chain_id_mismatch",
                    message=(
                        f"output chain {k} is named {chain.chain_id!r}; the request implies "
                        f"{expected_id!r} for entity {entity.label!r}"
                    ),
                    evidence={"position": k, "expected": expected_id, "got": chain.chain_id},
                )
            )
        if chain.sequence != entity.sequence:
            actual = short_sha(chain.sequence)
            holder = f" (entity {label_by_sha[actual]!r})" if actual in label_by_sha else ""
            findings.append(
                Finding(
                    code="sequence_mismatch",
                    message=(
                        f"output chain {k} ({chain.chain_id}) holds sequence {actual}{holder}, "
                        f"but the request has {entity.label!r} ({short_sha(entity.sequence)}) there"
                    ),
                    evidence={
                        "position": k,
                        "chain_id": chain.chain_id,
                        "expected_entity": entity.label,
                        "expected_sha256": short_sha(entity.sequence),
                        "actual_sha256": actual,
                        "actual_matches_entity": label_by_sha.get(actual),
                    },
                )
            )
    for k in range(len(entities), len(chains)):
        findings.append(
            Finding(
                code="extra_chain",
                message=f"output chain {k} ({chains[k].chain_id}) has no request entity",
                evidence={"position": k, "chain_id": chains[k].chain_id},
            )
        )
    return findings


@oracle("record_count")
def record_count(ctx: OracleContext) -> list[Finding]:
    findings: list[Finding] = []
    n_entities, n_chains = len(ctx.request.entities), len(ctx.output.chains)
    if ctx.contract.slots.record_count is not None and n_chains != n_entities:
        findings.append(
            Finding(
                code="chain_count_mismatch",
                message=f"request has {n_entities} entities but the output has {n_chains} chains",
                evidence={"expected": n_entities, "got": n_chains},
            )
        )
    slot = ctx.contract.slots.chain_order
    if slot is not None and slot.chain_ids == "key" and slot.required:
        present = {c.chain_id for c in ctx.output.chains}
        missing = [k for k in slot.required if k not in present]
        if missing:
            findings.append(
                Finding(
                    code="required_chain_missing",
                    message=f"output has no chain for required {slot.key} value(s) {missing}",
                    evidence={"missing": missing, "present": sorted(present)},
                )
            )
    return findings


@oracle("finite_coordinates_and_scale")
def finite_coordinates_and_scale(ctx: OracleContext) -> list[Finding]:
    out = ctx.output
    if out.n_atoms == 0:
        return [Finding(code="empty_structure", message="output structure has no atoms")]
    findings: list[Finding] = []
    if not out.coords_finite:
        findings.append(
            Finding(code="non_finite_coordinates", message="output has NaN or infinite coordinates")
        )
    spec = ctx.contract.slots.metric_scale.get("plddt")
    if spec is None:
        return findings
    eps = 1e-6
    observed: dict[str, float] = {}
    if out.plddt:
        observed["structure_min"], observed["structure_max"] = min(out.plddt), max(out.plddt)
    mean = (out.confidence or {}).get("plddt_mean")
    if isinstance(mean, int | float) and math.isfinite(mean):
        observed["confidence_mean"] = float(mean)
    outside = {k: v for k, v in observed.items() if not spec.min - eps <= v <= spec.max + eps}
    if outside:
        findings.append(
            Finding(
                code="metric_out_of_scale",
                message=f"pLDDT values {outside} fall outside the declared {spec.min}-{spec.max}",
                evidence={
                    "metric": "plddt",
                    "declared": [spec.min, spec.max],
                    "observed": observed,
                },
            )
        )
    peak = observed.get("structure_max", observed.get("confidence_mean"))
    if spec.max > 1.0 and peak is not None and 0.0 < peak <= 1.0 + eps:
        findings.append(
            Finding(
                code="metric_scale_suspect",
                message=(
                    f"all pLDDT values are <= 1.0 under a declared {spec.min}-{spec.max} scale; "
                    "a consumer would read this file as near-zero confidence"
                ),
                evidence={
                    "metric": "plddt",
                    "declared": [spec.min, spec.max],
                    "observed": observed,
                },
            )
        )
    return findings


def run_invariants(
    contract: ToolContract,
    request: Request,
    pdb_path: Path,
    confidence_path: Path | None = None,
) -> list[OracleResult]:
    """Run every invariant the contract lists; unreadable output fails them all."""
    try:
        ctx = OracleContext(contract, request, parse_output(pdb_path, confidence_path))
    except OutputError as exc:
        broken = Finding(code="unreadable_output", message=str(exc))
        return [
            OracleResult(
                invariant_id=inv.id, oracle=inv.oracle, slot=inv.slot, ok=False, findings=[broken]
            )
            for inv in contract.invariants
        ]
    results = []
    for inv in contract.invariants:
        findings = _REGISTRY[inv.oracle](ctx)
        results.append(
            OracleResult(
                invariant_id=inv.id,
                oracle=inv.oracle,
                slot=inv.slot,
                ok=not findings,
                findings=findings,
            )
        )
    return results
