"""Metamorphic probes: run the tool on related inputs and check the contract tells the truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from biowarden.contracts.request import Request
from biowarden.contracts.schema import Probe, ToolContract
from biowarden.oracles.invariants import Finding, short_sha
from biowarden.oracles.structure import OutputError, parse_output


class RunResult(Protocol):
    exit_code: int
    stderr: str
    pdb: Path | None
    confidence: Path | None


class Runner(Protocol):
    def run(self, request: Request, out_dir: Path) -> RunResult: ...


class ProbeResult(BaseModel):
    """``finding`` is a documented property of the tool; ``fail`` means the contract is wrong."""

    model_config = ConfigDict(frozen=True)

    probe_id: str
    kind: str
    slot: str
    status: Literal["pass", "finding", "fail"]
    declared_policy: str
    observed_policy: str
    findings: list[Finding] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


def run_permute_entities(
    contract: ToolContract,
    probe: Probe,
    runner: Runner,
    base_request: Request,
    work_dir: Path,
) -> ProbeResult:
    """M1: reverse the entity order and see whether output chains follow the request."""
    slot = contract.slots.chain_order
    if slot is None:
        raise ValueError("permute_entities needs a chain_order slot")
    declared = slot.tool_policy

    def result(
        status: Literal["pass", "finding", "fail"],
        observed: str,
        findings: list[Finding],
        evidence: dict[str, Any] | None = None,
    ) -> ProbeResult:
        return ProbeResult(
            probe_id=probe.id,
            kind=probe.kind,
            slot=probe.slot,
            status=status,
            declared_policy=declared,
            observed_policy=observed,
            findings=findings,
            evidence=evidence or {},
        )

    entities = base_request.entities
    sequences = [e.sequence for e in entities]
    if len(entities) < 2 or len(set(sequences)) != len(sequences):
        return result(
            "fail",
            "unknown",
            [
                Finding(
                    code="probe_inconclusive", message="needs >=2 entities with distinct sequences"
                )
            ],
        )

    permuted = Request(entities=list(reversed(entities)))
    runs = [
        runner.run(req, work_dir / name)
        for name, req in (("base", base_request), ("permuted", permuted))
    ]
    for run in runs:
        if run.exit_code != 0 or run.pdb is None:
            return result(
                "fail",
                "unknown",
                [
                    Finding(
                        code="tool_error",
                        message=f"tool exited {run.exit_code}: {run.stderr.strip()}",
                    )
                ],
            )
    try:
        base_out = parse_output(runs[0].pdb, runs[0].confidence)  # type: ignore[arg-type]
        perm_out = parse_output(runs[1].pdb, runs[1].confidence)  # type: ignore[arg-type]
    except OutputError as exc:
        return result("fail", "unknown", [Finding(code="unreadable_output", message=str(exc))])

    base_seqs = [c.sequence for c in base_out.chains]
    perm_seqs = [c.sequence for c in perm_out.chains]
    follows = base_seqs == sequences and perm_seqs == [e.sequence for e in permuted.entities]
    if follows:
        observed = "input_order"
    elif perm_seqs == base_seqs:
        observed = "declared_by_tool"
    else:
        observed = "inconsistent"
    evidence = {
        "base_chain_ids": [c.chain_id for c in base_out.chains],
        "permuted_chain_ids": [c.chain_id for c in perm_out.chains],
        "base_sequence_sha256": [short_sha(s) for s in base_seqs],
        "permuted_sequence_sha256": [short_sha(s) for s in perm_seqs],
    }

    if observed != declared:
        return result(
            "fail",
            observed,
            [
                Finding(
                    code="contract_mismatch",
                    message=f"contract declares {declared} but the tool behaves as {observed}",
                )
            ],
            evidence,
        )
    if observed == "declared_by_tool":
        return result(
            "finding",
            observed,
            [
                Finding(
                    code="order_insensitive",
                    message=(
                        "tool ignores request order; positional pairing of output chains with "
                        "request entities is only safe when the request follows the declared order"
                    ),
                )
            ],
            evidence,
        )
    return result("pass", observed, [], evidence)
