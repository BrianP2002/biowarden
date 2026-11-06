"""The call path every request takes: pre-call validation, the tool, post-call oracles, ledger."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from biowarden.contracts.loader import PACKAGED, load_packaged
from biowarden.contracts.request import Entity, Request
from biowarden.contracts.schema import ToolContract
from biowarden.gateway.runner import MockfoldRunner
from biowarden.ledger import Ledger
from biowarden.oracles import (
    OracleResult,
    ProbeResult,
    known_oracles,
    run_invariants,
    run_permute_entities,
)
from biowarden.oracles.structure import parse_output
from biowarden.validators.precall import Verdict, Violation, validate_request

Status = Literal["ok", "blocked", "flagged", "tool_error"]


class EntityChain(BaseModel):
    position: int
    label: str
    role: str
    chain_id: str


class GatewayResult(BaseModel):
    status: Status
    call_id: str
    backend: str
    tool_version: str | None = None
    contract_hash: str | None = None
    violations: list[Violation] = Field(default_factory=list)
    proposed_fix: Request | None = None
    fix_applied: bool = False
    oracle_results: list[OracleResult] = Field(default_factory=list)
    entity_map: list[EntityChain] | None = None
    artifacts: dict[str, str] | None = None
    metrics: dict[str, Any] | None = None
    text: str

    @property
    def is_error(self) -> bool:
        return self.status != "ok"


@dataclass
class _Call:
    call_id: str
    backend: str
    contract: ToolContract | None
    request: Request
    used: Request
    version: str | None = None
    fix: dict[str, Any] | None = None
    results: list[OracleResult] = field(default_factory=list)
    exit_code: int | None = None


def _blocked_text(backend: str, violations: Sequence[Violation], fix: Request | None) -> str:
    lines = [f"BLOCKED before running {backend}: {len(violations)} violation(s)."]
    lines += [f"- [{v.slot}/{v.code}] {v.message}" for v in violations[:4]]
    if fix is not None:
        order = ",".join(e.label for e in fix.entities)
        lines.append(
            f"Proposed fix (NOT applied): list the entities as {order}. Call again with "
            "apply_proposed_fix=true to accept it; the original request is kept in the ledger."
        )
    return "\n".join(lines)


def _flagged_text(backend: str, failed: Sequence[OracleResult]) -> str:
    lines = [f"FLAGGED: {backend} exited 0 but {len(failed)} invariant(s) failed. Do not use it."]
    for result in failed:
        lines += [
            f"- {result.invariant_id} {result.oracle}: {f.message}" for f in result.findings[:2]
        ]
    return "\n".join(lines)


class Gateway:
    """Wraps the tools. ``precall=False`` is an ablation switch for benchmark arms."""

    def __init__(
        self,
        work_dir: str | Path,
        ledger_path: str | Path,
        *,
        contracts: Mapping[str, ToolContract] | None = None,
        faults: Mapping[str, Sequence[str]] | None = None,
        precall: bool = True,
        runner_factory: Callable[[str], MockfoldRunner] | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.ledger = Ledger(ledger_path)
        self.contracts: dict[str, ToolContract] = dict(
            contracts or {c.tool.backend: c for c in map(load_packaged, PACKAGED)}
        )
        self.faults = {k: tuple(v) for k, v in (faults or {}).items()}
        self.precall = precall
        self._runner_factory = runner_factory or (
            lambda backend: MockfoldRunner(backend, self.faults.get(backend, ()))
        )
        self._runners: dict[str, MockfoldRunner] = {}
        for contract in self.contracts.values():
            unknown = {i.oracle for i in contract.invariants} - known_oracles()
            if unknown:
                raise ValueError(f"{contract.tool.backend}: unknown oracle(s) {sorted(unknown)}")

    def _runner(self, backend: str) -> MockfoldRunner:
        if backend not in self._runners:
            self._runners[backend] = self._runner_factory(backend)
        return self._runners[backend]

    def _log(self, contract: ToolContract | None, **fields: Any) -> None:
        self.ledger.append(
            {
                "tool": contract.tool.name if contract else None,
                "contract_hash": contract.content_hash() if contract else None,
                "config": {"precall": self.precall, "faults": self.faults},
                **fields,
            }
        )

    def _finish(
        self,
        call: _Call,
        status: Status,
        text: str,
        *,
        violations: Sequence[Violation] = (),
        proposed_fix: Request | None = None,
        **fields: Any,
    ) -> GatewayResult:
        self._log(
            call.contract,
            kind="call",
            call_id=call.call_id,
            backend=call.backend,
            tool_version=call.version,
            request_hash=call.request.content_hash(),
            request=call.request.to_tool_json(),
            used_request_hash=call.used.content_hash(),
            verdict={
                "ok": not violations,
                "violations": [v.model_dump(mode="json") for v in violations],
                "proposed_fix": proposed_fix.to_tool_json() if proposed_fix else None,
            },
            fix=call.fix,
            oracle_results=[r.model_dump(mode="json") for r in call.results],
            exit_code=call.exit_code,
            outcome=status,
        )
        return GatewayResult(
            status=status,
            call_id=call.call_id,
            backend=call.backend,
            tool_version=call.version,
            contract_hash=call.contract.content_hash() if call.contract else None,
            violations=list(violations),
            proposed_fix=proposed_fix,
            fix_applied=bool(call.fix and call.fix.get("applied")),
            oracle_results=call.results,
            text=text,
            **fields,
        )

    def _block(
        self, call: _Call, violations: Sequence[Violation], proposed: Request | None = None
    ) -> GatewayResult:
        text = _blocked_text(call.backend, violations, proposed)
        return self._finish(call, "blocked", text, violations=violations, proposed_fix=proposed)

    def predict_complex(
        self,
        entities: Sequence[Entity | Mapping[str, Any]],
        backend: str,
        apply_proposed_fix: bool = False,
    ) -> GatewayResult:
        request = Request(entities=[Entity.model_validate(e) for e in entities])
        call = _Call(uuid.uuid4().hex, backend, self.contracts.get(backend), request, request)
        contract = call.contract
        if contract is None:
            known = sorted(self.contracts)
            return self._block(
                call,
                [
                    Violation(
                        slot="backend",
                        code="unknown_backend",
                        expected=known,
                        got=backend,
                        message=f"no contract for backend {backend!r}; known: {known}",
                    )
                ],
            )

        runner = self._runner(backend)
        call.version = runner.version()
        if call.version != contract.tool.version:
            return self._block(
                call,
                [
                    Violation(
                        slot="tool_identity",
                        code="version_mismatch",
                        expected=contract.tool.version,
                        got=call.version,
                        message=f"{contract.tool.name} reports version {call.version}; "
                        f"the contract pins {contract.tool.version}",
                    )
                ],
            )

        verdict = validate_request(contract, request) if self.precall else Verdict(ok=True)
        if not verdict.ok:
            proposed = verdict.proposed_fix
            if not (apply_proposed_fix and proposed is not None):
                if proposed is not None:
                    call.fix = {"applied": False, "proposed_request_hash": proposed.content_hash()}
                return self._block(call, verdict.violations, proposed)
            call.used = proposed
            call.fix = {
                "applied": True,
                "original_request": request.to_tool_json(),
                "original_request_hash": request.content_hash(),
                "applied_request_hash": proposed.content_hash(),
            }

        run = runner.run(call.used, self.work_dir / call.call_id)
        call.exit_code = run.exit_code
        if run.exit_code != 0 or run.pdb is None:
            tail = run.stderr.strip()[-300:]
            return self._finish(
                call, "tool_error", f"TOOL ERROR: {backend} exited {run.exit_code}: {tail}"
            )

        call.results = run_invariants(contract, call.used, run.pdb, run.confidence)
        artifacts = {"pdb": str(run.pdb)}
        if run.confidence:
            artifacts["confidence"] = str(run.confidence)
        failed = [r for r in call.results if not r.ok]
        if failed:
            return self._finish(
                call, "flagged", _flagged_text(backend, failed), artifacts=artifacts
            )

        output = parse_output(run.pdb, run.confidence)
        entity_map = [
            EntityChain(position=k, label=e.label, role=e.role, chain_id=output.chains[k].chain_id)
            for k, e in enumerate(call.used.entities)
        ]
        scale = contract.slots.metric_scale.get("plddt")
        mean = (output.confidence or {}).get("plddt_mean")
        text = (
            f"OK {backend} ({contract.tool.name} {call.version}): {len(entity_map)} chains "
            f"[{','.join(m.chain_id for m in entity_map)}] in request order; "
            f"{len(call.results)}/{len(call.results)} invariants passed"
        )
        if scale is not None:
            text += f"; plddt_mean {mean} on a {scale.min:g}-{scale.max:g} scale"
        if call.fix:
            text += "; the proposed reordering was applied at your request"
        return self._finish(
            call,
            "ok",
            text + ".",
            entity_map=entity_map,
            artifacts=artifacts,
            metrics={"plddt_mean": mean, "plddt_scale": [scale.min, scale.max] if scale else None},
        )

    def run_probes(
        self, backend: str, base_entities: Sequence[Entity | Mapping[str, Any]]
    ) -> list[ProbeResult]:
        """Run the contract's probes against the live tool and log each outcome."""
        contract = self.contracts[backend]
        runner = self._runner(backend)
        base = Request(entities=[Entity.model_validate(e) for e in base_entities])
        out = []
        for probe in contract.probes:
            scratch = self.work_dir / f"probe-{probe.id}-{uuid.uuid4().hex[:8]}"
            result = run_permute_entities(contract, probe, runner, base, scratch)
            self._log(
                contract,
                kind="probe",
                backend=backend,
                tool_version=runner.version(),
                probe_id=probe.id,
                outcome=result.status,
                result=result.model_dump(mode="json"),
            )
            out.append(result)
        return out
