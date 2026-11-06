"""T2: pre-call validation blocks, proposes, and never rewrites on its own."""

from __future__ import annotations

from biowarden.contracts import Request, load_packaged
from biowarden.gateway import Gateway
from biowarden.validators import validate_request
from helpers import SWAPPED, TOOL_ORDER, entities, read_ledger


def test_t2_validator_blocks_permuted_request_with_structured_violation_and_unapplied_fix():
    contract = load_packaged("mockfold_tfoldlike")
    request = Request(entities=entities(SWAPPED))
    before = request.model_dump()

    verdict = validate_request(contract, request)

    assert not verdict.ok
    [violation] = verdict.violations
    assert (violation.slot, violation.code) == ("chain_order", "order_mismatch")
    assert violation.expected == list(TOOL_ORDER)
    assert violation.got == list(SWAPPED)
    assert verdict.proposed_fix is not None
    assert [e.role for e in verdict.proposed_fix.entities] == list(TOOL_ORDER)
    assert request.model_dump() == before  # the caller's request was not touched
    assert validate_request(contract, verdict.proposed_fix).ok


def test_t2_gateway_neither_applies_the_fix_nor_runs_the_tool_unless_asked(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    result = gateway.predict_complex(entities(SWAPPED), "tfoldlike")

    assert result.status == "blocked" and result.is_error
    assert result.proposed_fix is not None and not result.fix_applied
    assert result.entity_map is None and result.artifacts is None
    assert not (tmp_path / "work").exists()  # the tool never ran
    [record] = read_ledger(tmp_path / "ledger.jsonl")
    assert record["outcome"] == "blocked"
    assert record["fix"]["applied"] is False


def test_t2_applying_the_fix_is_opt_in_and_recorded_with_the_original_request(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")
    original = Request(entities=entities(SWAPPED))

    result = gateway.predict_complex(original.entities, "tfoldlike", apply_proposed_fix=True)

    assert result.status == "ok" and result.fix_applied
    assert [m.chain_id for m in result.entity_map] == list(TOOL_ORDER)
    [record] = read_ledger(tmp_path / "ledger.jsonl")
    assert record["fix"]["applied"] is True
    assert record["fix"]["original_request"] == original.to_tool_json()
    assert record["request"] == original.to_tool_json()
    assert record["used_request_hash"] != record["request_hash"]
    assert record["request_hash"] == original.content_hash()


def test_t2_request_hygiene_violations_are_structured():
    contract = load_packaged("mockfold_boltzlike")
    good = entities("BA")
    bad = [
        good[0].model_copy(update={"sequence": "ACDXZ"}),
        good[1].model_copy(update={"label": good[0].label}),
    ]

    verdict = validate_request(contract, Request(entities=bad))

    assert {v.code for v in verdict.violations} == {"invalid_sequence", "duplicate_label"}
    assert verdict.proposed_fix is None
