"""T6: metamorphic probe M1 checks that the contract describes the tool's real behaviour."""

from __future__ import annotations

from biowarden.gateway import Gateway
from helpers import TOOL_ORDER, entities, read_ledger, variant


def test_t6_m1_passes_for_boltzlike(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    [probe] = gateway.run_probes("boltzlike", entities(TOOL_ORDER))

    assert probe.status == "pass"
    assert probe.declared_policy == probe.observed_policy == "input_order"
    assert probe.findings == []


def test_t6_m1_flags_the_order_insensitivity_of_tfoldlike_as_the_documented_finding(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    [probe] = gateway.run_probes("tfoldlike", entities(TOOL_ORDER))

    assert probe.status == "finding"
    assert probe.declared_policy == probe.observed_policy == "declared_by_tool"
    assert [f.code for f in probe.findings] == ["order_insensitive"]
    assert (
        probe.evidence["base_chain_ids"] == probe.evidence["permuted_chain_ids"] == list(TOOL_ORDER)
    )
    assert probe.evidence["base_sequence_sha256"] == probe.evidence["permuted_sequence_sha256"]


def test_t6_m1_fails_when_the_contract_misdescribes_the_tool(tmp_path):
    def claim_input_order(data):
        data["slots"]["chain_order"] = {"tool_policy": "input_order", "chain_ids": "positional"}

    liar = variant("tfoldlike", claim_input_order)
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl", contracts={"tfoldlike": liar})

    [probe] = gateway.run_probes("tfoldlike", entities(TOOL_ORDER))

    assert probe.status == "fail"
    assert probe.declared_policy == "input_order" and probe.observed_policy == "declared_by_tool"
    assert [f.code for f in probe.findings] == ["contract_mismatch"]


def test_t6_probe_outcomes_are_written_to_the_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    gateway = Gateway(tmp_path / "work", path)

    gateway.run_probes("tfoldlike", entities(TOOL_ORDER))

    [record] = read_ledger(path)
    assert record["kind"] == "probe" and record["probe_id"] == "M1"
    assert record["outcome"] == "finding"
    assert record["result"]["observed_policy"] == "declared_by_tool"


def test_t6_probe_needs_distinct_sequences_to_be_conclusive(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")
    twins = [entities("B")[0], entities("B")[0].model_copy(update={"label": "copy"})]

    [probe] = gateway.run_probes("boltzlike", twins)

    assert probe.status == "fail" and probe.findings[0].code == "probe_inconclusive"
