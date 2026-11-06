"""Contract schema: loading, hashing, JSON Schema export, and the gateway's identity checks."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from biowarden.contracts import (
    PACKAGED,
    contract_json_schema,
    load_contract,
    load_packaged,
)
from biowarden.gateway import Gateway
from helpers import TOOL_ORDER, entities, variant


def test_packaged_contracts_load_and_name_their_backends():
    contracts = {name: load_packaged(name) for name in PACKAGED}

    assert {c.tool.backend for c in contracts.values()} == {"tfoldlike", "boltzlike"}
    assert all(c.invariants and c.probes for c in contracts.values())


def test_contract_hash_ignores_yaml_key_order_and_tracks_content(tmp_path):
    original = load_packaged("mockfold_tfoldlike")
    data = original.model_dump(mode="json")
    shuffled = dict(reversed(list(data.items())))
    path = tmp_path / "shuffled.yaml"
    path.write_text(yaml.safe_dump(shuffled, sort_keys=False))

    assert load_contract(path).content_hash() == original.content_hash()
    assert variant("tfoldlike", lambda d: d["tool"].update(version="0.2.0")).content_hash() != (
        original.content_hash()
    )


def test_contract_exports_a_json_schema():
    schema = contract_json_schema()

    assert schema["title"] == "ToolContract"
    assert {"tool", "slots", "invariants", "probes"} <= schema["properties"].keys()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["slots"]["chain_order"].update(order=[]),
        lambda d: d["slots"]["chain_order"].update(required=["Z"]),
        lambda d: d["slots"]["chain_order"].update(order=["B", "B", "A"]),
        lambda d: d["slots"].update(chain_order={"tool_policy": "input_order", "order": ["A"]}),
        lambda d: d["tool"].update(unexpected="field"),
    ],
)
def test_inconsistent_contracts_are_rejected(mutate):
    with pytest.raises(ValidationError):
        variant("tfoldlike", mutate)


def test_gateway_blocks_when_the_tool_version_differs_from_the_pin(tmp_path):
    pinned = variant("tfoldlike", lambda d: d["tool"].update(version="9.9.9"))
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl", contracts={"tfoldlike": pinned})

    result = gateway.predict_complex(entities(TOOL_ORDER), "tfoldlike")

    assert result.status == "blocked"
    [violation] = result.violations
    assert (violation.slot, violation.code) == ("tool_identity", "version_mismatch")
    assert (violation.expected, violation.got) == ("9.9.9", "0.1.0")
    assert not (tmp_path / "work").exists()


def test_gateway_blocks_unknown_backends_with_a_structured_violation(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    result = gateway.predict_complex(entities(TOOL_ORDER), "alphafold")

    assert result.status == "blocked"
    assert result.violations[0].code == "unknown_backend"
    assert result.contract_hash is None


def test_gateway_refuses_contracts_that_name_an_unknown_oracle(tmp_path):
    bad = variant("boltzlike", lambda d: d["invariants"][0].update(oracle="no_such_oracle"))

    with pytest.raises(ValueError, match="no_such_oracle"):
        Gateway(tmp_path / "work", tmp_path / "ledger.jsonl", contracts={"boltzlike": bad})
