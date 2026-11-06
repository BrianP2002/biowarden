"""T8: calls that follow the contracts raise no alarm."""

from __future__ import annotations

from pathlib import Path

from biowarden.gateway import Gateway
from helpers import TOOL_ORDER, entities, read_ledger


def test_t8_clean_tfoldlike_call_raises_no_false_alarm(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    result = gateway.predict_complex(entities(TOOL_ORDER), "tfoldlike")

    assert result.status == "ok" and not result.is_error
    assert result.violations == [] and not result.fix_applied
    assert result.oracle_results and all(r.ok and not r.findings for r in result.oracle_results)
    assert [(m.position, m.chain_id) for m in result.entity_map] == list(enumerate(TOOL_ORDER))
    assert read_ledger(tmp_path / "ledger.jsonl")[-1]["outcome"] == "ok"


def test_t8_clean_gamma_delta_call_raises_no_false_alarm(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    result = gateway.predict_complex(entities("DGMNP"), "tfoldlike")

    assert result.status == "ok"
    assert [m.chain_id for m in result.entity_map] == list("DGMNP")


def test_t8_clean_boltzlike_call_in_any_order_raises_no_false_alarm(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    result = gateway.predict_complex(entities("PMNAB"), "boltzlike")

    assert result.status == "ok" and result.violations == []
    assert [(m.label, m.chain_id) for m in result.entity_map] == [
        ("peptide", "A"),
        ("mhc", "B"),
        ("b2m", "C"),
        ("tcr_alpha", "D"),
        ("tcr_beta", "E"),
    ]
    assert result.metrics["plddt_scale"] == [0.0, 100.0]


def test_t8_repeated_clean_calls_are_deterministic(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl")

    first = gateway.predict_complex(entities(TOOL_ORDER), "tfoldlike")
    second = gateway.predict_complex(entities(TOOL_ORDER), "tfoldlike")

    assert first.call_id != second.call_id
    assert first.artifacts and second.artifacts
    assert Path(first.artifacts["pdb"]).read_text() == Path(second.artifacts["pdb"]).read_text()
