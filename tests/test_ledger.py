"""T5: the ledger is an append-only hash chain and says when it was cut or edited."""

from __future__ import annotations

import pytest

from biowarden.gateway import Gateway
from biowarden.ledger import Ledger, LedgerCorruptError, verify_ledger
from helpers import SWAPPED, TOOL_ORDER, entities, read_ledger


@pytest.fixture
def ledger_path(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for i in range(3):
        ledger.append({"kind": "call", "i": i})
    return ledger.path


def test_t5_chain_verifies_and_links_each_record_to_the_previous_one(ledger_path):
    report = verify_ledger(ledger_path)
    records = read_ledger(ledger_path)

    assert report.ok and report.n_records == 3 and not report.warnings
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert records[0]["prev_hash"] == "0" * 64
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert records[2]["prev_hash"] == records[1]["hash"]


def test_t5_modified_record_is_detected(ledger_path):
    text = ledger_path.read_text()
    ledger_path.write_text(text.replace('"i":1', '"i":9'))

    report = verify_ledger(ledger_path)

    assert not report.ok
    assert any("line 2" in e and "hash mismatch" in e for e in report.errors)


def test_t5_deleted_middle_record_is_detected(ledger_path):
    lines = ledger_path.read_text().splitlines(keepends=True)
    ledger_path.write_text(lines[0] + lines[2])

    report = verify_ledger(ledger_path)

    assert not report.ok
    assert any("seq" in e or "prev_hash" in e for e in report.errors)


def test_t5_truncated_tail_is_detected_through_the_head_anchor(ledger_path):
    lines = ledger_path.read_text().splitlines(keepends=True)
    ledger_path.write_text("".join(lines[:2]))  # the remaining records are still a valid chain

    report = verify_ledger(ledger_path)

    assert not report.ok
    assert any("head anchor" in e for e in report.errors)


def test_t5_partially_written_last_line_is_detected(ledger_path):
    data = ledger_path.read_bytes()
    ledger_path.write_bytes(data[: len(data) - 40])

    report = verify_ledger(ledger_path)

    assert not report.ok
    assert any("invalid JSON" in e for e in report.errors)


def test_t5_missing_head_is_a_warning_unless_required(ledger_path):
    ledger_path.with_name(ledger_path.name + ".head").unlink()

    assert verify_ledger(ledger_path).ok
    assert verify_ledger(ledger_path).warnings
    assert not verify_ledger(ledger_path, require_head=True).ok


def test_t5_a_corrupt_ledger_refuses_further_appends(ledger_path):
    ledger_path.write_text(ledger_path.read_text().replace('"i":1', '"i":9'))

    with pytest.raises(LedgerCorruptError):
        Ledger(ledger_path).append({"kind": "call", "i": 3})


def test_t5_reserved_fields_cannot_be_set_by_callers(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "ledger.jsonl").append({"hash": "forged"})


def test_t5_gateway_records_carry_the_provenance_fields(tmp_path):
    path = tmp_path / "ledger.jsonl"
    gateway = Gateway(tmp_path / "work", path)
    gateway.predict_complex(entities(TOOL_ORDER), "tfoldlike")
    gateway.predict_complex(entities(SWAPPED), "tfoldlike")

    ok, blocked = read_ledger(path)

    for record in (ok, blocked):
        assert {
            "call_id",
            "ts",
            "tool",
            "tool_version",
            "contract_hash",
            "request_hash",
        } <= record.keys()
        assert record["tool"] == "mockfold" and record["tool_version"] == "0.1.0"
        assert len(record["contract_hash"]) == 64
    assert ok["call_id"] != blocked["call_id"]
    assert ok["outcome"] == "ok" and all(r["ok"] for r in ok["oracle_results"])
    assert blocked["outcome"] == "blocked" and blocked["verdict"]["violations"]
    assert verify_ledger(path).ok


def test_line_separator_characters_inside_a_field_do_not_split_a_record(tmp_path):
    from biowarden.ledger import Ledger, verify_ledger

    ledger = Ledger(tmp_path / "l.jsonl")
    for text in ("a b", "c\x85d", "e f\x1cg"):
        ledger.append({"event": "x", "argv": [text]})

    assert verify_ledger(tmp_path / "l.jsonl").ok


def test_lone_surrogates_in_model_output_are_scrubbed_not_fatal(tmp_path):
    from biowarden.ledger import Ledger, verify_ledger

    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append({"event": "x", "argv": ["ok\udce9bad"], "nested": {"k\ud800": "v"}})

    assert verify_ledger(tmp_path / "l.jsonl").ok
