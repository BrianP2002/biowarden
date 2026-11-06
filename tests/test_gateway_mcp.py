"""T4: the FastMCP gateway end to end, in process, as an MCP client sees it."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from biowarden.gateway import Gateway
from biowarden.gateway.server import build_server
from biowarden.ledger import verify_ledger
from helpers import SWAPPED, TOOL_ORDER, entities, read_ledger


def call_predict(server: Any, order: str, backend: str, **extra: Any):
    arguments = {"entities": [e.model_dump() for e in entities(order)], "backend": backend, **extra}

    async def go():
        async with Client(server) as client:
            return await client.call_tool_mcp("predict_complex", arguments)

    return asyncio.run(go())


def test_t4_allowed_call_returns_structured_fields_and_text(tmp_path):
    server = build_server(Gateway(tmp_path / "work", tmp_path / "ledger.jsonl"))

    result = call_predict(server, TOOL_ORDER, "tfoldlike")

    assert result.is_error is False
    data = result.structured_content
    assert data["status"] == "ok" and data["violations"] == []
    assert [m["chain_id"] for m in data["entity_map"]] == list(TOOL_ORDER)
    assert [m["label"] for m in data["entity_map"]][:2] == ["tcr_beta", "tcr_alpha"]
    assert data["metrics"]["plddt_scale"] == [0.0, 1.0]
    assert all(r["ok"] for r in data["oracle_results"])
    assert Path(data["artifacts"]["pdb"]).exists()
    assert result.content[0].text.startswith("OK tfoldlike")


def test_t4_blocked_call_is_an_error_with_violations_and_a_proposed_fix(tmp_path):
    server = build_server(Gateway(tmp_path / "work", tmp_path / "ledger.jsonl"))

    result = call_predict(server, SWAPPED, "tfoldlike")

    assert result.is_error is True
    data = result.structured_content
    assert data["status"] == "blocked"
    assert [(v["slot"], v["code"]) for v in data["violations"]] == [
        ("chain_order", "order_mismatch")
    ]
    assert [e["role"] for e in data["proposed_fix"]["entities"]] == list(TOOL_ORDER)
    assert data["fix_applied"] is False and data["entity_map"] is None
    assert result.content[0].text.startswith("BLOCKED")
    assert "NOT applied" in result.content[0].text


def test_t4_accepting_the_fix_through_mcp_runs_the_tool(tmp_path):
    server = build_server(Gateway(tmp_path / "work", tmp_path / "ledger.jsonl"))

    result = call_predict(server, SWAPPED, "tfoldlike", apply_proposed_fix=True)

    assert result.is_error is False
    assert result.structured_content["fix_applied"] is True
    assert [m["label"] for m in result.structured_content["entity_map"]][:2] == [
        "tcr_beta",
        "tcr_alpha",
    ]


def test_t4_oracle_flagged_call_is_an_error_and_never_looks_successful(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    server = build_server(Gateway(tmp_path / "work", ledger, precall=False))

    result = call_predict(server, SWAPPED, "tfoldlike")

    assert result.is_error is True
    data = result.structured_content
    assert data["status"] == "flagged"
    assert data["entity_map"] is None and data["metrics"] is None
    failed = [r for r in data["oracle_results"] if not r["ok"]]
    assert [r["oracle"] for r in failed] == ["chain_sequence_roundtrip"]
    assert result.content[0].text.startswith("FLAGGED")
    assert read_ledger(ledger)[-1]["outcome"] == "flagged"


def test_t4_every_mcp_call_lands_in_one_verifiable_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    server = build_server(Gateway(tmp_path / "work", ledger))

    call_predict(server, TOOL_ORDER, "tfoldlike")
    call_predict(server, SWAPPED, "tfoldlike")
    call_predict(server, "PMNAB", "boltzlike")

    assert [r["outcome"] for r in read_ledger(ledger)] == ["ok", "blocked", "ok"]
    assert verify_ledger(ledger).ok


def test_t4_stdio_entry_point_serves_the_same_contract_checks(tmp_path):
    transport = StdioTransport(
        command=sys.executable,
        args=["-I", "-m", "biowarden.gateway.server"]
        + ["--work-dir", str(tmp_path / "work"), "--ledger", str(tmp_path / "ledger.jsonl")],
    )
    arguments = {"entities": [e.model_dump() for e in entities(SWAPPED)], "backend": "tfoldlike"}

    async def go():
        async with Client(transport) as client:
            return await client.call_tool_mcp("predict_complex", arguments)

    result = asyncio.run(go())

    assert result.is_error is True
    assert result.structured_content["status"] == "blocked"
    assert verify_ledger(tmp_path / "ledger.jsonl").n_records == 1
