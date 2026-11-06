"""The MCP front end of the engine, driven in process by a real MCP client."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastmcp import Client

from biowarden.engine import Engine, Promotion
from biowarden.engine.mcp_server import build_mcp
from biowarden.engine.spec import packaged_specs


def ask(server, arguments):
    async def go():
        async with Client(server) as client:
            return await client.call_tool_mcp("run_tool", arguments)

    return asyncio.run(go())


def server_with(tmp_path, ran, mode="block"):
    engine = Engine(packaged_specs(), mode=mode, promotion=Promotion())

    def runner(tool, args, cwd):
        ran.append((tool, list(args), cwd))
        return 0, "out\n", ""

    return build_mcp(engine, runner, tmp_path, ["samtools", "bcftools", "bedtools"])


def test_a_call_without_the_required_declaration_is_an_error_with_the_fix_first(tmp_path):
    ran = []
    server = server_with(tmp_path, ran)

    result = ask(server, {"tool": "bcftools", "args": ["view", "-R", "r.bed", "v.vcf.gz"]})

    assert result.is_error is True and ran == []
    first = result.content[0].text
    assert first.startswith("FIX:") and "region_semantics" in first


def test_a_declared_call_runs_the_tool_and_reports_exit_code_and_output(tmp_path):
    ran = []
    server = server_with(tmp_path, ran)

    result = ask(
        server,
        {
            "tool": "bcftools",
            "args": ["view", "-R", "r.bed", "v.vcf.gz"],
            "declared": {"region_semantics": "overlap"},
        },
    )

    assert result.is_error is False and ran == [
        ("bcftools", ["view", "-R", "r.bed", "v.vcf.gz"], tmp_path)
    ]
    assert result.content[0].text.startswith("[exit code 0]") and "out" in result.content[0].text


def test_unknown_tools_are_rejected_by_the_input_schema(tmp_path):
    ran = []
    server = server_with(tmp_path, ran)

    result = ask(server, {"tool": "rm", "args": ["-rf", "x"]})

    assert result.is_error is True and ran == []


def test_a_string_where_a_list_belongs_is_not_coerced(tmp_path):
    ran = []
    server = server_with(tmp_path, ran)

    result = ask(server, {"tool": "bcftools", "args": "view -h v.vcf.gz"})

    assert result.is_error is True and ran == []


def test_warn_mode_runs_the_call_and_prepends_the_warning(tmp_path):
    ran = []
    server = server_with(tmp_path, ran, mode="warn")

    result = ask(server, {"tool": "bcftools", "args": ["view", "-R", "r.bed", "v.vcf.gz"]})

    assert result.is_error is False and len(ran) == 1
    assert result.content[0].text.startswith("WARNING")


def test_the_tool_description_lists_the_declarations(tmp_path):
    server = server_with(tmp_path, [])

    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    (tool,) = asyncio.run(go())
    assert "region_semantics = overlap|start_only" in tool.description
    assert "fastq_quality_offset" in tool.description


__all__ = ["Path"]
