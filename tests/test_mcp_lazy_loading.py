"""Lazy exposure of large MCP tool catalogs."""
from __future__ import annotations

import pytest

from agent.mcp.client import MCPClient, MCPTool as MCPToolInfo
from agent.tools.mcp import MCPTool as MCPMetaTool
from agent.tools.registry import ToolRegistry


def _server(name: str = "big", **overrides):
    cfg = {
        "name": name,
        "transport": "stdio",
        "command": "true",
        "args": [],
        "auto_reconnect": False,
    }
    cfg.update(overrides)
    return cfg


def _client_with_tools(names, server="big", **overrides):
    client = MCPClient({"enabled": True, "servers": [_server(server, **overrides)]})
    for name in names:
        client.tools[f"mcp__{server}__{name}"] = MCPToolInfo(
            server=server,
            name=name,
            description=f"{name} tool for testing",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    return client


def test_large_catalog_is_hidden_behind_meta_tool():
    names = [f"tool_{i}" for i in range(229)]
    client = _client_with_tools(names)
    registry = ToolRegistry()

    registered = client.register_tools_into(registry)

    assert registered == 0
    assert registry.list_tools() == []
    assert len(client.list_tools()) == 229


def test_small_catalog_is_exposed_directly():
    client = _client_with_tools(["read_file", "write_file"])
    registry = ToolRegistry()

    assert client.register_tools_into(registry) == 2
    assert set(registry.list_tools()) == {
        "mcp__big__read_file",
        "mcp__big__write_file",
    }


def test_allowed_tools_are_always_direct():
    client = _client_with_tools(
        [f"tool_{i}" for i in range(30)],
        allowed_tools=["tool_3"],
    )
    registry = ToolRegistry()

    client.register_tools_into(registry)

    assert registry.list_tools() == ["mcp__big__tool_3"]


def test_activation_exposes_selected_tool_and_reset_removes_it():
    client = _client_with_tools([f"tool_{i}" for i in range(30)])
    registry = ToolRegistry()
    client.register_tools_into(registry)

    assert client.activate_tool("mcp__big__tool_7") == "mcp__big__tool_7"
    assert registry.list_tools() == ["mcp__big__tool_7"]

    client.reset_active_tools()
    assert registry.list_tools() == []


def test_search_ranks_matching_tools_and_lists_required_args():
    client = _client_with_tools(["create_issue", "delete_bucket", "list_issues"])

    results = client.search_tools("issue create", limit=2)

    assert results
    assert results[0]["name"] == "mcp__big__create_issue"
    assert results[0]["required"] == ["query"]
    assert all("inputSchema" not in r for r in results)


def test_meta_tool_search_activates_and_call_uses_full_name():
    client = _client_with_tools(["create_issue", "delete_bucket"])
    registry = ToolRegistry()
    client.register_tools_into(registry)
    assert "mcp" not in registry.tools

    meta = MCPMetaTool(client)
    import asyncio

    result = asyncio.run(meta.execute({"action": "search", "query": "issue"}))
    assert result["success"] is True
    assert "mcp__big__create_issue" in result["activated"]
    assert "mcp__big__create_issue" in registry.list_tools()

    calls = []

    async def fake_call(name, arguments, timeout=None):
        calls.append((name, arguments))
        return {"success": True}

    client.call_tool = fake_call
    called = asyncio.run(
        meta.execute({
            "action": "call",
            "tool": "create_issue",
            "arguments": {"query": "x"},
        })
    )
    assert called["success"] is True
    assert calls == [("mcp__big__create_issue", {"query": "x"})]


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected():
    client = _client_with_tools(["known"])
    assert client.activate_tool("missing") is None
    assert client.get_tool_schema("missing") is None
