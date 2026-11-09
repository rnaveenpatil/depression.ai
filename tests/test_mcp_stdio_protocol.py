"""stdio MCP framing tests, including frames larger than asyncio's default."""
from __future__ import annotations

import sys

import pytest

from agent.mcp.client import MCPClient, MCPServerConfig, StdioTransport, MCPTransport


# A minimal MCP server whose tools/list payload is deliberately huge and all on
# one line, mirroring real servers such as REA (229 tools, ~3.5 MB).
_FAKE_SERVER = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    message = json.loads(raw)
    method = message.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": message["params"].get(
                "protocolVersion", "2025-03-26"
            ),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "big",
                    "description": "x" * 200000,
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
    else:
        continue
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result})
        + "\n"
    )
    sys.stdout.flush()
"""


def _server_config(**overrides):
    cfg = {
        "name": "fake",
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-c", _FAKE_SERVER],
        "auto_reconnect": False,
        "timeout": 10.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.asyncio
async def test_stdio_discovers_tools_larger_than_asyncio_default_limit():
    client = MCPClient({"enabled": True, "servers": [_server_config()]})
    try:
        await client.initialize()
        assert client.connection_errors == {}
        assert list(client.tools) == ["mcp__fake__big"]
        assert len(client.tools["mcp__fake__big"].description) == 200000
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_stdio_rejects_frame_above_configured_limit():
    client = MCPClient({
        "enabled": True,
        "servers": [_server_config(max_message_bytes=64 * 1024)],
    })
    try:
        await client.initialize()
        assert client.tools == {}
        assert "exceeded" in client.connection_errors["fake"]
    finally:
        await client.shutdown()


def test_stdio_config_resolves_message_limit_and_root_timeout():
    server = _server_config()
    server.pop("timeout")
    client = MCPClient({
        "enabled": True,
        "timeout": 75.0,
        "max_message_bytes": 8 * 1024 * 1024,
        "servers": [server],
    })
    config = client.servers_cfg[0]
    assert config.max_message_bytes == 8 * 1024 * 1024
    assert config.timeout == 75.0


def test_stdio_config_rejects_too_small_message_limit():
    client = MCPClient({
        "enabled": True,
        "servers": [_server_config(max_message_bytes=1024)],
    })
    assert client.servers_cfg[0].max_message_bytes == 64 * 1024 * 1024


def test_stdio_transport_accepts_large_frame_bound():
    transport = StdioTransport(MCPServerConfig(
        name="fake", transport=MCPTransport.STDIO,
        command=sys.executable, args=["-c", _FAKE_SERVER],
    ))
    assert transport.config.max_message_bytes == 64 * 1024 * 1024
