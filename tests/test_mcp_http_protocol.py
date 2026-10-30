"""Mock HTTP MCP protocol tests (no network or credentials required)."""
from __future__ import annotations

import json

import httpx
import pytest

from agent.mcp.client import HTTPTransport, MCPServerConfig, MCPTransport, MCPState


@pytest.mark.asyncio
async def test_streamable_http_propagates_session_and_protocol_headers():
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(204)
        observed.append(request)
        data = json.loads(request.content)
        if data.get("method") == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": data["id"],
                            "result": {"protocolVersion": "2025-03-26"}},
                headers={"Mcp-Session-Id": "session-123"},
            )
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": data["id"],
                        "result": {"tools": []}},
        )

    config = MCPServerConfig(
        name="server", transport=MCPTransport.STREAMABLE_HTTP,
        url="https://example.com/mcp",
    )
    transport = HTTPTransport(config)
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport.state = MCPState.CONNECTED
    first = await transport.request("initialize", {"protocolVersion": "2025-03-26"})
    assert first["protocolVersion"] == "2025-03-26"
    transport._protocol_version = first["protocolVersion"]
    assert (await transport.request("tools/list")) == {"tools": []}
    assert observed[1].headers["mcp-session-id"] == "session-123"
    assert observed[1].headers["mcp-protocol-version"] == "2025-03-26"
    assert observed[1].url.path == "/mcp"
    await transport.disconnect()


@pytest.mark.asyncio
async def test_legacy_sse_uses_server_advertised_message_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            text="event: endpoint\ndata: /mcp/messages?session=123\n\n",
            headers={"content-type": "text/event-stream"},
        )

    transport = HTTPTransport(MCPServerConfig(
        name="legacy", transport=MCPTransport.SSE,
        url="https://example.com/mcp/sse",
    ))
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport.state = MCPState.CONNECTED
    await transport._sse_loop()
    assert transport._endpoint_ready.is_set()
    assert transport._post_url == "https://example.com/mcp/messages?session=123"
    await transport.disconnect()


@pytest.mark.asyncio
async def test_sse_endpoint_rejects_cross_origin_post_url():
    transport = HTTPTransport(MCPServerConfig(
        name="legacy", transport=MCPTransport.SSE,
        url="https://example.com/sse",
    ))
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(
            200, text="event: endpoint\ndata: https://attacker.test/messages\n\n",
            headers={"content-type": "text/event-stream"},
        ),
    ))
    transport.state = MCPState.CONNECTED
    await transport._sse_loop()
    assert not transport._endpoint_ready.is_set()
    assert transport._post_url == "https://example.com/sse"
    assert transport.state == MCPState.ERROR
    await transport.disconnect()
