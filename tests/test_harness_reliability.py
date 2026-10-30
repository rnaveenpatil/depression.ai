"""Regression coverage for MCP lifecycle and deterministic agent tool ordering."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.mcp.client import MCPClient, MCPServerConfig, MCPState, MCPTransportBase, MCPTransport


@pytest.mark.asyncio
async def test_tool_order_does_not_move_reads_ahead_of_writes():
    loop = AgentLoop(
        agent=SimpleNamespace(), llm=SimpleNamespace(),
        tool_registry=SimpleNamespace(), planner=None,
        config={"enable_parallel_reads": True},
    )
    observed = []

    async def fake_action(call, **kwargs):
        observed.append(call.name)
        return {"tool": call.name, "result": {"success": True}}

    loop._act_one = fake_action
    loop._is_read_only = lambda name: name == "read"
    calls = [
        SimpleNamespace(name="write", arguments={"filePath": "a"}),
        SimpleNamespace(name="read", arguments={"filePath": "a"}),
        SimpleNamespace(name="write", arguments={"filePath": "b"}),
        SimpleNamespace(name="read", arguments={"filePath": "b"}),
    ]
    await loop._act(calls)
    assert observed == ["write", "read", "write", "read"]


@pytest.mark.asyncio
async def test_parallel_reads_run_together_without_crossing_write():
    loop = AgentLoop(
        agent=SimpleNamespace(), llm=SimpleNamespace(),
        tool_registry=SimpleNamespace(), planner=None,
        config={"enable_parallel_reads": True},
    )
    times = []

    async def fake_action(call, **kwargs):
        times.append(("start", call.name))
        await asyncio.sleep(0)
        times.append(("done", call.name))
        return {"tool": call.name, "result": {"success": True}}

    loop._act_one = fake_action
    loop._is_read_only = lambda name: name.startswith("read")
    await loop._act([
        SimpleNamespace(name="read-a", arguments={}),
        SimpleNamespace(name="read-b", arguments={}),
        SimpleNamespace(name="write", arguments={}),
    ])
    assert times.index(("start", "read-b")) < times.index(("done", "read-a"))
    assert times.index(("start", "write")) > times.index(("done", "read-b"))


class DummyTransport(MCPTransportBase):
    async def connect(self):
        self.state = MCPState.CONNECTED

    async def disconnect(self):
        self.state = MCPState.CLOSED
        self._fail_pending("disconnected")

    async def _send(self, msg):
        pass

    async def _receive_loop(self):
        pass


@pytest.mark.asyncio
async def test_mcp_notification_handler_can_issue_requests_without_blocking_reader():
    cfg = MCPServerConfig(name="test", transport=MCPTransport.STDIO)
    transport = DummyTransport(cfg)
    await transport.connect()
    async def notification(message):
        assert await transport.request("tools/list", timeout=1) == {"tools": []}

    transport.add_notification_handler(notification)
    await asyncio.wait_for(transport._handle_message({
        "jsonrpc": "2.0", "method": "notifications/tools/list_changed",
    }), timeout=0.1)
    await asyncio.sleep(0)
    # Reader can still deliver the response required by the notification.
    ids = list(transport._pending)
    assert len(ids) == 1
    await transport._handle_message({
        "jsonrpc": "2.0", "id": ids[0], "result": {"tools": []},
    })
    await asyncio.sleep(0)
    await transport.disconnect()


@pytest.mark.asyncio
async def test_mcp_disconnect_fails_pending_requests_immediately():
    cfg = MCPServerConfig(name="test", transport=MCPTransport.STDIO)
    transport = DummyTransport(cfg)
    await transport.connect()
    task = asyncio.create_task(transport.request("tools/list", timeout=10))
    await asyncio.sleep(0)
    assert transport._pending
    await transport.disconnect()
    with pytest.raises(Exception, match="disconnected"):
        await task
    assert transport._pending == {}


@pytest.mark.asyncio
async def test_failed_mcp_initialization_exposes_connection_errors(monkeypatch):
    async def fail(cfg):
        raise RuntimeError("wrong credentials")
    client = MCPClient({
        "enabled": True, "servers": [{
            "name": "bad", "transport": "stdio", "command": "missing",
        }],
    })
    monkeypatch.setattr(client, "_connect_server", fail)
    await client.initialize()
    assert client.connection_errors["bad"] == "wrong credentials"
    assert client.get_status()["errors"]["bad"] == "wrong credentials"
    await client.shutdown()


@pytest.mark.asyncio
async def test_reconnect_retries_servers_missing_from_live_transports(monkeypatch):
    client = MCPClient({
        "enabled": True, "servers": [{
            "name": "lost", "transport": "stdio", "command": "missing",
        }],
    })
    seen = []
    async def fake_reconnect(name, cfg):
        seen.append(name)
    monkeypatch.setattr(client, "_reconnect", fake_reconnect)
    await client._health_check()
    assert seen == ["lost"]


@pytest.mark.asyncio
async def test_simple_question_does_not_call_intent_classifier(monkeypatch):
    loop = AgentLoop(
        agent=SimpleNamespace(), llm=SimpleNamespace(),
        tool_registry=SimpleNamespace(), planner=None,
        config={"enable_planning": True},
    )
    async def fail_classifier(query):
        raise AssertionError("unnecessary classifier called")
    async def fake_run(query):
        return {"success": True, "response": "answer"}
    monkeypatch.setattr(loop, "_classify_intent", fail_classifier)
    monkeypatch.setattr(loop, "_run_free_form", fake_run)
    assert (await loop.run("What is Python?"))["success"]
    assert loop.context.intent == "question"
    assert loop.context.metadata["intent_needs_plan"] is False


@pytest.mark.asyncio
async def test_coordinator_avoids_nested_build_planning():
    from agent.agent.dual_agent import AgentCoordinator

    loop = SimpleNamespace(enable_planning=True)
    observed = []
    async def execute(**kwargs):
        observed.append(loop.enable_planning)
        return {"success": True, "response": "completed", "tool_calls": 1}

    coordinator = AgentCoordinator(
        plan_agent=SimpleNamespace(),
        build_agent=SimpleNamespace(loop=loop, process_query=execute),
        config={},
    )
    result = await coordinator._execute_plan_with_build("Already planned", {})
    assert result["success"] is True
    assert observed == [False]
    assert loop.enable_planning is True
