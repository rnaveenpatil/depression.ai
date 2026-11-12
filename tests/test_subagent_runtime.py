"""Subagent runtime selection, limits, and lifecycle coverage."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.agent.subagent import (
    BUILTIN_SUBAGENTS,
    SubAgentManager,
    SubAgentRole,
)
from agent.tools.task import TaskTool


class FakeRegistry:
    def __init__(self, current: str | None = "active-model"):
        self.current = current
        self.available = {current} if current else set()
        self.used_models: list[str | None] = []
        self.calls = 0

    def get_current_model(self):
        return self.current

    def get_model_info(self, model):
        if model not in self.available:
            return None
        return {"model": model, "available": True}

    def set_api_key(self, _provider, _key):
        pass

    async def complete_with_tools(self, *, messages, tools, model, max_tokens):
        self.calls += 1
        self.used_models.append(model)
        return SimpleNamespace(content="done", tool_calls=[])


class ToolRegistryStub:
    def get_schemas(self):
        return []

    async def execute(self, _name, _arguments):
        return {"success": True}


def make_manager(monkeypatch, config=None, registry=None):
    manager = SubAgentManager(SimpleNamespace(config={"llm": {}}), config or {})
    manager.llm_registry = registry or FakeRegistry()

    async def fake_tools(_cfg, _subagent):
        return ToolRegistryStub()

    monkeypatch.setattr(manager, "_create_subagent_tools", fake_tools)
    return manager


def test_builtin_subagents_do_not_hardcode_models():
    assert all(cfg.model == "" for cfg in BUILTIN_SUBAGENTS.values())


@pytest.mark.asyncio
async def test_subagent_inherits_active_runtime_model(monkeypatch):
    registry = FakeRegistry("the-connected-model")
    manager = make_manager(monkeypatch, registry=registry)

    result = await manager.invoke_subagent(SubAgentRole.GENERAL, "hello")

    assert result["success"] is True
    assert result["model"] == "the-connected-model"
    assert registry.used_models == ["the-connected-model"]


@pytest.mark.asyncio
async def test_subagent_model_override_falls_back_when_unavailable(monkeypatch):
    registry = FakeRegistry("active-model")
    registry.available.add("role-model")
    manager = make_manager(monkeypatch, registry=registry)
    manager.subagent_configs[SubAgentRole.EXPLORE].model = "role-model"

    selected = await manager.invoke_subagent(SubAgentRole.EXPLORE, "first")
    assert selected["model"] == "role-model"

    registry.available.discard("role-model")
    fallback = await manager.invoke_subagent(SubAgentRole.EXPLORE, "second")
    assert fallback["model"] == "active-model"
    assert registry.used_models == ["role-model", "active-model"]


@pytest.mark.asyncio
async def test_subagent_uses_configured_subagent_model(monkeypatch):
    registry = FakeRegistry("active-model")
    registry.available.add("shared-subagent-model")
    manager = make_manager(
        monkeypatch,
        {"subagent_model": "shared-subagent-model"},
        registry=registry,
    )

    result = await manager.invoke_subagent(SubAgentRole.SCOUT, "research")

    assert result["model"] == "shared-subagent-model"


@pytest.mark.asyncio
async def test_subagent_max_turns_is_a_failure(monkeypatch):
    class LoopingRegistry(FakeRegistry):
        async def complete_with_tools(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(id=f"call-{self.calls}", name="noop", arguments={})
                ],
            )

    manager = make_manager(
        monkeypatch,
        {"max_turns": 2},
        registry=LoopingRegistry(),
    )

    result = await manager.invoke_subagent(SubAgentRole.GENERAL, "loop")

    assert result["success"] is False
    assert "max turns" in result["error"]


@pytest.mark.asyncio
async def test_subagent_timeout_is_reported(monkeypatch):
    class SlowRegistry(FakeRegistry):
        async def complete_with_tools(self, **_kwargs):
            await asyncio.sleep(5)
            return SimpleNamespace(content="late", tool_calls=[])

    manager = make_manager(
        monkeypatch,
        {"default_timeout": 0.01},
        registry=SlowRegistry(),
    )

    result = await manager.invoke_subagent(SubAgentRole.GENERAL, "slow")

    assert result["success"] is False
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_active_subagent(monkeypatch):
    started = asyncio.Event()
    blocker = asyncio.Event()

    class BlockingRegistry(FakeRegistry):
        async def complete_with_tools(self, **_kwargs):
            started.set()
            await blocker.wait()
            return SimpleNamespace(content="late", tool_calls=[])

    manager = make_manager(monkeypatch, registry=BlockingRegistry())
    task = asyncio.create_task(
        manager.invoke_subagent(SubAgentRole.GENERAL, "block")
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    await manager.shutdown()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, SubAgentRole.GENERAL),
        ({"role": "explore"}, SubAgentRole.EXPLORE),
        ({"role": "scout"}, SubAgentRole.SCOUT),
        ({"capabilities": ["read", "grep", "glob"]}, SubAgentRole.EXPLORE),
        ({"capabilities": ["research", "docs"]}, SubAgentRole.SCOUT),
        ({"capabilities": ["read", "write"]}, SubAgentRole.GENERAL),
    ],
)
def test_task_tool_maps_capabilities_and_role(params, expected):
    role, error = TaskTool._resolve_role(params)
    assert error is None
    assert role is expected


def test_task_tool_rejects_unknown_role():
    role, error = TaskTool._resolve_role({"role": "unknown"})
    assert role is None
    assert "unknown subagent role" in error


@pytest.mark.asyncio
async def test_max_subagents_limits_concurrency(monkeypatch):
    class ConcurrentRegistry(FakeRegistry):
        def __init__(self):
            super().__init__("active-model")
            self.active = 0
            self.max_active = 0

        async def complete_with_tools(self, **_kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.02)
                return SimpleNamespace(content="done", tool_calls=[])
            finally:
                self.active -= 1

    registry = ConcurrentRegistry()
    manager = make_manager(
        monkeypatch,
        {"max_subagents": 1},
        registry=registry,
    )

    results = await asyncio.gather(*(
        manager.invoke_subagent(SubAgentRole.GENERAL, str(index))
        for index in range(3)
    ))

    assert all(result["success"] for result in results)
    assert registry.max_active == 1
