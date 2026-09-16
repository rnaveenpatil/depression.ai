"""
Event bridge between the agent runtime and the TUI.

Agent workers run in Textual's thread pool. Their loops emit events on
the worker's event loop. Textual owns the UI loop. This module marshals
events onto the UI loop safely and exposes a single subscribe() entry
point for the app.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class AgentEvent:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


class EventBridge:
    """
    Collects events from agent loops and replays them on the UI loop.

    Usage:
        bridge = EventBridge()
        bridge.attach(coordinator)          # wires on_tool_executed, etc.
        bridge.subscribe("on_tool_executed", handler)
    """

    def __init__(self, ui_loop: Optional[asyncio.AbstractEventLoop] = None):
        self._ui_loop = ui_loop
        self._subscribers: Dict[str, List[Callable[[AgentEvent], Awaitable[None]]]] = {}
        self._lock = threading.Lock()
        self._attached: List[Any] = []

    # ------------------------------------------------------------------

    def set_ui_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._ui_loop = loop

    def subscribe(
        self, name: str, handler: Callable[[AgentEvent], Awaitable[None]]
    ) -> None:
        with self._lock:
            self._subscribers.setdefault(name, []).append(handler)

    def unsubscribe(self, name: str, handler: Callable[[AgentEvent], Awaitable[None]]) -> None:
        with self._lock:
            if name in self._subscribers and handler in self._subscribers[name]:
                self._subscribers[name].remove(handler)

    # ------------------------------------------------------------------

    def attach(self, coordinator: Any) -> None:
        """Hook every agent loop in the coordinator."""
        for agent in self._iter_agents(coordinator):
            loop = getattr(agent, "loop", None)
            if loop is None:
                continue
            try:
                loop.add_event_handler("on_tool_executed", self._on_tool_executed)
                loop.add_event_handler("on_query_complete", self._on_query_complete)
                self._attached.append(loop)
            except Exception:
                pass

    def detach(self) -> None:
        # AgentLoop has no remove_event_handler; clearing the handlers we
        # installed is best-effort since the loop is discarded on shutdown.
        self._attached.clear()

    @staticmethod
    def _iter_agents(coordinator: Any):
        for attr in ("plan_agent", "build_agent"):
            agent = getattr(coordinator, attr, None)
            if agent is not None:
                yield agent

    # ------------------------------------------------------------------

    async def _on_tool_executed(self, data: Dict[str, Any]) -> None:
        await self._emit("on_tool_executed", data)

    async def _on_query_complete(self, data: Dict[str, Any]) -> None:
        await self._emit("on_query_complete", data)

    async def _emit(self, name: str, payload: Dict[str, Any]) -> None:
        event = AgentEvent(name=name, payload=payload or {})
        with self._lock:
            handlers = list(self._subscribers.get(name, []))

        # Marshal onto the UI loop when we're not on it.
        if self._ui_loop is not None and self._ui_loop.is_running():
            current = asyncio.get_event_loop()
            if current is not self._ui_loop:
                self._ui_loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self._dispatch(handlers, event))
                )
                return

        await self._dispatch(handlers, event)

    async def _dispatch(
        self,
        handlers: List[Callable[[AgentEvent], Awaitable[None]]],
        event: AgentEvent,
    ) -> None:
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Never let a UI handler crash the agent.
                pass