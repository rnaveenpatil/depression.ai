"""
Event bridge between the agent runtime and the TUI.

Agent workers run in Textual's thread pool. Their loops emit events on
the worker's event loop. Textual owns the UI loop. This module marshals
events onto the UI loop safely and exposes a single subscribe() entry
point for the app.

[Streaming fix] attach() now forwards EVERY event a loop emits, not just
on_tool_executed and on_query_complete. Streaming tokens (`on_token`),
turn boundaries (`on_token_end`), and phase transitions (`on_phase`)
reach the UI the same way.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


# Every event name a loop may emit. attach() registers a forwarder for
# each of these on every agent loop.
_FORWARDED_EVENTS: tuple = (
    "on_tool_executed",
    "on_query_complete",
    "on_token",
    "on_token_end",
    "on_phase",
    "on_plan_created",
    "on_plan_updated",
    "on_plan_rendered",
    "on_plan_task_complete",
    "on_plan_task_failed",
)


@dataclass
class AgentEvent:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


class EventBridge:
    """
    Collects events from agent loops and replays them on the UI loop.

    Usage:
        bridge = EventBridge()
        bridge.set_ui_loop(asyncio.get_running_loop())
        bridge.attach(coordinator)
        bridge.subscribe("on_tool_executed", handler)
        bridge.subscribe("on_token", handler)
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

    def unsubscribe(
        self,
        name: str,
        handler: Callable[[AgentEvent], Awaitable[None]],
    ) -> None:
        with self._lock:
            if name in self._subscribers and handler in self._subscribers[name]:
                self._subscribers[name].remove(handler)

    # ------------------------------------------------------------------

    def attach(self, coordinator: Any) -> None:
        """
        Hook every agent loop in the coordinator.

        Registers a forwarder for every event in _FORWARDED_EVENTS. The
        forwarder is a closure that captures the event name so the loop's
        _trigger_event sees a real callable.
        """
        for agent in self._iter_agents(coordinator):
            loop = getattr(agent, "loop", None)
            if loop is None:
                continue
            for event_name in _FORWARDED_EVENTS:
                try:
                    loop.add_event_handler(
                        event_name,
                        self._make_forwarder(event_name),
                    )
                except Exception:
                    pass
            self._attached.append(loop)

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

    def _make_forwarder(self, event_name: str) -> Callable[[Dict[str, Any]], Awaitable[None]]:
        """Return a coroutine function that forwards one event by name."""
        async def _fwd(data: Dict[str, Any]) -> None:
            await self._emit(event_name, data or {})
        return _fwd

    # ------------------------------------------------------------------

    async def _emit(self, name: str, payload: Dict[str, Any]) -> None:
        event = AgentEvent(name=name, payload=payload or {})
        with self._lock:
            handlers = list(self._subscribers.get(name, []))

        if not handlers:
            return

        # If we are on the UI loop already, dispatch directly.
        # Otherwise marshal.
        target_loop = self._ui_loop
        if target_loop is None:
            # No UI loop registered — dispatch inline.
            await self._dispatch(handlers, event)
            return

        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None

        if current is target_loop:
            await self._dispatch(handlers, event)
            return

        # Cross-loop: schedule on the UI loop.
        if not target_loop.is_running():
            # UI loop not running — fall back to inline.
            await self._dispatch(handlers, event)
            return

        target_loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._dispatch(handlers, event))
        )

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