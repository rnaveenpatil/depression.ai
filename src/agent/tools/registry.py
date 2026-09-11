"""
Tool Registry - Central catalog of all tools available to the agent.

Responsibilities:
    - Register / unregister tools by name
    - Provide OpenAI-style tool schemas for the LLM
    - Dispatch tool calls to the right tool
    - Track execution stats
    - Enforce simple rate limits and timeouts
"""

from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.utils.errors import ToolError, ToolNotFoundError, ToolExecutionError

logger = get_logger(__name__)


# ======================================================================
# BASE TOOL
# ======================================================================

class BaseTool(ABC):
    """Base class for all tools."""

    name: str = "base"
    description: str = ""
    parameters: Dict[str, Any] = {}
    timeout: float = 60.0
    enabled: bool = True

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


# ======================================================================
# REGISTRY
# ======================================================================

@dataclass
class ToolStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    last_error: Optional[str] = None


class ToolRegistry:
    """Central catalog of tools."""

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.tools: Dict[str, BaseTool] = {}
        self._external: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, ToolStats] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------

    async def register_tool(self, tool: BaseTool) -> None:
        if not getattr(tool, "name", None):
            raise ToolError("Tool must have a name")
        self.tools[tool.name] = tool
        self.stats.setdefault(tool.name, ToolStats())
        logger.debug(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def register_external(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._external[name] = {
            "handler": handler,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }
        self.stats.setdefault(name, ToolStats())
        logger.debug(f"Registered external tool: {name}")

    def list_tools(self, include_disabled: bool = False) -> List[str]:
        names = list(self.tools.keys()) + list(self._external.keys())
        if not include_disabled:
            names = [
                n for n in names
                if n in self._external or getattr(self.tools[n], "enabled", True)
            ]
        return sorted(names)

    # ------------------------------------------------------------------
    # SCHEMAS
    # ------------------------------------------------------------------

    def get_schemas(self) -> List[Dict[str, Any]]:
        out = []
        for name, tool in self.tools.items():
            if not getattr(tool, "enabled", True):
                continue
            out.append(tool.to_schema())
        for name, info in self._external.items():
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": info.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return out

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        params = params or {}

        # External tool
        if name in self._external:
            return await self._execute_external(name, params, timeout)

        tool = self.tools.get(name)
        if not tool:
            raise ToolNotFoundError(f"Tool not found: {name}")

        if not getattr(tool, "enabled", True):
            return {"success": False, "error": f"Tool '{name}' is disabled"}

        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                tool.execute(params),
                timeout=timeout or getattr(tool, "timeout", 60.0),
            )
            if not isinstance(result, dict):
                result = {"success": True, "result": result}

            stats = self.stats[name]
            stats.calls += 1
            if result.get("success", True):
                stats.successes += 1
            else:
                stats.failures += 1
                stats.last_error = str(result.get("error", ""))
            stats.total_time += time.time() - t0

            return result

        except asyncio.TimeoutError:
            self.stats[name].calls += 1
            self.stats[name].failures += 1
            self.stats[name].last_error = "timeout"
            return {"success": False, "error": f"Tool '{name}' timed out"}

        except Exception as e:
            self.stats[name].calls += 1
            self.stats[name].failures += 1
            self.stats[name].last_error = str(e)
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _execute_external(
        self, name: str, params: Dict[str, Any], timeout: Optional[float]
    ) -> Dict[str, Any]:
        info = self._external[name]
        handler = info["handler"]
        try:
            result = await asyncio.wait_for(handler(**params), timeout=timeout or 60.0)
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            self.stats[name].calls += 1
            self.stats[name].successes += 1
            return result
        except asyncio.TimeoutError:
            self.stats[name].calls += 1
            self.stats[name].failures += 1
            return {"success": False, "error": f"Tool '{name}' timed out"}
        except Exception as e:
            self.stats[name].calls += 1
            self.stats[name].failures += 1
            self.stats[name].last_error = str(e)
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "calls": s.calls,
                "successes": s.successes,
                "failures": s.failures,
                "avg_time": (s.total_time / s.calls) if s.calls else 0,
                "last_error": s.last_error,
            }
            for name, s in self.stats.items()
        }