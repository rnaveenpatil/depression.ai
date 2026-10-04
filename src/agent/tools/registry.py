"""
Tool Registry - Central catalog of all tools available to the agent.

Single authoritative source for:
    - tool schemas sent to the LLM
    - tool descriptions embedded in the system prompt
    - argument validation
    - execution (built-in + external/MCP)
    - stats
    - tool metadata (read-only / mutating / category)
    - task-based tool routing
    - mutation epoch (for cache invalidation)

Rule: whatever `get_schemas()` returns is EXACTLY what the loop advertises
to the model and EXACTLY what `execute_safe()` can dispatch.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from agent.utils.logging import get_logger
from agent.utils.errors import ToolError, ToolNotFoundError
from agent.utils.redact import redact

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

    # Metadata used for task routing and cache invalidation.
    # Subclasses should override:
    #   read_only   True  => safe to cache result indefinitely (within TTL)
    #   mutating    True  => successful run invalidates the loop's cache
    #   category    one of: "inspect", "edit", "run", "proc", "vcs",
    #               "cloud", "net", "plan", "ui", "misc"
    read_only: bool = False
    mutating: bool = False
    category: str = "misc"

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
# STATS
# ======================================================================

@dataclass
class ToolStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    last_error: Optional[str] = None


# ======================================================================
# LIGHTWEIGHT SCHEMA VALIDATOR
# ======================================================================

def _validate_schema(value: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
    """
    Minimal JSON-Schema validator covering the subset tools actually use:
    type, required, enum, properties, items, additionalProperties.
    Returns a list of human-readable error strings (empty == valid).
    """
    errors: List[str] = []
    if not schema:
        return errors

    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path or 'arguments'} must be an object, got {type(value).__name__}"]
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        for req in required:
            if req not in value:
                errors.append(f"{path or 'arguments'}: missing required field '{req}'")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path or 'arguments'}: unexpected field '{k}'")
        for k, v in value.items():
            sub = props.get(k)
            if sub is not None:
                errors.extend(_validate_schema(v, sub, f"{path}.{k}" if path else k))
    elif t == "array":
        if not isinstance(value, list):
            return [f"{path or 'arguments'} must be an array"]
        item_schema = schema.get("items") or {}
        for i, item in enumerate(value):
            errors.extend(_validate_schema(item, item_schema, f"{path}[{i}]"))
    elif t == "string":
        if not isinstance(value, str):
            errors.append(f"{path or 'arguments'} must be a string")
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path or 'arguments'} must be an integer")
    elif t == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path or 'arguments'} must be a number")
    elif t == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path or 'arguments'} must be a boolean")

    enum = schema.get("enum")
    if enum and value not in enum:
        errors.append(f"{path or 'arguments'} must be one of {enum}")

    return errors


# ======================================================================
# TASK ROUTING
# ======================================================================

# Which categories each intent cares about. Order matters only for logging.
INTENT_CATEGORIES: Dict[str, Set[str]] = {
    "feature":   {"inspect", "edit", "run", "proc", "vcs"},
    "bugfix":    {"inspect", "edit", "run", "proc", "vcs"},
    "refactor":  {"inspect", "edit", "vcs"},
    "analysis":  {"inspect", "run", "vcs"},
    "question":  {"inspect", "net"},
    "unknown":   {"inspect", "edit", "run", "proc", "vcs", "cloud", "net", "plan", "ui", "misc"},
}

# Fallback: when an intent's category set would leave out a tool that
# should always be visible (plan updates, todo).
ALWAYS_VISIBLE_CATEGORIES: Set[str] = {"plan", "ui"}


# ======================================================================
# REGISTRY
# ======================================================================

class ToolRegistry:
    """Central catalog of tools — the single authoritative source."""

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.tools: Dict[str, BaseTool] = {}
        self._external: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, ToolStats] = {}
        self._lock = asyncio.Lock()

        # Monotonic counter, incremented on every successful mutating call.
        # Consumers (loop) use this to decide when to drop caches.
        self.mutation_epoch: int = 0

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------

    async def register_tool(self, tool: BaseTool) -> None:
        if not getattr(tool, "name", None):
            raise ToolError("Tool must have a name")
        self.tools[tool.name] = tool
        self.stats.setdefault(tool.name, ToolStats())
        logger.debug("Registered tool: %s", tool.name)

    def unregister_tool(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def register_external(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        *,
        read_only: bool = False,
        mutating: bool = False,
        category: str = "misc",
    ) -> None:
        self._external[name] = {
            "handler": handler,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
            "read_only": bool(read_only),
            "mutating": bool(mutating),
            "category": category or "misc",
        }
        self.stats.setdefault(name, ToolStats())
        logger.debug("Registered external tool: %s", name)

    def list_tools(self, include_disabled: bool = False) -> List[str]:
        names = list(self.tools.keys()) + list(self._external.keys())
        if not include_disabled:
            names = [
                n for n in names
                if n in self._external or getattr(self.tools[n], "enabled", True)
            ]
        return sorted(names)

    # ------------------------------------------------------------------
    # METADATA ACCESSORS
    # ------------------------------------------------------------------

    def is_read_only(self, name: str) -> bool:
        if name in self._external:
            return bool(self._external[name].get("read_only"))
        t = self.tools.get(name)
        return bool(getattr(t, "read_only", False))

    def is_mutating(self, name: str) -> bool:
        if name in self._external:
            return bool(self._external[name].get("mutating"))
        t = self.tools.get(name)
        return bool(getattr(t, "mutating", False))

    def get_category(self, name: str) -> str:
        if name in self._external:
            return str(self._external[name].get("category") or "misc")
        t = self.tools.get(name)
        return str(getattr(t, "category", "misc"))

    # ------------------------------------------------------------------
    # SCHEMAS — single source of truth
    # ------------------------------------------------------------------

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Merged, deduped list of every callable tool schema."""
        out: List[Dict[str, Any]] = []
        seen = set()

        for name, tool in self.tools.items():
            if not getattr(tool, "enabled", True):
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(tool.to_schema())

        for name, info in self._external.items():
            if name in seen:
                continue
            seen.add(name)
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": info.get("parameters") or {
                        "type": "object", "properties": {}
                    },
                },
            })
        return out

    def select_for_task(self, intent: str) -> List[Dict[str, Any]]:
        """
        Return a narrowed tool list based on intent.

        Used by the loop to cut down the schema size (and improve local-
        model tool selection) without ever hiding a tool the runtime can
        still dispatch.
        """
        cats = set(INTENT_CATEGORIES.get(intent or "unknown", INTENT_CATEGORIES["unknown"]))
        cats |= ALWAYS_VISIBLE_CATEGORIES

        out: List[Dict[str, Any]] = []
        for schema in self.get_schemas():
            name = schema["function"]["name"]
            if self.get_category(name) in cats:
                out.append(schema)
        return out or self.get_schemas()

    def estimate_schema_tokens(self, schemas: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Estimated token cost of the tool list passed to the model on EVERY
        request. Callers subtract this from the context budget.
        """
        schemas = schemas if schemas is not None else self.get_schemas()
        if not schemas:
            return 0
        try:
            blob = json.dumps(schemas, default=str)
        except Exception:
            return 512  # conservative
        return max(1, len(blob) // 3)

    def describe_for_prompt(
        self, schemas: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Human-readable tool block for the system prompt."""
        schemas = schemas if schemas is not None else self.get_schemas()
        lines = []
        for schema in schemas:
            fn = schema["function"]
            lines.append(
                f"- {fn['name']}: {fn['description']}\n"
                f"  Parameters: {json.dumps(fn['parameters'], default=str)}"
            )
        return "\n".join(lines) if lines else "(no tools available)"

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        for schema in self.get_schemas():
            if schema["function"]["name"] == name:
                return schema
        return None

    def has_tool(self, name: str) -> bool:
        return name in self.tools or name in self._external

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_arguments(self, name: str, args: Any) -> Optional[str]:
        if not isinstance(args, dict):
            return f"arguments must be an object, got {type(args).__name__}"

        if name in self._external:
            schema = self._external[name].get("parameters") or {}
        else:
            tool = self.tools.get(name)
            if tool is None:
                matches = get_close_matches(name, self.list_tools(), n=1)
                hint = f" Did you mean '{matches[0]}'?" if matches else ""
                return (
                    f"Unknown tool '{name}'.{hint} "
                    f"Available: {', '.join(self.list_tools())}"
                )
            schema = tool.parameters or {}

        errors = _validate_schema(args, schema)
        if errors:
            return "Invalid arguments for '" + name + "': " + "; ".join(errors)
        return None

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def execute_safe(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        params = params or {}

        err = self.validate_arguments(name, params)
        if err:
            return {
                "success": False,
                "error": err,
                "recoverable": True,
                "invalid_arguments": True,
                "suggestion": (
                    "Re-emit the tool call with valid JSON matching the "
                    "declared parameters. Do not retry the exact same call."
                ),
            }

        if name in self._external:
            return await self._execute_external(name, params, timeout)

        tool = self.tools[name]
        if not getattr(tool, "enabled", True):
            return {
                "success": False,
                "error": f"Tool '{name}' is disabled",
                "recoverable": False,
            }

        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                tool.execute(params),
                timeout=timeout or getattr(tool, "timeout", 60.0),
            )
        except asyncio.TimeoutError:
            self._bump(name, success=False, elapsed=time.time() - t0,
                       error=f"timeout after {timeout or getattr(tool, 'timeout', 60.0)}s")
            return {
                "success": False,
                "error": f"Tool '{name}' timed out after "
                         f"{timeout or getattr(tool, 'timeout', 60.0)}s",
                "recoverable": True,
                "suggestion": "Reduce the scope of the request or raise the timeout.",
            }
        except Exception as e:
            self._bump(name, success=False, elapsed=time.time() - t0, error=str(e))
            logger.error("Tool '%s' raised: %s", name, e, exc_info=True)
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "recoverable": True,
                "suggestion": "Inspect the error and retry with corrected arguments.",
            }

        if not isinstance(result, dict):
            result = {"success": True, "result": result}
        if "success" not in result:
            result["success"] = True
        try:
            result = redact(result)
        except Exception:
            pass

        success = bool(result.get("success"))
        self._bump(
            name, success=success, elapsed=time.time() - t0,
            error=None if success else str(result.get("error", "")),
        )

        # Bump mutation epoch on any successful mutating call.
        if success and self.is_mutating(name):
            self.mutation_epoch += 1
            result.setdefault("_mutation_epoch", self.mutation_epoch)

        return result

    async def _execute_external(
        self, name: str, params: Dict[str, Any], timeout: Optional[float]
    ) -> Dict[str, Any]:
        info = self._external[name]
        handler = info["handler"]
        t0 = time.time()
        try:
            result = await asyncio.wait_for(handler(**params), timeout=timeout or 60.0)
        except asyncio.TimeoutError:
            self._bump(name, False, time.time() - t0, "timeout")
            return {"success": False, "error": f"Tool '{name}' timed out",
                    "recoverable": True}
        except TypeError as e:
            self._bump(name, False, time.time() - t0, str(e))
            return {
                "success": False,
                "error": f"Invalid arguments for '{name}': {e}",
                "recoverable": True,
                "invalid_arguments": True,
                "suggestion": "Check parameter names match the tool schema.",
            }
        except Exception as e:
            self._bump(name, False, time.time() - t0, str(e))
            return {"success": False, "error": f"{type(e).__name__}: {e}",
                    "recoverable": True}

        if not isinstance(result, dict):
            result = {"success": True, "result": result}
        if "success" not in result:
            result["success"] = True
        try:
            result = redact(result)
        except Exception:
            pass

        success = bool(result.get("success"))
        self._bump(name, success=success, elapsed=time.time() - t0,
                   error=None if success else str(result.get("error", "")))

        if success and self.is_mutating(name):
            self.mutation_epoch += 1
            result.setdefault("_mutation_epoch", self.mutation_epoch)

        return result

    async def execute_parallel(
        self,
        calls: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(5)

        async def _run(c: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                name = c.get("name") or c.get("tool")
                params = c.get("params") or c.get("arguments") or {}
                res = await self.execute_safe(name, params, timeout=timeout)
                return {"tool": name, **res}

        return await asyncio.gather(*[_run(c) for c in calls])

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def _bump(self, name: str, success: bool, elapsed: float,
              error: Optional[str]) -> None:
        s = self.stats.setdefault(name, ToolStats())
        s.calls += 1
        s.total_time += elapsed
        if success:
            s.successes += 1
        else:
            s.failures += 1
            s.last_error = error

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


__all__ = ["BaseTool", "ToolRegistry", "ToolStats", "INTENT_CATEGORIES"]