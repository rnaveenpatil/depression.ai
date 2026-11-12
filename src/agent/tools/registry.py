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
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.utils.errors import ToolError
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
    #   destructive True  => may delete/overwrite user data
    #   category    one of: "inspect", "edit", "run", "proc", "vcs",
    #               "cloud", "net", "plan", "ui", "misc"
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
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

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path or 'arguments'} must be >= {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path or 'arguments'} must be <= {maximum}")
    if isinstance(value, str):
        if schema.get("minLength") is not None and len(value) < schema["minLength"]:
            errors.append(f"{path or 'arguments'} is too short")
        if schema.get("maxLength") is not None and len(value) > schema["maxLength"]:
            errors.append(f"{path or 'arguments'} is too long")
    if isinstance(value, list):
        if schema.get("minItems") is not None and len(value) < schema["minItems"]:
            errors.append(f"{path or 'arguments'} needs at least {schema['minItems']} items")
        if schema.get("maxItems") is not None and len(value) > schema["maxItems"]:
            errors.append(f"{path or 'arguments'} allows at most {schema['maxItems']} items")

    return errors


# ======================================================================
# TASK ROUTING
# ======================================================================

# ======================================================================
# OUTPUT CAP
# ======================================================================

MAX_TOOL_OUTPUT_BYTES = 256_000
_TOOL_PREVIEW_BYTES = 96_000


def _cap_tool_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Bound the payload returned to the model/context.

    Small results are untouched. Huge results are replaced by a compact
    preview with success/error metadata preserved.
    """
    try:
        blob = json.dumps(result, default=str)
    except Exception:
        return result
    if len(blob.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES:
        return result

    small: Dict[str, Any] = {}
    for key in ("success", "error", "path", "tool", "exit_code", "status",
                "truncated", "truncated_reason", "count", "size"):
        if key in result:
            small[key] = result[key]
    small["truncated"] = True
    small["truncated_reason"] = "tool output exceeded limit"
    small["output_preview"] = blob[:_TOOL_PREVIEW_BYTES]
    return small


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
        self._hidden_tools: set[str] = set()
        self._lock = threading.RLock()

        # Monotonic counter, incremented on every successful mutating call.
        # Consumers (loop) use this to decide when to drop caches.
        self.mutation_epoch: int = 0

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------

    async def register_tool(self, tool: BaseTool) -> None:
        if not getattr(tool, "name", None):
            raise ToolError("Tool must have a name")
        with self._lock:
            self.tools[tool.name] = tool
            self.stats.setdefault(tool.name, ToolStats())
        logger.debug("Registered tool: %s", tool.name)

    def unregister_tool(self, name: str) -> bool:
        with self._lock:
            removed = self.tools.pop(name, None) is not None
            removed = self._external.pop(name, None) is not None or removed
            if removed:
                self.stats.pop(name, None)
            self._hidden_tools.discard(name)
        return removed

    def hide_tool(self, name: str) -> bool:
        """Hide a registered alias from model-facing schemas.

        The tool remains executable for backward compatibility, but duplicate
        aliases no longer consume prompt tokens or confuse tool selection.
        """
        with self._lock:
            if name not in self.tools and name not in self._external:
                return False
            self._hidden_tools.add(name)
            return True

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
        destructive: bool = False,
    ) -> None:
        with self._lock:
            self._external[name] = {
                "handler": handler,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}},
                "read_only": bool(read_only),
                "mutating": bool(mutating),
                "destructive": bool(destructive),
                "category": category or "misc",
            }
            self.stats.setdefault(name, ToolStats())
        logger.debug("Registered external tool: %s", name)

    def list_tools(self, include_disabled: bool = False) -> List[str]:
        with self._lock:
            tools = dict(self.tools)
            external = dict(self._external)
        names = list(tools.keys()) + list(external.keys())
        if not include_disabled:
            names = [
                n for n in names
                if n in external or getattr(tools[n], "enabled", True)
            ]
        return sorted(names)

    # ------------------------------------------------------------------
    # METADATA ACCESSORS
    # ------------------------------------------------------------------

    def is_read_only(self, name: str) -> bool:
        t = self.tools.get(name)
        if t is not None:
            return bool(getattr(t, "read_only", False))
        ext = self._external.get(name)
        return bool(ext.get("read_only")) if ext else False

    def is_mutating(self, name: str) -> bool:
        t = self.tools.get(name)
        if t is not None:
            return bool(getattr(t, "mutating", False))
        ext = self._external.get(name)
        return bool(ext.get("mutating")) if ext else False

    def is_destructive(self, name: str) -> bool:
        t = self.tools.get(name)
        if t is not None:
            return bool(getattr(t, "destructive", False))
        ext = self._external.get(name)
        return bool(ext.get("destructive")) if ext else False

    def get_category(self, name: str) -> str:
        t = self.tools.get(name)
        if t is not None:
            return str(getattr(t, "category", "misc"))
        ext = self._external.get(name)
        return str(ext.get("category") or "misc") if ext else "misc"

    # ------------------------------------------------------------------
    # SCHEMAS — single source of truth
    # ------------------------------------------------------------------

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Merged, deduped list of every callable tool schema."""
        out: List[Dict[str, Any]] = []
        seen = set()
        with self._lock:
            tools = dict(self.tools)
            external = dict(self._external)

        for name, tool in tools.items():
            if name in self._hidden_tools or not getattr(tool, "enabled", True):
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(tool.to_schema())

        for name, info in external.items():
            if name in self._hidden_tools or name in seen:
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
        # Intent is advisory, not an authority to hide tools. The model
        # can change its approach as it learns more about the user's goal.
        # Permission checks remain in the execution path, not here.
        return self.get_schemas()

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

        tool = self.tools.get(name)
        if tool is not None:
            schema = tool.parameters or {}
        elif name in self._external:
            schema = self._external[name].get("parameters") or {}
        else:
            matches = get_close_matches(name, self.list_tools(), n=1)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            return (
                f"Unknown tool '{name}'.{hint} "
                f"Available: {', '.join(self.list_tools())}"
            )

        errors = _validate_schema(args, schema)
        if errors:
            return "Invalid arguments for '" + name + "': " + "; ".join(errors)
        return None

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compatibility entry point used by the main agent and subagents."""
        return await self.execute_safe(name, params, timeout=timeout)

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

        tool = self.tools.get(name)
        if tool is None:
            if name in self._external:
                return await self._execute_external(name, params, timeout)
            return {
                "success": False,
                "error": f"Unknown tool '{name}'",
                "recoverable": True,
            }
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
            message = str(redact({"error": str(e)}).get("error", str(e)))
            self._bump(name, success=False, elapsed=time.time() - t0, error=message)
            logger.error("Tool '%s' raised: %s", name, e, exc_info=True)
            return {
                "success": False,
                "error": f"{type(e).__name__}: {message}",
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
        result = _cap_tool_output(result)

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
            message = str(redact({"error": str(e)}).get("error", str(e)))
            self._bump(name, False, time.time() - t0, message)
            return {
                "success": False,
                "error": f"Invalid arguments for '{name}': {message}",
                "recoverable": True,
                "invalid_arguments": True,
                "suggestion": "Check parameter names match the tool schema.",
            }
        except Exception as e:
            message = str(redact({"error": str(e)}).get("error", str(e)))
            self._bump(name, False, time.time() - t0, message)
            return {"success": False, "error": f"{type(e).__name__}: {message}",
                    "recoverable": True}

        if not isinstance(result, dict):
            result = {"success": True, "result": result}
        if "success" not in result:
            result["success"] = True
        try:
            result = redact(result)
        except Exception:
            pass
        result = _cap_tool_output(result)

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
                try:
                    res = await self.execute_safe(name, params, timeout=timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    res = {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "recoverable": True,
                    }
                if not isinstance(res, dict):
                    res = {"success": True, "result": res}
                return {**res, "tool": name}

        results = await asyncio.gather(
            *[_run(c) for c in calls], return_exceptions=True
        )
        normalized: List[Dict[str, Any]] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                name = calls[index].get("name") or calls[index].get("tool")
                normalized.append({
                    "success": False,
                    "tool": name,
                    "error": f"{type(result).__name__}: {result}",
                    "recoverable": True,
                })
            else:
                normalized.append(result)
        return normalized

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


__all__ = ["BaseTool", "ToolRegistry", "ToolStats"]