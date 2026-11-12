"""
Task Tool - Spawn and manage subagent tasks for parallel work.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskRun:
    id: str
    description: str
    agent: Any = None
    task: Optional[asyncio.Task] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None


class TaskTool(BaseTool):
    name = "task"
    description = "Spawn a subagent to work on a self-contained task; inspect or wait for results."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["spawn", "list", "status", "wait", "cancel"]},
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "role": {"type": "string", "enum": ["general", "explore", "scout"]},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "task_id": {"type": "string"},
            "timeout": {"type": "number", "default": 300},
        },
        "required": ["action"],
    }
    timeout = 3600.0

    read_only = False
    mutating = True
    category = "misc"

    _MAX_RUNS = 32
    _MAX_COMPLETED = 100

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.runs: Dict[str, TaskRun] = {}
        self._shutdown = False

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "")
        if action == "spawn":
            return await self._spawn(params)
        if action == "list":
            return self._list()
        if action == "status":
            return self._status(params)
        if action == "wait":
            return await self._wait(params)
        if action == "cancel":
            return await self._cancel(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    async def _spawn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "spawn requires 'prompt'"}
        if self.agent is None:
            return {"success": False, "error": "no parent agent available"}
        if self._shutdown:
            return {"success": False, "error": "task tool is shutting down"}

        manager = getattr(self.agent, "subagent_manager", None)
        if manager is None:
            return {
                "success": False,
                "error": "subagent manager unavailable; refusing to run in the parent agent",
            }

        self._prune()
        running = sum(1 for r in self.runs.values() if r.task and not r.task.done())
        if running >= self._MAX_RUNS:
            return {
                "success": False,
                "error": f"too many running tasks (max {self._MAX_RUNS})",
            }

        role, role_error = self._resolve_role(params)
        if role_error:
            return {"success": False, "error": role_error}

        tid = str(uuid.uuid4())[:8]
        run = TaskRun(id=tid, description=params.get("description", prompt[:60]))
        run.started_at = time.time()

        async def _run():
            try:
                result = await manager.invoke_subagent(
                    role,
                    prompt,
                    {"description": run.description, "task_id": tid},
                )
                if result.get("success"):
                    run.result = result.get("result")
                else:
                    run.error = result.get("error", "subagent failed")
            except asyncio.CancelledError:
                run.error = "cancelled"
                raise
            except Exception as e:
                run.error = str(e)
            finally:
                run.finished_at = time.time()

        run.task = asyncio.create_task(_run(), name=f"task-{tid}")
        self.runs[tid] = run
        return {
            "success": True,
            "task_id": tid,
            "description": run.description,
            "role": role.value,
        }

    @staticmethod
    def _resolve_role(params: Dict[str, Any]) -> tuple[Any, Optional[str]]:
        from agent.agent.subagent import SubAgentRole

        requested = str(params.get("role") or "").strip().lower()
        if requested:
            roles = {
                "general": SubAgentRole.GENERAL,
                "explore": SubAgentRole.EXPLORE,
                "scout": SubAgentRole.SCOUT,
            }
            if requested not in roles:
                return None, f"unknown subagent role: {requested}"
            return roles[requested], None

        raw = params.get("capabilities") or []
        if isinstance(raw, str):
            raw = [raw]
        caps = {str(c).strip().lower() for c in raw if str(c).strip()}
        if caps & {
            "write", "edit", "patch", "terminal", "shell", "execute",
            "process", "general", "filesystem",
        }:
            return SubAgentRole.GENERAL, None
        if caps & {
            "research", "docs", "documentation", "dependency", "web", "browser",
            "scout",
        }:
            return SubAgentRole.SCOUT, None
        if caps and caps <= {
            "read", "readonly", "read-only", "search", "grep", "glob",
            "explore", "git", "list",
        }:
            return SubAgentRole.EXPLORE, None
        return SubAgentRole.GENERAL, None

    def _prune(self) -> None:
        completed = [
            (r.finished_at or 0.0, r.id)
            for r in self.runs.values()
            if r.task is not None and r.task.done()
        ]
        if len(completed) <= self._MAX_COMPLETED:
            return
        completed.sort()
        for _, tid in completed[: len(completed) - self._MAX_COMPLETED]:
            self.runs.pop(tid, None)

    def _list(self) -> Dict[str, Any]:
        return {
            "success": True,
            "tasks": [
                {
                    "id": r.id,
                    "description": r.description,
                    "running": r.task and not r.task.done(),
                    "has_error": bool(r.error),
                }
                for r in self.runs.values()
            ],
        }

    def _status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("task_id")
        r = self.runs.get(tid)
        if not r:
            return {"success": False, "error": f"Task {tid} not found"}
        return {
            "success": True,
            "id": r.id,
            "running": r.task and not r.task.done(),
            "result": r.result,
            "error": r.error,
        }

    async def _wait(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("task_id")
        r = self.runs.get(tid)
        if not r:
            return {"success": False, "error": f"Task {tid} not found"}
        try:
            timeout = max(0.1, min(float(params.get("timeout", 300)), 3600.0))
        except (TypeError, ValueError):
            return {"success": False, "error": "timeout must be a number"}
        if r.task is None:
            return {"success": False, "error": f"Task {tid} has no runner"}
        try:
            await asyncio.wait_for(asyncio.shield(r.task), timeout=timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout", "id": tid}
        return {"success": True, "id": tid, "result": r.result, "error": r.error}

    async def _cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("task_id")
        r = self.runs.get(tid)
        if not r:
            return {"success": False, "error": f"Task {tid} not found"}
        if r.task and not r.task.done():
            r.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(r.task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                pass
        return {"success": True, "id": tid, "cancelled": True}

    async def shutdown(self) -> None:
        self._shutdown = True
        pending = [r.task for r in self.runs.values() if r.task and not r.task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.runs.clear()
