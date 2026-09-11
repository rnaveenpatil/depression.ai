"""
Task Tool - Spawn and manage subagent tasks for parallel work.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "task_id": {"type": "string"},
            "timeout": {"type": "number", "default": 300},
        },
        "required": ["action"],
    }
    timeout = 600.0

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.runs: Dict[str, TaskRun] = {}

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
            return self._cancel(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    async def _spawn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "spawn requires 'prompt'"}
        if self.agent is None:
            return {"success": False, "error": "no parent agent available"}

        tid = str(uuid.uuid4())[:8]
        run = TaskRun(id=tid, description=params.get("description", prompt[:60]))

        async def _run():
            try:
                result = await self.agent.process_query(prompt)
                run.result = result
            except Exception as e:
                run.error = str(e)
            finally:
                import time
                run.finished_at = time.time()

        run.task = asyncio.create_task(_run())
        self.runs[tid] = run
        return {"success": True, "task_id": tid, "description": run.description}

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
        timeout = float(params.get("timeout", 300))
        try:
            await asyncio.wait_for(r.task, timeout=timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout", "id": tid}
        return {"success": True, "id": tid, "result": r.result, "error": r.error}

    def _cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("task_id")
        r = self.runs.get(tid)
        if not r:
            return {"success": False, "error": f"Task {tid} not found"}
        if r.task and not r.task.done():
            r.task.cancel()
        return {"success": True, "id": tid}