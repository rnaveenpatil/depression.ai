"""
Process Tool - Start, list, and manage long-running background processes.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# Maximum number of stdout/stderr lines retained per process.
_MAX_BUFFER_LINES = 5000


@dataclass
class ManagedProcess:
    id: str
    command: str
    pid: int
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.time)
    stdout: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    stderr: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    _pumps: List[asyncio.Task] = field(default_factory=list)


class ProcessTool(BaseTool):
    name = "process"
    description = (
        "Start, list, inspect, and stop background processes "
        "(dev servers, watchers, etc.)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "list", "status", "stop", "kill", "logs", "wait"],
            },
            "command": {"type": "string", "description": "For start"},
            "cwd": {"type": "string"},
            "process_id": {
                "type": "string",
                "description": "Tool-managed process id returned by 'start' (not the OS pid).",
            },
            "lines": {"type": "integer", "default": 100},
            "timeout": {"type": "number", "default": 30},
        },
        "required": ["action"],
    }
    timeout = 300.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()
        self.processes: Dict[str, ManagedProcess] = {}

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "")
        if action == "start":
            return await self._start(params)
        if action == "list":
            return self._list()
        if action == "status":
            return self._status(params)
        if action in ("stop", "kill"):
            return await self._stop(params, force=(action == "kill"))
        if action == "logs":
            return self._logs(params)
        if action == "wait":
            return await self._wait(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    async def _start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = params.get("command")
        if not command:
            return {"success": False, "error": "start requires 'command'"}

        cwd = params.get("cwd") or self.cwd
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        pid = str(uuid.uuid4())[:8]
        mp = ManagedProcess(id=pid, command=command, pid=proc.pid or 0, proc=proc)
        self.processes[pid] = mp

        async def pump(stream: Any, buf: Deque[str], tag: str) -> None:
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    buf.append(line.decode(errors="replace").rstrip())
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("process pump %s error: %s", tag, e)

        mp._pumps = [
            asyncio.create_task(pump(proc.stdout, mp.stdout, "out")),
            asyncio.create_task(pump(proc.stderr, mp.stderr, "err")),
        ]

        return {
            "success": True,
            "process_id": pid,
            "os_pid": proc.pid,
            "command": command,
            "note": "Use process_id (not os_pid) with the 'process' tool.",
        }

    def _list(self) -> Dict[str, Any]:
        return {
            "success": True,
            "processes": [
                {
                    "id": mp.id,
                    "os_pid": mp.pid,
                    "command": mp.command,
                    "running": mp.proc.returncode is None,
                    "returncode": mp.proc.returncode,
                    "started_at": mp.started_at,
                }
                for mp in self.processes.values()
            ],
        }

    def _get(self, params: Dict[str, Any]) -> Optional[ManagedProcess]:
        pid = params.get("process_id")
        if not pid:
            return None
        return self.processes.get(pid)

    def _status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
            }
        return {
            "success": True,
            "id": mp.id,
            "os_pid": mp.pid,
            "running": mp.proc.returncode is None,
            "returncode": mp.proc.returncode,
            "uptime": time.time() - mp.started_at,
            "stdout_lines": len(mp.stdout),
            "stderr_lines": len(mp.stderr),
        }

    async def _stop(
        self, params: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
            }

        returncode = mp.proc.returncode
        if returncode is None:
            try:
                if force:
                    mp.proc.kill()
                else:
                    mp.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(mp.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    mp.proc.kill()
                    await asyncio.wait_for(mp.proc.wait(), timeout=2)
                except Exception:
                    pass

        for t in mp._pumps:
            t.cancel()
        for t in mp._pumps:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # Remove from the live registry so it does not leak.
        self.processes.pop(mp.id, None)

        return {
            "success": True,
            "id": mp.id,
            "os_pid": mp.pid,
            "killed": force,
            "returncode": mp.proc.returncode,
        }

    def _logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
            }
        n = int(params.get("lines", 100))
        return {
            "success": True,
            "id": mp.id,
            "stdout": "\n".join(list(mp.stdout)[-n:]),
            "stderr": "\n".join(list(mp.stderr)[-n:]),
        }

    async def _wait(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
            }
        timeout = float(params.get("timeout", 30))
        try:
            code = await asyncio.wait_for(mp.proc.wait(), timeout=timeout)
            return {"success": True, "id": mp.id, "returncode": code}
        except asyncio.TimeoutError:
            return {"success": False, "error": "still running", "id": mp.id}

    async def shutdown(self) -> None:
        for pid in list(self.processes.keys()):
            try:
                await self._stop({"process_id": pid}, force=True)
            except Exception:
                pass