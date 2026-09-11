"""
Todo Tool - Task planning and checklist management for multi-step work.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TodoItem:
    id: str
    title: str
    status: str = "pending"        # pending | in_progress | done | blocked | cancelled
    notes: str = ""
    priority: int = 3              # 1=critical, 5=low
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TodoTool(BaseTool):
    name = "todo"
    description = "Maintain a task checklist across the session (add, update, list, complete)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "update", "list", "complete", "delete", "clear", "summary"],
            },
            "title": {"type": "string"},
            "todo_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked", "cancelled"]},
            "notes": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["action"],
    }
    timeout = 10.0

    def __init__(self, session: Any = None):
        self.session = session
        self.items: Dict[str, TodoItem] = {}

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "")
        if action == "add":
            return self._add(params)
        if action == "update":
            return self._update(params)
        if action == "list":
            return self._list()
        if action == "complete":
            return self._complete(params)
        if action == "delete":
            return self._delete(params)
        if action == "clear":
            self.items.clear()
            return {"success": True, "cleared": True}
        if action == "summary":
            return self._summary()
        return {"success": False, "error": f"Unknown action: {action}"}

    def _add(self, params: Dict[str, Any]) -> Dict[str, Any]:
        title = params.get("title")
        if not title:
            return {"success": False, "error": "add requires 'title'"}
        tid = str(uuid.uuid4())[:8]
        item = TodoItem(
            id=tid,
            title=title,
            priority=int(params.get("priority", 3)),
            notes=params.get("notes", ""),
        )
        self.items[tid] = item
        return {"success": True, "todo_id": tid, "item": asdict(item)}

    def _update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("todo_id")
        item = self.items.get(tid)
        if not item:
            return {"success": False, "error": f"Todo {tid} not found"}
        if "title" in params and params["title"]:
            item.title = params["title"]
        if "status" in params and params["status"]:
            item.status = params["status"]
        if "notes" in params:
            item.notes = params["notes"]
        if "priority" in params:
            item.priority = int(params["priority"])
        item.updated_at = time.time()
        return {"success": True, "item": asdict(item)}

    def _complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("todo_id")
        item = self.items.get(tid)
        if not item:
            return {"success": False, "error": f"Todo {tid} not found"}
        item.status = "done"
        item.updated_at = time.time()
        return {"success": True, "todo_id": tid}

    def _delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("todo_id")
        if tid not in self.items:
            return {"success": False, "error": f"Todo {tid} not found"}
        del self.items[tid]
        return {"success": True, "todo_id": tid}

    def _list(self) -> Dict[str, Any]:
        items = sorted(
            self.items.values(),
            key=lambda i: (i.status != "in_progress", i.priority, i.created_at),
        )
        return {"success": True, "items": [asdict(i) for i in items], "count": len(items)}

    def _summary(self) -> Dict[str, Any]:
        stats: Dict[str, int] = {}
        for i in self.items.values():
            stats[i.status] = stats.get(i.status, 0) + 1
        return {"success": True, "stats": stats, "total": len(self.items)}