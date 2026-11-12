"""
Todo Tool - Task planning and checklist management for multi-step work.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

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
            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "completed", "blocked", "cancelled"]},
            "notes": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["action"],
    }
    timeout = 99999.0

    read_only = False
    mutating = True
    category = "plan"

    # Shared store per session so `todo`, `todowrite`, and `todoread` see the
    # same data. Plain dict — session-scoped cleanup is handled by
    # forget_session(), not by weak references.
    _shared_stores: Dict[Any, Dict[str, TodoItem]] = {}
    # Alias kept for compatibility with callers that read _strong_keys.
    _strong_keys: Dict[Any, Dict[str, TodoItem]] = _shared_stores

    @staticmethod
    def _session_key(session: Any) -> Any:
        if session is None:
            return "__default__"
        sid = getattr(session, "id", None)
        if isinstance(sid, str) and sid:
            return sid
        return id(session)

    def __init__(self, session: Any = None):
        self.session = session
        sid = self._session_key(session)
        if sid not in self._strong_keys:
            self._strong_keys[sid] = {}
        # self.items is a reference to the shared dict for this session
        self.items: Dict[str, TodoItem] = self._strong_keys[sid]

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

    @staticmethod
    def _normalize_status(value: Any) -> Optional[str]:
        status = str(value or "")
        if status == "completed":
            return "done"
        if status in ("pending", "in_progress", "done", "blocked", "cancelled"):
            return status
        return None

    @staticmethod
    def _priority(value: Any) -> int:
        try:
            return max(1, min(int(value), 5))
        except (TypeError, ValueError):
            return 3

    @staticmethod
    def _public_item(item: TodoItem) -> Dict[str, Any]:
        data = asdict(item)
        if data.get("status") == "done":
            data["status"] = "completed"
        return data

    def _add(self, params: Dict[str, Any]) -> Dict[str, Any]:
        title = params.get("title")
        if not title or not isinstance(title, str):
            return {"success": False, "error": "add requires 'title'"}
        status = self._normalize_status(params.get("status", "pending"))
        if status is None:
            return {"success": False, "error": "invalid status"}
        tid = str(uuid.uuid4())[:8]
        item = TodoItem(
            id=tid,
            title=title[:2000],
            status=status,
            priority=self._priority(params.get("priority", 3)),
            notes=str(params.get("notes", ""))[:4000],
        )
        self.items[tid] = item
        return {"success": True, "todo_id": tid, "item": self._public_item(item)}

    def _update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tid = params.get("todo_id")
        item = self.items.get(tid)
        if not item:
            return {"success": False, "error": f"Todo {tid} not found"}
        if "title" in params and params["title"]:
            item.title = str(params["title"])[:2000]
        if "status" in params and params["status"]:
            status = self._normalize_status(params["status"])
            if status is None:
                return {"success": False, "error": "invalid status"}
            item.status = status
        if "notes" in params:
            item.notes = str(params["notes"])[:4000]
        if "priority" in params:
            item.priority = self._priority(params["priority"])
        item.updated_at = time.time()
        return {"success": True, "item": self._public_item(item)}

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
        out = [self._public_item(item) for item in items]
        return {"success": True, "items": out, "count": len(items)}

    def _summary(self) -> Dict[str, Any]:
        stats: Dict[str, int] = {}
        for i in self.items.values():
            status = "completed" if i.status == "done" else i.status
            stats[status] = stats.get(status, 0) + 1
        return {"success": True, "stats": stats, "total": len(self.items)}

    @classmethod
    def forget_session(cls, session: Any) -> None:
        """Called on session teardown so the shared store doesn't leak."""
        cls._strong_keys.pop(cls._session_key(session), None)