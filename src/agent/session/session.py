"""
Session and SessionManager.

Session         — a single conversation with id, name, messages, state.
SessionManager  — create / load / save / list / delete sessions on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import SessionError
from agent.session.history import SessionHistory, HistoryEntry
from agent.session.state import SessionState, SessionStatus

logger = get_logger(__name__)


# ======================================================================
# SESSION
# ======================================================================

class Session:
    """
    A single conversation session.

    Holds:
        - identity (id, name, created_at, updated_at)
        - history (SessionHistory)
        - runtime state (SessionState)
        - optional context snapshot
        - metadata / tags
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_messages: int = 5000,
    ):
        self.id: str = session_id or str(uuid.uuid4())
        self.name: str = name or "session"
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.metadata: Dict[str, Any] = metadata or {}

        self.history = SessionHistory(max_entries=max_messages)
        self.state = SessionState()

        # Optional: snapshot of context
        self.context_snapshot: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # MESSAGES
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        tokens: int = 0,
        pinned: bool = False,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        entry = self.history.append(
            role=role,
            content=content,
            tokens=tokens,
            pinned=pinned,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )
        self.touch()
        return entry

    def add_user_message(self, content: str, tokens: int = 0) -> HistoryEntry:
        return self.add_message("user", content, tokens=tokens)

    def add_assistant_message(self, content: str, tokens: int = 0) -> HistoryEntry:
        return self.add_message("assistant", content, tokens=tokens)

    def add_system_message(self, content: str, tokens: int = 0, pinned: bool = True) -> HistoryEntry:
        return self.add_message("system", content, tokens=tokens, pinned=pinned)

    def add_tool_message(
        self,
        content: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tokens: int = 0,
    ) -> HistoryEntry:
        return self.add_message(
            "tool",
            content,
            tokens=tokens,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def touch(self) -> None:
        self.updated_at = time.time()
        self.state.touch()

    def set_status(self, status: str | SessionStatus) -> None:
        self.state.set_status(status)
        self.touch()

    def add_usage(self, tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0) -> None:
        self.state.add_usage(tokens_in, tokens_out, cost)
        self.touch()

    def update_context(self, snapshot: Dict[str, Any]) -> None:
        self.context_snapshot = snapshot
        self.touch()

    def reset(self) -> None:
        self.history.clear()
        self.state.reset()
        self.context_snapshot = {}
        self.touch()

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "state": self.state.to_dict(),
            "context_snapshot": self.context_snapshot,
            "message_count": len(self.history),
            "token_total": self.history.total_tokens(),
        }
        if include_history:
            d["history"] = self.history.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        session = cls(
            session_id=data.get("id"),
            name=data.get("name", "session"),
            metadata=data.get("metadata", {}),
        )
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.context_snapshot = data.get("context_snapshot", {})
        if "state" in data and isinstance(data["state"], dict):
            session.state = SessionState.from_dict(data["state"])
        if "history" in data:
            session.history.from_dict(data["history"])
        return session

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": len(self.history),
            "tokens": self.history.total_tokens(),
            "status": self.state.status,
            "model": self.state.model,
        }

    def __repr__(self) -> str:
        return (
            f"<Session id={self.id[:8]} name={self.name!r} "
            f"msgs={len(self.history)} state={self.state.status}>"
        )


# ======================================================================
# SESSION MANAGER
# ======================================================================

class SessionManager:
    """
    Create, load, save, list, and delete sessions on disk.

    Storage layout:
        ~/.agent/sessions/
            index.json             — fast listing (id, name, updated_at, ...)
            <session_id>.json      — full session
            <session_id>.jsonl     — optional streaming history export
    """

    def __init__(
        self,
        database: Any = None,                      # optional persistent DB
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.config = cfg
        self.database = database

        storage_dir = os.path.expanduser(
            cfg.get("storage_dir", "~/.agent/sessions")
        )
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "index.json"

        self.auto_save: bool = cfg.get("auto_save", True)
        self.save_interval: float = cfg.get("save_interval", 30.0)
        self.max_sessions: int = cfg.get("max_sessions", 100)
        self.max_messages_per_session: int = cfg.get("max_messages_per_session", 5000)
        self.default_name: str = cfg.get("default_name", "session")

        self._current: Optional[Session] = None
        self._last_save: float = 0.0
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

        logger.info(
            f"SessionManager initialized (dir={self.storage_dir}, "
            f"auto_save={self.auto_save}, sessions={len(self._index)})"
        )

    # ------------------------------------------------------------------
    # INDEX
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        self._index = {}
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._index = data.get("sessions", {})
                elif isinstance(data, list):
                    self._index = {s["id"]: s for s in data if "id" in s}
            except Exception as e:
                logger.warning(f"Failed to load session index: {e}")

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"sessions": self._index}, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save session index: {e}")

    def _path_for(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.storage_dir / f"{safe}.json"

    # ------------------------------------------------------------------
    # CREATE / LOAD / SAVE
    # ------------------------------------------------------------------

    async def create_session(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        session = Session(
            name=name or self.default_name,
            metadata=metadata or {},
            max_messages=self.max_messages_per_session,
        )
        self._current = session
        self._index[session.id] = session.to_summary()
        self._save_index()
        await self.save_current_session()
        logger.info(f"Created session: {session.id}")
        return session

    async def load_session(self, session_id: str) -> Optional[Session]:
        path = self._path_for(session_id)

        # Try database first
        if self.database is not None:
            try:
                data = await self.database.load_session(session_id)
                if data:
                    session = Session.from_dict(data)
                    self._current = session
                    logger.info(f"Loaded session from DB: {session_id}")
                    return session
            except Exception as e:
                logger.debug(f"DB load failed: {e}")

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data)
            self._current = session
            logger.info(f"Loaded session: {session_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def save_current_session(self) -> bool:
        if not self._current:
            return False
        return await self.save_session(self._current)

    async def save_session(self, session: Session) -> bool:
        session.touch()
        path = self._path_for(session.id)

        try:
            data = session.to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")
            return False

        # Update index
        self._index[session.id] = session.to_summary()
        self._save_index()

        # Optional DB mirror
        if self.database is not None:
            try:
                await self.database.save_session(session.to_dict())
            except Exception as e:
                logger.debug(f"DB save failed: {e}")

        self._last_save = time.time()
        self._enforce_max_sessions()
        logger.debug(f"Saved session: {session.id}")
        return True

    async def autosave(self) -> bool:
        if not self.auto_save or not self._current:
            return False
        if time.time() - self._last_save >= self.save_interval:
            return await self.save_current_session()
        return False

    # ------------------------------------------------------------------
    # LIST / DELETE
    # ------------------------------------------------------------------

    async def list_sessions(self) -> List[Dict[str, Any]]:
        out = []
        for sid, info in self._index.items():
            item = dict(info)
            item["id"] = sid
            item["current"] = bool(self._current and self._current.id == sid)
            out.append(item)
        out.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return out

    async def delete_session(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        removed = False
        try:
            if path.exists():
                path.unlink()
                removed = True
        except Exception as e:
            logger.warning(f"Failed to delete session file: {e}")

        self._index.pop(session_id, None)
        self._save_index()

        if self._current and self._current.id == session_id:
            self._current = None

        if self.database is not None:
            try:
                await self.database.delete_session(session_id)
            except Exception:
                pass

        logger.info(f"Deleted session: {session_id} (removed={removed})")
        return removed

    async def delete_all_sessions(self) -> int:
        count = 0
        for sid in list(self._index.keys()):
            if await self.delete_session(sid):
                count += 1
        return count

    def _enforce_max_sessions(self) -> None:
        if self.max_sessions <= 0 or len(self._index) <= self.max_sessions:
            return
        sorted_sids = sorted(
            self._index.keys(),
            key=lambda s: self._index[s].get("updated_at", 0),
        )
        for sid in sorted_sids[: len(self._index) - self.max_sessions]:
            try:
                p = self._path_for(sid)
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            self._index.pop(sid, None)
        self._save_index()

    # ------------------------------------------------------------------
    # CURRENT SESSION
    # ------------------------------------------------------------------

    async def get_or_create_session(self) -> Session:
        if self._current:
            return self._current
        # Try to load the most recently updated session
        sessions = await self.list_sessions()
        if sessions:
            loaded = await self.load_session(sessions[0]["id"])
            if loaded:
                return loaded
        return await self.create_session()

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current.id if self._current else None

    def current(self) -> Optional[Session]:
        return self._current

    def set_current(self, session: Session) -> None:
        self._current = session

    async def get_current_info(self) -> Dict[str, Any]:
        if not self._current:
            return {"id": None, "state": "none"}
        s = self._current
        return {
            "id": s.id,
            "name": s.name,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "messages": len(s.history),
            "tokens": s.history.total_tokens(),
            "status": s.state.status,
            "model": s.state.model,
            "provider": s.state.provider,
            "cost": s.state.cost,
            "turn": s.state.turn,
        }

    async def reset_current(self) -> bool:
        if not self._current:
            return False
        self._current.reset()
        await self.save_current_session()
        return True

    # ------------------------------------------------------------------
    # HISTORY / MESSAGES
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, Any]]:
        if not self._current:
            return []
        return [e.to_dict() for e in self._current.history]

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self._current:
            return []
        return self._current.history.to_messages(limit=limit)

    # ------------------------------------------------------------------
    # EXPORT / IMPORT
    # ------------------------------------------------------------------

    async def export_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        session = self._current
        if session_id:
            session = await self.load_session(session_id)
        if not session:
            raise SessionError("No session to export")
        return session.to_dict()

    async def import_session(self, data: Dict[str, Any]) -> Session:
        session = Session.from_dict(data)
        self._current = session
        await self.save_session(session)
        logger.info(f"Imported session: {session.id}")
        return session

    # ------------------------------------------------------------------
    # SUMMARY FOR PROMPT
    # ------------------------------------------------------------------

    async def as_summary(self) -> str:
        if not self._current:
            return "No active session."
        s = self._current
        return (
            f"Session {s.id[:8]} — {s.name} "
            f"({len(s.history)} msgs, {s.history.total_tokens()} tokens, "
            f"state={s.state.status})"
        )

    def __repr__(self) -> str:
        cur = self._current.id[:8] if self._current else "-"
        return f"<SessionManager sessions={len(self._index)} current={cur}>"