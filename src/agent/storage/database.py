"""
Database - SQLite-backed persistence for the agent.

Stores:
    - sessions             (id, name, metadata, timestamps)
    - messages             (per-session history)
    - agent_state          (per-session runtime state snapshots)
    - tool_outputs         (cache of tool calls for context replay)
    - kv                   (arbitrary key/value for plugins & misc)
    - plans                (persisted plans from the planner)

Features:
    - Async API wrapping sqlite3 (executed in a thread pool)
    - WAL mode for concurrent reads
    - Automatic schema migration
    - Full-text search on messages
    - Backup / restore helpers
    - Safe by default (foreign keys, transactions)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import StorageError

logger = get_logger(__name__)


SCHEMA_VERSION = 1


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class SessionRecord:
    id: str
    name: str = "session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    token_total: int = 0


@dataclass
class AgentStateRecord:
    session_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    saved_at: float = field(default_factory=time.time)


# ======================================================================
# DATABASE
# ======================================================================

class Database:
    """
    Async SQLite wrapper.

    Usage:
        db = Database("~/.agent/agent.db")
        await db.initialize()
        await db.save_session({...})
    """

    def __init__(self, path: str | Path = "~/.agent/agent.db"):
        self.path = Path(os.path.expanduser(str(path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._open)
        await asyncio.to_thread(self._migrate)
        self._initialized = True
        logger.info(f"Database ready: {self.path}")

    def _open(self) -> None:
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,   # autocommit
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA temp_store=MEMORY")

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT,
            state TEXT,
            context_snapshot TEXT,
            message_count INTEGER DEFAULT 0,
            token_total INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions(updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0,
            tool_name TEXT,
            tool_call_id TEXT,
            metadata TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, id);
        CREATE INDEX IF NOT EXISTS idx_messages_created
            ON messages(created_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(content, content='messages', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES('delete', old.id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES('delete', old.id, old.content);
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TABLE IF NOT EXISTS agent_state (
            session_id TEXT PRIMARY KEY,
            state TEXT,
            saved_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tool_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            tool TEXT NOT NULL,
            params TEXT,
            result TEXT,
            success INTEGER DEFAULT 1,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tool_outputs_session
            ON tool_outputs(session_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            goal TEXT,
            data TEXT,
            status TEXT,
            created_at REAL,
            updated_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_plans_session
            ON plans(session_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL NOT NULL
        );
        """)

        # Record schema version
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        logger.debug(f"Schema migrated to version {SCHEMA_VERSION}")

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
        self._initialized = False
        logger.debug("Database closed")

    # ------------------------------------------------------------------
    # LOW-LEVEL HELPERS
    # ------------------------------------------------------------------

    async def _execute(self, sql: str, params: Tuple = ()) -> None:
        if not self._conn:
            raise StorageError("Database not initialized")
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, params)

    def _execute_sync(self, sql: str, params: Tuple = ()) -> None:
        self._conn.execute(sql, params)

    async def _fetchall(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        if not self._conn:
            raise StorageError("Database not initialized")
        async with self._lock:
            return await asyncio.to_thread(self._fetchall_sync, sql, params)

    def _fetchall_sync(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    async def _fetchone(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        if not self._conn:
            raise StorageError("Database not initialized")
        async with self._lock:
            return await asyncio.to_thread(self._fetchone_sync, sql, params)

    def _fetchone_sync(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchone()

    @staticmethod
    def _to_json(obj: Any) -> str:
        try:
            return json.dumps(obj or {}, default=str)
        except Exception:
            return "{}"

    @staticmethod
    def _from_json(text: Optional[str]) -> Any:
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # SESSIONS
    # ------------------------------------------------------------------

    async def save_session(self, session: Dict[str, Any]) -> None:
        sid = session.get("id")
        if not sid:
            raise StorageError("Session missing id")

        created = float(session.get("created_at") or time.time())
        updated = float(session.get("updated_at") or time.time())
        name = session.get("name", "session")
        metadata = self._to_json(session.get("metadata"))
        state = self._to_json(session.get("state"))
        ctx = self._to_json(session.get("context_snapshot"))

        history = session.get("history") or []
        msg_count = session.get("message_count", len(history))
        token_total = session.get("token_total") or sum(
            int(m.get("tokens") or 0) for m in history
        )

        await self._execute("""
            INSERT INTO sessions
              (id, name, created_at, updated_at, metadata, state,
               context_snapshot, message_count, token_total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name,
              updated_at=excluded.updated_at,
              metadata=excluded.metadata,
              state=excluded.state,
              context_snapshot=excluded.context_snapshot,
              message_count=excluded.message_count,
              token_total=excluded.token_total
        """, (sid, name, created, updated, metadata, state, ctx, msg_count, token_total))

        # Replace messages
        await self._execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        for m in history:
            await self._execute("""
                INSERT INTO messages
                  (session_id, role, content, tokens, pinned,
                   tool_name, tool_call_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sid,
                m.get("role", "user"),
                m.get("content", ""),
                int(m.get("tokens") or 0),
                1 if m.get("pinned") else 0,
                m.get("tool_name"),
                m.get("tool_call_id"),
                self._to_json(m.get("metadata")),
                float(m.get("timestamp") or time.time()),
            ))

    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if not row:
            return None

        msgs = await self._fetchall(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        history = []
        for m in msgs:
            history.append({
                "id": f"{m['id']}",
                "role": m["role"],
                "content": m["content"],
                "tokens": m["tokens"],
                "pinned": bool(m["pinned"]),
                "tool_name": m["tool_name"],
                "tool_call_id": m["tool_call_id"],
                "metadata": self._from_json(m["metadata"]),
                "timestamp": m["created_at"],
            })

        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": self._from_json(row["metadata"]),
            "state": self._from_json(row["state"]),
            "context_snapshot": self._from_json(row["context_snapshot"]),
            "message_count": row["message_count"],
            "token_total": row["token_total"],
            "history": history,
        }

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = await self._fetchall("""
            SELECT id, name, created_at, updated_at, message_count, token_total
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in rows]

    async def delete_session(self, session_id: str) -> bool:
        await self._execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    async def session_exists(self, session_id: str) -> bool:
        row = await self._fetchone("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,))
        return row is not None

    # ------------------------------------------------------------------
    # AGENT STATE
    # ------------------------------------------------------------------

    async def save_agent_state(self, state: Dict[str, Any]) -> None:
        sid = state.get("session_id")
        if not sid:
            raise StorageError("Agent state missing session_id")
        payload = self._to_json(state)
        saved_at = float(state.get("saved_at") or time.time())

        # Ensure session row exists (create if needed)
        if not await self.session_exists(sid):
            await self._execute("""
                INSERT INTO sessions (id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (sid, "session", time.time(), time.time()))

        await self._execute("""
            INSERT INTO agent_state (session_id, state, saved_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              state=excluded.state,
              saved_at=excluded.saved_at
        """, (sid, payload, saved_at))

    async def load_agent_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetchone(
            "SELECT state, saved_at FROM agent_state WHERE session_id = ?",
            (session_id,),
        )
        if not row:
            return None
        data = self._from_json(row["state"])
        data.setdefault("session_id", session_id)
        data["saved_at"] = row["saved_at"]
        return data

    # ------------------------------------------------------------------
    # TOOL OUTPUTS
    # ------------------------------------------------------------------

    async def save_tool_output(
        self,
        session_id: str,
        tool: str,
        params: Dict[str, Any],
        result: Any,
        success: bool = True,
    ) -> None:
        await self._execute("""
            INSERT INTO tool_outputs
              (session_id, tool, params, result, success, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            tool,
            self._to_json(params),
            self._to_json(result),
            1 if success else 0,
            time.time(),
        ))

    async def recent_tool_outputs(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        rows = await self._fetchall("""
            SELECT tool, params, result, success, created_at
            FROM tool_outputs
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit))
        return [
            {
                "tool": r["tool"],
                "params": self._from_json(r["params"]),
                "result": self._from_json(r["result"]),
                "success": bool(r["success"]),
                "timestamp": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # PLANS
    # ------------------------------------------------------------------

    async def save_plan(self, plan_id: str, data: Dict[str, Any]) -> None:
        session_id = data.get("session_id")
        goal = data.get("goal", "")
        status = data.get("status", "active")
        now = time.time()

        await self._execute("""
            INSERT INTO plans (id, session_id, goal, data, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              data=excluded.data,
              status=excluded.status,
              updated_at=excluded.updated_at
        """, (
            plan_id,
            session_id,
            goal,
            self._to_json(data),
            status,
            float(data.get("created_at") or now),
            now,
        ))

    async def load_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetchone("SELECT data FROM plans WHERE id = ?", (plan_id,))
        return self._from_json(row["data"]) if row else None

    async def list_plans(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        rows = await self._fetchall("""
            SELECT id, goal, status, updated_at
            FROM plans
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (session_id, limit))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # KEY-VALUE
    # ------------------------------------------------------------------

    async def set(self, key: str, value: Any) -> None:
        await self._execute("""
            INSERT INTO kv (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value,
              updated_at=excluded.updated_at
        """, (key, self._to_json(value), time.time()))

    async def get(self, key: str, default: Any = None) -> Any:
        row = await self._fetchone("SELECT value FROM kv WHERE key = ?", (key,))
        if not row:
            return default
        return self._from_json(row["value"])

    async def delete(self, key: str) -> bool:
        await self._execute("DELETE FROM kv WHERE key = ?", (key,))
        return True

    async def keys(self, prefix: str = "") -> List[str]:
        if prefix:
            rows = await self._fetchall(
                "SELECT key FROM kv WHERE key LIKE ? ORDER BY key", (prefix + "%",)
            )
        else:
            rows = await self._fetchall("SELECT key FROM kv ORDER BY key")
        return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    async def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        if session_id:
            sql = """
                SELECT m.id, m.session_id, m.role, m.content, m.created_at
                FROM messages_fts f
                JOIN messages m ON m.id = f.rowid
                WHERE messages_fts MATCH ? AND m.session_id = ?
                ORDER BY m.id DESC
                LIMIT ?
            """
            params = (query, session_id, limit)
        else:
            sql = """
                SELECT m.id, m.session_id, m.role, m.content, m.created_at
                FROM messages_fts f
                JOIN messages m ON m.id = f.rowid
                WHERE messages_fts MATCH ?
                ORDER BY m.id DESC
                LIMIT ?
            """
            params = (query, limit)

        rows = await self._fetchall(sql, params)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # MAINTENANCE
    # ------------------------------------------------------------------

    async def vacuum(self) -> None:
        await self._execute("VACUUM")

    async def backup(self, dest: str | Path) -> Path:
        dest = Path(os.path.expanduser(str(dest)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        def _bk():
            target = sqlite3.connect(str(dest))
            try:
                self._conn.backup(target)
            finally:
                target.close()
        await asyncio.to_thread(_bk)
        logger.info(f"Database backed up to: {dest}")
        return dest

    async def stats(self) -> Dict[str, Any]:
        tables = ["sessions", "messages", "tool_outputs", "plans", "kv"]
        out: Dict[str, int] = {}
        for t in tables:
            row = await self._fetchone(f"SELECT COUNT(*) AS c FROM {t}")
            out[t] = row["c"] if row else 0
        row = await self._fetchone("SELECT value FROM meta WHERE key='schema_version'")
        out["schema_version"] = int(row["value"]) if row else SCHEMA_VERSION
        try:
            out["file_size_bytes"] = self.path.stat().st_size
        except Exception:
            out["file_size_bytes"] = 0
        return out

    def __repr__(self) -> str:
        return f"<Database path={self.path}>"