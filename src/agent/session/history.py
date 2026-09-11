"""
Session History - Append-only message log with search and compaction.

Responsibilities:
    - Store every turn (user/assistant/system/tool) with metadata
    - Support search by role, keyword, time range
    - Support slicing, tail, and pagination
    - Track token counts per entry
    - Provide export/import for persistence
    - Integrate with the compaction engine
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# HISTORY ENTRY
# ======================================================================

@dataclass
class HistoryEntry:
    """A single message in the history."""

    id: str
    role: str                     # "user" | "assistant" | "system" | "tool"
    content: str
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    pinned: bool = False
    parent_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_message(self) -> Dict[str, Any]:
        """Convert to OpenAI-style chat message."""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            msg["name"] = self.tool_name
        return msg


# ======================================================================
# HISTORY
# ======================================================================

class SessionHistory:
    """
    Ordered log of every message in a session.
    """

    def __init__(self, max_entries: int = 5000):
        self.max_entries = max_entries
        self._entries: List[HistoryEntry] = []
        self._by_id: Dict[str, HistoryEntry] = {}

    # ------------------------------------------------------------------
    # APPEND
    # ------------------------------------------------------------------

    def append(
        self,
        role: str,
        content: str,
        tokens: int = 0,
        pinned: bool = False,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            tokens=tokens,
            pinned=pinned,
            parent_id=parent_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._by_id[entry.id] = entry

        # Trim non-pinned overflow
        if len(self._entries) > self.max_entries:
            self._trim()

        return entry

    def append_entry(self, entry: HistoryEntry) -> HistoryEntry:
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        if len(self._entries) > self.max_entries:
            self._trim()
        return entry

    def _trim(self) -> None:
        pinned = [e for e in self._entries if e.pinned]
        unpinned = [e for e in self._entries if not e.pinned]
        keep = max(0, self.max_entries - len(pinned))
        unpinned = unpinned[-keep:] if keep else []
        self._entries = sorted(pinned + unpinned, key=lambda e: e.timestamp)
        self._by_id = {e.id: e for e in self._entries}

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[HistoryEntry]:
        return iter(self._entries)

    def all(self) -> List[HistoryEntry]:
        return list(self._entries)

    def get(self, entry_id: str) -> Optional[HistoryEntry]:
        return self._by_id.get(entry_id)

    def tail(self, n: int = 20) -> List[HistoryEntry]:
        return self._entries[-n:]

    def head(self, n: int = 20) -> List[HistoryEntry]:
        return self._entries[:n]

    def slice(self, start: int, end: Optional[int] = None) -> List[HistoryEntry]:
        return self._entries[start:end]

    def by_role(self, role: str) -> List[HistoryEntry]:
        return [e for e in self._entries if e.role == role]

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        case_sensitive: bool = False,
        limit: int = 100,
    ) -> List[HistoryEntry]:
        if not query:
            return []
        needle = query if case_sensitive else query.lower()
        results: List[HistoryEntry] = []
        for e in self._entries:
            if role and e.role != role:
                continue
            haystack = e.content if case_sensitive else e.content.lower()
            if needle in haystack:
                results.append(e)
                if len(results) >= limit:
                    break
        return results

    def since(self, timestamp: float) -> List[HistoryEntry]:
        return [e for e in self._entries if e.timestamp >= timestamp]

    def between(self, start: float, end: float) -> List[HistoryEntry]:
        return [e for e in self._entries if start <= e.timestamp <= end]

    # ------------------------------------------------------------------
    # TOKEN COUNTS
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        return sum(e.tokens for e in self._entries)

    def tokens_by_role(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self._entries:
            out[e.role] = out.get(e.role, 0) + e.tokens
        return out

    # ------------------------------------------------------------------
    # MESSAGES FOR LLM
    # ------------------------------------------------------------------

    def to_messages(
        self,
        limit: Optional[int] = None,
        include_pinned: bool = True,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        entries = self._entries

        if not include_system:
            entries = [e for e in entries if e.role != "system"]

        if limit is not None and len(entries) > limit:
            pinned = [e for e in entries if e.pinned] if include_pinned else []
            recent = entries[-limit:]
            seen = set()
            combined = []
            for e in pinned + recent:
                if e.id in seen:
                    continue
                seen.add(e.id)
                combined.append(e)
            combined.sort(key=lambda e: e.timestamp)
            entries = combined

        return [e.to_message() for e in entries]

    # ------------------------------------------------------------------
    # MUTATION
    # ------------------------------------------------------------------

    def update(self, entry_id: str, **fields) -> bool:
        entry = self._by_id.get(entry_id)
        if not entry:
            return False
        for k, v in fields.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return True

    def pin(self, entry_id: str) -> bool:
        return self.update(entry_id, pinned=True)

    def unpin(self, entry_id: str) -> bool:
        return self.update(entry_id, pinned=False)

    def remove(self, entry_id: str) -> bool:
        entry = self._by_id.pop(entry_id, None)
        if not entry:
            return False
        self._entries = [e for e in self._entries if e.id != entry_id]
        return True

    def clear(self, keep_pinned: bool = False, keep_system: bool = True) -> None:
        if keep_pinned or keep_system:
            self._entries = [
                e for e in self._entries
                if (keep_pinned and e.pinned) or (keep_system and e.role == "system")
            ]
        else:
            self._entries = []
        self._by_id = {e.id: e for e in self._entries}

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def from_dict(self, data: List[Dict[str, Any]]) -> None:
        self._entries = []
        self._by_id = {}
        for d in data or []:
            try:
                entry = HistoryEntry.from_dict(d)
                self._entries.append(entry)
                self._by_id[entry.id] = entry
            except Exception as e:
                logger.debug(f"Skip invalid history entry: {e}")

    def export_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self._entries)

    def import_jsonl(self, text: str) -> int:
        count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entry = HistoryEntry.from_dict(d)
                self._entries.append(entry)
                self._by_id[entry.id] = entry
                count += 1
            except Exception:
                continue
        return count

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {
                "count": 0,
                "tokens": 0,
                "roles": {},
                "first_at": None,
                "last_at": None,
                "duration": 0.0,
            }
        return {
            "count": len(self._entries),
            "tokens": self.total_tokens(),
            "roles": self.tokens_by_role(),
            "first_at": self._entries[0].timestamp,
            "last_at": self._entries[-1].timestamp,
            "duration": self._entries[-1].timestamp - self._entries[0].timestamp,
        }