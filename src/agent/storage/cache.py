"""
Cache - Two-tier (memory + disk) cache with TTL and size limits.

Tiers:
    1. L1: in-memory LRU (fast, small)
    2. L2: disk JSON files under cache_dir (larger, persistent)

Features:
    - Namespaced keys (chat:, tool:, embed:, etc.)
    - TTL per entry
    - LRU eviction when over capacity
    - Size cap for disk tier (with automatic GC)
    - Atomic writes (temp file + rename)
    - Async-first API
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# DATA MODEL
# ======================================================================

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    size: int = 0
    namespace: str = ""

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
            "size": self.size,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ======================================================================
# CACHE
# ======================================================================

class Cache:
    """
    Two-tier cache with LRU memory tier and disk persistence.
    """

    def __init__(
        self,
        cache_dir: str | Path = "~/.agent/cache",
        max_memory_entries: int = 500,
        max_memory_bytes: int = 50 * 1024 * 1024,   # 50 MB
        max_disk_bytes: int = 500 * 1024 * 1024,    # 500 MB
        default_ttl: Optional[float] = None,        # seconds; None = no expiry
        enabled: bool = True,
    ):
        self.cache_dir = Path(os.path.expanduser(str(cache_dir)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_entries = max_memory_entries
        self.max_memory_bytes = max_memory_bytes
        self.max_disk_bytes = max_disk_bytes
        self.default_ttl = default_ttl
        self.enabled = enabled

        # L1 in-memory (LRU)
        self._memory: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._memory_bytes: int = 0
        self._lock = asyncio.Lock()

        # Stats
        self.stats_hits = 0
        self.stats_misses = 0
        self.stats_evictions = 0
        self.stats_disk_writes = 0
        self.stats_disk_reads = 0

    # ------------------------------------------------------------------
    # KEY HELPERS
    # ------------------------------------------------------------------

    def _normalize_key(self, key: str) -> Tuple[str, str]:
        """Split `namespace:key` (namespace optional)."""
        if ":" in key:
            ns, rest = key.split(":", 1)
            return ns, rest
        return "", key

    def _hash_key(self, key: str) -> str:
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Path:
        ns, _ = self._normalize_key(key)
        subdir = self.cache_dir / (ns or "_")
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{self._hash_key(key)}.json"

    # ------------------------------------------------------------------
    # GET / SET / DELETE
    # ------------------------------------------------------------------

    async def get(self, key: str, default: Any = None) -> Any:
        if not self.enabled:
            return default

        async with self._lock:
            # L1 lookup
            entry = self._memory.get(key)
            if entry is not None:
                if entry.is_expired:
                    self._memory.pop(key, None)
                    self._memory_bytes -= entry.size
                else:
                    entry.hits += 1
                    self._memory.move_to_end(key)
                    self.stats_hits += 1
                    return entry.value

            # L2 lookup
            disk = self._disk_path(key)
            if disk.exists():
                try:
                    data = await asyncio.to_thread(disk.read_text, "utf-8")
                    payload = json.loads(data)
                    entry = CacheEntry.from_dict(payload)
                    if entry.is_expired:
                        try:
                            disk.unlink()
                        except Exception:
                            pass
                    else:
                        # Promote to L1
                        self._memory[key] = entry
                        self._memory_bytes += entry.size
                        self._trim_memory()
                        self.stats_hits += 1
                        self.stats_disk_reads += 1
                        return entry.value
                except Exception as e:
                    logger.debug(f"Cache disk read failed for {key}: {e}")

            self.stats_misses += 1
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        persist: bool = True,
    ) -> None:
        if not self.enabled:
            return

        ns, _ = self._normalize_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (time.time() + ttl) if ttl else None

        try:
            size = len(json.dumps(value, default=str))
        except Exception:
            size = len(str(value))

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size=size,
            namespace=ns,
        )

        async with self._lock:
            self._memory[key] = entry
            self._memory.move_to_end(key)
            self._memory_bytes += size
            self._trim_memory()

            if persist:
                await self._write_disk(entry)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            removed = False
            entry = self._memory.pop(key, None)
            if entry:
                self._memory_bytes -= entry.size
                removed = True

            disk = self._disk_path(key)
            if disk.exists():
                try:
                    await asyncio.to_thread(disk.unlink)
                    removed = True
                except Exception:
                    pass
            return removed

    async def has(self, key: str) -> bool:
        if key in self._memory and not self._memory[key].is_expired:
            return True
        disk = self._disk_path(key)
        if disk.exists():
            try:
                data = await asyncio.to_thread(disk.read_text, "utf-8")
                entry = CacheEntry.from_dict(json.loads(data))
                return not entry.is_expired
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # BULK
    # ------------------------------------------------------------------

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            v = await self.get(k)
            if v is not None:
                out[k] = v
        return out

    async def set_many(
        self, items: Dict[str, Any], ttl: Optional[float] = None
    ) -> None:
        for k, v in items.items():
            await self.set(k, v, ttl=ttl)

    async def delete_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace."""
        removed = 0
        async with self._lock:
            for k in list(self._memory.keys()):
                ns, _ = self._normalize_key(k)
                if ns == namespace:
                    self._memory_bytes -= self._memory[k].size
                    self._memory.pop(k, None)
                    removed += 1

        subdir = self.cache_dir / (namespace or "_")
        if subdir.exists():
            def _rm():
                for p in subdir.glob("*.json"):
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
            await asyncio.to_thread(_rm)
        return removed

    # ------------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------------

    def _trim_memory(self) -> None:
        # Enforce entry count
        while len(self._memory) > self.max_memory_entries:
            _, entry = self._memory.popitem(last=False)
            self._memory_bytes -= entry.size
            self.stats_evictions += 1

        # Enforce byte cap
        while self._memory_bytes > self.max_memory_bytes and self._memory:
            _, entry = self._memory.popitem(last=False)
            self._memory_bytes -= entry.size
            self.stats_evictions += 1

    # ------------------------------------------------------------------
    # DISK MANAGEMENT
    # ------------------------------------------------------------------

    async def _write_disk(self, entry: CacheEntry) -> None:
        path = self._disk_path(entry.key)
        payload = json.dumps(entry.to_dict(), default=str)

        def _do():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(payload, "utf-8")
            tmp.replace(path)
        try:
            await asyncio.to_thread(_do)
            self.stats_disk_writes += 1
        except Exception as e:
            logger.debug(f"Cache disk write failed for {entry.key}: {e}")

    async def enforce_disk_limit(self) -> int:
        """Evict oldest disk entries if over the size cap."""
        files: List[Tuple[Path, float, int]] = []
        total = 0
        for p in self.cache_dir.rglob("*.json"):
            try:
                st = p.stat()
                files.append((p, st.st_mtime, st.st_size))
                total += st.st_size
            except Exception:
                continue

        if total <= self.max_disk_bytes:
            return 0

        files.sort(key=lambda x: x[1])  # oldest first
        removed = 0
        for p, _, size in files:
            if total <= self.max_disk_bytes:
                break
            try:
                p.unlink()
                total -= size
                removed += 1
            except Exception:
                continue
        if removed:
            logger.info(f"Cache disk GC removed {removed} files")
        return removed

    async def clear(self, namespace: Optional[str] = None) -> int:
        removed = 0
        async with self._lock:
            if namespace:
                for k in list(self._memory.keys()):
                    ns, _ = self._normalize_key(k)
                    if ns == namespace:
                        self._memory_bytes -= self._memory[k].size
                        self._memory.pop(k, None)
                        removed += 1
            else:
                removed += len(self._memory)
                self._memory.clear()
                self._memory_bytes = 0

        if namespace:
            subdir = self.cache_dir / (namespace or "_")
            if subdir.exists():
                def _rm():
                    nonlocal removed
                    for p in subdir.glob("*.json"):
                        try:
                            p.unlink()
                            removed += 1
                        except Exception:
                            pass
                await asyncio.to_thread(_rm)
        else:
            def _rm_all():
                nonlocal removed
                if self.cache_dir.exists():
                    for p in self.cache_dir.rglob("*.json"):
                        try:
                            p.unlink()
                            removed += 1
                        except Exception:
                            pass
            await asyncio.to_thread(_rm_all)

        return removed

    # ------------------------------------------------------------------
    # GC / STATS
    # ------------------------------------------------------------------

    async def gc_expired(self) -> int:
        """Delete expired entries from both tiers."""
        removed = 0
        now = time.time()

        async with self._lock:
            for k in list(self._memory.keys()):
                if self._memory[k].is_expired:
                    self._memory_bytes -= self._memory[k].size
                    self._memory.pop(k, None)
                    removed += 1

        def _disk_gc():
            nonlocal removed
            for p in self.cache_dir.rglob("*.json"):
                try:
                    data = json.loads(p.read_text("utf-8"))
                    exp = data.get("expires_at")
                    if exp is not None and exp < now:
                        p.unlink()
                        removed += 1
                except Exception:
                    continue
        await asyncio.to_thread(_disk_gc)

        if removed:
            logger.debug(f"Cache GC removed {removed} expired entries")
        return removed

    def stats(self) -> Dict[str, Any]:
        disk_files = 0
        disk_bytes = 0
        for p in self.cache_dir.rglob("*.json"):
            try:
                st = p.stat()
                disk_files += 1
                disk_bytes += st.st_size
            except Exception:
                continue

        total = self.stats_hits + self.stats_misses
        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory),
            "memory_bytes": self._memory_bytes,
            "memory_cap_bytes": self.max_memory_bytes,
            "disk_files": disk_files,
            "disk_bytes": disk_bytes,
            "disk_cap_bytes": self.max_disk_bytes,
            "hits": self.stats_hits,
            "misses": self.stats_misses,
            "hit_rate": (self.stats_hits / total) if total else 0.0,
            "evictions": self.stats_evictions,
            "disk_writes": self.stats_disk_writes,
            "disk_reads": self.stats_disk_reads,
        }

    def reset_stats(self) -> None:
        self.stats_hits = 0
        self.stats_misses = 0
        self.stats_evictions = 0
        self.stats_disk_writes = 0
        self.stats_disk_reads = 0

    # ------------------------------------------------------------------
    # DECORATOR
    # ------------------------------------------------------------------

    def memoize(self, ttl: Optional[float] = None, key_prefix: str = ""):
        """
        Decorator for caching async function results.

            @cache.memoize(ttl=300, key_prefix="fetch")
            async def fetch(url: str) -> str: ...
        """
        def decorator(fn):
            async def wrapper(*args, **kwargs):
                key_parts = [key_prefix or fn.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = ":".join(key_parts)
                cached = await self.get(key)
                if cached is not None:
                    return cached
                result = await fn(*args, **kwargs)
                await self.set(key, result, ttl=ttl)
                return result
            return wrapper
        return decorator

    def __repr__(self) -> str:
        return (
            f"<Cache dir={self.cache_dir} "
            f"mem={len(self._memory)} entries "
            f"disk={sum(1 for _ in self.cache_dir.rglob('*.json'))}>"
        )