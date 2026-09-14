"""
Filesystem Tool - Read / write / edit / list / delete files inside the workspace.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class FileSystemTool(BaseTool):
    name = "filesystem"
    description = (
        "Read, write, edit, list, and delete files in the project. "
        "All paths are relative to the workspace unless absolute paths are allowed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "edit", "list", "stat", "delete", "mkdir", "move", "copy"],
                "description": "Operation to perform",
            },
            "path": {"type": "string", "description": "Target file or directory"},
            "content": {"type": "string", "description": "For write/append/edit"},
            "old": {"type": "string", "description": "For edit: text to replace"},
            "new": {"type": "string", "description": "For edit: replacement text"},
            "dest": {"type": "string", "description": "For move/copy: destination"},
            "pattern": {"type": "string", "description": "For list: glob pattern"},
            "recursive": {"type": "boolean", "default": False},
            "max_bytes": {"type": "integer", "default": 1048576},
        },
        "required": ["action", "path"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    # ------------------------------------------------------------------

    # Cache for stat calls within a single list operation
    _stat_cache: Dict[str, Any] = {}

    def _resolve(self, path: str) -> Path:
        # Allow absolute paths outside workspace for /tmp/opencode-style temp handling
        # OpenCode permits /tmp for external work — mirror that
        if path.startswith("/tmp/") or path.startswith("/tmp\\"):
            p = Path(os.path.expanduser(path))
            return p.resolve()
        if self.workspace:
            return self.workspace.assert_inside_workspace(path)
        p = Path(os.path.expanduser(path))
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "read").lower()
        path = params.get("path", "")
        if not path:
            return {"success": False, "error": "Missing path"}

        try:
            target = self._resolve(path)
        except Exception as e:
            return {"success": False, "error": str(e)}

        try:
            if action == "read":
                return await self._read(target, params)
            if action == "write":
                return await self._write(target, params, mode="w")
            if action == "append":
                return await self._write(target, params, mode="a")
            if action == "edit":
                return await self._edit(target, params)
            if action == "list":
                return await self._list(target, params)
            if action == "stat":
                return await self._stat(target)
            if action == "delete":
                return await self._delete(target, params)
            if action == "mkdir":
                return await self._mkdir(target)
            if action == "move":
                return await self._move(target, params)
            if action == "copy":
                return await self._copy(target, params)
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------

    async def _read(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}
        if target.is_dir():
            return await self._list(target, params)

        max_bytes = int(params.get("max_bytes", 1_048_576))
        # Use thread pool for large files to avoid blocking event loop
        def _read_bytes() -> tuple[bytes, bool, int]:
            with open(target, "rb") as f:
                raw = f.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]
            size = target.stat().st_size
            return raw, truncated, size
        import asyncio
        raw, truncated, size = await asyncio.to_thread(_read_bytes)
        if b"\x00" in raw[:2048]:
            return {"success": False, "error": "Binary file not supported"}
        # Fast decode with error replace
        content = raw.decode("utf-8", errors="replace")
        return {
            "success": True,
            "path": str(target),
            "content": content,
            "truncated": truncated,
            "size": size,
        }

    async def _write(self, target: Path, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
        content = params.get("content", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        return {
            "success": True,
            "path": str(target),
            "bytes_written": len(content.encode("utf-8")),
        }

    async def _edit(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        old = params.get("old")
        new = params.get("new")
        if old is None or new is None:
            return {"success": False, "error": "edit requires 'old' and 'new'"}
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}

        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return {"success": False, "error": "old text not found"}
        updated = text.replace(old, new, 1 if not params.get("all") else -1)
        target.write_text(updated, encoding="utf-8")
        return {
            "success": True,
            "path": str(target),
            "replacements": 1 if not params.get("all") else count,
        }

    async def _list(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}
        recursive = bool(params.get("recursive", False))
        pattern = params.get("pattern", "*")
        max_files = int(params.get("max_files", 500))

        if target.is_file():
            return {"success": True, "path": str(target), "entries": [target.name]}

        entries: List[Dict[str, Any]] = []
        if recursive:
            iterator = target.rglob(pattern)
        else:
            iterator = target.glob(pattern)

        for p in iterator:
            try:
                st = p.stat()
                entries.append({
                    "path": str(p.relative_to(target)),
                    "type": "dir" if p.is_dir() else "file",
                    "size": st.st_size if p.is_file() else None,
                })
                if len(entries) >= max_files:
                    break
            except Exception:
                continue

        return {"success": True, "path": str(target), "entries": entries, "count": len(entries)}

    async def _stat(self, target: Path) -> Dict[str, Any]:
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}
        st = target.stat()
        return {
            "success": True,
            "path": str(target),
            "type": "dir" if target.is_dir() else "file",
            "size": st.st_size,
            "mtime": st.st_mtime,
            "mode": oct(st.st_mode),
        }

    async def _delete(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}
        if target.is_dir():
            if not params.get("recursive", False):
                return {"success": False, "error": "Directory deletion requires recursive=true"}
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"success": True, "path": str(target)}

    async def _mkdir(self, target: Path) -> Dict[str, Any]:
        target.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(target)}

    async def _move(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        dest = params.get("dest")
        if not dest:
            return {"success": False, "error": "move requires 'dest'"}
        dest_p = self._resolve(dest)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dest_p))
        return {"success": True, "from": str(target), "to": str(dest_p)}

    async def _copy(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        dest = params.get("dest")
        if not dest:
            return {"success": False, "error": "copy requires 'dest'"}
        dest_p = self._resolve(dest)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.copytree(str(target), str(dest_p), dirs_exist_ok=True)
        else:
            shutil.copy2(str(target), str(dest_p))
        return {"success": True, "from": str(target), "to": str(dest_p)}