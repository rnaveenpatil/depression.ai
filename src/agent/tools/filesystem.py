"""
Filesystem Tool - Read / write / edit / list / delete files inside the workspace.
"""

from __future__ import annotations

import asyncio
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
        "All paths are relative to the workspace unless allowlisted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "read", "write", "append", "edit", "list",
                    "stat", "delete", "mkdir", "move", "copy",
                ],
            },
            "path": {"type": "string"},
            "content": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "dest": {"type": "string"},
            "pattern": {"type": "string"},
            "recursive": {"type": "boolean", "default": False},
            "max_bytes": {"type": "integer", "default": 1048576},
            "max_files": {"type": "integer", "default": 500},
            # Optional: include before/after content in the response so the
            # TUI can render a red/green diff. Off by default to keep the
            # LLM prompt small; on when requested by an explicit flag.
            "include_diff": {"type": "boolean", "default": False},
        },
        "required": ["action", "path"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        # Temp paths are allowed outside the workspace.
        expanded = os.path.expanduser(path)
        if expanded.startswith(("/tmp/", "/tmp\\", "/var/tmp/")):
            return Path(expanded).resolve()
        if self.workspace:
            return self.workspace.assert_inside_workspace(path)
        p = Path(expanded)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = (params.get("action") or "read").lower()
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

        def _read_bytes() -> tuple[bytes, bool, int]:
            with open(target, "rb") as f:
                raw = f.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]
            size = target.stat().st_size
            return raw, truncated, size

        raw, truncated, size = await asyncio.to_thread(_read_bytes)
        if b"\x00" in raw[:2048]:
            return {"success": False, "error": "Binary file not supported"}
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
        include_diff = bool(params.get("include_diff", False))

        # Capture pre-image for append/write when a diff was requested.
        before = ""
        existed = target.exists()
        if include_diff and existed and target.is_file():
            try:
                before = target.read_text(encoding="utf-8", errors="replace")
            except Exception:
                before = ""

        target.parent.mkdir(parents=True, exist_ok=True)

        def _do_write() -> int:
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)
            return len(content.encode("utf-8"))

        written = await asyncio.to_thread(_do_write)

        result: Dict[str, Any] = {
            "success": True,
            "path": str(target),
            "bytes_written": written,
        }
        if include_diff:
            after = ""
            try:
                after = target.read_text(encoding="utf-8", errors="replace")
            except Exception:
                after = ""
            result["before"] = before
            result["after"] = after
            result["created"] = not existed
        return result

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
        replace_all = bool(params.get("all"))
        updated = text.replace(old, new, -1 if replace_all else 1)
        target.write_text(updated, encoding="utf-8")

        result: Dict[str, Any] = {
            "success": True,
            "path": str(target),
            "replacements": count if replace_all else 1,
        }
        if bool(params.get("include_diff", False)):
            result["before"] = text
            result["after"] = updated
        return result

    async def _list(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return {"success": False, "error": f"Not found: {target}"}
        if target.is_file():
            return {"success": True, "path": str(target), "entries": [target.name]}

        recursive = bool(params.get("recursive", False))
        pattern = params.get("pattern", "*")
        max_files = int(params.get("max_files", 500))

        iterator = target.rglob(pattern) if recursive else target.glob(pattern)

        entries: List[Dict[str, Any]] = []
        for p in iterator:
            try:
                st = p.stat()
                entries.append(
                    {
                        "path": str(p.relative_to(target)),
                        "type": "dir" if p.is_dir() else "file",
                        "size": st.st_size if p.is_file() else None,
                    }
                )
                if len(entries) >= max_files:
                    break
            except Exception:
                continue

        return {
            "success": True,
            "path": str(target),
            "entries": entries,
            "count": len(entries),
        }

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
                return {
                    "success": False,
                    "error": "Directory deletion requires recursive=true",
                }
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