"""
Filesystem Tool - Read / write / edit / list / delete files inside the workspace.

Every action returns a structured dict:
    {"success": True, ...}
    {"success": False, "error": str, "recoverable": bool, "suggestion": str}

Path policy:
    * Relative paths are resolved against the WORKSPACE ROOT, not the
      process cwd. This makes tool calls stable no matter where the agent
      process was launched from.
    * Absolute paths under the workspace are allowed.
    * Absolute paths outside the workspace are allowed ONLY for
      /tmp, /var/tmp, and ~/.cache unless the caller is the workspace.
    * `..` traversal that escapes the workspace is rejected.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


DIFF_SIZE_CAP = 50 * 1024           # bytes per side, for diff rendering
DEFAULT_MAX_BYTES = 1_048_576       # 1 MiB cap on read() content
READ_CONTENT_HARD_CAP = 512 * 1024  # never exceed 512 KiB of content returned
ALLOWED_ABS_ROOTS = ("/tmp", "/var/tmp", os.path.expanduser("~/.cache"))


def _err(error: str, *, recoverable: bool = True,
         suggestion: str = "", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error": error,
        "recoverable": recoverable,
    }
    if suggestion:
        out["suggestion"] = suggestion
    out.update(extra)
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileSystemTool(BaseTool):
    name = "filesystem"
    description = (
        "Read, write, edit, list, stat, delete, mkdir, move, copy files. "
        "Relative paths resolve against the workspace root. Returns "
        "structured results with success/error/recoverable fields."
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
            "max_bytes": {"type": "integer", "default": DEFAULT_MAX_BYTES},
            "max_files": {"type": "integer", "default": 500},
            "include_diff": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Capture before/after content for diff rendering. "
                    "Defaults to true; capped at 50 KB per side."
                ),
            },
            "include_content": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Set false to only stat the file without reading it. "
                    "Useful for verification without burning context."
                ),
            },
            "expected_sha256": {
                "type": "string",
                "description": (
                    "Optional. Reject the edit if the file's current hash "
                    "does not match. Prevents stale-file edits."
                ),
            },
            "all": {
                "type": "boolean",
                "default": False,
                "description": "For edit: replace ALL occurrences instead of the first.",
            },
        },
        "required": ["action", "path"],
    }
    timeout = 30.0

    read_only = False
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        # Base cwd for relative paths. Prefer the workspace root.
        try:
            if workspace is not None:
                self.base_cwd = Path(str(workspace.get_project_dir())).resolve()
            else:
                self.base_cwd = Path.cwd().resolve()
        except Exception:
            self.base_cwd = Path.cwd().resolve()

    # ------------------------------------------------------------------
    # PATH RESOLUTION (single choke point)
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """
        Resolve a user-supplied path to an absolute Path.

        Policy:
          * "~" is expanded.
          * Relative paths resolve against self.base_cwd.
          * Absolute paths outside the workspace are allowed ONLY if they
            live under /tmp, /var/tmp, or ~/.cache.
          * A resolved path must not escape the workspace via "..".
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")

        expanded = os.path.expanduser(path.strip())
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = (self.base_cwd / candidate)

        try:
            resolved = candidate.resolve(strict=False)
        except Exception as e:
            raise ValueError(f"cannot resolve path: {e}") from e

        # Workspace containment check.
        if self.workspace is not None:
            try:
                ws_root = Path(str(self.workspace.get_project_dir())).resolve()
            except Exception:
                ws_root = self.base_cwd
            try:
                resolved.relative_to(ws_root)
                return resolved
            except ValueError:
                pass
            # Allowlist outside workspace.
            for root in ALLOWED_ABS_ROOTS:
                try:
                    resolved.relative_to(Path(root).resolve())
                    return resolved
                except ValueError:
                    continue
            raise PermissionError(
                f"path escapes workspace: {resolved}"
            )
        return resolved

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = (params.get("action") or "read").lower()
        path = params.get("path", "")
        if not path:
            return _err("Missing required field 'path'",
                        suggestion="Provide a path, e.g. 'src/main.py'.")

        try:
            target = self._resolve(path)
        except PermissionError as e:
            return _err(f"Permission denied resolving '{path}': {e}",
                        recoverable=False)
        except Exception as e:
            return _err(f"Invalid path '{path}': {e}",
                        suggestion="Use a path inside the workspace.")

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
            return _err(f"Unknown action: {action}",
                        suggestion="Use one of: read, write, append, edit, list, "
                                   "stat, delete, mkdir, move, copy.")
        except PermissionError as e:
            return _err(f"Permission denied: {e}", recoverable=False)
        except IsADirectoryError as e:
            return _err(f"Expected a file but found a directory: {e}",
                        suggestion="Use action='list' for directories.")
        except NotADirectoryError as e:
            return _err(f"Expected a directory but found a file: {e}")
        except FileNotFoundError as e:
            return _err(f"Not found: {e}",
                        suggestion="Use action='list' to see what's there.")
        except OSError as e:
            # Transient OS errors (EAGAIN, EINTR) are usually recoverable.
            return _err(f"OS error: {e}", recoverable=True)
        except Exception as e:
            logger.error("filesystem.%s failed: %s", action, e, exc_info=True)
            return _err(f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    async def _read(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        include_content = bool(params.get("include_content", True))

        if not target.exists():
            return _err(f"File not found: {target}",
                        suggestion="Use action='list' or the glob tool to locate it.")

        if target.is_dir():
            if include_content:
                return await self._list(target, params)
            return {
                "success": True,
                "path": str(target),
                "type": "dir",
                "content_omitted": True,
            }

        if not target.is_file():
            return _err(f"Not a regular file: {target}")

        max_bytes = int(params.get("max_bytes", DEFAULT_MAX_BYTES))
        max_bytes = max(1, min(max_bytes, READ_CONTENT_HARD_CAP))

        def _read_bytes():
            with open(target, "rb") as f:
                raw = f.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]
            size = target.stat().st_size
            return raw, truncated, size

        try:
            raw, truncated, size = await asyncio.to_thread(_read_bytes)
        except PermissionError:
            return _err(f"Permission denied reading {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to read {target}: {e}")

        digest = _sha256_bytes(raw)

        if not include_content:
            return {
                "success": True,
                "path": str(target),
                "type": "file",
                "size": size,
                "sha256": digest,
                "truncated": truncated,
                "content_omitted": True,
            }

        if b"\x00" in raw[:2048]:
            return _err(
                f"Binary file not supported: {target}",
                suggestion="Use the terminal tool with `xxd` or `file` if you must inspect it.",
                binary=True,
                sha256=digest,
            )

        content, encoding = self._decode(raw)
        if content is None:
            return _err(
                f"Could not decode {target} as text",
                suggestion="The file may be binary or use an unsupported encoding.",
                encoding_failed=True,
            )

        return {
            "success": True,
            "path": str(target),
            "type": "file",
            "content": content,
            "encoding": encoding,
            "truncated": truncated,
            "size": size,
            "lines": content.count("\n") + (
                1 if content and not content.endswith("\n") else 0
            ),
            "sha256": digest,
        }

    @staticmethod
    def _decode(raw: bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(enc), enc
            except UnicodeDecodeError:
                continue
        return None, None

    # ------------------------------------------------------------------
    # WRITE / APPEND
    # ------------------------------------------------------------------

    async def _write(self, target: Path, params: Dict[str, Any],
                     mode: str) -> Dict[str, Any]:
        content = params.get("content", "")
        if not isinstance(content, str):
            return _err("'content' must be a string")

        include_diff = bool(params.get("include_diff", True))
        before = ""
        existed = target.exists()

        if existed and target.is_dir():
            return _err(f"Cannot write: {target} is a directory",
                        suggestion="Delete it or choose a different path.")

        if existed and not target.is_file():
            return _err(f"Cannot write: {target} is not a regular file")

        if include_diff and existed and target.is_file():
            try:
                before = target.read_text(encoding="utf-8", errors="replace")
            except Exception:
                before = ""

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return _err(f"Cannot create parent directory for {target}",
                        recoverable=False)
        except Exception as e:
            return _err(f"Cannot create parent directory: {e}")

        data = content.encode("utf-8")

        def _do_write() -> None:
            with open(target, mode, "wb") as f:
                f.write(data)

        try:
            await asyncio.to_thread(_do_write)
        except PermissionError:
            return _err(f"Permission denied writing {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to write {target}: {e}")

        after_sha = _sha256_bytes(data) if mode == "w" else _sha256_bytes(
            (before.encode("utf-8") if before else b"") + data
        )

        result: Dict[str, Any] = {
            "success": True,
            "path": str(target),
            "bytes_written": len(data),
            "created": not existed,
            "mode": "append" if mode == "a" else "write",
            "after_sha256": after_sha,
        }
        if include_diff:
            try:
                after = target.read_text(encoding="utf-8", errors="replace")
            except Exception:
                after = ""
            if len(before.encode("utf-8")) <= DIFF_SIZE_CAP and len(after.encode("utf-8")) <= DIFF_SIZE_CAP:
                result["before"] = before
                result["after"] = after
            else:
                result["diff_omitted"] = True
                result["diff_omitted_reason"] = (
                    f"file exceeds {DIFF_SIZE_CAP} bytes per side"
                )
        return result

    # ------------------------------------------------------------------
    # EDIT
    # ------------------------------------------------------------------

    async def _edit(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        old = params.get("old")
        new = params.get("new")
        if old is None or new is None:
            return _err("edit requires both 'old' and 'new'",
                        suggestion="Pass {'action':'edit','path':...,'old':...,'new':...}.")
        if not isinstance(old, str) or not isinstance(new, str):
            return _err("'old' and 'new' must be strings")

        if not target.exists():
            return _err(f"File not found: {target}",
                        suggestion="Read the file first to confirm the exact text.")
        if target.is_dir():
            return _err(f"Cannot edit a directory: {target}")

        try:
            raw = target.read_bytes()
        except PermissionError:
            return _err(f"Permission denied reading {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to read {target}: {e}")

        before_sha = _sha256_bytes(raw)

        expected = params.get("expected_sha256")
        if expected and expected != before_sha:
            return _err(
                "file changed since it was read (stale edit)",
                recoverable=True,
                stale_file=True,
                expected_sha256=expected,
                actual_sha256=before_sha,
                suggestion=(
                    "Read the file again, then re-apply the edit against the "
                    "current content."
                ),
            )

        text, encoding = self._decode(raw)
        if text is None:
            return _err(f"Could not decode {target} as text")

        count = text.count(old)
        if count == 0:
            snippet = text[:400]
            return _err(
                "old text not found in file",
                suggestion=(
                    "Call action='read' and copy the exact text (including "
                    "whitespace). First 400 chars of the file:\n" + snippet
                ),
            )

        replace_all = bool(params.get("all"))
        replaced = count if replace_all else 1
        updated = text.replace(old, new, -1 if replace_all else 1)
        new_bytes = updated.encode(encoding or "utf-8")

        try:
            target.write_bytes(new_bytes)
        except PermissionError:
            return _err(f"Permission denied writing {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to write edit: {e}")

        after_sha = _sha256_bytes(new_bytes)

        result: Dict[str, Any] = {
            "success": True,
            "path": str(target),
            "replacements": replaced,
            "occurrences_found": count,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        }
        include_diff = bool(params.get("include_diff", True))
        if include_diff:
            if (len(text.encode("utf-8")) <= DIFF_SIZE_CAP
                    and len(updated.encode("utf-8")) <= DIFF_SIZE_CAP):
                result["before"] = text
                result["after"] = updated
            else:
                result["diff_omitted"] = True
        return result

    # ------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------

    async def _list(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return _err(f"Not found: {target}",
                        suggestion="Check the path or use the glob tool.")
        if target.is_file():
            return {"success": True, "path": str(target), "entries": [target.name],
                    "count": 1}

        recursive = bool(params.get("recursive", False))
        pattern = params.get("pattern", "*")
        max_files = int(params.get("max_files", 500))

        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        entries: List[Dict[str, Any]] = []
        try:
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
        except PermissionError:
            return _err(f"Permission denied listing {target}", recoverable=False)

        return {"success": True, "path": str(target),
                "entries": entries, "count": len(entries),
                "truncated": len(entries) >= max_files}

    # ------------------------------------------------------------------
    # STAT
    # ------------------------------------------------------------------

    async def _stat(self, target: Path) -> Dict[str, Any]:
        if not target.exists():
            return _err(f"Not found: {target}")
        try:
            st = target.stat()
        except PermissionError:
            return _err(f"Permission denied: {target}", recoverable=False)
        return {
            "success": True,
            "path": str(target),
            "type": "dir" if target.is_dir() else ("file" if target.is_file() else "other"),
            "size": st.st_size,
            "mtime": st.st_mtime,
            "mode": oct(st.st_mode),
        }

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    async def _delete(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        if not target.exists():
            return _err(f"Not found: {target}")
        try:
            if target.is_dir():
                if not params.get("recursive", False):
                    return _err(
                        "Directory deletion requires recursive=true",
                        suggestion="Pass recursive=true if you really mean it.",
                    )
                shutil.rmtree(target)
            else:
                target.unlink()
        except PermissionError:
            return _err(f"Permission denied deleting {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to delete {target}: {e}")
        return {"success": True, "path": str(target), "deleted": True}

    # ------------------------------------------------------------------
    # MKDIR / MOVE / COPY
    # ------------------------------------------------------------------

    async def _mkdir(self, target: Path) -> Dict[str, Any]:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return _err(f"Permission denied creating {target}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to create directory: {e}")
        return {"success": True, "path": str(target), "created": True}

    async def _move(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        dest = params.get("dest")
        if not dest:
            return _err("move requires 'dest'")
        if not target.exists():
            return _err(f"Source not found: {target}")
        try:
            dest_p = self._resolve(dest)
            dest_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(dest_p))
        except PermissionError as e:
            return _err(f"Permission denied moving: {e}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to move: {e}")
        return {"success": True, "from": str(target), "to": str(dest_p)}

    async def _copy(self, target: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        dest = params.get("dest")
        if not dest:
            return _err("copy requires 'dest'")
        if not target.exists():
            return _err(f"Source not found: {target}")
        try:
            dest_p = self._resolve(dest)
            dest_p.parent.mkdir(parents=True, exist_ok=True)
            if target.is_dir():
                shutil.copytree(str(target), str(dest_p), dirs_exist_ok=True)
            else:
                shutil.copy2(str(target), str(dest_p))
        except PermissionError as e:
            return _err(f"Permission denied copying: {e}", recoverable=False)
        except Exception as e:
            return _err(f"Failed to copy: {e}")
        return {"success": True, "from": str(target), "to": str(dest_p)}