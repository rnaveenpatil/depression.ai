"""
Compatibility Tools

Wraps existing implementations to expose canonical tool names:
  bash, read, write, edit, grep, glob, apply_patch, todowrite, todoread,
  webfetch, websearch, skill, question, lsp

Write, edit, and apply_patch forward include_diff=true by default so the
TUI can render an inline diff without the caller having to opt in.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.file_walk import WalkStats, glob_match, walk_paths
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# bash -> terminal
# ----------------------------------------------------------------------
class BashTool(BaseTool):
    name = "bash"
    description = "Execute shell commands (canonical alias for terminal)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "workdir": {"type": "string", "description": "Working directory"},
            "cwd": {"type": "string", "description": "Working directory (alias)"},
            "timeout": {"type": "number", "default": 60},
            "description": {"type": "string", "description": "Human-readable description"},
            "include_aws_credentials": {
                "type": "boolean",
                "default": False,
                "description": "Opt in to injecting saved AWS credentials",
            },
        },
        "required": ["command"],
    }
    timeout = 120.0

    read_only = False
    mutating = True
    category = "run"

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        from agent.tools.terminal import TerminalTool
        self._inner = TerminalTool(workspace, config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        if "command" in params:
            p["command"] = params["command"]
        if "cwd" in params and params["cwd"]:
            p["cwd"] = params["cwd"]
        elif "workdir" in params and params["workdir"]:
            p["cwd"] = params["workdir"]
        if "timeout" in params:
            p["timeout"] = params["timeout"]
        if "shell" in params:
            p["shell"] = params["shell"]
        if "env" in params:
            p["env"] = params["env"]
        if params.get("include_aws_credentials"):
            p["include_aws_credentials"] = True
        return await self._inner.execute(p)


# ----------------------------------------------------------------------
# read
# ----------------------------------------------------------------------
class ReadTool(BaseTool):
    name = "read"
    description = "Read file contents. Supports line ranges via limit/offset."
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Path to file"},
            "path": {"type": "string", "description": "Alias for filePath"},
            "limit": {"type": "integer", "description": "Max lines to read"},
            "offset": {"type": "integer", "description": "Line offset (0-indexed)"},
        },
        "required": [],
    }
    timeout = 30.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None):
        from agent.tools.filesystem import FileSystemTool
        self._inner = FileSystemTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("filePath") or params.get("path") or ""
        if not path:
            return {"success": False, "error": "read requires filePath"}

        try:
            offset = max(0, int(params.get("offset", 0) or 0))
            limit = (
                int(params["limit"]) if params.get("limit") is not None else None
            )
        except (TypeError, ValueError):
            return {"success": False, "error": "offset and limit must be integers"}
        if limit is not None and limit < 0:
            return {"success": False, "error": "limit must be >= 0"}

        inner: Dict[str, Any] = {"action": "read", "path": path}
        if limit == 0:
            checked = await self._inner.execute({"action": "stat", "path": path})
            if not checked.get("success"):
                return checked
            return {
                "success": True,
                "path": checked.get("path", path),
                "type": checked.get("type"),
                "content": "",
                "truncated": False,
            }
        if limit is not None:
            # offset is 0-indexed; filesystem line ranges are 1-indexed.
            inner["start_line"] = offset + 1
            inner["end_line"] = offset + limit
        elif offset:
            # Preserve the "read from offset to end" behavior while letting
            # FileSystemTool stream from the requested line instead of loading
            # the first 32 KB and slicing it in memory.
            inner["start_line"] = offset + 1
            inner["end_line"] = 2**31 - 1

        return await self._inner.execute(inner)


class WriteTool(BaseTool):
    name = "write"
    description = (
        "Create or overwrite a file. Returns before/after content for "
        "diff rendering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
            "include_diff": {"type": "boolean", "default": True},
        },
        "required": ["content"],
    }
    timeout = 30.0

    read_only = False
    mutating = True
    category = "edit"

    def __init__(self, workspace: Any = None):
        from agent.tools.filesystem import FileSystemTool
        self._inner = FileSystemTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("filePath") or params.get("path") or ""
        content = params.get("content", "")
        if not path:
            return {"success": False, "error": "write requires filePath"}
        inner_params: Dict[str, Any] = {
            "action": "write",
            "path": path,
            "content": content,
            "include_diff": bool(params.get("include_diff", True)),
        }
        return await self._inner.execute(inner_params)


class EditTool(BaseTool):
    name = "edit"
    description = (
        "Modify an existing file via exact string replacement. Returns "
        "before/after content for diff rendering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "path": {"type": "string"},
            "oldString": {"type": "string"},
            "old_string": {"type": "string"},
            "newString": {"type": "string"},
            "new_string": {"type": "string"},
            "replaceAll": {"type": "boolean", "default": False},
            "include_diff": {"type": "boolean", "default": True},
        },
        "required": [],
    }
    timeout = 30.0

    read_only = False
    mutating = True
    category = "edit"

    def __init__(self, workspace: Any = None):
        from agent.tools.filesystem import FileSystemTool
        self._inner = FileSystemTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("filePath") or params.get("path") or ""
        old = params.get("oldString") or params.get("old_string") or params.get("old") or ""
        new = params.get("newString") or params.get("new_string") or params.get("new") or ""
        replace_all = bool(params.get("replaceAll", False))
        if not path:
            return {"success": False, "error": "edit requires filePath"}
        if old == "":
            return {"success": False, "error": "edit requires a non-empty oldString"}
        p: Dict[str, Any] = {
            "action": "edit",
            "path": path,
            "old": old,
            "new": new,
            "include_diff": bool(params.get("include_diff", True)),
        }
        if replace_all:
            p["all"] = True
        return await self._inner.execute(p)


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = (
        "Apply a patch using the marker format. Returns before/after content "
        "for diff rendering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "patchText": {"type": "string"},
            "patch": {"type": "string"},
            "diff": {"type": "string"},
            "include_diff": {"type": "boolean", "default": True},
        },
        "required": [],
    }
    timeout = 30.0

    read_only = False
    mutating = True
    category = "edit"

    def __init__(self, workspace: Any = None):
        from agent.tools.patch import PatchTool
        self._inner = PatchTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        patch_text = (
            params.get("patchText")
            or params.get("patch")
            or params.get("diff")
            or ""
        )
        if not patch_text:
            return {"success": False, "error": "apply_patch requires patchText"}
        include_diff = bool(params.get("include_diff", True))
        if "***" in patch_text:
            return await self._apply_marker_patch(
                patch_text, include_diff=include_diff
            )
        inner_params: Dict[str, Any] = {"action": "apply", "diff": patch_text}
        if include_diff:
            inner_params["include_diff"] = True
        return await self._inner.execute(inner_params)

    # ------------------------------------------------------------------

    @staticmethod
    def _write_bytes_atomic(
        target: Path, data: bytes, mode: Optional[int] = None
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".patch-", dir=str(target.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if mode is not None:
                os.chmod(tmp, mode)
            os.replace(tmp, target)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _snapshot_path(
        snapshots: Dict[Path, Dict[str, Any]], path: Path
    ) -> None:
        if path in snapshots:
            return
        if path.is_file():
            stat = path.stat()
            snapshots[path] = {
                "kind": "file",
                "data": path.read_bytes(),
                "mode": stat.st_mode & 0o7777,
            }
        elif path.is_dir():
            snapshots[path] = {"kind": "dir"}
        elif path.exists():
            snapshots[path] = {"kind": "other"}
        else:
            snapshots[path] = {"kind": "missing"}

    def _rollback(self, snapshots: Dict[Path, Dict[str, Any]]) -> List[str]:
        errors: List[str] = []
        for path, snap in reversed(list(snapshots.items())):
            try:
                if snap["kind"] == "missing":
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                elif snap["kind"] == "file":
                    self._write_bytes_atomic(path, snap["data"], snap["mode"])
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return errors

    async def _apply_marker_patch(
        self, patch_text: str, include_diff: bool = True
    ) -> Dict[str, Any]:
        sections = self._split_sections(patch_text)
        if not sections:
            return {"success": False, "error": "No file sections found in patch"}

        results: List[Dict[str, Any]] = []
        snapshots: Dict[Path, Dict[str, Any]] = {}
        applied: List[Dict[str, Any]] = []
        for section in sections:
            try:
                paths = [self._inner._resolve(section["path"])]  # type: ignore[attr-defined]
                if section.get("move_to"):
                    paths.append(self._inner._resolve(section["move_to"]))  # type: ignore[attr-defined]
                for path in paths:
                    self._snapshot_path(snapshots, path)
                result = await self._apply_section(section, include_diff=include_diff)
            except Exception as e:
                result = {"success": False, "path": section["path"], "error": str(e)}
            results.append(result)
            if result.get("success"):
                applied.append(result)
                continue

            rollback_errors = self._rollback(snapshots)
            for previous in applied:
                previous["success"] = False
                previous["rolled_back"] = True
                previous.setdefault(
                    "error", "rolled back because a later file section failed"
                )
            if rollback_errors:
                result["rollback_errors"] = rollback_errors
            break

        ok = bool(results) and all(r.get("success") for r in results)
        return {"success": ok, "results": results}

    def _split_sections(self, patch_text: str) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for raw in patch_text.splitlines():
            line = raw.rstrip("\r")
            if line.startswith("*** Begin Patch"):
                continue
            if line.startswith("*** End Patch"):
                if current:
                    sections.append(current)
                    current = None
                continue
            if line.startswith("*** Add File:"):
                if current:
                    sections.append(current)
                current = {"action": "add", "path": line.split(":", 1)[1].strip(), "lines": []}
                continue
            if line.startswith("*** Update File:"):
                if current:
                    sections.append(current)
                current = {"action": "update", "path": line.split(":", 1)[1].strip(), "lines": [], "move_to": None}
                continue
            if line.startswith("*** Delete File:"):
                if current:
                    sections.append(current)
                current = {"action": "delete", "path": line.split(":", 1)[1].strip(), "lines": []}
                continue
            if line.startswith("*** Move to:"):
                dest = line.split(":", 1)[1].strip()
                if current is None:
                    raise ValueError("Move to: without a preceding Update File:")
                current["move_to"] = dest
                continue
            if current is None:
                continue
            current["lines"].append(line)

        if current:
            sections.append(current)
        return sections

    async def _apply_section(
        self, section: Dict[str, Any], include_diff: bool = True
    ) -> Dict[str, Any]:
        action = section["action"]
        path = section["path"]
        target = self._inner._resolve(path)  # type: ignore[attr-defined]

        if action == "add":
            if target.exists():
                return {
                    "success": False,
                    "path": str(target),
                    "error": "Add File target already exists",
                }
            content = self._decode_add_lines(section["lines"])
            self._inner._write_atomic(target, content)  # type: ignore[attr-defined]
            entry: Dict[str, Any] = {
                "success": True,
                "path": str(target),
                "action": "add",
                "bytes_written": len(content.encode("utf-8")),
                "created": True,
            }
            if include_diff:
                self._inner._capture(entry, target, "", content)  # type: ignore[attr-defined]
            return entry

        if action == "delete":
            if not target.exists():
                return {"success": False, "path": str(target), "error": "not found"}
            if target.is_dir():
                return {
                    "success": False,
                    "path": str(target),
                    "error": "refusing to delete a directory; remove files explicitly",
                }
            before = ""
            if include_diff:
                try:
                    before = self._inner._read_text(target)  # type: ignore[attr-defined]
                except Exception:
                    before = ""
            target.unlink()
            entry: Dict[str, Any] = {"success": True, "path": str(target), "action": "delete"}
            if include_diff:
                self._inner._capture(entry, target, before, "")  # type: ignore[attr-defined]
            return entry

        if action == "update":
            if not target.exists() or not target.is_file():
                return {"success": False, "path": str(target), "error": "not found"}
            original = self._inner._read_text(target)  # type: ignore[attr-defined]
            updated = self._apply_update_hunks(original, section["lines"])

            final_path = target
            if section.get("move_to"):
                final_path = self._inner._resolve(section["move_to"])  # type: ignore[attr-defined]
                if final_path != target and final_path.exists():
                    return {
                        "success": False,
                        "path": str(final_path),
                        "error": "Move to target already exists",
                    }

            self._inner._write_atomic(final_path, updated)  # type: ignore[attr-defined]
            if final_path != target and target.exists():
                target.unlink()

            entry: Dict[str, Any] = {
                "success": True,
                "path": str(final_path),
                "action": "update",
                "bytes_before": len(original.encode("utf-8")),
                "bytes_after": len(updated.encode("utf-8")),
            }
            if include_diff:
                self._inner._capture(entry, final_path, original, updated)  # type: ignore[attr-defined]
            return entry

        return {"success": False, "path": path, "error": f"Unknown action: {action}"}

    def _decode_add_lines(self, lines: List[str]) -> str:
        out: List[str] = []
        for line in lines:
            if line.startswith("+"):
                out.append(line[1:])
            else:
                out.append(line)
        text = "\n".join(out)
        if out and not text.endswith("\n"):
            text += "\n"
        return text

    def _apply_update_hunks(self, original: str, hunk_lines: List[str]) -> str:
        src_lines = original.splitlines(keepends=True)
        newline = "\r\n" if "\r\n" in original else "\n"

        chunks: List[tuple[List[str], bool]] = []
        current: List[str] = []
        current_new_no_newline = False
        last_kind = ""

        def flush() -> None:
            nonlocal current, current_new_no_newline
            if current:
                chunks.append((current, current_new_no_newline))
            current = []
            current_new_no_newline = False

        for line in hunk_lines:
            # Standard apply_patch hunk headers are metadata, not context.
            if line.startswith("@@"):
                flush()
                last_kind = ""
                continue
            if line.startswith("\\"):
                if current and last_kind in ("+", " "):
                    current_new_no_newline = True
                last_kind = ""
                continue
            if line.startswith(("+", "-", " ")):
                current.append(line)
                last_kind = line[0]
            else:
                flush()
                current.append(" " + line)
                last_kind = " "
        flush()

        for chunk, new_no_newline in chunks:
            old_lines = [c[1:] for c in chunk if c.startswith((" ", "-"))]
            new_lines = [c[1:] for c in chunk if c.startswith((" ", "+"))]
            replacement = [self._inner._line_body(l) + newline for l in new_lines]
            if new_no_newline and replacement:
                replacement[-1] = self._inner._line_body(new_lines[-1])
            if not old_lines:
                src_lines.extend(replacement)
                continue
            start = self._find_chunk(src_lines, old_lines)
            if start is None:
                raise ValueError(f"could not find hunk anchor: {old_lines[:1]!r}")
            src_lines[start:start + len(old_lines)] = replacement
        return "".join(src_lines)

    def _find_chunk(self, src_lines: List[str], old_lines: List[str]) -> Optional[int]:
        target = [self._inner._line_body(l) for l in old_lines]
        n = len(target)
        for i in range(0, len(src_lines) - n + 1):
            window = [self._inner._line_body(l) for l in src_lines[i:i + n]]
            if window == target:
                return i
        return None


# ----------------------------------------------------------------------
# grep / glob
# ----------------------------------------------------------------------
class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents with regex."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "include": {"type": "string"},
            "glob": {"type": "string"},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["pattern"],
    }
    timeout = 60.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None):
        from agent.tools.search import SearchTool
        self._inner = SearchTool(workspace)
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = params.get("pattern") or ""
        if not pattern:
            return {"success": False, "error": "grep requires pattern"}
        glob_pat = params.get("include") or params.get("glob") or "*"
        root = params.get("path") or (str(self.workspace.get_project_dir()) if self.workspace else ".")
        try:
            max_results = max(1, min(int(params.get("max_results", 100)), 1000))
        except (TypeError, ValueError):
            return {"success": False, "error": "max_results must be an integer"}
        return await self._inner.execute({
            "query": pattern,
            "path": root,
            "glob": glob_pat,
            "regex": True,
            "max_results": max_results,
        })


class GlobTool(BaseTool):
    name = "glob"
    description = "Find files by glob pattern, newest first."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "max_results": {"type": "integer", "default": 500},
            "max_seconds": {"type": "number", "default": 5.0},
        },
        "required": ["pattern"],
    }
    timeout = 30.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    @staticmethod
    def _split_absolute(pattern: str) -> tuple[Path, str]:
        parts = Path(pattern).parts
        wildcard = next(
            (i for i, part in enumerate(parts)
             if any(ch in part for ch in "*?[")),
            len(parts),
        )
        if wildcard >= len(parts):
            return Path(pattern).parent, Path(pattern).name
        root = Path(*parts[:wildcard]) if wildcard else Path(os.sep)
        return root, "/".join(parts[wildcard:])

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = params.get("pattern") or ""
        if not pattern:
            return {"success": False, "error": "glob requires pattern"}
        try:
            max_results = max(1, min(int(params.get("max_results", 500)), 5000))
            max_seconds = max(0.05, min(float(params.get("max_seconds", 5.0)), 30.0))
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "max_results and max_seconds must be numbers",
            }

        root_value = params.get("path") or (
            str(self.workspace.get_project_dir()) if self.workspace else "."
        )
        if os.path.isabs(pattern):
            root_path, pattern = self._split_absolute(pattern)
        else:
            root_path = Path(root_value).expanduser()
            pattern = pattern.replace("\\", "/")
            if pattern.startswith("./"):
                pattern = pattern[2:]

        from agent.tools.search import _is_secret_file, resolve_search_root

        try:
            root_path = resolve_search_root(self.workspace, str(root_path))
        except Exception as e:
            return {"success": False, "error": f"Invalid glob root: {e}"}

        # A non-recursive glob pattern only needs to scan to its fixed depth.
        max_depth = None if "**" in pattern else pattern.count("/")

        def _search() -> tuple[List[tuple[str, float]], WalkStats]:
            stats = WalkStats()

            def accept(relative: str, name: str, is_dir: bool) -> bool:
                return glob_match(relative, pattern)

            found: List[tuple[str, float]] = []
            for item in walk_paths(
                root_path,
                recursive=True,
                max_results=max_results,
                max_seconds=max_seconds,
                max_depth=max_depth,
                accept=accept,
                stats=stats,
            ):
                if not _is_secret_file(item.path):
                    found.append((item.relative, item.mtime))
            return found, stats

        try:
            found, stats = await asyncio.to_thread(_search)
        except FileNotFoundError:
            return {
                "success": True,
                "pattern": pattern,
                "matches": [],
                "count": 0,
                "truncated": False,
                "truncated_reason": None,
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {root_path}"}
        except OSError as exc:
            return {"success": False, "error": f"glob failed: {exc}"}

        found.sort(key=lambda item: item[1], reverse=True)
        return {
            "success": True,
            "pattern": pattern,
            "matches": [item[0] for item in found],
            "count": len(found),
            "truncated": stats.truncated,
            "truncated_reason": stats.reason or None,
        }


# ----------------------------------------------------------------------
# webfetch / websearch
# ----------------------------------------------------------------------
class WebFetchTool(BaseTool):
    name = "webfetch"
    description = "Fetch web content from a URL (static HTML)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "maxChars": {"type": "integer", "default": 500000},
            "maxBytes": {"type": "integer"},
            "extractMode": {"type": "string", "enum": ["text", "markdown", "html"], "default": "text"},
        },
        "required": ["url"],
    }
    timeout = 60.0

    read_only = True
    mutating = False
    category = "net"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from agent.tools.web import WebTool
        self._inner = WebTool(config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url") or ""
        if not url:
            return {"success": False, "error": "webfetch requires url"}
        try:
            max_bytes = int(params.get("maxBytes") or params.get("maxChars", 500000))
        except (TypeError, ValueError):
            return {"success": False, "error": "maxChars/maxBytes must be an integer"}
        max_bytes = max(1024, min(max_bytes, 2_000_000))
        mode = str(params.get("extractMode", "text")).lower()
        if mode not in {"text", "markdown", "html"}:
            return {"success": False, "error": "extractMode must be text, markdown, or html"}
        return await self._inner.execute({
            "action": "fetch",
            "url": url,
            "max_bytes": max_bytes,
            "text_only": mode != "html",
        })


class WebSearchTool(BaseTool):
    name = "websearch"
    description = "Search the web (static search results)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    timeout = 60.0

    read_only = True
    mutating = False
    category = "net"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from agent.tools.web import WebTool
        self._inner = WebTool(config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query") or ""
        if not query:
            return {"success": False, "error": "websearch requires query"}
        try:
            limit = max(1, min(int(params.get("count", 10)), 20))
        except (TypeError, ValueError):
            return {"success": False, "error": "count must be an integer"}
        return await self._inner.execute({"action": "search", "query": query, "limit": limit})


# ----------------------------------------------------------------------
# todowrite / todoread
# ----------------------------------------------------------------------
class TodoWriteTool(BaseTool):
    name = "todowrite"
    description = "Create or replace the todo list."
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    timeout = 10.0

    read_only = False
    mutating = True
    category = "plan"

    def __init__(self, session: Any = None):
        from agent.tools.todo import TodoTool, TodoItem
        self.session = session
        self._TodoItem = TodoItem
        sid = TodoTool._session_key(session)
        if sid not in TodoTool._shared_stores:
            TodoTool._shared_stores[sid] = {}
        self._store = TodoTool._shared_stores[sid]

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import uuid
        todos = params.get("todos")
        if todos is None:
            return {"success": False, "error": "todowrite requires 'todos' array"}
        if not isinstance(todos, list):
            return {"success": False, "error": "'todos' must be an array"}

        normalized: List[Dict[str, Any]] = []
        in_progress_count = 0
        for idx, t in enumerate(todos):
            if not isinstance(t, dict):
                return {"success": False, "error": f"todos[{idx}] must be object"}
            content = t.get("content")
            status = t.get("status")
            if not content or not isinstance(content, str):
                return {"success": False, "error": f"todos[{idx}].content is required"}
            if status not in ("pending", "in_progress", "completed", "cancelled"):
                return {"success": False, "error": f"todos[{idx}].status invalid"}
            if status == "in_progress":
                in_progress_count += 1
            prio = t.get("priority", "medium")
            prio_map = {"high": 1, "medium": 3, "low": 5}
            prio_int = prio if isinstance(prio, int) else prio_map.get(str(prio).lower(), 3)
            tid = t.get("id")
            if tid is not None and not isinstance(tid, str):
                tid = str(tid)
            normalized.append({"content": content, "status": status, "priority_str": prio if isinstance(prio, str) else None, "priority_int": prio_int, "id": tid})

        warning = None
        if in_progress_count > 1:
            warning = f"Expected exactly one in_progress, got {in_progress_count}."

        self._store.clear()
        result_todos: List[Dict[str, Any]] = []
        for n in normalized:
            internal_status = "done" if n["status"] == "completed" else n["status"]
            tid = n["id"] or str(uuid.uuid4())[:8]
            while tid in self._store:
                tid = str(uuid.uuid4())[:8]
            item = self._TodoItem(id=tid, title=n["content"], status=internal_status, priority=n["priority_int"])
            self._store[tid] = item
            if n["priority_str"]:
                prio_out = n["priority_str"]
            elif n["priority_int"] <= 2:
                prio_out = "high"
            elif n["priority_int"] >= 5:
                prio_out = "low"
            else:
                prio_out = "medium"
            result_todos.append({"content": n["content"], "status": n["status"], "priority": prio_out, "id": tid})

        resp: Dict[str, Any] = {"success": True, "count": len(result_todos), "todos": result_todos}
        if warning:
            resp["warning"] = warning
        return resp


class TodoReadTool(BaseTool):
    name = "todoread"
    description = "Read the current todo list."
    parameters = {"type": "object", "properties": {}, "required": []}
    timeout = 10.0

    read_only = True
    mutating = False
    category = "plan"

    def __init__(self, session: Any = None):
        from agent.tools.todo import TodoTool
        self.session = session
        sid = TodoTool._session_key(session)
        if sid not in TodoTool._shared_stores:
            TodoTool._shared_stores[sid] = {}
        self._store = TodoTool._shared_stores[sid]

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        items = sorted(self._store.values(), key=lambda i: (i.status != "in_progress", i.priority, i.created_at))
        todos: List[Dict[str, Any]] = []
        for it in items:
            status = "completed" if it.status == "done" else it.status
            if it.priority <= 2:
                prio = "high"
            elif it.priority >= 5:
                prio = "low"
            else:
                prio = "medium"
            todos.append({"content": it.title, "status": status, "priority": prio, "id": it.id})
        return {"success": True, "todos": todos, "count": len(todos)}


# ----------------------------------------------------------------------
# skill
# ----------------------------------------------------------------------
class SkillTool(BaseTool):
    name = "skill"
    description = "Load a skill file and return its content."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "skill": {"type": "string"},
        },
        "required": [],
    }
    timeout = 10.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        cfg = config or {}
        default_dirs = [".agent/skills", ".skills", os.path.join(Path.home(), ".config", "agent", "skills")]
        self.skill_dirs: List[str] = cfg.get("skill_dirs") or default_dirs

    def _skill_roots(self, base: Path) -> List[Path]:
        roots: List[Path] = []
        for d in self.skill_dirs:
            try:
                root = (base / d) if not os.path.isabs(d) else Path(d)
                roots.append(root.expanduser().resolve())
            except (OSError, ValueError):
                continue
        return roots

    def _safe_skill_name(self, skill_name: str) -> Optional[str]:
        name = (skill_name or "").strip()
        if not name or "\x00" in name:
            return None
        if os.path.isabs(name) or "/" in name or "\\" in name:
            return None
        if name in (".", "..") or ".." in Path(name).parts:
            return None
        return name

    def _candidate_paths(self, roots: List[Path], skill_name: str) -> List[Path]:
        out: List[Path] = []
        for root in roots:
            for candidate in (root / f"{skill_name}.md", root / skill_name / "SKILL.md"):
                try:
                    resolved = candidate.resolve()
                    if resolved.is_relative_to(root):
                        out.append(resolved)
                except (OSError, ValueError):
                    continue
        return out

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raw_name = params.get("name") or params.get("skill") or ""
        skill_name = self._safe_skill_name(str(raw_name))
        if skill_name is None:
            return {
                "success": False,
                "error": "invalid skill name; use a simple name without path separators",
            }

        if self.workspace:
            try:
                base = Path(str(self.workspace.get_project_dir()))
            except Exception:
                base = Path.cwd()
        else:
            base = Path.cwd()

        roots = self._skill_roots(base)
        candidates = self._candidate_paths(roots, skill_name)

        from agent.permissions.manager import _path_is_secret

        for p in candidates:
            try:
                if not p.exists() or not p.is_file() or _path_is_secret(str(p)):
                    continue
                if p.stat().st_size > 512 * 1024:
                    return {
                        "success": False,
                        "error": f"skill '{skill_name}' exceeds 512 KB",
                    }
                content = p.read_text(encoding="utf-8", errors="replace")
                return {"success": True, "skill": skill_name, "content": content, "path": str(p)}
            except Exception:
                continue

        available: List[str] = []
        for root in roots:
            try:
                if root.is_dir():
                    available.extend(x.stem for x in root.glob("*.md") if x.is_file())
            except Exception:
                pass
        return {
            "success": False,
            "error": f"Skill '{skill_name}' not found",
            "available": sorted(set(available)),
        }

# ----------------------------------------------------------------------
# question
# ----------------------------------------------------------------------
class QuestionTool(BaseTool):
    name = "question"
    description = "Ask the user questions during execution."
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "header": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label"],
                            },
                        },
                        "multiple": {"type": "boolean"},
                    },
                    "required": ["question", "header", "options"],
                },
            }
        },
        "required": ["questions"],
    }
    timeout = 600.0

    read_only = False
    mutating = False
    category = "ui"

    def __init__(self, input_handler: Any = None):
        self.input_handler = input_handler

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        questions = params.get("questions") or []
        if not questions:
            return {"success": False, "error": "question requires questions array"}
        if self.input_handler and hasattr(self.input_handler, "get_input"):
            answers: List[str] = []
            for q in questions:
                header = q.get("header", "")
                text = q.get("question", "")
                opts = q.get("options") or []
                prompt = f"{header}: {text}\n"
                for i, o in enumerate(opts, 1):
                    prompt += f"  {i}. {o.get('label','')} - {o.get('description','')}\n"
                prompt += "Your answer: "
                try:
                    ans = await self.input_handler.get_input(prompt=prompt)
                    answers.append(ans.strip() if ans else "")
                except Exception as e:
                    answers.append(f"error: {e}")
            return {"success": True, "answers": answers, "count": len(answers)}
        return {"success": True, "questions": questions, "needs_user_input": True}


# ----------------------------------------------------------------------
# lsp (stub)
# ----------------------------------------------------------------------
class LspTool(BaseTool):
    name = "lsp"
    description = "Code intelligence via LSP."
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol", "workspaceSymbol", "goToImplementation", "prepareCallHierarchy", "incomingCalls", "outgoingCalls"],
            },
            "filePath": {"type": "string"},
            "line": {"type": "integer"},
            "character": {"type": "integer"},
            "symbol": {"type": "string"},
            "query": {"type": "string"},
        },
        "required": ["operation"],
    }
    timeout = 30.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        self.config = config or {}
        env_flag = self.config.get("experimental_env_var", "AGENT_EXPERIMENTAL_LSP")
        self.enabled = bool(os.environ.get(env_flag))

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "LSP tool disabled."}
        op = params.get("operation")
        return {"success": False, "error": f"LSP '{op}' not yet implemented", "operation": op}