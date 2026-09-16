"""
Compatibility Tools

Wraps existing implementations to expose canonical tool names:
  bash, read, write, edit, grep, glob, apply_patch, todowrite, todoread,
  webfetch, websearch, skill, question, lsp

These keep original terminal/filesystem/search/patch/todo/web tools intact
so existing configs keep working.
"""

from __future__ import annotations

import glob as _glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.registry import BaseTool
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
        },
        "required": ["command"],
    }
    timeout = 120.0

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        from agent.tools.terminal import TerminalTool
        self._inner = TerminalTool(workspace, config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        if "command" in params:
            p["command"] = params["command"]
        # workdir is an alias for cwd
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
        "required": ["filePath"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        from agent.tools.filesystem import FileSystemTool
        self._inner = FileSystemTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("filePath") or params.get("path") or ""
        if not path:
            return {"success": False, "error": "read requires filePath"}
        limit = params.get("limit")
        offset = params.get("offset", 0)
        res = await self._inner.execute({"action": "read", "path": path})
        if not res.get("success"):
            return res
        content = res.get("content", "")
        if limit is not None or offset:
            lines = content.splitlines()
            try:
                off = int(offset or 0)
                lim = int(limit) if limit is not None else None
                sliced = lines[off: off + lim if lim is not None else None]
                content = "\n".join(sliced)
            except Exception:
                pass
            res["content"] = content
        return res


class WriteTool(BaseTool):
    name = "write"
    description = "Create or overwrite a file."
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
            "include_diff": {
                "type": "boolean",
                "default": False,
                "description": "Include before/after content in the response for diff rendering",
            },
        },
        "required": ["filePath", "content"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        from agent.tools.filesystem import FileSystemTool
        self._inner = FileSystemTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("filePath") or params.get("path") or ""
        content = params.get("content", "")
        if not path:
            return {"success": False, "error": "write requires filePath"}
        inner_params: Dict[str, Any] = {
            "action": "write", "path": path, "content": content,
        }
        if "include_diff" in params:
            inner_params["include_diff"] = bool(params["include_diff"])
        return await self._inner.execute(inner_params)


class EditTool(BaseTool):
    name = "edit"
    description = "Modify existing file via exact string replacement."
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "path": {"type": "string"},
            "oldString": {"type": "string", "description": "Text to replace"},
            "old_string": {"type": "string"},
            "newString": {"type": "string", "description": "Replacement"},
            "new_string": {"type": "string"},
            "replaceAll": {"type": "boolean", "default": False},
            "include_diff": {
                "type": "boolean",
                "default": False,
                "description": "Include before/after content in the response for diff rendering",
            },
        },
        "required": ["filePath"],
    }
    timeout = 30.0

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
        if old == "" and new == "":
            return {"success": False, "error": "edit requires oldString/newString"}
        p: Dict[str, Any] = {"action": "edit", "path": path, "old": old, "new": new}
        if replace_all:
            p["all"] = True
        if "include_diff" in params:
            p["include_diff"] = bool(params["include_diff"])
        return await self._inner.execute(p)


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = "Apply a patch using the marker format (*** Add/Update/Delete File:)."
    parameters = {
        "type": "object",
        "properties": {
            "patchText": {
                "type": "string",
                "description": "Patch with *** markers or unified diff",
            },
            "patch": {"type": "string", "description": "Alias for patchText"},
            "diff": {"type": "string", "description": "Alias for patchText"},
            "include_diff": {
                "type": "boolean",
                "default": False,
                "description": "Include before/after content in the response for diff rendering",
            },
        },
        "required": ["patchText"],
    }
    timeout = 30.0

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
        include_diff = bool(params.get("include_diff", False))
        if "***" in patch_text:
            return await self._apply_marker_patch(patch_text, include_diff=include_diff)
        inner_params: Dict[str, Any] = {"action": "apply", "diff": patch_text}
        if include_diff:
            inner_params["include_diff"] = True
        return await self._inner.execute(inner_params)

    # ------------------------------------------------------------------

    async def _apply_marker_patch(
        self, patch_text: str, include_diff: bool = False
    ) -> Dict[str, Any]:
        """
        Parse and apply a patch written with *** markers. Handles context
        lines (' '), deletions ('-'), and additions ('+') on Update
        sections. Add sections use every line as content; Delete removes
        the file.
        """
        sections = self._split_sections(patch_text)
        if not sections:
            return {"success": False, "error": "No file sections found in patch"}

        results: List[Dict[str, Any]] = []
        for section in sections:
            try:
                result = await self._apply_section(section, include_diff=include_diff)
            except Exception as e:
                result = {"success": False, "path": section["path"], "error": str(e)}
            results.append(result)

        ok = all(r.get("success") for r in results)
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
                current = {
                    "action": "add",
                    "path": line.split(":", 1)[1].strip(),
                    "lines": [],
                }
                continue
            if line.startswith("*** Update File:"):
                if current:
                    sections.append(current)
                current = {
                    "action": "update",
                    "path": line.split(":", 1)[1].strip(),
                    "lines": [],
                    "move_to": None,
                }
                continue
            if line.startswith("*** Delete File:"):
                if current:
                    sections.append(current)
                current = {
                    "action": "delete",
                    "path": line.split(":", 1)[1].strip(),
                    "lines": [],
                }
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
        self, section: Dict[str, Any], include_diff: bool = False
    ) -> Dict[str, Any]:
        action = section["action"]
        path = section["path"]
        target = self._inner._resolve(path)  # type: ignore[attr-defined]

        if action == "add":
            content = self._decode_add_lines(section["lines"])
            target.parent.mkdir(parents=True, exist_ok=True)
            before = ""
            existed = target.exists()
            if include_diff and existed and target.is_file():
                try:
                    before = target.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    before = ""
            target.write_text(content, encoding="utf-8")
            entry: Dict[str, Any] = {
                "success": True,
                "path": str(target),
                "action": "add",
                "bytes_written": len(content.encode("utf-8")),
            }
            if include_diff:
                entry["before"] = before
                entry["after"] = content
                entry["created"] = not existed
            return entry

        if action == "delete":
            if not target.exists():
                return {"success": False, "path": str(target), "error": "not found"}
            before = ""
            if include_diff and target.is_file():
                try:
                    before = target.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    before = ""
            if target.is_dir():
                import shutil
                shutil.rmtree(target)
            else:
                target.unlink()
            entry: Dict[str, Any] = {
                "success": True, "path": str(target), "action": "delete",
            }
            if include_diff:
                entry["before"] = before
                entry["after"] = ""
            return entry

        if action == "update":
            if not target.exists():
                return {"success": False, "path": str(target), "error": "not found"}

            original = target.read_text(encoding="utf-8", errors="replace")
            updated = self._apply_update_hunks(original, section["lines"])

            final_path = target
            if section.get("move_to"):
                final_path = self._inner._resolve(section["move_to"])  # type: ignore[attr-defined]
                final_path.parent.mkdir(parents=True, exist_ok=True)

            final_path.write_text(updated, encoding="utf-8")
            if final_path != target and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass

            entry: Dict[str, Any] = {
                "success": True,
                "path": str(final_path),
                "action": "update",
                "bytes_before": len(original),
                "bytes_after": len(updated),
            }
            if include_diff:
                entry["before"] = original
                entry["after"] = updated
            return entry

        return {"success": False, "path": path, "error": f"Unknown action: {action}"}

    def _decode_add_lines(self, lines: List[str]) -> str:
        """Add sections: every line is content. A leading '+' is optional."""
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
        """
        Apply a sequence of context/delete/add lines to the original text.
        Uses a simple anchor-match: context lines locate the position, then
        '-' lines are removed and '+' lines are inserted.
        """
        src_lines = original.splitlines(keepends=True)

        chunks: List[List[str]] = []
        current: List[str] = []
        for line in hunk_lines:
            if line.startswith(("+", "-", " ")):
                current.append(line)
            else:
                if current:
                    chunks.append(current)
                    current = []
                current.append(" " + line)
        if current:
            chunks.append(current)

        for chunk in chunks:
            old_lines = [c[1:] for c in chunk if c.startswith((" ", "-"))]
            new_lines = [c[1:] for c in chunk if c.startswith((" ", "+"))]
            if not old_lines:
                src_lines.extend(l + "\n" for l in new_lines if not l.endswith("\n"))
                continue

            start = self._find_chunk(src_lines, old_lines)
            if start is None:
                raise ValueError(
                    f"could not find hunk anchor: {old_lines[:1]!r}"
                )
            replacement = [
                (l if l.endswith("\n") else l + "\n") for l in new_lines
            ]
            src_lines[start:start + len(old_lines)] = replacement

        return "".join(src_lines)

    def _find_chunk(self, src_lines: List[str], old_lines: List[str]) -> Optional[int]:
        target = [l.rstrip("\n") for l in old_lines]
        n = len(target)
        for i in range(0, len(src_lines) - n + 1):
            window = [l.rstrip("\n") for l in src_lines[i:i + n]]
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
            "path": {"type": "string", "description": "Root path/glob", "default": "."},
            "include": {"type": "string", "description": "File glob filter"},
            "glob": {"type": "string"},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["pattern"],
    }
    timeout = 60.0

    def __init__(self, workspace: Any = None):
        from agent.tools.search import SearchTool
        self._inner = SearchTool(workspace)
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = params.get("pattern") or ""
        if not pattern:
            return {"success": False, "error": "grep requires pattern"}
        glob_pat = params.get("include") or params.get("glob") or "*"
        root = params.get("path") or (
            str(self.workspace.get_project_dir()) if self.workspace else "."
        )
        return await self._inner.execute(
            {
                "query": pattern,
                "path": root,
                "glob": glob_pat,
                "regex": True,
                "max_results": int(params.get("max_results", 100)),
            }
        )


class GlobTool(BaseTool):
    name = "glob"
    description = "Find files by glob pattern, newest first."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Root dir", "default": "."},
        },
        "required": ["pattern"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = params.get("pattern") or ""
        if not pattern:
            return {"success": False, "error": "glob requires pattern"}
        root = params.get("path") or (
            str(self.workspace.get_project_dir()) if self.workspace else "."
        )
        pattern_full = os.path.join(root, pattern)
        matches = _glob.glob(pattern_full, recursive=True)

        root_resolved = Path(root).resolve()
        entries: List[Tuple[str, float]] = []
        for m in matches:
            try:
                resolved = Path(m).resolve()
                rel = str(resolved.relative_to(root_resolved))
            except Exception:
                rel = m
            try:
                mtime = os.path.getmtime(m)
            except OSError:
                mtime = 0.0
            entries.append((rel, mtime))

        entries.sort(key=lambda t: t[1], reverse=True)
        return {
            "success": True,
            "pattern": pattern,
            "matches": [e[0] for e in entries],
            "count": len(entries),
        }


# ----------------------------------------------------------------------
# webfetch / websearch split
# ----------------------------------------------------------------------
class WebFetchTool(BaseTool):
    name = "webfetch"
    description = "Fetch web content from a URL (static HTML)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "maxChars": {"type": "integer", "default": 500000},
            "extractMode": {"type": "string"},
        },
        "required": ["url"],
    }
    timeout = 60.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from agent.tools.web import WebTool
        self._inner = WebTool(config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url") or ""
        if not url:
            return {"success": False, "error": "webfetch requires url"}
        max_bytes = int(params.get("maxChars", 500000))
        return await self._inner.execute(
            {"action": "fetch", "url": url, "max_bytes": max_bytes}
        )


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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from agent.tools.web import WebTool
        self._inner = WebTool(config)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query") or ""
        if not query:
            return {"success": False, "error": "websearch requires query"}
        return await self._inner.execute(
            {
                "action": "search",
                "query": query,
                "limit": int(params.get("count", 10)),
            }
        )


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
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    timeout = 10.0

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
                return {
                    "success": False,
                    "error": (
                        f"todos[{idx}].status must be "
                        "pending|in_progress|completed|cancelled, "
                        f"got {status!r}"
                    ),
                }
            if status == "in_progress":
                in_progress_count += 1

            prio = t.get("priority", "medium")
            prio_map = {"high": 1, "medium": 3, "low": 5}
            if isinstance(prio, int):
                prio_int = prio
            else:
                prio_int = prio_map.get(str(prio).lower(), 3)

            tid = t.get("id")
            if tid is not None and not isinstance(tid, str):
                tid = str(tid)

            normalized.append(
                {
                    "content": content,
                    "status": status,
                    "priority_str": prio if isinstance(prio, str) else None,
                    "priority_int": prio_int,
                    "id": tid,
                }
            )

        warning: Optional[str] = None
        if in_progress_count > 1:
            warning = (
                f"Expected exactly one in_progress, got {in_progress_count}. "
                "Keep exactly one task in_progress at a time."
            )

        # Full-list replacement: clear the store then rebuild.
        self._store.clear()
        result_todos: List[Dict[str, Any]] = []
        for n in normalized:
            internal_status = "done" if n["status"] == "completed" else n["status"]
            tid = n["id"] or str(uuid.uuid4())[:8]
            while tid in self._store:
                tid = str(uuid.uuid4())[:8]

            item = self._TodoItem(
                id=tid,
                title=n["content"],
                status=internal_status,
                priority=n["priority_int"],
            )
            self._store[tid] = item

            if n["priority_str"]:
                prio_out = n["priority_str"]
            elif n["priority_int"] <= 2:
                prio_out = "high"
            elif n["priority_int"] >= 5:
                prio_out = "low"
            else:
                prio_out = "medium"

            result_todos.append(
                {
                    "content": n["content"],
                    "status": n["status"],
                    "priority": prio_out,
                    "id": tid,
                }
            )

        resp: Dict[str, Any] = {
            "success": True,
            "count": len(result_todos),
            "todos": result_todos,
        }
        if warning:
            resp["warning"] = warning
        return resp


class TodoReadTool(BaseTool):
    name = "todoread"
    description = "Read the current todo list."
    parameters = {"type": "object", "properties": {}, "required": []}
    timeout = 10.0

    def __init__(self, session: Any = None):
        from agent.tools.todo import TodoTool
        self.session = session
        sid = TodoTool._session_key(session)
        if sid not in TodoTool._shared_stores:
            TodoTool._shared_stores[sid] = {}
        self._store = TodoTool._shared_stores[sid]

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        items = sorted(
            self._store.values(),
            key=lambda i: (i.status != "in_progress", i.priority, i.created_at),
        )
        todos: List[Dict[str, Any]] = []
        for it in items:
            status = "completed" if it.status == "done" else it.status
            if it.priority <= 2:
                prio = "high"
            elif it.priority >= 5:
                prio = "low"
            else:
                prio = "medium"
            todos.append(
                {"content": it.title, "status": status, "priority": prio, "id": it.id}
            )
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
            "name": {"type": "string", "description": "Skill name"},
            "skill": {"type": "string"},
        },
        "required": [],
    }
    timeout = 10.0

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        cfg = config or {}
        # Directories searched, in order. Overridable via config.
        default_dirs = [
            ".agent/skills",
            ".skills",
            os.path.join(Path.home(), ".config", "agent", "skills"),
        ]
        self.skill_dirs: List[str] = cfg.get("skill_dirs") or default_dirs

    def _candidate_paths(self, base: Path, skill_name: str) -> List[Path]:
        out: List[Path] = []
        for d in self.skill_dirs:
            root = (base / d) if not os.path.isabs(d) else Path(d)
            out.append(root / f"{skill_name}.md")
            out.append(root / skill_name / "SKILL.md")
        return out

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        skill_name = params.get("name") or params.get("skill") or ""

        candidates: List[Path] = []
        if self.workspace:
            base = Path(str(self.workspace.get_project_dir()))
            candidates.extend(self._candidate_paths(base, skill_name))
        else:
            candidates.extend(self._candidate_paths(Path.cwd(), skill_name))

        if skill_name and os.path.sep in skill_name:
            candidates.append(Path(skill_name))

        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    return {
                        "success": True,
                        "skill": skill_name,
                        "content": content,
                        "path": str(p),
                    }
            except Exception:
                continue

        available: List[str] = []
        for d in self.skill_dirs:
            try:
                root = (Path.cwd() / d) if not os.path.isabs(d) else Path(d)
                if root.exists():
                    available.extend([x.stem for x in root.glob("*.md")])
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

        return {
            "success": True,
            "questions": questions,
            "needs_user_input": True,
        }


# ----------------------------------------------------------------------
# lsp (stub)
# ----------------------------------------------------------------------
class LspTool(BaseTool):
    name = "lsp"
    description = "Code intelligence via LSP (goToDefinition, findReferences, hover)."
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "goToDefinition", "findReferences", "hover",
                    "documentSymbol", "workspaceSymbol",
                    "goToImplementation", "prepareCallHierarchy",
                    "incomingCalls", "outgoingCalls",
                ],
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

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        self.config = config or {}
        env_flag = self.config.get("experimental_env_var", "AGENT_EXPERIMENTAL_LSP")
        self.enabled = bool(os.environ.get(env_flag))

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "success": False,
                "error": (
                    "LSP tool disabled. Set "
                    f"{self.config.get('experimental_env_var','AGENT_EXPERIMENTAL_LSP')}=1"
                ),
            }
        op = params.get("operation")
        return {
            "success": False,
            "error": f"LSP '{op}' not yet implemented",
            "operation": op,
        }