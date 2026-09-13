"""
Opencode Compatibility Tools

Wraps existing implementations to expose opencode-canonical tool names:
  bash, read, write, edit, grep, glob, apply_patch, todowrite, todoread,
  webfetch, websearch, skill, question, lsp

These keep original `terminal/filesystem/search/patch/todo/web` intact
so existing configs keep working — new names satisfy opencode permission keys.
"""
from __future__ import annotations

import os
import json
import glob as _glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# bash -> terminal
# ----------------------------------------------------------------------
class BashTool(BaseTool):
    name = "bash"
    description = "Execute shell commands (opencode-compatible alias for terminal)."
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
        # opencode bash uses workdir, our terminal uses cwd
        if "workdir" in params and "cwd" not in params:
            params = {**params, "cwd": params["workdir"]}
        # strip opencode-only keys
        p = {k: v for k, v in params.items() if k in ("command", "cwd", "timeout", "shell", "env")}
        if "command" not in p and "command" in params:
            p["command"] = params["command"]
        return await self._inner.execute(p)


# ----------------------------------------------------------------------
# read
# ----------------------------------------------------------------------
class ReadTool(BaseTool):
    name = "read"
    description = "Read file contents (opencode-compatible). Supports line ranges."
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
    description = "Create or overwrite a file (opencode-compatible)."
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
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
        return await self._inner.execute({"action": "write", "path": path, "content": content})


class EditTool(BaseTool):
    name = "edit"
    description = "Modify existing file via exact string replacement (opencode-compatible)."
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
        # filesystem edit needs old/new
        p: Dict[str, Any] = {"action": "edit", "path": path, "old": old, "new": new}
        if replace_all:
            p["all"] = True
        return await self._inner.execute(p)


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = "Apply patch (opencode-compatible, aliases patch tool)."
    parameters = {
        "type": "object",
        "properties": {
            " patchText": {"type": "string"},
            "patchText": {"type": "string", "description": "Patch with *** markers or unified diff"},
            "patch": {"type": "string"},
            "filePath": {"type": "string"},
        },
        "required": [],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        from agent.tools.patch import PatchTool
        self._inner = PatchTool(workspace)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # opencode apply_patch uses output.args.patchText with embedded file markers
        patch_text = params.get("patchText") or params.get("patch") or params.get("diff") or ""
        if not patch_text:
            return {"success": False, "error": "apply_patch requires patchText"}
        # Detect opencode marker format
        if "***" in patch_text:
            return await self._apply_opencode_markers(patch_text)
        return await self._inner.execute({"action": "apply", "diff": patch_text})

    async def _apply_opencode_markers(self, patch_text: str) -> Dict[str, Any]:
        lines = patch_text.splitlines()
        results = []
        current_path: Optional[str] = None
        current_action: Optional[str] = None
        buf: List[str] = []

        async def flush():
            nonlocal buf, current_path, current_action
            if not current_path or not buf:
                buf = []
                return
            content = "\n".join(buf)
            try:
                target = self._inner._resolve(current_path)  # type: ignore
                if current_action == "delete":
                    if target.exists():
                        if target.is_dir():
                            import shutil
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                        results.append({"success": True, "path": current_path, "action": "delete"})
                    else:
                        results.append({"success": False, "path": current_path, "error": "not found"})
                elif current_action == "add":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    results.append({"success": True, "path": current_path, "action": "add"})
                else:  # update
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    results.append({"success": True, "path": current_path, "action": "update"})
            except Exception as e:
                results.append({"success": False, "path": current_path, "error": str(e)})
            buf = []

        for line in lines:
            if line.startswith("*** Add File:"):
                await flush()
                current_path = line.split(":", 1)[1].strip()
                current_action = "add"
                buf = []
            elif line.startswith("*** Update File:"):
                await flush()
                current_path = line.split(":", 1)[1].strip()
                current_action = "update"
                buf = []
            elif line.startswith("*** Move to:"):
                await flush()
                current_path = line.split(":", 1)[1].strip()
                current_action = "update"
                buf = []
            elif line.startswith("*** Delete File:"):
                await flush()
                current_path = line.split(":", 1)[1].strip()
                current_action = "delete"
                buf = []
            elif line.startswith("*** End"):
                await flush()
                current_path = None
                current_action = None
            else:
                if current_action != "delete":
                    buf.append(line)
        await flush()
        ok = all(r.get("success") for r in results) if results else False
        return {"success": ok, "results": results}


# ----------------------------------------------------------------------
# grep / glob
# ----------------------------------------------------------------------
class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents with regex (opencode-compatible)."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Root path/glob", "default": "."},
            "include": {"type": "string", "description": "File glob filter"},
            "glob": {"type": "string"},
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
        root = params.get("path") or (str(self.workspace.get_project_dir()) if self.workspace else ".")
        return await self._inner.execute({
            "query": pattern,
            "path": root,
            "glob": glob_pat,
            "regex": True,
            "max_results": params.get("max_results", 100),
        })


class GlobTool(BaseTool):
    name = "glob"
    description = "Find files by glob pattern (opencode-compatible)."
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
        root = params.get("path") or (str(self.workspace.get_project_dir()) if self.workspace else ".")
        # Respect .gitignore not needed for stub; simple glob
        import fnmatch, os
        pattern_full = os.path.join(root, pattern)
        matches = _glob.glob(pattern_full, recursive=True)
        # Normalize to relative if inside root
        rel = []
        rp = Path(root).resolve()
        for m in matches:
            try:
                rel.append(str(Path(m).resolve().relative_to(rp)))
            except Exception:
                rel.append(m)
        rel = sorted(rel, key=lambda p: os.path.getmtime(os.path.join(root, p)) if os.path.exists(os.path.join(root, p)) else 0, reverse=True)
        return {"success": True, "pattern": pattern, "matches": rel, "count": len(rel)}


# ----------------------------------------------------------------------
# webfetch / websearch split
# ----------------------------------------------------------------------
class WebFetchTool(BaseTool):
    name = "webfetch"
    description = "Fetch web content from URL (opencode-compatible)."
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
        return await self._inner.execute({"action": "fetch", "url": url, "max_bytes": max_bytes})


class WebSearchTool(BaseTool):
    name = "websearch"
    description = "Search the web (opencode-compatible)."
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
        return await self._inner.execute({"action": "search", "query": query, "limit": int(params.get("count", 10))})


# ----------------------------------------------------------------------
# todowrite / todoread
# ----------------------------------------------------------------------
class TodoWriteTool(BaseTool):
    name = "todowrite"
    description = "Create/update todo list (opencode-compatible)."
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

    # Shared store awareness: TodoTool now uses class-level dict, so this wrapper will see same data

    def __init__(self, session: Any = None):
        from agent.tools.todo import TodoTool, TodoItem
        self.session = session
        self._TodoItem = TodoItem
        # Use same shared dict as TodoTool for this session
        sid = TodoTool._session_key(session)
        if sid not in TodoTool._shared_stores:
            TodoTool._shared_stores[sid] = {}
        self._store = TodoTool._shared_stores[sid]

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agent.tools.todo import TodoTool
        import time, uuid

        todos = params.get("todos")
        if todos is None:
            return {"success": False, "error": "todowrite requires 'todos' array"}
        if not isinstance(todos, list):
            return {"success": False, "error": "'todos' must be an array"}

        # Validate items and normalize
        normalized = []
        in_progress_count = 0
        for idx, t in enumerate(todos):
            if not isinstance(t, dict):
                return {"success": False, "error": f"todos[{idx}] must be object"}
            content = t.get("content")
            status = t.get("status")
            if not content or not isinstance(content, str):
                return {"success": False, "error": f"todos[{idx}].content is required"}
            if status not in ("pending", "in_progress", "completed", "cancelled"):
                return {"success": False, "error": f"todos[{idx}].status must be pending|in_progress|completed|cancelled, got {status!r}"}
            if status == "in_progress":
                in_progress_count += 1
            # priority mapping
            prio = t.get("priority", "medium")
            if isinstance(prio, int):
                prio_int = prio
            else:
                prio_map = {"high": 1, "medium": 3, "low": 5}
                prio_int = prio_map.get(str(prio).lower(), 3)
            # id handling: use provided id if present, else generate
            tid = t.get("id")
            if tid and not isinstance(tid, str):
                tid = str(tid)
            normalized.append({"content": content, "status": status, "priority": prio_int, "id": tid})

        # Subtle validation: exactly one in_progress while work remains (warn, not hard fail)
        # Opencode guideline: keep exactly one in_progress. If multiple, we still accept but report warning.
        warning = None
        if in_progress_count > 1:
            warning = f"Expected exactly one in_progress, got {in_progress_count}. Keep exactly one task in_progress at a time."

        # Sync store: opencode replaces entire list — implement as full sync preserving ids when possible
        # Clear and recreate with preserved ids
        self._store.clear()
        result_todos = []
        for n in normalized:
            # map completed -> done for internal status, in_progress stays, cancelled stays
            internal_status = n["status"]
            if internal_status == "completed":
                internal_status = "done"
            tid = n["id"] or str(uuid.uuid4())[:8]
            # ensure unique
            while tid in self._store:
                tid = str(uuid.uuid4())[:8]
            item = self._TodoItem(
                id=tid,
                title=n["content"],
                status=internal_status,
                priority=n["priority"],
            )
            self._store[tid] = item
            # reflect back with opencode status (completed not done)
            out_status = n["status"]
            result_todos.append({"content": n["content"], "status": out_status, "priority": n["priority"] if isinstance(n["priority"], str) else ("high" if n["priority"]==1 else "low" if n["priority"]==5 else "medium"), "id": tid})

        resp: Dict[str, Any] = {"success": True, "count": len(result_todos), "todos": result_todos}
        if warning:
            resp["warning"] = warning
        return resp


class TodoReadTool(BaseTool):
    name = "todoread"
    description = "Read current todo list (opencode-compatible)."
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
        # Return in opencode format: [{content, status, priority, id}]
        items = sorted(self._store.values(), key=lambda i: (i.status != "in_progress", i.priority, i.created_at))
        todos = []
        for it in items:
            status = it.status
            if status == "done":
                status = "completed"
            prio = "medium"
            if it.priority == 1:
                prio = "high"
            elif it.priority == 5:
                prio = "low"
            elif it.priority == 2:
                prio = "high"
            todos.append({"content": it.title, "status": status, "priority": prio, "id": it.id})
        return {"success": True, "todos": todos, "count": len(todos)}


# ----------------------------------------------------------------------
# skill
# ----------------------------------------------------------------------
class SkillTool(BaseTool):
    name = "skill"
    description = "Load a skill (SKILL.md) and return its content."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name"}, "skill": {"type": "string"}},
        "required": [],
    }
    timeout = 10.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        skill_name = params.get("name") or params.get("skill") or ""
        # search dirs: .opencode/skills, .agent/skills, src/.agent/skills, ~/.config/opencode/skills
        candidates = []
        if self.workspace:
            base = Path(str(self.workspace.get_project_dir()))
            candidates.extend([
                base / ".opencode" / "skills" / f"{skill_name}.md",
                base / ".opencode" / "skills" / skill_name / "SKILL.md",
                base / ".agent" / "skills" / f"{skill_name}.md",
                Path.home() / ".config" / "opencode" / "skills" / f"{skill_name}.md",
            ])
        # also try skill_name as path
        if skill_name:
            candidates.append(Path(skill_name))
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    return {"success": True, "skill": skill_name, "content": content, "path": str(p)}
            except Exception:
                continue
        # fallback: list available
        available = []
        for base in candidates:
            try:
                parent = base.parent
                if parent.exists():
                    available.extend([x.name for x in parent.glob("*.md")])
            except Exception:
                pass
        return {"success": False, "error": f"Skill '{skill_name}' not found", "available": sorted(set(available))}


# ----------------------------------------------------------------------
# question
# ----------------------------------------------------------------------
class QuestionTool(BaseTool):
    name = "question"
    description = "Ask the user questions during execution (opencode-compatible)."
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
        # If input_handler supports question flow, delegate
        if self.input_handler and hasattr(self.input_handler, "get_input"):
            # Fallback: render questions and ask via input_handler sequentially
            answers = []
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
        # No handler: return questions for caller to display
        return {"success": True, "questions": questions, "needs_user_input": True}


# ----------------------------------------------------------------------
# lsp (stub, enables permission key, returns not-enabled info)
# ----------------------------------------------------------------------
class LspTool(BaseTool):
    name = "lsp"
    description = "Code intelligence via LSP (goToDefinition, findReferences, hover, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol", "workspaceSymbol", "goToImplementation", "prepareCallHierarchy", "incomingCalls", "outgoingCalls"]},
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
        self.enabled = bool(os.environ.get("OPENCODE_EXPERIMENTAL_LSP_TOOL") or os.environ.get("OPENCODE_EXPERIMENTAL"))

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "LSP tool disabled. Set OPENCODE_EXPERIMENTAL_LSP_TOOL=true", "hint": "export OPENCODE_EXPERIMENTAL_LSP_TOOL=1"}
        op = params.get("operation")
        return {"success": False, "error": f"LSP '{op}' not yet implemented", "operation": op}
