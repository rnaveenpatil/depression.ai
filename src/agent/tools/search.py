"""
Search Tool - Regex / glob / text search across the project.

Backends:
    - "ripgrep"  if `rg` is available
    - "grep"     fallback
    - "python"   pure-Python fallback
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class SearchTool(BaseTool):
    name = "search"
    description = (
        "Search for text or regex patterns across project files. "
        "Supports glob filtering, case-insensitive search, and file name search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text or regex to search for"},
            "path": {"type": "string", "description": "Root path", "default": "."},
            "glob": {"type": "string", "description": "File glob filter", "default": "*"},
            "regex": {"type": "boolean", "default": False},
            "case_sensitive": {"type": "boolean", "default": False},
            "name_only": {"type": "boolean", "default": False, "description": "Search file names only"},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["query"],
    }
    timeout = 60.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Missing query"}

        root = params.get("path") or (
            str(self.workspace.get_project_dir()) if self.workspace else os.getcwd()
        )

        if shutil.which("rg"):
            return await self._ripgrep(query, root, params)
        return await self._python_search(query, root, params)

    # ------------------------------------------------------------------

    async def _ripgrep(
        self, query: str, root: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        args = ["rg", "--line-number", "--no-heading", "--color=never", "--hidden", "--glob", "!.git"]
        if not params.get("case_sensitive"):
            args.append("-i")
        if not params.get("regex"):
            args.append("-F")
        if params.get("name_only"):
            args.append("--files")
        glob = params.get("glob")
        if glob and glob != "*":
            args.extend(["--glob", glob])
        # Respect .gitignore already via rg defaults; cap results
        args.extend(["--max-count", str(int(params.get("max_results", 100)))])
        args.extend([query, root])

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            # If rg succeeded, parse
            if proc.returncode not in (0, 1):  # 1 = no matches, still success
                return await self._python_search(query, root, params)
        except Exception:
            return await self._python_search(query, root, params)

        max_results = int(params.get("max_results", 100))
        matches: List[Dict[str, Any]] = []
        for line in stdout.decode(errors="replace").splitlines():
            if len(matches) >= max_results:
                break
            if params.get("name_only"):
                matches.append({"path": line})
            else:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    matches.append({
                        "path": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else None,
                        "text": parts[2],
                    })

        return {"success": True, "matches": matches, "count": len(matches), "backend": "ripgrep"}

    async def _python_search(
        self, query: str, root: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        max_results = int(params.get("max_results", 100))
        glob = params.get("glob") or "*"
        case_sensitive = bool(params.get("case_sensitive"))
        is_regex = bool(params.get("regex"))
        name_only = bool(params.get("name_only"))

        try:
            pattern = re.compile(query if is_regex else re.escape(query),
                                0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}"}

        matches: List[Dict[str, Any]] = []
        root_path = Path(root)

        def _scan():
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
                for fn in filenames:
                    if not fnmatch.fnmatch(fn, glob):
                        continue
                    p = Path(dirpath) / fn
                    if name_only:
                        if pattern.search(fn):
                            matches.append({"path": str(p)})
                    else:
                        try:
                            with open(p, "r", encoding="utf-8", errors="replace") as f:
                                for i, line in enumerate(f, 1):
                                    if pattern.search(line):
                                        matches.append({
                                            "path": str(p),
                                            "line": i,
                                            "text": line.rstrip("\n"),
                                        })
                                        if len(matches) >= max_results:
                                            return
                        except Exception:
                            continue
                    if len(matches) >= max_results:
                        return

        await asyncio.to_thread(_scan)
        return {"success": True, "matches": matches, "count": len(matches), "backend": "python"}