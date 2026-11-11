"""
Search Tool - Regex / glob / text search across the project.

Backends:
    - "ripgrep"  if `rg` is available
    - "python"   pure-Python fallback
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.file_walk import WalkStats, glob_match, walk_paths
from agent.utils.logging import get_logger

logger = get_logger(__name__)


RG_IGNORE_GLOBS = (
    "!.git/**", "!**/.git/**",
    "!**/.env", "!**/.env.*", "!**/*.env",
    "!**/.aws/**", "!**/.ssh/**", "!**/.npmrc", "!**/.pypirc",
    "!**/.netrc", "!**/.git-credentials", "!**/id_rsa*",
    "!**/id_ed25519*", "!**/*.pem", "!**/*.key",
    "!**/node_modules/**", "!**/bower_components/**",
    "!**/__pycache__/**", "!**/.pytest_cache/**",
    "!**/.venv/**", "!**/venv/**", "!**/env/**",
    "!**/.tox/**", "!**/.nox/**",
    "!**/dist/**", "!**/build/**", "!**/target/**",
    "!**/coverage/**", "!**/htmlcov/**",
    "!**/site-packages/**", "!**/lib/python*/**",
)


_SECRET_BASENAMES = {
    ".env", ".npmrc", ".pypirc", ".netrc", ".git-credentials",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials",
}


def resolve_search_root(workspace: Any, raw: str) -> Path:
    """Resolve and constrain a caller-supplied search root."""
    candidate = Path(str(raw)).expanduser()
    if workspace is not None:
        base = Path(str(workspace.get_project_dir())).expanduser().resolve()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        if resolved != base and not resolved.is_relative_to(base):
            raise ValueError("search root must stay inside the workspace")
        return resolved
    return candidate.resolve()


def _is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    if name in _SECRET_BASENAMES:
        return True
    if name.startswith(".env.") or name.endswith(".env"):
        return True
    if name.endswith((".pem", ".key")) and name not in {"public.pem"}:
        return True
    try:
        from agent.permissions.manager import _path_is_secret
        return _path_is_secret(str(path))
    except Exception:
        return False


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
            "max_seconds": {"type": "number", "default": 10.0},
        },
        "required": ["query"],
    }
    timeout = 60.0

    read_only = True
    mutating = False
    category = "inspect"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Missing query"}

        raw_root = params.get("path") or (
            str(self.workspace.get_project_dir()) if self.workspace else os.getcwd()
        )
        try:
            root = str(resolve_search_root(self.workspace, str(raw_root)))
        except Exception as e:
            return {"success": False, "error": f"Invalid search root: {e}"}

        # name_only relies on Python filtering even when rg is present, because
        # `rg --files` ignores the query entirely.
        if shutil.which("rg") and not params.get("name_only"):
            rg_result = await self._ripgrep(query, root, params)
            if rg_result is not None:
                return rg_result

        return await self._python_search(query, root, params)

    # ------------------------------------------------------------------

    async def _ripgrep(
        self, query: str, root: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            max_results = max(1, min(int(params.get("max_results", 100)), 1000))
            max_seconds = max(0.1, min(float(params.get("max_seconds", 10.0)), 60.0))
        except (TypeError, ValueError):
            return None

        args = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--hidden",
        ]
        for ignored in RG_IGNORE_GLOBS:
            args.extend(["--glob", ignored])
        if not params.get("case_sensitive"):
            args.append("-i")
        if not params.get("regex"):
            args.append("-F")
        glob = params.get("glob")
        if glob and glob != "*":
            args.extend(["--glob", glob])
        # Per-file cap only; the global cap is enforced by the reader below.
        args.extend(["--max-count", str(max_results)])
        args.extend(["--", query, root])

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.debug("ripgrep failed (%s); falling back to Python", e)
            return None

        matches: List[Dict[str, Any]] = []

        async def _consume() -> None:
            assert proc.stdout is not None
            while len(matches) < max_results:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip("\n")
                parts = line.split(":", 2)
                if len(parts) == 3:
                    matches.append(
                        {
                            "path": parts[0],
                            "line": int(parts[1]) if parts[1].isdigit() else None,
                            "text": parts[2],
                        }
                    )
            if len(matches) >= max_results:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()

        try:
            await asyncio.wait_for(_consume(), timeout=max_seconds)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
            return None
        except Exception as e:
            logger.debug("ripgrep reader failed (%s); falling back to Python", e)
            return None

        # rg: 0 = matches found, 1 = no matches, -9 = intentionally capped.
        if proc.returncode not in (0, 1, -9) and not matches:
            return None

        return {
            "success": True,
            "matches": matches,
            "count": len(matches),
            "backend": "ripgrep",
            "truncated": len(matches) >= max_results,
        }

    async def _python_search(
        self, query: str, root: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            max_results = max(1, min(int(params.get("max_results", 100)), 1000))
            max_seconds = max(0.05, min(float(params.get("max_seconds", 10.0)), 60.0))
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "max_results and max_seconds must be numbers",
            }
        glob = params.get("glob") or "*"
        case_sensitive = bool(params.get("case_sensitive"))
        is_regex = bool(params.get("regex"))
        name_only = bool(params.get("name_only"))

        try:
            pattern = re.compile(
                query if is_regex else re.escape(query),
                0 if case_sensitive else re.IGNORECASE,
            )
        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}"}

        matches: List[Dict[str, Any]] = []
        root_path = Path(root)
        if not root_path.exists():
            return {
                "success": True,
                "matches": [],
                "count": 0,
                "backend": "python",
                "truncated": False,
            }

        scan_state = {"truncated": False, "reason": ""}

        def _scan() -> None:
            stats = WalkStats()

            def accept(relative: str, name: str, is_dir: bool) -> bool:
                return not is_dir and glob_match(name, glob)

            try:
                for item in walk_paths(
                    root_path,
                    recursive=True,
                    max_results=None,
                    max_seconds=max_seconds,
                    accept=accept,
                    stats=stats,
                ):
                    p = item.path
                    if _is_secret_file(p):
                        continue
                    if name_only:
                        if pattern.search(item.name):
                            matches.append({"path": str(p)})
                            if len(matches) >= max_results:
                                return
                        continue
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if pattern.search(line):
                                    matches.append(
                                        {
                                            "path": str(p),
                                            "line": i,
                                            "text": line.rstrip("\n"),
                                        }
                                    )
                                    if len(matches) >= max_results:
                                        return
                    except Exception:
                        continue
            finally:
                scan_state["truncated"] = stats.truncated
                scan_state["reason"] = stats.reason

        await asyncio.to_thread(_scan)
        truncated = len(matches) >= max_results or scan_state["truncated"]
        return {
            "success": True,
            "matches": matches,
            "count": len(matches),
            "backend": "python",
            "truncated": truncated,
            "truncated_reason": (
                "max_results" if len(matches) >= max_results
                else (scan_state["reason"] or None)
            ),
        }
