"""
Git Tool - Wrapper around git commands for the agent.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger
from agent.utils.shell import terminate_process

logger = get_logger(__name__)


# Mutating actions that must be rejected when read_only=True.
_WRITE_ACTIONS = {
    "add", "commit", "checkout", "pull", "push", "stash", "reset",
    "branch_delete", "tag_delete", "clean", "rebase", "merge", "cherry_pick",
}

# Git options that can execute helpers, redirect output, or escape the repo.
_UNSAFE_GIT_FLAGS = {
    "-c", "--config-env", "--exec-path", "--git-dir", "--work-tree",
    "--namespace", "--super-prefix", "--upload-pack", "--receive-pack",
    "--output", "--ext-diff", "--textconv", "--open-files-in-pager",
}

_SAFE_ACTION_ARGS = {
    "add": set(),
    "pull": {"--ff-only", "--rebase", "--no-rebase", "--autostash", "--prune", "--tags", "--no-tags"},
    "push": {"--force-with-lease", "--force", "--tags", "--all", "-u", "--set-upstream", "--dry-run"},
    "stash": {"--include-untracked", "--all", "--keep-index"},
    "reset": {"--soft", "--mixed", "--hard", "--keep", "--merge"},
}


class GitTool(BaseTool):
    name = "git"
    description = (
        "Inspect and manipulate git repositories "
        "(status, diff, log, add, commit, branch)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "diff", "log", "show", "branch",
                    "add", "commit", "checkout", "pull", "push",
                    "stash", "reset", "blame", "remote",
                ],
            },
            "path": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
            "ref": {"type": "string"},
            "staged": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["action"],
    }
    timeout = 60.0

    read_only = False
    mutating = True
    category = "vcs"

    def __init__(self, workspace: Any = None, read_only: bool = False):
        self.workspace = workspace
        self.read_only = bool(read_only)
        self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()

    @staticmethod
    def _check_path(value: Any) -> Optional[str]:
        if value is None:
            return None
        path = str(value)
        if not path or "\x00" in path or "\n" in path or "\r" in path:
            raise ValueError("invalid path")
        if path.startswith("-"):
            raise ValueError(f"path must not start with '-': {path!r}")
        try:
            from agent.permissions.manager import _path_is_secret
            candidates = [path]
            if ":" in path:
                candidates.append(path.split(":", 1)[1])
            if any(_path_is_secret(c) for c in candidates):
                raise ValueError("secret files are not accessible through git")
        except ValueError:
            raise
        except Exception:
            pass
        return path

    def _safe_extra_args(self, action: str, raw: Any) -> List[str]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError("args must be an array of strings")
        allowed = _SAFE_ACTION_ARGS.get(action, set())
        out: List[str] = []
        for item in raw:
            arg = str(item)
            if not arg or "\x00" in arg or "\n" in arg or "\r" in arg:
                raise ValueError("invalid git argument")
            if arg in _UNSAFE_GIT_FLAGS or arg.startswith(tuple(f + "=" for f in _UNSAFE_GIT_FLAGS)):
                raise ValueError(f"unsafe git option: {arg!r}")
            if arg.startswith("-") and arg not in allowed:
                raise ValueError(f"git option not allowed for {action}: {arg!r}")
            out.append(arg)
        return out

    async def _git(self, *args: str, timeout: float = 30.0) -> Dict[str, Any]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
        })
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                cwd=self.cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return {"success": False, "error": "git executable not found"}

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await terminate_process(proc, force=True)
            return {"success": False, "error": "git command timed out"}
        except asyncio.CancelledError:
            await terminate_process(proc, force=True)
            raise

        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = (params.get("action") or "").lower()

        if self.read_only and action in _WRITE_ACTIONS:
            return {
                "success": False,
                "error": f"Action '{action}' not allowed in read-only mode",
                "permission_denied": True,
                "tool": self.name,
            }

        args: List[str] = []

        try:
            path = self._check_path(params.get("path"))
            ref = self._check_path(params.get("ref"))
            extra = self._safe_extra_args(action, params.get("args"))
        except ValueError as e:
            return {"success": False, "error": str(e), "action": action}

        if action == "status":
            args = ["status", "--short", "--branch"]
        elif action == "diff":
            args = ["diff"]
            if params.get("staged"):
                args.append("--staged")
            if path:
                args.extend(["--", path])
        elif action == "log":
            try:
                limit = max(1, min(int(params.get("limit", 20)), 1000))
            except (TypeError, ValueError):
                return {"success": False, "error": "limit must be an integer"}
            args = ["log", f"--max-count={limit}", "--oneline"]
            if path:
                args.extend(["--", path])
        elif action == "show":
            args = ["show", "--no-ext-diff", "--no-textconv", ref or "HEAD"]
        elif action == "branch":
            args = ["branch", "-a"]
        elif action == "add":
            add_args = extra or ["."]
            try:
                for item in add_args:
                    self._check_path(item)
            except ValueError as e:
                return {"success": False, "error": str(e), "action": action}
            args = ["add", "--", *add_args]
        elif action == "commit":
            msg = params.get("message", "")
            if not msg:
                return {"success": False, "error": "commit requires 'message'"}
            args = ["commit", "-m", msg]
        elif action == "checkout":
            checkout_ref = ref or (extra[0] if extra else None)
            if not checkout_ref:
                return {"success": False, "error": "checkout requires 'ref'"}
            try:
                self._check_path(checkout_ref)
            except ValueError as e:
                return {"success": False, "error": str(e), "action": action}
            args = ["checkout", checkout_ref]
        elif action == "pull":
            args = ["pull", *extra]
        elif action == "push":
            args = ["push", *extra]
        elif action == "stash":
            args = ["stash", *extra]
        elif action == "reset":
            args = ["reset", *extra]
        elif action == "blame":
            if not path:
                return {"success": False, "error": "blame requires 'path'"}
            args = ["blame", "--", path]
        elif action == "remote":
            args = ["remote", "-v"]
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        result = await self._git(*args, timeout=self.timeout)
        result["action"] = action
        result["args"] = args
        result["read_only"] = self.read_only
        return result