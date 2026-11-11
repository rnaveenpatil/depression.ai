"""
Diagnostics Tool - Errors, lint, and compiler diagnostics.

Runs linters / type checkers / compilers and returns structured diagnostics.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)

# Flags that mutate files, redirect output, or execute helper programs.
_DANGEROUS_FLAGS = {
    "--fix", "--fix-only", "--unsafe-fixes", "--write", "--write-changes",
    "--output", "--output-file", "-o", "--out", "--out-file", "--json-output",
    "--cache-dir", "--cache-location", "--log-file", "--report",
}

_SAFE_BOOL_FLAGS = {
    "--no-cache", "--no-fix", "--preview", "--show-fixes", "--statistics",
    "--respect-gitignore", "--diff", "--no-color", "--color", "--quiet",
    "--verbose", "--strict", "--no-error-summary", "--pretty",
    "--show-source", "--show-column", "--show-error-codes",
    "--warn-unused-ignores", "--ignore-missing-imports",
    "--no-implicit-optional", "--show-traceback", "--fast", "--no-install",
}

_SAFE_VALUE_FLAGS = {
    "--select", "--ignore", "--extend-select", "--extend-ignore",
    "--line-length", "--target-version", "--exclude",
    "--per-file-ignores", "--output-format", "--max-line-length",
    "--extend-exclude", "--disable", "--enable", "--follow-imports",
    "--python-version", "--platform", "--project", "--max-warnings",
    "--ignore-patterns", "--ignore-path",
}


def _find_executable(name: str) -> Optional[str]:
    """Find an executable on PATH, then in the running interpreter's bin dir.

    The second check matters when the agent is launched via a venv console
    script by absolute path: ``venv/bin`` is not necessarily on PATH, so
    ``shutil.which("ruff")`` fails even though ``venv/bin/ruff`` exists.
    """
    found = shutil.which(name)
    if found:
        return found
    exe_name = f"{name}.exe" if os.name == "nt" else name
    candidate = Path(sys.executable).parent / exe_name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


class DiagnosticsTool(BaseTool):
    name = "diagnostics"
    description = "Run linters, type checkers, and compilers to surface errors and warnings."
    parameters = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["auto", "ruff", "flake8", "mypy", "pyright", "eslint", "tsc", "cargo", "go", "pylint"],
                "default": "auto",
            },
            "path": {"type": "string", "description": "File or directory to analyze"},
            "args": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "number", "default": 60},
        },
        "required": [],
    }
    timeout = 120.0

    read_only = True
    mutating = False
    category = "run"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()

    def _validate_path(self, raw: Any) -> str:
        value = str(raw or ".")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("invalid path")
        if value.startswith("-"):
            raise ValueError("path must not start with '-'")
        base = Path(self.cwd).resolve()
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        if resolved != base and not resolved.is_relative_to(base):
            raise ValueError("path must stay inside the workspace")
        return str(resolved)

    def _validate_extra_args(self, extra: Any) -> List[str]:
        if extra is None:
            return []
        if not isinstance(extra, list):
            raise ValueError("args must be an array of strings")
        out: List[str] = []
        i = 0
        while i < len(extra):
            arg = str(extra[i])
            if not arg or "\x00" in arg or "\n" in arg or "\r" in arg:
                raise ValueError("invalid diagnostics argument")
            flag = arg.split("=", 1)[0]
            if flag in _DANGEROUS_FLAGS or flag.startswith("--output"):
                raise ValueError(f"unsafe diagnostics option: {flag!r}")
            if arg.startswith("-"):
                if "=" in arg:
                    if flag not in _SAFE_VALUE_FLAGS:
                        raise ValueError(f"diagnostics option not allowed: {flag!r}")
                    out.append(arg)
                    i += 1
                    continue
                if arg in _SAFE_BOOL_FLAGS:
                    out.append(arg)
                    i += 1
                    continue
                if arg in _SAFE_VALUE_FLAGS:
                    if i + 1 >= len(extra):
                        raise ValueError(f"{arg} requires a value")
                    value = str(extra[i + 1])
                    if "\x00" in value or "\n" in value or "\r" in value:
                        raise ValueError("invalid diagnostics argument value")
                    out.extend([arg, value])
                    i += 2
                    continue
                raise ValueError(f"diagnostics option not allowed: {arg!r}")
            # Bare values are only valid immediately after a value flag.
            raise ValueError(f"unexpected diagnostics argument: {arg!r}")
        return out

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = params.get("tool", "auto")
        if tool == "auto":
            tool = self._detect_tool()

        try:
            path = self._validate_path(params.get("path"))
            extra = self._validate_extra_args(params.get("args"))
            timeout = max(1.0, min(float(params.get("timeout", 60)), 600.0))
        except (TypeError, ValueError) as e:
            return {"success": False, "error": str(e), "tool": tool}

        cmd = self._build_command(tool, path, extra)
        if not cmd:
            return {"success": False, "error": f"No diagnostics backend available for '{tool}'"}

        executable = _find_executable(cmd[0])
        if executable is None:
            return {
                "success": False,
                "error": f"Command not found: {cmd[0]}",
                "tool": tool,
                "recoverable": True,
            }
        cmd[0] = executable

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
            return {"success": False, "error": "diagnostics timed out"}
        except FileNotFoundError:
            return {"success": False, "error": f"Command not found: {cmd[0]}"}

        # Linters use non-zero exit codes for findings; the command itself
        # still succeeded. Reserve success=False for invocation failures.
        expected = {0, 1}
        if tool == "tsc":
            expected = {0, 1, 2}
        elif tool == "pylint":
            expected = set(range(0, 64))
        elif tool == "cargo":
            expected = {0, 101}
        return {
            "success": proc.returncode in expected,
            "tool": tool,
            "exit_code": proc.returncode,
            "has_findings": proc.returncode != 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _detect_tool(self) -> str:
        if (os.path.isdir(os.path.join(self.cwd, "src")) or
                any(f.endswith(".py") for f in os.listdir(self.cwd) if os.path.isfile(os.path.join(self.cwd, f)))):
            if _find_executable("ruff"):
                return "ruff"
            if _find_executable("flake8"):
                return "flake8"
            if _find_executable("mypy"):
                return "mypy"
        if os.path.exists(os.path.join(self.cwd, "tsconfig.json")):
            return "tsc"
        if os.path.exists(os.path.join(self.cwd, "package.json")):
            return "eslint"
        if os.path.exists(os.path.join(self.cwd, "Cargo.toml")):
            return "cargo"
        if os.path.exists(os.path.join(self.cwd, "go.mod")):
            return "go"
        return "ruff"

    def _build_command(self, tool: str, path: str, extra: List[str]) -> Optional[List[str]]:
        if tool == "ruff":
            return ["ruff", "check", "--output-format", "concise", *extra, "--", path]
        if tool == "flake8":
            return ["flake8", *extra, "--", path]
        if tool == "mypy":
            return ["mypy", *extra, "--", path]
        if tool == "pyright":
            return ["pyright", *extra, "--", path]
        if tool == "pylint":
            return ["pylint", *extra, "--", path]
        if tool == "eslint":
            return ["npx", "--no-install", "eslint", *extra, "--", path]
        if tool == "tsc":
            return ["npx", "--no-install", "tsc", "--noEmit", *extra]
        if tool == "cargo":
            return ["cargo", "check", *extra]
        if tool == "go":
            return ["go", "vet", *extra, "--", path]
        return None