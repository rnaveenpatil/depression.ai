"""
Patch Tool - Apply unified diffs and structured find-replace patches.

Captures the pre-image and post-image (capped at 50 KB per side) so the
TUI can render an inline diff alongside the result.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


DIFF_SIZE_CAP = 50 * 1024


class PatchTool(BaseTool):
    name = "patch"
    description = "Apply a unified diff or a structured find-replace patch."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["apply", "preview"],
            },
            "diff": {"type": "string"},
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                },
            },
            "include_diff": {
                "type": "boolean",
                "default": True,
                "description": "Capture before/after content (capped at 50 KB per side).",
            },
        },
        "required": ["action"],
    }
    timeout = 30.0

    read_only = False
    mutating = True
    category = "edit"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    def _resolve(self, path: str) -> Path:
        if not path or not str(path).strip():
            raise ValueError("patch path is required")
        if self.workspace:
            return self.workspace.assert_inside_workspace(str(path))
        return Path(os.path.expanduser(str(path))).resolve()

    @staticmethod
    def _read_text(target: Path) -> str:
        return target.read_bytes().decode("utf-8")

    @staticmethod
    def _write_atomic(target: Path, text: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = None
        try:
            mode = target.stat().st_mode & 0o7777
        except OSError:
            pass
        fd, tmp_name = tempfile.mkstemp(prefix=".patch-", dir=str(target.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(text.encode("utf-8"))
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
    def _is_secret(path: Path) -> bool:
        try:
            from agent.permissions.manager import _path_is_secret
            return _path_is_secret(str(path))
        except Exception:
            return False

    @classmethod
    def _capture(cls, entry: Dict[str, Any], target: Path, before: str, after: str) -> None:
        if cls._is_secret(target):
            entry["redacted"] = True
            entry["before"] = "[redacted secret file]"
            entry["after"] = "[redacted secret file]"
            return
        if len(before) <= DIFF_SIZE_CAP and len(after) <= DIFF_SIZE_CAP:
            entry["before"] = before
            entry["after"] = after
        else:
            entry["diff_omitted"] = True

    def _commit(
        self,
        pending: List[Dict[str, Any]],
        dry_run: bool,
        include_diff: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        written: List[Dict[str, Any]] = []
        try:
            if not dry_run:
                for item in pending:
                    self._write_atomic(item["target"], item["after"])
                    written.append(item)
            for item in pending:
                entry = dict(item["entry"])
                entry["success"] = True
                if include_diff:
                    self._capture(entry, item["target"], item["before"], item["after"])
                results.append(entry)
            return results
        except Exception as exc:
            for item in reversed(written):
                try:
                    self._write_atomic(item["target"], item["before"])
                except Exception:
                    pass
            return [
                {
                    "success": False,
                    "path": str(item["target"]),
                    "error": f"write failed: {exc}; changes rolled back",
                }
                for item in pending
            ]

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "apply")
        if action not in ("apply", "preview"):
            return {"success": False, "error": f"Unknown action: {action}"}

        dry_run = action == "preview"
        include_diff = bool(params.get("include_diff", True))

        if params.get("patches"):
            results = await self._apply_structured(
                params["patches"], dry_run=dry_run, include_diff=include_diff
            )
        elif params.get("diff"):
            results = await self._apply_unified(
                params["diff"], dry_run=dry_run, include_diff=include_diff
            )
        else:
            return {"success": False, "error": "Provide either 'patches' or 'diff'"}

        ok = bool(results) and all(r.get("success") for r in results)
        return {"success": ok, "results": results, "applied": not dry_run}

    # ------------------------------------------------------------------

    async def _apply_structured(
        self,
        patches: List[Dict[str, Any]],
        dry_run: bool,
        include_diff: bool = True,
    ) -> List[Dict[str, Any]]:
        if not isinstance(patches, list) or not patches:
            return [{"success": False, "error": "patches must be a non-empty array"}]

        current: Dict[Path, str] = {}
        before: Dict[Path, str] = {}
        pending: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for p in patches:
            if not isinstance(p, dict):
                errors.append({"success": False, "error": "patch entry must be an object"})
                continue
            try:
                target = self._resolve(str(p.get("path") or ""))
                old = p.get("old", "")
                new = p.get("new", "")
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError("old and new must be strings")
                if not old:
                    raise ValueError("old must not be empty")
                if not target.exists() or not target.is_file():
                    raise ValueError("not found")
                if target not in current:
                    current[target] = self._read_text(target)
                    before[target] = current[target]
                text = current[target]
                if old not in text:
                    raise ValueError("old text not found")
                updated = text.replace(old, new, 1)
                current[target] = updated
                pending.append(
                    {
                        "target": target,
                        "before": before[target],
                        "after": updated,
                        "entry": {
                            "path": str(target),
                            "bytes_before": len(before[target].encode("utf-8")),
                            "bytes_after": len(updated.encode("utf-8")),
                        },
                    }
                )
            except Exception as e:
                errors.append(
                    {"success": False, "path": str(p.get("path") or ""), "error": str(e)}
                )

        if errors:
            return errors + [
                {
                    "success": False,
                    "path": str(item["target"]),
                    "error": "not applied because another patch in this batch failed",
                }
                for item in pending
            ]
        return self._commit(pending, dry_run=dry_run, include_diff=include_diff)

    async def _apply_unified(
        self,
        diff_text: str,
        dry_run: bool,
        include_diff: bool = True,
    ) -> List[Dict[str, Any]]:
        files = self._parse_unified_diff(diff_text)
        if not files:
            return [{"success": False, "error": "no file hunks found in diff"}]

        pending: List[Dict[str, Any]] = []
        try:
            for file_hunks in files:
                path = file_hunks["path"]
                target = self._resolve(path)
                if not target.exists() or not target.is_file():
                    raise ValueError(f"{path}: not found")
                text = self._read_text(target)
                lines = text.splitlines(keepends=True)
                newline = "\r\n" if "\r\n" in text else "\n"

                for hunk in file_hunks["hunks"]:
                    old_lines = hunk["old_lines"]
                    new_lines = hunk["new_lines"]
                    start = max(0, hunk["old_start"] - 1)
                    if hunk["old_count"] == 0:
                        start = max(0, hunk["old_start"])

                    actual = [self._line_body(l) for l in lines[start:start + len(old_lines)]]
                    expected = [self._line_body(l) for l in old_lines]
                    if actual != expected:
                        found = self._find_hunk(lines, old_lines, start)
                        if found is None:
                            raise ValueError(
                                f"hunk at line {hunk['old_start']} doesn't match"
                            )
                        start = found

                    replacement = [self._line_body(l) + newline for l in new_lines]
                    lines[start:start + len(old_lines)] = replacement

                updated = "".join(lines)
                pending.append(
                    {
                        "target": target,
                        "before": text,
                        "after": updated,
                        "entry": {
                            "path": str(target),
                            "hunks_applied": len(file_hunks["hunks"]),
                        },
                    }
                )
        except Exception as e:
            return [{"success": False, "error": str(e)}]

        return self._commit(pending, dry_run=dry_run, include_diff=include_diff)

    def _parse_unified_diff(self, text: str) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        current_hunk: Optional[Dict[str, Any]] = None

        for line in text.splitlines():
            if line.startswith("--- "):
                continue
            if line.startswith("+++ "):
                raw_path = line[4:].strip()
                if "\t" in raw_path:
                    raw_path = raw_path.split("\t", 1)[0]
                if raw_path.startswith(("a/", "b/")):
                    raw_path = raw_path[2:]
                current = {"path": raw_path, "hunks": []}
                files.append(current)
                current_hunk = None
                continue
            if line.startswith("@@ ") and current is not None:
                m = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line
                )
                if not m:
                    continue
                current_hunk = {
                    "old_start": int(m.group(1)),
                    "old_count": int(m.group(2) or 1),
                    "new_start": int(m.group(3)),
                    "new_count": int(m.group(4) or 1),
                    "old_lines": [],
                    "new_lines": [],
                }
                current["hunks"].append(current_hunk)
                continue
            if current_hunk is None:
                continue
            if line.startswith("\\"):
                continue
            if line.startswith(" "):
                current_hunk["old_lines"].append(line[1:])
                current_hunk["new_lines"].append(line[1:])
            elif line.startswith("-"):
                current_hunk["old_lines"].append(line[1:])
            elif line.startswith("+"):
                current_hunk["new_lines"].append(line[1:])
        return files

    @staticmethod
    def _line_body(line: str) -> str:
        return line.rstrip("\r\n")

    def _find_hunk(
        self, lines: List[str], old: List[str], start: int
    ) -> Optional[int]:
        window = 20
        expected = [l.rstrip("\n") for l in old]
        for offset in range(1, window):
            for delta in (start + offset, start - offset):
                if 0 <= delta <= len(lines) - len(old):
                    candidate = [
                        l.rstrip("\n") for l in lines[delta:delta + len(old)]
                    ]
                    if candidate == expected:
                        return delta
        return None