"""
Patch Tool - Apply unified diffs and structured patches to files.

Supports:
    - Unified diff format (--- / +++ / @@ ... @@)
    - Structured find-replace patches
    - Dry-run mode to preview changes
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class PatchTool(BaseTool):
    name = "patch"
    description = "Apply a unified diff or a structured find-replace patch to one or more files."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["apply", "preview"],
                "description": "apply applies the patch; preview shows the diff only",
            },
            "diff": {"type": "string", "description": "Unified diff text"},
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
        },
        "required": ["action"],
    }
    timeout = 30.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace

    def _resolve(self, path: str) -> Path:
        if self.workspace:
            return self.workspace.assert_inside_workspace(path)
        return Path(os.path.expanduser(path)).resolve()

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "apply")
        if action not in ("apply", "preview"):
            return {"success": False, "error": f"Unknown action: {action}"}

        results = []
        if params.get("patches"):
            results = await self._apply_structured(params["patches"], dry_run=(action == "preview"))
        elif params.get("diff"):
            results = await self._apply_unified(params["diff"], dry_run=(action == "preview"))
        else:
            return {"success": False, "error": "Provide either 'patches' or 'diff'"}

        ok = all(r.get("success") for r in results)
        return {"success": ok, "results": results, "applied": action == "apply"}

    # ------------------------------------------------------------------

    async def _apply_structured(
        self, patches: List[Dict[str, Any]], dry_run: bool
    ) -> List[Dict[str, Any]]:
        results = []
        for p in patches:
            path = p.get("path")
            old = p.get("old", "")
            new = p.get("new", "")
            try:
                target = self._resolve(path)
                if not target.exists():
                    results.append({"success": False, "path": path, "error": "not found"})
                    continue
                text = target.read_text(encoding="utf-8", errors="replace")
                if old not in text:
                    results.append({"success": False, "path": path, "error": "old text not found"})
                    continue
                updated = text.replace(old, new, 1)
                if not dry_run:
                    target.write_text(updated, encoding="utf-8")
                results.append({
                    "success": True,
                    "path": str(target),
                    "bytes_before": len(text),
                    "bytes_after": len(updated),
                })
            except Exception as e:
                results.append({"success": False, "path": path, "error": str(e)})
        return results

    async def _apply_unified(
        self, diff_text: str, dry_run: bool
    ) -> List[Dict[str, Any]]:
        hunks = self._parse_unified_diff(diff_text)
        results = []
        for file_hunks in hunks:
            path = file_hunks["path"]
            try:
                target = self._resolve(path)
                if not target.exists():
                    results.append({"success": False, "path": path, "error": "not found"})
                    continue
                text = target.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines(keepends=True)

                for hunk in file_hunks["hunks"]:
                    old_lines = hunk["old_lines"]
                    new_lines = hunk["new_lines"]
                    start = hunk["old_start"] - 1
                    # Verify content matches
                    actual = lines[start:start + len(old_lines)]
                    if [l.rstrip("\n") for l in actual] != [l.rstrip("\n") for l in old_lines]:
                        # Try fuzzy search nearby
                        found = self._find_hunk(lines, old_lines, start)
                        if found is None:
                            raise ValueError(f"hunk at line {hunk['old_start']} doesn't match")
                        start = found

                    replacement = [l + "\n" if not l.endswith("\n") else l for l in new_lines]
                    lines[start:start + len(old_lines)] = replacement

                updated = "".join(lines)
                if not dry_run:
                    target.write_text(updated, encoding="utf-8")
                results.append({
                    "success": True,
                    "path": str(target),
                    "hunks_applied": len(file_hunks["hunks"]),
                })
            except Exception as e:
                results.append({"success": False, "path": path, "error": str(e)})
        return results

    def _parse_unified_diff(self, text: str) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        current_hunk: Optional[Dict[str, Any]] = None

        for line in text.splitlines():
            if line.startswith("--- "):
                continue
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                current = {"path": path, "hunks": []}
                files.append(current)
                current_hunk = None
                continue
            if line.startswith("@@ ") and current is not None:
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
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
            if line.startswith(" "):
                current_hunk["old_lines"].append(line[1:])
                current_hunk["new_lines"].append(line[1:])
            elif line.startswith("-"):
                current_hunk["old_lines"].append(line[1:])
            elif line.startswith("+"):
                current_hunk["new_lines"].append(line[1:])
        return files

    def _find_hunk(self, lines: List[str], old: List[str], start: int) -> Optional[int]:
        window = 20
        for offset in range(1, window):
            for delta in (start + offset, start - offset):
                if 0 <= delta <= len(lines) - len(old):
                    candidate = [l.rstrip("\n") for l in lines[delta:delta + len(old)]]
                    if candidate == [l.rstrip("\n") for l in old]:
                        return delta
        return None