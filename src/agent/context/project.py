"""
Project Context - Codebase awareness.

Responsibilities:
- Detect project type (Python, Node, Rust, Go, etc.)
- Summarize directory structure
- Extract dependencies, scripts, entry points
- Provide a compact, LLM-friendly summary of the project
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# PROJECT DETECTION
# ======================================================================

PROJECT_MARKERS = {
    "python":     ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
    "node":       ["package.json", "yarn.lock", "pnpm-lock.yaml"],
    "rust":       ["Cargo.toml"],
    "go":         ["go.mod"],
    "java":       ["pom.xml", "build.gradle"],
    "ruby":       ["Gemfile"],
    "php":        ["composer.json"],
    "csharp":     ["*.csproj", "*.sln"],
    "elixir":     ["mix.exs"],
    "haskell":    ["*.cabal", "stack.yaml"],
}

LANG_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go",
    ".java": "java", ".kt": "kotlin",
    ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".fs": "fsharp",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".swift": "swift", ".m": "objc",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".lhs": "haskell",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".md": "markdown", ".rst": "restructuredtext",
    ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".sql": "sql",
}

IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "target", ".next", ".nuxt",
    ".idea", ".vscode", ".DS_Store",
    "coverage", ".coverage", "htmlcov",
}


# ======================================================================
# PROJECT CONTEXT
# ======================================================================

class ProjectContext:
    """
    Detects and summarizes the project structure.
    """

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self.root: Optional[Path] = self._resolve_root()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0  # seconds

    def _resolve_root(self) -> Optional[Path]:
        try:
            if self.workspace and hasattr(self.workspace, "project_dir"):
                return Path(self.workspace.project_dir).resolve()
        except Exception:
            pass
        return Path.cwd().resolve()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def refresh(self) -> Dict[str, Any]:
        """Force a re-scan"""
        self._cache = None
        return await self.get_summary(force=True)

    async def get_summary(self, force: bool = False) -> Dict[str, Any]:
        """Return a cached summary of the project"""
        now = time.time()
        if not force and self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        summary = await asyncio.to_thread(self._scan_sync)
        self._cache = summary
        self._cache_time = now
        return summary

    # ------------------------------------------------------------------
    # SCANNING
    # ------------------------------------------------------------------

    def _scan_sync(self) -> Dict[str, Any]:
        if not self.root or not self.root.exists():
            return {"root": None, "error": "project root not found"}

        lang_counts: Dict[str, int] = {}
        file_count = 0
        total_size = 0
        top_level_entries: List[str] = []

        try:
            top_level_entries = sorted([
                e.name for e in self.root.iterdir()
                if not e.name.startswith(".")
            ])[:30]
        except Exception:
            top_level_entries = []

        # Walk the tree (bounded)
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

            # Depth limit: stop at 5 levels deep
            rel = Path(dirpath).relative_to(self.root)
            if len(rel.parts) > 5:
                dirnames[:] = []
                continue

            for fn in filenames:
                file_count += 1
                ext = Path(fn).suffix.lower()
                lang = LANG_EXT.get(ext)
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

                try:
                    total_size += (Path(dirpath) / fn).stat().st_size
                except Exception:
                    pass

            if file_count > 5000:  # safety cap
                break

        # Detect project types via marker files
        detected_types: List[str] = []
        for lang, markers in PROJECT_MARKERS.items():
            for marker in markers:
                if "*" in marker:
                    if list(self.root.glob(marker)):
                        detected_types.append(lang)
                        break
                elif (self.root / marker).exists():
                    detected_types.append(lang)
                    break

        primary_lang = None
        if lang_counts:
            primary_lang = max(lang_counts.items(), key=lambda x: x[1])[0]

        # Metadata extraction
        metadata = self._extract_metadata(detected_types)

        return {
            "root": str(self.root),
            "name": self.root.name,
            "detected_types": detected_types,
            "primary_language": primary_lang,
            "language_counts": dict(sorted(
                lang_counts.items(), key=lambda x: -x[1]
            )[:10]),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "top_level_entries": top_level_entries,
            "metadata": metadata,
            "scanned_at": time.time(),
        }

    def _extract_metadata(self, detected_types: List[str]) -> Dict[str, Any]:
        """Extract dependency / script info from common project files"""
        meta: Dict[str, Any] = {}
        if not self.root:
            return meta

        # Python
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                project = data.get("project", {})
                meta["python"] = {
                    "name": project.get("name"),
                    "version": project.get("version"),
                    "dependencies": project.get("dependencies", [])[:20],
                    "python_requires": project.get("requires-python"),
                }
            except Exception:
                pass

        # Node
        pkg = self.root / "package.json"
        if pkg.exists():
            try:
                with open(pkg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta["node"] = {
                    "name": data.get("name"),
                    "version": data.get("version"),
                    "scripts": list(data.get("scripts", {}).keys()),
                    "dependencies": list(data.get("dependencies", {}).keys())[:20],
                    "dev_dependencies": list(data.get("devDependencies", {}).keys())[:20],
                }
            except Exception:
                pass

        # Rust
        cargo = self.root / "Cargo.toml"
        if cargo.exists():
            try:
                import tomllib
                with open(cargo, "rb") as f:
                    data = tomllib.load(f)
                pkg_data = data.get("package", {})
                meta["rust"] = {
                    "name": pkg_data.get("name"),
                    "version": pkg_data.get("version"),
                    "edition": pkg_data.get("edition"),
                }
            except Exception:
                pass

        # Go
        gomod = self.root / "go.mod"
        if gomod.exists():
            try:
                with open(gomod, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                meta["go"] = {"module": first_line.replace("module ", "")}
            except Exception:
                pass

        return meta

    # ------------------------------------------------------------------
    # LLM-FRIENDLY SUMMARY
    # ------------------------------------------------------------------

    async def as_prompt(self, max_chars: int = 1500) -> str:
        """Return a compact, LLM-friendly summary string"""
        summary = await self.get_summary()
        parts = []

        if summary.get("name"):
            parts.append(f"Project: {summary['name']}")
        if summary.get("primary_language"):
            parts.append(f"Primary language: {summary['primary_language']}")
        if summary.get("detected_types"):
            parts.append(f"Detected stack: {', '.join(summary['detected_types'])}")
        if summary.get("file_count"):
            parts.append(f"Files: {summary['file_count']}")

        langs = summary.get("language_counts", {})
        if langs:
            top = list(langs.items())[:5]
            parts.append("Languages: " + ", ".join(f"{k} ({v})" for k, v in top))

        meta = summary.get("metadata", {})
        if "node" in meta and meta["node"].get("scripts"):
            parts.append(f"npm scripts: {', '.join(meta['node']['scripts'][:8])}")
        if "python" in meta and meta["python"].get("dependencies"):
            deps = meta["python"]["dependencies"][:8]
            parts.append(f"Python deps: {', '.join(deps)}")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return self._cache or {}