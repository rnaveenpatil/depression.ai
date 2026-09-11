"""
Project Scanner - Builds an index of the codebase.

Responsibilities:
    - Walk the project tree (respecting ignore rules)
    - Classify files by language, size, and role
    - Detect entry points, tests, configs, docs
    - Produce a compact, LLM-friendly summary
    - Cache the index with mtime-based invalidation
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# LANGUAGE MAP
# ======================================================================

LANG_BY_EXT: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".fs": "fsharp",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".swift": "swift", ".m": "objc", ".mm": "objc",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell",
    ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".ps1": "powershell",
    ".md": "markdown", ".rst": "rst",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".xml": "xml", ".sql": "sql",
    ".tf": "terraform", ".tfvars": "terraform",
    ".proto": "protobuf",
    ".graphql": "graphql", ".gql": "graphql",
    ".vue": "vue", ".svelte": "svelte",
}

ENTRY_NAMES = {
    "main.py", "__main__.py", "app.py", "cli.py", "server.py",
    "index.js", "index.ts", "main.js", "main.ts", "app.js", "app.ts",
    "main.go", "main.rs", "main.c", "main.cpp",
    "manage.py", "wsgi.py", "asgi.py",
}

TEST_PATTERNS = [
    "test_*.py", "*_test.py", "tests/*", "test/*",
    "*.test.js", "*.test.ts", "*.spec.js", "*.spec.ts",
    "__tests__/*", "spec/*",
]

CONFIG_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "Pipfile", "poetry.lock", "package.json", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "tsconfig.json",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
    "pom.xml", "build.gradle", "Gemfile", "composer.json",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env", ".env.example", "Makefile", "justfile",
    "serverless.yml", "cdk.json", "samconfig.toml",
    "terraform.tfstate", "main.tf", "variables.tf",
    ".gitignore", ".dockerignore", "README.md",
}

DOC_NAMES = {
    "README.md", "README.rst", "README.txt",
    "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE",
    "docs", "doc", "wiki",
}

ENTRY_DIRS = {"src", "app", "lib", "source", "cmd", "bin", "internal"}


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class FileInfo:
    """Metadata for a single file"""
    path: str                # relative to project root
    abs_path: str
    name: str
    ext: str
    language: str = "unknown"
    size: int = 0
    mtime: float = 0.0
    lines: int = 0
    is_entry_point: bool = False
    is_test: bool = False
    is_config: bool = False
    is_doc: bool = False
    is_binary: bool = False


@dataclass
class ScanResult:
    """Full scan output"""
    root: str
    files: List[FileInfo] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    language_counts: Dict[str, int] = field(default_factory=dict)
    language_bytes: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_size: int = 0
    total_lines: int = 0
    primary_language: Optional[str] = None
    entry_points: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    doc_files: List[str] = field(default_factory=list)
    detected_stacks: List[str] = field(default_factory=list)
    scanned_at: float = field(default_factory=time.time)
    scan_duration: float = 0.0
    truncated: bool = False

    def top_languages(self, n: int = 5) -> List[Tuple[str, int]]:
        return sorted(self.language_counts.items(), key=lambda x: -x[1])[:n]

    def as_summary(self, max_chars: int = 1500) -> str:
        parts = []
        parts.append(f"Project: {Path(self.root).name}")
        if self.primary_language:
            parts.append(f"Primary language: {self.primary_language}")
        parts.append(f"Files: {self.total_files} ({self.total_size // 1024} KB)")

        if self.language_counts:
            top = self.top_languages(6)
            parts.append("Languages: " + ", ".join(f"{k} ({v})" for k, v in top))

        if self.detected_stacks:
            parts.append(f"Stack: {', '.join(self.detected_stacks)}")

        if self.entry_points:
            parts.append(f"Entry points: {', '.join(self.entry_points[:5])}")

        if self.config_files:
            parts.append(f"Config: {', '.join(self.config_files[:5])}")

        if self.test_files:
            parts.append(f"Tests: {len(self.test_files)} files")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text


# ======================================================================
# SCANNER
# ======================================================================

class ProjectScanner:
    """
    Walks the project tree and produces a ScanResult.
    """

    def __init__(
        self,
        project_dir: str | Path,
        ignore_patterns: Optional[List[str]] = None,
        max_depth: int = 5,
        max_files: int = 20_000,
        max_file_bytes: int = 5 * 1024 * 1024,
        respect_gitignore: bool = True,
    ):
        self.root = Path(project_dir).resolve()
        self.ignore_patterns: List[str] = list(ignore_patterns or [
            ".git", "node_modules", "__pycache__", "*.pyc",
            ".venv", "venv", "env", "dist", "build", ".next",
            ".DS_Store", "*.log", "*.tmp", ".cache",
            ".idea", ".vscode", "target", "coverage", ".pytest_cache",
            "*.min.js", "*.min.css", "*.map",
        ])
        self.max_depth = max_depth
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.respect_gitignore = respect_gitignore

        self._gitignore: Optional[List[str]] = None
        self._cache: Optional[ScanResult] = None
        self._cache_mtime: float = 0.0

    # ------------------------------------------------------------------

    async def scan(self, force: bool = False) -> ScanResult:
        """Scan the project (cached if unchanged)."""
        # Cache invalidation via root mtime
        try:
            root_mtime = self.root.stat().st_mtime
        except Exception:
            root_mtime = 0.0

        if not force and self._cache and root_mtime <= self._cache_mtime:
            return self._cache

        start = time.time()
        result = await asyncio.to_thread(self._scan_sync)
        result.scan_duration = time.time() - start

        self._cache = result
        self._cache_mtime = root_mtime
        return result

    def _scan_sync(self) -> ScanResult:
        result = ScanResult(root=str(self.root))

        if not self.root.exists():
            logger.warning(f"Project root does not exist: {self.root}")
            return result

        lang_counts: Dict[str, int] = {}
        lang_bytes: Dict[str, int] = {}
        total_size = 0
        total_lines = 0
        file_count = 0
        truncated = False

        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root)

            # Depth prune
            if len(rel_dir.parts) > self.max_depth:
                dirnames[:] = []
                continue

            # Ignore prune
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore(d, rel_dir / d)
            ]

            if rel_dir != Path("."):
                result.directories.append(str(rel_dir))

            for fn in filenames:
                if file_count >= self.max_files:
                    truncated = True
                    break

                abs_p = Path(dirpath) / fn
                rel_p = abs_p.relative_to(self.root)

                if self._should_ignore(fn, rel_p):
                    continue

                try:
                    stat = abs_p.stat()
                except Exception:
                    continue

                if stat.st_size > self.max_file_bytes:
                    continue

                ext = abs_p.suffix.lower()
                lang = LANG_BY_EXT.get(ext, "unknown")

                # Detect roles
                name_lower = fn.lower()
                is_entry = fn in ENTRY_NAMES or any(
                    str(rel_p).startswith(d + "/") and fn in ENTRY_NAMES
                    for d in ENTRY_DIRS
                )
                is_test = self._matches_any(str(rel_p), TEST_PATTERNS)
                is_config = fn in CONFIG_NAMES
                is_doc = fn in DOC_NAMES

                info = FileInfo(
                    path=str(rel_p),
                    abs_path=str(abs_p),
                    name=fn,
                    ext=ext,
                    language=lang,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    is_entry_point=is_entry,
                    is_test=is_test,
                    is_config=is_config,
                    is_doc=is_doc,
                )

                # Count lines for text files (best-effort, small files only)
                if lang not in ("unknown",) and stat.st_size < 512_000:
                    try:
                        with open(abs_p, "rb") as f:
                            chunk = f.read(64_000)
                            # Binary check
                            if b"\x00" not in chunk[:1024]:
                                info.lines = chunk.count(b"\n") + 1
                                total_lines += info.lines
                            else:
                                info.is_binary = True
                    except Exception:
                        pass

                result.files.append(info)
                file_count += 1

                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                lang_bytes[lang] = lang_bytes.get(lang, 0) + stat.st_size
                total_size += stat.st_size

                if is_entry:
                    result.entry_points.append(str(rel_p))
                if is_test:
                    result.test_files.append(str(rel_p))
                if is_config:
                    result.config_files.append(str(rel_p))
                if is_doc:
                    result.doc_files.append(str(rel_p))

            if truncated:
                break

        result.language_counts = lang_counts
        result.language_bytes = lang_bytes
        result.total_files = file_count
        result.total_size = total_size
        result.total_lines = total_lines
        result.truncated = truncated

        if lang_counts:
            result.primary_language = max(lang_counts.items(), key=lambda x: x[1])[0]

        result.detected_stacks = self._detect_stacks(result)

        logger.info(
            f"Scanned {file_count} files, {total_size // 1024} KB, "
            f"{total_lines} lines in {len(result.directories)} dirs"
        )
        return result

    # ------------------------------------------------------------------
    # IGNORE LOGIC
    # ------------------------------------------------------------------

    def _should_ignore(self, name: str, rel_path: Path) -> bool:
        for pat in self.ignore_patterns:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(str(rel_path), pat):
                return True
            if pat in str(rel_path):
                return True

        if self.respect_gitignore:
            if self._gitignore is None:
                self._load_gitignore()
            for pat in self._gitignore or []:
                if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(str(rel_path), pat):
                    return True
                if pat.endswith("/") and str(rel_path).startswith(pat.rstrip("/") + "/"):
                    return True

        return False

    def _load_gitignore(self) -> None:
        self._gitignore = []
        gi = self.root / ".gitignore"
        if not gi.exists():
            return
        try:
            for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self._gitignore.append(line)
        except Exception as e:
            logger.debug(f"Failed to load .gitignore: {e}")

    @staticmethod
    def _matches_any(path: str, patterns: Iterable[str]) -> bool:
        for pat in patterns:
            if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(Path(path).name, pat):
                return True
            if "/" in pat and path.startswith(pat.split("*")[0]):
                return True
        return False

    # ------------------------------------------------------------------
    # STACK DETECTION
    # ------------------------------------------------------------------

    def _detect_stacks(self, result: ScanResult) -> List[str]:
        stacks: List[str] = []
        names = {Path(f).name for f in result.config_files}

        if "pyproject.toml" in names or "setup.py" in names:
            stacks.append("python")
        if "package.json" in names:
            stacks.append("node")
        if "Cargo.toml" in names:
            stacks.append("rust")
        if "go.mod" in names:
            stacks.append("go")
        if "pom.xml" in names or "build.gradle" in names:
            stacks.append("java")
        if "Gemfile" in names:
            stacks.append("ruby")
        if "composer.json" in names:
            stacks.append("php")
        if "Dockerfile" in names or "docker-compose.yml" in names:
            stacks.append("docker")
        if any(n.endswith(".tf") for n in names) or "terraform" in result.language_counts:
            stacks.append("terraform")
        if "serverless.yml" in names:
            stacks.append("serverless")
        if "cdk.json" in names:
            stacks.append("aws-cdk")
        if "samconfig.toml" in names:
            stacks.append("aws-sam")

        return stacks

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    async def get_file(self, rel_path: str) -> Optional[FileInfo]:
        result = await self.scan()
        for f in result.files:
            if f.path == rel_path:
                return f
        return None

    async def find_files(
        self,
        pattern: str,
        limit: int = 50,
    ) -> List[FileInfo]:
        result = await self.scan()
        out = []
        for f in result.files:
            if fnmatch.fnmatch(f.path, pattern) or fnmatch.fnmatch(f.name, pattern):
                out.append(f)
                if len(out) >= limit:
                    break
        return out

    async def files_by_language(self, language: str) -> List[FileInfo]:
        result = await self.scan()
        return [f for f in result.files if f.language == language]