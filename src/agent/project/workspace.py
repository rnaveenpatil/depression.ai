"""
Workspace Manager - Owns the working directory and project root.

Responsibilities:
    - Resolve and track the current project directory
    - Enforce path boundaries (never read/write outside the workspace)
    - Provide helpers for path normalization and safety checks
    - Track workspace metadata (project name, git presence, AWS presence)
    - Support switching projects at runtime (/cd, /project cd)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import WorkspaceError

logger = get_logger(__name__)


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class WorkspaceInfo:
    """Snapshot of the workspace state"""
    project_dir: str
    workspace_dir: str
    project_name: str
    is_git_repo: bool = False
    is_aws_project: bool = False
    has_pyproject: bool = False
    has_package_json: bool = False
    has_dockerfile: bool = False
    has_terraform: bool = False
    has_serverless: bool = False
    total_files: int = 0
    total_size: int = 0
    resolved_at: float = field(default_factory=time.time)


# ======================================================================
# WORKSPACE MANAGER
# ======================================================================

class WorkspaceManager:
    """
    Owns the working directory. Everything the agent reads/writes flows
    through this class so we can enforce boundaries and track state.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.config = cfg

        # Workspace dir (where the agent stores its state: .agent/)
        self.workspace_dir = Path(
            os.path.expanduser(workspace_dir or cfg.get("workspace_dir", "."))
        ).resolve()

        # Project dir (the actual codebase)
        self.project_dir = Path(
            os.path.expanduser(project_dir or cfg.get("path", os.getcwd()))
        ).resolve()

        # Boundary control
        self.enforce_boundaries: bool = cfg.get("enforce_boundaries", True)
        self.allowed_extra_dirs: List[Path] = [
            Path(os.path.expanduser(p)).resolve()
            for p in cfg.get("allowed_extra_dirs", [])
        ]

        # Auto-scan behavior
        self.auto_scan: bool = cfg.get("auto_scan", True)
        self.scan_depth: int = cfg.get("scan_depth", 5)
        self.max_file_size: int = cfg.get("max_file_size", 1024 * 1024)

        # Ignore patterns
        self.ignore_patterns: List[str] = list(cfg.get("ignore_patterns", [
            ".git", "node_modules", "__pycache__", "*.pyc",
            ".venv", "venv", "env", "dist", "build", ".next",
            ".DS_Store", "*.log", "*.tmp", ".cache", ".idea",
            ".vscode", "target", "coverage", ".pytest_cache",
        ]))
        self.respect_gitignore: bool = cfg.get("respect_gitignore", True)

        # State
        self._info: Optional[WorkspaceInfo] = None
        self._gitignore_cache: Optional[List[str]] = None

        logger.info(
            f"WorkspaceManager initialized "
            f"(project={self.project_dir}, workspace={self.workspace_dir})"
        )

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Validate the project dir and (optionally) scan it."""
        if not self.project_dir.exists():
            logger.warning(
                f"Project directory does not exist: {self.project_dir} — "
                f"falling back to cwd"
            )
            self.project_dir = Path.cwd().resolve()

        if not self.project_dir.is_dir():
            raise WorkspaceError(
                f"Project path is not a directory: {self.project_dir}"
            )

        # Create workspace dir if needed
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create workspace dir: {e}")

        # Build info snapshot
        self._info = await self._build_info()

        if self.auto_scan:
            await self.scan_metadata()

        logger.info(
            f"Workspace ready: {self._info.project_name} "
            f"(files={self._info.total_files}, git={self._info.is_git_repo}, "
            f"aws={self._info.is_aws_project})"
        )

    async def cleanup(self) -> None:
        """Called during agent shutdown."""
        logger.debug("Workspace cleanup complete")

    # ------------------------------------------------------------------
    # PROJECT DIR MANAGEMENT
    # ------------------------------------------------------------------

    async def set_project_dir(self, path: str) -> WorkspaceInfo:
        """Change the active project directory."""
        new_dir = Path(os.path.expanduser(path)).resolve()
        if not new_dir.exists() or not new_dir.is_dir():
            raise WorkspaceError(f"Invalid project directory: {new_dir}")

        self.project_dir = new_dir
        self._gitignore_cache = None
        self._info = await self._build_info()

        logger.info(f"Project directory changed to: {new_dir}")
        return self._info

    def get_project_dir(self) -> Path:
        return self.project_dir

    def get_workspace_dir(self) -> Path:
        return self.workspace_dir

    async def get_info(self) -> Dict[str, Any]:
        if self._info is None:
            self._info = await self._build_info()
        return self._info.__dict__

    # ------------------------------------------------------------------
    # PATH SAFETY
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        """Resolve a path relative to the project dir."""
        p = Path(os.path.expanduser(path))
        if not p.is_absolute():
            p = self.project_dir / p
        return p.resolve()

    def is_inside_workspace(self, path: str | Path) -> bool:
        """Check whether a path is inside the allowed workspace."""
        try:
            p = self.resolve(str(path)) if isinstance(path, str) else path.resolve()
        except Exception:
            return False

        if self._is_under(p, self.project_dir):
            return True
        if self._is_under(p, self.workspace_dir):
            return True
        return any(self._is_under(p, d) for d in self.allowed_extra_dirs)

    def assert_inside_workspace(self, path: str | Path) -> Path:
        """Raise if the path is outside the workspace."""
        p = self.resolve(str(path)) if isinstance(path, str) else path.resolve()
        if self.enforce_boundaries and not self.is_inside_workspace(p):
            raise WorkspaceError(
                f"Path outside workspace: {p} (workspace={self.project_dir})"
            )
        return p

    @staticmethod
    def _is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def relative(self, path: str | Path) -> str:
        """Get a path relative to the project dir."""
        try:
            p = Path(path).resolve() if isinstance(path, Path) else self.resolve(str(path))
            return str(p.relative_to(self.project_dir))
        except Exception:
            return str(path)

    # ------------------------------------------------------------------
    # IGNORE HANDLING
    # ------------------------------------------------------------------

    def should_ignore(self, path: Path) -> bool:
        """Check if a file/dir should be ignored during scanning."""
        name = path.name
        rel = self.relative(path)

        # Hard-coded ignore patterns
        for pattern in self.ignore_patterns:
            if self._fnmatch(name, pattern) or self._fnmatch(rel, pattern):
                return True

        # .gitignore patterns
        if self.respect_gitignore:
            for pattern in self._get_gitignore_patterns():
                if self._fnmatch(name, pattern) or self._fnmatch(rel, pattern):
                    return True

        return False

    def _get_gitignore_patterns(self) -> List[str]:
        if self._gitignore_cache is not None:
            return self._gitignore_cache

        patterns: List[str] = []
        gi = self.project_dir / ".gitignore"
        if gi.exists():
            try:
                for line in gi.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line.rstrip("/"))
            except Exception as e:
                logger.debug(f"Failed to read .gitignore: {e}")

        self._gitignore_cache = patterns
        return patterns

    @staticmethod
    def _fnmatch(name: str, pattern: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, f"*{pattern}*")

    # ------------------------------------------------------------------
    # METADATA SCAN
    # ------------------------------------------------------------------

    async def _build_info(self) -> WorkspaceInfo:
        """Build a WorkspaceInfo snapshot."""
        info = WorkspaceInfo(
            project_dir=str(self.project_dir),
            workspace_dir=str(self.workspace_dir),
            project_name=self.project_dir.name,
        )

        # Detect markers
        info.is_git_repo = (self.project_dir / ".git").exists()
        info.has_pyproject = (self.project_dir / "pyproject.toml").exists()
        info.has_package_json = (self.project_dir / "package.json").exists()
        info.has_dockerfile = (
            (self.project_dir / "Dockerfile").exists()
            or (self.project_dir / "docker-compose.yml").exists()
        )
        info.has_terraform = any(
            p.suffix in (".tf", ".tfvars")
            for p in self.project_dir.glob("*")
            if p.is_file()
        )
        info.has_serverless = (self.project_dir / "serverless.yml").exists()

        # AWS detection
        info.is_aws_project = (
            info.has_terraform
            or info.has_serverless
            or (self.project_dir / ".aws").exists()
            or (self.project_dir / "samconfig.toml").exists()
            or (self.project_dir / "cdk.json").exists()
            or (self.project_dir / "amplify.yml").exists()
        )

        return info

    async def scan_metadata(self) -> WorkspaceInfo:
        """Lightweight scan: count files and total size."""
        if self._info is None:
            self._info = await self._build_info()

        count = 0
        total_size = 0

        try:
            for dirpath, dirnames, filenames in os.walk(self.project_dir):
                # Prune ignored dirs in-place
                dirnames[:] = [
                    d for d in dirnames
                    if not self.should_ignore(Path(dirpath) / d)
                ]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if self.should_ignore(p):
                        continue
                    count += 1
                    try:
                        total_size += p.stat().st_size
                    except Exception:
                        pass
                if count > 50_000:  # safety cap
                    break
        except Exception as e:
            logger.debug(f"Scan failed: {e}")

        self._info.total_files = count
        self._info.total_size = total_size
        self._info.resolved_at = time.time()
        return self._info

    # ------------------------------------------------------------------
    # FILE OPERATIONS (safe, boundary-checked)
    # ------------------------------------------------------------------

    async def list_files(
        self,
        ext: Optional[str] = None,
        tree: bool = False,
        max_files: int = 200,
        include_dirs: bool = False,
    ) -> List[str]:
        """List files inside the workspace."""
        results: List[str] = []

        try:
            for dirpath, dirnames, filenames in os.walk(self.project_dir):
                dirnames[:] = [
                    d for d in dirnames
                    if not self.should_ignore(Path(dirpath) / d)
                ]

                # Depth limit
                rel_dir = Path(dirpath).relative_to(self.project_dir)
                if len(rel_dir.parts) > self.scan_depth:
                    dirnames[:] = []
                    continue

                if include_dirs and not tree:
                    for d in dirnames:
                        results.append(str(rel_dir / d) + "/")
                if tree:
                    prefix = "  " * len(rel_dir.parts)
                    if rel_dir != Path("."):
                        results.append(f"{prefix}{rel_dir.name}/")

                for fn in sorted(filenames):
                    p = Path(dirpath) / fn
                    if self.should_ignore(p):
                        continue
                    if ext and not fn.endswith(ext):
                        continue
                    rel = p.relative_to(self.project_dir)
                    if tree:
                        results.append(f"  " * (len(rel.parts) - 1) + fn)
                    else:
                        results.append(str(rel))

                if len(results) >= max_files:
                    break
        except Exception as e:
            logger.debug(f"list_files failed: {e}")

        return results[:max_files]

    async def read_file(self, path: str, max_bytes: Optional[int] = None) -> str:
        """Read a file inside the workspace."""
        p = self.assert_inside_workspace(path)
        if not p.exists():
            raise WorkspaceError(f"File not found: {p}")
        if not p.is_file():
            raise WorkspaceError(f"Not a file: {p}")

        limit = max_bytes or self.max_file_size
        try:
            with open(p, "rb") as f:
                raw = f.read(limit + 1)
            truncated = len(raw) > limit
            if truncated:
                raw = raw[:limit]
            if b"\x00" in raw[:1024]:
                raise WorkspaceError(f"Binary file not supported: {p}")
            return raw.decode("utf-8", errors="replace")
        except WorkspaceError:
            raise
        except Exception as e:
            raise WorkspaceError(f"Read failed: {e}")

    # ------------------------------------------------------------------
    # GIT HELPERS
    # ------------------------------------------------------------------

    async def get_git_info(self) -> Dict[str, Any]:
        """Return basic git info about the workspace."""
        if not (self.project_dir / ".git").exists():
            return {"is_git_repo": False}

        try:
            from agent.project.git import GitProject
            gp = GitProject(self.project_dir)
            status = await gp.get_status()
            return {
                "is_git_repo": True,
                "branch": status.branch,
                "is_dirty": status.is_dirty,
                "ahead": status.ahead,
                "behind": status.behind,
                "staged": len(status.staged),
                "modified": len(status.modified),
                "untracked": len(status.untracked),
                "last_commit": (
                    status.last_commit.__dict__
                    if status.last_commit else None
                ),
            }
        except Exception as e:
            logger.debug(f"Git info failed: {e}")
            return {"is_git_repo": True, "error": str(e)}

    async def scan(self) -> Dict[str, Any]:
        """Full scan: refresh metadata and return info dict."""
        await self.scan_metadata()
        return await self.get_info()