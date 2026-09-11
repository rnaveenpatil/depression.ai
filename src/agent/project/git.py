"""
Git Project - Wrapper around git operations and analysis.

Responsibilities:
    - Status: branch, staged/modified/untracked files
    - Diff: working tree, staged, commit-to-commit
    - Log: recent commits, authors, dates
    - Branches: local, remote, current
    - Blame and file history
    - Safe defaults (read-only unless explicitly enabled)
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import GitError

logger = get_logger(__name__)


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class GitCommit:
    sha: str
    short_sha: str
    author: str
    email: str
    date: str
    subject: str
    body: str = ""
    parents: List[str] = field(default_factory=list)
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


@dataclass
class GitBranch:
    name: str
    sha: str
    is_current: bool = False
    is_remote: bool = False
    upstream: Optional[str] = None
    ahead: int = 0
    behind: int = 0


@dataclass
class GitStatus:
    branch: str
    is_dirty: bool = False
    is_detached: bool = False
    ahead: int = 0
    behind: int = 0
    staged: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    conflicted: List[str] = field(default_factory=list)
    last_commit: Optional[GitCommit] = None


# ======================================================================
# GIT PROJECT
# ======================================================================

class GitProject:
    """
    Git-aware view of a project directory.

    All operations shell out to `git` via asyncio subprocess.
    """

    def __init__(
        self,
        project_dir: str | Path,
        git_binary: str = "git",
        timeout: float = 30.0,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.git_binary = git_binary
        self.timeout = timeout

    # ------------------------------------------------------------------
    # LOW-LEVEL EXEC
    # ------------------------------------------------------------------

    async def _run(
        self,
        args: List[str],
        check: bool = True,
        timeout: Optional[float] = None,
    ) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        cmd = [self.git_binary, *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout or self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise GitError(f"git {args[0]} timed out")

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")

            if check and proc.returncode != 0:
                raise GitError(f"git {args[0]} failed: {err.strip() or out.strip()}")

            return proc.returncode, out, err

        except FileNotFoundError:
            raise GitError(f"git binary not found: {self.git_binary}")

    # ------------------------------------------------------------------
    # REPO DETECTION
    # ------------------------------------------------------------------

    def is_git_repo(self) -> bool:
        return (self.project_dir / ".git").exists()

    async def is_valid_repo(self) -> bool:
        try:
            rc, _, _ = await self._run(["rev-parse", "--git-dir"], check=False)
            return rc == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    async def get_status(self) -> GitStatus:
        """Parse `git status --porcelain=v2 --branch`"""
        rc, out, err = await self._run(
            ["status", "--porcelain=v2", "--branch", "-z"],
            check=False,
        )
        if rc != 0:
            raise GitError(f"git status failed: {err.strip()}")

        status = GitStatus(branch="HEAD")
        entries = out.split("\x00")
        for line in entries:
            if not line:
                continue

            if line.startswith("# branch.head "):
                status.branch = line[len("# branch.head "):].strip()
            elif line.startswith("# branch.upstream "):
                pass
            elif line.startswith("# branch.ab "):
                m = re.match(r"# branch\.ab \+(\d+) -(\d+)", line)
                if m:
                    status.ahead = int(m.group(1))
                    status.behind = int(m.group(2))
            elif line.startswith("1 ") or line.startswith("2 "):
                parts = line.split(" ")
                xy = parts[1] if len(parts) > 1 else ".."
                path = parts[-1]
                x, y = xy[0], xy[1]

                if x == "R":
                    status.renamed.append((path, path))
                elif x in "MAD":
                    if x == "M": status.staged.append(path)
                    elif x == "A": status.staged.append(path)
                    elif x == "D": status.deleted.append(path)

                if y == "M":
                    status.modified.append(path)
                elif y == "D":
                    status.deleted.append(path)

            elif line.startswith("? "):
                status.untracked.append(line[2:])
            elif line.startswith("u "):
                parts = line.split(" ")
                if len(parts) > 1:
                    status.conflicted.append(parts[-1])

        status.is_dirty = bool(
            status.staged or status.modified or status.untracked
            or status.deleted or status.renamed or status.conflicted
        )

        # Last commit
        try:
            status.last_commit = await self.get_last_commit()
        except Exception:
            pass

        return status

    # ------------------------------------------------------------------
    # DIFF
    # ------------------------------------------------------------------

    async def diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        ref: Optional[str] = None,
        context: int = 3,
        stat_only: bool = False,
    ) -> str:
        args = ["diff", f"-U{context}"]
        if staged:
            args.append("--staged")
        if stat_only:
            args.append("--stat")
        if ref:
            args.append(ref)
        if path:
            args.extend(["--", path])

        _, out, _ = await self._run(args, check=False)
        return out

    async def diff_stat(
        self, ref: Optional[str] = None, staged: bool = False
    ) -> Dict[str, Any]:
        args = ["diff", "--numstat"]
        if staged:
            args.append("--staged")
        if ref:
            args.append(ref)

        _, out, _ = await self._run(args, check=False)
        files = []
        total_ins = 0
        total_del = 0
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                ins, dele, fname = parts
                try:
                    i = int(ins); d = int(dele)
                except ValueError:
                    i = d = 0
                files.append({"file": fname, "insertions": i, "deletions": d})
                total_ins += i
                total_del += d

        return {
            "files": files,
            "total_files": len(files),
            "insertions": total_ins,
            "deletions": total_del,
        }

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------

    async def log(
        self,
        limit: int = 20,
        oneline: bool = True,
        author: Optional[str] = None,
        path: Optional[str] = None,
    ) -> List[GitCommit]:
        fmt = "%H%x00%h%x00%an%x00%ae%x00%aI%x00%s%x00%b%x00%P"
        args = ["log", f"--max-count={limit}", f"--pretty=format:{fmt}"]
        if oneline:
            args.append("--no-merges")
        if author:
            args.append(f"--author={author}")
        if path:
            args.extend(["--", path])

        _, out, _ = await self._run(args, check=False)

        commits: List[GitCommit] = []
        for record in out.split("\x00\x00"):
            if not record.strip():
                continue
            parts = record.split("\x00")
            if len(parts) < 8:
                continue
            sha, short, author, email, date, subject, body, parents = parts[:8]
            commits.append(GitCommit(
                sha=sha,
                short_sha=short,
                author=author,
                email=email,
                date=date,
                subject=subject,
                body=body.strip(),
                parents=parents.split() if parents else [],
            ))
        return commits

    async def get_last_commit(self) -> Optional[GitCommit]:
        commits = await self.log(limit=1)
        return commits[0] if commits else None

    async def show_commit(
        self, ref: str, stat: bool = True, patch: bool = False
    ) -> str:
        args = ["show", ref]
        if stat:
            args.append("--stat")
        if not patch:
            args.append("--no-patch")
        _, out, _ = await self._run(args, check=False)
        return out

    # ------------------------------------------------------------------
    # BRANCHES
    # ------------------------------------------------------------------

    async def branches(self, include_remote: bool = False) -> List[GitBranch]:
        args = [
            "for-each-ref",
            "--format=%(refname:short)%00%(objectname:short)%00%(upstream:short)",
            "refs/heads",
        ]
        if include_remote:
            args.append("refs/remotes")

        _, out, _ = await self._run(args, check=False)

        current = await self.current_branch()
        branches = []
        for line in out.splitlines():
            parts = line.split("\x00")
            if not parts or not parts[0]:
                continue
            name = parts[0]
            sha = parts[1] if len(parts) > 1 else ""
            upstream = parts[2] if len(parts) > 2 else ""
            branches.append(GitBranch(
                name=name,
                sha=sha,
                is_current=(name == current),
                is_remote=name.startswith("origin/") or name.startswith("upstream/"),
                upstream=upstream or None,
            ))
        return branches

    async def current_branch(self) -> str:
        rc, out, _ = await self._run(
            ["symbolic-ref", "--short", "HEAD"], check=False
        )
        if rc == 0:
            return out.strip()
        rc, out, _ = await self._run(["rev-parse", "--short", "HEAD"], check=False)
        return out.strip() if rc == 0 else "DETACHED"

    # ------------------------------------------------------------------
    # FILE HISTORY
    # ------------------------------------------------------------------

    async def file_history(
        self, path: str, limit: int = 20
    ) -> List[GitCommit]:
        return await self.log(limit=limit, path=path)

    async def blame(
        self, path: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        args = ["blame", "--line-porcelain"]
        if start is not None and end is not None:
            args.extend([f"-L{start},{end}"])
        args.append(path)

        _, out, _ = await self._run(args, check=False)

        entries = []
        current: Dict[str, Any] = {}
        for line in out.splitlines():
            if re.match(r"^[0-9a-f]{40} ", line):
                parts = line.split(" ")
                if current:
                    entries.append(current)
                current = {
                    "sha": parts[0],
                    "line": int(parts[2]) if len(parts) > 2 else 0,
                }
            elif line.startswith("author "):
                current["author"] = line[len("author "):]
            elif line.startswith("author-mail "):
                current["email"] = line[len("author-mail "):].strip("<>")
            elif line.startswith("author-time "):
                current["time"] = line[len("author-time "):]
            elif line.startswith("\t"):
                current["content"] = line[1:]
        if current:
            entries.append(current)
        return entries

    # ------------------------------------------------------------------
    # SAFE WRITE OPS (opt-in)
    # ------------------------------------------------------------------

    async def add(self, paths: Optional[List[str]] = None) -> None:
        args = ["add"]
        args.extend(paths or ["."])
        await self._run(args)

    async def commit(self, message: str, allow_empty: bool = False) -> str:
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        _, out, _ = await self._run(args)
        m = re.search(r"\[.*?\s([0-9a-f]+)\]", out)
        return m.group(1) if m else ""

    async def checkout(self, ref: str, create: bool = False) -> None:
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(ref)
        await self._run(args)

    async def pull(self, remote: str = "origin", branch: Optional[str] = None) -> str:
        args = ["pull", remote]
        if branch:
            args.append(branch)
        _, out, _ = await self._run(args)
        return out

    async def push(
        self, remote: str = "origin", branch: Optional[str] = None, set_upstream: bool = False
    ) -> str:
        args = ["push"]
        if set_upstream:
            args.append("-u")
        args.append(remote)
        if branch:
            args.append(branch)
        _, out, _ = await self._run(args)
        return out

    # ------------------------------------------------------------------
    # SUMMARY FOR LLM
    # ------------------------------------------------------------------

    async def as_summary(self, max_chars: int = 1200) -> str:
        try:
            status = await self.get_status()
        except Exception as e:
            return f"git: unavailable ({e})"

        parts = [f"Branch: {status.branch}"]
        if status.last_commit:
            c = status.last_commit
            parts.append(f"Last commit: {c.short_sha} — {c.subject} ({c.author})")
        if status.is_dirty:
            parts.append(
                f"Dirty: {len(status.staged)} staged, "
                f"{len(status.modified)} modified, "
                f"{len(status.untracked)} untracked"
            )
        else:
            parts.append("Working tree clean")
        if status.ahead or status.behind:
            parts.append(f"Ahead {status.ahead} / behind {status.behind}")

        text = " | ".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text