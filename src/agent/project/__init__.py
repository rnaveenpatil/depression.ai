"""
Project Module - Codebase awareness and workspace management.

Exports:
    WorkspaceManager  — manages the working directory and project root
    ProjectScanner    — scans and indexes project files
    GitProject        — git-level operations and analysis
    AWSProject        — AWS project detection and awareness
"""

from agent.project.workspace import WorkspaceManager
from agent.project.scanner import ProjectScanner, FileInfo, ScanResult
from agent.project.git import GitProject, GitStatus, GitCommit, GitBranch
from agent.project.awsproject import AWSProject, AWSProjectInfo

__all__ = [
    "WorkspaceManager",
    "ProjectScanner",
    "FileInfo",
    "ScanResult",
    "GitProject",
    "GitStatus",
    "GitCommit",
    "GitBranch",
    "AWSProject",
    "AWSProjectInfo",
]