"""
AWS Project - Detection and awareness of AWS-related projects.

Responsibilities:
    - Detect AWS tooling (CDK, SAM, Serverless, Terraform, Amplify, Chalice)
    - Parse configuration (cdk.json, samconfig.toml, serverless.yml)
    - Detect region, account, stack names
    - Parse CloudFormation / SAM / Serverless templates for resources
    - Summarize AWS project state for the LLM
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class AWSProjectInfo:
    root: str
    is_aws_project: bool = False
    frameworks: List[str] = field(default_factory=list)
    region: Optional[str] = None
    account_id: Optional[str] = None
    profile: Optional[str] = None
    stacks: List[str] = field(default_factory=list)
    resources: Dict[str, List[str]] = field(default_factory=dict)
    config_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ======================================================================
# DETECTORS
# ======================================================================

FRAMEWORK_MARKERS = {
    "aws-cdk": ["cdk.json", "cdk.context.json"],
    "aws-sam": ["samconfig.toml", "template.yaml", "template.yml"],
    "serverless": ["serverless.yml", "serverless.yaml", "serverless.ts"],
    "terraform": [],              # detected by *.tf files
    "amplify": ["amplify.yml", "amplifyconfiguration.json"],
    "chalice": [".chalice/config.json", "chalicelib"],
    "elastic-beanstalk": [".elasticbeanstalk/config.yml"],
    "copilot": ["copilot/"],
    "app-runner": ["apprunner.yaml"],
    "cdk8s": ["cdk8s.yaml"],
    "pulumi-aws": ["Pulumi.yaml"],
    "aws-cdk-python": ["app.py"],  # combined with cdk.json
}

REGION_PATTERN = re.compile(r"\b([a-z]{2}(?:-gov)?-[a-z]+-\d)\b")
ACCOUNT_PATTERN = re.compile(r"\b\d{12}\b")


# ======================================================================
# AWS PROJECT
# ======================================================================

class AWSProject:
    """
    Detects and describes AWS-flavored projects.
    """

    def __init__(self, project_dir: str | Path):
        self.root = Path(project_dir).resolve()
        self._info: Optional[AWSProjectInfo] = None

    # ------------------------------------------------------------------

    async def detect(self, force: bool = False) -> AWSProjectInfo:
        if self._info is not None and not force:
            return self._info

        info = AWSProjectInfo(root=str(self.root))

        try:
            self._detect_frameworks(info)
            self._detect_region_and_account(info)
            self._detect_stacks(info)
        except Exception as e:
            logger.debug(f"AWS project detection failed: {e}")
            info.notes.append(f"detection error: {e}")

        info.is_aws_project = bool(info.frameworks) or bool(info.config_files)
        self._info = info
        return info

    # ------------------------------------------------------------------

    def _detect_frameworks(self, info: AWSProjectInfo) -> None:
        for framework, markers in FRAMEWORK_MARKERS.items():
            for marker in markers:
                p = self.root / marker
                if p.exists():
                    info.frameworks.append(framework)
                    info.config_files.append(marker)
                    break

        # Terraform via glob
        if any(self.root.glob("*.tf")):
            if "terraform" not in info.frameworks:
                info.frameworks.append("terraform")
            for p in list(self.root.glob("*.tf"))[:5]:
                info.config_files.append(p.name)

        # CDK Python (app.py + cdk.json)
        if (self.root / "cdk.json").exists() and (self.root / "app.py").exists():
            if "aws-cdk-python" not in info.frameworks:
                info.frameworks.append("aws-cdk-python")

    def _detect_region_and_account(self, info: AWSProjectInfo) -> None:
        # Priority order for region
        sources: List[Path] = []
        for name in (
            "cdk.json", "samconfig.toml", "serverless.yml", "serverless.yaml",
            ".aws/config", "amplify.yml",
        ):
            p = self.root / name
            if p.exists():
                sources.append(p)

        for src in sources:
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            m = REGION_PATTERN.search(text)
            if m and not info.region:
                info.region = m.group(1)

            m = ACCOUNT_PATTERN.search(text)
            if m and not info.account_id:
                info.account_id = m.group(0)

        # Env vars as last resort
        if not info.region:
            info.region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not info.profile:
            info.profile = os.environ.get("AWS_PROFILE")

    def _detect_stacks(self, info: AWSProjectInfo) -> None:
        # CDK stacks from cdk.json / python files
        if "aws-cdk" in info.frameworks or "aws-cdk-python" in info.frameworks:
            for py in list(self.root.glob("**/*_stack.py"))[:20]:
                m = re.search(r"class\s+(\w+Stack)\b", py.read_text(errors="replace"))
                if m:
                    info.stacks.append(m.group(1))
            for ts in list(self.root.glob("**/*-stack.ts"))[:20]:
                m = re.search(r"class\s+(\w+Stack)\b", ts.read_text(errors="replace"))
                if m:
                    info.stacks.append(m.group(1))

        # Serverless service name
        for name in ("serverless.yml", "serverless.yaml"):
            p = self.root / name
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    m = re.search(r"^service:\s*(.+)$", text, re.MULTILINE)
                    if m:
                        info.stacks.append(m.group(1).strip().strip("'\""))
                    self._scan_serverless_resources(text, info)
                except Exception:
                    pass

        # SAM template.yaml
        for name in ("template.yaml", "template.yml"):
            p = self.root / name
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    info.notes.append(f"{name} present ({len(text.splitlines())} lines)")
                    self._scan_sam_resources(text, info)
                except Exception:
                    pass

        # Terraform resource blocks
        if "terraform" in info.frameworks:
            resources = set()
            for tf in list(self.root.glob("*.tf"))[:30]:
                try:
                    text = tf.read_text(encoding="utf-8", errors="replace")
                    for m in re.finditer(r'resource\s+"(aws_[a-z0-9_]+)"', text):
                        resources.add(m.group(1))
                except Exception:
                    pass
            if resources:
                info.resources["terraform"] = sorted(resources)[:40]

    def _scan_serverless_resources(self, text: str, info: AWSProjectInfo) -> None:
        """Extract function names from serverless.yml (naive YAML scan)."""
        funcs = []
        in_functions = False
        for line in text.splitlines():
            if re.match(r"^functions:", line):
                in_functions = True
                continue
            if in_functions:
                if re.match(r"^[a-zA-Z]", line) and not line.startswith(" "):
                    break
                m = re.match(r"^  (\w+):", line)
                if m:
                    funcs.append(m.group(1))
        if funcs:
            info.resources["serverless_functions"] = funcs

    def _scan_sam_resources(self, text: str, info: AWSProjectInfo) -> None:
        """Extract logical resource IDs from a SAM template."""
        resources = re.findall(r"^  (\w+):\s*$", text, re.MULTILINE)
        if resources:
            info.resources["sam_resources"] = resources[:40]

    # ------------------------------------------------------------------

    async def as_prompt(self, max_chars: int = 800) -> str:
        info = await self.detect()
        if not info.is_aws_project:
            return ""

        parts = ["AWS project detected:"]
        if info.frameworks:
            parts.append(f"  Frameworks: {', '.join(sorted(set(info.frameworks)))}")
        if info.region:
            parts.append(f"  Region: {info.region}")
        if info.account_id:
            parts.append(f"  Account: {info.account_id}")
        if info.profile:
            parts.append(f"  Profile: {info.profile}")
        if info.stacks:
            parts.append(f"  Stacks: {', '.join(info.stacks[:5])}")
        if "terraform" in info.resources:
            resources = info.resources["terraform"]
            parts.append(
                f"  Terraform resources: {', '.join(resources[:8])}"
                + (f" (+{len(resources) - 8} more)" if len(resources) > 8 else "")
            )
        if "serverless_functions" in info.resources:
            funcs = info.resources["serverless_functions"]
            parts.append(f"  Serverless functions: {', '.join(funcs[:8])}")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text

    async def get_info(self) -> Dict[str, Any]:
        info = await self.detect()
        return info.__dict__

    def reset(self) -> None:
        self._info = None