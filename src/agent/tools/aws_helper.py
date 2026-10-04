"""
AWS Helper Tool - unified AWS access with automatic fallback.

Preference order:
    1. Call an MCP `mcp__aws__*` tool via the MCP client.
    2. Fall back to the AWS CLI via subprocess.
    3. Fall back to `~/.aws/credentials` when EnvManager is empty.

Credential source: EnvManager.get_aws_env() (canonical), with a fallback
to `~/.aws/credentials` + `~/.aws/config` so a user with an existing AWS
setup works without pasting keys into the panel first.
"""

from __future__ import annotations

import asyncio
import configparser
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


# The MCP AWS server has historically exposed different tool names.
# We try them in order until one succeeds.
_MCP_CALL_CANDIDATES = (
    "mcp__aws__call_aws",
    "mcp__aws__aws_cli",
    "mcp__aws__execute",
    "mcp__aws__run",
)


class AWSHelperTool(BaseTool):
    name = "aws"
    description = (
        "Execute AWS operations. Prefers the AWS MCP server for structured, "
        "audited calls; automatically falls back to the AWS CLI if MCP is "
        "unavailable. Use this instead of raw bash for any AWS task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call", "identity", "list_services"],
                "description": "'call' executes an AWS CLI/service operation",
            },
            "service": {
                "type": "string",
                "description": "AWS service, e.g. s3, ec2, iam, lambda",
            },
            "operation": {
                "type": "string",
                "description": "Operation, e.g. list-buckets, describe-instances",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional CLI args as separate tokens",
            },
            "mcp_tool": {
                "type": "string",
                "description": "Direct MCP tool name if known (mcp__aws__...)",
            },
            "mcp_arguments": {
                "type": "object",
                "description": "Arguments for the MCP tool",
            },
            "prefer": {
                "type": "string",
                "enum": ["auto", "mcp", "cli"],
                "default": "auto",
            },
            "profile": {
                "type": "string",
                "description": "Optional AWS profile name",
            },
        },
        "required": ["action"],
    }
    timeout = 120.0

    def __init__(self, agent: Any = None, workspace: Any = None):
        self.agent = agent
        self.workspace = workspace
        try:
            self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()
        except Exception:
            self.cwd = os.getcwd()

    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "call")

        if action == "identity":
            return await self._identity(params)
        if action == "list_services":
            return self._list_services()
        if action == "call":
            return await self._call(params)
        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "recoverable": True,
        }

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def _identity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prefer = params.get("prefer", "auto")

        for tool_name in _MCP_CALL_CANDIDATES:
            mcp_attempt = await self._try_mcp(
                tool_name=tool_name,
                arguments={
                    "operation": "sts get-caller-identity",
                    "profile": params.get("profile"),
                },
                prefer=prefer,
            )
            if mcp_attempt.get("success") and not mcp_attempt.get("_fallback"):
                return mcp_attempt
            if mcp_attempt.get("_not_registered"):
                continue
            if mcp_attempt.get("_fallback") and prefer == "mcp":
                return mcp_attempt

        return await self._run_cli(
            ["sts", "get-caller-identity"],
            prefer=prefer,
            profile=params.get("profile"),
        )

    def _list_services(self) -> Dict[str, Any]:
        services = [
            "s3", "ec2", "iam", "lambda", "rds", "dynamodb", "sqs", "sns",
            "cloudformation", "cloudwatch", "logs", "ecs", "eks", "ecr",
            "route53", "apigateway", "secretsmanager", "ssm", "kms",
        ]
        return {"success": True, "services": services, "count": len(services)}

    async def _call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prefer = params.get("prefer", "auto")

        # Explicit MCP tool name — caller knows what they want.
        if params.get("mcp_tool"):
            return await self._try_mcp(
                tool_name=str(params["mcp_tool"]),
                arguments=params.get("mcp_arguments") or {},
                prefer=prefer,
            )

        service = params.get("service")
        operation = params.get("operation")
        if not (service and operation):
            return {
                "success": False,
                "error": "Provide 'service' + 'operation', or 'mcp_tool'",
                "recoverable": True,
            }

        # Try MCP call tool candidates.
        for tool_name in _MCP_CALL_CANDIDATES:
            mcp_attempt = await self._try_mcp(
                tool_name=tool_name,
                arguments={
                    "operation": f"{service} {operation}",
                    "params": list(params.get("args") or []),
                },
                prefer=prefer,
            )
            if mcp_attempt.get("success") and not mcp_attempt.get("_fallback"):
                return mcp_attempt
            if mcp_attempt.get("_not_registered"):
                continue
            if mcp_attempt.get("_fallback") and prefer == "mcp":
                return mcp_attempt

        return await self._run_cli(
            [service, operation, *(params.get("args") or [])],
            prefer=prefer,
            profile=params.get("profile"),
        )

    # ------------------------------------------------------------------
    # MCP path
    # ------------------------------------------------------------------

    async def _try_mcp(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        prefer: str = "auto",
    ) -> Dict[str, Any]:
        if prefer == "cli":
            return {"success": False, "_fallback": True, "error": "cli requested"}

        client = self._get_mcp_client()
        if client is None:
            return {
                "success": False,
                "_fallback": True,
                "error": "MCP client not available",
            }

        try:
            available = {t["function"]["name"] for t in client.list_tools()}
        except Exception as exc:
            return {
                "success": False,
                "_fallback": True,
                "error": f"MCP tool listing failed: {exc}",
            }

        if tool_name not in available:
            return {
                "success": False,
                "_fallback": True,
                "_not_registered": True,
                "error": f"MCP tool '{tool_name}' not registered",
            }

        try:
            result = await client.call_tool(tool_name, arguments)
            if isinstance(result, dict) and result.get("success"):
                result["_source"] = "mcp"
                return result
            return {
                "success": False,
                "_fallback": True,
                "error": (result or {}).get("error", "MCP call failed"),
                "mcp_result": result,
            }
        except Exception as exc:
            logger.warning("MCP call failed for %s: %s", tool_name, exc)
            return {"success": False, "_fallback": True, "error": str(exc)}

    def _get_mcp_client(self) -> Optional[Any]:
        if self.agent is None:
            return None
        return getattr(self.agent, "mcp_client", None)

    # ------------------------------------------------------------------
    # Credential resolution (EnvManager + ~/.aws fallback)
    # ------------------------------------------------------------------

    def _read_aws_credentials_file(
        self, profile: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Read ~/.aws/credentials (and config for region) as a fallback when
        EnvManager has nothing. Only used when the canonical store is empty.
        """
        out: Dict[str, str] = {}
        creds_path = Path(os.path.expanduser("~/.aws/credentials"))
        cfg_path = Path(os.path.expanduser("~/.aws/config"))

        section_name = profile or "default"

        if creds_path.is_file():
            try:
                parser = configparser.RawConfigParser()
                parser.read(creds_path)
                if parser.has_section(section_name):
                    for k, v in parser.items(section_name):
                        out[k] = v
            except Exception as exc:
                logger.debug("Could not read %s: %s", creds_path, exc)

        if cfg_path.is_file():
            try:
                parser = configparser.RawConfigParser()
                parser.read(cfg_path)
                # In config, named profiles are "profile <name>", default is "default".
                cfg_section = section_name if section_name == "default" else f"profile {section_name}"
                if parser.has_section(cfg_section):
                    for k, v in parser.items(cfg_section):
                        out.setdefault(k, v)
            except Exception as exc:
                logger.debug("Could not read %s: %s", cfg_path, exc)

        return out

    def _resolve_env(self, profile: Optional[str] = None) -> Tuple[Dict[str, str], str]:
        """
        Build the env for the AWS CLI subprocess.

        Returns (env, source) where source is "env_manager" or "aws_file"
        or "none".
        """
        env = {**os.environ}

        # 1. Canonical: EnvManager (writes here go to ~/.agent/env).
        try:
            canonical = EnvManager.get().get_aws_env()
        except Exception:
            canonical = {}

        if canonical.get("AWS_ACCESS_KEY_ID") and canonical.get("AWS_SECRET_ACCESS_KEY"):
            env.update(canonical)
            if profile:
                env["AWS_PROFILE"] = profile
            return env, "env_manager"

        # 2. Fallback: ~/.aws/credentials + ~/.aws/config.
        file_creds = self._read_aws_credentials_file(profile)
        mapped: Dict[str, str] = {}
        if file_creds.get("aws_access_key_id"):
            mapped["AWS_ACCESS_KEY_ID"] = file_creds["aws_access_key_id"]
        if file_creds.get("aws_secret_access_key"):
            mapped["AWS_SECRET_ACCESS_KEY"] = file_creds["aws_secret_access_key"]
        if file_creds.get("aws_session_token"):
            mapped["AWS_SESSION_TOKEN"] = file_creds["aws_session_token"]
        if file_creds.get("region"):
            mapped["AWS_DEFAULT_REGION"] = file_creds["region"]
            mapped["AWS_REGION"] = file_creds["region"]

        if mapped:
            env.update(mapped)
            if profile:
                env["AWS_PROFILE"] = profile
            return env, "aws_file"

        # 3. Nothing.
        if profile:
            env["AWS_PROFILE"] = profile
            return env, "profile_only"
        return env, "none"

    # ------------------------------------------------------------------
    # CLI path
    # ------------------------------------------------------------------

    async def _run_cli(
        self,
        args: List[str],
        prefer: str = "auto",
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        if prefer == "mcp":
            return {
                "success": False,
                "error": "mcp requested but unavailable; set prefer='auto' for fallback",
                "recoverable": True,
            }

        if not shutil.which("aws"):
            return {
                "success": False,
                "error": (
                    "AWS CLI not found. Install it, or fix the MCP server so "
                    "the MCP path is used."
                ),
                "recoverable": False,
            }

        cmd = ["aws", *args]
        env, source = self._resolve_env(profile=profile)

        if source == "none":
            return {
                "success": False,
                "_source": "cli",
                "error": (
                    "No AWS credentials found. Save them via the AWS panel "
                    "(they go to ~/.agent/env), or set AWS_ACCESS_KEY_ID + "
                    "AWS_SECRET_ACCESS_KEY in the environment, or create "
                    "~/.aws/credentials."
                ),
                "credentials_source": "none",
                "command": " ".join(cmd),
                "recoverable": True,
            }

        t0 = time.time()
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except Exception:
                    pass
            return {
                "success": False,
                "_source": "cli",
                "error": f"aws CLI timed out after {self.timeout}s",
                "command": " ".join(cmd),
                "credentials_source": source,
            }

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")

        result: Dict[str, Any] = {
            "success": proc.returncode == 0,
            "_source": "cli",
            "command": " ".join(cmd),
            "credentials_source": source,
            "exit_code": proc.returncode,
            "stdout": out,
            "stderr": err,
            "duration": time.time() - t0,
        }

        if out.strip().startswith(("{", "[")):
            try:
                result["json"] = json.loads(out)
            except Exception:
                pass

        return result


__all__ = ["AWSHelperTool"]