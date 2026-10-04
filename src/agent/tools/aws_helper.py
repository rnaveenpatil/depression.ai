"""
AWS Helper Tool - unified AWS access with automatic fallback.

Preference order:
    1. Call `mcp__aws__<tool>` via the MCP client (structured, audited)
    2. Fall back to the AWS CLI via the shell (bash/terminal)

The LLM gets a single tool name to remember; the fallback logic lives
here so the model never has to guess.

Credential source: EnvManager.get_aws_env(). This is the one canonical
place all AWS consumers (CLI, boto3, MCP) must read from.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


class AWSHelperTool(BaseTool):
    name = "aws"
    description = (
        "Execute AWS operations. Prefers the AWS MCP server for structured, "
        "audited calls; automatically falls back to the AWS CLI via bash if "
        "MCP is unavailable. Use this instead of raw bash for any AWS task."
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
        self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()

    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "call")

        if action == "identity":
            return await self._identity(params)
        if action == "list_services":
            return self._list_services()
        if action == "call":
            return await self._call(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def _identity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mcp_attempt = await self._try_mcp(
            tool_name="mcp__aws__call_aws",
            arguments={
                "operation": "sts get-caller-identity",
                "profile": params.get("profile"),
            },
            prefer=params.get("prefer", "auto"),
        )
        if mcp_attempt.get("success") and not mcp_attempt.get("_fallback"):
            return mcp_attempt

        return await self._run_cli(
            ["sts", "get-caller-identity"],
            prefer=params.get("prefer", "auto"),
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

        if params.get("mcp_tool"):
            return await self._try_mcp(
                tool_name=str(params["mcp_tool"]),
                arguments=params.get("mcp_arguments") or {},
                prefer=prefer,
            )

        service = params.get("service")
        operation = params.get("operation")
        if service and operation:
            mcp_attempt = await self._try_mcp(
                tool_name="mcp__aws__call_aws",
                arguments={
                    "operation": f"{service} {operation}",
                    "params": list(params.get("args") or []),
                },
                prefer=prefer,
            )
            if mcp_attempt.get("success") and not mcp_attempt.get("_fallback"):
                return mcp_attempt

            return await self._run_cli(
                [service, operation, *(params.get("args") or [])],
                prefer=prefer,
            )

        return {
            "success": False,
            "error": "Provide 'service' + 'operation', or 'mcp_tool'",
        }

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
    # CLI fallback
    # ------------------------------------------------------------------

    def _build_aws_subprocess_env(self, profile: Optional[str] = None) -> Dict[str, str]:
        """
        Build the env the AWS CLI subprocess should see.

        = os.environ (updated live by EnvManager)
        + canonical AWS vars from EnvManager.get_aws_env()   <- always overrides
        + optional AWS_PROFILE
        """
        env = os.environ.copy()
        env.update(EnvManager.get().get_aws_env())
        if profile:
            env["AWS_PROFILE"] = profile
        return env

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
            }

        if not shutil.which("aws"):
            return {
                "success": False,
                "error": (
                    "AWS CLI not found. Install it, or fix the MCP server so "
                    "the MCP path is used."
                ),
            }

        cmd = ["aws", *args]
        env = self._build_aws_subprocess_env(profile=profile)

        # Fail fast with a clear message if no creds at all
        if not env.get("AWS_ACCESS_KEY_ID") and not env.get("AWS_PROFILE"):
            return {
                "success": False,
                "_source": "cli",
                "error": (
                    "No AWS credentials available. Add them in the AWS panel "
                    "or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
                ),
                "command": " ".join(cmd),
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
            }

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")

        result: Dict[str, Any] = {
            "success": proc.returncode == 0,
            "_source": "cli",
            "command": " ".join(cmd),
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