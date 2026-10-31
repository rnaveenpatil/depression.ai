"""Single verification controller shared by every loop execution path.

The controller owns the bookkeeping that used to be scattered across
``AgentLoop``:

* which mutations are still unverified,
* which targets changed in the current batch,
* how many batch fix->verify rounds have been spent,
* reusable verification evidence (a successful test/build command), and
* the user-visible ``UNVERIFIED`` report.

Keeping this in one place means the free-form loop and the plan-task loop
apply the same rules, and a successful test run can satisfy verification
without spending another model turn.

The controller operates on the existing ``LoopContext`` fields so callers
and telemetry keep observing the same state.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional

VERIFICATION_MODES = ("per_step", "batch", "off")
MAX_BATCH_VERIFY_ATTEMPTS = 4

# Commands whose successful output is itself evidence that pending changes
# are consistent. Deliberately excludes arbitrary shell commands so a plain
# ``ls`` can never clear a pending write.
_TEST_COMMAND_RE = re.compile(
    r"(?:^|[\s;&|])("
    r"pytest|py\.test|tox|nox|"
    r"npm\s+(?:run\s+)?(?:test|build|lint|typecheck)|"
    r"pnpm\s+(?:run\s+)?(?:test|build|lint|typecheck)|"
    r"yarn\s+(?:run\s+)?(?:test|build|lint|typecheck)|"
    r"bun\s+(?:run\s+)?(?:test|build|lint|typecheck)|"
    r"cargo\s+(?:test|check|build|clippy)|"
    r"go\s+(?:test|build|vet)|"
    r"mvn\s+(?:test|verify|package)|"
    r"gradle(?:w)?\s+(?:test|check|build)|"
    r"make\s+(?:test|check|build)|"
    r"ruff|mypy|pyright|eslint|tsc|"
    r"jest|vitest|mocha|playwright\s+test|"
    r"python\s+-m\s+(?:pytest|unittest|compileall|mypy|ruff)"
    r")\b",
    re.I,
)


def _command_from_args(args: Dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    for key in ("command", "cmd", "script", "shell", "exec", "cmdline"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


class VerificationController:
    """Owns pending-verification state and reusable evidence."""

    def __init__(
        self,
        context: Any,
        *,
        mode: str = "batch",
        max_batch_attempts: int = MAX_BATCH_VERIFY_ATTEMPTS,
        max_nudges: int = 1,
    ) -> None:
        self.context = context
        self.mode = mode if mode in VERIFICATION_MODES else "batch"
        self.max_batch_attempts = max(0, int(max_batch_attempts))
        self.max_nudges = max(0, int(max_nudges))

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def pending(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return list(getattr(self.context, "unverified_mutations", []) or [])

    @property
    def attempts(self) -> int:
        return int(getattr(self.context, "batch_verify_attempts", 0) or 0)

    @property
    def changed_targets(self) -> List[str]:
        return list(getattr(self.context, "batch_changed_targets", []) or [])

    # ------------------------------------------------------------------
    # Recording mutations / reads
    # ------------------------------------------------------------------

    def record_mutation(
        self,
        tool: str,
        args: Dict[str, Any],
        target: Optional[str],
        call_id: str,
    ) -> None:
        if not self.enabled:
            return
        # A plain shell command is evidence in its own output. Only shell
        # commands that redirect to a durable target need read-back.
        if target and str(target).startswith("cmd::"):
            return
        if target and target not in self.context.batch_changed_targets:
            self.context.batch_changed_targets.append(target)

        for entry in list(self.context.unverified_mutations):
            if entry.get("tool") == tool and entry.get("target") == target:
                entry["time"] = time.time()
                entry["call_id"] = call_id
                return
        self.context.unverified_mutations.append({
            "tool": tool,
            "args": args,
            "target": target,
            "call_id": call_id,
            "time": time.time(),
        })

    def record_read_target(self, target: Optional[str]) -> None:
        if not self.enabled or not target:
            return
        self.clear_target(target)

    def clear_target(self, target: str) -> int:
        if not getattr(self.context, "unverified_mutations", None):
            return 0
        remaining = [
            m for m in self.context.unverified_mutations
            if m.get("target") != target
        ]
        cleared = len(self.context.unverified_mutations) - len(remaining)
        self.context.unverified_mutations = remaining
        return cleared

    def clear_all(self) -> None:
        self.context.unverified_mutations = []

    # ------------------------------------------------------------------
    # Reusable evidence
    # ------------------------------------------------------------------

    def record_test_evidence(
        self,
        tool: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> bool:
        """Reuse a successful test/build run as verification evidence.

        Returns True when evidence was recorded. Only commands that actually
        exercise the project count; a successful redirect write or ``ls`` does
        not.
        """
        if not self.enabled:
            return False
        if not isinstance(result, dict) or not result.get("success", False):
            return False
        command = _command_from_args(args)
        if not command or not _TEST_COMMAND_RE.search(command):
            return False

        output = (
            result.get("content")
            or result.get("output")
            or result.get("stdout")
            or result.get("result")
            or ""
        )
        evidence = {
            "time": time.time(),
            "tool": tool,
            "command": command[:400],
            "fingerprint": _fingerprint(command + "\n" + str(output)[:4000]),
        }
        self.context.metadata["verification_evidence"] = evidence
        self.refresh_from_evidence()
        return True

    def refresh_from_evidence(self) -> int:
        """Clear mutations that predate the newest successful test run."""
        if not self.enabled:
            return 0
        evidence = self.context.metadata.get("verification_evidence") or {}
        try:
            evidence_time = float(evidence.get("time") or 0)
        except (TypeError, ValueError):
            evidence_time = 0.0
        if evidence_time <= 0:
            return 0

        before = len(self.context.unverified_mutations)
        self.context.unverified_mutations = [
            m for m in self.context.unverified_mutations
            if float(m.get("time") or 0) > evidence_time
        ]
        return before - len(self.context.unverified_mutations)

    def evidence_summary(self) -> str:
        evidence = self.context.metadata.get("verification_evidence") or {}
        if not evidence.get("command"):
            return ""
        return (
            f"Reused test evidence ({evidence.get('command')}) at "
            f"{time.strftime('%H:%M:%S', time.localtime(evidence.get('time', 0)))}."
        )

    # ------------------------------------------------------------------
    # Batch verification gate
    # ------------------------------------------------------------------

    def should_verify(
        self,
        response_text: str,
        *,
        declared_unverified: bool,
        should_stop: bool,
        iteration: int,
        max_iterations: int,
        max_batch_attempts: Optional[int] = None,
    ) -> bool:
        if self.mode != "batch" or should_stop:
            return False
        if not self.enabled:
            return False
        self.refresh_from_evidence()
        if not self.pending:
            return False
        budget = (
            self.max_batch_attempts
            if max_batch_attempts is None
            else max(0, int(max_batch_attempts))
        )
        if self.attempts >= budget:
            return False
        if iteration >= max_iterations:
            return False
        if declared_unverified:
            return False
        return True

    def begin_attempt(self) -> int:
        self.context.batch_verify_attempts += 1
        return self.context.batch_verify_attempts

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def finalize_report(
        self,
        text: str,
        *,
        declared_unverified: bool,
    ) -> tuple[str, bool]:
        self.refresh_from_evidence()
        pending = self.pending
        if self.mode != "batch" or not pending:
            return text, False

        targets = sorted({str(m.get("target") or "?") for m in pending})
        tools = sorted({str(m.get("tool") or "?") for m in pending})
        attempted = self.attempts
        total = self.max_batch_attempts
        self.context.metadata["unverified_targets"] = targets
        self.context.metadata["unverified_declared"] = True

        bullets = "\n".join(f"  - {t}" for t in targets)
        if declared_unverified:
            report = "\n\nRemaining unverified targets:\n" + bullets
        else:
            report = (
                f"\n\nUNVERIFIED: {len(targets)} change(s) could not be "
                f"verified after {attempted}/{total} verification attempt(s).\n"
                f"Changed via: {', '.join(tools)}\n"
                f"Targets:\n{bullets}\n"
                "Run the relevant checks or read the files back before "
                "treating these changes as complete."
            )
        if not text.strip():
            return report.lstrip("\n"), True
        return text.rstrip() + report, True


__all__ = [
    "VerificationController",
    "VERIFICATION_MODES",
    "MAX_BATCH_VERIFY_ATTEMPTS",
]
