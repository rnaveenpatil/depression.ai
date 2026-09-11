"""
Session State - Runtime state of an active session.

Tracks:
    - Status (idle, thinking, executing, paused, error)
    - Token usage and cost
    - Current plan and current task
    - Tool call counters and last error
    - Session-scoped metadata
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger

logger = get_logger(__name__)


class SessionStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"        # waiting for user permission
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class SessionState:
    """Mutable runtime state for a single session."""

    # Status
    status: str = SessionStatus.IDLE.value
    turn: int = 0

    # Metrics
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    tool_calls: int = 0
    errors: int = 0

    # Model
    provider: str = ""
    model: str = ""
    last_temperature: float = 0.1

    # Current work
    current_plan_id: Optional[str] = None
    current_task_id: Optional[str] = None
    last_error: Optional[str] = None

    # Timing
    started_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    # Free-form
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def touch(self) -> None:
        self.last_active_at = time.time()

    def add_usage(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
    ) -> None:
        self.tokens_in += int(tokens_in or 0)
        self.tokens_out += int(tokens_out or 0)
        self.cost += float(cost or 0.0)
        self.touch()

    def record_tool_call(self) -> None:
        self.tool_calls += 1
        self.touch()

    def record_error(self, message: str) -> None:
        self.errors += 1
        self.last_error = message
        self.touch()

    def set_status(self, status: str | SessionStatus) -> None:
        self.status = status.value if isinstance(status, SessionStatus) else str(status)
        self.touch()

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def duration(self) -> float:
        return time.time() - self.started_at

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def reset(self) -> None:
        self.status = SessionStatus.IDLE.value
        self.turn = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.tool_calls = 0
        self.errors = 0
        self.current_plan_id = None
        self.current_task_id = None
        self.last_error = None
        self.started_at = time.time()
        self.last_active_at = self.started_at
        self.metadata.clear()