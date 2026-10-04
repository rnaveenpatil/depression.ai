"""Regression tests for permission action inference and the session cache.

Guards the bug where ``PermissionManager._infer_action`` ignored the explicit
``action`` param that unified tools (``filesystem``, ``git``, ...) dispatch on.
Every ``filesystem`` call was therefore inferred as the generic action
``"execute"``, which had two consequences:

1. the confirmation prompt mislabelled the operation, and
2. the session cache key (``tool:action:path``) collided across *different*
   operations on the same path -- so approving one ``delete`` silently
   pre-authorised every later filesystem action on that same path.
"""

from __future__ import annotations

import pytest

from agent.permissions.manager import PermissionManager, _looks_destructive


class _Recorder:
    """Records confirmation prompts and optionally approves them."""

    def __init__(self, approve: bool = True):
        self.approve = approve
        self.prompts: list[tuple[str, str]] = []

    async def __call__(self, request, verdict) -> bool:
        self.prompts.append((request.tool, request.action))
        return self.approve

    @property
    def count(self) -> int:
        return len(self.prompts)


def _manager(approve: bool = True) -> tuple[PermissionManager, _Recorder]:
    pm = PermissionManager()
    rec = _Recorder(approve=approve)
    pm.set_confirm_callback(rec)
    return pm, rec


# ----------------------------------------------------------------------
# action inference
# ----------------------------------------------------------------------

@pytest.mark.parametrize("action", ["read", "write", "delete", "list", "stat", "mkdir", "move", "copy"])
def test_infer_action_prefers_explicit_action_param(action):
    """The explicit ``action`` param must win over tool-name matching."""
    pm = PermissionManager()
    assert pm._infer_action("filesystem", {"action": action}) == action


def test_infer_action_does_not_collapse_to_execute():
    """Regression: delete/write must not both infer to the generic 'execute'."""
    pm = PermissionManager()
    delete = pm._infer_action("filesystem", {"action": "delete"})
    write = pm._infer_action("filesystem", {"action": "write"})
    assert delete != write
    assert delete != "execute"


@pytest.mark.parametrize("key", ["action", "operation", "op"])
def test_infer_action_accepts_operation_aliases(key):
    pm = PermissionManager()
    assert pm._infer_action("filesystem", {key: "delete"}) == "delete"


def test_infer_action_falls_back_to_tool_name():
    """With no explicit action, name-based inference still applies."""
    pm = PermissionManager()
    assert pm._infer_action("filesystem_read", {}) == "read"
    assert pm._infer_action("write_file", {}) == "write"


def test_infer_action_ignores_blank_explicit_action():
    pm = PermissionManager()
    assert pm._infer_action("write_file", {"action": "   "}) == "write"


# ----------------------------------------------------------------------
# session-cache isolation
# ----------------------------------------------------------------------

async def test_approving_delete_does_not_authorize_rmdir_on_same_path():
    """Regression: distinct destructive actions on one path must each ask."""
    pm, rec = _manager(approve=True)

    await pm.check_permission("filesystem", {"action": "delete", "path": "src/a.py"})
    assert rec.count == 1, "destructive 'delete' should prompt"

    await pm.check_permission("filesystem", {"action": "rmdir", "path": "src/a.py"})
    assert rec.count == 2, (
        "a previously approved 'delete' must not silently authorise 'rmdir' "
        "on the same path via a colliding session-cache key"
    )


async def test_allowed_delete_is_remembered_for_the_same_action():
    """The cache must still work within one action (no spurious re-prompts)."""
    pm, rec = _manager(approve=True)
    await pm.check_permission("filesystem", {"action": "delete", "path": "src/a.py"})
    await pm.check_permission("filesystem", {"action": "delete", "path": "src/a.py"})
    assert rec.count == 1


async def test_denied_delete_is_not_remembered_as_allowed():
    pm, rec = _manager(approve=False)
    ok, _ = await pm.check_permission("filesystem", {"action": "delete", "path": "src/a.py"})
    assert ok is False
    ok, _ = await pm.check_permission("filesystem", {"action": "delete", "path": "src/a.py"})
    assert ok is False, "a denied action must not become allowed on retry"


# ----------------------------------------------------------------------
# decision matrix
# ----------------------------------------------------------------------

async def test_env_read_is_denied_without_prompting():
    pm, rec = _manager(approve=True)
    ok, _ = await pm.check_permission("filesystem", {"action": "read", "path": ".env"})
    assert ok is False
    assert rec.count == 0, ".env denial is hard-coded and must not prompt"


async def test_benign_filesystem_write_is_silent_allow():
    pm, rec = _manager(approve=False)
    ok, _ = await pm.check_permission("filesystem", {"action": "write", "path": "src/a.py"})
    assert ok is True
    assert rec.count == 0


async def test_benign_terminal_command_is_silent_allow():
    pm, rec = _manager(approve=False)
    ok, _ = await pm.check_permission("terminal", {"command": "ls -la"})
    assert ok is True
    assert rec.count == 0


async def test_destructive_terminal_command_prompts():
    pm, rec = _manager(approve=True)
    ok, _ = await pm.check_permission("terminal", {"command": "rm -rf build"})
    assert ok is True
    assert rec.count == 1


async def test_destructive_terminal_command_denied_when_user_declines():
    pm, rec = _manager(approve=False)
    ok, _ = await pm.check_permission("terminal", {"command": "rm -rf build"})
    assert ok is False
    assert rec.count == 1


# ----------------------------------------------------------------------
# destructive detection
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool,action,params",
    [
        ("filesystem", "delete", {"path": "a.py"}),
        ("filesystem", "rmdir", {"path": "a"}),
        ("filesystem", "remove", {"path": "a"}),
        ("terminal", "execute", {"command": "rm -rf build"}),
        ("git", "reset_hard", {}),
    ],
)
def test_looks_destructive_flags_destructive_calls(tool, action, params):
    is_destructive, reason = _looks_destructive(tool, action, params)
    assert is_destructive is True
    assert reason


@pytest.mark.parametrize(
    "tool,action,params",
    [
        ("filesystem", "read", {"path": "a.py"}),
        ("filesystem", "write", {"path": "a.py"}),
        ("filesystem", "list", {"path": "."}),
        ("terminal", "execute", {"command": "ls -la"}),
        ("git", "status", {}),
    ],
)
def test_looks_destructive_allows_benign_calls(tool, action, params):
    is_destructive, _ = _looks_destructive(tool, action, params)
    assert is_destructive is False
