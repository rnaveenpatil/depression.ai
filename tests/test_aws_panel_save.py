"""
Regression tests for the AWS credential panel + env write path.

Covers the three reported defects:
  1. the secret field was masked with a literal value in .value, so a fresh
     paste looked like it "disappeared";
  2. there was no confirmation that a pasted secret was actually stored;
  3. write_env_file() swallowed filesystem errors, so a failed save could
     still report success.
"""

import os
from pathlib import Path

import pytest

from agent.tui.widgets.aws_panel import (
    AWSPanel,
    _parse_env_text,
    _written_env_path,
)
from agent.utils import env_manager as em


# ----------------------------------------------------------------------
# _parse_env_text
# ----------------------------------------------------------------------

def test_parse_env_text_basic_and_comments():
    text = (
        "# comment\n"
        "\n"
        "AWS_ACCESS_KEY_ID=AKIAEXAMPLE\n"
        "AWS_SECRET_ACCESS_KEY=abc123\n"
    )
    assert _parse_env_text(text) == {
        "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "abc123",
    }


def test_parse_env_text_keeps_equals_signs_in_value():
    # Base64-ish secrets can contain '='; only the first one splits.
    parsed = _parse_env_text("AWS_SECRET_ACCESS_KEY=ab=cd=ef\n")
    assert parsed["AWS_SECRET_ACCESS_KEY"] == "ab=cd=ef"


# ----------------------------------------------------------------------
# write_env_file strict mode
# ----------------------------------------------------------------------

def test_write_env_file_non_strict_does_not_raise(tmp_path):
    # A directory where the file should be makes write_text fail.
    target = tmp_path / ".env"
    target.mkdir()
    em.write_env_file({"A": "1"}, target)  # must not raise


def test_write_env_file_strict_raises_on_failure(tmp_path):
    target = tmp_path / ".env"
    target.mkdir()
    with pytest.raises(Exception):
        em.write_env_file({"A": "1"}, target, strict=True)


def test_write_env_file_round_trip_and_overwrites(tmp_path):
    target = tmp_path / ".env"
    em.write_env_file({"AWS_ACCESS_KEY_ID": "AKIA1", "KEEP": "yes"}, target)
    em.write_env_file({"AWS_ACCESS_KEY_ID": "AKIA2"}, target)

    data = em.load_env_file(target)
    assert data["AWS_ACCESS_KEY_ID"] == "AKIA2"
    assert data["KEEP"] == "yes"
    # overwritten in place, not duplicated
    raw = target.read_text()
    assert raw.count("AWS_ACCESS_KEY_ID=") == 1


def test_write_env_file_sets_0600(tmp_path):
    target = tmp_path / ".env"
    em.write_env_file({"A": "1"}, target)
    assert oct(target.stat().st_mode)[-3:] == "600"


# ----------------------------------------------------------------------
# Secret field: placeholder, never a literal mask value
# ----------------------------------------------------------------------

class _Panel(AWSPanel):
    """Instantiate AWSPanel without a running Textual app."""

    def __init__(self, app_ref, store):
        self._app = app_ref
        self._scan_pos = 0
        self._scan_active = False
        self._scan_passes = 0
        self._radar_i = 0
        self._radar_active = False
        self._breath_i = 0
        self._trace_i = 0
        self._trace_active = False
        self._trace_ok = True
        self._title = None
        self._strength = None
        self._trace = None
        self._status = None
        self._region_static = None
        self._mcp_status = None
        self._saved_at = None
        self._secret_replaced = False
        self._env_cb = lambda snapshot: None
        self._store = store
        self._sentinel = object()

    # Bypass EnvManager so tests don't touch the real ~/.agent/env
    def _read_canonical(self):
        return dict(self._store)

    # Animation helpers need a live Textual app/event loop.
    def start_scan(self, passes: int = 2) -> None:
        pass

    def start_radar(self, duration_ticks: int = 30) -> None:
        pass

    def start_trace(self, ok: bool = True) -> None:
        pass

    def refresh_mcp_status(self) -> None:
        pass

    def refresh_values(self) -> None:
        pass

    def _render_title(self) -> None:
        pass

    def _render_strength(self) -> None:
        pass

    def _render_saved_at(self) -> None:
        pass

    def _render_trace_running(self) -> None:
        pass

    def _render_trace_final(self) -> None:
        pass


def test_secret_field_is_prefilled_with_stored_secret():
    """The field keeps the stored secret (masked) so a paste stays visible.

    Regression: the panel used to write a literal '••••••••' into .value and
    blank the field on every save, so a pasted key looked like it vanished.
    """
    panel = _Panel(
        None,
        {"access_key": "AKIA1", "secret_key": "realsecret", "region": "us-east-1"},
    )
    assert panel._stored_secret() == "realsecret"

    empty = _Panel(None, {"access_key": "", "secret_key": "", "region": ""})
    assert empty._stored_secret() == ""


@pytest.fixture(autouse=True)
def _isolate_real_env_files(monkeypatch, tmp_path):
    """Belt-and-braces: no test in this module may touch the real .env."""
    monkeypatch.setattr(em, "find_env_file", lambda: tmp_path / ".env")
    monkeypatch.setenv("HOME", str(tmp_path))


def test_save_reports_secret_replaced(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    store = {}
    panel = _Panel(None, store)

    # CRITICAL: redirect EnvManager's own write target too, not just the
    # panel's verification path. save_aws_credentials() calls
    # find_env_file() internally, which would otherwise write the real
    # project .env.
    monkeypatch.setattr(em, "find_env_file", lambda: env)
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel._written_env_path", lambda: str(env)
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel._current_region",
        lambda self: "us-west-2",
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel.query_one",
        lambda self, sel, cls=None: _Field(sel, store),
    )

    store.update(
        {"access_key": "AKIAOLD", "secret_key": "oldsecret", "region": "us-east-1"}
    )
    _Field.values = {"#aws-key": "AKIAREAL", "#aws-secret": "brandnewsecret"}

    result = panel.save_credentials()

    assert result["ok"] is True, result
    assert panel._secret_replaced is True
    assert panel._saved_at is not None

    # the new secret really landed in the file
    written = em.load_env_file(env)
    assert written["AWS_SECRET_ACCESS_KEY"] == "brandnewsecret"
    assert written["AWS_ACCESS_KEY_ID"] == "AKIAREAL"


def test_save_keeps_stored_secret_when_field_left_alone(monkeypatch, tmp_path):
    """Field prefilled with the stored secret: saving must be a no-op for it.

    Because the field now holds the real stored value, 'unchanged' is
    detected by comparing the typed value to the canonical one.
    """
    env = tmp_path / ".env"
    store = {
        "access_key": "AKIAKEEP",
        "secret_key": "keepme",
        "region": "us-east-1",
    }
    panel = _Panel(None, store)

    monkeypatch.setattr(em, "find_env_file", lambda: env)
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel._written_env_path", lambda: str(env)
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel._current_region",
        lambda self: "us-east-1",
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel.query_one",
        lambda self, sel, cls=None: _Field(sel, store),
    )

    # unchanged: field shows the stored secret
    _Field.values = {"#aws-key": "AKIAKEEP", "#aws-secret": "keepme"}

    result = panel.save_credentials()

    assert result["ok"] is True, result
    assert panel._secret_replaced is False
    written = em.load_env_file(env)
    assert written["AWS_SECRET_ACCESS_KEY"] == "keepme"


def test_save_with_empty_secret_keeps_stored_one(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    store = {
        "access_key": "AKIAKEEP",
        "secret_key": "keepme",
        "region": "us-east-1",
    }
    panel = _Panel(None, store)

    monkeypatch.setattr(em, "find_env_file", lambda: env)
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel._written_env_path", lambda: str(env)
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel._current_region",
        lambda self: "us-east-1",
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel.query_one",
        lambda self, sel, cls=None: _Field(sel, store),
    )

    _Field.values = {"#aws-key": "AKIAKEEP", "#aws-secret": ""}

    result = panel.save_credentials()

    assert result["ok"] is True, result
    assert panel._secret_replaced is False
    written = em.load_env_file(env)
    assert written["AWS_SECRET_ACCESS_KEY"] == "keepme"


def test_save_fails_when_disk_holds_a_different_secret(monkeypatch, tmp_path):
    """A write that silently kept the old value must not report success."""
    env = tmp_path / ".env"
    env.write_text(
        "AWS_ACCESS_KEY_ID=AKIAREAL\nAWS_SECRET_ACCESS_KEY=stale\n"
    )
    store = {"access_key": "AKIAREAL", "secret_key": "stale", "region": "us-east-1"}
    panel = _Panel(None, store)

    monkeypatch.setattr(em, "find_env_file", lambda: env)
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel._written_env_path", lambda: str(env)
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel._current_region",
        lambda self: "us-east-1",
    )
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.AWSPanel.query_one",
        lambda self, sel, cls=None: _Field(sel, store),
    )
    # force the "write" to be a no-op so disk keeps the stale secret
    monkeypatch.setattr(
        "agent.tui.widgets.aws_panel.EnvManager.get",
        lambda: _NoWriteEnvManager(),
    )

    _Field.values = {"#aws-key": "AKIAREAL", "#aws-secret": "brandnewsecret"}

    result = panel.save_credentials()

    assert result["ok"] is False
    assert "different secret" in result["error"]


class _NoWriteEnvManager:
    """EnvManager stub whose save silently does nothing."""

    def save_aws_credentials(self, **kwargs):
        return None

    def get_aws_credentials(self):
        return {}

    def subscribe(self, cb):
        return None

    def unsubscribe(self, cb):
        return None


class _Field:
    """Minimal stand-in for a Textual Input."""

    values: dict = {}

    def __init__(self, selector, store):
        self.selector = selector
        self.store = store
        self._value = self.values.get(selector, "")

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def __getattr__(self, item):
        if item == "value":
            return self._value
        raise AttributeError(item)