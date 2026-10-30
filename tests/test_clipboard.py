"""Regression tests for real clipboard backends and honest COPY status."""
from __future__ import annotations

import subprocess

import pytest

from agent.utils import clipboard


def test_linux_wayland_uses_native_clipboard(monkeypatch):
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda program: "/bin/" + program if program == "wl-copy" else None)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    assert clipboard.copy_text("héllo") == "copied"
    assert calls[0][0] == ["wl-copy"]
    assert calls[0][1]["input"] == "héllo".encode("utf-8")
    assert calls[0][1]["check"] is True
    assert calls[0][1]["timeout"] == 3


def test_linux_x11_uses_xclip(monkeypatch):
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda p: "/usr/bin/xclip" if p == "xclip" else None)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert clipboard.copy_text("test") == "copied"
    assert calls == [["xclip", "-selection", "clipboard"]]


def test_linux_falls_back_to_xsel_if_xclip_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda p: p if p in ("xclip", "xsel") else None)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "xclip":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert clipboard.copy_text("hello") == "copied"
    assert calls == [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]


def test_macos_uses_pbcopy(monkeypatch):
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: name if name == "pbcopy" else None)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert clipboard.copy_text("text") == "copied"
    assert calls == [["pbcopy"]]


def test_windows_uses_native_clipboard(monkeypatch):
    seen = []
    monkeypatch.setattr(clipboard.sys, "platform", "win32")
    monkeypatch.setattr(clipboard, "_windows_copy", lambda text: seen.append(text))
    assert clipboard.copy_text("🌍") == "copied"
    assert seen == ["🌍"]


def test_remote_ssh_uses_terminal_not_remote_desktop(monkeypatch):
    emitted = []
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":10")
    monkeypatch.setenv("SSH_CONNECTION", "remote")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *a, **kw: pytest.fail("should not copy to remote desktop"))
    assert clipboard.copy_text("hello", terminal_copy=emitted.append) == "terminal"
    assert emitted == ["hello"]


def test_no_backend_is_not_false_success(monkeypatch):
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert clipboard.copy_text("hello") == "unavailable"
    assert clipboard.copy_text("hello", terminal_copy=lambda x: None) == "terminal"
    assert clipboard.copy_text("hello", terminal_copy=lambda x: (_ for _ in ()).throw(RuntimeError())) == "unavailable"


def test_native_failure_does_not_report_copied(monkeypatch):
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "xclip" if name == "xclip" else None)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "xclip")))
    assert clipboard.copy_text("data") == "unavailable"
