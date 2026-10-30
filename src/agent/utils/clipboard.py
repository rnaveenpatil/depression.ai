"""Cross-platform clipboard with honest success reporting for terminal applications.

Native clipboard transfers return "copied" only after a successful platform call.
Terminal OSC 52 fallback cannot be verified; callers must not label it "copied".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable, Optional


def _windows_copy(text: str) -> None:
    """Write Unicode to the Windows clipboard without external packages."""
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = (ctypes.c_void_p,)
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.argtypes = ()
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.SetClipboardData.argtypes = (ctypes.c_uint, ctypes.c_void_p)
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = ()
    user32.CloseClipboard.restype = ctypes.c_int
    kernel32.GlobalAlloc.argtypes = (ctypes.c_uint, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalFree.argtypes = (ctypes.c_void_p,)
    kernel32.GlobalFree.restype = ctypes.c_void_p

    raw = (text + "\0").encode("utf-16-le")
    # GMEM_MOVEABLE is required by SetClipboardData.
    handle = kernel32.GlobalAlloc(0x0002, len(raw))
    if not handle:
        raise OSError("Could not allocate clipboard memory")
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        kernel32.GlobalFree(handle)
        raise OSError("Could not lock clipboard memory")
    ctypes.memmove(pointer, raw, len(raw))
    kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        kernel32.GlobalFree(handle)
        raise OSError("Could not open Windows clipboard")
    try:
        if not user32.EmptyClipboard():
            raise OSError("Could not clear Windows clipboard")
        if not user32.SetClipboardData(13, handle):  # CF_UNICODETEXT
            raise OSError("Could not set Windows clipboard text")
        handle = None  # Ownership transferred to the clipboard.
    finally:
        user32.CloseClipboard()
        if handle is not None:
            kernel32.GlobalFree(handle)


def _native_commands() -> list[list[str]]:
    """Prefer the active Linux display server; never invoke a shell."""
    if sys.platform == "darwin":
        return [["pbcopy"]] if shutil.which("pbcopy") else []
    if not sys.platform.startswith("linux"):
        return []

    commands: list[list[str]] = []
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        commands.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        if shutil.which("xclip"):
            commands.append(["xclip", "-selection", "clipboard"])
        if shutil.which("xsel"):
            commands.append(["xsel", "--clipboard", "--input"])
    return commands


def copy_text(
    text: str,
    *,
    terminal_copy: Optional[Callable[[str], None]] = None,
) -> str:
    """Copy text; return 'copied', 'terminal', or 'unavailable'.

    The OSC 52 terminal method is a *best-effort request*, not a verified
    clipboard write. Never claim its success as a native copy.
    """
    if not isinstance(text, str):
        raise TypeError("Clipboard content must be text")

    if sys.platform == "win32":
        try:
            _windows_copy(text)
            return "copied"
        except (OSError, RuntimeError):
            pass
    else:
        # Remote SSH clients cannot generally access the user's local desktop
        # clipboard; route copy requests to the connecting terminal instead.
        if not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
            for cmd in _native_commands():
                try:
                    subprocess.run(
                        cmd,
                        input=text.encode("utf-8"),
                        check=True,
                        timeout=3,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return "copied"
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue

    if terminal_copy is not None:
        try:
            terminal_copy(text)
            return "terminal"
        except Exception:
            pass
    return "unavailable"
