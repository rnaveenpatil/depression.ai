"""Real-PTY integration test for the permission modal.

Unlike headless Textual tests, this boots the real application on a pseudo
terminal and sends the allow key through the actual tty input path.
"""
from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path


REPO = str(Path(__file__).resolve().parents[1])

DRIVER = textwrap.dedent(f"""
    import asyncio
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, {REPO!r} + "/src")

    from agent.tui.app import DepressionApp, PermissionModal
    from agent.permissions.manager import PermissionVerdict, RiskLevel


    class PTYApp(DepressionApp):
        def on_mount(self):
            super().on_mount()
            request = SimpleNamespace(
                tool="bash",
                action="execute",
                params={{"command": "printf pty-panel-ok"}},
            )
            verdict = PermissionVerdict.ask(
                "destructive action detected",
                risk=RiskLevel.HIGH,
            )
            self.set_timer(0.2, lambda: self.push_screen(PermissionModal(request, verdict)))


    def main():
        print("__PTY_BOOT__", flush=True)
        PTYApp().run()


    main()
""")


def _spawn():
    master, slave = pty.openpty()
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", DRIVER],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
    )
    os.close(slave)
    return proc, master


def _read(master, buf):
    ready, _, _ = select.select([master], [], [], 0.2)
    if not ready:
        return buf, True
    try:
        return buf + os.read(master, 65536), True
    except OSError:
        return buf, False


def test_permission_modal_responsive_over_real_pty():
    proc, master = _spawn()
    buf = b""
    t0 = time.time()
    boot_seen = modal_seen = allowed = False
    try:
        while time.time() - t0 < 20:
            buf, alive = _read(master, buf)
            text = buf.decode(errors="replace")
            if "__PTY_BOOT__" in text:
                boot_seen = True
            if "PERMISSION REQUEST" in text:
                modal_seen = True
                if not allowed:
                    os.write(master, b"a")
                    allowed = True
            if allowed:
                break
            if not alive or proc.poll() is not None:
                break
    finally:
        try:
            os.write(master, b"\x11")
        except OSError:
            pass
        for _ in range(30):
            if proc.poll() is not None:
                break
            buf, _ = _read(master, buf)
            time.sleep(0.05)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)

    assert boot_seen, "app bootstrap marker was not emitted"
    assert proc.returncode not in (1, 2), (
        "app crashed under the pty: " + repr(buf[-1000:])
    )
    assert modal_seen, "permission modal never rendered: " + repr(buf[-1000:])
    assert allowed, "failed to send allow key over the real tty"
