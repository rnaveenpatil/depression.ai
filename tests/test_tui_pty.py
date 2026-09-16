"""Real-PTY integration test for the permission modal.

Headless `pilot` tests inject synthetic key events, which bypass the real
terminal input layer (pty -> Textual driver -> app). This test boots the actual
app on a pseudo-terminal, types a prompt, waits for the permission modal to
render, presses 'a' (allow) through the real tty, and asserts the underlying
bash tool actually ran. Guards against "permission panel not responsive"
regressions.
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

import pytest

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])

DRIVER = textwrap.dedent(f"""\
    import asyncio
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, "{REPO}/src")
    sys.path.insert(0, "{REPO}")
    sys.path.insert(0, "{REPO}/tests")

    from agent.agent.dual_agent import AgentCoordinator, AgentRole
    from agent.permissions.manager import PermissionManager
    from agent.tui.app import DepressionApp
    from tests.test_agents_roles import _make_agent, _make_workspace

    D = Path(tempfile.mkdtemp(prefix="ptyinv_"))
    subprocess.run(["git", "init", "-q"], cwd=D, check=True)

    async def build():
        ws = _make_workspace(D)
        await ws.initialize()
        script = [
            {{"label": "pty-inv", "content": "running", "tool_calls": [
                {{"name": "bash",
                  "arguments": {{"command": "printf pty-panel-ok > out.txt"}}}}]}},
            {{"label": "pty-final", "content": "Done.", "tool_calls": []}},
        ]
        plan, _ = await _make_agent(
            ws, AgentRole.PLAN,
            [{{"label": "p", "content": "plan", "tool_calls": []}}],
            with_planner=False)
        bld, _ = await _make_agent(ws, AgentRole.BUILD, script, with_planner=False)
        bld.permission_manager = PermissionManager(
            {{"mode": "manual", "*": "ask", "default": "allow"}}, ui=None,
            input_handler=None)
        coord = AgentCoordinator(plan_agent=plan, build_agent=bld, config={{}})
        app = DepressionApp(coordinator=coord)
        app._wire_permission_confirmation()
        return app

    def main():
        print("__D__" + str(D), flush=True)
        app = asyncio.run(build())
        app.run()

    main()
""")


def _spawn():
    master, slave = pty.openpty()
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", DRIVER],
        stdin=slave, stdout=slave, stderr=slave, env=env,
    )
    os.close(slave)
    return proc, master


def _read(master, buf):
    r, _, _ = select.select([master], [], [], 0.2)
    if r:
        try:
            return buf + os.read(master, 65536), True
        except OSError:
            return buf, False
    return buf, True


def test_permission_modal_responsive_over_real_pty(tmp_path):
    proc, master = _spawn()
    buf = b""
    t0 = time.time()
    workdir = None
    booted = submitted = modal_seen = allowed = done = False
    try:
        while time.time() - t0 < 90 and not done:
            buf, alive = _read(master, buf)
            if not alive:
                break
            text = buf.decode(errors="replace")
            m = re.search(r"__D__(/[^\s]*)", text)
            if m is not None:
                workdir = m.group(1)
            if workdir and not booted:
                time.sleep(2.5)  # let Textual boot and focus the prompt
                booted = True
                os.write(master, b"make flag\r")
                submitted = True
            if submitted and not modal_seen and "PERMISSION REQUEST" in text:
                modal_seen = True
            if modal_seen and not allowed:
                os.write(master, b"a")  # 'a' = allow
                allowed = True
            if workdir and os.path.exists(os.path.join(workdir, "out.txt")):
                done = True
            if proc.poll() is not None:
                break
    finally:
        try:
            os.write(master, b"\x11")  # ctrl+q quit
        except OSError:
            pass
        for _ in range(40):
            if proc.poll() is not None:
                break
            buf, alive = _read(master, buf)
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)

    assert booted, "app did not boot under the pty"
    assert modal_seen, "permission modal never rendered (" + repr(buf[:400]) + ")"
    assert allowed, "failed to send allow over the real tty"
    assert done, "tool did not run after allow (" + repr(buf[:400]) + ")"
    assert workdir is not None
    assert open(os.path.join(workdir, "out.txt")).read() == "pty-panel-ok"