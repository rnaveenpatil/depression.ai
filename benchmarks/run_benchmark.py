#!/usr/bin/env python3
"""Small, dependency-free benchmark harness for coding agents.

Each task is materialised in a throwaway workspace, the agent is invoked
non-interactively, and a verification command decides pass/fail. This makes the
same suite runnable against this project, Claude Code, Codex, or any CLI that
accepts a prompt.

Examples:
    python benchmarks/run_benchmark.py --agent depression
    python benchmarks/run_benchmark.py --agent claude --tasks benchmarks/tasks.json
    python benchmarks/run_benchmark.py --agent-cmd 'my-agent --yes {prompt}'
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_TASKS = Path(__file__).with_name("tasks.json")

AGENT_PRESETS: Dict[str, str] = {
    "depression": (
        "depression --non-interactive --yolo --quiet "
        "--max-turns 30 {prompt}"
    ),
    "claude": "claude -p {prompt} --dangerously-skip-permissions",
    "codex": "codex exec --full-auto {prompt}",
}


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("tasks file must contain a JSON list")
    return data


def run(command: str, cwd: Path, timeout: int) -> tuple[int, str, float]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return 124, output, time.monotonic() - started


def seed(workspace: Path, task: Dict[str, Any]) -> None:
    for name, content in (task.get("files") or {}).items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_task(task: Dict[str, Any], agent_cmd: str) -> Dict[str, Any]:
    timeout = int(task.get("max_seconds", 240))
    with tempfile.TemporaryDirectory(prefix=f"agentbench-{task['id']}-") as tmp:
        workspace = Path(tmp)
        seed(workspace, task)
        prompt = str(task["prompt"])
        command = agent_cmd.format(prompt=shlex.quote(prompt))
        code, output, elapsed = run(command, workspace, timeout)
        verify_code, verify_output, verify_elapsed = run(
            str(task["verify"]), workspace, min(timeout, 120)
        )
        return {
            "id": task["id"],
            "passed": verify_code == 0,
            "task_exit": code,
            "duration_s": round(elapsed, 2),
            "verify_s": round(verify_elapsed, 2),
            "output_tail": output[-2000:],
            "verify_tail": verify_output[-1000:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENT_PRESETS))
    parser.add_argument(
        "--agent-cmd",
        help="Command template; use {prompt} where the request goes.",
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run only these task ids.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    if args.agent_cmd:
        agent_cmd = args.agent_cmd
    elif args.agent:
        agent_cmd = AGENT_PRESETS[args.agent]
    else:
        parser.error("pass --agent or --agent-cmd")

    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = set(args.only)
        tasks = [t for t in tasks if t["id"] in wanted]
    if not tasks:
        raise SystemExit("no tasks selected")

    results = [run_task(task, agent_cmd) for task in tasks]
    passed = sum(1 for r in results if r["passed"])
    summary = {
        "agent_command": agent_cmd,
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3),
        "total_duration_s": round(sum(r["duration_s"] for r in results), 2),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(
                f"{mark}  {r['id']:<24} agent={r['duration_s']:>6.1f}s "
                f"verify={r['verify_s']:>5.1f}s exit={r['task_exit']}"
            )
            if not r["passed"] and r["verify_tail"].strip():
                print("      " + r["verify_tail"].strip().replace("\n", "\n      "))
        print(
            f"\n{passed}/{len(results)} passed "
            f"({summary['pass_rate'] * 100:.0f}%) in "
            f"{summary['total_duration_s']:.1f}s"
        )

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
