"""Regression tests for safe command construction in tool fallbacks."""
import shlex
import subprocess
from pathlib import Path

from agent.agent.loop import (
    _bash_fallback_edit,
    _bash_fallback_glob,
    _bash_fallback_grep,
    _bash_fallback_list,
    _bash_fallback_read,
    _bash_fallback_write,
)


def _run_command(command: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, shell=True, check=False, capture_output=True,
        text=True, timeout=10, cwd=cwd,
    )


def test_write_fallback_preserves_content_including_heredoc_marker(tmp_path: Path) -> None:
    target = tmp_path / "written.txt"
    content = "first line\nEOF\nsecond line\n$(touch should_not_exist)\n"
    result = _run_command(
        _bash_fallback_write({"path": str(target), "content": content})["command"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == content
    assert not (tmp_path / "should_not_exist").exists()


def test_write_fallback_does_not_interpret_shell_metacharacters_in_path(tmp_path: Path) -> None:
    marker = tmp_path / "INJECTED"
    target_name = "odd'; touch INJECTED; #.txt"
    target = tmp_path / target_name
    result = _run_command(
        _bash_fallback_write({"path": target_name, "content": "safe content"})["command"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "safe content"
    assert not marker.exists()


def test_edit_fallback_replaces_exact_text_and_preserves_special_characters(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("before OLD after\n", encoding="utf-8")
    replacement = "quote ' ; $(touch should_not_exist) & / \\\nEOF"
    result = _run_command(
        _bash_fallback_edit({
            "path": str(target),
            "oldString": "OLD",
            "newString": replacement,
        })["command"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "before " + replacement + " after\n"
    assert not (tmp_path / "should_not_exist").exists()


def test_read_list_glob_and_grep_quote_untrusted_arguments(tmp_path: Path) -> None:
    marker = tmp_path / "INJECTED"
    strange = "name'; touch INJECTED; #"
    commands = [
        _bash_fallback_read({"path": strange})["command"],
        _bash_fallback_list({"path": strange})["command"],
        _bash_fallback_glob({"path": strange, "pattern": "*'; touch INJECTED; #"})["command"],
        _bash_fallback_grep({"path": strange, "pattern": "x'; touch INJECTED; #"})["command"],
    ]
    for command in commands:
        assert command
        shlex.split(command)
        _run_command(command, tmp_path)
    assert not marker.exists()
