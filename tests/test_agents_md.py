"""Project instructions (AGENTS.md / CLAUDE.md) loading."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.context.instructions import (
    clear_instruction_cache,
    load_project_instructions,
)


def _make_loop(project_dir: Path, configured=None) -> AgentLoop:
    config = {}
    if configured is not None:
        config["context"] = {"instructions": configured}
    agent = SimpleNamespace(
        workspace=SimpleNamespace(project_dir=str(project_dir)),
        config=config,
        permission_manager=SimpleNamespace(enabled=True, auto_approve=False),
    )
    return AgentLoop(
        agent=agent,
        llm=SimpleNamespace(),
        tool_registry=SimpleNamespace(),
        planner=None,
        config={"enable_planning": False},
    )


def test_loads_agents_md_with_fingerprint(tmp_path):
    clear_instruction_cache()
    (tmp_path / "AGENTS.md").write_text(
        "Always run tests with make test.\n", encoding="utf-8"
    )

    text, meta = load_project_instructions(str(tmp_path))

    assert "Always run tests with make test." in text
    assert meta["paths"] == [str(tmp_path / "AGENTS.md")]
    assert meta["fingerprint"]
    assert meta["truncated"] is False


def test_parent_agents_md_is_found_from_subdirectory(tmp_path):
    clear_instruction_cache()
    (tmp_path / "AGENTS.md").write_text("Root rule", encoding="utf-8")
    child = tmp_path / "src"
    child.mkdir()

    text, _meta = load_project_instructions(str(child))

    assert "Root rule" in text


def test_instruction_cap_truncates_large_file(tmp_path):
    clear_instruction_cache()
    (tmp_path / "AGENTS.md").write_text("x" * 500, encoding="utf-8")

    text, meta = load_project_instructions(str(tmp_path), max_bytes=100)

    assert text.endswith("(truncated)")
    assert text.count("x") == 100


@pytest.mark.asyncio
async def test_system_prompt_includes_project_instructions(tmp_path):
    clear_instruction_cache()
    (tmp_path / "AGENTS.md").write_text(
        "This project uses tabs, not spaces.", encoding="utf-8"
    )
    loop = _make_loop(tmp_path)

    prompt = await loop._get_system_prompt()

    assert "## Project Instructions" in prompt
    assert "This project uses tabs, not spaces." in prompt


@pytest.mark.asyncio
async def test_configured_instruction_path_is_loaded(tmp_path):
    clear_instruction_cache()
    (tmp_path / "custom-rules.md").write_text(
        "Use pytest -q.", encoding="utf-8"
    )
    loop = _make_loop(tmp_path, configured=["custom-rules.md"])

    prompt = await loop._get_system_prompt()

    assert "Use pytest -q." in prompt
