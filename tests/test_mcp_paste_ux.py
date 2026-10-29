"""Paste-first MCP/LLM UI regression tests."""
import json
import pytest
from textual.widgets import Button, Input, Select, TextArea
from agent.tui.app import DepressionApp
from agent.tui.mcp_config_ui import save_mcp_server

def test_private_token_and_existing_settings(tmp_path):
    config = tmp_path / "config.json"
    env = tmp_path / "env"
    config.write_text(json.dumps({"ui": {"theme": "matrix"}, "mcp": {"servers": [
        {"name": "old", "transport": "stdio", "command": "python", "args": ["-V"]}
    ]}}))
    save_mcp_server({"name": "demo", "url": "https://example.com/mcp", "transport": "streamable-http", "api_key": "SECRET_TOKEN"},
                    config_path=config, env_path=env)
    obj = json.loads(config.read_text())
    assert obj["ui"]["theme"] == "matrix"
    assert len(obj["mcp"]["servers"]) == 2
    assert "SECRET_TOKEN" not in config.read_text()
    assert "SECRET_TOKEN" in env.read_text()
    assert env.stat().st_mode & 0o077 == 0

def test_pasted_json_stdio(tmp_path):
    config, env = tmp_path / "config.json", tmp_path / "env"
    payload = {"mcpServers": {"filesystem": {
        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "env": {"EXAMPLE_TOKEN": "private"}
    }}}
    saved = save_mcp_server({"json": json.dumps(payload)}, config_path=config, env_path=env)
    assert saved["config"]["servers"][0]["transport"] == "stdio"
    assert "private" not in config.read_text()
    assert "private" in env.read_text()

@pytest.mark.parametrize("url", ["http://example.com/mcp", "https://user:pass@example.com/mcp", "file:///tmp/mcp"])
def test_invalid_remote_url_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        save_mcp_server({"name": "test", "url": url}, config_path=tmp_path / "config.json", env_path=tmp_path / "env")

@pytest.mark.asyncio
async def test_manual_llm_fields_and_mcp_form():
    app = DepressionApp(coordinator=None, skip_onboarding=True)
    async with app.run_test() as pilot:
        assert app.query_one("#base-url", Input)
        assert app.query_one("#api-key", Input)
        assert app.query_one("#model-id", Input)
        assert not app.query("#model-select")
        app._slash("/mcp")
        await pilot.pause()
        assert app.query_one("#mcp-name", Input)
        assert app.query_one("#mcp-url", Input)
        assert app.query_one("#mcp-token", Input).password
        assert app.query_one("#mcp-transport", Select)
        assert app.query_one("#mcp-paste", TextArea)
        assert app.query_one("#mcp-save", Button)
