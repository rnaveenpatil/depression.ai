from pathlib import Path

from agent.llm.runtime import configure_runtime_provider, load_runtime_config
from agent.utils import env_manager as em
from agent.utils.env_manager import load_env_file, set_aws_credentials


class FakeRegistry:
    def __init__(self):
        self.providers = {}
        self._api_keys = {}
        self._current_provider = None
        self._current_model = None

    def install_provider(self, name, provider, api_key, model, metadata=None):
        self.providers[name] = provider
        self._api_keys[name] = api_key
        self._current_provider = name
        self._current_model = model


def test_runtime_provider_accepts_arbitrary_model_and_endpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    private_env = tmp_path / "agent-env"
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", private_env)
    registry = FakeRegistry()

    provider = configure_runtime_provider(
        registry, "https://llm.example/v1", "dummy-key", "vendor/custom-agent"
    )

    assert provider.base_url == "https://llm.example/v1"
    assert provider.api_key == "dummy-key"
    assert registry._current_provider == "custom"
    assert registry._current_model == "vendor/custom-agent"
    assert load_runtime_config()["model"] == "vendor/custom-agent"
    assert load_env_file(tmp_path / ".env")["DEPRESSION_API_KEY"] == "dummy-key"
    assert not private_env.exists()


def test_llm_credentials_use_one_write(tmp_path, monkeypatch):
    from agent.utils import credential_store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", tmp_path / "agent-env")
    writes = []
    original = credential_store.write_env_file

    def counting_write(*args, **kwargs):
        writes.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(credential_store, "write_env_file", counting_write)
    configure_runtime_provider(
        FakeRegistry(), "https://llm.example/v1", "dummy-key", "vendor/custom"
    )
    assert len(writes) == 1


def test_aws_credentials_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    private_env = tmp_path / "agent-env"
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", private_env)
    set_aws_credentials("AKIA_TEST", "SECRET_TEST", "ap-south-1")
    values = load_env_file(tmp_path / ".env")
    assert values["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
    assert values["AWS_SECRET_ACCESS_KEY"] == "SECRET_TEST"
    assert values["AWS_DEFAULT_REGION"] == "ap-south-1"
    assert not private_env.exists()


def test_all_credentials_share_one_project_file(tmp_path, monkeypatch):
    import json
    from agent.tui import mcp_config_ui as mcp_ui
    from agent.llm.runtime import save_runtime_config

    private_env = tmp_path / ".agent" / "env"
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", private_env)
    monkeypatch.chdir(tmp_path)

    save_runtime_config("https://llm.example/v1", "llm-private", "my-model")
    em.EnvManager.get().save_aws_credentials(
        access_key="AKIA_TEST", secret_key="aws-private", region="ap-south-1",
    )
    mcp_ui.save_mcp_server(
        {"name": "demo", "url": "https://mcp.example/mcp", "api_key": "mcp-private"},
        config_path=tmp_path / "config.json",
    )

    project_env = tmp_path / ".env"
    data = load_env_file(project_env)
    assert data["DEPRESSION_API_KEY"] == "llm-private"
    assert data["AWS_SECRET_ACCESS_KEY"] == "aws-private"
    assert data["DEPRESSION_MCP_DEMO_API_KEY"] == "mcp-private"
    assert project_env.stat().st_mode & 0o077 == 0
    assert not private_env.exists()
    config = json.loads((tmp_path / "config.json").read_text())
    assert "mcp-private" not in json.dumps(config)


def test_project_credentials_override_legacy_global_file(tmp_path, monkeypatch):
    project = tmp_path / ".env"
    project.write_text(
        "DEPRESSION_API_KEY=new-llm\n"
        "AWS_ACCESS_KEY_ID=NEW_ID\n"
        "AWS_SECRET_ACCESS_KEY=new-secret\n"
    )
    private_env = tmp_path / "env"
    em.write_env_file({
        "DEPRESSION_API_KEY": "old-llm",
        "AWS_ACCESS_KEY_ID": "OLD_ID",
        "AWS_SECRET_ACCESS_KEY": "old-secret",
    }, private_env, strict=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", private_env)
    assert em.get_env_var("DEPRESSION_API_KEY") == "new-llm"
    assert em.EnvManager.get().get_aws_credentials()["access_key"] == "NEW_ID"
    assert em.EnvManager.get().get_aws_credentials()["secret_key"] == "new-secret"
    assert "old-llm" in private_env.read_text()


def test_project_credentials_remain_readable(tmp_path, monkeypatch):
    project = tmp_path / ".env"
    project.write_text("DEPRESSION_API_KEY=project-llm\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", tmp_path / "not-yet-created")
    assert em.get_env_var("DEPRESSION_API_KEY") == "project-llm"


def test_legacy_global_credentials_remain_readable(tmp_path, monkeypatch):
    private_env = tmp_path / "env"
    private_env.write_text("DEPRESSION_API_KEY=legacy-llm\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(em, "GLOBAL_ENV_PATH", private_env)
    assert em.get_env_var("DEPRESSION_API_KEY") == "legacy-llm"
