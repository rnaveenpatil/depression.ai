from pathlib import Path

from agent.llm.runtime import configure_runtime_provider, load_runtime_config
from agent.utils.env_manager import load_env_file, set_aws_credentials


class FakeRegistry:
    def __init__(self):
        self.providers = {}
        self._api_keys = {}
        self._current_provider = None
        self._current_model = None


def test_runtime_provider_accepts_arbitrary_model_and_endpoint(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    registry = FakeRegistry()

    provider = configure_runtime_provider(
        registry, "https://llm.example/v1", "secret-test-key", "vendor/custom-agent"
    )

    assert provider.base_url == "https://llm.example/v1"
    assert provider.api_key == "secret-test-key"
    assert registry._current_provider == "custom"
    assert registry._current_model == "vendor/custom-agent"
    assert load_runtime_config()["model"] == "vendor/custom-agent"
    assert load_env_file(env)["DEPRESSION_API_KEY"] == "secret-test-key"


def test_aws_credentials_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_aws_credentials("AKIA_TEST", "SECRET_TEST", "ap-south-1")
    values = load_env_file(Path(tmp_path) / ".env")
    assert values["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
    assert values["AWS_SECRET_ACCESS_KEY"] == "SECRET_TEST"
    assert values["AWS_DEFAULT_REGION"] == "ap-south-1"
