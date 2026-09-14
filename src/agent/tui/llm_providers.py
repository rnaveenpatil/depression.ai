"""LLM catalog and persistent user connection settings for the TUI.

The TUI owns presentation; the runtime provider registry remains responsible
for actual completions. Keys are stored locally with restrictive permissions
and are never rendered back into the interface in clear text.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LLMModel:
    name: str
    display_name: str
    provider: str
    base_url: str
    context_window: int
    supports_tools: bool = True
    rating: int = 4

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1] if "/" in self.name else self.name

    @property
    def stars(self) -> str:
        return "★" * self.rating + "☆" * (5 - self.rating)


# Curated catalog requested for Depression.AI. Provider/model IDs are kept
# separate from display names so the UI can remain clean and searchable.
PROVIDER_MODELS: List[LLMModel] = [
    LLMModel("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B A55B", "nvidia", "https://integrate.api.nvidia.com/v1", 1_000_000, rating=5),
    LLMModel("deepseek/DeepSeek-V4-Pro", "DeepSeek V4 Pro", "deepseek", "https://api.deepseek.com", 1_000_000, rating=5),
    LLMModel("deepseek/DeepSeek-V4-Flash", "DeepSeek V4 Flash", "deepseek", "https://api.deepseek.com", 1_000_000, rating=5),
    LLMModel("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B A12B", "nvidia", "https://integrate.api.nvidia.com/v1", 1_000_000, rating=5),
    LLMModel("nvidia/nemotron-3.5-lightning-30b-a3b", "Nemotron 3.5 Lightning 30B A3B", "nvidia", "https://integrate.api.nvidia.com/v1", 1_000_000, rating=5),
    LLMModel("nvidia/nemotron-3-nano-30b-a3b", "Nemotron 3 Nano 30B A3B", "nvidia", "https://integrate.api.nvidia.com/v1", 1_000_000, rating=4),
    LLMModel("minimax/MiniMax-M2.5", "MiniMax M2.5", "minimax", "https://api.minimax.io/v1", 200_000, rating=5),
    LLMModel("mistral/mistral-medium-3.5", "Mistral Medium 3.5", "mistral", "https://api.mistral.ai/v1", 256_000, rating=5),
    LLMModel("mistral/devstral", "Devstral", "mistral", "https://api.mistral.ai/v1", 256_000, rating=5),
    LLMModel("anthropic/claude-sonnet", "Claude Sonnet", "anthropic", "https://api.anthropic.com", 200_000, rating=5),
    LLMModel("anthropic/claude-opus", "Claude Opus", "anthropic", "https://api.anthropic.com", 200_000, rating=5),
    LLMModel("google/gemini-flash", "Gemini Flash", "google", "https://generativelanguage.googleapis.com/v1beta/openai", 1_000_000, rating=5),
    LLMModel("google/gemini-pro", "Gemini Pro", "google", "https://generativelanguage.googleapis.com/v1beta/openai", 1_000_000, rating=5),
    LLMModel("openai/gpt-5.x", "GPT-5.x", "openai", "https://api.openai.com/v1", 400_000, rating=5),
    LLMModel("openai/gpt-oss-120b", "GPT-OSS 120B", "groq", "https://api.groq.com/openai/v1", 131_000, rating=5),
    LLMModel("openai/gpt-oss-20b", "GPT-OSS 20B", "groq", "https://api.groq.com/openai/v1", 131_000, rating=4),
    LLMModel("moonshot/kimi-k2", "Kimi K2.x", "moonshot", "https://api.moonshot.ai/v1", 200_000, rating=5),
    LLMModel("moonshot/kimi-k3", "Kimi K3", "moonshot", "https://api.moonshot.ai/v1", 200_000, rating=5),
    LLMModel("zai/glm-5.x", "GLM-5.x", "zai", "https://api.z.ai/api/paas/v4", 200_000, rating=5),
    LLMModel("qwen/qwen-coder", "Qwen Coder", "qwen", "", 200_000, rating=5),
    LLMModel("meta/llama-4", "Llama 4 / latest Llama", "meta", "", 200_000, rating=4),
    LLMModel("alibaba/qwen-3.x", "Qwen 3.x", "alibaba", "", 200_000, rating=4),
    LLMModel("openrouter/coding-agent", "OpenRouter Coding / Agent Models", "openrouter", "https://openrouter.ai/api/v1", 200_000, rating=5),
]

PROVIDER_INFO: Dict[str, Dict] = {
    "nvidia": {"name": "NVIDIA", "icon": "◆", "api_key_env": "NVIDIA_API_KEY"},
    "deepseek": {"name": "DeepSeek", "icon": "◇", "api_key_env": "DEEPSEEK_API_KEY"},
    "minimax": {"name": "MiniMax", "icon": "✦", "api_key_env": "MINIMAX_API_KEY"},
    "mistral": {"name": "Mistral", "icon": "M", "api_key_env": "MISTRAL_API_KEY"},
    "anthropic": {"name": "Anthropic", "icon": "A", "api_key_env": "ANTHROPIC_API_KEY"},
    "google": {"name": "Google", "icon": "G", "api_key_env": "GOOGLE_API_KEY"},
    "openai": {"name": "OpenAI", "icon": "O", "api_key_env": "OPENAI_API_KEY"},
    "groq": {"name": "Groq", "icon": "⚡", "api_key_env": "GROQ_API_KEY"},
    "moonshot": {"name": "Moonshot", "icon": "☾", "api_key_env": "MOONSHOT_API_KEY"},
    "zai": {"name": "Z.AI", "icon": "Z", "api_key_env": "ZAI_API_KEY"},
    "qwen": {"name": "Qwen", "icon": "Q", "api_key_env": "DASHSCOPE_API_KEY"},
    "meta": {"name": "Meta", "icon": "M", "api_key_env": "META_API_KEY"},
    "alibaba": {"name": "Alibaba", "icon": "A", "api_key_env": "DASHSCOPE_API_KEY"},
    "openrouter": {"name": "OpenRouter", "icon": "↗", "api_key_env": "OPENROUTER_API_KEY"},
}


def _config_file() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    directory = base / "depression"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "llm.json"


class LLMConfig:
    """Persistent selected model, endpoint and API-key configuration."""
    def __init__(self) -> None:
        self._selected_model: Optional[str] = None
        self._api_keys: Dict[str, str] = {}
        self._base_urls: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with _config_file().open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._selected_model = data.get("selected_model")
            self._api_keys = dict(data.get("api_keys", {}))
            self._base_urls = dict(data.get("base_urls", {}))
        except (OSError, ValueError, TypeError):
            return

    def save(self) -> None:
        path = _config_file()
        tmp = path.with_suffix(".tmp")
        data = {"selected_model": self._selected_model, "api_keys": self._api_keys, "base_urls": self._base_urls}
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except OSError:
            try: tmp.unlink(missing_ok=True)
            except OSError: pass

    @property
    def selected_model(self) -> Optional[str]: return self._selected_model
    @selected_model.setter
    def selected_model(self, value: str) -> None:
        self._selected_model = value; self.save()

    def get_api_key(self, provider: str) -> str:
        value = self._api_keys.get(provider)
        if value: return value
        env = PROVIDER_INFO.get(provider, {}).get("api_key_env", "")
        return os.environ.get(env, "") if env else ""

    def set_api_key(self, provider: str, key: str) -> None:
        if key: self._api_keys[provider] = key
        else: self._api_keys.pop(provider, None)
        self.save()

    def get_base_url(self, provider: str) -> str:
        if provider in self._base_urls: return self._base_urls[provider]
        for model in PROVIDER_MODELS:
            if model.provider == provider and model.base_url: return model.base_url
        return ""

    def set_base_url(self, provider: str, url: str) -> None:
        if url: self._base_urls[provider] = url.rstrip("/")
        else: self._base_urls.pop(provider, None)
        self.save()

    def get_selected_model_info(self) -> Optional[LLMModel]:
        return next((m for m in PROVIDER_MODELS if m.name == self._selected_model), None)

    def build_llm_config(self, model: LLMModel) -> Dict:
        return {"llm": {"provider": model.provider, "model": model.name, "api_key": self.get_api_key(model.provider), "base_url": self.get_base_url(model.provider), "params": {"temperature": 0.1}}}

    def get_provider_status(self) -> List[Dict]:
        result = []
        for provider, info in PROVIDER_INFO.items():
            models = [m for m in PROVIDER_MODELS if m.provider == provider]
            if models:
                result.append({"id": provider, "name": info["name"], "icon": info["icon"], "has_key": bool(self.get_api_key(provider)), "models": models})
        return result


_llm_config: Optional[LLMConfig] = None

def get_llm_config() -> LLMConfig:
    global _llm_config
    if _llm_config is None: _llm_config = LLMConfig()
    return _llm_config
