"""
LLM Provider Registry & Persistent Config Storage

Manages:
- Pre-configured LLM providers with models (from user's table)
- API key storage (obfuscated on disk)
- Base URL management
- Provider selection and switching
- Persistent configuration at ~/.config/depression/llm.json
"""

from __future__ import annotations

import json
import os
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ======================================================================
# PROVIDER DATA (from user's table)
# ======================================================================

@dataclass(frozen=True)
class LLMModel:
    """A single LLM model configuration."""
    name: str                    # Internal name (provider/model)
    display_name: str           # Human-readable name
    provider: str               # Provider ID (deepseek, together, groq, nvidia, mistral, google)
    base_url: str               # OpenAI-compatible base URL
    context_window: int         # Context window in tokens
    supports_tools: bool = True # Function calling support
    rating: int = 4             # 1-5 stars

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1] if "/" in self.name else self.name

    @property
    def stars(self) -> str:
        return "★" * self.rating + "☆" * (5 - self.rating)


# Exact models from user's table
PROVIDER_MODELS: List[LLMModel] = [
    LLMModel(
        name="deepseek-ai/DeepSeek-V4.1-Flash",
        display_name="DeepSeek V4.1 Flash",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        context_window=1_000_000,
        supports_tools=True,
        rating=5,
    ),
    LLMModel(
        name="deepseek-ai/DeepSeek-V4-Flash-0731",
        display_name="DeepSeek V4 Flash 0731",
        provider="together",
        base_url="https://api.together.xyz/v1",
        context_window=1_000_000,
        supports_tools=True,
        rating=5,
    ),
    LLMModel(
        name="openai/gpt-oss-120b",
        display_name="GPT-OSS 120B",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        context_window=131_000,
        supports_tools=True,
        rating=5,
    ),
    LLMModel(
        name="openai/gpt-oss-20b",
        display_name="GPT-OSS 20B",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        context_window=131_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="nvidia/nemotron-3-ultra-550b",
        display_name="Nemotron 3 Ultra 550B",
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        context_window=1_000_000,
        supports_tools=True,
        rating=5,
    ),
    LLMModel(
        name="nvidia/nemotron-3-super-120b",
        display_name="Nemotron 3 Super 120B",
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        context_window=1_000_000,
        supports_tools=True,
        rating=5,
    ),
    LLMModel(
        name="nvidia/nemotron-3-nano-30b",
        display_name="Nemotron 3 Nano 30B",
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        context_window=1_000_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="mistralai/Mistral-Small-4",
        display_name="Mistral Small 4",
        provider="mistral",
        base_url="https://api.mistral.ai/v1",
        context_window=256_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="google/gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        context_window=1_000_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="meta-llama/llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        context_window=131_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="qwen/qwen3-27b",
        display_name="Qwen 3.6 27B",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        context_window=131_000,
        supports_tools=True,
        rating=4,
    ),
    LLMModel(
        name="minimax/minimax-m2.7",
        display_name="MiniMax M2.7",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        context_window=131_000,
        supports_tools=True,
        rating=4,
    ),
]


# Provider-specific info
PROVIDER_INFO: Dict[str, Dict] = {
    "deepseek": {
        "name": "DeepSeek",
        "color": "#00c8ff",
        "icon": "◆",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key_url": "https://platform.deepseek.com/api_keys",
    },
    "together": {
        "name": "Together AI",
        "color": "#b464ff",
        "icon": "⟡",
        "api_key_env": "TOGETHER_API_KEY",
        "api_key_url": "https://api.together.xyz/settings/api-keys",
    },
    "groq": {
        "name": "Groq",
        "color": "#f55036",
        "icon": "⚡",
        "api_key_env": "GROQ_API_KEY",
        "api_key_url": "https://console.groq.com/keys",
    },
    "nvidia": {
        "name": "NVIDIA",
        "color": "#76b900",
        "icon": "◈",
        "api_key_env": "NVIDIA_API_KEY",
        "api_key_url": "https://build.nvidia.com/settings/api-keys",
    },
    "mistral": {
        "name": "Mistral AI",
        "color": "#ff7000",
        "icon": "◧",
        "api_key_env": "MISTRAL_API_KEY",
        "api_key_url": "https://console.mistral.ai/api-keys/",
    },
    "google": {
        "name": "Google AI",
        "color": "#4285f4",
        "icon": "◎",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://aistudio.google.com/apikey",
    },
}


# ======================================================================
# CONFIG STORAGE
# ======================================================================

def _get_config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base / "depression"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_config_file() -> Path:
    return _get_config_dir() / "llm.json"


def _obfuscate(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def _deobfuscate(text: str) -> str:
    try:
        return base64.urlsafe_b64decode(text.encode()).decode()
    except Exception:
        return text


class LLMConfig:
    """Persistent LLM configuration manager."""

    def __init__(self):
        self._selected_model: Optional[str] = None
        self._api_keys: Dict[str, str] = {}
        self._custom_base_urls: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        config_file = _get_config_file()
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                self._selected_model = data.get("selected_model")
                encrypted_keys = data.get("api_keys", {})
                self._api_keys = {k: _deobfuscate(v) for k, v in encrypted_keys.items()}
                self._custom_base_urls = data.get("base_urls", {})
            except Exception:
                pass

    def save(self) -> None:
        config_file = _get_config_file()
        data = {
            "selected_model": self._selected_model,
            "api_keys": {k: _obfuscate(v) for k, v in self._api_keys.items()},
            "base_urls": self._custom_base_urls,
        }
        try:
            with open(config_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save LLM config: {e}")

    @property
    def selected_model(self) -> Optional[str]:
        return self._selected_model

    @selected_model.setter
    def selected_model(self, model_name: str) -> None:
        self._selected_model = model_name
        self.save()

    def get_api_key(self, provider: str) -> str:
        if provider in self._api_keys and self._api_keys[provider]:
            return self._api_keys[provider]
        info = PROVIDER_INFO.get(provider, {})
        env_var = info.get("api_key_env", "")
        if env_var:
            return os.environ.get(env_var, "")
        return ""

    def set_api_key(self, provider: str, key: str) -> None:
        if key:
            self._api_keys[provider] = key
        else:
            self._api_keys.pop(provider, None)
        self.save()

    def has_api_key(self, provider: str) -> bool:
        return bool(self.get_api_key(provider))

    def get_base_url(self, provider: str) -> str:
        if provider in self._custom_base_urls:
            return self._custom_base_urls[provider]
        for model in PROVIDER_MODELS:
            if model.provider == provider:
                return model.base_url
        return ""

    def set_base_url(self, provider: str, url: str) -> None:
        if url:
            self._custom_base_urls[provider] = url
        else:
            self._custom_base_urls.pop(provider, None)
        self.save()

    def get_selected_model_info(self) -> Optional[LLMModel]:
        if not self._selected_model:
            return None
        for model in PROVIDER_MODELS:
            if model.name == self._selected_model:
                return model
        return None

    def get_provider_status(self) -> List[Dict]:
        providers = {}
        for model in PROVIDER_MODELS:
            p = model.provider
            if p not in providers:
                info = PROVIDER_INFO.get(p, {})
                providers[p] = {
                    "id": p,
                    "name": info.get("name", p),
                    "color": info.get("color", "#ffffff"),
                    "icon": info.get("icon", "●"),
                    "has_key": self.has_api_key(p),
                    "models": [],
                }
            providers[p]["models"].append(model)
        return list(providers.values())

    def build_llm_config(self, model: LLMModel) -> Dict:
        api_key = self.get_api_key(model.provider)
        base_url = self.get_base_url(model.provider)
        return {
            "llm": {
                "provider": model.provider,
                "model": model.name,
                "api_key": api_key,
                "base_url": base_url,
                "params": {"temperature": 0.7},
            }
        }


_llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config