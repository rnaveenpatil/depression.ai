"""Environment file management for Depression.AI.

Handles reading/writing .env files for persistent credential storage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List


ENV_FILE_NAME = ".env"
DEFAULT_ENV_PATH = Path.cwd() / ENV_FILE_NAME


# Provider to env var mappings
PROVIDER_API_KEY_ENVS = {
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zai": "ZAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "meta": "META_API_KEY",
    "alibaba": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_BASE_URL_ENVS = {
    "nvidia": "NVIDIA_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "minimax": "MINIMAX_BASE_URL",
    "mistral": "MISTRAL_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "google": "GOOGLE_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "groq": "GROQ_BASE_URL",
    "moonshot": "MOONSHOT_BASE_URL",
    "zai": "ZAI_BASE_URL",
    "qwen": "QWEN_BASE_URL",
    "meta": "META_BASE_URL",
    "alibaba": "ALIBABA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}

AWS_ENVS = {
    "access_key": "AWS_ACCESS_KEY_ID",
    "secret_key": "AWS_SECRET_ACCESS_KEY",
    "region": "AWS_DEFAULT_REGION",
}

SELECTED_MODEL_ENV = "DEPRESSION_SELECTED_MODEL"


def find_env_file(start_path: Optional[Path] = None) -> Path:
    """Find the nearest .env file walking up from start_path."""
    path = start_path or Path.cwd()
    for parent in [path, *path.parents]:
        env_file = parent / ENV_FILE_NAME
        if env_file.exists():
            return env_file
    return DEFAULT_ENV_PATH


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    path = env_path or find_env_file()
    result: Dict[str, str] = {}
    
    if not path.exists():
        return result
    
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    except Exception:
        pass
    
    return result


def write_env_file(env_vars: Dict[str, str], env_path: Optional[Path] = None) -> None:
    """Write environment variables to a .env file."""
    path = env_path or DEFAULT_ENV_PATH
    
    # Load existing to preserve comments and other vars
    existing = load_env_file(path)
    existing.update(env_vars)
    
    # Write back
    lines = []
    for key, value in sorted(existing.items()):
        lines.append(f"{key}={value}")
    
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Secure the file
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except Exception:
        pass


def set_env_var(key: str, value: str, env_path: Optional[Path] = None) -> None:
    """Set a single environment variable in .env file."""
    env_vars = load_env_file(env_path)
    if value:
        env_vars[key] = value
    else:
        env_vars.pop(key, None)
    write_env_file(env_vars, env_path)


def get_env_var(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Get a single environment variable from .env file."""
    env_vars = load_env_file(env_path)
    return env_vars.get(key)


def set_provider_credentials(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    env_path: Optional[Path] = None
) -> None:
    """Set API key and/or base URL for a provider in .env."""
    env_vars = load_env_file(env_path)
    
    if api_key is not None:
        key_env = PROVIDER_API_KEY_ENVS.get(provider.lower())
        if key_env:
            if api_key:
                env_vars[key_env] = api_key
            else:
                env_vars.pop(key_env, None)
    
    if base_url is not None:
        url_env = PROVIDER_BASE_URL_ENVS.get(provider.lower())
        if url_env:
            if base_url:
                env_vars[url_env] = base_url.rstrip("/")
            else:
                env_vars.pop(url_env, None)
    
    write_env_file(env_vars, env_path)


def get_provider_credentials(provider: str, env_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Get API key and base URL for a provider from .env."""
    env_vars = load_env_file(env_path)
    return {
        "api_key": env_vars.get(PROVIDER_API_KEY_ENVS.get(provider.lower(), "")),
        "base_url": env_vars.get(PROVIDER_BASE_URL_ENVS.get(provider.lower(), "")),
    }


def set_selected_model(model: str, env_path: Optional[Path] = None) -> None:
    """Set the selected model in .env."""
    set_env_var(SELECTED_MODEL_ENV, model, env_path)


def get_selected_model(env_path: Optional[Path] = None) -> Optional[str]:
    """Get the selected model from .env."""
    return get_env_var(SELECTED_MODEL_ENV, env_path)


def set_aws_credentials(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
    env_path: Optional[Path] = None
) -> None:
    """Set AWS credentials in .env."""
    env_vars = load_env_file(env_path)
    
    if access_key is not None:
        if access_key:
            env_vars[AWS_ENVS["access_key"]] = access_key
        else:
            env_vars.pop(AWS_ENVS["access_key"], None)
    
    if secret_key is not None:
        if secret_key:
            env_vars[AWS_ENVS["secret_key"]] = secret_key
        else:
            env_vars.pop(AWS_ENVS["secret_key"], None)
    
    if region is not None:
        if region:
            env_vars[AWS_ENVS["region"]] = region
        else:
            env_vars.pop(AWS_ENVS["region"], None)
    
    write_env_file(env_vars, env_path)


def get_aws_credentials(env_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Get AWS credentials from .env."""
    env_vars = load_env_file(env_path)
    return {
        "access_key": env_vars.get(AWS_ENVS["access_key"]),
        "secret_key": env_vars.get(AWS_ENVS["secret_key"]),
        "region": env_vars.get(AWS_ENVS["region"], "us-east-1"),
    }


def load_into_os_environ(env_path: Optional[Path] = None) -> None:
    """Load .env variables into os.environ (for subprocesses and late-binding)."""
    env_vars = load_env_file(env_path)
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


# Auto-load on import
load_into_os_environ()