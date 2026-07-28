"""
Provider preset system for ViMax chat model configuration.

Supports auto-detection and resolution of LLM provider settings,
allowing users to specify a provider name (e.g., ``minimax``) instead
of manually configuring base_url and model details.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "env_key": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M3",
        "models": [
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
        ],
        "temperature_range": (0.0, 1.0),
    },
    "atlascloud": {
        "base_url": "https://api.atlascloud.ai/v1",
        "env_key": "ATLASCLOUD_API_KEY",
        "default_model": "deepseek-ai/deepseek-v4-pro",
        "models": [
            "deepseek-ai/deepseek-v4-pro",
        ],
    },
}


def resolve_chat_model_config(init_args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve provider presets and return final ``init_chat_model`` kwargs.

    If ``model_provider`` matches a known preset (e.g. ``minimax``), the
    returned dict will have:

    * ``model_provider`` rewritten to ``"openai"`` (OpenAI-compatible API)
    * ``base_url`` filled in from the preset when not already set
    * ``api_key`` sourced from the environment when not already set
    * ``model`` defaulted to the preset's default model when not already set
    * ``temperature`` clamped to the provider's supported range

    For unknown providers the dict is returned unchanged.
    """
    args = dict(init_args)  # shallow copy
    provider = args.get("model_provider", "openai")

    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return args

    # base_url
    if not args.get("base_url"):
        args["base_url"] = preset["base_url"]

    # api_key – fall back to env var
    if not args.get("api_key"):
        env_key = preset.get("env_key", "")
        env_val = os.environ.get(env_key, "")
        if env_val:
            args["api_key"] = env_val
            logger.info("Using %s API key from environment variable %s", provider, env_key)

    # default model
    if not args.get("model"):
        args["model"] = preset["default_model"]
        logger.info("Defaulting to model %s for provider %s", args["model"], provider)

    # temperature clamping
    temp_range = preset.get("temperature_range")
    if temp_range and "temperature" in args and args["temperature"] is not None:
        lo, hi = temp_range
        original = args["temperature"]
        args["temperature"] = max(lo, min(hi, original))
        if args["temperature"] != original:
            logger.warning(
                "Clamped temperature %.2f -> %.2f for provider %s",
                original, args["temperature"], provider,
            )

    # rewrite to openai-compatible provider for LangChain
    args["model_provider"] = "openai"

    return args


def detect_provider_from_env() -> Optional[str]:
    """Return the name of a provider whose API key is found in the environment.

    Checks ``PROVIDER_PRESETS`` in definition order and returns the first
    match, or ``None`` if no key is set.
    """
    for name, preset in PROVIDER_PRESETS.items():
        env_key = preset.get("env_key", "")
        if env_key and os.environ.get(env_key):
            return name
    return None
