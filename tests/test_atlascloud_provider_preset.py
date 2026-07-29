"""Focused tests for the Atlas Cloud provider preset."""

import os
import unittest
from unittest.mock import patch

from utils.provider_presets import (
    PROVIDER_PRESETS,
    detect_provider_from_env,
    resolve_chat_model_config,
)


class TestAtlasCloudProviderPreset(unittest.TestCase):
    def test_preset_uses_openai_compatible_endpoint(self):
        preset = PROVIDER_PRESETS["atlascloud"]

        self.assertEqual(preset["base_url"], "https://api.atlascloud.ai/v1")
        self.assertEqual(preset["env_key"], "ATLASCLOUD_API_KEY")
        self.assertEqual(preset["default_model"], "deepseek-ai/deepseek-v4-pro")

    def test_resolver_rewrites_provider_and_applies_defaults(self):
        result = resolve_chat_model_config(
            {"model_provider": "atlascloud", "api_key": "test-key"}
        )

        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["base_url"], "https://api.atlascloud.ai/v1")
        self.assertEqual(result["model"], "deepseek-ai/deepseek-v4-pro")

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "env-key"}, clear=True)
    def test_resolver_reads_api_key_from_environment(self):
        result = resolve_chat_model_config(
            {"model_provider": "atlascloud", "model": "custom-model"}
        )

        self.assertEqual(result["api_key"], "env-key")
        self.assertEqual(result["model"], "custom-model")

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "env-key"}, clear=True)
    def test_environment_detection_includes_atlascloud(self):
        self.assertEqual(detect_provider_from_env(), "atlascloud")

    def test_explicit_values_are_preserved(self):
        result = resolve_chat_model_config(
            {
                "model_provider": "atlascloud",
                "model": "custom-model",
                "base_url": "https://proxy.example.com/v1",
                "api_key": "explicit-key",
            }
        )

        self.assertEqual(result["model"], "custom-model")
        self.assertEqual(result["base_url"], "https://proxy.example.com/v1")
        self.assertEqual(result["api_key"], "explicit-key")


if __name__ == "__main__":
    unittest.main()
