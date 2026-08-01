import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from agent_runtime.config import api_provider_from_base_url
from agent_runtime.vimax_adapters import _build_image_generator
from tools.image_generator_orcarouter_api import (
    ImageGeneratorOrcaRouterAPI,
    OrcaRouterImageAPIError,
    _is_retryable_image_error,
)


def _encoded_png(size: tuple[int, int] = (16, 9)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OrcaRouterImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_image_with_openai_compatible_images_api(self):
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        progress = []
        generator = ImageGeneratorOrcaRouterAPI(api_key="secret", model="openai/gpt-image-2")
        with patch("tools.image_generator_orcarouter_api._post_json", fake_post):
            result = await generator.generate_single_image(
                "a cinematic beach",
                aspect_ratio="16:9",
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertEqual(captured["url"], "https://api.orcarouter.ai/v1/images/generations")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")
        self.assertEqual(captured["payload"]["model"], "openai/gpt-image-2")
        self.assertNotIn("aspect_ratio", captured["payload"])
        self.assertIn("landscape image", captured["payload"]["prompt"])
        self.assertEqual(result.data.size, (16, 9))
        self.assertEqual(result.ext, "png")
        self.assertEqual([item[0] for item in progress], ["image_generation", "image_completed"])

    async def test_reference_images_use_the_edits_endpoint(self):
        post = AsyncMock(return_value=(200, {"data": [{"b64_json": _encoded_png()}]}))
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 9), "red").save(reference_path)
            generator = ImageGeneratorOrcaRouterAPI(api_key="secret")
            with patch("tools.image_generator_orcarouter_api._post_multipart", post):
                await generator.generate_single_image("edit this", [str(reference_path)])

            self.assertEqual(
                post.await_args.args[0],
                "https://api.orcarouter.ai/v1/images/edits",
            )
            self.assertEqual(post.await_args.kwargs["reference_paths"], [str(reference_path)])

        # multipart requests must not pin Content-Type; aiohttp adds the boundary
        self.assertNotIn("Content-Type", post.await_args.kwargs["headers"])
        self.assertEqual(post.await_args.kwargs["fields"]["model"], "openai/gpt-image-2")

    async def test_too_many_reference_images_is_rejected(self):
        generator = ImageGeneratorOrcaRouterAPI(api_key="secret")
        with self.assertRaises(ValueError):
            await generator.generate_single_image("edit this", [f"ref-{index}.png" for index in range(17)])

    async def test_non_retryable_client_error_is_not_repeated(self):
        post = AsyncMock(return_value=(400, {"error": {"message": "bad request"}}))
        generator = ImageGeneratorOrcaRouterAPI(api_key="secret")
        with patch("tools.image_generator_orcarouter_api._post_json", post):
            with self.assertRaises(OrcaRouterImageAPIError):
                await generator.generate_single_image("bad request")
        self.assertEqual(post.await_count, 1)

    def test_retry_policy_is_bounded_to_transient_errors_and_portrait_outputs(self):
        self.assertTrue(_is_retryable_image_error(OrcaRouterImageAPIError(429, {})))
        self.assertTrue(_is_retryable_image_error(OrcaRouterImageAPIError(500, {})))
        self.assertTrue(_is_retryable_image_error(ValueError("Generated image is portrait-oriented (9x16); retrying for a landscape frame")))
        self.assertFalse(_is_retryable_image_error(OrcaRouterImageAPIError(401, {})))
        self.assertFalse(_is_retryable_image_error(ValueError("invalid reference image")))

    def test_base_url_is_mapped_to_the_orcarouter_provider(self):
        self.assertEqual(api_provider_from_base_url("https://api.orcarouter.ai/v1"), "orcarouter")
        self.assertEqual(api_provider_from_base_url("https://openrouter.ai/api/v1"), "openrouter")

    def test_agent_factory_selects_orcarouter_from_image_base_url(self):
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="openai/gpt-image-2"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://api.orcarouter.ai/v1"):
            generator = _build_image_generator()
        self.assertIsInstance(generator, ImageGeneratorOrcaRouterAPI)
        self.assertEqual(generator.model, "openai/gpt-image-2")


if __name__ == "__main__":
    unittest.main()
