import unittest
from unittest.mock import AsyncMock, patch

from tools.image_generator_muapi import ImageGeneratorMuAPI
from tools.muapi_client import MuAPIClient, MuAPIError
from tools.video_generator_muapi import VideoGeneratorMuAPI
from agent_runtime.vimax_adapters import _build_image_generator, _build_video_generator
from agent_runtime.config import api_provider_from_base_url


class FakeMuAPIClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.uploaded_paths = []
        self.generate_calls = []

    async def upload_file(self, path):
        self.uploaded_paths.append(path)
        return f"https://uploads.example/{len(self.uploaded_paths)}.png"

    async def generate(self, endpoint, payload):
        self.generate_calls.append((endpoint, payload))
        return self.outputs


class MuAPIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_submits_polls_and_returns_outputs(self):
        client = MuAPIClient(api_key="test-key", poll_interval=0, max_poll_attempts=3)
        client._request_json = AsyncMock(
            side_effect=[
                {"request_id": "request-1"},
                {"status": "processing"},
                {"status": "completed", "outputs": ["https://cdn.example/image.png"]},
            ]
        )

        outputs = await client.generate("flux-kontext-dev-t2i", {"prompt": "a tree"})

        self.assertEqual(outputs, ["https://cdn.example/image.png"])
        self.assertEqual(client._request_json.await_count, 3)

    async def test_client_raises_for_failed_prediction(self):
        client = MuAPIClient(api_key="test-key", poll_interval=0, max_poll_attempts=2)
        client._request_json = AsyncMock(
            side_effect=[
                {"request_id": "request-2"},
                {"status": "failed", "error": "model unavailable"},
            ]
        )

        with self.assertRaisesRegex(MuAPIError, "model unavailable"):
            await client.generate("veo3-fast-text-to-video", {"prompt": "a tree"})

    async def test_image_provider_uploads_references_and_uses_override(self):
        provider = ImageGeneratorMuAPI(
            api_key="test-key",
            text_to_image_model="custom-t2i",
            image_to_image_model="custom-i2i",
        )
        fake_client = FakeMuAPIClient(["https://cdn.example/edited.webp"])
        provider.client = fake_client

        output = await provider.generate_single_image(
            prompt="change the coat to red",
            reference_image_paths=["character.png"],
            aspect_ratio="1:1",
        )

        self.assertEqual(output.data, "https://cdn.example/edited.webp")
        self.assertEqual(output.ext, "webp")
        self.assertEqual(fake_client.uploaded_paths, ["character.png"])
        self.assertEqual(fake_client.generate_calls[0][0], "custom-i2i")
        self.assertEqual(
            fake_client.generate_calls[0][1]["images_list"],
            ["https://uploads.example/1.png"],
        )

    async def test_video_provider_uses_text_to_video_override(self):
        provider = VideoGeneratorMuAPI(
            api_key="test-key",
            t2v_model="custom-video-model",
        )
        fake_client = FakeMuAPIClient(["https://cdn.example/clip.mp4"])
        provider.client = fake_client

        output = await provider.generate_single_video(
            prompt="a slow camera move",
            aspect_ratio="9:16",
        )

        self.assertEqual(output.data, "https://cdn.example/clip.mp4")
        self.assertEqual(fake_client.generate_calls[0][0], "custom-video-model")
        self.assertEqual(fake_client.generate_calls[0][1]["aspect_ratio"], "9:16")

    async def test_video_provider_uploads_two_frames_for_image_to_video(self):
        provider = VideoGeneratorMuAPI(api_key="test-key", i2v_model="custom-i2v")
        fake_client = FakeMuAPIClient(["https://cdn.example/clip.mp4"])
        provider.client = fake_client

        await provider.generate_single_video(
            prompt="match the final frame",
            reference_image_paths=["first.png", "last.png"],
        )

        self.assertEqual(fake_client.generate_calls[0][0], "custom-i2v")
        self.assertEqual(
            fake_client.generate_calls[0][1]["images_list"],
            ["https://uploads.example/1.png", "https://uploads.example/2.png"],
        )

    def test_single_model_names_pair_muapi_endpoints(self):
        image_provider = ImageGeneratorMuAPI(
            api_key="test-key",
            model="flux-kontext-dev-t2i",
        )
        video_provider = VideoGeneratorMuAPI(
            api_key="test-key",
            model="veo3-fast-text-to-video",
        )

        self.assertEqual(image_provider.text_to_image_model, "flux-kontext-dev-t2i")
        self.assertEqual(image_provider.image_to_image_model, "flux-kontext-dev-i2i")
        self.assertEqual(video_provider.t2v_model, "veo3-fast-text-to-video")
        self.assertEqual(video_provider.i2v_model, "veo3-fast-image-to-video")

        reverse_video_provider = VideoGeneratorMuAPI(
            api_key="test-key",
            model="veo3-fast-image-to-video",
        )
        self.assertEqual(reverse_video_provider.t2v_model, "veo3-fast-text-to-video")
        self.assertEqual(reverse_video_provider.i2v_model, "veo3-fast-image-to-video")

        default_video_provider = VideoGeneratorMuAPI(
            api_key="test-key",
            model="veo3.1-fast",
        )
        self.assertEqual(default_video_provider.t2v_model, "veo3.1-fast-text-to-video")
        self.assertEqual(default_video_provider.i2v_model, "veo3.1-fast-image-to-video")

    def test_agent_factory_selects_muapi_from_base_url(self):
        self.assertEqual(api_provider_from_base_url("https://api.muapi.ai/api/v1"), "muapi")
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="flux-kontext-dev-t2i"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://api.muapi.ai/api/v1"):
            image_provider = _build_image_generator()
        with patch("agent_runtime.vimax_adapters.video_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.video_model", return_value="veo3-fast-text-to-video"), \
             patch("agent_runtime.vimax_adapters.video_base_url", return_value="https://api.muapi.ai/api/v1"), \
             patch("agent_runtime.vimax_adapters.video_provider", return_value="muapi"):
            video_provider = _build_video_generator()

        self.assertIsInstance(image_provider, ImageGeneratorMuAPI)
        self.assertIsInstance(video_provider, VideoGeneratorMuAPI)


if __name__ == "__main__":
    unittest.main()
