"""MuAPI image provider for ViMax."""

import math
import re
from typing import Any, List, Optional
from urllib.parse import urlparse

from interfaces.image_output import ImageOutput
from tools.muapi_client import MuAPIClient
from utils.rate_limiter import RateLimiter


class ImageGeneratorMuAPI:
    """Generate ViMax images through MuAPI's text and image endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        text_to_image_model: Optional[str] = None,
        image_to_image_model: Optional[str] = None,
        base_url: str = "https://api.muapi.ai/api/v1",
        rate_limiter: Optional[RateLimiter] = None,
        poll_interval: float = 3.0,
        max_poll_attempts: int = 200,
        timeout: float = 120.0,
    ):
        paired_text_model, paired_image_model = self._paired_models(model)
        self.text_to_image_model = text_to_image_model or paired_text_model
        self.image_to_image_model = image_to_image_model or paired_image_model
        # Keep a single model attribute for the agent runtime's provider
        # introspection and for consistency with the other ViMax adapters.
        self.model = model or self.text_to_image_model
        self.client = MuAPIClient(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            timeout=timeout,
        )

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = None,
        num_images: int = 1,
        **kwargs,
    ) -> ImageOutput:
        """Generate one image and return the first completed output.

        ViMax supplies local reference paths. MuAPI requires hosted URLs, so
        references are uploaded before the generation request is submitted.
        """

        reference_image_paths = reference_image_paths or []
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not 1 <= num_images <= 4:
            raise ValueError("num_images must be between 1 and 4")

        progress = kwargs.get("progress")
        resolved_aspect_ratio = self._resolve_aspect_ratio(
            aspect_ratio=aspect_ratio,
            size=kwargs.get("size"),
        )
        payload = {
            "prompt": prompt,
            "aspect_ratio": resolved_aspect_ratio,
            "num_images": num_images,
        }

        if reference_image_paths:
            payload["images_list"] = [
                await self.client.upload_file(path) for path in reference_image_paths
            ]
            endpoint = self.image_to_image_model
        else:
            endpoint = self.text_to_image_model

        self._emit_progress(
            progress,
            "image_generation",
            f"Generating image with {endpoint}",
            {"model": endpoint, "reference_count": len(reference_image_paths)},
        )
        outputs = await self.client.generate(endpoint, payload)
        output_url = outputs[0]
        self._emit_progress(
            progress,
            "image_completed",
            "MuAPI image generation completed",
            {"model": endpoint},
        )
        return ImageOutput(
            fmt="url",
            ext=self._extension_from_url(output_url, default="png"),
            data=output_url,
        )

    @staticmethod
    def _paired_models(model: Optional[str]) -> tuple[str, str]:
        if not model:
            return "flux-kontext-dev-t2i", "flux-kontext-dev-i2i"
        if model.endswith("-t2i"):
            return model, f"{model[:-4]}-i2i"
        if model.endswith("-i2i"):
            return f"{model[:-4]}-t2i", model
        return model, model

    @staticmethod
    def _resolve_aspect_ratio(
        aspect_ratio: Optional[str],
        size: Optional[str],
    ) -> str:
        if aspect_ratio:
            return aspect_ratio
        if size:
            match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", str(size))
            if match:
                width, height = (int(value) for value in match.groups())
                if width > 0 and height > 0:
                    divisor = math.gcd(width, height)
                    normalized = f"{width // divisor}:{height // divisor}"
                    if normalized in {
                        "9:16",
                        "16:9",
                        "1:1",
                        "4:3",
                        "3:4",
                        "3:2",
                        "2:3",
                        "21:9",
                        "9:21",
                        "16:21",
                    }:
                        return normalized
        return "16:9"

    @staticmethod
    def _extension_from_url(url: str, default: str) -> str:
        suffix = urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 1)
        if len(suffix) == 2 and suffix[1].isalnum():
            return suffix[1].lower()
        return default

    @staticmethod
    def _emit_progress(progress: Any, stage: str, message: str, metadata: dict[str, Any]) -> None:
        if progress is not None:
            progress(stage, message, metadata)
