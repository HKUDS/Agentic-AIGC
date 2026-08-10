"""MuAPI video provider for ViMax."""

from typing import Any, List, Optional
from urllib.parse import urlparse

from interfaces.video_output import VideoOutput
from tools.muapi_client import MuAPIClient
from utils.rate_limiter import RateLimiter


class VideoGeneratorMuAPI:
    """Generate ViMax clips through MuAPI text/image-to-video endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        t2v_model: Optional[str] = None,
        i2v_model: Optional[str] = None,
        base_url: str = "https://api.muapi.ai/api/v1",
        rate_limiter: Optional[RateLimiter] = None,
        poll_interval: float = 3.0,
        max_poll_attempts: int = 200,
        timeout: float = 120.0,
    ):
        paired_text_model, paired_image_model = self._paired_models(model)
        self.t2v_model = t2v_model or paired_text_model
        self.i2v_model = i2v_model or paired_image_model
        self.model = model or self.t2v_model
        self.client = MuAPIClient(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            timeout=timeout,
        )

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> VideoOutput:
        """Generate one clip and return its first completed output.

        The default Veo 3 / Veo 3.1 endpoints accept one or more image URLs in
        ``images_list``. The ``resolution`` and ``duration`` arguments used
        by some other ViMax providers are intentionally not sent because
        they are not part of the default MuAPI Veo 3 request schema.
        """

        reference_image_paths = reference_image_paths or []
        if not prompt:
            raise ValueError("prompt must not be empty")
        if len(reference_image_paths) > 2:
            raise ValueError("reference_image_paths must contain no more than 2 images")
        if aspect_ratio not in {"9:16", "16:9"}:
            raise ValueError("MuAPI Veo 3 endpoints support 9:16 and 16:9 aspect ratios")

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
        }
        if reference_image_paths:
            payload["images_list"] = [
                await self.client.upload_file(path) for path in reference_image_paths
            ]
            endpoint = self.i2v_model
        else:
            endpoint = self.t2v_model

        progress = kwargs.get("progress")
        self._emit_progress(
            progress,
            "video_create",
            f"Creating MuAPI video generation task with {endpoint}",
            {"model": endpoint, "frame_count": len(reference_image_paths)},
        )
        outputs = await self.client.generate(endpoint, payload)
        output_url = outputs[0]
        self._emit_progress(
            progress,
            "video_completed",
            "MuAPI video generation completed",
            {"model": endpoint},
        )
        return VideoOutput(
            fmt="url",
            ext=self._extension_from_url(output_url, default="mp4"),
            data=output_url,
        )

    @staticmethod
    def _paired_models(model: Optional[str]) -> tuple[str, str]:
        if not model:
            return "veo3-fast-text-to-video", "veo3-fast-image-to-video"
        aliases = {
            "veo3-fast": ("veo3-fast-text-to-video", "veo3-fast-image-to-video"),
            "veo3.1-fast": ("veo3.1-fast-text-to-video", "veo3.1-fast-image-to-video"),
        }
        if model in aliases:
            return aliases[model]

        text_suffix = "-text-to-video"
        image_suffix = "-image-to-video"
        if model.endswith(text_suffix):
            return model, f"{model[:-len(text_suffix)]}{image_suffix}"
        if model.endswith(image_suffix):
            return f"{model[:-len(image_suffix)]}{text_suffix}", model
        return model, model

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
