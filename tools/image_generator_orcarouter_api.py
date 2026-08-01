from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any, List

import aiohttp
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from interfaces.image_output import ImageOutput
from tools.image_orientation import ensure_not_portrait, landscape_guard_requested
from utils.rate_limiter import RateLimiter
from utils.retry import after_func

DEFAULT_ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
MAX_REFERENCE_IMAGES = 16


class OrcaRouterImageAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        super().__init__(f"OrcaRouter image generation failed with HTTP {status_code}: {payload}")


def _request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_IMAGE_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _is_retryable_image_error(exc: BaseException) -> bool:
    if isinstance(exc, OrcaRouterImageAPIError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, ValueError) and "portrait-oriented" in str(exc)


class ImageGeneratorOrcaRouterAPI:
    """Generate images through OrcaRouter's OpenAI-compatible Images API.

    Text-to-image goes to ``POST {base_url}/images/generations`` (JSON).
    When reference images are supplied the request goes to
    ``POST {base_url}/images/edits`` instead, which takes the references as
    multipart ``image[]`` parts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-image-2",
        base_url: str = DEFAULT_ORCAROUTER_BASE_URL,
        quality: str = "auto",
        background: str = "auto",
        output_compression: int | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.quality = quality
        self.background = background
        self.output_compression = output_compression
        self.rate_limiter = rate_limiter

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_image_error),
        after=after_func,
        reraise=True,
    )
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] | None = None,
        aspect_ratio: str | None = "16:9",
        **kwargs: Any,
    ) -> ImageOutput:
        references = list(reference_image_paths or [])
        if len(references) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"OrcaRouter image editing supports at most {MAX_REFERENCE_IMAGES} reference images"
            )
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        enforce_landscape = landscape_guard_requested(
            size=kwargs.get("size"),
            aspect_ratio=aspect_ratio,
            enforce_landscape=kwargs.get("enforce_landscape", True),
            allow_portrait=kwargs.get("allow_portrait", False),
        )
        request_prompt = _prompt_with_landscape_requirement(prompt, aspect_ratio) if enforce_landscape else prompt
        fields: dict[str, Any] = {
            "model": self.model,
            "prompt": request_prompt,
            "n": 1,
            "quality": kwargs.get("quality", self.quality),
            "background": kwargs.get("background", self.background),
        }
        compression = kwargs.get("output_compression", self.output_compression)
        if compression is not None:
            fields["output_compression"] = compression

        progress = kwargs.get("progress")
        _emit_progress(
            progress,
            "image_generation",
            f"Generating image with {self.model}",
            {"model": self.model, "reference_count": len(references)},
        )
        timeout = aiohttp.ClientTimeout(total=_request_timeout_seconds())
        if references:
            status, response = await _post_multipart(
                f"{self.base_url}/images/edits",
                headers=self._headers(),
                fields=fields,
                reference_paths=references,
                timeout=timeout,
            )
        else:
            status, response = await _post_json(
                f"{self.base_url}/images/generations",
                headers=self._headers(json_body=True),
                payload=fields,
                timeout=timeout,
            )
        if status >= 400:
            raise OrcaRouterImageAPIError(status, response)

        image, extension = _decode_image_response(response)
        if enforce_landscape:
            ensure_not_portrait(image)
        _emit_progress(
            progress,
            "image_completed",
            "OrcaRouter image generation completed",
            {"model": self.model, "width": image.width, "height": image.height},
        )
        return ImageOutput(fmt="pil", ext=extension, data=image)

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_body:
            # multipart requests must let aiohttp set Content-Type (it carries the boundary)
            headers["Content-Type"] = "application/json"
        return headers


def _prompt_with_landscape_requirement(prompt: str, aspect_ratio: str | None) -> str:
    ratio = aspect_ratio or "16:9"
    return f"{prompt}\n\nComposition requirement: create a landscape image with an approximate {ratio} aspect ratio; the width must be greater than the height."


def _decode_image_response(payload: Any) -> tuple[Image.Image, str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
    encoded = item.get("b64_json") if item else None
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"OrcaRouter image response missing data[0].b64_json: {payload}")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            image = opened.copy()
    except Exception as exc:
        raise ValueError("OrcaRouter image response contained invalid image data") from exc
    media_type = item.get("media_type", "image/png")
    extension = {"image/jpeg": "jpg", "image/webp": "webp"}.get(media_type, "png")
    return image, extension


def _emit_progress(progress: Any, stage: str, message: str, metadata: dict[str, Any]) -> None:
    if progress is not None:
        progress(stage, message, metadata)


async def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            return response.status, await _read_body(response)


async def _post_multipart(
    url: str,
    *,
    headers: dict[str, str],
    fields: dict[str, Any],
    reference_paths: List[str],
    timeout: aiohttp.ClientTimeout,
) -> tuple[int, Any]:
    form = aiohttp.FormData()
    for key, value in fields.items():
        form.add_field(key, str(value))
    for path in reference_paths:
        resolved = Path(path)
        content_type = mimetypes.guess_type(resolved.name)[0] or "image/png"
        form.add_field(
            "image[]",
            resolved.read_bytes(),
            filename=resolved.name,
            content_type=content_type,
        )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, data=form) as response:
            return response.status, await _read_body(response)


async def _read_body(response: aiohttp.ClientResponse) -> Any:
    text = await response.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}
