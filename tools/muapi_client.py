"""Small async client for MuAPI's submit-then-poll REST API."""

import asyncio
import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

import aiohttp

from utils.rate_limiter import RateLimiter


class MuAPIError(RuntimeError):
    """Raised when MuAPI rejects a request or a generation fails."""


class MuAPIClient:
    """Shared transport used by the image and video provider adapters.

    MuAPI accepts local media through ``/upload_file`` and exposes the same
    asynchronous lifecycle for every model: submit a job, then poll its
    prediction result until it completes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.muapi.ai/api/v1",
        rate_limiter: Optional[RateLimiter] = None,
        poll_interval: float = 3.0,
        max_poll_attempts: int = 200,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("MUAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MuAPI API key is required. Pass api_key or set MUAPI_API_KEY."
            )

        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if max_poll_attempts <= 0:
            raise ValueError("max_poll_attempts must be positive")

        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.timeout = timeout

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
        form_data: Optional[aiohttp.FormData] = None,
    ) -> Any:
        headers = {"x-api-key": self.api_key}
        request_kwargs: Dict[str, Any] = {"headers": headers}
        if json_payload is not None:
            headers["Content-Type"] = "application/json"
            request_kwargs["json"] = json_payload
        elif form_data is not None:
            request_kwargs["data"] = form_data

        url = f"{self.base_url}/{path.lstrip('/')}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **request_kwargs) as response:
                response_text = await response.text()
                try:
                    payload = json.loads(response_text) if response_text else {}
                except json.JSONDecodeError as exc:
                    raise MuAPIError(
                        f"MuAPI returned invalid JSON ({response.status}) from {path}: "
                        f"{response_text[:500]}"
                    ) from exc

                if response.status >= 400:
                    raise MuAPIError(
                        f"MuAPI request failed with HTTP {response.status} at {path}: "
                        f"{payload}"
                    )
                return payload

    async def submit(self, endpoint: str, payload: Dict[str, Any]) -> str:
        """Submit a model request and return its prediction ID."""

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        response = await self._request_json("POST", endpoint, json_payload=payload)
        request_id = response.get("request_id")
        if request_id is None and isinstance(response.get("data"), dict):
            request_id = response["data"].get("request_id")
        if not request_id:
            raise MuAPIError(f"MuAPI submission did not return request_id: {response}")
        return request_id

    async def generate(self, endpoint: str, payload: Dict[str, Any]) -> List[str]:
        """Submit a job, poll it, and return its output URLs."""

        request_id = await self.submit(endpoint, payload)
        return await self.poll_result(request_id)

    async def upload_file(self, path: str) -> str:
        """Upload a local media file and return its hosted URL."""

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        form_data = aiohttp.FormData()
        with open(path, "rb") as media_file:
            form_data.add_field(
                "file",
                media_file,
                filename=os.path.basename(path),
                content_type=content_type,
            )
            response = await self._request_json(
                "POST",
                "upload_file",
                form_data=form_data,
            )

        upload_url = response.get("url")
        if upload_url is None and isinstance(response.get("data"), dict):
            upload_url = response["data"].get("url")
        if not upload_url:
            raise MuAPIError(f"MuAPI upload did not return a URL: {response}")
        return upload_url

    async def poll_result(self, request_id: str) -> List[str]:
        """Poll one prediction until it completes or fails."""

        for attempt in range(self.max_poll_attempts):
            response = await self._request_json(
                "GET",
                f"predictions/{request_id}/result",
            )
            result = response
            if isinstance(response.get("data"), dict) and "status" not in response:
                result = response["data"]

            status = result.get("status")
            if status == "completed":
                outputs = self._extract_outputs(result)
                if not outputs:
                    raise MuAPIError(
                        f"MuAPI prediction {request_id} completed without outputs: {response}"
                    )
                return outputs

            if status in {"failed", "cancelled"}:
                error = result.get("error") or result.get("message") or result
                raise MuAPIError(f"MuAPI prediction {request_id} {status}: {error}")

            if attempt == self.max_poll_attempts - 1:
                raise MuAPIError(
                    f"MuAPI prediction {request_id} did not complete after "
                    f"{self.max_poll_attempts} polls (last status: {status!r})"
                )

            logging.info(
                "MuAPI prediction %s is %s; polling again in %ss",
                request_id,
                status or "unknown",
                self.poll_interval,
            )
            await asyncio.sleep(self.poll_interval)

        raise AssertionError("poll loop should always return or raise")

    @staticmethod
    def _extract_outputs(result: Dict[str, Any]) -> List[str]:
        outputs = result.get("outputs")
        if outputs is None:
            outputs = result.get("output")

        if outputs is None:
            for key in ("image", "video", "audio"):
                media = result.get(key)
                if isinstance(media, dict) and media.get("url"):
                    outputs = [media["url"]]
                    break
                if isinstance(media, str):
                    outputs = [media]
                    break

        if isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list):
            return []

        normalized: List[str] = []
        for output in outputs:
            if isinstance(output, str) and output:
                normalized.append(output)
            elif isinstance(output, dict) and output.get("url"):
                normalized.append(output["url"])
        return normalized
