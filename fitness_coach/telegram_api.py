"""Minimal async Telegram Bot API client built on aiohttp.

Only the handful of methods the bot actually uses is implemented, which keeps
the dependency surface at `aiohttp` (already required by the repository) and
makes the client easy to fake in tests.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"


class TelegramError(RuntimeError):
    """Raised when the Bot API replies with ok=false."""

    def __init__(self, method: str, description: str, error_code: Optional[int] = None):
        super().__init__(f"{method} failed: {description}")
        self.method = method
        self.description = description
        self.error_code = error_code


class TelegramClient:
    """Thin wrapper over the Bot API HTTP interface."""

    def __init__(self, token: str, session: Optional[aiohttp.ClientSession] = None):
        self.token = token
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "TelegramClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()

    async def call(
        self,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 30.0,
    ) -> Any:
        """
        Invoke a Bot API method.

        Retries transient network failures and honours the `retry_after` hint
        Telegram sends when the bot is rate limited.
        """
        session = await self._ensure_session()
        url = f"{API_ROOT}/bot{self.token}/{method}"
        body = {key: value for key, value in (payload or {}).items() if value is not None}

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with session.post(
                    url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    data = await response.json()
                    if data.get("ok"):
                        return data.get("result")

                    description = data.get("description", "unknown error")
                    error_code = data.get("error_code")
                    retry_after = (data.get("parameters") or {}).get("retry_after")
                    if retry_after:
                        await asyncio.sleep(float(retry_after))
                        continue
                    raise TelegramError(method, description, error_code)
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                last_error = error
                await asyncio.sleep(2**attempt)

        raise TelegramError(method, f"network error: {last_error}")

    async def get_updates(
        self, offset: Optional[int] = None, timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Long-poll for updates. The HTTP timeout outlives the poll timeout."""
        result = await self.call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query", "pre_checkout_query"],
            },
            timeout=timeout + 15,
        )
        return result or []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = True,
    ) -> Dict[str, Any]:
        # Telegram rejects messages over 4096 characters; send long content in
        # chunks so a verbose program never silently fails to deliver.
        chunks = _split_message(text)
        result: Dict[str, Any] = {}
        for index, chunk in enumerate(chunks):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview,
                # Attach the keyboard to the last chunk only.
                "reply_markup": reply_markup if index == len(chunks) - 1 else None,
            }
            try:
                result = await self.call("sendMessage", payload)
            except TelegramError as error:
                if parse_mode is None or "parse" not in error.description.lower():
                    raise
                # An unbalanced "<" from a model answer or a user's own text
                # must not swallow the whole message: resend it as plain text.
                logger.info("Retrying message to %s without HTML parsing", chat_id)
                payload["parse_mode"] = None
                result = await self.call("sendMessage", payload)
        return result

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: Optional[str] = "HTML",
    ) -> Any:
        return await self.call(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text[:4096],
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
                "disable_web_page_preview": True,
            },
        )

    async def answer_callback_query(
        self, callback_query_id: str, text: str = "", show_alert: bool = False
    ) -> Any:
        return await self.call(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": text or None,
                "show_alert": show_alert,
            },
        )

    async def send_invoice(
        self,
        chat_id: int,
        title: str,
        description: str,
        payload: str,
        prices: List[Dict[str, Any]],
        currency: str = "XTR",
        provider_token: str = "",
    ) -> Any:
        """
        Send a payment invoice.

        With `currency="XTR"` this bills Telegram Stars, which needs no payment
        provider token and works in every country Telegram supports.
        """
        return await self.call(
            "sendInvoice",
            {
                "chat_id": chat_id,
                "title": title,
                "description": description,
                "payload": payload,
                "provider_token": provider_token,
                "currency": currency,
                "prices": prices,
            },
        )

    async def answer_pre_checkout_query(
        self, pre_checkout_query_id: str, ok: bool = True, error_message: str = ""
    ) -> Any:
        return await self.call(
            "answerPreCheckoutQuery",
            {
                "pre_checkout_query_id": pre_checkout_query_id,
                "ok": ok,
                "error_message": error_message or None,
            },
        )

    async def send_document(
        self, chat_id: int, filename: str, content: bytes, caption: str = ""
    ) -> Any:
        """Upload an in-memory file (used by the Pro data export)."""
        session = await self._ensure_session()
        url = f"{API_ROOT}/bot{self.token}/sendDocument"
        form = aiohttp.FormData()
        form.add_field("chat_id", str(chat_id))
        if caption:
            form.add_field("caption", caption[:1024])
        form.add_field("document", content, filename=filename)
        try:
            async with session.post(
                url, data=form, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                try:
                    data = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    # A proxy or gateway can answer with a non-JSON body; the
                    # caller still needs a TelegramError it can handle.
                    detail = await response.text()
                    raise TelegramError(
                        "sendDocument", f"HTTP {response.status}: {detail[:200]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise TelegramError("sendDocument", f"network error: {error}") from error

        if not data.get("ok"):
            raise TelegramError("sendDocument", data.get("description", "unknown error"))
        return data.get("result")

    async def set_my_commands(self, commands: List[Dict[str, str]]) -> Any:
        return await self.call("setMyCommands", {"commands": commands})


def escape_html(text: str) -> str:
    """Escape text that goes into an HTML-formatted message."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _split_message(text: str, limit: int = 4000) -> List[str]:
    """Split a long message on paragraph, then line, then hard boundaries."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n\n")
        if split_at < limit // 2:
            split_at = window.rfind("\n")
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
