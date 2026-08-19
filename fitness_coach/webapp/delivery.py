"""Delivering verification codes over email or SMS.

Three backends, picked from configuration:

* SMTP for email;
* an HTTP webhook for SMS, which fits every provider that accepts a JSON POST
  (and lets the deployment swap providers without touching this code);
* a console backend that only logs the code, used in development so the whole
  sign-up flow can be exercised without any provider account.
"""

import asyncio
import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

EMAIL_SUBJECT = "Код для входа в ИИ-тренер"
EMAIL_BODY = (
    "Твой код для входа: {code}\n\n"
    "Код действует 10 минут. Если это был не ты — просто проигнорируй письмо, "
    "без кода в аккаунт никто не войдёт."
)
SMS_BODY = "Код для входа в ИИ-тренер: {code}. Действует 10 минут."


@dataclass
class DeliveryResult:
    ok: bool
    error: str = ""


class Deliverer:
    """Base class: a backend that can deliver a code to one channel."""

    async def send(self, channel: str, identifier: str, code: str) -> DeliveryResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class ConsoleDeliverer(Deliverer):
    """Logs the code instead of sending it. Development only."""

    async def send(self, channel: str, identifier: str, code: str) -> DeliveryResult:
        logger.warning(
            "[dev] verification code for %s (%s): %s", identifier, channel, code
        )
        return DeliveryResult(True)


class SmtpDeliverer(Deliverer):
    """Sends codes by email over SMTP."""

    def __init__(self, config):
        self.config = config

    async def send(self, channel: str, identifier: str, code: str) -> DeliveryResult:
        if channel != "email":
            return DeliveryResult(False, "channel_unsupported")

        message = EmailMessage()
        message["Subject"] = EMAIL_SUBJECT
        message["From"] = self.config.smtp_from or self.config.smtp_user
        message["To"] = identifier
        message.set_content(EMAIL_BODY.format(code=code))

        try:
            # smtplib is blocking, so it runs off the event loop.
            await asyncio.to_thread(self._send_blocking, message)
            return DeliveryResult(True)
        except Exception as error:
            logger.error("SMTP delivery to %s failed: %s", identifier, error)
            return DeliveryResult(False, "smtp_failed")

    def _send_blocking(self, message: EmailMessage) -> None:
        config = self.config
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=20) as server:
            if config.smtp_use_tls:
                server.starttls()
            if config.smtp_user:
                server.login(config.smtp_user, config.smtp_password)
            server.send_message(message)


class WebhookDeliverer(Deliverer):
    """
    Sends codes by SMS through an HTTP webhook.

    The provider gets `{"phone": ..., "text": ...}`; most SMS gateways either
    accept that shape directly or sit behind a tiny adapter.
    """

    def __init__(self, url: str, token: str = "",
                 session: Optional[aiohttp.ClientSession] = None):
        self.url = url
        self.token = token
        self._session = session
        self._owns_session = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def send(self, channel: str, identifier: str, code: str) -> DeliveryResult:
        if channel != "phone":
            return DeliveryResult(False, "channel_unsupported")

        session = await self._ensure_session()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            async with session.post(
                self.url,
                json={"phone": identifier, "text": SMS_BODY.format(code=code)},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    logger.error(
                        "SMS webhook returned %s: %s", response.status, detail[:200]
                    )
                    return DeliveryResult(False, "sms_failed")
        except Exception as error:
            logger.error("SMS delivery to %s failed: %s", identifier, error)
            return DeliveryResult(False, "sms_failed")
        return DeliveryResult(True)

    async def close(self) -> None:
        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()


class RoutingDeliverer(Deliverer):
    """Picks a backend per channel and falls back to the console one."""

    def __init__(self, email: Optional[Deliverer] = None, sms: Optional[Deliverer] = None):
        self.email = email
        self.sms = sms
        self.console = ConsoleDeliverer()

    def backend_for(self, channel: str) -> Deliverer:
        backend = self.email if channel == "email" else self.sms
        return backend or self.console

    def is_real(self, channel: str) -> bool:
        """True when a channel has an actual provider behind it."""
        return self.backend_for(channel) is not self.console

    async def send(self, channel: str, identifier: str, code: str) -> DeliveryResult:
        return await self.backend_for(channel).send(channel, identifier, code)

    async def close(self) -> None:
        for backend in (self.email, self.sms):
            if backend is not None:
                await backend.close()


def build_deliverer(config) -> RoutingDeliverer:
    """Assemble the delivery backends the configuration enables."""
    email = SmtpDeliverer(config) if config.smtp_host else None
    sms = (
        WebhookDeliverer(config.sms_webhook_url, config.sms_webhook_token)
        if config.sms_webhook_url
        else None
    )
    if email is None:
        logger.warning("SMTP is not configured — email codes will only be logged.")
    if sms is None:
        logger.warning("SMS webhook is not configured — phone codes will only be logged.")
    return RoutingDeliverer(email=email, sms=sms)
