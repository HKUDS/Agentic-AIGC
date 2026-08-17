"""Subscription payments via Telegram Stars, plus promo codes and admin grants."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import Config
from .models import Subscription, User
from .storage import Storage
from .telegram_api import TelegramClient

# Invoice payload. Versioned so a future price or period change can be told
# apart from old, still-in-flight invoices.
PRO_PAYLOAD = "pro_subscription_v1"
STARS_CURRENCY = "XTR"


@dataclass
class ActivationResult:
    """Outcome of granting Pro access."""

    activated: bool
    expires_at: Optional[float]
    reason: str = ""


def activate_pro(
    storage: Storage,
    user_id: int,
    days: int,
    source: str,
    now: Optional[float] = None,
) -> ActivationResult:
    """
    Grant or extend Pro access.

    Buying again while Pro is still active extends the existing subscription
    instead of overwriting it, so nobody loses paid days by renewing early.
    """
    moment = now if now is not None else time.time()
    current = storage.get_subscription(user_id)

    base = moment
    if current is not None and current.is_active_pro(moment) and current.expires_at:
        base = current.expires_at

    expires_at = base + days * 86400
    storage.set_subscription(user_id, "pro", expires_at, source)
    return ActivationResult(activated=True, expires_at=expires_at)


def grant_lifetime_pro(storage: Storage, user_id: int, source: str = "admin") -> ActivationResult:
    """Give unlimited Pro (used by `/grant` from an admin account)."""
    storage.set_subscription(user_id, "pro", None, source)
    return ActivationResult(activated=True, expires_at=None)


def revoke_pro(storage: Storage, user_id: int) -> None:
    storage.set_subscription(user_id, "free", None, "revoked")


def redeem_promo(
    storage: Storage,
    config: Config,
    user: User,
    code: str,
    now: Optional[float] = None,
) -> ActivationResult:
    """
    Apply a promo code.

    Codes come from `FITNESS_PROMO_CODES` and are single-use per user.
    """
    normalised = (code or "").strip().upper()
    if not normalised:
        return ActivationResult(False, None, "empty")
    if normalised not in config.promo_codes:
        return ActivationResult(False, None, "unknown")
    if not storage.redeem_promo(user.id, normalised):
        return ActivationResult(False, None, "already_used")
    return activate_pro(
        storage, user.id, config.pro_period_days, f"promo:{normalised}", now
    )


async def send_pro_invoice(client: TelegramClient, config: Config, chat_id: int) -> None:
    """Send the Telegram Stars invoice for the Pro subscription."""
    description = (
        f"Pro-доступ на {config.pro_period_days} дней: адаптивная программа на "
        "4 недели, безлимитный чат с ИИ-тренером, план питания с расчётом БЖУ, "
        "до 10 напоминаний, аналитика прогресса и экспорт данных."
    )
    await client.send_invoice(
        chat_id=chat_id,
        title="ИИ-тренер Pro",
        description=description,
        payload=PRO_PAYLOAD,
        prices=[{"label": "Pro-подписка", "amount": config.pro_price_stars}],
        currency=STARS_CURRENCY,
    )


def handle_successful_payment(
    storage: Storage,
    config: Config,
    user: User,
    payment: dict,
    now: Optional[float] = None,
) -> Tuple[ActivationResult, Subscription]:
    """
    Process a `successful_payment` update.

    Telegram may redeliver an update; the charge id is stored with a unique
    index so a repeat delivery never grants a second month.
    """
    charge_id = payment.get("telegram_payment_charge_id", "")
    amount = int(payment.get("total_amount") or 0)
    currency = payment.get("currency", "")
    payload = payment.get("invoice_payload", "")

    fresh = storage.record_payment(user.id, amount, currency, charge_id, payload)
    if not fresh:
        subscription = storage.get_subscription(user.id) or Subscription(user_id=user.id)
        return (
            ActivationResult(False, subscription.expires_at, "duplicate"),
            subscription,
        )

    result = activate_pro(
        storage, user.id, config.pro_period_days, f"stars:{charge_id}", now
    )
    subscription = storage.get_subscription(user.id) or Subscription(user_id=user.id)
    return result, subscription
