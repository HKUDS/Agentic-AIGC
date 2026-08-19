"""Environment-driven configuration for the AI fitness coach bot."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_list(name: str) -> List[str]:
    raw = _env_str(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Config:
    """
    Runtime configuration of the bot.

    Everything is read from environment variables so the same code can run
    locally, in Docker, or on a server without code changes. See
    `fitness_coach/.env.example` for the full list.
    """

    telegram_token: str = ""
    database_path: str = ".fitness_coach/bot.sqlite3"

    # LLM backend (any OpenAI-compatible endpoint). When the key is missing the
    # coach falls back to deterministic, rule-based answers instead of failing.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 60

    # Subscription / payments.
    pro_price_stars: int = 250
    pro_period_days: int = 30
    promo_codes: List[str] = field(default_factory=list)
    admin_ids: List[int] = field(default_factory=list)

    # Scheduler.
    reminder_tick_seconds: int = 30
    poll_timeout_seconds: int = 30

    # Web sign-up.
    web_enabled: bool = True
    bot_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    # Signs session tokens and verification codes. Generated per start when
    # unset, which logs everybody out on restart — set it in production.
    secret_key: str = ""
    public_url: str = ""
    bot_username: str = ""
    # Returning the code in the API response is a development shortcut and a
    # serious hole in production, so it is opt-in and refuses to stay on when
    # a real delivery backend exists.
    dev_show_code: bool = False

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True

    sms_webhook_url: str = ""
    sms_webhook_token: str = ""

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token)

    def telegram_link(self, code: str) -> str:
        """Deep link that hands a link code to the bot."""
        if not self.bot_username:
            return ""
        return f"https://t.me/{self.bot_username.lstrip('@')}?start={code}"

    def is_admin(self, telegram_id: int) -> bool:
        return telegram_id in self.admin_ids

    @classmethod
    def from_env(cls) -> "Config":
        """Build a config from the process environment."""
        admin_ids: List[int] = []
        for raw in _env_list("FITNESS_ADMIN_IDS"):
            try:
                admin_ids.append(int(raw))
            except ValueError:
                continue

        return cls(
            telegram_token=_env_str("TELEGRAM_BOT_TOKEN"),
            database_path=_env_str("FITNESS_DB_PATH", ".fitness_coach/bot.sqlite3"),
            llm_api_key=_env_str("FITNESS_LLM_API_KEY") or _env_str("OPENAI_API_KEY"),
            llm_base_url=_env_str("FITNESS_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=_env_str("FITNESS_LLM_MODEL", "gpt-4o-mini"),
            llm_timeout_seconds=_env_int("FITNESS_LLM_TIMEOUT", 60),
            pro_price_stars=_env_int("FITNESS_PRO_PRICE_STARS", 250),
            pro_period_days=_env_int("FITNESS_PRO_PERIOD_DAYS", 30),
            promo_codes=[code.upper() for code in _env_list("FITNESS_PROMO_CODES")],
            admin_ids=admin_ids,
            reminder_tick_seconds=_env_int("FITNESS_REMINDER_TICK", 30),
            poll_timeout_seconds=_env_int("FITNESS_POLL_TIMEOUT", 30),
            web_enabled=_env_bool("FITNESS_WEB_ENABLED", True),
            bot_enabled=_env_bool("FITNESS_BOT_ENABLED", True),
            web_host=_env_str("FITNESS_WEB_HOST", "0.0.0.0"),
            # Railway and most PaaS hand the port over in PORT.
            web_port=_env_int("PORT", _env_int("FITNESS_WEB_PORT", 8080)),
            secret_key=_env_str("FITNESS_SECRET_KEY"),
            public_url=_env_str("FITNESS_PUBLIC_URL").rstrip("/"),
            bot_username=_env_str("FITNESS_BOT_USERNAME"),
            dev_show_code=_env_bool("FITNESS_DEV_SHOW_CODE", False),
            smtp_host=_env_str("FITNESS_SMTP_HOST"),
            smtp_port=_env_int("FITNESS_SMTP_PORT", 587),
            smtp_user=_env_str("FITNESS_SMTP_USER"),
            smtp_password=os.environ.get("FITNESS_SMTP_PASSWORD", ""),
            smtp_from=_env_str("FITNESS_SMTP_FROM"),
            smtp_use_tls=_env_bool("FITNESS_SMTP_TLS", True),
            sms_webhook_url=_env_str("FITNESS_SMS_WEBHOOK_URL"),
            sms_webhook_token=_env_str("FITNESS_SMS_WEBHOOK_TOKEN"),
        )

    def validate(self) -> Optional[str]:
        """Return a human-readable error if the config cannot be used."""
        if not self.telegram_token and not self.web_enabled:
            return (
                "Nothing to run: set TELEGRAM_BOT_TOKEN for the bot, or enable "
                "the web app with FITNESS_WEB_ENABLED=1."
            )
        if self.pro_period_days <= 0:
            return "FITNESS_PRO_PERIOD_DAYS must be a positive number of days."
        if not 1 <= self.web_port <= 65535:
            return f"Invalid web port: {self.web_port}."
        return None
