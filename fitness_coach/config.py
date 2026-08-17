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

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

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
        )

    def validate(self) -> Optional[str]:
        """Return a human-readable error if the config cannot be used."""
        if not self.telegram_token:
            return (
                "TELEGRAM_BOT_TOKEN is not set. Create a bot via @BotFather "
                "and export the token before starting."
            )
        if self.pro_period_days <= 0:
            return "FITNESS_PRO_PERIOD_DAYS must be a positive number of days."
        return None
