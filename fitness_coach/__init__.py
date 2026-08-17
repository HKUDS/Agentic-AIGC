"""AI fitness trainer for Telegram: free tier + Pro subscription.

Layout:
    config.py        environment-driven settings
    models.py        domain dataclasses
    storage.py       SQLite persistence
    subscription.py  free/pro limits and feature gating
    workouts.py      rule-based program generator
    nutrition.py     calorie and macro calculations (Pro)
    analytics.py     progress statistics and export (Pro)
    coach.py         LLM-backed trainer with an offline fallback
    scheduler.py     reminder timing and the background loop
    telegram_api.py  minimal Bot API client
    payments.py      Telegram Stars subscription and promo codes
    bot.py           handlers and dialog state machine
"""

from .config import Config
from .subscription import FREE_PLAN, PRO_PLAN, plan_for

__all__ = ["Config", "FREE_PLAN", "PRO_PLAN", "plan_for"]
