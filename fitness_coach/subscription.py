"""Plan definitions and the single place where free/pro limits are decided."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

from .models import Subscription

# Feature identifiers used by the bot when gating a handler.
FEATURE_AI_CHAT = "ai_chat"
FEATURE_BASIC_PLAN = "basic_plan"
FEATURE_WORKOUT_LOG = "workout_log"
FEATURE_WORKOUT_REMINDER = "workout_reminder"
FEATURE_CUSTOM_REMINDERS = "custom_reminders"
FEATURE_ADAPTIVE_PLAN = "adaptive_plan"
FEATURE_NUTRITION = "nutrition"
FEATURE_ANALYTICS = "analytics"
FEATURE_EXPORT = "export"
FEATURE_TECHNIQUE_REVIEW = "technique_review"


@dataclass(frozen=True)
class Plan:
    """Limits of a single tier. Instances are immutable and module-level."""

    name: str
    title: str
    ai_messages_per_day: int
    max_reminders: int
    program_weeks: int
    features: FrozenSet[str] = field(default_factory=frozenset)

    def has(self, feature: str) -> bool:
        return feature in self.features


FREE_PLAN = Plan(
    name="free",
    title="Free",
    ai_messages_per_day=5,
    max_reminders=1,
    program_weeks=1,
    features=frozenset(
        {
            FEATURE_AI_CHAT,
            FEATURE_BASIC_PLAN,
            FEATURE_WORKOUT_LOG,
            FEATURE_WORKOUT_REMINDER,
        }
    ),
)

PRO_PLAN = Plan(
    name="pro",
    title="Pro",
    ai_messages_per_day=0,  # 0 means unlimited.
    max_reminders=10,
    program_weeks=4,
    features=frozenset(
        {
            FEATURE_AI_CHAT,
            FEATURE_BASIC_PLAN,
            FEATURE_WORKOUT_LOG,
            FEATURE_WORKOUT_REMINDER,
            FEATURE_CUSTOM_REMINDERS,
            FEATURE_ADAPTIVE_PLAN,
            FEATURE_NUTRITION,
            FEATURE_ANALYTICS,
            FEATURE_EXPORT,
            FEATURE_TECHNIQUE_REVIEW,
        }
    ),
)

PLANS: Dict[str, Plan] = {FREE_PLAN.name: FREE_PLAN, PRO_PLAN.name: PRO_PLAN}

# Human-readable labels for the paywall screen, in display order.
PRO_FEATURE_LABELS = (
    (FEATURE_ADAPTIVE_PLAN, "Адаптивная программа на 4 недели, которая меняется по твоим отчётам"),
    (FEATURE_AI_CHAT, "Безлимитный чат с ИИ-тренером (на Free — 5 сообщений в день)"),
    (FEATURE_NUTRITION, "Расчёт калорий, БЖУ и план питания"),
    (FEATURE_CUSTOM_REMINDERS, "До 10 напоминаний: тренировки, вода, взвешивание, свои"),
    (FEATURE_ANALYTICS, "Аналитика прогресса: объём, регулярность, динамика веса"),
    (FEATURE_TECHNIQUE_REVIEW, "Разбор техники упражнений по твоему описанию"),
    (FEATURE_EXPORT, "Экспорт всех данных и программы в файл"),
)


def plan_for(subscription: Optional[Subscription], now: Optional[float] = None) -> Plan:
    """Resolve the effective plan, downgrading expired Pro back to Free."""
    if subscription is not None and subscription.is_active_pro(now):
        return PRO_PLAN
    return FREE_PLAN


def can_use(
    feature: str,
    subscription: Optional[Subscription],
    now: Optional[float] = None,
) -> bool:
    """True when the subscription's effective plan unlocks `feature`."""
    return plan_for(subscription, now).has(feature)


def ai_messages_left(
    subscription: Optional[Subscription],
    used_today: int,
    now: Optional[float] = None,
) -> Optional[int]:
    """
    Remaining AI messages for today.

    Returns None when the plan is unlimited, otherwise a non-negative count.
    """
    plan = plan_for(subscription, now)
    if plan.ai_messages_per_day <= 0:
        return None
    return max(0, plan.ai_messages_per_day - used_today)


def reminder_slots_left(
    subscription: Optional[Subscription],
    active_reminders: int,
    now: Optional[float] = None,
) -> int:
    """How many more reminders the user may create."""
    plan = plan_for(subscription, now)
    return max(0, plan.max_reminders - active_reminders)
