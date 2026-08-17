"""Domain models shared by storage, business logic, and the Telegram layer."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

# Profile enumerations. Values are stored verbatim in SQLite, so they double as
# the wire format for callback data — keep them short and stable.
GOALS = ("lose_weight", "build_muscle", "keep_fit", "endurance")
LEVELS = ("beginner", "intermediate", "advanced")
GENDERS = ("male", "female", "other")
EQUIPMENT = ("none", "home", "gym")

REMINDER_KINDS = ("workout", "water", "weigh_in", "custom")


@dataclass
class Profile:
    """Everything the coach needs to build a personalised program."""

    user_id: int
    gender: str = "other"
    age: Optional[int] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[float] = None
    goal: str = "keep_fit"
    level: str = "beginner"
    equipment: str = "none"
    days_per_week: int = 3
    # Minutes to add to UTC to get the user's wall clock, e.g. 180 for UTC+3.
    timezone_offset_minutes: int = 0

    @property
    def is_complete(self) -> bool:
        """True once onboarding collected everything the planners need."""
        return all(
            value is not None
            for value in (self.age, self.height_cm, self.weight_kg)
        )


@dataclass
class Subscription:
    """A user's access tier. `expires_at` is a UTC unix timestamp."""

    user_id: int
    plan: str = "free"
    expires_at: Optional[float] = None
    source: str = ""

    def is_active_pro(self, now: Optional[float] = None) -> bool:
        if self.plan != "pro":
            return False
        if self.expires_at is None:
            # Lifetime grant (used by admin grants and tests).
            return True
        return (now if now is not None else time.time()) < self.expires_at

    def days_left(self, now: Optional[float] = None) -> Optional[int]:
        if self.expires_at is None:
            return None
        remaining = self.expires_at - (now if now is not None else time.time())
        if remaining <= 0:
            return 0
        return max(1, int(remaining // 86400))


@dataclass
class User:
    """A Telegram user plus the state the bot keeps for them."""

    id: int
    telegram_id: int
    first_name: str = ""
    username: str = ""
    created_at: float = field(default_factory=time.time)
    # Name of the pending onboarding/dialog step, empty when idle.
    state: str = ""
    profile: Profile = field(default_factory=lambda: Profile(user_id=0))
    subscription: Subscription = field(default_factory=lambda: Subscription(user_id=0))


@dataclass
class Reminder:
    """A recurring Telegram nudge scheduled in the user's local timezone."""

    id: int
    user_id: int
    kind: str
    time_local: str  # "HH:MM" in the user's timezone.
    days: str  # Comma-separated weekday numbers, Monday = 0.
    text: str = ""
    enabled: bool = True
    next_fire_at: float = 0.0
    last_fired_at: Optional[float] = None

    def weekdays(self) -> List[int]:
        result: List[int] = []
        for chunk in self.days.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                day = int(chunk)
            except ValueError:
                continue
            if 0 <= day <= 6:
                result.append(day)
        return sorted(set(result))


@dataclass
class WorkoutLog:
    """One completed (or skipped) training session."""

    id: int
    user_id: int
    created_at: float
    workout_name: str
    duration_minutes: int = 0
    difficulty: int = 0  # 1..5, self-reported.
    note: str = ""
    completed: bool = True


@dataclass
class WeightLog:
    """A body-weight measurement used for progress analytics."""

    id: int
    user_id: int
    created_at: float
    weight_kg: float


@dataclass
class Exercise:
    """A single movement inside a generated workout."""

    name: str
    sets: int
    reps: str
    rest_seconds: int
    muscle_group: str
    equipment: str = "none"

    def render(self) -> str:
        return f"{self.name} — {self.sets}×{self.reps} (отдых {self.rest_seconds} сек)"


@dataclass
class Workout:
    """One training day of a program."""

    day_index: int
    title: str
    focus: str
    exercises: List[Exercise] = field(default_factory=list)
    estimated_minutes: int = 0


@dataclass
class Program:
    """A generated multi-week training program."""

    goal: str
    level: str
    equipment: str
    weeks: int
    workouts: List[Workout] = field(default_factory=list)
    notes: str = ""
