"""Reminder scheduling: next-fire computation and the background loop."""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Optional, Sequence

from .models import Reminder

logger = logging.getLogger(__name__)

WEEKDAY_LABELS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
EVERY_DAY = "0,1,2,3,4,5,6"


def parse_time_of_day(raw: str) -> Optional[str]:
    """
    Normalise user input like "7", "7:5", "07.05", "19-30" into "HH:MM".

    Returns None when the text is not a valid time of day.
    """
    text = raw.strip().replace(".", ":").replace("-", ":").replace(" ", "")
    if not text:
        return None
    if ":" not in text:
        if not text.isdigit():
            return None
        if len(text) == 4:  # "0730"
            hours, minutes = text[:2], text[2:]
        else:
            hours, minutes = text, "0"
    else:
        hours, _, minutes = text.partition(":")
    if not hours.isdigit() or not minutes.isdigit():
        return None
    hour, minute = int(hours), int(minutes)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def parse_timezone_offset(raw: str) -> Optional[int]:
    """Parse "+3", "UTC+3", "-05:30", "3" into an offset in minutes."""
    text = raw.strip().upper().replace("UTC", "").replace("GMT", "").replace(" ", "")
    if not text:
        return None
    sign = 1
    if text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:]
    if not text:
        return None
    hours_text, _, minutes_text = text.partition(":")
    if not hours_text.isdigit():
        return None
    minutes_text = minutes_text or "0"
    if not minutes_text.isdigit():
        return None
    hours, minutes = int(hours_text), int(minutes_text)
    if hours > 14 or minutes > 59:
        return None
    return sign * (hours * 60 + minutes)


def format_timezone_offset(offset_minutes: int) -> str:
    sign = "+" if offset_minutes >= 0 else "-"
    magnitude = abs(offset_minutes)
    return f"UTC{sign}{magnitude // 60:02d}:{magnitude % 60:02d}"


def format_days(days: Sequence[int]) -> str:
    """Human label for a weekday set, e.g. "ежедневно" or "Пн, Ср, Пт"."""
    ordered = sorted(set(days))
    if not ordered:
        return "—"
    if ordered == [0, 1, 2, 3, 4, 5, 6]:
        return "ежедневно"
    if ordered == [0, 1, 2, 3, 4]:
        return "по будням"
    if ordered == [5, 6]:
        return "по выходным"
    return ", ".join(WEEKDAY_LABELS[day] for day in ordered)


def parse_days(raw: str) -> Optional[List[int]]:
    """
    Parse a weekday selection: "ежедневно", "будни", "выходные", "1,3,5"
    (1 = Monday, matching how people count days out loud), or "пн,ср,пт".
    """
    text = raw.strip().lower()
    if not text:
        return None
    if text in ("ежедневно", "каждый день", "все", "all", "daily"):
        return [0, 1, 2, 3, 4, 5, 6]
    if text in ("будни", "по будням", "weekdays"):
        return [0, 1, 2, 3, 4]
    if text in ("выходные", "по выходным", "weekend", "weekends"):
        return [5, 6]

    aliases = {
        "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    result: List[int] = []
    for chunk in text.replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isdigit():
            number = int(chunk)
            if not 1 <= number <= 7:
                return None
            result.append(number - 1)
            continue
        key = chunk[:3] if chunk[:3] in aliases else chunk[:2]
        if key not in aliases:
            return None
        result.append(aliases[key])
    if not result:
        return None
    return sorted(set(result))


def compute_next_fire(
    time_local: str,
    weekdays: Sequence[int],
    offset_minutes: int,
    after: Optional[float] = None,
) -> float:
    """
    Next UTC timestamp at which a reminder should fire.

    Args:
        time_local: "HH:MM" in the user's own timezone.
        weekdays: Days the reminder is active, Monday = 0. Empty means daily.
        offset_minutes: Minutes to add to UTC to get the user's wall clock.
        after: Compute the first fire strictly after this UTC timestamp
            (defaults to now).

    Returns:
        A UTC unix timestamp, always strictly in the future relative to `after`.
    """
    moment = after if after is not None else time.time()
    active = sorted(set(weekdays)) or [0, 1, 2, 3, 4, 5, 6]

    hour_text, _, minute_text = time_local.partition(":")
    hour, minute = int(hour_text), int(minute_text or 0)

    local_now = datetime.fromtimestamp(moment, tz=timezone.utc) + timedelta(
        minutes=offset_minutes
    )
    # Search a full week plus one day so a DST-free fixed offset always lands.
    for delta_days in range(0, 9):
        candidate = (local_now + timedelta(days=delta_days)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate <= local_now:
            continue
        if candidate.weekday() not in active:
            continue
        return (candidate - timedelta(minutes=offset_minutes)).timestamp()

    # Unreachable for a non-empty weekday set, but keep the contract of always
    # returning a future timestamp.
    return moment + 86400


def next_fire_for(reminder: Reminder, offset_minutes: int, after: Optional[float] = None) -> float:
    """Convenience wrapper around `compute_next_fire` for a stored reminder."""
    return compute_next_fire(
        reminder.time_local, reminder.weekdays(), offset_minutes, after
    )


class ReminderScheduler:
    """
    Polls the database for due reminders and hands them to a delivery callback.

    Delivery and rescheduling are separated so the bot can decide what a
    reminder message looks like, while this class only owns the timing.
    """

    def __init__(
        self,
        storage,
        deliver: Callable[[Reminder], Awaitable[bool]],
        tick_seconds: int = 30,
    ):
        self.storage = storage
        self.deliver = deliver
        self.tick_seconds = max(5, tick_seconds)
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def run_once(self, now: Optional[float] = None) -> int:
        """
        Fire everything that is due. Returns the number of reminders delivered.

        A reminder is always rescheduled, even when delivery fails (the user
        may have blocked the bot), so a single failure cannot wedge the loop.
        """
        moment = now if now is not None else time.time()
        delivered = 0
        for reminder in self.storage.due_reminders(moment):
            user = self.storage.get_user(reminder.user_id)
            offset = user.profile.timezone_offset_minutes if user else 0
            try:
                if user is not None and await self.deliver(reminder):
                    delivered += 1
            except Exception:  # pragma: no cover - defensive, logged and skipped
                logger.exception("Failed to deliver reminder %s", reminder.id)
            finally:
                self.storage.mark_reminder_fired(
                    reminder.id,
                    moment,
                    next_fire_for(reminder, offset, moment),
                )
        return delivered

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:  # pragma: no cover - keep the loop alive
                logger.exception("Reminder tick failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.tick_seconds)
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass
            self._task = None
