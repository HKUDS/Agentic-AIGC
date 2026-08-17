"""Progress statistics and data export."""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from .models import WeightLog, WorkoutLog

DAY = 86400.0


@dataclass
class ProgressReport:
    """Aggregated training statistics for a user."""

    total_workouts: int
    workouts_last_7_days: int
    workouts_last_30_days: int
    minutes_last_30_days: int
    current_streak_weeks: int
    average_difficulty: float
    weight_start: Optional[float]
    weight_latest: Optional[float]
    weight_delta: Optional[float]
    best_week: int


def _local_date(timestamp: float, offset_minutes: int) -> datetime:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        + timedelta(minutes=offset_minutes)
    ).replace(hour=0, minute=0, second=0, microsecond=0)


def training_streak_weeks(
    logs: Sequence[WorkoutLog],
    offset_minutes: int = 0,
    now: Optional[float] = None,
    minimum_per_week: int = 1,
) -> int:
    """
    Number of consecutive weeks (counting back from this one) with at least
    `minimum_per_week` completed workouts.

    The current week is allowed to be still empty without breaking the streak —
    it simply is not counted yet.
    """
    moment = now if now is not None else time.time()
    if not logs:
        return 0

    today = _local_date(moment, offset_minutes)
    week_start = today - timedelta(days=today.weekday())

    counts = {}
    for log in logs:
        if not log.completed:
            continue
        log_day = _local_date(log.created_at, offset_minutes)
        log_week = log_day - timedelta(days=log_day.weekday())
        counts[log_week] = counts.get(log_week, 0) + 1

    streak = 0
    cursor = week_start
    if counts.get(cursor, 0) < minimum_per_week:
        # Current week not finished yet: start counting from the previous one.
        cursor -= timedelta(days=7)
    while counts.get(cursor, 0) >= minimum_per_week:
        streak += 1
        cursor -= timedelta(days=7)
    return streak


def build_report(
    workouts: Sequence[WorkoutLog],
    weights: Sequence[WeightLog],
    offset_minutes: int = 0,
    now: Optional[float] = None,
) -> ProgressReport:
    """Aggregate raw logs into the numbers shown in the analytics screen."""
    moment = now if now is not None else time.time()
    completed = [log for log in workouts if log.completed]

    last_7 = [log for log in completed if moment - log.created_at <= 7 * DAY]
    last_30 = [log for log in completed if moment - log.created_at <= 30 * DAY]

    rated = [log.difficulty for log in completed if log.difficulty > 0]
    average_difficulty = sum(rated) / len(rated) if rated else 0.0

    # Weight logs arrive newest-first from storage.
    ordered_weights = sorted(weights, key=lambda log: log.created_at)
    weight_start = ordered_weights[0].weight_kg if ordered_weights else None
    weight_latest = ordered_weights[-1].weight_kg if ordered_weights else None
    weight_delta = (
        round(weight_latest - weight_start, 1)
        if weight_start is not None and weight_latest is not None
        else None
    )

    weekly_counts = {}
    for log in completed:
        day = _local_date(log.created_at, offset_minutes)
        week = day - timedelta(days=day.weekday())
        weekly_counts[week] = weekly_counts.get(week, 0) + 1

    return ProgressReport(
        total_workouts=len(completed),
        workouts_last_7_days=len(last_7),
        workouts_last_30_days=len(last_30),
        minutes_last_30_days=sum(log.duration_minutes for log in last_30),
        current_streak_weeks=training_streak_weeks(completed, offset_minutes, moment),
        average_difficulty=round(average_difficulty, 1),
        weight_start=weight_start,
        weight_latest=weight_latest,
        weight_delta=weight_delta,
        best_week=max(weekly_counts.values()) if weekly_counts else 0,
    )


def render_report(report: ProgressReport) -> str:
    """Format a progress report for a Telegram message."""
    lines = [
        "<b>Аналитика прогресса</b>",
        "",
        f"Всего тренировок: <b>{report.total_workouts}</b>",
        f"За 7 дней: <b>{report.workouts_last_7_days}</b>",
        f"За 30 дней: <b>{report.workouts_last_30_days}</b> "
        f"({report.minutes_last_30_days} мин)",
        f"Лучшая неделя: <b>{report.best_week}</b> тренировки",
        f"Серия недель подряд: <b>{report.current_streak_weeks}</b>",
    ]
    if report.average_difficulty:
        lines.append(f"Средняя тяжесть: <b>{report.average_difficulty}</b> из 5")

    if report.weight_start is not None and report.weight_latest is not None:
        delta = report.weight_delta or 0.0
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
        lines.extend(
            [
                "",
                f"Вес: {report.weight_start} кг → <b>{report.weight_latest} кг</b> "
                f"({arrow} {abs(delta)} кг)",
            ]
        )
    else:
        lines.extend(["", "<i>Отправь /weight 78.5, чтобы отслеживать динамику веса.</i>"])

    lines.extend(["", _verdict(report)])
    return "\n".join(lines)


def _verdict(report: ProgressReport) -> str:
    if report.workouts_last_7_days >= 3:
        return "Отличная неделя — держи темп и добавляй нагрузку понемногу."
    if report.workouts_last_7_days > 0:
        return "Неделя идёт. Ещё одна тренировка — и план выполнен."
    if report.total_workouts == 0:
        return "Пока пусто. Отметь первую тренировку — с неё начинается статистика."
    return "На этой неделе тренировок не было. Начни с короткой — 20 минут тоже считаются."


def export_payload(
    profile_dict: dict,
    workouts: Sequence[WorkoutLog],
    weights: Sequence[WeightLog],
    program_text: str = "",
) -> bytes:
    """Serialise everything the bot knows about a user into a JSON file."""
    data = {
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "profile": profile_dict,
        "program": program_text,
        "workouts": [
            {
                "date": datetime.fromtimestamp(log.created_at, tz=timezone.utc).isoformat(),
                "name": log.workout_name,
                "duration_minutes": log.duration_minutes,
                "difficulty": log.difficulty,
                "note": log.note,
                "completed": log.completed,
            }
            for log in sorted(workouts, key=lambda item: item.created_at)
        ],
        "weights": [
            {
                "date": datetime.fromtimestamp(log.created_at, tz=timezone.utc).isoformat(),
                "weight_kg": log.weight_kg,
            }
            for log in sorted(weights, key=lambda item: item.created_at)
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def weekly_digest(workouts: Sequence[WorkoutLog], now: Optional[float] = None) -> str:
    """Short weekly summary used by the Pro weekly report reminder."""
    moment = now if now is not None else time.time()
    week: List[WorkoutLog] = [
        log for log in workouts if log.completed and moment - log.created_at <= 7 * DAY
    ]
    if not week:
        return (
            "Итоги недели: тренировок не отмечено. Давай запланируем ближайшую — "
            "нажми «Тренировка на сегодня»."
        )
    minutes = sum(log.duration_minutes for log in week)
    return (
        f"Итоги недели: <b>{len(week)}</b> тренировки, {minutes} минут работы. "
        "Продолжай — на следующей неделе добавь по одному повторению в подходе."
    )
