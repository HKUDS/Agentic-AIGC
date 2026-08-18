"""Inline keyboard builders.

Callback data follows the `action:argument` convention and must stay under
Telegram's 64-byte limit, so arguments are short identifiers, never free text.
"""

from typing import Any, Dict, List, Optional, Sequence

from .models import Reminder
from .scheduler import format_days


def _keyboard(rows: Sequence[Sequence[Dict[str, str]]]) -> Dict[str, Any]:
    return {"inline_keyboard": [list(row) for row in rows]}


def _button(text: str, data: str) -> Dict[str, str]:
    return {"text": text, "callback_data": data}


def main_menu(is_pro: bool) -> Dict[str, Any]:
    """Main menu. Pro-only entries are marked with a lock for free users."""
    lock = "" if is_pro else " 🔒"
    return _keyboard(
        [
            [_button("🏋️ Тренировка на сегодня", "menu:today")],
            [_button("📋 Моя программа", "menu:program"), _button("✅ Отметить", "menu:log")],
            [_button("💬 Спросить тренера", "menu:ask")],
            [_button("⏰ Напоминания", "menu:reminders")],
            [
                _button(f"🍎 Питание{lock}", "menu:nutrition"),
                _button(f"📊 Прогресс{lock}", "menu:progress"),
            ],
            [_button("⚙️ Профиль", "menu:profile"), _button("⭐ Подписка", "menu:subscription")],
        ]
    )


def gender_keyboard() -> Dict[str, Any]:
    return _keyboard(
        [
            [
                _button("Мужской", "onb_gender:male"),
                _button("Женский", "onb_gender:female"),
            ],
            [_button("Не указывать", "onb_gender:other")],
        ]
    )


def goal_keyboard(prefix: str = "onb_goal") -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("🔥 Снизить вес", f"{prefix}:lose_weight")],
            [_button("💪 Набрать мышцы", f"{prefix}:build_muscle")],
            [_button("🧘 Поддержать форму", f"{prefix}:keep_fit")],
            [_button("🏃 Выносливость", f"{prefix}:endurance")],
        ]
    )


def level_keyboard(prefix: str = "onb_level") -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("Новичок", f"{prefix}:beginner")],
            [_button("Средний", f"{prefix}:intermediate")],
            [_button("Продвинутый", f"{prefix}:advanced")],
        ]
    )


def equipment_keyboard(prefix: str = "onb_equipment") -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("🏠 Без оборудования", f"{prefix}:none")],
            [_button("🏋️ Гантели / резинки дома", f"{prefix}:home")],
            [_button("🏢 Тренажёрный зал", f"{prefix}:gym")],
        ]
    )


def days_keyboard(prefix: str = "onb_days") -> Dict[str, Any]:
    return _keyboard(
        [
            [
                _button("2", f"{prefix}:2"),
                _button("3", f"{prefix}:3"),
                _button("4", f"{prefix}:4"),
                _button("5", f"{prefix}:5"),
            ]
        ]
    )


def onboarding_reminder_keyboard() -> Dict[str, Any]:
    """Time picker shown at the end of onboarding."""
    return _keyboard(
        [
            [
                _button("07:00", "onb_reminder:07:00"),
                _button("08:30", "onb_reminder:08:30"),
            ],
            [
                _button("12:30", "onb_reminder:12:30"),
                _button("18:00", "onb_reminder:18:00"),
            ],
            [
                _button("19:00", "onb_reminder:19:00"),
                _button("20:30", "onb_reminder:20:30"),
            ],
            [_button("🕐 Другое время", "onb_reminder:custom")],
            [_button("Не напоминать", "onb_reminder:skip")],
        ]
    )


def difficulty_keyboard() -> Dict[str, Any]:
    return _keyboard(
        [
            [
                _button("1 · легко", "log_difficulty:1"),
                _button("2", "log_difficulty:2"),
                _button("3", "log_difficulty:3"),
                _button("4", "log_difficulty:4"),
                _button("5 · на пределе", "log_difficulty:5"),
            ]
        ]
    )


def today_keyboard() -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("✅ Выполнил", "menu:log")],
            [_button("🔄 Другая тренировка", "today:next")],
            [_button("💬 Спросить тренера", "menu:ask"), _button("◀️ Меню", "menu:main")],
        ]
    )


def reminders_keyboard(reminders: Sequence[Reminder], can_add: bool) -> Dict[str, Any]:
    """List of reminders with per-item toggle and delete controls."""
    rows: List[List[Dict[str, str]]] = []
    for reminder in reminders:
        state = "🔔" if reminder.enabled else "🔕"
        label = f"{state} {reminder.time_local} · {format_days(reminder.weekdays())}"
        rows.append(
            [
                _button(label, f"rem_toggle:{reminder.id}"),
                _button("🗑", f"rem_delete:{reminder.id}"),
            ]
        )
    if can_add:
        rows.append([_button("➕ Добавить напоминание", "rem_add:start")])
    rows.append([_button("◀️ Меню", "menu:main")])
    return _keyboard(rows)


def reminder_kind_keyboard(is_pro: bool) -> Dict[str, Any]:
    rows = [[_button("🏋️ Тренировка", "rem_kind:workout")]]
    if is_pro:
        rows.extend(
            [
                [_button("💧 Вода", "rem_kind:water")],
                [_button("⚖️ Взвешивание", "rem_kind:weigh_in")],
                [_button("✏️ Свой текст", "rem_kind:custom")],
            ]
        )
    rows.append([_button("◀️ Назад", "menu:reminders")])
    return _keyboard(rows)


def subscription_keyboard(price: int, is_pro: bool) -> Dict[str, Any]:
    rows: List[List[Dict[str, str]]] = []
    label = "⭐ Продлить" if is_pro else "⭐ Оформить Pro"
    rows.append([_button(f"{label} — {price} Stars", "pay:stars")])
    rows.append([_button("🎁 У меня есть промокод", "pay:promo")])
    rows.append([_button("◀️ Меню", "menu:main")])
    return _keyboard(rows)


def upgrade_keyboard() -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("⭐ Открыть Pro", "menu:subscription")],
            [_button("◀️ Меню", "menu:main")],
        ]
    )


def profile_keyboard() -> Dict[str, Any]:
    return _keyboard(
        [
            [_button("🎯 Цель", "edit:goal"), _button("📈 Уровень", "edit:level")],
            [_button("🏋️ Инвентарь", "edit:equipment"), _button("📅 Дни", "edit:days")],
            [_button("⚖️ Вес", "edit:weight"), _button("🕐 Часовой пояс", "edit:timezone")],
            [_button("◀️ Меню", "menu:main")],
        ]
    )


def back_to_menu(extra: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Any]:
    rows: List[List[Dict[str, str]]] = []
    if extra:
        rows.append(list(extra))
    rows.append([_button("◀️ Меню", "menu:main")])
    return _keyboard(rows)
