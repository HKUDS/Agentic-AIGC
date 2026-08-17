"""Telegram handlers, dialog state machine, and the polling loop."""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from . import keyboards, texts
from .analytics import build_report, export_payload, render_report
from .coach import AICoach, usage_summary
from .config import Config
from .models import Reminder, User
from .nutrition import build_nutrition_plan, render_nutrition_plan
from .payments import (
    PRO_PAYLOAD,
    activate_pro,
    grant_lifetime_pro,
    handle_successful_payment,
    redeem_promo,
    send_pro_invoice,
)
from .scheduler import (
    EVERY_DAY,
    ReminderScheduler,
    compute_next_fire,
    format_days,
    format_timezone_offset,
    next_fire_for,
    parse_days,
    parse_time_of_day,
    parse_timezone_offset,
)
from .storage import Storage
from .subscription import (
    FEATURE_ANALYTICS,
    FEATURE_CUSTOM_REMINDERS,
    FEATURE_EXPORT,
    FEATURE_NUTRITION,
    ai_messages_left,
    plan_for,
    reminder_slots_left,
)
from .telegram_api import TelegramClient, TelegramError, escape_html
from .workouts import (
    EQUIPMENT_LABELS,
    GENDER_LABELS,
    GOAL_LABELS,
    LEVEL_LABELS,
    generate_program,
    render_program,
    render_workout,
)

logger = logging.getLogger(__name__)

# Dialog steps collected during onboarding, in order.
ONBOARDING_ORDER = (
    "gender",
    "age",
    "height",
    "weight",
    "goal",
    "level",
    "equipment",
    "days",
    "timezone",
)

BOT_COMMANDS = [
    {"command": "menu", "description": "Главное меню"},
    {"command": "today", "description": "Тренировка на сегодня"},
    {"command": "program", "description": "Вся программа"},
    {"command": "ask", "description": "Спросить ИИ-тренера"},
    {"command": "log", "description": "Отметить тренировку"},
    {"command": "weight", "description": "Записать вес"},
    {"command": "reminders", "description": "Напоминания"},
    {"command": "nutrition", "description": "Питание и БЖУ (Pro)"},
    {"command": "progress", "description": "Аналитика прогресса (Pro)"},
    {"command": "subscribe", "description": "Подписка Pro"},
    {"command": "profile", "description": "Профиль"},
    {"command": "help", "description": "Справка"},
]

# Callback data and dialog state both need a separator; states use "|" because
# stored payloads contain times like "19:00".
STATE_SEP = "|"


class FitnessBot:
    """Routes Telegram updates to features and owns the dialog state machine."""

    def __init__(
        self,
        config: Config,
        storage: Storage,
        client: TelegramClient,
        coach: AICoach,
    ):
        self.config = config
        self.storage = storage
        self.client = client
        self.coach = coach
        self.scheduler = ReminderScheduler(
            storage, self._deliver_reminder, config.reminder_tick_seconds
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        try:
            await self.client.set_my_commands(BOT_COMMANDS)
        except TelegramError as error:
            logger.warning("Could not publish command list: %s", error)

    async def run(self) -> None:
        """Long-poll Telegram until cancelled."""
        await self.setup()
        self.scheduler.start()
        offset = self._load_offset()
        logger.info("Bot started, polling from offset %s", offset)

        try:
            while True:
                try:
                    updates = await self.client.get_updates(
                        offset=offset, timeout=self.config.poll_timeout_seconds
                    )
                except TelegramError as error:
                    logger.error("getUpdates failed: %s", error)
                    await asyncio.sleep(5)
                    continue

                for update in updates:
                    offset = int(update["update_id"]) + 1
                    try:
                        await self.handle_update(update)
                    except Exception:
                        logger.exception("Failed to handle update %s", update.get("update_id"))
                    finally:
                        # Persist after each update so a crash cannot replay a
                        # payment or re-answer a question already handled.
                        self.storage.set_meta("update_offset", str(offset))
        finally:
            await self.scheduler.stop()

    def _load_offset(self) -> Optional[int]:
        raw = self.storage.get_meta("update_offset")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Update routing
    # ------------------------------------------------------------------

    async def handle_update(self, update: Dict[str, Any]) -> None:
        if "message" in update:
            await self._handle_message(update["message"])
        elif "callback_query" in update:
            await self._handle_callback(update["callback_query"])
        elif "pre_checkout_query" in update:
            await self._handle_pre_checkout(update["pre_checkout_query"])

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        chat_id = chat.get("id")
        if chat_id is None or sender.get("is_bot"):
            return

        user = self.storage.get_or_create_user(
            telegram_id=int(sender["id"]),
            first_name=sender.get("first_name", ""),
            username=sender.get("username", ""),
        )

        if "successful_payment" in message:
            await self._on_successful_payment(chat_id, user, message["successful_payment"])
            return

        text = (message.get("text") or "").strip()
        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(chat_id, user, text)
            return

        if user.state:
            await self._handle_state_input(chat_id, user, text)
            return

        # Free text outside a dialog is treated as a question to the coach.
        await self._answer_question(chat_id, user, text)

    async def _handle_command(self, chat_id: int, user: User, text: str) -> None:
        command, _, argument = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        argument = argument.strip()

        handlers = {
            "/start": self._cmd_start,
            "/menu": self._cmd_menu,
            "/help": self._cmd_help,
            "/today": self._cmd_today,
            "/program": self._cmd_program,
            "/ask": self._cmd_ask,
            "/log": self._cmd_log,
            "/weight": self._cmd_weight,
            "/reminders": self._cmd_reminders,
            "/nutrition": self._cmd_nutrition,
            "/progress": self._cmd_progress,
            "/export": self._cmd_export,
            "/subscribe": self._cmd_subscribe,
            "/promo": self._cmd_promo,
            "/profile": self._cmd_profile,
            "/cancel": self._cmd_cancel,
            "/stats": self._cmd_stats,
            "/grant": self._cmd_grant,
        }

        handler = handlers.get(command)
        if handler is None:
            await self._send(chat_id, texts.HELP)
            return
        await handler(chat_id, user, argument)

    async def _handle_callback(self, query: Dict[str, Any]) -> None:
        sender = query.get("from") or {}
        message = query.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        data = query.get("data") or ""
        query_id = query.get("id")

        if chat_id is None:
            return

        user = self.storage.get_or_create_user(
            telegram_id=int(sender["id"]),
            first_name=sender.get("first_name", ""),
            username=sender.get("username", ""),
        )

        action, _, argument = data.partition(":")
        try:
            await self._dispatch_callback(chat_id, user, action, argument)
        finally:
            if query_id:
                try:
                    await self.client.answer_callback_query(query_id)
                except TelegramError:
                    pass

    async def _dispatch_callback(
        self, chat_id: int, user: User, action: str, argument: str
    ) -> None:
        if action == "menu":
            await self._menu_action(chat_id, user, argument)
        elif action.startswith("onb_"):
            await self._onboarding_choice(chat_id, user, action[4:], argument)
        elif action == "edit":
            await self._edit_field(chat_id, user, argument)
        elif action.startswith("set_"):
            await self._apply_profile_edit(chat_id, user, action[4:], argument)
        elif action == "today":
            await self._cmd_today(chat_id, user, "next" if argument == "next" else "")
        elif action == "log_difficulty":
            await self._save_workout_log(chat_id, user, argument)
        elif action == "rem_add":
            await self._reminder_add_start(chat_id, user)
        elif action == "rem_kind":
            await self._reminder_kind_chosen(chat_id, user, argument)
        elif action == "rem_toggle":
            await self._reminder_toggle(chat_id, user, argument)
        elif action == "rem_delete":
            await self._reminder_delete(chat_id, user, argument)
        elif action == "pay":
            await self._payment_action(chat_id, user, argument)

    async def _menu_action(self, chat_id: int, user: User, argument: str) -> None:
        actions = {
            "main": self._cmd_menu,
            "today": self._cmd_today,
            "program": self._cmd_program,
            "log": self._cmd_log,
            "ask": self._cmd_ask,
            "reminders": self._cmd_reminders,
            "nutrition": self._cmd_nutrition,
            "progress": self._cmd_progress,
            "profile": self._cmd_profile,
            "subscription": self._cmd_subscribe,
        }
        handler = actions.get(argument)
        if handler is not None:
            await handler(chat_id, user, "")

    async def _handle_pre_checkout(self, query: Dict[str, Any]) -> None:
        """Approve the checkout. Telegram gives ~10 seconds to answer."""
        payload = query.get("invoice_payload", "")
        ok = payload == PRO_PAYLOAD
        await self.client.answer_pre_checkout_query(
            query["id"],
            ok=ok,
            error_message="" if ok else "Счёт устарел, оформи подписку заново: /subscribe",
        )

    # ------------------------------------------------------------------
    # Onboarding
    # ------------------------------------------------------------------

    async def _cmd_start(self, chat_id: int, user: User, argument: str = "") -> None:
        await self._send(chat_id, texts.WELCOME.format(name=user.first_name or "друг"))
        await self._send_onboarding_step(chat_id, user, ONBOARDING_ORDER[0])

    async def _send_onboarding_step(self, chat_id: int, user: User, step: str) -> None:
        prompts = {
            "gender": (texts.ASK_GENDER, keyboards.gender_keyboard()),
            "age": (texts.ASK_AGE, None),
            "height": (texts.ASK_HEIGHT, None),
            "weight": (texts.ASK_WEIGHT, None),
            "goal": (texts.ASK_GOAL, keyboards.goal_keyboard()),
            "level": (texts.ASK_LEVEL, keyboards.level_keyboard()),
            "equipment": (texts.ASK_EQUIPMENT, keyboards.equipment_keyboard()),
            "days": (texts.ASK_DAYS, keyboards.days_keyboard()),
            "timezone": (texts.ASK_TIMEZONE, None),
        }
        prompt, markup = prompts[step]
        self.storage.set_state(user.id, f"onb_{step}")
        await self._send(chat_id, prompt, markup)

    async def _advance_onboarding(self, chat_id: int, user: User, current_step: str) -> None:
        index = ONBOARDING_ORDER.index(current_step)
        if index + 1 < len(ONBOARDING_ORDER):
            await self._send_onboarding_step(chat_id, user, ONBOARDING_ORDER[index + 1])
            return

        self.storage.set_state(user.id, "")
        await self._send(chat_id, texts.ONBOARDING_DONE)
        fresh = self.storage.get_user(user.id) or user
        await self._cmd_today(chat_id, fresh, "")

    async def _onboarding_choice(
        self, chat_id: int, user: User, step: str, value: str
    ) -> None:
        """Handle a button press during onboarding."""
        fields = {
            "gender": ("gender", str),
            "goal": ("goal", str),
            "level": ("level", str),
            "equipment": ("equipment", str),
            "days": ("days_per_week", int),
        }
        if step not in fields:
            return
        column, caster = fields[step]
        try:
            self.storage.update_profile(user.id, **{column: caster(value)})
        except ValueError:
            return

        # Buttons can be pressed on an old message after onboarding finished —
        # in that case just confirm the change instead of restarting the chain.
        if user.state != f"onb_{step}":
            await self._send(chat_id, "Обновил профиль ✅")
            await self._cmd_profile(chat_id, self.storage.get_user(user.id) or user, "")
            return
        await self._advance_onboarding(chat_id, user, step)

    # ------------------------------------------------------------------
    # Dialog state input
    # ------------------------------------------------------------------

    async def _handle_state_input(self, chat_id: int, user: User, text: str) -> None:
        state, _, payload = user.state.partition(STATE_SEP)

        if state == "onb_age":
            await self._on_age(chat_id, user, text, "onboarding")
        elif state == "onb_height":
            await self._on_height(chat_id, user, text)
        elif state == "onb_weight":
            await self._on_weight(chat_id, user, text, "onboarding")
        elif state == "onb_timezone":
            await self._on_timezone(chat_id, user, text, "onboarding")
        elif state == "edit_weight":
            await self._on_weight(chat_id, user, text, "edit")
        elif state == "edit_timezone":
            await self._on_timezone(chat_id, user, text, "edit")
        elif state == "ask":
            self.storage.set_state(user.id, "")
            await self._answer_question(chat_id, user, text)
        elif state == "promo":
            self.storage.set_state(user.id, "")
            await self._cmd_promo(chat_id, user, text)
        elif state == "rem_time":
            await self._reminder_time_entered(chat_id, user, payload, text)
        elif state == "rem_days":
            await self._reminder_days_entered(chat_id, user, payload, text)
        elif state == "rem_text":
            await self._reminder_text_entered(chat_id, user, payload, text)
        elif state == "log_difficulty" and text.strip() in ("1", "2", "3", "4", "5"):
            # The rating is normally a button press, but typing the digit works too.
            await self._save_workout_log(chat_id, user, text.strip())
        elif state.startswith("onb_"):
            # A button step answered with text: re-send the step instead of
            # silently dropping the user out of onboarding.
            await self._send_onboarding_step(chat_id, user, state[4:])
        else:
            self.storage.set_state(user.id, "")
            await self._answer_question(chat_id, user, text)

    async def _on_age(self, chat_id: int, user: User, text: str, mode: str) -> None:
        try:
            age = int(text.strip())
        except ValueError:
            await self._send(chat_id, texts.INVALID_NUMBER.format(example="29"))
            return
        if not 14 <= age <= 100:
            await self._send(chat_id, texts.INVALID_AGE)
            return
        self.storage.update_profile(user.id, age=age)
        if mode == "onboarding":
            await self._advance_onboarding(chat_id, user, "age")

    async def _on_height(self, chat_id: int, user: User, text: str) -> None:
        try:
            height = int(float(text.strip().replace(",", ".")))
        except ValueError:
            await self._send(chat_id, texts.INVALID_NUMBER.format(example="178"))
            return
        if not 120 <= height <= 230:
            await self._send(chat_id, texts.INVALID_HEIGHT)
            return
        self.storage.update_profile(user.id, height_cm=height)
        await self._advance_onboarding(chat_id, user, "height")

    async def _on_weight(self, chat_id: int, user: User, text: str, mode: str) -> None:
        try:
            weight = float(text.strip().replace(",", "."))
        except ValueError:
            await self._send(chat_id, texts.INVALID_NUMBER.format(example="78.5"))
            return
        if not 35 <= weight <= 300:
            await self._send(chat_id, texts.INVALID_WEIGHT)
            return
        self.storage.update_profile(user.id, weight_kg=weight)
        self.storage.add_weight_log(user.id, weight)
        if mode == "onboarding":
            await self._advance_onboarding(chat_id, user, "weight")
        else:
            self.storage.set_state(user.id, "")
            await self._send(chat_id, texts.WEIGHT_SAVED.format(weight=weight))

    async def _on_timezone(self, chat_id: int, user: User, text: str, mode: str) -> None:
        offset = parse_timezone_offset(text)
        if offset is None:
            await self._send(chat_id, texts.INVALID_TIMEZONE)
            return
        self.storage.update_profile(user.id, timezone_offset_minutes=offset)
        # Stored fire times are absolute, so a timezone change must move them.
        self.storage.reschedule_user_reminders(
            user.id, lambda reminder: next_fire_for(reminder, offset)
        )
        if mode == "onboarding":
            await self._advance_onboarding(chat_id, user, "timezone")
        else:
            self.storage.set_state(user.id, "")
            await self._send(
                chat_id,
                f"Часовой пояс обновлён: <b>{format_timezone_offset(offset)}</b>.",
            )

    # ------------------------------------------------------------------
    # Core features (free tier)
    # ------------------------------------------------------------------

    async def _cmd_menu(self, chat_id: int, user: User, argument: str = "") -> None:
        self.storage.set_state(user.id, "")
        await self._send(chat_id, texts.MENU, keyboards.main_menu(self._is_pro(user)))

    async def _cmd_help(self, chat_id: int, user: User, argument: str = "") -> None:
        await self._send(chat_id, texts.HELP)

    async def _cmd_cancel(self, chat_id: int, user: User, argument: str = "") -> None:
        if user.state:
            self.storage.set_state(user.id, "")
            await self._send(chat_id, texts.CANCELLED)
        else:
            await self._send(chat_id, texts.NOTHING_TO_CANCEL)

    async def _cmd_today(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self._require_profile(user):
            await self._send(chat_id, texts.PROFILE_INCOMPLETE)
            return

        program = self._program_for(user)
        index_key = f"day_index:{user.id}"
        try:
            index = int(self.storage.get_meta(index_key) or 0)
        except ValueError:
            index = 0
        if argument == "next":
            index += 1
        index %= len(program.workouts)
        self.storage.set_meta(index_key, str(index))

        workout = program.workouts[index]
        self.storage.set_meta(f"last_workout:{user.id}", workout.title)
        self.storage.set_meta(f"last_duration:{user.id}", str(workout.estimated_minutes))

        await self._send(
            chat_id,
            render_workout(workout) + "\n\n" + texts.DISCLAIMER_SHORT,
            keyboards.today_keyboard(),
        )

    async def _cmd_program(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self._require_profile(user):
            await self._send(chat_id, texts.PROFILE_INCOMPLETE)
            return

        program = self._program_for(user)
        text = render_program(program)

        if self._is_pro(user):
            # Pro gets the LLM pass over the generated draft.
            text = await self.coach.personalise_program(user, text)
        else:
            text += (
                "\n\n🔒 <i>В Pro программа расписана на 4 недели с прогрессией и "
                "адаптируется под твои отчёты — /subscribe</i>"
            )
        await self._send(chat_id, text, keyboards.back_to_menu())

    async def _cmd_log(self, chat_id: int, user: User, argument: str = "") -> None:
        self.storage.set_state(user.id, "log_difficulty")
        await self._send(chat_id, texts.LOG_ASK_DIFFICULTY, keyboards.difficulty_keyboard())

    async def _save_workout_log(self, chat_id: int, user: User, argument: str) -> None:
        try:
            difficulty = max(1, min(5, int(argument)))
        except ValueError:
            difficulty = 0

        name = self.storage.get_meta(f"last_workout:{user.id}") or "Тренировка"
        try:
            duration = int(self.storage.get_meta(f"last_duration:{user.id}") or 0)
        except ValueError:
            duration = 0

        self.storage.add_workout_log(
            user.id, workout_name=name, duration_minutes=duration, difficulty=difficulty
        )
        self.storage.set_state(user.id, "")

        logs = self.storage.list_workout_logs(user.id)
        week = sum(1 for log in logs if time.time() - log.created_at <= 7 * 86400)
        await self._send(
            chat_id,
            texts.LOG_SAVED.format(total=len(logs), week=week),
            keyboards.back_to_menu(),
        )

    async def _cmd_weight(self, chat_id: int, user: User, argument: str = "") -> None:
        if not argument:
            self.storage.set_state(user.id, "edit_weight")
            await self._send(chat_id, texts.WEIGHT_USAGE)
            return
        await self._on_weight(chat_id, user, argument, "edit")

    async def _cmd_ask(self, chat_id: int, user: User, argument: str = "") -> None:
        if argument:
            await self._answer_question(chat_id, user, argument)
            return
        self.storage.set_state(user.id, "ask")
        await self._send(chat_id, texts.ASK_PROMPT)

    async def _answer_question(self, chat_id: int, user: User, question: str) -> None:
        subscription = user.subscription
        offset = user.profile.timezone_offset_minutes
        used = self.storage.ai_messages_used_today(user.id, offset)
        left = ai_messages_left(subscription, used)

        if left is not None and left <= 0:
            await self._send(chat_id, texts.ASK_LIMIT_REACHED, keyboards.upgrade_keyboard())
            return

        history = self.storage.recent_chat(user.id, limit=8)
        context_note = self._history_note(user)
        answer = await self.coach.answer(user, question, history, context_note)

        used = self.storage.record_ai_message(user.id, offset)
        self.storage.append_chat_message(user.id, "user", question)
        self.storage.append_chat_message(user.id, "assistant", answer)

        plan = plan_for(subscription)
        limit = plan.ai_messages_per_day if plan.ai_messages_per_day > 0 else None
        await self._send(chat_id, answer + usage_summary(used, limit))

    def _history_note(self, user: User) -> str:
        """Extra LLM context describing recent training activity."""
        logs = self.storage.list_workout_logs(user.id, limit=5)
        if not logs:
            return "История тренировок: пока пусто."
        parts = []
        for log in logs:
            when = datetime.fromtimestamp(log.created_at, tz=timezone.utc).strftime("%d.%m")
            parts.append(f"{when} — {log.workout_name} (тяжесть {log.difficulty}/5)")
        return "Последние тренировки: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    async def _cmd_reminders(self, chat_id: int, user: User, argument: str = "") -> None:
        self.storage.set_state(user.id, "")
        reminders = self.storage.list_reminders(user.id)
        slots = reminder_slots_left(user.subscription, len(reminders))

        if not reminders:
            text = texts.REMINDERS_EMPTY
        else:
            lines = ["<b>Твои напоминания</b>", ""]
            for reminder in reminders:
                state = "включено" if reminder.enabled else "выключено"
                lines.append(
                    f"• {reminder.time_local} · {format_days(reminder.weekdays())} · "
                    f"{self._reminder_title(reminder)} ({state})"
                )
            lines.append("")
            lines.append("Нажми на напоминание, чтобы включить или выключить его.")
            text = "\n".join(lines)

        await self._send(
            chat_id, text, keyboards.reminders_keyboard(reminders, slots > 0)
        )

    @staticmethod
    def _reminder_title(reminder: Reminder) -> str:
        titles = {
            "workout": "тренировка",
            "water": "вода",
            "weigh_in": "взвешивание",
            # The text is the user's own, so it must be escaped for HTML mode.
            "custom": escape_html(reminder.text) or "своё",
        }
        return titles.get(reminder.kind, reminder.kind)

    async def _reminder_add_start(self, chat_id: int, user: User) -> None:
        reminders = self.storage.list_reminders(user.id)
        slots = reminder_slots_left(user.subscription, len(reminders))
        if slots <= 0:
            message = (
                texts.REMINDER_LIMIT_PRO if self._is_pro(user) else texts.REMINDER_LIMIT_FREE
            )
            await self._send(chat_id, message, keyboards.upgrade_keyboard())
            return
        await self._send(
            chat_id,
            "Какое напоминание добавить?",
            keyboards.reminder_kind_keyboard(self._is_pro(user)),
        )

    async def _reminder_kind_chosen(self, chat_id: int, user: User, kind: str) -> None:
        if kind != "workout" and not self._can(user, FEATURE_CUSTOM_REMINDERS):
            await self._send(
                chat_id,
                texts.pro_only(
                    "custom_reminders", self.config.pro_price_stars, self.config.pro_period_days
                ),
                keyboards.upgrade_keyboard(),
            )
            return
        self.storage.set_state(user.id, f"rem_time{STATE_SEP}{kind}")
        await self._send(chat_id, texts.REMINDER_ASK_TIME)

    async def _reminder_time_entered(
        self, chat_id: int, user: User, kind: str, text: str
    ) -> None:
        time_local = parse_time_of_day(text)
        if time_local is None:
            await self._send(chat_id, texts.INVALID_TIME)
            return
        self.storage.set_state(
            user.id, STATE_SEP.join(["rem_days", kind, time_local])
        )
        await self._send(chat_id, texts.REMINDER_ASK_DAYS)

    async def _reminder_days_entered(
        self, chat_id: int, user: User, payload: str, text: str
    ) -> None:
        kind, _, time_local = payload.partition(STATE_SEP)
        days = parse_days(text)
        if days is None:
            await self._send(chat_id, texts.INVALID_DAYS)
            return
        days_text = ",".join(str(day) for day in days)

        if kind == "custom":
            self.storage.set_state(
                user.id, STATE_SEP.join(["rem_text", kind, time_local, days_text])
            )
            await self._send(chat_id, texts.REMINDER_ASK_TEXT)
            return

        await self._create_reminder(chat_id, user, kind, time_local, days_text, "")

    async def _reminder_text_entered(
        self, chat_id: int, user: User, payload: str, text: str
    ) -> None:
        parts = payload.split(STATE_SEP)
        if len(parts) < 3:
            self.storage.set_state(user.id, "")
            await self._cmd_reminders(chat_id, user, "")
            return
        kind, time_local, days_text = parts[0], parts[1], parts[2]
        await self._create_reminder(chat_id, user, kind, time_local, days_text, text[:200])

    async def _create_reminder(
        self,
        chat_id: int,
        user: User,
        kind: str,
        time_local: str,
        days_text: str,
        text: str,
    ) -> None:
        # Re-check the quota: the dialog may have started before another
        # reminder was added, or Pro may have expired mid-dialog.
        existing = self.storage.list_reminders(user.id)
        if reminder_slots_left(user.subscription, len(existing)) <= 0:
            self.storage.set_state(user.id, "")
            message = (
                texts.REMINDER_LIMIT_PRO if self._is_pro(user) else texts.REMINDER_LIMIT_FREE
            )
            await self._send(chat_id, message, keyboards.upgrade_keyboard())
            return

        offset = user.profile.timezone_offset_minutes
        days = [int(day) for day in days_text.split(",") if day.strip().isdigit()]
        next_fire = compute_next_fire(time_local, days or [0, 1, 2, 3, 4, 5, 6], offset)

        self.storage.add_reminder(
            user_id=user.id,
            kind=kind,
            time_local=time_local,
            days=days_text or EVERY_DAY,
            text=text,
            next_fire_at=next_fire,
        )
        self.storage.set_state(user.id, "")

        await self._send(
            chat_id,
            texts.REMINDER_CREATED.format(
                time=time_local,
                days=format_days(days),
                next=self._format_local(next_fire, offset),
            ),
        )
        await self._cmd_reminders(chat_id, self.storage.get_user(user.id) or user, "")

    async def _reminder_toggle(self, chat_id: int, user: User, argument: str) -> None:
        try:
            reminder_id = int(argument)
        except ValueError:
            return
        reminder = self.storage.get_reminder(reminder_id)
        if reminder is None or reminder.user_id != user.id:
            return
        self.storage.set_reminder_enabled(reminder_id, user.id, not reminder.enabled)
        if not reminder.enabled:
            # Re-enabling: recompute so it does not fire immediately on a stale
            # timestamp from while it was off.
            self.storage.mark_reminder_fired(
                reminder_id,
                reminder.last_fired_at or 0.0,
                next_fire_for(reminder, user.profile.timezone_offset_minutes),
            )
        await self._cmd_reminders(chat_id, user, "")

    async def _reminder_delete(self, chat_id: int, user: User, argument: str) -> None:
        try:
            reminder_id = int(argument)
        except ValueError:
            return
        if self.storage.delete_reminder(reminder_id, user.id):
            await self._send(chat_id, texts.REMINDER_DELETED)
        await self._cmd_reminders(chat_id, user, "")

    async def _deliver_reminder(self, reminder: Reminder) -> bool:
        """Send one reminder. Returns False when the user is unreachable."""
        user = self.storage.get_user(reminder.user_id)
        if user is None:
            return False

        template = texts.REMINDER_MESSAGES.get(reminder.kind, texts.REMINDER_MESSAGES["custom"])
        text = template.format(text=escape_html(reminder.text) or "пора действовать")

        markup = keyboards.today_keyboard() if reminder.kind == "workout" else None
        try:
            await self.client.send_message(user.telegram_id, text, markup)
            return True
        except TelegramError as error:
            # 403 means the user blocked the bot: disable the reminder so the
            # scheduler stops retrying forever.
            if error.error_code == 403:
                self.storage.set_reminder_enabled(reminder.id, reminder.user_id, False)
                logger.info("Disabled reminder %s: user blocked the bot", reminder.id)
            else:
                logger.warning("Reminder %s failed: %s", reminder.id, error)
            return False

    # ------------------------------------------------------------------
    # Pro features
    # ------------------------------------------------------------------

    async def _cmd_nutrition(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self._can(user, FEATURE_NUTRITION):
            await self._deny(chat_id, "nutrition")
            return
        if not self._require_profile(user):
            await self._send(chat_id, texts.PROFILE_INCOMPLETE)
            return

        plan = build_nutrition_plan(user.profile)
        if plan is None:
            await self._send(chat_id, texts.PROFILE_INCOMPLETE)
            return
        await self._send(chat_id, render_nutrition_plan(plan), keyboards.back_to_menu())

    async def _cmd_progress(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self._can(user, FEATURE_ANALYTICS):
            await self._deny(chat_id, "analytics")
            return

        report = build_report(
            self.storage.list_workout_logs(user.id),
            self.storage.list_weight_logs(user.id),
            user.profile.timezone_offset_minutes,
        )
        await self._send(chat_id, render_report(report), keyboards.back_to_menu())

    async def _cmd_export(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self._can(user, FEATURE_EXPORT):
            await self._deny(chat_id, "export")
            return

        profile = user.profile
        program_text = (
            render_program(self._program_for(user)) if self._require_profile(user) else ""
        )
        payload = export_payload(
            {
                "goal": profile.goal,
                "level": profile.level,
                "equipment": profile.equipment,
                "gender": profile.gender,
                "age": profile.age,
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "days_per_week": profile.days_per_week,
                "timezone": format_timezone_offset(profile.timezone_offset_minutes),
            },
            self.storage.list_workout_logs(user.id, limit=1000),
            self.storage.list_weight_logs(user.id, limit=1000),
            program_text,
        )
        await self.client.send_document(
            chat_id,
            filename="fitness_export.json",
            content=payload,
            caption="Твои данные: профиль, программа, тренировки и вес.",
        )

    # ------------------------------------------------------------------
    # Profile and subscription
    # ------------------------------------------------------------------

    async def _cmd_profile(self, chat_id: int, user: User, argument: str = "") -> None:
        fresh = self.storage.get_user(user.id) or user
        profile = fresh.profile
        plan = plan_for(fresh.subscription)
        text = texts.profile_summary(
            goal=GOAL_LABELS.get(profile.goal, profile.goal),
            level=LEVEL_LABELS.get(profile.level, profile.level),
            equipment=EQUIPMENT_LABELS.get(profile.equipment, profile.equipment),
            gender=GENDER_LABELS.get(profile.gender, profile.gender),
            age=profile.age,
            height=profile.height_cm,
            weight=profile.weight_kg,
            days_per_week=profile.days_per_week,
            timezone_label=format_timezone_offset(profile.timezone_offset_minutes),
            plan_title=plan.title,
        )
        await self._send(chat_id, text, keyboards.profile_keyboard())

    async def _edit_field(self, chat_id: int, user: User, field: str) -> None:
        if field == "goal":
            await self._send(chat_id, texts.ASK_GOAL, keyboards.goal_keyboard("set_goal"))
        elif field == "level":
            await self._send(chat_id, texts.ASK_LEVEL, keyboards.level_keyboard("set_level"))
        elif field == "equipment":
            await self._send(
                chat_id, texts.ASK_EQUIPMENT, keyboards.equipment_keyboard("set_equipment")
            )
        elif field == "days":
            await self._send(chat_id, texts.ASK_DAYS, keyboards.days_keyboard("set_days"))
        elif field == "weight":
            self.storage.set_state(user.id, "edit_weight")
            await self._send(chat_id, texts.ASK_WEIGHT)
        elif field == "timezone":
            self.storage.set_state(user.id, "edit_timezone")
            await self._send(chat_id, texts.ASK_TIMEZONE)

    async def _apply_profile_edit(
        self, chat_id: int, user: User, field: str, value: str
    ) -> None:
        columns = {
            "goal": ("goal", str),
            "level": ("level", str),
            "equipment": ("equipment", str),
            "days": ("days_per_week", int),
        }
        if field not in columns:
            return
        column, caster = columns[field]
        try:
            self.storage.update_profile(user.id, **{column: caster(value)})
        except ValueError:
            return
        await self._send(chat_id, "Обновил ✅ Программа пересобрана под новые настройки.")
        await self._cmd_profile(chat_id, self.storage.get_user(user.id) or user, "")

    async def _cmd_subscribe(self, chat_id: int, user: User, argument: str = "") -> None:
        fresh = self.storage.get_user(user.id) or user
        subscription = fresh.subscription

        if subscription.is_active_pro():
            if subscription.expires_at is None:
                text = texts.SUBSCRIPTION_LIFETIME
            else:
                text = texts.SUBSCRIPTION_ACTIVE.format(
                    until=self._format_local(
                        subscription.expires_at, fresh.profile.timezone_offset_minutes
                    ),
                    days=subscription.days_left() or 0,
                )
        else:
            text = texts.paywall(self.config.pro_price_stars, self.config.pro_period_days)

        await self._send(
            chat_id,
            text,
            keyboards.subscription_keyboard(
                self.config.pro_price_stars, subscription.is_active_pro()
            ),
        )

    async def _payment_action(self, chat_id: int, user: User, argument: str) -> None:
        if argument == "stars":
            try:
                await send_pro_invoice(self.client, self.config, chat_id)
            except TelegramError as error:
                logger.error("sendInvoice failed: %s", error)
                await self._send(
                    chat_id,
                    "Не удалось выставить счёт. Попробуй ещё раз через минуту или "
                    "напиши /promo, если у тебя есть промокод.",
                )
        elif argument == "promo":
            self.storage.set_state(user.id, "promo")
            await self._send(chat_id, "Отправь промокод одним сообщением.")

    async def _cmd_promo(self, chat_id: int, user: User, argument: str = "") -> None:
        if not argument:
            self.storage.set_state(user.id, "promo")
            await self._send(chat_id, texts.PROMO_USAGE)
            return

        result = redeem_promo(self.storage, self.config, user, argument)
        if result.activated:
            await self._send(
                chat_id,
                texts.PROMO_OK.format(
                    until=self._format_local(
                        result.expires_at, user.profile.timezone_offset_minutes
                    )
                ),
                keyboards.main_menu(True),
            )
        elif result.reason == "already_used":
            await self._send(chat_id, texts.PROMO_USED)
        else:
            await self._send(chat_id, texts.PROMO_UNKNOWN)

    async def _on_successful_payment(
        self, chat_id: int, user: User, payment: Dict[str, Any]
    ) -> None:
        result, subscription = handle_successful_payment(
            self.storage, self.config, user, payment
        )
        if not result.activated and result.reason == "duplicate":
            await self._send(chat_id, texts.PAYMENT_DUPLICATE)
            return

        until = (
            self._format_local(subscription.expires_at, user.profile.timezone_offset_minutes)
            if subscription.expires_at
            else "бессрочно"
        )
        await self._send(
            chat_id, texts.PAYMENT_SUCCESS.format(until=until), keyboards.main_menu(True)
        )

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    async def _cmd_stats(self, chat_id: int, user: User, argument: str = "") -> None:
        if not self.config.is_admin(user.telegram_id):
            await self._send(chat_id, texts.HELP)
            return
        await self._send(
            chat_id,
            "<b>Статистика</b>\n"
            f"Пользователей: {self.storage.count_users()}\n"
            f"Активных Pro: {self.storage.count_pro_users()}\n"
            f"LLM: {'подключён' if self.config.llm_enabled else 'офлайн-режим'}",
        )

    async def _cmd_grant(self, chat_id: int, user: User, argument: str = "") -> None:
        """`/grant <telegram_id> [days]` — give Pro manually."""
        if not self.config.is_admin(user.telegram_id):
            await self._send(chat_id, texts.HELP)
            return

        parts = argument.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            await self._send(chat_id, "Использование: /grant &lt;telegram_id&gt; [дней]")
            return

        target = self.storage.get_user_by_telegram_id(int(parts[0]))
        if target is None:
            await self._send(chat_id, "Пользователь ещё не запускал бота.")
            return

        if len(parts) > 1 and parts[1].isdigit():
            result = activate_pro(self.storage, target.id, int(parts[1]), "admin")
            until = self._format_local(result.expires_at, 0)
        else:
            grant_lifetime_pro(self.storage, target.id)
            until = "бессрочно"

        await self._send(chat_id, f"Выдал Pro пользователю {parts[0]} до {until}.")
        try:
            await self.client.send_message(
                target.telegram_id, f"⭐ Тебе открыли Pro-доступ ({until}). Приятных тренировок!"
            )
        except TelegramError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(
        self, chat_id: int, text: str, markup: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            await self.client.send_message(chat_id, text, markup)
        except TelegramError as error:
            logger.warning("sendMessage to %s failed: %s", chat_id, error)

    async def _deny(self, chat_id: int, feature: str) -> None:
        await self._send(
            chat_id,
            texts.pro_only(
                feature, self.config.pro_price_stars, self.config.pro_period_days
            ),
            keyboards.upgrade_keyboard(),
        )

    def _is_pro(self, user: User) -> bool:
        return user.subscription.is_active_pro()

    def _can(self, user: User, feature: str) -> bool:
        return plan_for(user.subscription).has(feature)

    def _require_profile(self, user: User) -> bool:
        return user.profile.is_complete

    def _program_for(self, user: User):
        weeks = plan_for(user.subscription).program_weeks
        return generate_program(user.profile, weeks)

    @staticmethod
    def _format_local(timestamp: Optional[float], offset_minutes: int) -> str:
        if timestamp is None:
            return "—"
        moment = datetime.fromtimestamp(timestamp, tz=timezone.utc) + timedelta(
            minutes=offset_minutes
        )
        return moment.strftime("%d.%m.%Y %H:%M")
