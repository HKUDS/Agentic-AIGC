import os
import tempfile
import time
import unittest
from datetime import datetime, timezone

from fitness_coach.bot import FitnessBot
from fitness_coach.coach import AICoach
from fitness_coach.config import Config
from fitness_coach.storage import Storage
from fitness_coach.telegram_api import TelegramClient, TelegramError, escape_html

CHAT_ID = 4242


class FakeTelegramClient:
    """Records outgoing Bot API calls instead of performing them."""

    def __init__(self):
        self.messages = []
        self.invoices = []
        self.documents = []
        self.callback_answers = []
        self.pre_checkout_answers = []
        self.commands = []
        self.fail_send_with = None

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        if self.fail_send_with is not None:
            raise self.fail_send_with
        self.messages.append({"chat_id": chat_id, "text": text, "markup": reply_markup})
        return {"message_id": len(self.messages)}

    async def answer_callback_query(self, callback_query_id, text="", show_alert=False):
        self.callback_answers.append(callback_query_id)

    async def send_invoice(self, chat_id, title, description, payload, prices, **kwargs):
        self.invoices.append(
            {"chat_id": chat_id, "payload": payload, "prices": prices, **kwargs}
        )

    async def answer_pre_checkout_query(self, pre_checkout_query_id, ok=True, error_message=""):
        self.pre_checkout_answers.append({"id": pre_checkout_query_id, "ok": ok})

    async def send_document(self, chat_id, filename, content, caption=""):
        self.documents.append({"chat_id": chat_id, "filename": filename, "content": content})

    async def set_my_commands(self, commands):
        self.commands = commands

    # Test helpers -----------------------------------------------------

    @property
    def last_text(self):
        return self.messages[-1]["text"] if self.messages else ""

    @property
    def all_text(self):
        return "\n".join(message["text"] for message in self.messages)

    def reset(self):
        self.messages.clear()


def message_update(text, telegram_id=99, update_id=1, extra=None):
    message = {
        "message_id": update_id,
        "chat": {"id": CHAT_ID},
        "from": {"id": telegram_id, "first_name": "Тест", "username": "test"},
        "text": text,
    }
    if extra:
        message.update(extra)
    return {"update_id": update_id, "message": message}


def callback_update(data, telegram_id=99, update_id=1):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb-{update_id}",
            "from": {"id": telegram_id, "first_name": "Тест", "username": "test"},
            "message": {"message_id": update_id, "chat": {"id": CHAT_ID}},
            "data": data,
        },
    }


class BotTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.config = Config(
            telegram_token="test-token",
            database_path=os.path.join(self._dir.name, "bot.sqlite3"),
            pro_price_stars=250,
            pro_period_days=30,
            promo_codes=["WELCOME"],
            admin_ids=[1000],
        )
        self.storage = Storage(self.config.database_path)
        self.client = FakeTelegramClient()
        self.coach = AICoach(self.config)  # No API key: offline answers.
        self.bot = FitnessBot(self.config, self.storage, self.client, self.coach)

    def tearDown(self):
        self.storage.close()
        self._dir.cleanup()

    async def send(self, text, telegram_id=99):
        await self.bot.handle_update(message_update(text, telegram_id, self._next_id()))

    async def tap(self, data, telegram_id=99):
        await self.bot.handle_update(callback_update(data, telegram_id, self._next_id()))

    _counter = 0

    def _next_id(self):
        BotTestCase._counter += 1
        return BotTestCase._counter

    async def complete_onboarding(self, telegram_id=99, reminder_time="19:00"):
        await self.send("/start", telegram_id)
        await self.tap("onb_gender:male", telegram_id)
        await self.send("30", telegram_id)
        await self.send("180", telegram_id)
        await self.send("80", telegram_id)
        await self.tap("onb_goal:build_muscle", telegram_id)
        await self.tap("onb_level:beginner", telegram_id)
        await self.tap("onb_equipment:gym", telegram_id)
        await self.tap("onb_days:3", telegram_id)
        await self.send("+3", telegram_id)
        await self.tap(f"onb_reminder:{reminder_time}", telegram_id)
        return self.storage.get_user_by_telegram_id(telegram_id)

    def make_pro(self, telegram_id=99, days=30):
        user = self.storage.get_user_by_telegram_id(telegram_id)
        self.storage.set_subscription(user.id, "pro", time.time() + days * 86400, "test")
        return self.storage.get_user_by_telegram_id(telegram_id)


class OnboardingTests(BotTestCase):
    async def test_full_onboarding_fills_the_profile(self):
        user = await self.complete_onboarding()
        self.assertEqual(user.profile.gender, "male")
        self.assertEqual(user.profile.age, 30)
        self.assertEqual(user.profile.height_cm, 180)
        self.assertEqual(user.profile.weight_kg, 80.0)
        self.assertEqual(user.profile.goal, "build_muscle")
        self.assertEqual(user.profile.level, "beginner")
        self.assertEqual(user.profile.equipment, "gym")
        self.assertEqual(user.profile.days_per_week, 3)
        self.assertEqual(user.profile.timezone_offset_minutes, 180)
        self.assertEqual(user.state, "")
        self.assertTrue(user.profile.is_complete)

    async def test_onboarding_ends_by_showing_a_workout(self):
        await self.complete_onboarding()
        self.assertIn("Разминка", self.client.all_text)

    async def test_starting_weight_is_logged_for_progress_tracking(self):
        user = await self.complete_onboarding()
        weights = self.storage.list_weight_logs(user.id)
        self.assertEqual([log.weight_kg for log in weights], [80.0])

    async def test_invalid_age_is_rejected_and_the_step_repeats(self):
        await self.send("/start")
        await self.tap("onb_gender:male")
        await self.send("abc")
        self.assertIn("Напиши только цифры", self.client.last_text)
        await self.send("5")
        self.assertIn("от 14 до 100", self.client.last_text)

        user = self.storage.get_user_by_telegram_id(99)
        self.assertEqual(user.state, "onb_age")
        self.assertIsNone(user.profile.age)

    async def test_invalid_timezone_is_rejected(self):
        await self.send("/start")
        await self.tap("onb_gender:male")
        await self.send("30")
        await self.send("180")
        await self.send("80")
        await self.tap("onb_goal:keep_fit")
        await self.tap("onb_level:beginner")
        await self.tap("onb_equipment:none")
        await self.tap("onb_days:3")
        await self.send("Москва")
        self.assertIn("Не понял часовой пояс", self.client.last_text)

    async def test_features_ask_for_a_profile_before_it_exists(self):
        await self.send("/today")
        self.assertIn("/start", self.client.last_text)

    async def test_registration_creates_a_reminder_from_the_profile(self):
        user = await self.complete_onboarding(reminder_time="19:00")
        reminders = self.storage.list_reminders(user.id)

        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].kind, "workout")
        self.assertEqual(reminders[0].time_local, "19:00")
        # 3 workouts a week become Mon/Wed/Fri, matching the generated split.
        self.assertEqual(reminders[0].weekdays(), [0, 2, 4])
        self.assertTrue(reminders[0].enabled)
        self.assertGreater(reminders[0].next_fire_at, time.time())
        self.assertIn("Напоминания включены", self.client.all_text)

    async def test_reminder_days_follow_the_weekly_frequency(self):
        await self.send("/start")
        await self.tap("onb_gender:female")
        await self.send("28")
        await self.send("165")
        await self.send("60")
        await self.tap("onb_goal:lose_weight")
        await self.tap("onb_level:beginner")
        await self.tap("onb_equipment:none")
        await self.tap("onb_days:5")
        await self.send("+3")
        await self.tap("onb_reminder:07:00")

        user = self.storage.get_user_by_telegram_id(99)
        self.assertEqual(self.storage.list_reminders(user.id)[0].weekdays(), [0, 1, 2, 3, 4])

    async def test_reminder_time_is_stored_in_the_users_timezone(self):
        user = await self.complete_onboarding(reminder_time="07:00")
        reminder = self.storage.list_reminders(user.id)[0]
        fire = datetime.fromtimestamp(reminder.next_fire_at, tz=timezone.utc)
        local_hour = (fire.hour + 3) % 24  # Profile is UTC+3.
        self.assertEqual((local_hour, fire.minute), (7, 0))

    async def test_custom_reminder_time_during_registration(self):
        await self.send("/start")
        await self.tap("onb_gender:male")
        await self.send("30")
        await self.send("180")
        await self.send("80")
        await self.tap("onb_goal:keep_fit")
        await self.tap("onb_level:beginner")
        await self.tap("onb_equipment:home")
        await self.tap("onb_days:3")
        await self.send("+3")
        await self.tap("onb_reminder:custom")
        await self.send("нет")
        self.assertIn("Не понял время", self.client.last_text)

        await self.send("6:45")
        user = self.storage.get_user_by_telegram_id(99)
        self.assertEqual(self.storage.list_reminders(user.id)[0].time_local, "06:45")
        self.assertEqual(user.state, "")

    async def test_registration_can_finish_without_a_reminder(self):
        user = await self.complete_onboarding(reminder_time="skip")
        self.assertEqual(self.storage.list_reminders(user.id), [])
        self.assertIn("без напоминаний", self.client.all_text)
        self.assertEqual(user.state, "")

    async def test_rerunning_start_retimes_the_reminder_instead_of_duplicating(self):
        user = await self.complete_onboarding(reminder_time="19:00")
        await self.complete_onboarding(reminder_time="08:30")

        reminders = self.storage.list_reminders(user.id)
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].time_local, "08:30")

    async def test_registered_reminder_actually_fires(self):
        user = await self.complete_onboarding(reminder_time="19:00")
        reminder = self.storage.list_reminders(user.id)[0]
        # Pretend the scheduled moment arrived.
        self.storage.mark_reminder_fired(reminder.id, 0.0, time.time() - 1)
        self.client.reset()

        self.assertEqual(await self.bot.scheduler.run_once(), 1)
        self.assertIn("Время тренировки", self.client.last_text)
        self.assertEqual(self.client.messages[-1]["chat_id"], user.telegram_id)

    async def test_cancel_clears_a_pending_step(self):
        await self.send("/start")
        await self.send("/cancel")
        self.assertEqual(self.storage.get_user_by_telegram_id(99).state, "")


class FreeTierTests(BotTestCase):
    async def test_free_program_is_one_week_and_teases_pro(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/program")
        self.assertIn("1 нед.", self.client.all_text)
        self.assertIn("/subscribe", self.client.all_text)

    async def test_ai_chat_is_capped_at_five_messages_per_day(self):
        await self.complete_onboarding()
        self.client.reset()
        for _ in range(5):
            await self.send("Сколько белка нужно есть?")
        self.assertNotIn("закончились", self.client.all_text)

        await self.send("Ещё вопрос про белок")
        self.assertIn("закончились", self.client.last_text)

    async def test_quota_footer_counts_down(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("Сколько белка нужно есть?")
        self.assertIn("4 из 5", self.client.last_text)

    async def test_free_text_outside_a_dialog_reaches_the_coach(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("Что делать при крепатуре?")
        self.assertIn("Мышечная боль", self.client.last_text)

    async def test_pain_question_gets_the_medical_safety_answer(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("Болит колено при приседе, что делать?")
        self.assertIn("врач", self.client.last_text.lower())

    async def test_nutrition_is_locked_behind_pro(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/nutrition")
        self.assertIn("функция Pro", self.client.last_text)
        self.assertIn("250", self.client.last_text)

    async def test_progress_and_export_are_locked_behind_pro(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/progress")
        self.assertIn("функция Pro", self.client.last_text)
        await self.send("/export")
        self.assertIn("функция Pro", self.client.last_text)
        self.assertEqual(self.client.documents, [])

    async def test_workout_can_be_logged_and_counted(self):
        await self.complete_onboarding()
        await self.send("/today")
        await self.send("/log")
        await self.tap("log_difficulty:4")

        user = self.storage.get_user_by_telegram_id(99)
        logs = self.storage.list_workout_logs(user.id)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].difficulty, 4)
        self.assertTrue(logs[0].workout_name)
        self.assertIn("Записал", self.client.last_text)

    async def test_weight_command_records_a_measurement(self):
        user = await self.complete_onboarding()
        await self.send("/weight 78.5")
        weights = self.storage.list_weight_logs(user.id)
        self.assertEqual(weights[0].weight_kg, 78.5)
        self.assertEqual(self.storage.get_user(user.id).profile.weight_kg, 78.5)

    async def test_today_rotates_through_the_split(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/today")
        first = self.client.last_text
        await self.tap("today:next")
        second = self.client.last_text
        self.assertNotEqual(first, second)

    async def test_unknown_command_shows_help(self):
        await self.send("/nope")
        self.assertIn("Команды", self.client.last_text)


class ReminderTests(BotTestCase):
    async def test_free_user_can_create_one_workout_reminder(self):
        user = await self.complete_onboarding(reminder_time="skip")
        await self.tap("rem_add:start")
        await self.tap("rem_kind:workout")
        await self.send("19:00")
        await self.send("будни")

        reminders = self.storage.list_reminders(user.id)
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].time_local, "19:00")
        self.assertEqual(reminders[0].weekdays(), [0, 1, 2, 3, 4])
        self.assertGreater(reminders[0].next_fire_at, time.time())
        self.assertEqual(self.storage.get_user(user.id).state, "")

    async def test_second_reminder_is_blocked_on_free(self):
        user = await self.complete_onboarding(reminder_time="skip")
        await self.tap("rem_add:start")
        await self.tap("rem_kind:workout")
        await self.send("19:00")
        await self.send("ежедневно")

        self.client.reset()
        await self.tap("rem_add:start")
        self.assertIn("бесплатном тарифе", self.client.last_text)
        self.assertEqual(len(self.storage.list_reminders(user.id)), 1)

    async def test_water_reminder_requires_pro(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.tap("rem_kind:water")
        self.assertIn("функция Pro", self.client.last_text)

    async def test_pro_user_can_create_a_custom_reminder(self):
        user = await self.complete_onboarding(reminder_time="skip")
        self.make_pro()
        await self.tap("rem_add:start")
        await self.tap("rem_kind:custom")
        await self.send("07:30")
        await self.send("1,3,5")
        await self.send("Не забудь растяжку")

        reminders = self.storage.list_reminders(user.id)
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0].kind, "custom")
        self.assertEqual(reminders[0].text, "Не забудь растяжку")
        self.assertEqual(reminders[0].weekdays(), [0, 2, 4])

    async def test_invalid_time_keeps_the_dialog_open(self):
        user = await self.complete_onboarding(reminder_time="skip")
        await self.tap("rem_add:start")
        await self.tap("rem_kind:workout")
        await self.send("завтра")
        self.assertIn("Не понял время", self.client.last_text)
        self.assertTrue(self.storage.get_user(user.id).state.startswith("rem_time"))

    async def test_reminder_can_be_toggled_and_deleted(self):
        user = await self.complete_onboarding(reminder_time="skip")
        await self.tap("rem_add:start")
        await self.tap("rem_kind:workout")
        await self.send("19:00")
        await self.send("ежедневно")
        reminder = self.storage.list_reminders(user.id)[0]

        await self.tap(f"rem_toggle:{reminder.id}")
        self.assertFalse(self.storage.get_reminder(reminder.id).enabled)

        await self.tap(f"rem_toggle:{reminder.id}")
        self.assertTrue(self.storage.get_reminder(reminder.id).enabled)

        await self.tap(f"rem_delete:{reminder.id}")
        self.assertEqual(self.storage.list_reminders(user.id), [])

    async def test_users_cannot_touch_each_other_reminders(self):
        owner = await self.complete_onboarding(telegram_id=99)
        reminder = self.storage.add_reminder(owner.id, "workout", "19:00", "0", "", 0.0)
        await self.complete_onboarding(telegram_id=1234)

        await self.tap(f"rem_delete:{reminder.id}", telegram_id=1234)
        self.assertIsNotNone(self.storage.get_reminder(reminder.id))

        await self.tap(f"rem_toggle:{reminder.id}", telegram_id=1234)
        self.assertTrue(self.storage.get_reminder(reminder.id).enabled)

    async def test_timezone_change_reschedules_existing_reminders(self):
        user = await self.complete_onboarding(reminder_time="skip")
        await self.tap("rem_add:start")
        await self.tap("rem_kind:workout")
        await self.send("19:00")
        await self.send("ежедневно")
        before = self.storage.list_reminders(user.id)[0].next_fire_at

        await self.send("/profile")
        await self.tap("edit:timezone")
        await self.send("+9")

        after = self.storage.list_reminders(user.id)[0].next_fire_at
        self.assertNotEqual(before, after)
        self.assertGreater(after, time.time())

    async def test_scheduler_delivers_and_reschedules(self):
        user = await self.complete_onboarding(reminder_time="skip")
        past = time.time() - 60
        reminder = self.storage.add_reminder(
            user.id, "workout", "19:00", "0,1,2,3,4,5,6", "", past
        )
        self.client.reset()

        delivered = await self.bot.scheduler.run_once()
        self.assertEqual(delivered, 1)
        self.assertIn("Время тренировки", self.client.last_text)
        self.assertEqual(self.client.messages[-1]["chat_id"], user.telegram_id)

        refreshed = self.storage.get_reminder(reminder.id)
        self.assertGreater(refreshed.next_fire_at, time.time())
        self.assertEqual(len(self.storage.due_reminders()), 0)

    async def test_custom_reminder_text_is_delivered(self):
        user = await self.complete_onboarding()
        self.storage.add_reminder(
            user.id, "custom", "07:00", "0,1,2,3,4,5,6", "Выпей воды", time.time() - 5
        )
        self.client.reset()
        await self.bot.scheduler.run_once()
        self.assertIn("Выпей воды", self.client.last_text)

    async def test_blocked_user_reminder_is_disabled_instead_of_looping(self):
        user = await self.complete_onboarding()
        reminder = self.storage.add_reminder(
            user.id, "workout", "19:00", "0,1,2,3,4,5,6", "", time.time() - 5
        )
        self.client.fail_send_with = TelegramError("sendMessage", "bot was blocked", 403)

        delivered = await self.bot.scheduler.run_once()
        self.assertEqual(delivered, 0)
        self.assertFalse(self.storage.get_reminder(reminder.id).enabled)

    async def test_disabled_reminders_are_not_delivered(self):
        user = await self.complete_onboarding()
        reminder = self.storage.add_reminder(
            user.id, "workout", "19:00", "0,1,2,3,4,5,6", "", time.time() - 5
        )
        self.storage.set_reminder_enabled(reminder.id, user.id, False)
        self.client.reset()
        self.assertEqual(await self.bot.scheduler.run_once(), 0)
        self.assertEqual(self.client.messages, [])


class SubscriptionFlowTests(BotTestCase):
    async def test_paywall_lists_pro_features_and_price(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/subscribe")
        self.assertIn("Pro", self.client.last_text)
        self.assertIn("250", self.client.last_text)
        self.assertIn("Аналитика", self.client.all_text)

    async def test_stars_invoice_is_sent(self):
        await self.complete_onboarding()
        await self.tap("pay:stars")
        self.assertEqual(len(self.client.invoices), 1)
        self.assertEqual(self.client.invoices[0]["payload"], "pro_subscription_v1")
        self.assertEqual(self.client.invoices[0]["prices"][0]["amount"], 250)
        self.assertEqual(self.client.invoices[0].get("currency"), "XTR")

    async def test_pre_checkout_is_approved_for_a_known_payload(self):
        await self.bot.handle_update(
            {
                "update_id": 900,
                "pre_checkout_query": {
                    "id": "pcq-1",
                    "invoice_payload": "pro_subscription_v1",
                },
            }
        )
        self.assertTrue(self.client.pre_checkout_answers[0]["ok"])

    async def test_pre_checkout_is_rejected_for_a_stale_payload(self):
        await self.bot.handle_update(
            {
                "update_id": 901,
                "pre_checkout_query": {"id": "pcq-2", "invoice_payload": "old_payload"},
            }
        )
        self.assertFalse(self.client.pre_checkout_answers[0]["ok"])

    async def test_successful_payment_unlocks_pro_and_is_idempotent(self):
        user = await self.complete_onboarding()
        payment = {
            "successful_payment": {
                "telegram_payment_charge_id": "charge-1",
                "total_amount": 250,
                "currency": "XTR",
                "invoice_payload": "pro_subscription_v1",
            }
        }
        await self.bot.handle_update(
            message_update("", telegram_id=99, update_id=500, extra=payment)
        )
        self.assertIn("Pro активирован", self.client.last_text)
        subscription = self.storage.get_subscription(user.id)
        self.assertTrue(subscription.is_active_pro())

        first_expiry = subscription.expires_at
        await self.bot.handle_update(
            message_update("", telegram_id=99, update_id=501, extra=payment)
        )
        self.assertIn("уже был учтён", self.client.last_text)
        self.assertEqual(self.storage.get_subscription(user.id).expires_at, first_expiry)

    async def test_promo_code_activates_pro_once(self):
        await self.complete_onboarding()
        await self.send("/promo WELCOME")
        self.assertIn("Промокод принят", self.client.last_text)
        self.assertTrue(
            self.storage.get_user_by_telegram_id(99).subscription.is_active_pro()
        )

        await self.send("/promo WELCOME")
        self.assertIn("уже активировал", self.client.last_text)

    async def test_unknown_promo_is_rejected(self):
        await self.complete_onboarding()
        await self.send("/promo HACK")
        self.assertIn("Такого промокода нет", self.client.last_text)
        self.assertFalse(
            self.storage.get_user_by_telegram_id(99).subscription.is_active_pro()
        )


class ProFeatureTests(BotTestCase):
    async def test_nutrition_returns_calculated_targets(self):
        await self.complete_onboarding()
        self.make_pro()
        self.client.reset()
        await self.send("/nutrition")
        text = self.client.all_text
        self.assertIn("Белки", text)
        self.assertIn("ккал", text)
        self.assertNotIn("функция Pro", text)

    async def test_progress_report_reflects_logged_workouts(self):
        user = await self.complete_onboarding()
        self.make_pro()
        self.storage.add_workout_log(user.id, "День A", duration_minutes=45, difficulty=3)
        self.client.reset()

        await self.send("/progress")
        self.assertIn("Всего тренировок", self.client.all_text)
        self.assertIn("Аналитика прогресса", self.client.all_text)

    async def test_export_sends_a_json_document(self):
        await self.complete_onboarding()
        self.make_pro()
        await self.send("/export")
        self.assertEqual(len(self.client.documents), 1)
        self.assertEqual(self.client.documents[0]["filename"], "fitness_export.json")
        self.assertIn(b"workouts", self.client.documents[0]["content"])

    async def test_pro_program_covers_four_weeks(self):
        await self.complete_onboarding()
        self.make_pro()
        self.client.reset()
        await self.send("/program")
        self.assertIn("4 нед.", self.client.all_text)

    async def test_pro_chat_has_no_daily_limit(self):
        await self.complete_onboarding()
        self.make_pro()
        self.client.reset()
        for _ in range(8):
            await self.send("Сколько белка нужно есть?")
        self.assertNotIn("закончились", self.client.all_text)
        self.assertNotIn("Осталось сообщений", self.client.all_text)

    async def test_expired_pro_loses_access(self):
        user = await self.complete_onboarding()
        self.storage.set_subscription(user.id, "pro", time.time() - 10, "test")
        self.client.reset()
        await self.send("/nutrition")
        self.assertIn("функция Pro", self.client.last_text)


class ProfileAndAdminTests(BotTestCase):
    async def test_profile_screen_shows_current_settings(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/profile")
        text = self.client.last_text
        self.assertIn("набор мышц", text)
        self.assertIn("UTC+03:00", text)
        self.assertIn("Free", text)

    async def test_goal_can_be_changed_from_the_profile_screen(self):
        user = await self.complete_onboarding()
        await self.send("/profile")
        await self.tap("edit:goal")
        await self.tap("set_goal:lose_weight")
        self.assertEqual(self.storage.get_user(user.id).profile.goal, "lose_weight")

    async def test_days_change_reshapes_the_program(self):
        user = await self.complete_onboarding()
        await self.tap("set_days:5")
        self.assertEqual(self.storage.get_user(user.id).profile.days_per_week, 5)

    async def test_admin_commands_are_refused_for_regular_users(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.send("/stats")
        self.assertIn("Команды", self.client.last_text)

    async def test_admin_can_see_stats_and_grant_pro(self):
        target = await self.complete_onboarding(telegram_id=99)
        await self.send("/start", telegram_id=1000)
        self.client.reset()

        await self.send("/stats", telegram_id=1000)
        self.assertIn("Пользователей", self.client.last_text)

        await self.send("/grant 99 15", telegram_id=1000)
        subscription = self.storage.get_subscription(target.id)
        self.assertTrue(subscription.is_active_pro())
        self.assertLess(abs(subscription.expires_at - (time.time() + 15 * 86400)), 60)

    async def test_grant_reports_an_unknown_target(self):
        await self.send("/start", telegram_id=1000)
        await self.send("/grant 555555", telegram_id=1000)
        self.assertIn("не запускал бота", self.client.last_text)


class RobustnessTests(BotTestCase):
    async def test_bot_messages_are_ignored(self):
        update = message_update("/start")
        update["message"]["from"]["is_bot"] = True
        await self.bot.handle_update(update)
        self.assertEqual(self.client.messages, [])

    async def test_non_text_messages_do_not_crash(self):
        update = message_update("")
        update["message"].pop("text")
        update["message"]["photo"] = [{"file_id": "abc"}]
        await self.bot.handle_update(update)
        self.assertEqual(self.client.messages, [])

    async def test_callback_queries_are_always_answered(self):
        await self.complete_onboarding()
        await self.tap("menu:main")
        self.assertTrue(self.client.callback_answers)

    async def test_unknown_callback_data_is_ignored(self):
        await self.complete_onboarding()
        self.client.reset()
        await self.tap("nonsense:payload")
        self.assertEqual(self.client.messages, [])

    async def test_update_offset_survives_a_restart(self):
        self.storage.set_meta("update_offset", "777")
        self.assertEqual(self.bot._load_offset(), 777)
        self.storage.set_meta("update_offset", "broken")
        self.assertIsNone(self.bot._load_offset())

    async def test_send_failures_do_not_propagate(self):
        await self.complete_onboarding()
        self.client.fail_send_with = TelegramError("sendMessage", "chat not found", 400)
        await self.send("/menu")  # Must not raise.

    async def test_user_text_in_reminders_is_html_escaped(self):
        user = await self.complete_onboarding()
        self.make_pro()
        self.storage.add_reminder(
            user.id, "custom", "07:00", "0,1,2,3,4,5,6", "пресс <b> и 3 < 5", time.time() - 5
        )
        self.client.reset()

        await self.bot.scheduler.run_once()
        self.assertIn("&lt;b&gt;", self.client.last_text)
        self.assertNotIn("<b> и", self.client.last_text)

        self.client.reset()
        await self.send("/reminders")
        self.assertNotIn("<b> и", self.client.all_text)


class TelegramClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = TelegramClient("token")
        self.calls = []

    async def test_escape_html(self):
        self.assertEqual(escape_html("a < b & c > d"), "a &lt; b &amp; c &gt; d")
        self.assertEqual(escape_html(""), "")

    async def test_broken_html_is_resent_as_plain_text(self):
        async def fake_call(method, payload=None, timeout=None):
            self.calls.append(payload)
            if payload.get("parse_mode") == "HTML":
                raise TelegramError("sendMessage", "Bad Request: can't parse entities", 400)
            return {"message_id": 1}

        self.client.call = fake_call
        await self.client.send_message(1, "3 < 5 <b")

        self.assertEqual(len(self.calls), 2)
        self.assertIsNone(self.calls[1]["parse_mode"])

    async def test_unrelated_errors_are_not_retried(self):
        async def fake_call(method, payload=None, timeout=None):
            self.calls.append(payload)
            raise TelegramError("sendMessage", "Forbidden: bot was blocked", 403)

        self.client.call = fake_call
        with self.assertRaises(TelegramError):
            await self.client.send_message(1, "привет")
        self.assertEqual(len(self.calls), 1)

    async def test_long_messages_are_split_and_keyboard_goes_last(self):
        async def fake_call(method, payload=None, timeout=None):
            self.calls.append(payload)
            return {"message_id": len(self.calls)}

        self.client.call = fake_call
        markup = {"inline_keyboard": [[{"text": "ok", "callback_data": "x"}]]}
        await self.client.send_message(1, "абзац\n\n" * 2000, markup)

        self.assertGreater(len(self.calls), 1)
        self.assertTrue(all(len(call["text"]) <= 4096 for call in self.calls))
        self.assertIsNone(self.calls[0]["reply_markup"])
        self.assertEqual(self.calls[-1]["reply_markup"], markup)


if __name__ == "__main__":
    unittest.main()
