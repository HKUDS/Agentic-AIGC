import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from fitness_coach.analytics import build_report, export_payload, training_streak_weeks
from fitness_coach.coach import offline_answer, usage_summary
from fitness_coach.config import Config
from fitness_coach.models import Profile, Subscription
from fitness_coach.nutrition import build_nutrition_plan, mifflin_st_jeor
from fitness_coach.payments import activate_pro, handle_successful_payment, redeem_promo
from fitness_coach.scheduler import (
    compute_next_fire,
    format_days,
    parse_days,
    parse_time_of_day,
    parse_timezone_offset,
)
from fitness_coach.storage import Storage
from fitness_coach.subscription import (
    FEATURE_ANALYTICS,
    FEATURE_NUTRITION,
    FEATURE_WORKOUT_REMINDER,
    FREE_PLAN,
    PRO_PLAN,
    ai_messages_left,
    can_use,
    plan_for,
    reminder_slots_left,
)
from fitness_coach.workouts import generate_program, render_program


def make_profile(**overrides):
    defaults = dict(
        user_id=1,
        gender="male",
        age=30,
        height_cm=180,
        weight_kg=80.0,
        goal="build_muscle",
        level="intermediate",
        equipment="gym",
        days_per_week=3,
    )
    defaults.update(overrides)
    return Profile(**defaults)


class SubscriptionGatingTests(unittest.TestCase):
    def test_free_user_gets_free_plan(self):
        self.assertIs(plan_for(Subscription(user_id=1)), FREE_PLAN)

    def test_active_pro_gets_pro_plan(self):
        now = time.time()
        subscription = Subscription(user_id=1, plan="pro", expires_at=now + 86400)
        self.assertIs(plan_for(subscription, now), PRO_PLAN)

    def test_expired_pro_falls_back_to_free(self):
        now = time.time()
        subscription = Subscription(user_id=1, plan="pro", expires_at=now - 1)
        self.assertIs(plan_for(subscription, now), FREE_PLAN)
        self.assertFalse(can_use(FEATURE_NUTRITION, subscription, now))

    def test_lifetime_pro_never_expires(self):
        subscription = Subscription(user_id=1, plan="pro", expires_at=None)
        self.assertTrue(subscription.is_active_pro(time.time() + 10**9))

    def test_pro_only_features_are_locked_on_free(self):
        free = Subscription(user_id=1)
        self.assertFalse(can_use(FEATURE_NUTRITION, free))
        self.assertFalse(can_use(FEATURE_ANALYTICS, free))
        self.assertTrue(can_use(FEATURE_WORKOUT_REMINDER, free))

    def test_ai_quota_counts_down_on_free_and_is_unlimited_on_pro(self):
        free = Subscription(user_id=1)
        self.assertEqual(ai_messages_left(free, 0), 5)
        self.assertEqual(ai_messages_left(free, 5), 0)
        self.assertEqual(ai_messages_left(free, 99), 0)

        pro = Subscription(user_id=1, plan="pro", expires_at=None)
        self.assertIsNone(ai_messages_left(pro, 500))

    def test_reminder_slots(self):
        free = Subscription(user_id=1)
        pro = Subscription(user_id=1, plan="pro", expires_at=None)
        self.assertEqual(reminder_slots_left(free, 0), 1)
        self.assertEqual(reminder_slots_left(free, 1), 0)
        self.assertEqual(reminder_slots_left(pro, 3), 7)

    def test_days_left_is_rounded_up_to_whole_days(self):
        now = time.time()
        subscription = Subscription(user_id=1, plan="pro", expires_at=now + 3600)
        self.assertEqual(subscription.days_left(now), 1)


class WorkoutGeneratorTests(unittest.TestCase):
    def test_program_is_deterministic_for_the_same_profile(self):
        first = generate_program(make_profile())
        second = generate_program(make_profile())
        self.assertEqual(
            [exercise.name for workout in first.workouts for exercise in workout.exercises],
            [exercise.name for workout in second.workouts for exercise in workout.exercises],
        )

    def test_workout_count_matches_requested_days(self):
        for days in (2, 3, 4, 5):
            program = generate_program(make_profile(days_per_week=days))
            self.assertEqual(len(program.workouts), days)

    def test_days_are_clamped_into_the_supported_range(self):
        self.assertEqual(len(generate_program(make_profile(days_per_week=1)).workouts), 2)
        self.assertEqual(len(generate_program(make_profile(days_per_week=9)).workouts), 5)

    def test_beginner_never_gets_advanced_movements(self):
        program = generate_program(make_profile(level="beginner", equipment="none"))
        names = [exercise.name for workout in program.workouts for exercise in workout.exercises]
        self.assertNotIn("Приседания-пистолетик", names)
        self.assertNotIn("Уголок (L-sit)", names)

    def test_goal_drives_sets_and_rest(self):
        muscle = generate_program(make_profile(goal="build_muscle"))
        loss = generate_program(make_profile(goal="lose_weight"))
        self.assertEqual(muscle.workouts[0].exercises[0].sets, 4)
        self.assertEqual(muscle.workouts[0].exercises[0].rest_seconds, 90)
        self.assertEqual(loss.workouts[0].exercises[0].rest_seconds, 45)

    def test_weight_loss_program_includes_cardio(self):
        program = generate_program(make_profile(goal="lose_weight"))
        groups = {
            exercise.muscle_group
            for workout in program.workouts
            for exercise in workout.exercises
        }
        self.assertIn("cardio", groups)

    def test_equipment_free_program_uses_bodyweight_movements(self):
        program = generate_program(make_profile(equipment="none", level="advanced"))
        names = [exercise.name for workout in program.workouts for exercise in workout.exercises]
        self.assertTrue(all("штанг" not in name.lower() for name in names))

    def test_render_program_includes_every_workout(self):
        program = generate_program(make_profile(days_per_week=4))
        rendered = render_program(program)
        for workout in program.workouts:
            self.assertIn(workout.title, rendered)


class NutritionTests(unittest.TestCase):
    def test_mifflin_matches_the_reference_formula(self):
        profile = make_profile(gender="male", weight_kg=80, height_cm=180, age=30)
        # 10*80 + 6.25*180 - 5*30 + 5 = 1780
        self.assertAlmostEqual(mifflin_st_jeor(profile), 1780.0, places=4)

    def test_incomplete_profile_yields_no_plan(self):
        self.assertIsNone(mifflin_st_jeor(make_profile(weight_kg=None)))
        self.assertIsNone(build_nutrition_plan(make_profile(age=None)))

    def test_deficit_and_surplus_move_calories_the_right_way(self):
        cut = build_nutrition_plan(make_profile(goal="lose_weight"))
        bulk = build_nutrition_plan(make_profile(goal="build_muscle"))
        self.assertLess(cut.calories, cut.tdee)
        self.assertGreater(bulk.calories, bulk.tdee)

    def test_macros_add_up_to_the_calorie_target(self):
        plan = build_nutrition_plan(make_profile())
        total = plan.protein_g * 4 + plan.fat_g * 9 + plan.carbs_g * 4
        self.assertLess(abs(total - plan.calories), 25)

    def test_calories_never_drop_below_a_safe_floor(self):
        plan = build_nutrition_plan(
            make_profile(gender="female", weight_kg=40.0, height_cm=150, age=70, goal="lose_weight")
        )
        self.assertGreaterEqual(plan.calories, 1200)
        self.assertTrue(plan.warning)


class SchedulerParsingTests(unittest.TestCase):
    def test_time_parsing_accepts_common_shapes(self):
        self.assertEqual(parse_time_of_day("7"), "07:00")
        self.assertEqual(parse_time_of_day("7:5"), "07:05")
        self.assertEqual(parse_time_of_day("19.30"), "19:30")
        self.assertEqual(parse_time_of_day("0730"), "07:30")
        self.assertEqual(parse_time_of_day(" 23:59 "), "23:59")

    def test_time_parsing_rejects_invalid_input(self):
        for value in ("25:00", "12:99", "abc", "", "-1"):
            self.assertIsNone(parse_time_of_day(value), value)

    def test_timezone_parsing(self):
        self.assertEqual(parse_timezone_offset("+3"), 180)
        self.assertEqual(parse_timezone_offset("UTC+3"), 180)
        self.assertEqual(parse_timezone_offset("-05:30"), -330)
        self.assertEqual(parse_timezone_offset("0"), 0)
        self.assertIsNone(parse_timezone_offset("+20"))
        self.assertIsNone(parse_timezone_offset("moscow"))

    def test_day_parsing(self):
        self.assertEqual(parse_days("ежедневно"), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(parse_days("будни"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_days("1,3,5"), [0, 2, 4])
        self.assertEqual(parse_days("пн, ср"), [0, 2])
        self.assertIsNone(parse_days("0,9"))

    def test_day_formatting(self):
        self.assertEqual(format_days([0, 1, 2, 3, 4, 5, 6]), "ежедневно")
        self.assertEqual(format_days([5, 6]), "по выходным")
        self.assertEqual(format_days([0, 2, 4]), "Пн, Ср, Пт")


class NextFireTests(unittest.TestCase):
    @staticmethod
    def _utc(year, month, day, hour, minute=0):
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()

    def test_fires_later_today_when_the_time_has_not_passed(self):
        # 2026-01-05 is a Monday. 08:00 UTC with a +0 offset.
        now = self._utc(2026, 1, 5, 8)
        result = compute_next_fire("19:00", [0, 1, 2, 3, 4, 5, 6], 0, now)
        self.assertEqual(result, self._utc(2026, 1, 5, 19))

    def test_rolls_over_to_tomorrow_when_the_time_has_passed(self):
        now = self._utc(2026, 1, 5, 20)
        result = compute_next_fire("19:00", [0, 1, 2, 3, 4, 5, 6], 0, now)
        self.assertEqual(result, self._utc(2026, 1, 6, 19))

    def test_respects_the_weekday_selection(self):
        # Monday 20:00; reminder set for Wednesdays only.
        now = self._utc(2026, 1, 5, 20)
        result = compute_next_fire("07:00", [2], 0, now)
        self.assertEqual(result, self._utc(2026, 1, 7, 7))

    def test_timezone_offset_shifts_the_absolute_time(self):
        # 07:00 local at UTC+3 is 04:00 UTC.
        now = self._utc(2026, 1, 5, 0)
        result = compute_next_fire("07:00", [0], 180, now)
        self.assertEqual(result, self._utc(2026, 1, 5, 4))

    def test_result_is_always_in_the_future(self):
        now = self._utc(2026, 1, 5, 19)
        result = compute_next_fire("19:00", [0, 1, 2, 3, 4, 5, 6], 0, now)
        self.assertGreater(result, now)

    def test_negative_offset_can_move_the_fire_to_the_next_utc_day(self):
        # 23:00 local at UTC-5 is 04:00 UTC the next day.
        now = self._utc(2026, 1, 5, 12)
        result = compute_next_fire("23:00", [0, 1, 2, 3, 4, 5, 6], -300, now)
        self.assertEqual(result, self._utc(2026, 1, 6, 4))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.storage = Storage(os.path.join(self._dir.name, "test.sqlite3"))

    def tearDown(self):
        self.storage.close()
        self._dir.cleanup()

    def test_user_is_created_once_with_defaults(self):
        first = self.storage.get_or_create_user(555, "Аня", "anya")
        second = self.storage.get_or_create_user(555)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.first_name, "Аня")
        self.assertEqual(second.subscription.plan, "free")
        self.assertEqual(second.profile.days_per_week, 3)

    def test_profile_updates_are_persisted_and_unknown_keys_ignored(self):
        user = self.storage.get_or_create_user(1)
        self.storage.update_profile(user.id, weight_kg=72.5, goal="lose_weight", hacker="x")
        refreshed = self.storage.get_user(user.id)
        self.assertEqual(refreshed.profile.weight_kg, 72.5)
        self.assertEqual(refreshed.profile.goal, "lose_weight")

    def test_ai_quota_is_per_day(self):
        user = self.storage.get_or_create_user(1)
        self.assertEqual(self.storage.ai_messages_used_today(user.id), 0)
        self.storage.record_ai_message(user.id)
        self.storage.record_ai_message(user.id)
        self.assertEqual(self.storage.ai_messages_used_today(user.id), 2)

    def test_reminder_lifecycle(self):
        user = self.storage.get_or_create_user(1)
        reminder = self.storage.add_reminder(user.id, "workout", "19:00", "0,2,4", "", 100.0)
        self.assertEqual([r.id for r in self.storage.list_reminders(user.id)], [reminder.id])
        self.assertEqual(reminder.weekdays(), [0, 2, 4])

        self.storage.set_reminder_enabled(reminder.id, user.id, False)
        self.assertEqual(self.storage.list_reminders(user.id, only_enabled=True), [])

        self.storage.set_reminder_enabled(reminder.id, user.id, True)
        self.assertEqual(len(self.storage.due_reminders(now=200.0)), 1)
        self.assertEqual(len(self.storage.due_reminders(now=50.0)), 0)

        self.storage.mark_reminder_fired(reminder.id, 200.0, 400.0)
        self.assertEqual(len(self.storage.due_reminders(now=300.0)), 0)

        self.assertTrue(self.storage.delete_reminder(reminder.id, user.id))
        self.assertEqual(self.storage.list_reminders(user.id), [])

    def test_reminders_are_scoped_to_their_owner(self):
        owner = self.storage.get_or_create_user(1)
        other = self.storage.get_or_create_user(2)
        reminder = self.storage.add_reminder(owner.id, "workout", "19:00", "0", "", 0.0)
        self.assertFalse(self.storage.delete_reminder(reminder.id, other.id))
        self.assertEqual(len(self.storage.list_reminders(owner.id)), 1)

    def test_payments_are_idempotent_per_charge(self):
        user = self.storage.get_or_create_user(1)
        self.assertTrue(self.storage.record_payment(user.id, 250, "XTR", "charge-1", "p"))
        self.assertFalse(self.storage.record_payment(user.id, 250, "XTR", "charge-1", "p"))

    def test_promo_codes_are_single_use_per_user(self):
        first = self.storage.get_or_create_user(1)
        second = self.storage.get_or_create_user(2)
        self.assertTrue(self.storage.redeem_promo(first.id, "WELCOME"))
        self.assertFalse(self.storage.redeem_promo(first.id, "welcome"))
        self.assertTrue(self.storage.redeem_promo(second.id, "WELCOME"))

    def test_chat_history_is_returned_oldest_first(self):
        user = self.storage.get_or_create_user(1)
        self.storage.append_chat_message(user.id, "user", "первый")
        self.storage.append_chat_message(user.id, "assistant", "второй")
        history = self.storage.recent_chat(user.id)
        self.assertEqual([item["content"] for item in history], ["первый", "второй"])

    def test_pro_counters(self):
        first = self.storage.get_or_create_user(1)
        self.storage.get_or_create_user(2)
        self.storage.set_subscription(first.id, "pro", time.time() + 100, "test")
        self.assertEqual(self.storage.count_users(), 2)
        self.assertEqual(self.storage.count_pro_users(), 1)


class PaymentTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.storage = Storage(os.path.join(self._dir.name, "pay.sqlite3"))
        self.config = Config(
            telegram_token="t",
            pro_price_stars=250,
            pro_period_days=30,
            promo_codes=["WELCOME"],
        )
        self.user = self.storage.get_or_create_user(42)

    def tearDown(self):
        self.storage.close()
        self._dir.cleanup()

    def test_activation_sets_the_expected_expiry(self):
        now = 1_700_000_000.0
        result = activate_pro(self.storage, self.user.id, 30, "test", now)
        self.assertAlmostEqual(result.expires_at, now + 30 * 86400)
        self.assertTrue(self.storage.get_subscription(self.user.id).is_active_pro(now))

    def test_renewing_early_extends_instead_of_resetting(self):
        now = 1_700_000_000.0
        first = activate_pro(self.storage, self.user.id, 30, "test", now)
        second = activate_pro(self.storage, self.user.id, 30, "test", now + 86400)
        self.assertAlmostEqual(second.expires_at, first.expires_at + 30 * 86400)

    def test_successful_payment_activates_pro_once(self):
        payment = {
            "telegram_payment_charge_id": "charge-9",
            "total_amount": 250,
            "currency": "XTR",
            "invoice_payload": "pro_subscription_v1",
        }
        result, subscription = handle_successful_payment(
            self.storage, self.config, self.user, payment
        )
        self.assertTrue(result.activated)
        self.assertEqual(subscription.plan, "pro")

        repeat, _ = handle_successful_payment(
            self.storage, self.config, self.user, payment
        )
        self.assertFalse(repeat.activated)
        self.assertEqual(repeat.reason, "duplicate")

    def test_promo_flow(self):
        self.assertEqual(
            redeem_promo(self.storage, self.config, self.user, "NOPE").reason, "unknown"
        )
        self.assertTrue(redeem_promo(self.storage, self.config, self.user, "welcome").activated)
        self.assertEqual(
            redeem_promo(self.storage, self.config, self.user, "WELCOME").reason,
            "already_used",
        )


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.storage = Storage(os.path.join(self._dir.name, "stats.sqlite3"))
        self.user = self.storage.get_or_create_user(7)

    def tearDown(self):
        self.storage.close()
        self._dir.cleanup()

    def test_report_counts_recent_windows(self):
        now = time.time()
        for days_ago in (0, 2, 6, 10, 40):
            self.storage.add_workout_log(
                self.user.id,
                "День A",
                duration_minutes=45,
                difficulty=3,
                created_at=now - days_ago * 86400,
            )
        report = build_report(
            self.storage.list_workout_logs(self.user.id),
            self.storage.list_weight_logs(self.user.id),
            now=now,
        )
        self.assertEqual(report.total_workouts, 5)
        self.assertEqual(report.workouts_last_7_days, 3)
        self.assertEqual(report.workouts_last_30_days, 4)
        self.assertEqual(report.minutes_last_30_days, 180)

    def test_weight_delta_uses_first_and_last_measurement(self):
        now = time.time()
        self.storage.add_weight_log(self.user.id, 82.0, created_at=now - 30 * 86400)
        self.storage.add_weight_log(self.user.id, 79.5, created_at=now)
        report = build_report(
            [], self.storage.list_weight_logs(self.user.id), now=now
        )
        self.assertEqual(report.weight_start, 82.0)
        self.assertEqual(report.weight_latest, 79.5)
        self.assertEqual(report.weight_delta, -2.5)

    def test_streak_counts_consecutive_weeks(self):
        # Wednesday 2026-01-07 12:00 UTC.
        now = datetime(2026, 1, 7, 12, tzinfo=timezone.utc).timestamp()
        logs = []
        for week in range(3):
            logs.append(
                self.storage.add_workout_log(
                    self.user.id, "День A", created_at=now - week * 7 * 86400
                )
            )
        self.assertEqual(training_streak_weeks(logs, 0, now), 3)

    def test_streak_is_zero_without_logs(self):
        self.assertEqual(training_streak_weeks([], 0, time.time()), 0)

    def test_gap_breaks_the_streak(self):
        now = datetime(2026, 1, 7, 12, tzinfo=timezone.utc).timestamp()
        logs = [
            self.storage.add_workout_log(self.user.id, "A", created_at=now),
            self.storage.add_workout_log(
                self.user.id, "A", created_at=now - 21 * 86400
            ),
        ]
        self.assertEqual(training_streak_weeks(logs, 0, now), 1)

    def test_export_payload_is_valid_json_with_all_sections(self):
        import json

        self.storage.add_workout_log(self.user.id, "День A", duration_minutes=40)
        self.storage.add_weight_log(self.user.id, 80.0)
        payload = export_payload(
            {"goal": "keep_fit"},
            self.storage.list_workout_logs(self.user.id),
            self.storage.list_weight_logs(self.user.id),
            "программа",
        )
        data = json.loads(payload.decode("utf-8"))
        self.assertEqual(data["profile"]["goal"], "keep_fit")
        self.assertEqual(len(data["workouts"]), 1)
        self.assertEqual(len(data["weights"]), 1)
        self.assertEqual(data["program"], "программа")


class OfflineCoachTests(unittest.TestCase):
    def test_pain_questions_route_to_the_medical_answer(self):
        answer = offline_answer("У меня болит колено после приседаний")
        self.assertIn("врач", answer.lower())

    def test_known_topics_get_specific_answers(self):
        self.assertIn("г на кг", offline_answer("Сколько белка есть в день?"))
        self.assertIn("дефицит", offline_answer("Как быстрее похудеть?").lower())

    def test_unknown_topic_falls_back_with_profile_context(self):
        answer = offline_answer("Расскажи что-нибудь", make_profile(goal="keep_fit"))
        self.assertIn("поддержание формы", answer)

    def test_usage_summary_only_shown_for_limited_plans(self):
        self.assertEqual(usage_summary(3, None), "")
        self.assertIn("2 из 5", usage_summary(3, 5))


class ConfigTests(unittest.TestCase):
    def test_missing_token_is_rejected(self):
        self.assertIsNotNone(Config().validate())
        self.assertIsNone(Config(telegram_token="abc").validate())

    def test_env_parsing(self):
        env = {
            "TELEGRAM_BOT_TOKEN": " token ",
            "FITNESS_PRO_PRICE_STARS": "99",
            "FITNESS_PROMO_CODES": "one, two",
            "FITNESS_ADMIN_IDS": "1,2,oops",
        }
        original = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            config = Config.from_env()
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(config.telegram_token, "token")
        self.assertEqual(config.pro_price_stars, 99)
        self.assertEqual(config.promo_codes, ["ONE", "TWO"])
        self.assertEqual(config.admin_ids, [1, 2])
        self.assertFalse(config.is_admin(3))


if __name__ == "__main__":
    unittest.main()
