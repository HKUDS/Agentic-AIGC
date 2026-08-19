import os
import tempfile
import time
import unittest

from aiohttp.test_utils import TestClient, TestServer

from fitness_coach.config import Config
from fitness_coach.storage import Storage
from fitness_coach.webapp.auth import (
    CODE_TTL_SECONDS,
    MAX_SENDS_PER_WINDOW,
    MAX_VERIFY_ATTEMPTS,
    RESEND_COOLDOWN_SECONDS,
    AuthService,
    detect_channel,
    generate_code,
    issue_token,
    normalize_email,
    normalize_phone,
    verify_token,
)
from fitness_coach.webapp.delivery import DeliveryResult, Deliverer
from fitness_coach.webapp.server import WebApp, mask_identifier, parse_survey

SECRET = "test-secret-key"


class IdentifierTests(unittest.TestCase):
    def test_email_normalisation(self):
        self.assertEqual(normalize_email("  Ivan@Example.COM "), "ivan@example.com")
        self.assertIsNone(normalize_email("ivan@example"))
        self.assertIsNone(normalize_email("not-an-email"))
        self.assertIsNone(normalize_email("a@b.c" + "x" * 250))

    def test_phone_normalisation(self):
        self.assertEqual(normalize_phone("+7 999 123-45-67"), "+79991234567")
        self.assertEqual(normalize_phone("8 (999) 123 45 67"), "+79991234567")
        self.assertEqual(normalize_phone("+1-202-555-0143"), "+12025550143")
        self.assertIsNone(normalize_phone("12345"))
        self.assertIsNone(normalize_phone("+7999abc4567"))

    def test_channel_detection(self):
        self.assertEqual(detect_channel("a@b.co"), ("email", "a@b.co"))
        self.assertEqual(detect_channel("89991234567"), ("phone", "+79991234567"))
        self.assertEqual(detect_channel("нет"), (None, None))

    def test_masking_hides_most_of_the_identifier(self):
        masked = mask_identifier("email", "ivanov@example.com")
        self.assertTrue(masked.startswith("iv"))
        self.assertIn("@example.com", masked)
        self.assertNotIn("ivanov", masked)

        phone = mask_identifier("phone", "+79991234567")
        self.assertTrue(phone.endswith("4567"))
        self.assertNotIn("999123", phone)


class TokenTests(unittest.TestCase):
    def test_round_trip(self):
        token = issue_token(7, SECRET)
        self.assertEqual(verify_token(token, SECRET), 7)

    def test_tampered_or_foreign_tokens_are_rejected(self):
        token = issue_token(7, SECRET)
        self.assertIsNone(verify_token(token, "other-secret"))
        self.assertIsNone(verify_token(token.replace("7.", "8.", 1), SECRET))
        self.assertIsNone(verify_token("garbage", SECRET))
        self.assertIsNone(verify_token("", SECRET))

    def test_expired_token_is_rejected(self):
        now = time.time()
        token = issue_token(7, SECRET, ttl=10, now=now)
        self.assertEqual(verify_token(token, SECRET, now=now + 5), 7)
        self.assertIsNone(verify_token(token, SECRET, now=now + 11))

    def test_codes_are_numeric_and_random(self):
        codes = {generate_code() for _ in range(50)}
        self.assertGreater(len(codes), 40)
        self.assertTrue(all(code.isdigit() and len(code) == 6 for code in codes))


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.storage = Storage(os.path.join(self._dir.name, "auth.sqlite3"))
        self.auth = AuthService(self.storage, SECRET)

    def tearDown(self):
        self.storage.close()
        self._dir.cleanup()

    def test_code_is_never_stored_in_plain_text(self):
        result = self.auth.request_code("ivan@example.com")
        record = self.storage.get_auth_code("ivan@example.com")
        self.assertTrue(result.ok)
        self.assertNotIn(result.code, record["code_hash"])
        self.assertEqual(len(record["code_hash"]), 64)

    def test_first_verification_creates_the_account(self):
        request = self.auth.request_code("ivan@example.com")
        result = self.auth.verify("ivan@example.com", request.code)
        self.assertTrue(result.ok)
        self.assertTrue(result.created)

        user = self.storage.get_user(result.user_id)
        self.assertEqual(user.email, "ivan@example.com")
        self.assertIsNone(user.telegram_id)

    def test_second_login_reuses_the_same_account(self):
        first = self.auth.request_code("ivan@example.com")
        created = self.auth.verify("ivan@example.com", first.code)
        second = self.auth.request_code(
            "ivan@example.com", now=time.time() + RESEND_COOLDOWN_SECONDS + 1
        )
        returning = self.auth.verify("ivan@example.com", second.code)

        self.assertTrue(returning.ok)
        self.assertFalse(returning.created)
        self.assertEqual(returning.user_id, created.user_id)

    def test_identifier_is_normalised_before_lookup(self):
        request = self.auth.request_code("  IVAN@Example.com ")
        first = self.auth.verify("ivan@example.com", request.code)
        self.assertTrue(first.ok)
        self.assertEqual(self.storage.count_users(), 1)

    def test_code_cannot_be_replayed(self):
        request = self.auth.request_code("ivan@example.com")
        self.assertTrue(self.auth.verify("ivan@example.com", request.code).ok)
        repeat = self.auth.verify("ivan@example.com", request.code)
        self.assertFalse(repeat.ok)
        self.assertEqual(repeat.error, "no_code")

    def test_wrong_code_is_rejected_and_counted(self):
        self.auth.request_code("ivan@example.com")
        result = self.auth.verify("ivan@example.com", "000000")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "wrong_code")
        self.assertEqual(result.attempts_left, MAX_VERIFY_ATTEMPTS - 1)
        self.assertEqual(self.storage.count_users(), 0)

    def test_code_dies_after_too_many_guesses(self):
        request = self.auth.request_code("ivan@example.com")
        for _ in range(MAX_VERIFY_ATTEMPTS):
            self.auth.verify("ivan@example.com", "000000")

        # Even the correct code no longer works once the budget is spent.
        result = self.auth.verify("ivan@example.com", request.code)
        self.assertFalse(result.ok)
        self.assertIn(result.error, ("no_code", "too_many_attempts"))

    def test_expired_code_is_rejected(self):
        now = time.time()
        request = self.auth.request_code("ivan@example.com", now=now)
        result = self.auth.verify(
            "ivan@example.com", request.code, now=now + CODE_TTL_SECONDS + 1
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "expired")

    def test_resend_is_rate_limited(self):
        now = time.time()
        self.auth.request_code("ivan@example.com", now=now)
        blocked = self.auth.request_code("ivan@example.com", now=now + 5)
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.error, "cooldown")
        self.assertGreater(blocked.retry_after, 0)

        allowed = self.auth.request_code(
            "ivan@example.com", now=now + RESEND_COOLDOWN_SECONDS + 1
        )
        self.assertTrue(allowed.ok)

    def test_hourly_send_cap(self):
        now = time.time()
        for index in range(MAX_SENDS_PER_WINDOW):
            result = self.auth.request_code(
                "ivan@example.com", now=now + index * (RESEND_COOLDOWN_SECONDS + 1)
            )
            self.assertTrue(result.ok, index)

        blocked = self.auth.request_code(
            "ivan@example.com",
            now=now + MAX_SENDS_PER_WINDOW * (RESEND_COOLDOWN_SECONDS + 1),
        )
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.error, "too_many_requests")

        # The window rolls over an hour later.
        fresh = self.auth.request_code("ivan@example.com", now=now + 3700)
        self.assertTrue(fresh.ok)

    def test_a_code_for_one_identifier_does_not_work_for_another(self):
        first = self.auth.request_code("ivan@example.com")
        self.auth.request_code("petr@example.com")
        result = self.auth.verify("petr@example.com", first.code)
        self.assertFalse(result.ok)

    def test_invalid_identifier_is_refused(self):
        self.assertEqual(
            self.auth.request_code("не телефон").error, "invalid_identifier"
        )


class SurveyValidationTests(unittest.TestCase):
    def base(self, **overrides):
        payload = {
            "gender": "male", "age": 30, "height_cm": 180, "weight_kg": 80,
            "goal": "lose_weight", "level": "beginner", "equipment": "mix",
            "days_per_week": 4, "timezone_offset_minutes": 180,
            "reminder_time": "07:00",
        }
        payload.update(overrides)
        return payload

    def test_valid_payload(self):
        fields, error = parse_survey(self.base())
        self.assertEqual(error, "")
        self.assertEqual(fields["equipment"], "mix")
        self.assertEqual(fields["weight_kg"], 80.0)

    def test_each_field_is_validated(self):
        cases = {
            "gender": "хакер",
            "age": 5,
            "height_cm": 300,
            "weight_kg": 1000,
            "goal": "стать невидимым",
            "level": "бог",
            "equipment": "космос",
            "days_per_week": 9,
            "timezone_offset_minutes": 99999,
            "reminder_time": "25:99",
        }
        for field, value in cases.items():
            _, error = parse_survey(self.base(**{field: value}))
            self.assertEqual(error, field, f"{field} was not rejected")

    def test_missing_numbers_are_rejected(self):
        for field in ("age", "height_cm", "weight_kg"):
            payload = self.base()
            payload.pop(field)
            _, error = parse_survey(payload)
            self.assertEqual(error, field)

    def test_comma_decimals_are_accepted(self):
        fields, error = parse_survey(self.base(weight_kg="78,5"))
        self.assertEqual(error, "")
        self.assertEqual(fields["weight_kg"], 78.5)

    def test_reminder_time_is_optional(self):
        fields, error = parse_survey(self.base(reminder_time=""))
        self.assertEqual(error, "")
        self.assertEqual(fields["reminder_time"], "")


class CapturingDeliverer(Deliverer):
    """Records codes instead of sending them."""

    def __init__(self, fail: bool = False):
        self.sent = []
        self.fail = fail

    async def send(self, channel, identifier, code):
        if self.fail:
            return DeliveryResult(False, "smtp_failed")
        self.sent.append({"channel": channel, "identifier": identifier, "code": code})
        return DeliveryResult(True)

    def is_real(self, channel):
        return True

    @property
    def last_code(self):
        return self.sent[-1]["code"]


class WebApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.config = Config(
            telegram_token="",
            database_path=os.path.join(self._dir.name, "web.sqlite3"),
            secret_key=SECRET,
            bot_username="fitness_test_bot",
        )
        self.storage = Storage(self.config.database_path)
        self.deliverer = CapturingDeliverer()
        self.webapp = WebApp(
            self.config,
            self.storage,
            auth=AuthService(self.storage, SECRET),
            deliverer=self.deliverer,
        )
        self.client = TestClient(TestServer(self.webapp.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.storage.close()
        self._dir.cleanup()

    async def sign_in(self, identifier="ivan@example.com"):
        response = await self.client.post("/api/auth/request", json={"identifier": identifier})
        self.assertEqual(response.status, 200)
        verify = await self.client.post(
            "/api/auth/verify",
            json={"identifier": identifier, "code": self.deliverer.last_code},
        )
        self.assertEqual(verify.status, 200)
        return (await verify.json())["token"]

    def auth_header(self, token):
        return {"Authorization": f"Bearer {token}"}

    def survey_payload(self, **overrides):
        payload = {
            "gender": "male", "age": 34, "height_cm": 182, "weight_kg": 88.5,
            "goal": "lose_weight", "level": "beginner", "equipment": "mix",
            "days_per_week": 4, "timezone_offset_minutes": 180,
            "reminder_time": "07:00",
        }
        payload.update(overrides)
        return payload

    async def test_page_and_health_are_served(self):
        page = await self.client.get("/")
        body = await page.text()
        self.assertEqual(page.status, 200)
        self.assertIn("ИИ-фитнес-тренер", body)
        self.assertIn("viewport", body)  # mobile-ready

        health = await self.client.get("/healthz")
        self.assertEqual(health.status, 200)

    async def test_security_headers_are_present(self):
        response = await self.client.get("/")
        self.assertIn("Content-Security-Policy", response.headers)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    async def test_code_is_never_returned_to_the_client(self):
        response = await self.client.post(
            "/api/auth/request", json={"identifier": "ivan@example.com"}
        )
        data = await response.json()
        self.assertNotIn("dev_code", data)
        self.assertNotIn(self.deliverer.last_code, await response.text())
        self.assertEqual(data["masked"], "iv**@example.com")

    async def test_response_does_not_reveal_whether_the_account_exists(self):
        first = await self.client.post(
            "/api/auth/request", json={"identifier": "new@example.com"}
        )
        token = await self.sign_in("known@example.com")
        self.assertTrue(token)
        second = await self.client.post(
            "/api/auth/request", json={"identifier": "known@example.com"}
        )
        # A returning user gets the exact same shape as a brand-new one.
        self.assertEqual(
            set((await first.json()).keys()), set((await second.json()).keys())
        )

    async def test_invalid_identifier_is_refused(self):
        response = await self.client.post("/api/auth/request", json={"identifier": "нет"})
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"], "invalid_identifier")

    async def test_resend_cooldown_returns_429(self):
        await self.client.post("/api/auth/request", json={"identifier": "a@b.co"})
        again = await self.client.post("/api/auth/request", json={"identifier": "a@b.co"})
        self.assertEqual(again.status, 429)
        self.assertEqual((await again.json())["error"], "cooldown")

    async def test_failed_delivery_does_not_strand_the_user(self):
        self.deliverer.fail = True
        response = await self.client.post(
            "/api/auth/request", json={"identifier": "a@b.co"}
        )
        self.assertEqual(response.status, 502)
        # The pending code was dropped, so a retry is possible right away.
        self.assertIsNone(self.storage.get_auth_code("a@b.co"))

    async def test_wrong_code_is_rejected_with_attempts_left(self):
        await self.client.post("/api/auth/request", json={"identifier": "a@b.co"})
        response = await self.client.post(
            "/api/auth/verify", json={"identifier": "a@b.co", "code": "000000"}
        )
        data = await response.json()
        self.assertEqual(response.status, 400)
        self.assertEqual(data["error"], "wrong_code")
        self.assertEqual(data["attempts_left"], MAX_VERIFY_ATTEMPTS - 1)

    async def test_protected_endpoints_require_a_valid_token(self):
        for path in ("/api/me", "/api/plan"):
            self.assertEqual((await self.client.get(path)).status, 401)
        self.assertEqual((await self.client.post("/api/survey", json={})).status, 401)
        self.assertEqual((await self.client.post("/api/telegram/link", json={})).status, 401)

        forged = issue_token(1, "wrong-secret")
        response = await self.client.get("/api/me", headers=self.auth_header(forged))
        self.assertEqual(response.status, 401)

    async def test_new_account_is_asked_to_take_the_survey(self):
        token = await self.sign_in()
        response = await self.client.get("/api/me", headers=self.auth_header(token))
        data = await response.json()
        self.assertFalse(data["profile"]["complete"])
        self.assertFalse(data["telegram_linked"])
        self.assertEqual(data["plan"]["name"], "free")

    async def test_survey_returns_a_plan_and_stores_the_profile(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey", json=self.survey_payload(), headers=self.auth_header(token)
        )
        data = await response.json()
        self.assertEqual(response.status, 200)

        program = data["program"]
        self.assertEqual(len(program["workouts"]), 4)
        self.assertEqual(program["equipment"], "mix")
        self.assertEqual(data["profile"]["age"], 34)
        self.assertTrue(data["profile"]["complete"])

    async def test_mix_alternates_home_and_gym_days(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(equipment="mix", days_per_week=4),
            headers=self.auth_header(token),
        )
        venues = [w["venue"] for w in (await response.json())["program"]["workouts"]]
        self.assertEqual(venues, ["gym", "home", "gym", "home"])

        labels = [w["venue_label"] for w in (await response.json())["program"]["workouts"]]
        self.assertEqual(labels, ["в зале", "дома", "в зале", "дома"])
        # The place lives in the badge, not duplicated in the title.
        titles = [w["title"] for w in (await response.json())["program"]["workouts"]]
        self.assertTrue(all("·" not in title for title in titles), titles)

    async def test_home_only_plan_has_no_venue_badges(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(equipment="home"),
            headers=self.auth_header(token),
        )
        workouts = (await response.json())["program"]["workouts"]
        self.assertTrue(all(workout["venue"] == "home" for workout in workouts))
        self.assertTrue(all(workout["venue_label"] == "" for workout in workouts))

    async def test_survey_creates_the_reminder_from_the_profile(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(days_per_week=3, reminder_time="07:00"),
            headers=self.auth_header(token),
        )
        reminders = (await response.json())["reminders"]
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0]["time"], "07:00")
        self.assertEqual(reminders[0]["days"], "Пн, Ср, Пт")

        stored = self.storage.list_reminders(
            self.storage.get_user_by_identifier("email", "ivan@example.com").id
        )
        self.assertGreater(stored[0].next_fire_at, time.time())

    async def test_survey_without_a_reminder_creates_none(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(reminder_time=""),
            headers=self.auth_header(token),
        )
        self.assertEqual((await response.json())["reminders"], [])

    async def test_retaking_the_survey_does_not_duplicate_reminders(self):
        token = await self.sign_in()
        await self.client.post(
            "/api/survey", json=self.survey_payload(), headers=self.auth_header(token)
        )
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(reminder_time="20:30"),
            headers=self.auth_header(token),
        )
        reminders = (await response.json())["reminders"]
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0]["time"], "20:30")

    async def test_invalid_survey_names_the_bad_field(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/survey",
            json=self.survey_payload(age=5),
            headers=self.auth_header(token),
        )
        data = await response.json()
        self.assertEqual(response.status, 400)
        self.assertEqual(data["field"], "age")

    async def test_plan_endpoint_returns_the_saved_plan(self):
        token = await self.sign_in()
        await self.client.post(
            "/api/survey", json=self.survey_payload(), headers=self.auth_header(token)
        )
        response = await self.client.get("/api/plan", headers=self.auth_header(token))
        data = await response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["program"]["workouts"]), 4)
        # Nutrition and analytics stay behind the paywall.
        self.assertNotIn("nutrition", data)
        self.assertNotIn("progress", data)

    async def test_pro_plan_includes_nutrition_and_progress(self):
        token = await self.sign_in()
        await self.client.post(
            "/api/survey", json=self.survey_payload(), headers=self.auth_header(token)
        )
        user = self.storage.get_user_by_identifier("email", "ivan@example.com")
        self.storage.set_subscription(user.id, "pro", time.time() + 86400, "test")

        response = await self.client.get("/api/plan", headers=self.auth_header(token))
        data = await response.json()
        self.assertIn("nutrition", data)
        self.assertGreater(data["nutrition"]["calories"], 1200)
        self.assertIn("progress", data)
        self.assertEqual(data["program"]["weeks"], 4)

    async def test_plan_asks_for_the_survey_when_the_profile_is_empty(self):
        token = await self.sign_in()
        response = await self.client.get("/api/plan", headers=self.auth_header(token))
        self.assertTrue((await response.json())["needs_survey"])

    async def test_telegram_link_returns_a_deep_link(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/telegram/link", json={}, headers=self.auth_header(token)
        )
        data = await response.json()
        self.assertTrue(data["url"].startswith("https://t.me/fitness_test_bot?start="))
        self.assertIn(data["code"], data["url"])

        user = self.storage.get_user_by_identifier("email", "ivan@example.com")
        self.assertEqual(self.storage.consume_link_code(data["code"]), user.id)

    async def test_link_code_is_single_use_and_expires(self):
        token = await self.sign_in()
        response = await self.client.post(
            "/api/telegram/link", json={}, headers=self.auth_header(token)
        )
        code = (await response.json())["code"]
        self.assertIsNotNone(self.storage.consume_link_code(code))
        self.assertIsNone(self.storage.consume_link_code(code))

        second = await self.client.post(
            "/api/telegram/link", json={}, headers=self.auth_header(token)
        )
        fresh = (await second.json())["code"]
        self.assertIsNone(self.storage.consume_link_code(fresh, now=time.time() + 1000))

    async def test_phone_sign_up_works_too(self):
        token = await self.sign_in("8 999 123 45 67")
        self.assertTrue(token)
        user = self.storage.get_user_by_identifier("phone", "+79991234567")
        self.assertIsNotNone(user)
        self.assertIsNone(user.email)


if __name__ == "__main__":
    unittest.main()
