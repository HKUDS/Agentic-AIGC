"""HTTP API and static hosting for the mobile sign-up page."""

import json
import logging
import os
import secrets
import time
from typing import Any, Dict, Optional, Tuple

from aiohttp import web

from ..analytics import build_report, render_report
from ..models import EQUIPMENT, GENDERS, GOALS, LEVELS
from ..nutrition import build_nutrition_plan
from ..scheduler import compute_next_fire, format_days, parse_time_of_day
from ..subscription import plan_for
from ..workouts import (
    EQUIPMENT_LABELS,
    GOAL_LABELS,
    LEVEL_LABELS,
    VENUE_LABELS,
    default_training_days,
    generate_program,
)
from .auth import AuthService
from .delivery import build_deliverer

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LINK_CODE_TTL_SECONDS = 900

# Only the page's own inline script and styles may run; nothing loads from
# anywhere else, and the page cannot be framed.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
}

ERROR_MESSAGES = {
    "invalid_identifier": "Проверь номер телефона или адрес почты.",
    "cooldown": "Код уже отправлен. Запросить новый можно через {retry_after} сек.",
    "too_many_requests": "Слишком много попыток. Попробуй позже.",
    "no_code": "Сначала запроси код.",
    "expired": "Код истёк. Запроси новый.",
    "too_many_attempts": "Слишком много неверных попыток. Запроси новый код.",
    "wrong_code": "Неверный код. Осталось попыток: {attempts_left}.",
    "delivery_failed": "Не удалось отправить код. Попробуй другой способ связи.",
    "unauthorized": "Нужно войти заново.",
    "invalid_survey": "Проверь ответы анкеты.",
    "rate_limited": "Слишком много запросов. Подожди немного.",
}


def _message(error: str, **kwargs) -> str:
    template = ERROR_MESSAGES.get(error, "Что-то пошло не так. Попробуй ещё раз.")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def json_error(error: str, status: int = 400, **kwargs) -> web.Response:
    payload = {"error": error, "message": _message(error, **kwargs)}
    payload.update(kwargs)
    return web.json_response(payload, status=status)


def mask_identifier(channel: str, identifier: str) -> str:
    """Show just enough of the identifier for the user to recognise it."""
    if channel == "email":
        name, _, domain = identifier.partition("@")
        head = name[:2] if len(name) > 2 else name[:1]
        return f"{head}{'*' * max(1, len(name) - len(head))}@{domain}"
    return f"{identifier[:2]}{'*' * max(0, len(identifier) - 6)}{identifier[-4:]}"


class IpRateLimiter:
    """
    In-memory per-IP limiter for the unauthenticated endpoints.

    The per-identifier limits in `AuthService` stop one target being spammed;
    this stops one source spraying many targets.
    """

    def __init__(self, limit: int = 20, window_seconds: int = 600):
        self.limit = limit
        self.window = window_seconds
        self._hits: Dict[str, list] = {}

    def check(self, key: str, now: Optional[float] = None) -> bool:
        moment = now if now is not None else time.time()
        hits = [stamp for stamp in self._hits.get(key, []) if moment - stamp < self.window]
        if len(hits) >= self.limit:
            self._hits[key] = hits
            return False
        hits.append(moment)
        self._hits[key] = hits
        if len(self._hits) > 10000:  # Cheap guard against unbounded growth.
            self._hits = {key: hits}
        return True


def program_to_dict(program, profile) -> Dict[str, Any]:
    """Serialise a generated program for the web client."""
    # The venue badge only carries information when the plan mixes places.
    show_venue = program.equipment == "mix"
    return {
        "goal": program.goal,
        "goal_label": GOAL_LABELS.get(program.goal, program.goal),
        "level": program.level,
        "level_label": LEVEL_LABELS.get(program.level, program.level),
        "equipment": program.equipment,
        "equipment_label": EQUIPMENT_LABELS.get(program.equipment, program.equipment),
        "weeks": program.weeks,
        "notes": program.notes,
        "training_days": format_days(default_training_days(profile.days_per_week)),
        "workouts": [
            {
                "title": workout.title,
                "focus": workout.focus,
                "venue": workout.venue,
                "venue_label": VENUE_LABELS.get(workout.venue, "") if show_venue else "",
                "minutes": workout.estimated_minutes,
                "exercises": [
                    {
                        "name": exercise.name,
                        "sets": exercise.sets,
                        "reps": exercise.reps,
                        "rest_seconds": exercise.rest_seconds,
                    }
                    for exercise in workout.exercises
                ],
            }
            for workout in program.workouts
        ],
    }


def profile_to_dict(user) -> Dict[str, Any]:
    profile = user.profile
    return {
        "gender": profile.gender,
        "age": profile.age,
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
        "goal": profile.goal,
        "level": profile.level,
        "equipment": profile.equipment,
        "days_per_week": profile.days_per_week,
        "timezone_offset_minutes": profile.timezone_offset_minutes,
        "complete": profile.is_complete,
    }


class WebApp:
    """The aiohttp application serving the sign-up page and its API."""

    def __init__(self, config, storage, auth: Optional[AuthService] = None, deliverer=None):
        self.config = config
        self.storage = storage
        self.auth = auth or AuthService(storage, config.secret_key)
        self.deliverer = deliverer if deliverer is not None else build_deliverer(config)
        self.limiter = IpRateLimiter()
        self.app = self._build_app()

    def _build_app(self) -> web.Application:
        app = web.Application(middlewares=[self._security_middleware])
        app.router.add_get("/", self.page)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_post("/api/auth/request", self.auth_request)
        app.router.add_post("/api/auth/verify", self.auth_verify)
        app.router.add_get("/api/me", self.me)
        app.router.add_post("/api/survey", self.survey)
        app.router.add_get("/api/plan", self.plan)
        app.router.add_post("/api/telegram/link", self.telegram_link)
        if os.path.isdir(STATIC_DIR):
            app.router.add_static("/static/", STATIC_DIR)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _cleanup(self, app: web.Application) -> None:
        await self.deliverer.close()

    @web.middleware
    async def _security_middleware(self, request: web.Request, handler):
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = error
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _json_body(self, request: web.Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return {}
        return body if isinstance(body, dict) else {}

    def _client_key(self, request: web.Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.remote or "unknown"

    def _current_user(self, request: web.Request):
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        user_id = self.auth.user_id_from_token(token)
        if user_id is None:
            return None
        return self.storage.get_user(user_id)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    async def page(self, request: web.Request) -> web.Response:
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.isfile(path):
            return web.Response(text="UI is not installed", status=500)
        with open(path, encoding="utf-8") as handle:
            return web.Response(text=handle.read(), content_type="text/html")

    async def healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "users": self.storage.count_users()})

    async def auth_request(self, request: web.Request) -> web.Response:
        if not self.limiter.check(self._client_key(request)):
            return json_error("rate_limited", status=429)

        body = await self._json_body(request)
        result = self.auth.request_code(str(body.get("identifier", "")))
        if not result.ok:
            status = 429 if result.error in ("cooldown", "too_many_requests") else 400
            return json_error(result.error, status=status, retry_after=result.retry_after)

        delivery = await self.deliverer.send(result.channel, result.identifier, result.code)
        if not delivery.ok:
            # The code is useless if it never arrives; drop it so the user can
            # retry immediately instead of waiting out the cooldown.
            self.storage.delete_auth_code(result.identifier)
            return json_error("delivery_failed", status=502)

        payload = {
            "ok": True,
            "channel": result.channel,
            "masked": mask_identifier(result.channel, result.identifier),
        }
        # Development convenience, and only when nothing real can deliver.
        if self.config.dev_show_code and not self.deliverer.is_real(result.channel):
            payload["dev_code"] = result.code
        return web.json_response(payload)

    async def auth_verify(self, request: web.Request) -> web.Response:
        if not self.limiter.check(self._client_key(request)):
            return json_error("rate_limited", status=429)

        body = await self._json_body(request)
        result = self.auth.verify(
            str(body.get("identifier", "")), str(body.get("code", ""))
        )
        if not result.ok:
            return json_error(
                result.error, status=400, attempts_left=result.attempts_left
            )

        user = self.storage.get_user(result.user_id)
        return web.json_response(
            {
                "ok": True,
                "token": self.auth.issue_session(result.user_id),
                "created": result.created,
                "needs_survey": not user.profile.is_complete,
            }
        )

    async def me(self, request: web.Request) -> web.Response:
        user = self._current_user(request)
        if user is None:
            return json_error("unauthorized", status=401)

        plan = plan_for(user.subscription)
        reminders = self.storage.list_reminders(user.id)
        return web.json_response(
            {
                "profile": profile_to_dict(user),
                "plan": {"name": plan.name, "title": plan.title},
                "telegram_linked": user.telegram_id is not None,
                "reminders": [
                    {
                        "time": reminder.time_local,
                        "days": format_days(reminder.weekdays()),
                        "enabled": reminder.enabled,
                    }
                    for reminder in reminders
                ],
            }
        )

    async def survey(self, request: web.Request) -> web.Response:
        """Store the survey answers and hand back the generated plan."""
        user = self._current_user(request)
        if user is None:
            return json_error("unauthorized", status=401)

        body = await self._json_body(request)
        fields, error = parse_survey(body)
        if error:
            return json_error("invalid_survey", status=400, field=error)

        reminder_time = fields.pop("reminder_time", "")
        self.storage.update_profile(user.id, **fields)
        if fields.get("weight_kg"):
            self.storage.add_weight_log(user.id, fields["weight_kg"])

        fresh = self.storage.get_user(user.id)
        if reminder_time:
            self._schedule_reminder(fresh, reminder_time)
            fresh = self.storage.get_user(user.id)

        program = generate_program(fresh.profile, plan_for(fresh.subscription).program_weeks)
        return web.json_response(
            {
                "ok": True,
                "profile": profile_to_dict(fresh),
                "program": program_to_dict(program, fresh.profile),
                "reminders": [
                    {"time": item.time_local, "days": format_days(item.weekdays())}
                    for item in self.storage.list_reminders(fresh.id)
                ],
                "telegram_linked": fresh.telegram_id is not None,
            }
        )

    def _schedule_reminder(self, user, raw_time: str) -> None:
        """Create or retime the workout reminder implied by the profile."""
        time_local = parse_time_of_day(raw_time)
        if not time_local:
            return

        days = default_training_days(user.profile.days_per_week)
        days_text = ",".join(str(day) for day in days)
        offset = user.profile.timezone_offset_minutes
        next_fire = compute_next_fire(time_local, days, offset)

        existing = [
            item for item in self.storage.list_reminders(user.id) if item.kind == "workout"
        ]
        if existing:
            self.storage.update_reminder_schedule(
                existing[0].id, user.id, time_local, days_text, next_fire
            )
            self.storage.set_reminder_enabled(existing[0].id, user.id, True)
            return

        self.storage.add_reminder(
            user_id=user.id,
            kind="workout",
            time_local=time_local,
            days=days_text,
            text="",
            next_fire_at=next_fire,
        )

    async def plan(self, request: web.Request) -> web.Response:
        user = self._current_user(request)
        if user is None:
            return json_error("unauthorized", status=401)
        if not user.profile.is_complete:
            return web.json_response({"ok": False, "needs_survey": True})

        tier = plan_for(user.subscription)
        program = generate_program(user.profile, tier.program_weeks)
        payload = {
            "ok": True,
            "program": program_to_dict(program, user.profile),
            "plan": {"name": tier.name, "title": tier.title},
        }

        if tier.has("nutrition"):
            nutrition = build_nutrition_plan(user.profile)
            if nutrition is not None:
                payload["nutrition"] = {
                    "calories": nutrition.calories,
                    "protein_g": nutrition.protein_g,
                    "fat_g": nutrition.fat_g,
                    "carbs_g": nutrition.carbs_g,
                    "water_ml": nutrition.water_ml,
                }
        if tier.has("analytics"):
            report = build_report(
                self.storage.list_workout_logs(user.id),
                self.storage.list_weight_logs(user.id),
                user.profile.timezone_offset_minutes,
            )
            payload["progress"] = render_report(report)
        return web.json_response(payload)

    async def telegram_link(self, request: web.Request) -> web.Response:
        """Issue a one-time code that connects this account to the bot."""
        user = self._current_user(request)
        if user is None:
            return json_error("unauthorized", status=401)

        code = secrets.token_urlsafe(9)
        self.storage.create_link_code(
            user.id, code, time.time() + LINK_CODE_TTL_SECONDS
        )
        return web.json_response(
            {
                "ok": True,
                "code": code,
                "url": self.config.telegram_link(code),
                "expires_in": LINK_CODE_TTL_SECONDS,
            }
        )


def parse_survey(body: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Validate the survey payload.

    Returns (fields, "") on success, or ({}, offending field name).
    """
    fields: Dict[str, Any] = {}

    gender = str(body.get("gender", "other"))
    if gender not in GENDERS:
        return {}, "gender"
    fields["gender"] = gender

    for name, low, high, caster in (
        ("age", 14, 100, int),
        ("height_cm", 120, 230, int),
        ("weight_kg", 35, 300, float),
    ):
        raw = body.get(name)
        try:
            value = caster(str(raw).replace(",", "."))
        except (TypeError, ValueError):
            return {}, name
        if not low <= value <= high:
            return {}, name
        fields[name] = value

    for name, allowed in (("goal", GOALS), ("level", LEVELS), ("equipment", EQUIPMENT)):
        value = str(body.get(name, ""))
        if value not in allowed:
            return {}, name
        fields[name] = value

    try:
        days = int(body.get("days_per_week", 3))
    except (TypeError, ValueError):
        return {}, "days_per_week"
    if not 2 <= days <= 5:
        return {}, "days_per_week"
    fields["days_per_week"] = days

    try:
        offset = int(body.get("timezone_offset_minutes", 0))
    except (TypeError, ValueError):
        return {}, "timezone_offset_minutes"
    if not -840 <= offset <= 840:
        return {}, "timezone_offset_minutes"
    fields["timezone_offset_minutes"] = offset

    reminder_time = str(body.get("reminder_time", "") or "")
    if reminder_time and parse_time_of_day(reminder_time) is None:
        return {}, "reminder_time"
    fields["reminder_time"] = reminder_time

    return fields, ""
