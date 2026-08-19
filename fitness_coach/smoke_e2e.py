"""End-to-end smoke check: the real bot against a fake Telegram Bot API.

Run it with `python -m fitness_coach.smoke_e2e` (add `--transcript` to print
every message the bot sends). Unlike the unit tests, nothing here is mocked
inside the bot: real aiohttp requests, the real polling loop, real offset
persistence and the real reminder scheduler all run -- only api.telegram.org is
replaced by a local stub server, and no bot token is needed.

Exits 0 when every check passes, 1 otherwise, so it can gate a deploy.
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import time

import aiohttp
from aiohttp import web

from . import telegram_api
from .bot import FitnessBot
from .coach import AICoach
from .config import Config
from .storage import Storage
from .telegram_api import TelegramClient
from .webapp.auth import AuthService
from .webapp.delivery import Deliverer, DeliveryResult
from .webapp.server import WebApp


class CapturingDeliverer(Deliverer):
    """Captures verification codes so the check can read them back."""

    def __init__(self):
        self.sent = []

    async def send(self, channel, identifier, code):
        self.sent.append({"channel": channel, "identifier": identifier, "code": code})
        return DeliveryResult(True)

    def is_real(self, channel):
        return True

TOKEN = "111:FAKE"
CHAT = 500100
USER = {"id": 777001, "first_name": "Иван", "username": "ivan", "is_bot": False}

sent = []          # every outgoing API call the bot made
queue = []         # updates waiting to be polled
update_id = [1000]
failures = []


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(f"{label}: {detail}")
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))


async def api(request):
    method = request.match_info["method"]
    if request.content_type.startswith("multipart/"):
        # sendDocument uploads a real multipart body.
        reader = await request.post()
        body = {key: (value if isinstance(value, str) else value.filename)
                for key, value in reader.items()}
    else:
        body = await request.json()

    if method == "getUpdates":
        deadline = time.time() + 0.4
        while not queue and time.time() < deadline:
            await asyncio.sleep(0.02)
        batch, queue[:] = list(queue), []
        return web.json_response({"ok": True, "result": batch})

    sent.append((method, body))
    if method == "sendMessage":
        return web.json_response(
            {"ok": True, "result": {"message_id": len(sent), "text": body["text"]}}
        )
    return web.json_response({"ok": True, "result": True})


def push(update):
    update_id[0] += 1
    update["update_id"] = update_id[0]
    queue.append(update)


def text_update(text):
    return {"message": {"message_id": update_id[0], "chat": {"id": CHAT}, "from": USER, "text": text}}


def tap_update(data):
    return {
        "callback_query": {
            "id": f"cb{update_id[0]}",
            "from": USER,
            "message": {"message_id": update_id[0], "chat": {"id": CHAT}},
            "data": data,
        }
    }


def messages():
    return [payload["text"] for method, payload in sent if method == "sendMessage"]


async def step(update, timeout=6.0):
    """Send one update and return the messages the bot replied with."""
    before = len(messages())
    push(update)
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(0.05)
        if len(messages()) > before:
            # Give the bot a moment to finish a multi-message reply.
            await asyncio.sleep(0.35)
            break
    return messages()[before:]


async def main():
    app = web.Application()
    app.router.add_post("/bot{token}/{method}", api)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    telegram_api.API_ROOT = f"http://127.0.0.1:{port}"

    workdir = tempfile.mkdtemp()
    db_path = os.path.join(workdir, "bot.sqlite3")
    config = Config(
        telegram_token=TOKEN,
        database_path=db_path,
        pro_price_stars=250,
        pro_period_days=30,
        promo_codes=["WELCOME"],
        reminder_tick_seconds=5,
        poll_timeout_seconds=1,
        secret_key="smoke-test-secret",
        bot_username="smoke_fitness_bot",
    )
    storage = Storage(db_path)
    client = TelegramClient(TOKEN)
    bot = FitnessBot(config, storage, client, AICoach(config))
    task = asyncio.create_task(bot.run())
    await asyncio.sleep(0.5)

    transcript = []

    def log(title, replies):
        transcript.append((title, replies))
        return replies

    print("\n=== 1. Регистрация ===")
    r = log("/start", await step(text_update("/start")))
    check("бот отвечает на /start", any("Привет, Иван" in m for m in r), r[:1])
    check("спрашивает пол", any("пол" in m.lower() for m in r))

    log("пол", await step(tap_update("onb_gender:male")))
    log("возраст", await step(text_update("34")))
    log("рост", await step(text_update("182")))
    log("вес", await step(text_update("88.5")))
    log("цель", await step(tap_update("onb_goal:lose_weight")))
    log("уровень", await step(tap_update("onb_level:beginner")))
    log("инвентарь", await step(tap_update("onb_equipment:home")))
    r = log("дни", await step(tap_update("onb_days:4")))
    check("после дней спрашивает часовой пояс", any("часовой пояс" in m for m in r))

    r = log("часовой пояс", await step(text_update("+3")))
    check("предлагает время напоминания", any("Во сколько" in m for m in r))
    check("дни взяты из профиля (4/нед)", any("Пн, Вт, Чт, Пт" in m for m in r), r)

    r = log("время напоминания", await step(tap_update("onb_reminder:07:00")))
    joined = "\n".join(r)
    check("напоминания включены", "Напоминания включены" in joined)
    check("профиль готов", "Профиль готов" in joined)
    check("сразу показал тренировку", "Разминка" in joined)

    probe = sqlite3.connect(db_path)
    probe.row_factory = sqlite3.Row
    profile = probe.execute("SELECT * FROM profiles").fetchone()
    check("профиль в базе", (profile["age"], profile["height_cm"], profile["weight_kg"],
                             profile["goal"], profile["days_per_week"],
                             profile["timezone_offset_minutes"])
          == (34, 182, 88.5, "lose_weight", 4, 180), dict(profile))

    reminder = probe.execute("SELECT * FROM reminders").fetchone()
    check("напоминание создано", reminder is not None)
    check("время 07:00 и дни 0,1,3,4",
          (reminder["time_local"], reminder["days"]) == ("07:00", "0,1,3,4"),
          dict(reminder) if reminder else "")
    check("следующий запуск в будущем", reminder["next_fire_at"] > time.time())

    print("\n=== 2. Тренировка и журнал ===")
    r = log("/today", await step(text_update("/today")))
    check("выдал тренировку", any("Разминка" in m for m in r))
    check("кардио есть (цель — похудение)",
          any("мин" in m for m in r))

    await step(text_update("/log"))
    r = log("оценка тяжести", await step(tap_update("log_difficulty:4")))
    check("тренировка записана", any("Записал" in m for m in r), r)
    logs = probe.execute("SELECT * FROM workout_logs").fetchall()
    check("лог в базе", len(logs) == 1 and logs[0]["difficulty"] == 4)

    r = log("/weight 87.2", await step(text_update("/weight 87.2")))
    check("вес записан", any("87.2" in m for m in r))

    print("\n=== 3. ИИ-тренер и лимит Free ===")
    r = log("вопрос", await step(text_update("Сколько белка нужно есть?")))
    check("ответил по существу", any("г на кг" in m for m in r), r)
    check("показал остаток квоты", any("4 из 5" in m for m in r), r)

    r = log("вопрос про боль", await step(text_update("болит колено при приседе")))
    check("боль -> к врачу", any("врач" in m.lower() for m in r), r)

    for _ in range(3):
        await step(text_update("а что по кардио?"))
    r = log("6-й вопрос", await step(text_update("ещё вопрос")))
    check("лимит Free сработал", any("закончились" in m for m in r), r)

    print("\n=== 4. Платные функции закрыты ===")
    r = log("/nutrition (free)", await step(text_update("/nutrition")))
    check("питание закрыто", any("функция Pro" in m for m in r), r)
    r = log("/progress (free)", await step(text_update("/progress")))
    check("аналитика закрыта", any("функция Pro" in m for m in r), r)

    print("\n=== 5. Оплата Telegram Stars ===")
    r = log("/subscribe", await step(text_update("/subscribe")))
    check("показал пейволл с ценой", any("250" in m for m in r), r)

    before_invoices = sum(1 for m, _ in sent if m == "sendInvoice")
    push(tap_update("pay:stars"))
    await asyncio.sleep(0.8)
    invoices = [p for m, p in sent if m == "sendInvoice"]
    check("счёт выставлен", len(invoices) > before_invoices)
    check("валюта XTR (Stars)", invoices and invoices[-1]["currency"] == "XTR",
          invoices[-1] if invoices else "")
    check("цена 250", invoices and invoices[-1]["prices"][0]["amount"] == 250)

    push({"pre_checkout_query": {"id": "pcq1", "from": USER,
                                 "invoice_payload": "pro_subscription_v1",
                                 "currency": "XTR", "total_amount": 250}})
    await asyncio.sleep(0.8)
    pre = [p for m, p in sent if m == "answerPreCheckoutQuery"]
    check("pre-checkout подтверждён", pre and pre[-1]["ok"] is True, pre)

    payment = {"message": {"message_id": 900, "chat": {"id": CHAT}, "from": USER,
                           "successful_payment": {
                               "currency": "XTR", "total_amount": 250,
                               "invoice_payload": "pro_subscription_v1",
                               "telegram_payment_charge_id": "charge_abc"}}}
    r = log("оплата", await step(payment))
    check("Pro активирован", any("Pro активирован" in m for m in r), r)

    subscription = probe.execute("SELECT * FROM subscriptions").fetchone()
    check("подписка pro в базе", subscription["plan"] == "pro")
    check("срок ~30 дней",
          abs(subscription["expires_at"] - (time.time() + 30 * 86400)) < 120)

    r = log("повторная доставка платежа", await step(payment))
    check("повторный платёж не удваивает", any("уже был учтён" in m for m in r), r)

    print("\n=== 6. Pro-функции открылись ===")
    r = log("/nutrition (pro)", await step(text_update("/nutrition")))
    joined = "\n".join(r)
    check("расчёт калорий выдан", "ккал" in joined and "Белки" in joined)
    check("есть дефицит для похудения", "Цель по калориям" in joined)

    r = log("/progress (pro)", await step(text_update("/progress")))
    check("аналитика выдана", any("Всего тренировок" in m for m in r), r)

    r = log("вопрос на Pro", await step(text_update("сколько белка?")))
    check("лимит снят", not any("закончились" in m for m in r), r)
    check("нет счётчика квоты", not any("Осталось сообщений" in m for m in r))

    before_docs = sum(1 for m, _ in sent if m == "sendDocument")
    push(text_update("/export"))
    await asyncio.sleep(1.0)
    check("экспорт отправлен файлом",
          sum(1 for m, _ in sent if m == "sendDocument") > before_docs)

    print("\n=== 7. Планировщик реально шлёт напоминание ===")
    probe.execute("UPDATE reminders SET next_fire_at = ?", (time.time() - 1,))
    probe.commit()
    before = len(messages())
    deadline = time.time() + 12
    fired = []
    while time.time() < deadline:
        await asyncio.sleep(0.3)
        new = messages()[before:]
        if any("Время тренировки" in m for m in new):
            fired = new
            break
    check("напоминание доставлено планировщиком", bool(fired), "не пришло за 12 сек")

    row = probe.execute("SELECT * FROM reminders").fetchone()
    check("напоминание перепланировано вперёд", row["next_fire_at"] > time.time())

    print("\n=== 8. Веб-регистрация и связка с ботом ===")
    deliverer = CapturingDeliverer()
    webapp = WebApp(config, storage, auth=AuthService(storage, config.secret_key),
                    deliverer=deliverer)
    web_runner = web.AppRunner(webapp.app)
    await web_runner.setup()
    web_site = web.TCPSite(web_runner, "127.0.0.1", 0)
    await web_site.start()
    base = f"http://127.0.0.1:{web_site._server.sockets[0].getsockname()[1]}"

    async with aiohttp.ClientSession() as http:
        async def call(method, path, payload=None, token=None):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            async with http.request(
                method, base + path, json=payload, headers=headers
            ) as response:
                return response.status, await response.json()

        status, _ = await call("GET", "/healthz")
        check("страница отвечает", status == 200)

        status, data = await call("POST", "/api/auth/request",
                                  {"identifier": "8 (999) 111-22-33"})
        check("код запрошен по телефону", status == 200 and data.get("channel") == "phone",
              data)
        check("номер нормализован",
              deliverer.sent and deliverer.sent[-1]["identifier"] == "+79991112233",
              deliverer.sent[-1] if deliverer.sent else "")
        check("код не возвращается клиенту", "dev_code" not in data)
        check("номер замаскирован", data.get("masked", "").endswith("2233"), data)

        status, blocked = await call("POST", "/api/auth/request",
                                     {"identifier": "+79991112233"})
        check("повторный запрос ограничен", status == 429 and blocked["error"] == "cooldown")

        status, wrong = await call("POST", "/api/auth/verify",
                                   {"identifier": "+79991112233", "code": "000000"})
        check("неверный код отклонён", status == 400 and wrong["error"] == "wrong_code")

        status, session = await call(
            "POST", "/api/auth/verify",
            {"identifier": "+79991112233", "code": deliverer.sent[-1]["code"]},
        )
        token = session.get("token", "")
        check("вход по коду выполнен", status == 200 and bool(token), session)
        check("новому аккаунту предложен опрос", session.get("needs_survey") is True)

        status, _ = await call("GET", "/api/me")
        check("без токена доступа нет", status == 401)

        status, plan = await call("POST", "/api/survey", {
            "gender": "female", "age": 28, "height_cm": 168, "weight_kg": 63.0,
            "goal": "keep_fit", "level": "beginner", "equipment": "mix",
            "days_per_week": 4, "timezone_offset_minutes": 180,
            "reminder_time": "08:30",
        }, token=token)
        check("опрос принят и план собран", status == 200 and plan.get("ok") is True, plan)

        venues = [w["venue"] for w in plan["program"]["workouts"]]
        check("микс чередует зал и дом", venues == ["gym", "home", "gym", "home"], venues)
        check("в плане 4 тренировки", len(plan["program"]["workouts"]) == 4)
        check("напоминание создано опросом",
              plan["reminders"] and plan["reminders"][0]["time"] == "08:30",
              plan.get("reminders"))
        check("дни напоминания из профиля",
              plan["reminders"][0]["days"] == "Пн, Вт, Чт, Пт", plan["reminders"])

        status, bad = await call("POST", "/api/survey", {
            "gender": "male", "age": 5, "height_cm": 180, "weight_kg": 80,
            "goal": "keep_fit", "level": "beginner", "equipment": "gym",
            "days_per_week": 3,
        }, token=token)
        check("некорректная анкета отклонена", status == 400 and bad.get("field") == "age", bad)

        status, link = await call("POST", "/api/telegram/link", {}, token=token)
        check("ссылка на бота выдана",
              status == 200 and link["url"].startswith("https://t.me/smoke_fitness_bot?start="),
              link)

    web_user = storage.get_user_by_identifier("phone", "+79991112233")
    check("веб-аккаунт в базе", web_user is not None and web_user.telegram_id is None)

    users_before = storage.count_users()
    second_user = {"id": 777002, "first_name": "Мария", "username": "maria", "is_bot": False}
    push({"message": {"message_id": 1, "chat": {"id": CHAT + 1}, "from": second_user,
                      "text": f"/start {link['code']}"}})
    await asyncio.sleep(1.2)

    linked = storage.get_user(web_user.id)
    check("Telegram привязан к веб-аккаунту", linked.telegram_id == 777002,
          f"telegram_id={linked.telegram_id}")
    check("профиль с сайта сохранён",
          (linked.profile.equipment, linked.profile.weight_kg) == ("mix", 63.0),
          f"{linked.profile.equipment}/{linked.profile.weight_kg}")
    check("дубликат аккаунта не создан", storage.count_users() == users_before,
          f"{users_before} -> {storage.count_users()}")

    probe.execute("UPDATE reminders SET next_fire_at = ? WHERE user_id = ?",
                  (time.time() - 1, web_user.id))
    probe.commit()
    before = len(messages())
    deadline = time.time() + 12
    delivered = False
    while time.time() < deadline:
        await asyncio.sleep(0.3)
        for method, payload in sent[-40:]:
            if method == "sendMessage" and payload.get("chat_id") == 777002:
                delivered = True
        if delivered:
            break
    check("напоминание дошло в привязанный Telegram", delivered, "не пришло за 12 сек")

    await web_runner.cleanup()

    print("\n=== 9. Устойчивость ===")
    r = log("неизвестная команда", await step(text_update("/wat")))
    check("показал справку", any("Команды" in m for m in r))
    offset = probe.execute("SELECT value FROM meta WHERE key='update_offset'").fetchone()
    check("offset сохранён в базе", offset is not None and int(offset["value"]) > 1000,
          dict(offset) if offset else "")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await client.close()
    storage.close()
    probe.close()
    await runner.cleanup()

    if "--transcript" in sys.argv:
        print("\n" + "=" * 62)
        print("ПЕРЕПИСКА С БОТОМ\n")
        for title, replies in transcript:
            print(f"--- пользователь: {title}")
            for reply in replies:
                print(f"    бот: {reply}\n")

    print("\n" + "=" * 62)
    print(f"Всего вызовов Bot API: {len(sent)}, сообщений пользователю: {len(messages())}")
    if failures:
        print(f"ПРОВАЛЕНО ПРОВЕРОК: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
