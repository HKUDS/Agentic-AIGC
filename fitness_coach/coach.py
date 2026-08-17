"""The AI trainer: LLM-backed answers with a deterministic fallback.

When an OpenAI-compatible endpoint is configured the coach answers freely with
the user's profile and recent training history in context. When it is not
configured — or the call fails — the same questions still get useful answers
from a small rule-based knowledge base, so the free tier never shows an error
screen because someone forgot an API key.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .config import Config
from .models import Profile, User
from .workouts import EQUIPMENT_LABELS, GOAL_LABELS, LEVEL_LABELS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — опытный персональный фитнес-тренер и нутрициолог в Telegram-боте.

Правила:
- Отвечай по-русски, коротко и по делу: 3-8 предложений или компактный список.
- Опирайся на профиль пользователя и его историю тренировок, если они даны.
- Давай конкретику: упражнения, подходы, повторения, вес, темп, время отдыха.
- Ты не врач. При боли, травме, головокружении, беременности, хронических
  заболеваниях — прямо советуй обратиться к врачу и не ставь диагнозов.
- Никогда не предлагай анаболические стероиды, жёсткие голодания
  (менее 1200 ккал), приём препаратов и «сушку любой ценой».
- Не обещай конкретных сроков вроде «минус 10 кг за неделю».
- Без markdown-таблиц; можно простые списки и HTML-теги <b>, <i>.
"""


def build_profile_context(user: User, extra: str = "") -> str:
    """Render the user's profile and recent history as LLM context."""
    profile = user.profile
    parts = [
        "Профиль пользователя:",
        f"- имя: {user.first_name or 'не указано'}",
        f"- цель: {GOAL_LABELS.get(profile.goal, profile.goal)}",
        f"- уровень: {LEVEL_LABELS.get(profile.level, profile.level)}",
        f"- инвентарь: {EQUIPMENT_LABELS.get(profile.equipment, profile.equipment)}",
        f"- тренировок в неделю: {profile.days_per_week}",
    ]
    if profile.age:
        parts.append(f"- возраст: {profile.age}")
    if profile.height_cm:
        parts.append(f"- рост: {profile.height_cm} см")
    if profile.weight_kg:
        parts.append(f"- вес: {profile.weight_kg} кг")
    if extra:
        parts.append(extra)
    return "\n".join(parts)


class AICoach:
    """Answers user questions, with an LLM when available."""

    def __init__(self, config: Config, session: Optional[aiohttp.ClientSession] = None):
        self.config = config
        self._session = session
        self._owns_session = session is None

    async def close(self) -> None:
        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def answer(
        self,
        user: User,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_note: str = "",
    ) -> str:
        """
        Answer `question` for `user`.

        Never raises: any backend problem degrades to the offline knowledge
        base so the user always gets a reply.
        """
        if not self.config.llm_enabled:
            return offline_answer(question, user.profile)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": build_profile_context(user, context_note)},
        ]
        for item in history or []:
            if item.get("role") in ("user", "assistant") and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": question})

        try:
            return await self._chat_completion(messages)
        except Exception as error:
            logger.warning("LLM call failed, falling back offline: %s", error)
            return offline_answer(question, user.profile)

    async def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        session = await self._ensure_session()
        url = f"{self.config.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 700,
        }
        headers = {"Authorization": f"Bearer {self.config.llm_api_key}"}

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.config.llm_timeout_seconds),
        ) as response:
            if response.status >= 400:
                detail = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {detail[:200]}")
            data = await response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("empty completion")
        content = (choices[0].get("message") or {}).get("content") or ""
        content = content.strip()
        if not content:
            raise RuntimeError("empty completion content")
        return content

    async def personalise_program(self, user: User, base_program_text: str) -> str:
        """
        Pro feature: ask the LLM to adapt the generated program.

        Falls back to the rule-based program unchanged, which is already a
        valid plan — the Pro value in that case comes from the 4-week length
        and the progression notes.
        """
        if not self.config.llm_enabled:
            return base_program_text

        question = (
            "Ниже черновик программы, собранный по шаблону. Адаптируй его под "
            "профиль: замени неподходящие упражнения, поправь объём и добавь "
            "2-3 персональных совета в конце. Сохрани структуру по дням и "
            "форматирование HTML-тегами.\n\n" + base_program_text
        )
        try:
            return await self.answer(user, question)
        except Exception:  # pragma: no cover - answer() already degrades
            return base_program_text


# ----------------------------------------------------------------------
# Offline knowledge base
# ----------------------------------------------------------------------

_MEDICAL_KEYWORDS = (
    "боль", "болит", "травма", "колено", "спина", "хрустит", "защемил",
    "давление", "сердце", "головокружение", "беременн", "грыжа",
)

_TOPICS = (
    (
        ("белок", "белка", "протеин"),
        "Ориентир по белку — 1.6-2.2 г на кг веса в день. Разбей на 3-5 приёмов "
        "по 25-40 г: мясо, рыба, яйца, творог, бобовые. Протеиновый порошок — "
        "не обязателен, это просто удобный способ добрать норму.",
    ),
    (
        ("крепатур", "болят мышцы после", "ддомс", "doms"),
        "Мышечная боль через 1-2 дня после нагрузки — норма, особенно после новых "
        "упражнений. Помогают лёгкое кардио 15-20 минут, растяжка, сон и вода. "
        "Тренироваться можно, но снизь рабочий вес на 20-30%, пока боль не уйдёт.",
    ),
    (
        ("разминк", "разогрев", "заминк"),
        "Разминка — 8-10 минут: 3-5 минут лёгкого кардио, суставная гимнастика "
        "сверху вниз и 1-2 подхода первого упражнения с пустым грифом или "
        "половинным весом. Заминка — 5 минут спокойной растяжки на то, что "
        "работало.",
    ),
    (
        ("похуд", "сбросить", "жир", "дефицит"),
        "Жир уходит только при дефиците калорий: 300-500 ккал ниже нормы даёт "
        "0.4-0.7 кг в неделю — это устойчивый темп. Держи белок высоким, силовые "
        "2-3 раза в неделю, чтобы терять жир, а не мышцы, и добавь 7-10 тысяч "
        "шагов в день. Взвешивайся раз в неделю в одинаковых условиях.",
    ),
    (
        ("масс", "набрать", "мышц", "гипертроф"),
        "Для роста мышц нужен небольшой профицит (+10-15% калорий), белок "
        "1.6-2.2 г/кг, 10-20 рабочих подходов на мышечную группу в неделю и "
        "прогрессия нагрузки: каждую неделю +2 повторения или +2.5 кг. Сон 7-9 "
        "часов важнее любой добавки.",
    ),
    (
        ("мотивац", "лень", "не хочу", "бросил"),
        "Мотивация приходит после старта, а не до него. Три рабочих приёма: "
        "снизь планку до «10 минут разминки и решу дальше», поставь тренировки в "
        "календарь как встречи и отмечай выполненные в боте — серия не даёт "
        "сорваться. Пропуск одной тренировки ничего не ломает, пропуск двух "
        "подряд — уже привычка.",
    ),
    (
        ("плато", "вес стоит", "не уходит"),
        "Плато в 1-2 недели — это норма, вес колеблется из-за воды и гликогена. "
        "Если движения нет 3 недели подряд: перепроверь реальную калорийность "
        "(взвесь еду неделю), добавь шагов, поспи больше. Резать калории ещё "
        "сильнее — последнее, к чему стоит прибегать.",
    ),
    (
        ("сколько раз", "как часто", "частот"),
        "Оптимум для большинства — 3-4 силовые тренировки в неделю с днём отдыха "
        "между ними. Новичку хватит 3 занятий на всё тело: так каждая мышечная "
        "группа получает нагрузку трижды в неделю и успевает восстановиться.",
    ),
    (
        ("вода", "пить"),
        "Ориентир — 30 мл воды на кг веса в день плюс 0.5-0.7 л на каждый час "
        "тренировки. Простой индикатор — светлая моча и отсутствие жажды к "
        "вечеру.",
    ),
    (
        ("кардио",),
        "Кардио не мешает росту мышц, если это 2-3 сессии по 20-40 минут в "
        "спокойном темпе и не сразу перед силовой. Для жиросжигания ходьба "
        "работает не хуже бега и меньше нагружает суставы.",
    ),
    (
        ("сон", "восстановл"),
        "Восстановление — половина результата: 7-9 часов сна, день отдыха между "
        "тяжёлыми тренировками, разгрузочная неделя каждые 4-6 недель. Если "
        "пульс покоя вырос на 5-10 ударов и пропал аппетит — ты недовосстановлен, "
        "снизь объём.",
    ),
)

_DEFAULT_ANSWER = (
    "Я на связи. Чтобы ответить точнее, уточни детали: что именно не получается, "
    "сколько ты уже тренируешься и какой у тебя инвентарь.\n\n"
    "Базовые ориентиры: 3 силовые тренировки в неделю, белок 1.6-2.2 г на кг веса, "
    "сон 7-9 часов и прогрессия нагрузки — каждую неделю чуть больше повторений "
    "или веса. Программу можно взять в разделе «Моя программа»."
)

_MEDICAL_ANSWER = (
    "Если что-то болит — это стоп-сигнал, а не повод «дотерпеть». Останови "
    "упражнение, которое вызывает боль, и покажись врачу или спортивному "
    "реабилитологу: я не могу поставить диагноз по переписке.\n\n"
    "Пока ждёшь приём: не нагружай проблемную зону, оставь безболезненные "
    "движения на другие группы мышц и следи, не усиливается ли боль в покое."
)


def offline_answer(question: str, profile: Optional[Profile] = None) -> str:
    """Rule-based answer used when no LLM is configured or the call failed."""
    text = (question or "").lower()

    if any(keyword in text for keyword in _MEDICAL_KEYWORDS):
        return _MEDICAL_ANSWER

    for keywords, answer in _TOPICS:
        if any(keyword in text for keyword in keywords):
            return answer

    if profile is not None and profile.goal in GOAL_LABELS:
        goal_hint = (
            f"\n\nТвоя цель сейчас — {GOAL_LABELS[profile.goal]}, "
            f"{profile.days_per_week} тренировки в неделю."
        )
        return _DEFAULT_ANSWER + goal_hint
    return _DEFAULT_ANSWER


async def warm_up_backend(coach: AICoach, timeout: float = 5.0) -> bool:
    """
    Best-effort check that the LLM backend answers.

    Returns False (rather than raising) when the backend is unusable, letting
    the bot log a warning at start-up and keep running in offline mode.
    """
    if not coach.config.llm_enabled:
        return False
    try:
        await asyncio.wait_for(
            coach._chat_completion(
                [{"role": "user", "content": "ping"}]
            ),
            timeout=timeout,
        )
        return True
    except Exception as error:
        logger.warning("LLM backend is not reachable: %s", error)
        return False


def usage_summary(used_today: int, limit: Optional[int], now: Optional[float] = None) -> str:
    """One-line quota footer appended to free-tier AI answers."""
    if limit is None:
        return ""
    left = max(0, limit - used_today)
    _ = now or time.time()
    return f"\n\n<i>Осталось сообщений сегодня: {left} из {limit}. Pro — без лимита.</i>"
