"""Rule-based workout program generator.

The generator is fully deterministic: the same profile always produces the same
program. That keeps the free tier usable without an LLM, makes the output
testable, and gives the AI coach (see `coach.py`) a solid draft to personalise
for Pro users instead of inventing a program from scratch.
"""

import random
from typing import Dict, List, Sequence, Tuple

from .models import Exercise, Profile, Program, Workout

# Exercise library: equipment -> muscle group -> (name, minimum level rank).
# Level ranks: 0 beginner, 1 intermediate, 2 advanced.
_LIBRARY: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "none": {
        "legs": [
            ("Приседания с собственным весом", 0),
            ("Выпады на месте", 0),
            ("Ягодичный мостик", 0),
            ("Болгарские приседания", 1),
            ("Приседания-пистолетик", 2),
        ],
        "push": [
            ("Отжимания с колен", 0),
            ("Отжимания от пола", 0),
            ("Отжимания с узкой постановкой рук", 1),
            ("Отжимания с ногами на возвышении", 1),
            ("Отжимания в стойке у стены", 2),
        ],
        "pull": [
            ("Тяга полотенца к поясу (изометрия)", 0),
            ("Обратная гиперэкстензия лёжа", 0),
            ("Подтягивания австралийские (под столом)", 1),
            ("Подтягивания прямым хватом", 2),
        ],
        "core": [
            ("Планка", 0),
            ("Скручивания", 0),
            ("Боковая планка", 1),
            ("Подъёмы ног лёжа", 1),
            ("Уголок (L-sit)", 2),
        ],
        "cardio": [
            ("Ходьба быстрым шагом", 0),
            ("Джампинг-джек", 0),
            ("Бёрпи", 1),
            ("Скакалка", 1),
            ("Интервальный бег 30/30", 2),
        ],
    },
    "home": {
        "legs": [
            ("Приседания с гантелями", 0),
            ("Выпады с гантелями", 0),
            ("Румынская тяга с гантелями", 1),
            ("Болгарские приседания с гантелями", 1),
            ("Приседания с гантелью на груди (гоблет)", 0),
        ],
        "push": [
            ("Жим гантелей лёжа на полу", 0),
            ("Жим гантелей стоя", 0),
            ("Разводка гантелей лёжа", 1),
            ("Отжимания с гантелями (нейтральный хват)", 1),
        ],
        "pull": [
            ("Тяга гантели в наклоне", 0),
            ("Тяга гантелей к поясу двумя руками", 0),
            ("Тяга резинки к груди", 0),
            ("Подтягивания на турнике", 2),
        ],
        "core": [
            ("Планка", 0),
            ("Русский твист с гантелью", 1),
            ("Подъёмы ног лёжа", 1),
            ("Скручивания", 0),
        ],
        "cardio": [
            ("Джампинг-джек", 0),
            ("Скакалка", 1),
            ("Бёрпи", 1),
            ("Ходьба быстрым шагом", 0),
        ],
    },
    "gym": {
        "legs": [
            ("Приседания со штангой", 1),
            ("Жим ногами в тренажёре", 0),
            ("Румынская тяга со штангой", 1),
            ("Разгибания ног в тренажёре", 0),
            ("Сгибания ног в тренажёре", 0),
            ("Становая тяга", 2),
        ],
        "push": [
            ("Жим штанги лёжа", 1),
            ("Жим гантелей на наклонной скамье", 0),
            ("Жим штанги стоя", 1),
            ("Разводка в тренажёре «бабочка»", 0),
            ("Отжимания на брусьях", 2),
        ],
        "pull": [
            ("Тяга верхнего блока к груди", 0),
            ("Тяга горизонтального блока", 0),
            ("Тяга штанги в наклоне", 1),
            ("Подтягивания прямым хватом", 2),
            ("Подъём штанги на бицепс", 0),
        ],
        "core": [
            ("Планка", 0),
            ("Скручивания на римском стуле", 1),
            ("Подъём ног в висе", 2),
            ("Скручивания в блоке", 1),
        ],
        "cardio": [
            ("Беговая дорожка, ровный темп", 0),
            ("Эллипс, ровный темп", 0),
            ("Гребной тренажёр, интервалы", 1),
            ("Велотренажёр, интервалы", 1),
        ],
    },
}

_LEVEL_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}

# Goal -> (sets, reps, rest seconds, cardio blocks per workout).
_GOAL_SCHEME: Dict[str, Tuple[int, str, int, int]] = {
    "lose_weight": (3, "12-15", 45, 2),
    "build_muscle": (4, "8-12", 90, 0),
    "keep_fit": (3, "10-12", 60, 1),
    "endurance": (3, "15-20", 30, 2),
}

# Splits keyed by workouts per week: (title, [muscle groups in order]).
_SPLITS: Dict[int, Sequence[Tuple[str, str, Sequence[str]]]] = {
    2: (
        ("День A — всё тело", "всё тело", ("legs", "push", "pull", "core")),
        ("День B — всё тело", "всё тело", ("push", "legs", "pull", "core")),
    ),
    3: (
        ("День A — ноги и корпус", "ноги", ("legs", "legs", "core", "core")),
        ("День B — жимовой", "грудь, плечи, трицепс", ("push", "push", "core", "core")),
        ("День C — тяговый", "спина и бицепс", ("pull", "pull", "core", "core")),
    ),
    4: (
        ("День A — верх (жим)", "грудь, плечи, трицепс", ("push", "push", "core")),
        ("День B — низ", "ноги и ягодицы", ("legs", "legs", "core")),
        ("День C — верх (тяга)", "спина и бицепс", ("pull", "pull", "core")),
        ("День D — низ и корпус", "ноги и пресс", ("legs", "core", "core")),
    ),
    5: (
        ("День A — жимовой", "грудь, плечи, трицепс", ("push", "push", "core")),
        ("День B — тяговый", "спина и бицепс", ("pull", "pull", "core")),
        ("День C — ноги", "ноги и ягодицы", ("legs", "legs", "core")),
        ("День D — верх", "грудь и спина", ("push", "pull", "core")),
        ("День E — низ и пресс", "ноги и пресс", ("legs", "core", "core")),
    ),
}

# Human labels used when the split names a focus, kept separate from the split
# table so translations stay in one place.
_FOCUS_FALLBACK = "всё тело"


# "mix" alternates venues day by day instead of blending the libraries: a
# workout you can only half-do because the equipment is elsewhere is useless.
_VENUE_ROTATION = ("gym", "home")
_VENUE_SUFFIX = {"gym": " · в зале", "home": " · дома"}


def equipment_for_day(equipment: str, day_index: int) -> str:
    """Which equipment set a given training day uses."""
    if equipment != "mix":
        return equipment
    return _VENUE_ROTATION[day_index % len(_VENUE_ROTATION)]


def _clamp_days(days_per_week: int) -> int:
    if days_per_week < 2:
        return 2
    if days_per_week > 5:
        return 5
    return days_per_week


def _pick_exercises(
    rng: random.Random,
    equipment: str,
    muscle_group: str,
    level: str,
    used: set,
) -> Exercise:
    """Choose one not-yet-used exercise for the group, respecting the level."""
    library = _LIBRARY.get(equipment, _LIBRARY["none"])
    pool = library.get(muscle_group) or _LIBRARY["none"][muscle_group]
    max_rank = _LEVEL_RANK.get(level, 0)

    allowed = [name for name, rank in pool if rank <= max_rank]
    if not allowed:
        # A beginner in a group whose easiest option is above their level still
        # gets the easiest available movement rather than an empty workout.
        allowed = [min(pool, key=lambda item: item[1])[0]]

    fresh = [name for name in allowed if name not in used]
    candidates = fresh or allowed
    name = rng.choice(sorted(candidates))
    used.add(name)
    return Exercise(
        name=name,
        sets=0,
        reps="",
        rest_seconds=0,
        muscle_group=muscle_group,
        equipment=equipment,
    )


def generate_program(profile: Profile, weeks: int = 1) -> Program:
    """
    Build a training program for `profile`.

    Args:
        profile: The user's goal, level, equipment, and weekly availability.
        weeks: Program length. The free tier passes 1, Pro passes 4.

    Returns:
        A `Program` whose workouts cover a single week; `weeks` describes how
        many times the week repeats with progressive overload (see notes).
    """
    days = _clamp_days(profile.days_per_week)
    split = _SPLITS[days]
    sets, reps, rest, cardio_blocks = _GOAL_SCHEME.get(
        profile.goal, _GOAL_SCHEME["keep_fit"]
    )

    # Seeded on the profile so the program is stable between calls but still
    # differs between users.
    seed = f"{profile.user_id}|{profile.goal}|{profile.level}|{profile.equipment}|{days}"
    rng = random.Random(seed)

    workouts: List[Workout] = []
    for index, (title, focus, groups) in enumerate(split):
        venue = equipment_for_day(profile.equipment, index)
        used: set = set()
        exercises: List[Exercise] = []
        for group in groups:
            exercise = _pick_exercises(rng, venue, group, profile.level, used)
            exercise.sets = sets
            exercise.reps = reps
            exercise.rest_seconds = rest
            exercises.append(exercise)

        for _ in range(cardio_blocks):
            cardio = _pick_exercises(rng, venue, "cardio", profile.level, used)
            cardio.sets = 1
            cardio.reps = "8-12 мин"
            cardio.rest_seconds = 60
            exercises.append(cardio)

        # Rough duration: working time plus rest, plus 10 minutes warm-up.
        minutes = 10
        for exercise in exercises:
            minutes += exercise.sets * (1 + exercise.rest_seconds / 60.0)

        workouts.append(
            Workout(
                day_index=index,
                title=title,
                focus=focus or _FOCUS_FALLBACK,
                exercises=exercises,
                estimated_minutes=int(round(minutes)),
                venue=venue,
            )
        )

    notes = _progression_note(profile.goal, weeks)
    return Program(
        goal=profile.goal,
        level=profile.level,
        equipment=profile.equipment,
        weeks=max(1, weeks),
        workouts=workouts,
        notes=notes,
    )


def _progression_note(goal: str, weeks: int) -> str:
    if weeks <= 1:
        return (
            "Прогрессия: каждую следующую неделю добавляй по 1-2 повторения "
            "в подходе, пока не дойдёшь до верхней границы диапазона."
        )
    if goal == "build_muscle":
        return (
            "Прогрессия по неделям: 1 — освоение техники, 2 — +1 подход в базовых "
            "упражнениях, 3 — +2.5-5 кг или +2 повторения, 4 — разгрузка "
            "(-30% объёма) перед новым циклом."
        )
    if goal == "lose_weight":
        return (
            "Прогрессия по неделям: 1 — база, 2 — сокращаем отдых до 40 сек, "
            "3 — +1 кардио-блок, 4 — разгрузка и контрольное взвешивание."
        )
    return (
        "Прогрессия по неделям: 1 — база, 2 — +1 повторение, 3 — +1 подход "
        "в первом упражнении дня, 4 — лёгкая неделя для восстановления."
    )


def render_workout(workout: Workout, show_venue: bool = False) -> str:
    """
    Format a single workout for a Telegram message.

    `show_venue` appends "· в зале" / "· дома" — only useful for a mixed
    program, where the place changes from day to day.
    """
    suffix = _VENUE_SUFFIX.get(workout.venue, "") if show_venue else ""
    lines = [f"<b>{workout.title}{suffix}</b>", f"Фокус: {workout.focus}", ""]
    lines.append("Разминка: 8-10 минут — суставная гимнастика и лёгкое кардио.")
    lines.append("")
    for number, exercise in enumerate(workout.exercises, start=1):
        lines.append(f"{number}. {exercise.render()}")
    lines.append("")
    lines.append(f"Ориентировочная длительность: ~{workout.estimated_minutes} мин.")
    lines.append("Заминка: 5 минут растяжки на проработанные мышцы.")
    return "\n".join(lines)


def render_program(program: Program) -> str:
    """Format a whole program for a Telegram message."""
    header = (
        f"<b>Твоя программа</b>\n"
        f"Цель: {GOAL_LABELS.get(program.goal, program.goal)} · "
        f"уровень: {LEVEL_LABELS.get(program.level, program.level)} · "
        f"инвентарь: {EQUIPMENT_LABELS.get(program.equipment, program.equipment)}\n"
        f"Длительность: {program.weeks} нед., {len(program.workouts)} тренировки в неделю\n"
    )
    blocks = [header]
    show_venue = program.equipment == "mix"
    for workout in program.workouts:
        blocks.append(render_workout(workout, show_venue))
    blocks.append(f"<i>{program.notes}</i>")
    return "\n\n".join(blocks)


GOAL_LABELS = {
    "lose_weight": "снижение веса",
    "build_muscle": "набор мышц",
    "keep_fit": "поддержание формы",
    "endurance": "выносливость",
}

LEVEL_LABELS = {
    "beginner": "новичок",
    "intermediate": "средний",
    "advanced": "продвинутый",
}

EQUIPMENT_LABELS = {
    "none": "без оборудования",
    "home": "гантели/резинки дома",
    "gym": "зал",
    "mix": "микс: дом и зал",
}

VENUE_LABELS = {"home": "дома", "gym": "в зале", "none": "дома"}

GENDER_LABELS = {
    "male": "мужской",
    "female": "женский",
    "other": "не указан",
}


# Default training days by weekly frequency, spaced for recovery (Monday = 0).
_DEFAULT_TRAINING_DAYS = {
    2: [0, 3],
    3: [0, 2, 4],
    4: [0, 1, 3, 4],
    5: [0, 1, 2, 3, 4],
}


def default_training_days(days_per_week: int) -> list:
    """
    Weekdays to train given a weekly frequency.

    Used to schedule the reminder created right after onboarding, so the pings
    land on days that actually match the generated split.
    """
    return list(_DEFAULT_TRAINING_DAYS.get(_clamp_days(days_per_week), [0, 2, 4]))
