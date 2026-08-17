"""Calorie, macro, and meal-plan calculations (Pro feature)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Profile

# Activity multipliers keyed by weekly training days.
_ACTIVITY_BY_DAYS: Tuple[Tuple[int, float], ...] = (
    (0, 1.2),
    (2, 1.375),
    (4, 1.55),
    (6, 1.725),
)

# Goal -> (calorie delta ratio, protein g/kg, fat g/kg).
_GOAL_TUNING: Dict[str, Tuple[float, float, float]] = {
    "lose_weight": (-0.18, 2.0, 0.8),
    "build_muscle": (0.12, 1.8, 1.0),
    "keep_fit": (0.0, 1.6, 0.9),
    "endurance": (0.05, 1.5, 0.9),
}


@dataclass
class NutritionPlan:
    """Daily energy and macro targets plus a sample day of eating."""

    bmr: int
    tdee: int
    calories: int
    protein_g: int
    fat_g: int
    carbs_g: int
    water_ml: int
    meals: List[Tuple[str, str]]
    warning: str = ""


def mifflin_st_jeor(profile: Profile) -> Optional[float]:
    """
    Basal metabolic rate via the Mifflin-St Jeor equation.

    Returns None when the profile lacks age, height, or weight.
    """
    if profile.age is None or profile.height_cm is None or profile.weight_kg is None:
        return None
    base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    if profile.gender == "male":
        return base + 5
    if profile.gender == "female":
        return base - 161
    # Unspecified gender: average of the two constants keeps the estimate honest.
    return base - 78


def activity_multiplier(days_per_week: int) -> float:
    multiplier = _ACTIVITY_BY_DAYS[0][1]
    for threshold, value in _ACTIVITY_BY_DAYS:
        if days_per_week >= threshold:
            multiplier = value
    return multiplier


def build_nutrition_plan(profile: Profile) -> Optional[NutritionPlan]:
    """
    Compute daily targets and a sample menu for `profile`.

    Returns None when the profile is incomplete — the caller should ask the
    user to finish onboarding instead of showing made-up numbers.
    """
    bmr = mifflin_st_jeor(profile)
    if bmr is None or profile.weight_kg is None:
        return None

    tdee = bmr * activity_multiplier(profile.days_per_week)
    delta, protein_per_kg, fat_per_kg = _GOAL_TUNING.get(
        profile.goal, _GOAL_TUNING["keep_fit"]
    )
    calories = tdee * (1 + delta)

    warning = ""
    # Never prescribe a starvation diet: clamp to a conservative floor.
    floor = 1500 if profile.gender == "male" else 1200
    if calories < floor:
        calories = float(floor)
        warning = (
            "Расчётная норма получилась слишком низкой, поэтому она поднята до "
            f"безопасного минимума ({floor} ккал). При таком дефиците обязательно "
            "проконсультируйся с врачом."
        )

    protein_g = protein_per_kg * profile.weight_kg
    fat_g = fat_per_kg * profile.weight_kg
    carbs_kcal = calories - (protein_g * 4 + fat_g * 9)
    if carbs_kcal < 0:
        # Rebalance by trimming fat first — protein is the priority on a deficit.
        fat_g = max(0.5 * profile.weight_kg, (calories - protein_g * 4) / 9)
        carbs_kcal = max(0.0, calories - (protein_g * 4 + fat_g * 9))
    carbs_g = carbs_kcal / 4

    return NutritionPlan(
        bmr=int(round(bmr)),
        tdee=int(round(tdee)),
        calories=int(round(calories)),
        protein_g=int(round(protein_g)),
        fat_g=int(round(fat_g)),
        carbs_g=int(round(carbs_g)),
        water_ml=int(round(profile.weight_kg * 30 / 50.0) * 50),
        meals=_sample_menu(profile.goal),
        warning=warning,
    )


def _sample_menu(goal: str) -> List[Tuple[str, str]]:
    if goal == "build_muscle":
        return [
            ("Завтрак", "Овсянка на молоке, 3 яйца, банан, орехи"),
            ("Перекус", "Творог 5% с мёдом и ягодами"),
            ("Обед", "Рис, куриная грудка, овощной салат с оливковым маслом"),
            ("Перекус", "Протеиновый коктейль + тост с арахисовой пастой"),
            ("Ужин", "Говядина/рыба, гречка, тушёные овощи"),
        ]
    if goal == "lose_weight":
        return [
            ("Завтрак", "Омлет из 2 яиц + белки, овощи, цельнозерновой хлеб"),
            ("Перекус", "Греческий йогурт без сахара, яблоко"),
            ("Обед", "Куриная грудка/индейка, киноа, большая порция салата"),
            ("Перекус", "Творог 2% или горсть орехов (30 г)"),
            ("Ужин", "Белая рыба, овощи на пару, немного оливкового масла"),
        ]
    return [
        ("Завтрак", "Каша, яйца, фрукт"),
        ("Перекус", "Йогурт и орехи"),
        ("Обед", "Крупа, мясо или рыба, овощи"),
        ("Перекус", "Фрукт или творог"),
        ("Ужин", "Белок и овощи, немного медленных углеводов"),
    ]


def render_nutrition_plan(plan: NutritionPlan) -> str:
    """Format a nutrition plan for a Telegram message."""
    lines = [
        "<b>Питание — твои дневные ориентиры</b>",
        "",
        f"Базовый обмен (BMR): <b>{plan.bmr}</b> ккал",
        f"С учётом активности (TDEE): <b>{plan.tdee}</b> ккал",
        f"Цель по калориям: <b>{plan.calories}</b> ккал/день",
        "",
        f"Белки: <b>{plan.protein_g} г</b>",
        f"Жиры: <b>{plan.fat_g} г</b>",
        f"Углеводы: <b>{plan.carbs_g} г</b>",
        f"Вода: <b>{plan.water_ml} мл</b>",
        "",
        "<b>Пример дня</b>",
    ]
    for title, content in plan.meals:
        lines.append(f"• {title}: {content}")
    if plan.warning:
        lines.extend(["", f"⚠️ {plan.warning}"])
    lines.extend(
        [
            "",
            "<i>Это расчёт по формуле, а не медицинская рекомендация. "
            "При хронических заболеваниях, беременности или приёме лекарств "
            "сначала проконсультируйся с врачом.</i>",
        ]
    )
    return "\n".join(lines)
