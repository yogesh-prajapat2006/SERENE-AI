# habit_recommendation.py
"""Explainable, rule-based habit recommender.

Earlier versions trained a Decision Tree on a tiny dataset (16 rows, 16
different labels), so the model could only memorise it. This version uses
simple, readable rules instead. Every recommendation comes with a `reason`.

NOTE: these are general wellness suggestions, not medical advice. The rules
live in `RULES` below, so they are easy to review and change.
"""

METHOD = "rule_based"

# Ordinal scales: position in the list = level (0 = lowest).
EXERCISE_LEVELS = ["Never", "Rarely", "A few times a week", "Daily"]
SOCIAL_MEDIA_LEVELS = ["Less than 1 hour", "1-2 hours", "2-3 hours", "More than 3 hours"]
STRESS_LEVELS = ["Low", "Moderate", "High"]
MINDFULNESS_LEVELS = ["Never", "Rarely", "A few times a week", "Daily"]

OPTIONS = {
    "exercise_frequency": EXERCISE_LEVELS,
    "social_media_hours": SOCIAL_MEDIA_LEVELS,
    "stress_level": STRESS_LEVELS,
    "mindfulness_frequency": MINDFULNESS_LEVELS,
}

# Friendly alternative spellings (all lower case) -> canonical label.
ALIASES = {
    "exercise_frequency": {"weekly": "A few times a week", "few times a week": "A few times a week"},
    "mindfulness_frequency": {"weekly": "A few times a week", "few times a week": "A few times a week"},
    "social_media_hours": {
        "less than 1": "Less than 1 hour",
        "<1 hour": "Less than 1 hour",
        "1-2": "1-2 hours",
        "2-3": "2-3 hours",
        "more than 3": "More than 3 hours",
        "3+ hours": "More than 3 hours",
    },
    "stress_level": {},
}

FIELDS = tuple(OPTIONS)


def normalize_choice(field, value):
    """Return (canonical_label, level_index) for a user value, or raise ValueError."""
    options = OPTIONS[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text. Valid values: {', '.join(options)}.")
    key = " ".join(value.strip().lower().split())
    for index, option in enumerate(options):
        if option.lower() == key:
            return option, index
    alias = ALIASES.get(field, {}).get(key)
    if alias:
        return alias, options.index(alias)
    raise ValueError(f"Invalid value '{value}' for {field}. Valid values: {', '.join(options)}.")


# Each rule: (name, condition(levels) -> bool, habit, reason).
# `levels` has the keys of OPTIONS with numeric levels (higher = more).
# The FIRST matching rule wins, so more urgent rules come first.
RULES = [
    (
        "high_stress_low_mindfulness",
        lambda l: l["stress_level"] == 2 and l["mindfulness_frequency"] <= 1,
        "Practice 10 minutes of mindful breathing or meditation every day.",
        "Your stress is high and you rarely or never practise mindfulness, "
        "so a short daily breathing or meditation routine is a gentle place to start.",
    ),
    (
        "high_stress_low_exercise",
        lambda l: l["stress_level"] == 2 and l["exercise_frequency"] <= 1,
        "Take a relaxed 20-minute walk outdoors a few times this week.",
        "Your stress is high and you rarely exercise, so light movement can help you unwind.",
    ),
    (
        "heavy_social_media",
        lambda l: l["social_media_hours"] == 3,
        "Try a screen-free hour (for example before bed) every day.",
        "You spend more than 3 hours a day on social media, "
        "so a daily screen-free break is a simple way to cut down.",
    ),
    (
        "low_exercise",
        lambda l: l["exercise_frequency"] <= 1,
        "Start with a 20-minute walk or stretching session 3 times a week.",
        "You rarely or never exercise, so a small, regular activity is easier to keep up.",
    ),
    (
        "low_mindfulness",
        lambda l: l["mindfulness_frequency"] <= 1,
        "Try 5 minutes of mindfulness (breathing or a body scan) each morning.",
        "You rarely or never practise mindfulness, so a very short daily session is an easy start.",
    ),
    (
        "moderate_stress",
        lambda l: l["stress_level"] == 1,
        "Write in a journal for 10 minutes each evening.",
        "Your stress is moderate, and journaling is a simple way to process the day.",
    ),
    (
        "social_media_medium",
        lambda l: l["social_media_hours"] == 2,
        "Set a daily time limit for social media apps.",
        "You spend 2-3 hours a day on social media, so a daily limit can free up time.",
    ),
]

DEFAULT_HABIT = "Keep up your current routine and add a weekly gratitude journal."
DEFAULT_REASON = (
    "Your answers show a balanced routine, so the best step is to keep it going "
    "and add a small reflective habit."
)


def recommend_habit_with_reason(exercise_freq, social_media_hours, stress_level, mindfulness_freq):
    """Return a dict: recommended_habit, reason, method, rule, inputs (canonical values).

    Raises ValueError if any answer is not one of the valid options.
    """
    raw = {
        "exercise_frequency": exercise_freq,
        "social_media_hours": social_media_hours,
        "stress_level": stress_level,
        "mindfulness_frequency": mindfulness_freq,
    }
    labels, levels = {}, {}
    for field in FIELDS:
        labels[field], levels[field] = normalize_choice(field, raw[field])

    for name, condition, habit, reason in RULES:
        if condition(levels):
            return {
                "recommended_habit": habit,
                "reason": reason,
                "method": METHOD,
                "rule": name,
                "inputs": labels,
            }
    return {
        "recommended_habit": DEFAULT_HABIT,
        "reason": DEFAULT_REASON,
        "method": METHOD,
        "rule": "default",
        "inputs": labels,
    }


def recommend_habit(exercise_freq, social_media_hours, stress_level, mindfulness_freq):
    """Return only the habit text (same call signature as the old function)."""
    return recommend_habit_with_reason(
        exercise_freq, social_media_hours, stress_level, mindfulness_freq
    )["recommended_habit"]


def ask(prompt, field):
    options = OPTIONS[field]
    return input(f"{prompt} ({'/'.join(options)}): ")


# Simple questionnaire
def main():
    print("Please answer the following questions to get a habit recommendation.")
    answers = (
        ask("How often do you exercise?", "exercise_frequency"),
        ask("How many hours do you spend on social media?", "social_media_hours"),
        ask("How would you rate your stress level?", "stress_level"),
        ask("How often do you practice mindfulness?", "mindfulness_frequency"),
    )
    try:
        result = recommend_habit_with_reason(*answers)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"Recommended Habit: {result['recommended_habit']}")
    print(f"Why: {result['reason']}")
    print("(General wellness suggestion, not medical advice.)")


if __name__ == "__main__":
    main()
