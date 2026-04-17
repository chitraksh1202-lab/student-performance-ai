"""
Weakness detection and dynamic suggestion engine.

All features are normalized [0, 1] where 1 = best performance.
distraction_index is inverted (lower raw value = better).
"""

# Human-readable labels for each feature
FEATURE_LABELS = {
    "consistency":       "Study Consistency",
    "focus_efficiency":  "Focus Efficiency",
    "improvement":       "Improvement Trend",
    "revision_strength": "Revision Strength",
    "distraction_index": "Distraction Control",  # high raw = bad → we invert
    "subject_strength":  "Subject Strength",
}

# For distraction_index, higher value means MORE distracted (bad)
# We invert it so a high "score" = good (low distraction)
INVERTED_FEATURES = {"distraction_index"}


def score_feature(name: str, value: float) -> float:
    """
    Return a performance score in [0, 1] where 1 = best.
    Inverts distraction_index since higher distraction = worse performance.
    """
    if name in INVERTED_FEATURES:
        return 1.0 - float(value)
    return float(value)


def rank_features(features: dict) -> list[dict]:
    """
    Return all features ranked from weakest (index 0) to strongest (last).
    Each entry has: feature, label, score (0-100), raw value.
    """
    ranked = []
    for name, value in features.items():
        s = score_feature(name, value)
        ranked.append({
            "feature": name,
            "label":   FEATURE_LABELS.get(name, name),
            "score":   round(s * 100, 1),
            "raw":     round(float(value), 3),
        })
    ranked.sort(key=lambda x: x["score"])  # ascending: weakest first
    return ranked


def get_weakest(features: dict) -> dict:
    return rank_features(features)[0]


def get_strongest(features: dict) -> dict:
    return rank_features(features)[-1]


def _pct(score_0_to_1: float) -> str:
    """Format a 0-1 score as a percentage string."""
    return f"{round(score_0_to_1 * 100)}%"


def get_suggestions(features: dict) -> list[dict]:
    """
    Generate dynamic, specific suggestions based on actual feature values.

    Returns a list of dicts: {title, detail, priority}
    Priority: 'high' | 'medium' | 'low'
    """
    tips = []
    c  = features["consistency"]
    fe = features["focus_efficiency"]
    rs = features["revision_strength"]
    di = features["distraction_index"]
    ss = features["subject_strength"]
    im = features["improvement"]

    # Consistency
    if c < 0.4:
        tips.append({
            "title":    "Fix your study schedule",
            "detail":   f"Your consistency score is only {_pct(c)}. Irregular study hours hurt long-term retention. "
                        "Block fixed study slots in your calendar — even 1 hour daily beats 7 hours on Sunday.",
            "priority": "high",
        })
    elif c < 0.65:
        tips.append({
            "title":    "Make study hours more regular",
            "detail":   f"Consistency is at {_pct(c)}. Try studying at the same time each day to build a habit.",
            "priority": "medium",
        })

    # Focus efficiency
    if score_feature("focus_efficiency", fe) < 0.4:
        tips.append({
            "title":    "Boost focused study time",
            "detail":   f"Only {_pct(fe)} of your study time is actually focused. "
                        "Use Pomodoro (25 min on / 5 min break) and remove your phone from the room.",
            "priority": "high",
        })
    elif score_feature("focus_efficiency", fe) < 0.6:
        tips.append({
            "title":    "Improve study quality",
            "detail":   f"Focus efficiency is {_pct(fe)} — try studying in a quieter environment.",
            "priority": "medium",
        })

    # Revision
    if rs <= 0.4:
        tips.append({
            "title":    "Revise more frequently",
            "detail":   f"Revision strength is {_pct(rs)}. Without regular revision, you forget up to 70% "
                        "within a week (Ebbinghaus forgetting curve). Aim for 4–5 sessions/week.",
            "priority": "high",
        })
    elif rs < 0.7:
        tips.append({
            "title":    "Increase revision sessions",
            "detail":   f"You're revising moderately ({_pct(rs)}). Adding 1–2 more sessions per week "
                        "would noticeably improve retention.",
            "priority": "medium",
        })

    # Distraction
    if score_feature("distraction_index", di) < 0.4:
        tips.append({
            "title":    "Reduce distractions urgently",
            "detail":   f"Your distraction level is very high ({_pct(di)} of max). "
                        "Turn off notifications, use app blockers, and study in a dedicated space.",
            "priority": "high",
        })
    elif score_feature("distraction_index", di) < 0.6:
        tips.append({
            "title":    "Manage distractions better",
            "detail":   f"Distraction is at {_pct(di)}. Put your phone on Do Not Disturb during study sessions.",
            "priority": "medium",
        })

    # Improvement trend
    if im < 0.4:
        tips.append({
            "title":    "Your marks are declining",
            "detail":   f"Improvement trend is only {_pct(im)}. Review the last test you struggled with "
                        "and identify which topics you're weak in. A tutor or study group might help.",
            "priority": "high",
        })

    # Subject strength
    if ss < 0.3:
        tips.append({
            "title":    "Build subject foundation",
            "detail":   f"Subject strength is {_pct(ss)}. Go back to basics — re-read core chapters "
                        "and do practice problems before attempting harder questions.",
            "priority": "medium",
        })

    # All good
    if not tips:
        tips.append({
            "title":    "You're on the right track!",
            "detail":   "All areas are performing well. Focus on maintaining consistency and "
                        "push your weakest area a little higher to reach top marks.",
            "priority": "low",
        })

    # Sort by priority
    order = {"high": 0, "medium": 1, "low": 2}
    tips.sort(key=lambda x: order[x["priority"]])
    return tips


def get_grade(predicted_marks: float) -> tuple[str, str]:
    """Return (grade_letter, grade_label) based on predicted marks."""
    if predicted_marks >= 90:
        return "A+", "Outstanding"
    elif predicted_marks >= 80:
        return "A",  "Excellent"
    elif predicted_marks >= 70:
        return "B",  "Good"
    elif predicted_marks >= 60:
        return "C",  "Average"
    elif predicted_marks >= 50:
        return "D",  "Below Average"
    else:
        return "F",  "Needs Urgent Improvement"


def get_trend_label(improvement_norm: float) -> str:
    """Convert normalized improvement [0,1] to a readable label."""
    if improvement_norm >= 0.75:
        return "Strongly Increasing"
    elif improvement_norm >= 0.55:
        return "Increasing"
    elif improvement_norm >= 0.45:
        return "Stable"
    elif improvement_norm >= 0.30:
        return "Decreasing"
    else:
        return "Strongly Decreasing"
