"""
persona_engine.py
Behavioral persona assignment.
Step 1: deterministic rule-based assignment from feature scores.
Step 2: Groq polishes the wording (optional, only if Groq available).
"""

from __future__ import annotations

import pandas as pd

from src.utils import Persona, get_logger

logger = get_logger(__name__)

# ── Persona Definitions ───────────────────────────────────────────────────────

PERSONA_DEFINITIONS = {
    "Consistent Performer": {
        "description": (
            "Maintains stable, high-quality behavioral patterns across sleep, "
            "activity, and deep work. Low volatility and high reliability."
        ),
        "rules": {
            "behavioral_stability_score": (">=", 0.70),
            "productivity_score": (">=", 0.55),
            "behavioral_volatility": ("<=", 1.0),
        },
    },
    "Burnout Risk": {
        "description": (
            "Shows signs consistent with behavioral burnout: declining productivity, "
            "elevated screen time, and disrupted sleep patterns."
        ),
        "rules": {
            "productivity_score": ("<=", 0.45),
            "screen_time_trend": (">=", 0.05),
            "sleep_consistency": ("<=", 0.65),
        },
    },
    "Recovery Phase": {
        "description": (
            "Recently improving behavioral signals following a lower-performance period. "
            "Sleep and exercise metrics are trending upward."
        ),
        "rules": {
            "recovery_score": (">=", 0.60),
            "exercise_adherence": (">=", 0.50),
        },
    },
    "Highly Structured User": {
        "description": (
            "Exceptionally consistent routines with minimal variance across all signals. "
            "High exercise adherence and stable sleep schedule."
        ),
        "rules": {
            "sleep_consistency": (">=", 0.80),
            "activity_consistency": (">=", 0.75),
            "exercise_adherence": (">=", 0.70),
        },
    },
    "Digitally Fatigued": {
        "description": (
            "Screen time trending upward alongside declining sleep and productivity. "
            "Pattern suggests digital overexposure affecting recovery and focus."
        ),
        "rules": {
            "screen_time_trend": (">=", 0.08),
            "recovery_score": ("<=", 0.50),
            "productivity_score": ("<=", 0.50),
        },
    },
}

FALLBACK_PERSONA = "Developing Pattern"
FALLBACK_DESCRIPTION = (
    "Behavioral profile is still developing or does not clearly match "
    "a defined pattern. More data may reveal clearer trends."
)


class PersonaEngine:
    """Assign behavioral persona using deterministic feature rules."""

    def assign(self, udf: pd.DataFrame, user_id: str) -> Persona:
        """
        Assign persona to single-user enriched DataFrame.
        Evaluates rules in priority order; returns first match.
        Falls back to 'Developing Pattern' if no match.
        """
        feature_means = self._compute_means(udf)
        scores: list[tuple[str, int]] = []   # (persona_name, rules_matched)

        for persona_name, defn in PERSONA_DEFINITIONS.items():
            match_count = self._evaluate_rules(defn["rules"], feature_means)
            total_rules = len(defn["rules"])
            if match_count == total_rules:
                # Full match — use immediately
                logger.info(f"User {user_id}: persona = '{persona_name}' (full match).")
                return Persona(
                    label=persona_name,
                    description=defn["description"],
                    supporting_metrics=feature_means,
                )
            scores.append((persona_name, match_count))

        # Partial match: pick persona with most rules satisfied
        best_name, best_count = max(scores, key=lambda t: t[1])
        total_rules = len(PERSONA_DEFINITIONS[best_name]["rules"])

        if best_count >= max(1, total_rules - 1):
            logger.info(f"User {user_id}: persona = '{best_name}' (partial match {best_count}/{total_rules}).")
            return Persona(
                label=best_name,
                description=PERSONA_DEFINITIONS[best_name]["description"],
                supporting_metrics=feature_means,
            )

        logger.info(f"User {user_id}: no persona matched → fallback.")
        return Persona(
            label=FALLBACK_PERSONA,
            description=FALLBACK_DESCRIPTION,
            supporting_metrics=feature_means,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_means(udf: pd.DataFrame) -> dict[str, float]:
        """Compute mean of each engineered feature, skip NaN columns."""
        feat_cols = [
            "sleep_consistency", "activity_consistency", "productivity_score",
            "behavioral_volatility", "exercise_adherence", "screen_time_trend",
            "recovery_score", "behavioral_stability_score",
        ]
        return {
            col: round(float(udf[col].mean(skipna=True)), 4)
            for col in feat_cols
            if col in udf.columns
        }

    @staticmethod
    def _evaluate_rules(
        rules: dict[str, tuple[str, float]],
        feature_means: dict[str, float],
    ) -> int:
        """Return count of rules satisfied."""
        count = 0
        for feat, (op, threshold) in rules.items():
            val = feature_means.get(feat)
            if val is None:
                continue
            if op == ">=" and val >= threshold:
                count += 1
            elif op == "<=" and val <= threshold:
                count += 1
        return count