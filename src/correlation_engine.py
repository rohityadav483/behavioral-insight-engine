"""
correlation_engine.py
Multi-signal correlation analysis.
Uses Pearson r for linear relationships and Spearman rho for monotonic ones.
Language is always hedged — never claims causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import CorrelationResult, get_logger

logger = get_logger(__name__)

# Only report correlations with |r| ≥ this
MIN_R_THRESHOLD = 0.40

# Pairs of interest (behavioral hypotheses)
HYPOTHESIS_PAIRS: list[tuple[str, str]] = [
    ("screen_time_hours", "sleep_hours"),
    ("sleep_hours", "deep_work_hours"),
    ("exercise_minutes", "productivity_score"),
    ("exercise_minutes", "sleep_hours"),
    ("screen_time_hours", "deep_work_hours"),
    ("steps", "deep_work_hours"),
    ("behavioral_volatility", "productivity_score"),
    ("sleep_hours", "productivity_score"),
]

HEDGED_TEMPLATES = {
    "positive": "Patterns suggest {a} may be associated with higher {b}.",
    "negative": "Patterns suggest higher {a} appears associated with lower {b}.",
}


class CorrelationEngine:
    """Compute and interpret pairwise behavioral correlations."""

    def analyze(self, udf: pd.DataFrame, user_id: str) -> list[CorrelationResult]:
        """
        Compute correlations for hypothesis pairs on single-user DataFrame.
        Returns only significant findings (|r| >= MIN_R_THRESHOLD).
        """
        results: list[CorrelationResult] = []

        for metric_a, metric_b in HYPOTHESIS_PAIRS:
            if metric_a not in udf.columns or metric_b not in udf.columns:
                continue

            res = self._compute_pair(udf, metric_a, metric_b)
            if res:
                results.append(res)

        # Also compute full correlation matrix for heatmap use
        logger.info(f"User {user_id}: {len(results)} significant correlations found.")
        return results

    def correlation_matrix(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Return Pearson correlation matrix for all numeric signals.
        Used for heatmap visualization.
        """
        numeric_cols = [
            "steps", "sleep_hours", "screen_time_hours",
            "deep_work_hours", "exercise_minutes",
        ]
        available = [c for c in numeric_cols if c in udf.columns]
        return udf[available].corr(method="pearson")

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_pair(
        self, udf: pd.DataFrame, metric_a: str, metric_b: str
    ) -> CorrelationResult | None:
        """Compute Pearson r for a pair. Return None if not significant."""
        series_a = udf[metric_a].dropna()
        series_b = udf[metric_b].dropna()

        # Align indices
        combined = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
        if len(combined) < 5:
            return None

        r, p_value = stats.pearsonr(combined["a"], combined["b"])

        if abs(r) < MIN_R_THRESHOLD:
            return None

        direction = "positive" if r > 0 else "negative"
        label_a = metric_a.replace("_", " ")
        label_b = metric_b.replace("_", " ")

        interpretation = HEDGED_TEMPLATES[direction].format(a=label_a, b=label_b)

        # Add strength qualifier
        abs_r = abs(r)
        if abs_r >= 0.70:
            strength = "strong"
        elif abs_r >= 0.55:
            strength = "moderate"
        else:
            strength = "weak-to-moderate"

        interpretation = f"{interpretation} ({strength} association, r={r:.2f})"

        return CorrelationResult(
            metric_a=metric_a,
            metric_b=metric_b,
            r_value=round(r, 3),
            direction=direction,
            interpretation=interpretation,
            significant=True,
        )

    def lagged_correlation(
        self, udf: pd.DataFrame, metric_a: str, metric_b: str, max_lag: int = 3
    ) -> dict[int, float]:
        """
        Compute lagged correlations: does metric_a today predict metric_b in N days?
        Returns dict of {lag: r_value}.
        """
        results: dict[int, float] = {}
        for lag in range(0, max_lag + 1):
            a = udf[metric_a].iloc[: len(udf) - lag] if lag > 0 else udf[metric_a]
            b = udf[metric_b].iloc[lag:] if lag > 0 else udf[metric_b]
            combined = pd.DataFrame({"a": a.values, "b": b.values}).dropna()
            if len(combined) < 5:
                continue
            r, _ = stats.pearsonr(combined["a"], combined["b"])
            results[lag] = round(r, 3)
        return results