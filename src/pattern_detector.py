"""
pattern_detector.py
Detect behavioral trends and patterns per user.
Uses linear regression slopes, rolling averages, variance analysis,
and week-over-week comparisons. Fully deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import (
    MIN_DAYS_FOR_INSIGHT,
    ROLLING_WINDOW,
    Evidence,
    Pattern,
    get_logger,
)

logger = get_logger(__name__)

# Minimum |slope| to declare a trend (avoid noise)
SLOPE_THRESHOLD = 0.01


class PatternDetector:
    """Detect meaningful behavioral trends for a single user."""

    SIGNAL_DISPLAY = {
        "steps": "daily steps",
        "sleep_hours": "sleep duration",
        "screen_time_hours": "screen time",
        "deep_work_hours": "deep work",
        "exercise_minutes": "exercise",
        "productivity_score": "productivity score",
        "recovery_score": "recovery score",
        "behavioral_stability_score": "behavioral stability",
    }

    def detect(self, udf: pd.DataFrame, user_id: str) -> list[Pattern]:
        """
        Run all pattern detection on a single-user enriched DataFrame.
        Returns list of Pattern objects.
        """
        if len(udf) < MIN_DAYS_FOR_INSIGHT:
            logger.warning(f"User {user_id}: insufficient data for pattern detection.")
            return []

        patterns: list[Pattern] = []
        for metric in self.SIGNAL_DISPLAY:
            if metric not in udf.columns:
                continue
            p = self._analyze_metric(udf, metric, user_id)
            if p:
                patterns.append(p)

        patterns += self._detect_burnout_pattern(udf, user_id)
        patterns += self._detect_recovery_pattern(udf, user_id)
        patterns += self._detect_stabilization(udf, user_id)

        logger.info(f"User {user_id}: {len(patterns)} patterns detected.")
        return patterns

    # ── Per-Metric Trend Analysis ─────────────────────────────────────────────

    def _analyze_metric(self, udf: pd.DataFrame, metric: str,
                        user_id: str) -> Pattern | None:
        """Fit OLS slope over full user period; return Pattern if meaningful."""
        series = udf[metric].dropna()
        if len(series) < MIN_DAYS_FOR_INSIGHT:
            return None

        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Weak trend → skip
        if abs(slope) < SLOPE_THRESHOLD and r_squared < 0.10:
            return None

        direction = "increasing" if slope > 0 else "decreasing"

        # Week-over-week delta
        week_delta = self._week_over_week(udf, metric)

        display_name = self.SIGNAL_DISPLAY.get(metric, metric)
        first_val = float(series.iloc[:7].mean())
        last_val = float(series.iloc[-7:].mean())
        evidence = [
            Evidence(
                description=f"7-day average changed from {first_val:.2f} to {last_val:.2f}",
                metric=metric,
                value=last_val,
                baseline=first_val,
                delta=last_val - first_val,
            )
        ]

        return Pattern(
            name=f"{direction.capitalize()} {display_name}",
            description=(
                f"{display_name.capitalize()} shows {direction} trend "
                f"(slope={slope:.4f}, R²={r_squared:.2f})."
            ),
            metric=metric,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            week_delta=week_delta,
            evidence=evidence,
        )

    def _week_over_week(self, udf: pd.DataFrame, metric: str) -> float:
        """Return % change: last week mean vs previous week mean."""
        if "week" not in udf.columns:
            return 0.0
        weeks = sorted(udf["week"].unique())
        if len(weeks) < 2:
            return 0.0
        last_w = weeks[-1]
        prev_w = weeks[-2]
        last_mean = udf[udf["week"] == last_w][metric].mean()
        prev_mean = udf[udf["week"] == prev_w][metric].mean()
        if prev_mean == 0 or np.isnan(prev_mean):
            return 0.0
        return float((last_mean - prev_mean) / abs(prev_mean) * 100)

    # ── Composite Pattern Detectors ───────────────────────────────────────────

    def _detect_burnout_pattern(self, udf: pd.DataFrame, user_id: str) -> list[Pattern]:
        """
        Burnout indicator: declining deep_work + increasing screen_time + declining sleep.
        All three conditions must trend in burnout direction over last 14 days.
        """
        recent = udf.tail(14).copy()
        if len(recent) < 7:
            return []

        dw_slope = self._quick_slope(recent["deep_work_hours"].dropna())
        st_slope = self._quick_slope(recent["screen_time_hours"].dropna())
        sl_slope = self._quick_slope(recent["sleep_hours"].dropna())

        if dw_slope < -0.02 and st_slope > 0.02 and sl_slope < -0.01:
            ev = Evidence(
                description=(
                    f"Deep work slope={dw_slope:.3f}, "
                    f"screen time slope={st_slope:.3f}, "
                    f"sleep slope={sl_slope:.3f}"
                ),
                metric="composite",
                value=dw_slope,
                baseline=0.0,
                delta=dw_slope,
            )
            return [Pattern(
                name="Burnout-Like Pattern",
                description=(
                    "Declining deep work, rising screen time, and declining sleep "
                    "observed over recent 14 days. Pattern is consistent with early "
                    "burnout indicators."
                ),
                metric="composite",
                direction="decreasing",
                slope=dw_slope,
                r_squared=0.0,
                week_delta=0.0,
                evidence=[ev],
            )]
        return []

    def _detect_recovery_pattern(self, udf: pd.DataFrame, user_id: str) -> list[Pattern]:
        """
        Recovery: prior low period followed by improving sleep + exercise.
        Check if last 7 days are higher than prior 7 days on sleep + exercise.
        """
        if len(udf) < 14:
            return []
        recent = udf.tail(7)
        prior = udf.tail(14).head(7)

        sl_recent = recent["sleep_hours"].mean()
        sl_prior = prior["sleep_hours"].mean()
        ex_recent = recent["exercise_minutes"].mean()
        ex_prior = prior["exercise_minutes"].mean()

        if sl_recent > sl_prior * 1.05 and ex_recent > ex_prior * 1.05:
            ev = Evidence(
                description=(
                    f"Sleep: {sl_prior:.1f}h → {sl_recent:.1f}h; "
                    f"Exercise: {ex_prior:.0f} → {ex_recent:.0f} min"
                ),
                metric="composite",
                value=sl_recent,
                baseline=sl_prior,
                delta=sl_recent - sl_prior,
            )
            return [Pattern(
                name="Recovery Phase",
                description=(
                    "Sleep and exercise have improved compared to the prior week, "
                    "suggesting a recovery or positive behavioral adjustment phase."
                ),
                metric="composite",
                direction="increasing",
                slope=sl_recent - sl_prior,
                r_squared=0.0,
                week_delta=float((sl_recent - sl_prior) / sl_prior * 100),
                evidence=[ev],
            )]
        return []

    def _detect_stabilization(self, udf: pd.DataFrame, user_id: str) -> list[Pattern]:
        """
        Stabilization: behavioral_stability_score high and not declining.
        """
        if "behavioral_stability_score" not in udf.columns:
            return []
        recent = udf.tail(7)["behavioral_stability_score"].dropna()
        if len(recent) < 4:
            return []
        mean_stab = recent.mean()
        slope = self._quick_slope(recent)
        if mean_stab >= 0.70 and slope >= -0.005:
            ev = Evidence(
                description=f"Mean stability score over last 7 days: {mean_stab:.2f}",
                metric="behavioral_stability_score",
                value=mean_stab,
                baseline=0.70,
                delta=mean_stab - 0.70,
            )
            return [Pattern(
                name="Behavioral Stabilization",
                description=(
                    "Behavioral signals show low cross-metric variance, "
                    "suggesting a stable and consistent routine."
                ),
                metric="behavioral_stability_score",
                direction="stable",
                slope=slope,
                r_squared=0.0,
                week_delta=0.0,
                evidence=[ev],
            )]
        return []

    # ── Util ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _quick_slope(series: pd.Series) -> float:
        """OLS slope of a short series. Returns 0 if insufficient data."""
        s = series.dropna()
        if len(s) < 3:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = s.values.astype(float)
        slope, *_ = stats.linregress(x, y)
        return float(slope)