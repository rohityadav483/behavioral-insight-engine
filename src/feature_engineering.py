"""
feature_engineering.py
Compute derived behavioral features from preprocessed data.
All features are deterministic and interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ROLLING_WINDOW, get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Compute 8 engineered behavioral features per user per day.

    Features:
      - sleep_consistency         low std dev → high consistency
      - activity_consistency      low std dev → high consistency
      - productivity_score        weighted deep work + steps
      - behavioral_volatility     mean |z-score| across signals
      - exercise_adherence        % of rolling window days with ≥30 min exercise
      - screen_time_trend         linear slope of screen_time over 7d
      - recovery_score            sleep quality × exercise regularity composite
      - behavioral_stability_score inverse of cross-signal variance
    """

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self.window = window

    # ── Public ────────────────────────────────────────────────────────────────

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to a preprocessed full-dataset DataFrame.
        Returns enriched copy.
        """
        df = df.copy()
        results = []
        for user_id, udf in df.groupby("user_id"):
            udf = udf.sort_values("date").reset_index(drop=True)
            udf = self._sleep_consistency(udf)
            udf = self._activity_consistency(udf)
            udf = self._productivity_score(udf)
            udf = self._behavioral_volatility(udf)
            udf = self._exercise_adherence(udf)
            udf = self._screen_time_trend(udf)
            udf = self._recovery_score(udf)
            udf = self._behavioral_stability(udf)
            results.append(udf)
            logger.info(f"Features engineered for user {user_id}.")
        return pd.concat(results, ignore_index=True)

    def get_feature_summary(self, udf: pd.DataFrame) -> dict[str, float]:
        """Return mean of each engineered feature for a single-user DataFrame."""
        feats = [
            "sleep_consistency", "activity_consistency", "productivity_score",
            "behavioral_volatility", "exercise_adherence", "screen_time_trend",
            "recovery_score", "behavioral_stability_score",
        ]
        return {f: round(float(udf[f].mean(skipna=True)), 3) for f in feats if f in udf.columns}

    # ── Feature Implementations ───────────────────────────────────────────────

    def _sleep_consistency(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Inverted rolling std of sleep_hours.
        High value = very consistent sleep schedule.
        """
        roll_std = udf["sleep_hours"].rolling(self.window, min_periods=3).std()
        # normalise: invert and scale to 0-1
        udf["sleep_consistency"] = 1 / (1 + roll_std)
        return udf

    def _activity_consistency(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Inverted rolling std of steps."""
        roll_std = udf["steps"].rolling(self.window, min_periods=3).std()
        udf["activity_consistency"] = 1 / (1 + roll_std / 1000)  # scale to similar range
        return udf

    def _productivity_score(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Composite: 60% deep_work_hours (normalized 0-6h) + 40% steps (normalized 0-12k).
        Range: 0-1.
        """
        dw_norm = (udf["deep_work_hours"] / 6.0).clip(0, 1)
        steps_norm = (udf["steps"] / 12000.0).clip(0, 1)
        udf["productivity_score"] = 0.60 * dw_norm + 0.40 * steps_norm
        return udf

    def _behavioral_volatility(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Mean of absolute z-scores across all signals.
        High value = large deviations from personal baseline.
        """
        z_cols = [c for c in udf.columns if c.endswith("_zscore")]
        if z_cols:
            udf["behavioral_volatility"] = udf[z_cols].abs().mean(axis=1)
        else:
            udf["behavioral_volatility"] = np.nan
        return udf

    def _exercise_adherence(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Fraction of rolling-window days where exercise_minutes >= 30.
        Range: 0-1.
        """
        meets_target = (udf["exercise_minutes"] >= 30).astype(float)
        udf["exercise_adherence"] = (
            meets_target.rolling(self.window, min_periods=3).mean()
        )
        return udf

    def _screen_time_trend(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Linear slope of screen_time_hours over rolling window.
        Positive = increasing screen time. Uses OLS per window.
        """
        n = len(udf)
        slopes = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            chunk = udf["screen_time_hours"].iloc[i - self.window + 1: i + 1].values
            if len(chunk) >= 3:
                x = np.arange(len(chunk), dtype=float)
                p = np.polyfit(x, chunk, 1)
                slopes[i] = p[0]  # slope
        udf["screen_time_trend"] = slopes
        return udf

    def _recovery_score(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Composite of sleep quality and exercise regularity.
        sleep quality = sleep_hours / 8 (capped at 1)
        exercise regularity = exercise_adherence
        """
        sleep_q = (udf["sleep_hours"] / 8.0).clip(0, 1)
        ex_reg = udf.get("exercise_adherence", pd.Series(0.5, index=udf.index))
        udf["recovery_score"] = 0.55 * sleep_q + 0.45 * ex_reg
        return udf

    def _behavioral_stability(self, udf: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse of mean rolling coefficient of variation across key signals.
        High = stable, predictable behavior.
        """
        signals = ["steps", "sleep_hours", "deep_work_hours", "exercise_minutes"]
        cvs = []
        for sig in signals:
            roll_mean = udf[sig].rolling(self.window, min_periods=3).mean()
            roll_std = udf[sig].rolling(self.window, min_periods=3).std()
            cv = roll_std / roll_mean.replace(0, np.nan)
            cvs.append(cv)
        mean_cv = pd.concat(cvs, axis=1).mean(axis=1)
        udf["behavioral_stability_score"] = 1 / (1 + mean_cv)
        return udf