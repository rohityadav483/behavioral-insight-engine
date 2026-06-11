"""
preprocessing.py
Per-user rolling statistics, baselines, and normalization.
Operates on clean DataFrames from DataLoader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ROLLING_WINDOW, get_logger

logger = get_logger(__name__)

SIGNAL_COLS = [
    "steps", "sleep_hours", "screen_time_hours",
    "deep_work_hours", "exercise_minutes",
]


class Preprocessor:
    """Add rolling stats and baseline columns to per-user DataFrames."""

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self.window = window

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process full multi-user DataFrame.
        Adds rolling mean/std, z-scores, and baselines per user.
        Returns enriched DataFrame (does not mutate input).
        """
        df = df.copy()
        results = []
        for user_id, udf in df.groupby("user_id"):
            udf = udf.sort_values("date").reset_index(drop=True)
            udf = self._add_rolling_stats(udf)
            udf = self._add_baselines(udf)
            udf = self._add_zscores(udf)
            udf = self._add_week_number(udf)
            results.append(udf)
            logger.info(f"Preprocessed user {user_id}: {len(udf)} rows.")
        return pd.concat(results, ignore_index=True)

    def get_user_baseline(self, df: pd.DataFrame, user_id: str) -> dict[str, float]:
        """Return whole-period mean baselines for a single user."""
        udf = df[df["user_id"] == user_id]
        return {col: float(udf[col].mean()) for col in SIGNAL_COLS}

    # ── Private ───────────────────────────────────────────────────────────────

    def _add_rolling_stats(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Add <col>_roll_mean and <col>_roll_std for each signal column."""
        for col in SIGNAL_COLS:
            udf[f"{col}_roll_mean"] = (
                udf[col].rolling(self.window, min_periods=3).mean()
            )
            udf[f"{col}_roll_std"] = (
                udf[col].rolling(self.window, min_periods=3).std()
            )
        return udf

    def _add_baselines(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Add <col>_baseline = expanding mean (all prior days for that user)."""
        for col in SIGNAL_COLS:
            udf[f"{col}_baseline"] = udf[col].expanding(min_periods=3).mean()
        return udf

    def _add_zscores(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Add <col>_zscore relative to rolling mean/std."""
        for col in SIGNAL_COLS:
            mean_col = f"{col}_roll_mean"
            std_col = f"{col}_roll_std"
            # avoid division by zero
            std_safe = udf[std_col].replace(0, np.nan)
            udf[f"{col}_zscore"] = (udf[col] - udf[mean_col]) / std_safe
        return udf

    def _add_week_number(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Tag each row with week number (1-indexed from user's first date)."""
        first_date = udf["date"].min()
        udf["week"] = ((udf["date"] - first_date).dt.days // 7 + 1).astype(int)
        return udf

    # ── Normalization (0-1 min-max, per-user) ─────────────────────────────────

    def normalize(self, udf: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize signal cols within a single user's DataFrame."""
        udf = udf.copy()
        for col in SIGNAL_COLS:
            col_min = udf[col].min()
            col_max = udf[col].max()
            denom = col_max - col_min if col_max != col_min else 1.0
            udf[f"{col}_norm"] = (udf[col] - col_min) / denom
        return udf