"""
data_loader.py
Load and validate the behavioral dataset.
All IO lives here — rest of pipeline consumes clean DataFrames only.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from src.utils import COLUMNS_REQUIRED, get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load, validate, and parse behavioral CSV data."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._raw: Optional[pd.DataFrame] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load CSV → validate schema → parse dates → sort.
        Returns a clean DataFrame ready for preprocessing.
        Raises FileNotFoundError or ValueError on bad input.
        """
        logger.info(f"Loading dataset from: {self.csv_path}")
        self._check_file_exists()
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
        self._validate_schema(df)
        df = self._parse_dates(df)
        df = self._sort(df)
        df = self._coerce_types(df)
        self._raw = df
        logger.info(
            f"Users: {sorted(df['user_id'].unique())} | "
            f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
        )
        return df

    @property
    def users(self) -> list[str]:
        if self._raw is None:
            raise RuntimeError("Call load() first.")
        return sorted(self._raw["user_id"].unique().tolist())

    def get_user_df(self, user_id: str) -> pd.DataFrame:
        """Return subset for a single user, reset index."""
        if self._raw is None:
            raise RuntimeError("Call load() first.")
        return (
            self._raw[self._raw["user_id"] == user_id]
            .copy()
            .reset_index(drop=True)
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _check_file_exists(self) -> None:
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in COLUMNS_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        logger.info("Schema validation passed.")

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad = df["date"].isna().sum()
        if bad:
            logger.warning(f"{bad} rows had unparseable dates → dropped.")
            df = df.dropna(subset=["date"])
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["user_id", "date"]).reset_index(drop=True)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns are correct dtype; fill tiny gaps forward."""
        numeric_cols = ["steps", "sleep_hours", "screen_time_hours",
                        "deep_work_hours", "exercise_minutes"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            null_count = df[col].isna().sum()
            if null_count:
                logger.warning(f"Column '{col}' has {null_count} nulls → forward-filled.")
                df[col] = df.groupby("user_id")[col].transform(
                    lambda s: s.fillna(method="ffill").fillna(method="bfill")
                )
        return df