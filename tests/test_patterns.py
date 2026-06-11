"""
Tests for pattern detection logic.
Uses synthetic DataFrames — no file I/O required.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.pattern_detector import PatternDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_udf(steps_values: list[float], sleep_values: list[float] | None = None,
              dw_values: list[float] | None = None, st_values: list[float] | None = None,
              ex_values: list[float] | None = None) -> pd.DataFrame:
    """Build a single-user preprocessed+engineered DataFrame for testing."""
    n = len(steps_values)
    dates = pd.date_range("2026-01-01", periods=n)
    sleep_values = sleep_values or [7.0] * n
    dw_values = dw_values or [4.0] * n
    st_values = st_values or [3.0] * n
    ex_values = ex_values or [45] * n

    df = pd.DataFrame({
        "user_id": ["U_TEST"] * n,
        "date": dates,
        "steps": steps_values,
        "sleep_hours": sleep_values,
        "screen_time_hours": st_values,
        "deep_work_hours": dw_values,
        "exercise_minutes": ex_values,
    })

    preprocessor = Preprocessor()
    engineer = FeatureEngineer()
    df = preprocessor.process(df)
    df = engineer.engineer(df)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPatternDetector:
    detector = PatternDetector()

    def test_increasing_steps_detected(self):
        """Clear upward steps trend should yield an increasing pattern."""
        steps = list(range(4000, 4000 + 30 * 200, 200))  # 4000 → 9800, +200/day
        udf = _make_udf(steps)
        patterns = self.detector.detect(udf, "U_TEST")
        names = [p.name for p in patterns]
        increasing = [p for p in patterns if p.metric == "steps" and p.direction == "increasing"]
        assert len(increasing) >= 1, f"Expected increasing steps pattern, got: {names}"

    def test_decreasing_steps_detected(self):
        """Clear downward steps trend should yield a decreasing pattern."""
        steps = list(range(10000, 10000 - 30 * 200, -200))  # 10000 → 4200
        udf = _make_udf(steps)
        patterns = self.detector.detect(udf, "U_TEST")
        decreasing = [p for p in patterns if p.metric == "steps" and p.direction == "decreasing"]
        assert len(decreasing) >= 1

    def test_flat_signal_no_trend(self):
        """Constant steps should NOT produce a trend pattern."""
        steps = [7000] * 30
        udf = _make_udf(steps)
        patterns = self.detector.detect(udf, "U_TEST")
        step_trends = [p for p in patterns if p.metric == "steps"]
        # Allow 0 or 1 pattern; if present, r² should be very low
        for p in step_trends:
            assert p.r_squared < 0.10, f"Flat signal should have low R², got {p.r_squared}"

    def test_burnout_pattern_detected(self):
        """Declining DW + rising screen time + declining sleep = burnout pattern."""
        n = 30
        dw = [5.0 - i * 0.1 for i in range(n)]       # declining
        st = [2.0 + i * 0.12 for i in range(n)]       # rising
        sleep = [8.0 - i * 0.05 for i in range(n)]    # declining
        steps = [7000] * n
        udf = _make_udf(steps, sleep_values=sleep, dw_values=dw, st_values=st)
        patterns = self.detector.detect(udf, "U_TEST")
        burnout = [p for p in patterns if "Burnout" in p.name]
        assert len(burnout) >= 1, "Expected burnout-like pattern"

    def test_recovery_pattern_detected(self):
        """Better sleep and exercise in second half vs first half = recovery."""
        n = 30
        sleep = [6.0] * 15 + [7.5] * 15  # improves in second half
        exercise = [20] * 15 + [60] * 15  # improves in second half
        steps = [7000] * n
        udf = _make_udf(steps, sleep_values=sleep, ex_values=exercise)
        patterns = self.detector.detect(udf, "U_TEST")
        recovery = [p for p in patterns if "Recovery" in p.name]
        assert len(recovery) >= 1, "Expected recovery pattern"

    def test_insufficient_data_returns_empty(self):
        """Less than MIN_DAYS_FOR_INSIGHT rows → empty list."""
        udf = _make_udf([7000] * 4)  # only 4 rows
        patterns = self.detector.detect(udf, "U_TEST")
        assert patterns == []

    def test_pattern_has_required_fields(self):
        """All detected patterns must have name, metric, direction, r_squared."""
        steps = list(range(4000, 10000, 200))
        udf = _make_udf(steps)
        patterns = self.detector.detect(udf, "U_TEST")
        for p in patterns:
            assert p.name, "Pattern must have a name"
            assert p.metric, "Pattern must have a metric"
            assert p.direction in ("increasing", "decreasing", "stable")
            assert 0.0 <= p.r_squared <= 1.0, f"R² out of range: {p.r_squared}"

    def test_pattern_evidence_populated(self):
        """Each pattern should carry at least one evidence item."""
        steps = list(range(4000, 10000, 200))
        udf = _make_udf(steps)
        patterns = self.detector.detect(udf, "U_TEST")
        for p in patterns:
            if p.metric != "composite":
                assert len(p.evidence) >= 1, f"Pattern '{p.name}' has no evidence"