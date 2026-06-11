"""
Tests for anomaly detection logic.
Verifies z-score detection, severity, evidence, and IForest corroboration.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detector import AnomalyDetector


# ── Fixture helper ────────────────────────────────────────────────────────────

def _make_udf(steps_values: list[float],
              sleep_values: list[float] | None = None) -> pd.DataFrame:
    n = len(steps_values)
    dates = pd.date_range("2026-01-01", periods=n)
    sleep_values = sleep_values or [7.0] * n
    df = pd.DataFrame({
        "user_id": ["U_TEST"] * n,
        "date": dates,
        "steps": steps_values,
        "sleep_hours": sleep_values,
        "screen_time_hours": [3.0] * n,
        "deep_work_hours": [4.0] * n,
        "exercise_minutes": [45] * n,
    })
    preprocessor = Preprocessor()
    engineer = FeatureEngineer()
    df = preprocessor.process(df)
    df = engineer.engineer(df)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:
    detector = AnomalyDetector(use_iforest=True)
    detector_no_iforest = AnomalyDetector(use_iforest=False)

    def test_sudden_drop_detected(self):
        """Single-day crash in steps should be flagged."""
        steps = [8000] * 25 + [1000] + [8000] * 4  # huge drop on day 26
        udf = _make_udf(steps)
        anomalies = self.detector.detect(udf, "U_TEST")
        step_anomalies = [a for a in anomalies if a.metric == "steps"]
        assert len(step_anomalies) >= 1, "Sudden drop in steps should be anomaly"
        # The anomaly should be the low day
        low_day = min(step_anomalies, key=lambda a: a.value)
        assert low_day.value < 5000

    def test_sudden_spike_detected(self):
        """Single-day spike in screen_time should be flagged."""
        screen = [3.0] * 25 + [12.0] + [3.0] * 4  # spike on day 26
        n = 30
        df = pd.DataFrame({
            "user_id": ["U_TEST"] * n,
            "date": pd.date_range("2026-01-01", periods=n),
            "steps": [7000] * n,
            "sleep_hours": [7.0] * n,
            "screen_time_hours": screen,
            "deep_work_hours": [4.0] * n,
            "exercise_minutes": [45] * n,
        })
        pre = Preprocessor()
        eng = FeatureEngineer()
        udf = pre.process(df)
        udf = eng.engineer(udf)
        anomalies = self.detector.detect(udf, "U_TEST")
        screen_anomalies = [a for a in anomalies if a.metric == "screen_time_hours"]
        assert len(screen_anomalies) >= 1, "Screen time spike should be detected"

    def test_stable_signal_no_anomaly(self):
        """Perfectly stable steps should yield no anomaly."""
        steps = [7000] * 30
        udf = _make_udf(steps)
        anomalies = self.detector_no_iforest.detect(udf, "U_TEST")
        step_anomalies = [a for a in anomalies if a.metric == "steps"]
        assert len(step_anomalies) == 0, "Stable steps should have no anomaly"

    def test_anomaly_has_required_fields(self):
        """Every anomaly must carry required fields."""
        steps = [8000] * 25 + [1000] + [8000] * 4
        udf = _make_udf(steps)
        anomalies = self.detector.detect(udf, "U_TEST")
        for a in anomalies:
            assert a.date, "Anomaly must have a date"
            assert a.metric, "Anomaly must have a metric"
            assert a.reason, "Anomaly must have a reason"
            assert a.evidence, "Anomaly must have evidence"
            assert a.severity in ("low", "medium", "high")
            assert isinstance(a.z_score, float)

    def test_severity_scales_with_zscore(self):
        """Higher z-score deviation → higher severity."""
        steps = [8000] * 25 + [500] + [8000] * 4  # extreme drop
        udf = _make_udf(steps)
        anomalies = self.detector.detect(udf, "U_TEST")
        step_anomalies = [a for a in anomalies if a.metric == "steps"]
        if step_anomalies:
            worst = min(step_anomalies, key=lambda a: a.z_score)
            assert worst.severity in ("medium", "high"), \
                f"Extreme drop should be medium/high severity, got {worst.severity}"

    def test_insufficient_data_returns_empty(self):
        """Less than 5 rows → empty anomaly list."""
        udf = _make_udf([7000] * 4)
        anomalies = self.detector.detect(udf, "U_TEST")
        assert anomalies == []

    def test_iforest_corroboration_flag(self):
        """IForest corroboration field must be boolean."""
        steps = [8000] * 25 + [1000] + [8000] * 4
        udf = _make_udf(steps)
        anomalies = self.detector.detect(udf, "U_TEST")
        for a in anomalies:
            assert isinstance(a.confirmed_by_iforest, bool)

    def test_anomaly_baseline_is_reasonable(self):
        """Baseline should be close to mean of stable period."""
        steps = [7500] * 25 + [1000] + [7500] * 4
        udf = _make_udf(steps)
        anomalies = self.detector.detect(udf, "U_TEST")
        step_anomalies = [a for a in anomalies if a.metric == "steps"]
        if step_anomalies:
            a = step_anomalies[0]
            assert 5000 < a.baseline < 9000, f"Baseline {a.baseline} seems wrong"