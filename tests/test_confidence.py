"""
Tests for confidence scoring engine.
Verifies numeric output range, tier classification, and scoring logic.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.confidence_engine import ConfidenceEngine
from src.utils import Pattern, Anomaly, Evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pattern(r_squared: float, metric: str = "steps") -> Pattern:
    return Pattern(
        name="Test Pattern",
        description="test",
        metric=metric,
        direction="increasing",
        slope=0.1,
        r_squared=r_squared,
        week_delta=5.0,
        evidence=[Evidence("test ev", metric, 100.0, 90.0, 10.0)],
    )


def _make_anomaly(z_score: float, iforest: bool = False) -> Anomaly:
    return Anomaly(
        date="2026-01-15",
        metric="steps",
        value=2000.0,
        baseline=8000.0,
        z_score=z_score,
        reason="test reason",
        evidence="test evidence",
        severity="medium",
        confirmed_by_iforest=iforest,
    )


def _simple_udf(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"steps": np.random.normal(7000, 500, n)})


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestConfidenceEngine:
    engine = ConfidenceEngine()

    def test_score_in_valid_range(self):
        """Confidence must always be in [0.0, 1.0]."""
        for r2 in [0.0, 0.3, 0.5, 0.8, 1.0]:
            pattern = _make_pattern(r2)
            score, tier = self.engine.score_pattern(pattern, _simple_udf())
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for R²={r2}"

    def test_high_r2_yields_higher_confidence(self):
        """Pattern with R²=0.9 should score higher than R²=0.1."""
        udf = _simple_udf()
        score_high, _ = self.engine.score_pattern(_make_pattern(0.9), udf)
        score_low, _ = self.engine.score_pattern(_make_pattern(0.1), udf)
        assert score_high > score_low

    def test_tier_classification(self):
        """Tier boundaries: ≥0.75 = high, ≥0.50 = medium, <0.50 = low."""
        from src.utils import classify_confidence
        assert classify_confidence(0.80) == "high"
        assert classify_confidence(0.75) == "high"
        assert classify_confidence(0.60) == "medium"
        assert classify_confidence(0.50) == "medium"
        assert classify_confidence(0.49) == "low"
        assert classify_confidence(0.0) == "low"

    def test_anomaly_score_scales_with_zscore(self):
        """Higher |z| → higher anomaly confidence."""
        score_z2, _ = self.engine.score_anomaly(_make_anomaly(-2.1))
        score_z4, _ = self.engine.score_anomaly(_make_anomaly(-4.0))
        assert score_z4 > score_z2

    def test_iforest_corroboration_increases_score(self):
        """IForest corroboration should push score slightly higher."""
        without = _make_anomaly(-2.5, iforest=False)
        with_corr = _make_anomaly(-2.5, iforest=True)
        score_w, _ = self.engine.score_anomaly(without)
        score_c, _ = self.engine.score_anomaly(with_corr)
        assert score_c >= score_w

    def test_correlation_score_in_range(self):
        """Correlation confidence must be in [0, 1]."""
        for r in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            score, tier = self.engine.score_correlation(r, n_samples=30)
            assert 0.0 <= score <= 1.0
            assert tier in ("low", "medium", "high")

    def test_composite_pattern_penalized(self):
        """Composite metric patterns should score lower than direct metrics."""
        udf = _simple_udf()
        direct = _make_pattern(0.7, metric="steps")
        composite = _make_pattern(0.7, metric="composite")
        score_d, _ = self.engine.score_pattern(direct, udf)
        score_c, _ = self.engine.score_pattern(composite, udf)
        assert score_d > score_c, "Composite should be penalized vs direct metric"

    def test_anomaly_at_threshold_z2(self):
        """Anomaly exactly at z=2.0 should have low-medium confidence."""
        score, tier = self.engine.score_anomaly(_make_anomaly(2.0))
        assert tier in ("low", "medium"), f"z=2.0 anomaly expected low/medium, got {tier}"

    def test_extreme_anomaly_high_confidence(self):
        """z=4.0 anomaly should have high confidence."""
        score, tier = self.engine.score_anomaly(_make_anomaly(-4.0))
        assert tier == "high", f"Extreme anomaly should be high confidence, got {tier}"