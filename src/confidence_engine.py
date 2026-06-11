"""
confidence_engine.py
Confidence scoring for patterns and insights.
Score = weighted combination of:
  - trend strength (R²)
  - evidence volume (n supporting days normalized)
  - consistency (1 - coefficient_of_variation)
Range: 0.0 → 1.0
Tier: Low | Medium | High
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import Pattern, Anomaly, classify_confidence, get_logger

logger = get_logger(__name__)

# Weight distribution for confidence formula
W_TREND_STRENGTH = 0.40     # R² of linear fit
W_EVIDENCE_VOLUME = 0.35    # normalized day count
W_CONSISTENCY = 0.25        # 1 - CV

MAX_EVIDENCE_DAYS = 14      # saturation point for evidence volume


class ConfidenceEngine:
    """Compute confidence scores for patterns and anomalies."""

    # ── Patterns ──────────────────────────────────────────────────────────────

    def score_pattern(self, pattern: Pattern, udf: pd.DataFrame) -> tuple[float, str]:
        """
        Return (confidence_score, tier) for a detected pattern.
        Uses pattern R², evidence count, and metric consistency.
        """
        # 1. Trend strength
        trend_strength = min(float(pattern.r_squared), 1.0)

        # 2. Evidence volume
        evidence_count = len(pattern.evidence)
        ev_volume = min(evidence_count / MAX_EVIDENCE_DAYS, 1.0)

        # 3. Metric consistency in the udf
        consistency = self._metric_consistency(udf, pattern.metric)

        score = (
            W_TREND_STRENGTH * trend_strength
            + W_EVIDENCE_VOLUME * ev_volume
            + W_CONSISTENCY * consistency
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Penalize composite patterns slightly (weaker causal chain)
        if pattern.metric == "composite":
            score = score * 0.85

        tier = classify_confidence(score)
        logger.debug(f"Pattern '{pattern.name}': confidence={score:.2f} ({tier})")
        return round(score, 3), tier

    # ── Anomalies ─────────────────────────────────────────────────────────────

    def score_anomaly(self, anomaly: Anomaly) -> tuple[float, str]:
        """
        Return (confidence_score, tier) for an anomaly.
        Based on |z-score| magnitude and IForest corroboration.
        """
        abs_z = abs(anomaly.z_score)

        # Map z-score to 0-1 confidence (saturates at |z|=4)
        z_based = min(abs_z / 4.0, 1.0)

        # Corroboration bonus
        corroboration_bonus = 0.10 if anomaly.confirmed_by_iforest else 0.0

        score = float(np.clip(z_based + corroboration_bonus, 0.0, 1.0))
        tier = classify_confidence(score)
        logger.debug(f"Anomaly on {anomaly.date} ({anomaly.metric}): confidence={score:.2f} ({tier})")
        return round(score, 3), tier

    # ── Correlation Insights ──────────────────────────────────────────────────

    def score_correlation(self, r_value: float, n_samples: int) -> tuple[float, str]:
        """
        Confidence for a correlation finding.
        |r| × sample_adequacy_factor.
        """
        abs_r = abs(r_value)
        # sample adequacy: saturates at n=30
        sample_factor = min(n_samples / 30.0, 1.0)
        score = float(np.clip(abs_r * sample_factor, 0.0, 1.0))
        tier = classify_confidence(score)
        return round(score, 3), tier

    # ── Util ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _metric_consistency(udf: pd.DataFrame, metric: str) -> float:
        """
        1 - coefficient_of_variation for the metric in this DataFrame.
        Returns 0.5 if metric not available or zero mean.
        """
        if metric not in udf.columns or metric == "composite":
            return 0.5
        series = udf[metric].dropna()
        if len(series) < 3 or series.mean() == 0:
            return 0.5
        cv = series.std() / series.mean()
        return float(np.clip(1 - cv, 0.0, 1.0))