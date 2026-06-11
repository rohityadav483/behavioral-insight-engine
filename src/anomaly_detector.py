"""
anomaly_detector.py
Explainable anomaly detection per user.
Primary method: z-score vs personal rolling baseline.
Secondary method: Isolation Forest (corroboration only, not primary signal).
Every anomaly carries: reason, evidence, severity, deviation_from_baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils import (
    ISOLATION_CONTAMINATION,
    MIN_DAYS_FOR_INSIGHT,
    ZSCORE_THRESHOLD,
    Anomaly,
    get_logger,
    severity_from_zscore,
)

logger = get_logger(__name__)

SIGNAL_LABELS = {
    "steps": "daily steps",
    "sleep_hours": "sleep duration",
    "screen_time_hours": "screen time",
    "deep_work_hours": "deep work hours",
    "exercise_minutes": "exercise duration",
}

SIGNAL_DIRECTIONS = {
    "steps": "drop",
    "sleep_hours": "drop",
    "screen_time_hours": "spike",
    "deep_work_hours": "drop",
    "exercise_minutes": "drop",
}


class AnomalyDetector:
    """Detect and explain per-user behavioral anomalies."""

    def __init__(
        self,
        z_threshold: float = ZSCORE_THRESHOLD,
        use_iforest: bool = True,
        contamination: float = ISOLATION_CONTAMINATION,
    ) -> None:
        self.z_threshold = z_threshold
        self.use_iforest = use_iforest
        self.contamination = contamination

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, udf: pd.DataFrame, user_id: str) -> list[Anomaly]:
        """
        Run anomaly detection on single-user enriched DataFrame.
        Returns list of Anomaly objects.
        """
        if len(udf) < MIN_DAYS_FOR_INSIGHT:
            logger.warning(f"User {user_id}: not enough data for anomaly detection.")
            return []

        # 1. Z-score anomalies (primary)
        flagged: dict[str, set[int]] = {}  # metric → set of row indices
        anomalies: list[Anomaly] = []

        for metric in SIGNAL_LABELS:
            z_col = f"{metric}_zscore"
            if z_col not in udf.columns:
                continue
            anomaly_rows = udf[udf[z_col].abs() > self.z_threshold].index.tolist()
            flagged[metric] = set(anomaly_rows)

        # 2. Isolation Forest (secondary — flag rows as suspicious)
        iforest_flagged: set[int] = set()
        if self.use_iforest:
            iforest_flagged = self._run_iforest(udf)

        # 3. Build Anomaly objects
        seen: set[tuple] = set()
        for metric, row_indices in flagged.items():
            for idx in row_indices:
                key = (str(udf.at[idx, "date"]), metric)
                if key in seen:
                    continue
                seen.add(key)

                z_col = f"{metric}_zscore"
                baseline_col = f"{metric}_baseline"

                z_score = float(udf.at[idx, z_col])
                value = float(udf.at[idx, metric])
                baseline = float(udf.at[idx, baseline_col]) if baseline_col in udf.columns else float(udf[metric].mean())
                deviation = value - baseline
                date_str = str(udf.at[idx, "date"])[:10]

                reason, evidence = self._build_explanation(
                    metric, value, baseline, z_score, deviation
                )

                anomalies.append(Anomaly(
                    date=date_str,
                    metric=metric,
                    value=value,
                    baseline=round(baseline, 2),
                    z_score=round(z_score, 2),
                    reason=reason,
                    evidence=evidence,
                    severity=severity_from_zscore(z_score),
                    confirmed_by_iforest=(idx in iforest_flagged),
                ))

        anomalies.sort(key=lambda a: a.date)
        logger.info(f"User {user_id}: {len(anomalies)} anomalies detected.")
        return anomalies

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_iforest(self, udf: pd.DataFrame) -> set[int]:
        """
        Run Isolation Forest on multi-signal matrix.
        Returns set of row indices flagged as anomalous.
        """
        feature_cols = [c for c in SIGNAL_LABELS if c in udf.columns]
        X = udf[feature_cols].fillna(udf[feature_cols].mean())

        if len(X) < 10:
            return set()

        clf = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        preds = clf.fit_predict(X)  # -1 = anomaly
        flagged_indices = set(udf.index[preds == -1].tolist())
        logger.debug(f"IForest flagged {len(flagged_indices)} rows.")
        return flagged_indices

    def _build_explanation(
        self,
        metric: str,
        value: float,
        baseline: float,
        z_score: float,
        deviation: float,
    ) -> tuple[str, str]:
        """
        Build human-readable reason and evidence strings.
        Uses only the provided numeric evidence — no LLM here.
        """
        label = SIGNAL_LABELS.get(metric, metric)
        direction = SIGNAL_DIRECTIONS.get(metric, "deviation")
        abs_z = abs(z_score)
        pct_dev = ((value - baseline) / abs(baseline) * 100) if baseline != 0 else 0

        # Reason
        if z_score < 0:
            reason = (
                f"Unusually low {label}: {value:.1f} vs personal baseline of {baseline:.1f} "
                f"({abs(pct_dev):.0f}% below average)."
            )
        else:
            reason = (
                f"Unusually high {label}: {value:.1f} vs personal baseline of {baseline:.1f} "
                f"({abs(pct_dev):.0f}% above average)."
            )

        # Evidence
        evidence = (
            f"z-score = {z_score:.2f} (threshold: ±{self.z_threshold}). "
            f"Deviation from baseline: {deviation:+.2f} units ({pct_dev:+.1f}%). "
            f"|z| = {abs_z:.2f} → severity: {severity_from_zscore(z_score)}."
        )

        return reason, evidence