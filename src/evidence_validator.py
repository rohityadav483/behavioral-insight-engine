"""
evidence_validator.py
Evidence sufficiency gate.
Blocks insight generation when evidence is weak, noisy, or insufficient.
Any insight that fails validation returns an abstention message instead of a claim.
"""

from __future__ import annotations

from src.utils import CONFIDENCE_ABSTAIN_FLOOR, MIN_DAYS_FOR_INSIGHT, get_logger

logger = get_logger(__name__)

ABSTAIN_MESSAGE = "Insufficient evidence for reliable behavioral interpretation."


class EvidenceValidator:
    """
    Gate keeper between engines and insight generation.
    Validates that sufficient, consistent evidence exists before allowing claims.
    """

    def __init__(
        self,
        min_confidence: float = CONFIDENCE_ABSTAIN_FLOOR,
        min_days: int = MIN_DAYS_FOR_INSIGHT,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_days = min_days

    # ── Public ────────────────────────────────────────────────────────────────

    def validate_pattern(
        self,
        confidence: float,
        n_days: int,
        r_squared: float,
    ) -> tuple[bool, str]:
        """
        Validate a pattern insight.
        Returns (is_valid, reason_if_invalid).
        """
        if n_days < self.min_days:
            return False, f"Only {n_days} data points (minimum: {self.min_days})."
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}."
        if r_squared < 0.05:
            return False, f"Trend R²={r_squared:.3f} — no reliable directional signal."
        return True, ""

    def validate_anomaly(
        self,
        confidence: float,
        z_score: float,
    ) -> tuple[bool, str]:
        """
        Validate an anomaly.
        Anomalies are gated on confidence and z-score magnitude.
        """
        if confidence < self.min_confidence:
            return False, f"Anomaly confidence {confidence:.2f} below threshold."
        if abs(z_score) < 2.0:
            return False, f"|z-score| {abs(z_score):.2f} does not meet minimum 2.0."
        return True, ""

    def validate_correlation(
        self,
        confidence: float,
        r_value: float,
        n_samples: int,
    ) -> tuple[bool, str]:
        """
        Validate a correlation insight.
        Requires |r| ≥ 0.4 and adequate sample size.
        """
        if n_samples < self.min_days:
            return False, f"Only {n_samples} samples for correlation (minimum: {self.min_days})."
        if abs(r_value) < 0.40:
            return False, f"|r|={abs(r_value):.2f} — weak correlation, not reported."
        if confidence < self.min_confidence:
            return False, f"Correlation confidence {confidence:.2f} below threshold."
        return True, ""

    def validate_persona(
        self,
        confidence: float,
        n_days: int,
    ) -> tuple[bool, str]:
        """
        Validate persona assignment.
        Requires enough history and reasonable confidence.
        """
        if n_days < self.min_days:
            return False, f"Only {n_days} days of history for persona assignment."
        if confidence < self.min_confidence:
            return False, f"Persona confidence {confidence:.2f} below threshold."
        return True, ""

    @staticmethod
    def abstain_message() -> str:
        """Standard abstention message."""
        return ABSTAIN_MESSAGE