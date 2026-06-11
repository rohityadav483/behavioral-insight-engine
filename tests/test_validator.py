"""
Tests for evidence sufficiency and abstention logic.
Verifies that weak evidence correctly triggers abstention.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evidence_validator import EvidenceValidator, ABSTAIN_MESSAGE


class TestEvidenceValidator:
    validator = EvidenceValidator(min_confidence=0.40, min_days=5)

    # ── Pattern validation ─────────────────────────────────────────────────

    def test_valid_pattern_passes(self):
        """High confidence + enough days + decent R² should pass."""
        valid, reason = self.validator.validate_pattern(
            confidence=0.75, n_days=30, r_squared=0.60
        )
        assert valid is True
        assert reason == ""

    def test_low_confidence_pattern_fails(self):
        """Confidence below 0.40 should fail."""
        valid, reason = self.validator.validate_pattern(
            confidence=0.30, n_days=30, r_squared=0.60
        )
        assert valid is False
        assert "Confidence" in reason

    def test_insufficient_days_fails(self):
        """Less than min_days should fail."""
        valid, reason = self.validator.validate_pattern(
            confidence=0.80, n_days=3, r_squared=0.70
        )
        assert valid is False
        assert "data points" in reason

    def test_low_r2_fails(self):
        """R² < 0.05 should fail even with high confidence."""
        valid, reason = self.validator.validate_pattern(
            confidence=0.80, n_days=30, r_squared=0.02
        )
        assert valid is False
        assert "R²" in reason

    # ── Anomaly validation ─────────────────────────────────────────────────

    def test_valid_anomaly_passes(self):
        """z=3.0 + decent confidence should pass."""
        valid, reason = self.validator.validate_anomaly(confidence=0.70, z_score=-3.0)
        assert valid is True

    def test_low_confidence_anomaly_fails(self):
        """Anomaly confidence below threshold should fail."""
        valid, reason = self.validator.validate_anomaly(confidence=0.30, z_score=-2.5)
        assert valid is False

    def test_below_z2_fails(self):
        """|z| < 2.0 should not be reported."""
        valid, reason = self.validator.validate_anomaly(confidence=0.80, z_score=-1.8)
        assert valid is False
        assert "z-score" in reason.lower() or "2.0" in reason

    def test_exactly_z2_passes_if_confident(self):
        """|z| == 2.0 is the boundary — should pass if confidence adequate."""
        valid, _ = self.validator.validate_anomaly(confidence=0.50, z_score=-2.0)
        assert valid is True

    # ── Correlation validation ─────────────────────────────────────────────

    def test_strong_correlation_passes(self):
        valid, _ = self.validator.validate_correlation(
            confidence=0.70, r_value=-0.65, n_samples=30
        )
        assert valid is True

    def test_weak_correlation_blocked(self):
        """|r| < 0.40 should be blocked."""
        valid, reason = self.validator.validate_correlation(
            confidence=0.80, r_value=0.30, n_samples=30
        )
        assert valid is False
        assert "weak" in reason.lower() or "not reported" in reason.lower()

    def test_small_sample_correlation_fails(self):
        valid, reason = self.validator.validate_correlation(
            confidence=0.80, r_value=0.70, n_samples=3
        )
        assert valid is False

    # ── Persona validation ─────────────────────────────────────────────────

    def test_valid_persona_passes(self):
        valid, _ = self.validator.validate_persona(confidence=0.65, n_days=30)
        assert valid is True

    def test_insufficient_days_persona_fails(self):
        valid, _ = self.validator.validate_persona(confidence=0.80, n_days=2)
        assert valid is False

    # ── Abstain message ────────────────────────────────────────────────────

    def test_abstain_message_is_defined(self):
        msg = EvidenceValidator.abstain_message()
        assert isinstance(msg, str)
        assert len(msg) > 10
        assert msg == ABSTAIN_MESSAGE

    def test_abstain_message_does_not_make_claims(self):
        """Abstain message must not contain hedged or direct claims."""
        msg = ABSTAIN_MESSAGE.lower()
        assert "patterns suggest" not in msg
        assert "confidence" in msg or "insufficient" in msg or "evidence" in msg