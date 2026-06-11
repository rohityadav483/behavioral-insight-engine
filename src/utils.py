"""Shared utilities, constants, type aliases, logging setup."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Constants ─────────────────────────────────────────────────────────────────

ZSCORE_THRESHOLD = 2.0          # flag anomaly if |z| > this
CONFIDENCE_ABSTAIN_FLOOR = 0.40 # abstain if confidence < this
ROLLING_WINDOW = 7              # days for rolling stats
MIN_DAYS_FOR_INSIGHT = 5        # min data points needed before any insight
ISOLATION_CONTAMINATION = 0.10  # Isolation Forest expected anomaly fraction

CONFIDENCE_TIERS = {
    "high":   (0.75, 1.0),
    "medium": (0.50, 0.75),
    "low":    (0.0,  0.50),
}

COLUMNS_REQUIRED = [
    "user_id", "date", "steps", "sleep_hours",
    "screen_time_hours", "deep_work_hours", "exercise_minutes",
]

# ── Data Classes (shared Insight / Anomaly structs) ───────────────────────────

@dataclass
class Evidence:
    description: str
    metric: str
    value: float
    baseline: float
    delta: float


@dataclass
class Anomaly:
    date: str
    metric: str
    value: float
    baseline: float
    z_score: float
    reason: str
    evidence: str
    severity: str          # "low" | "medium" | "high"
    confirmed_by_iforest: bool = False


@dataclass
class Pattern:
    name: str
    description: str
    metric: str
    direction: str          # "increasing" | "decreasing" | "stable"
    slope: float
    r_squared: float
    week_delta: float       # week-over-week % change
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Insight:
    user_id: str
    insight_type: str       # "pattern" | "anomaly" | "correlation" | "persona"
    title: str
    raw_description: str    # deterministic text
    narration: str          # Groq-generated text (empty if Groq disabled)
    confidence: float
    confidence_tier: str
    evidence: list[Evidence] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""
    week: Optional[int] = None


@dataclass
class CorrelationResult:
    metric_a: str
    metric_b: str
    r_value: float
    direction: str          # "positive" | "negative"
    interpretation: str     # human-readable, hedged language
    significant: bool       # |r| >= 0.40


@dataclass
class Persona:
    label: str
    description: str
    supporting_metrics: dict[str, float]
    groq_wording: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_confidence(score: float) -> str:
    """Map numeric confidence → tier label."""
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    return "low"


def severity_from_zscore(z: float) -> str:
    """Map |z-score| → severity label."""
    abs_z = abs(z)
    if abs_z >= 3.5:
        return "high"
    elif abs_z >= 2.5:
        return "medium"
    return "low"


def ensure_outputs_dir() -> str:
    """Create outputs/ dir if missing. Returns path."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(path, exist_ok=True)
    return path