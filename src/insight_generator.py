"""
insight_generator.py
Central orchestrator: runs all engines per user, validates evidence,
generates Insight objects with optional Groq narration.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.anomaly_detector import AnomalyDetector
from src.confidence_engine import ConfidenceEngine
from src.correlation_engine import CorrelationEngine
from src.evidence_validator import EvidenceValidator
from src.feature_engineering import FeatureEngineer
from src.groq_client import GroqClient
from src.pattern_detector import PatternDetector
from src.persona_engine import PersonaEngine
from src.preprocessing import Preprocessor
from src.utils import (
    Anomaly,
    CorrelationResult,
    Evidence,
    Insight,
    Pattern,
    Persona,
    get_logger,
)

logger = get_logger(__name__)


class InsightGenerator:
    """
    Full pipeline orchestrator per user.
    Produces list of validated, scored, narrated Insight objects.
    """

    def __init__(self, groq_client: Optional[GroqClient] = None) -> None:
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()
        self.confidence_engine = ConfidenceEngine()
        self.evidence_validator = EvidenceValidator()
        self.correlation_engine = CorrelationEngine()
        self.persona_engine = PersonaEngine()
        self.groq = groq_client or GroqClient()

    # ── Public ────────────────────────────────────────────────────────────────

    def generate_for_user(self, raw_udf: pd.DataFrame, user_id: str) -> dict:
        """
        Full pipeline for one user.
        Returns a dict containing all results.
        """
        logger.info(f"=== Generating insights for user {user_id} ===")

        # 1. Preprocess + enrich
        udf = self.preprocessor.process(raw_udf)
        udf = self.feature_engineer.engineer(udf)

        # 2. Persona
        persona = self.persona_engine.assign(udf, user_id)
        persona_confidence, _ = self.confidence_engine.score_correlation(0.65, len(udf))
        valid_persona, persona_fail = self.evidence_validator.validate_persona(
            persona_confidence, len(udf)
        )
        if valid_persona and self.groq.available:
            persona.groq_wording = self.groq.narrate_persona(
                persona.label, persona.description, persona.supporting_metrics, user_id
            )

        # 3. Patterns → Insights
        patterns = self.pattern_detector.detect(udf, user_id)
        pattern_insights = self._build_pattern_insights(patterns, udf, user_id)

        # 4. Anomalies → Insights
        anomalies = self.anomaly_detector.detect(udf, user_id)
        anomaly_insights = self._build_anomaly_insights(anomalies, udf, user_id)

        # 5. Correlations → Insights
        correlations = self.correlation_engine.analyze(udf, user_id)
        correlation_insights = self._build_correlation_insights(correlations, udf, user_id)

        # 6. Feature summary
        feature_summary = self.feature_engineer.get_feature_summary(udf)

        all_insights = pattern_insights + anomaly_insights + correlation_insights

        return {
            "user_id": user_id,
            "enriched_df": udf,
            "persona": persona,
            "patterns": patterns,
            "anomalies": anomalies,
            "correlations": correlations,
            "insights": all_insights,
            "feature_summary": feature_summary,
            "groq_available": self.groq.available,
        }

    def generate_all_users(
        self, full_df: pd.DataFrame
    ) -> dict[str, dict]:
        """Run pipeline for every user. Returns {user_id: results_dict}."""
        results = {}
        for user_id, udf in full_df.groupby("user_id"):
            udf_reset = udf.reset_index(drop=True)
            results[str(user_id)] = self.generate_for_user(udf_reset, str(user_id))
        return results

    def answer_query(self, query: str, user_results: dict, user_id: str) -> str:
        """Answer a free-text question using structured insight context."""
        context = {
            "user_id": user_id,
            "persona": user_results["persona"].label,
            "feature_summary": user_results["feature_summary"],
            "anomaly_count": len(user_results["anomalies"]),
            "anomalies": [
                {"date": a.date, "metric": a.metric, "reason": a.reason}
                for a in user_results["anomalies"][:5]
            ],
            "patterns": [
                {"name": p.name, "metric": p.metric, "direction": p.direction}
                for p in user_results["patterns"][:5]
            ],
        }
        return self.groq.answer_query(query, json.dumps(context, indent=2), user_id)

    # ── Private: Pattern Insights ─────────────────────────────────────────────

    def _build_pattern_insights(
        self, patterns: list[Pattern], udf: pd.DataFrame, user_id: str
    ) -> list[Insight]:
        insights = []
        for pattern in patterns:
            confidence, tier = self.confidence_engine.score_pattern(pattern, udf)
            valid, fail_reason = self.evidence_validator.validate_pattern(
                confidence, len(udf), pattern.r_squared
            )

            if not valid:
                insights.append(Insight(
                    user_id=user_id,
                    insight_type="pattern",
                    title=pattern.name,
                    raw_description=self.evidence_validator.abstain_message(),
                    narration=self.evidence_validator.abstain_message(),
                    confidence=confidence,
                    confidence_tier=tier,
                    evidence=pattern.evidence,
                    abstained=True,
                    abstain_reason=fail_reason,
                ))
                continue

            ev_bullets = [e.description for e in pattern.evidence]
            narration = self.groq.narrate_insight(
                pattern.name, pattern.description, ev_bullets, confidence, user_id
            ) if self.groq.available else pattern.description

            insights.append(Insight(
                user_id=user_id,
                insight_type="pattern",
                title=pattern.name,
                raw_description=pattern.description,
                narration=narration,
                confidence=confidence,
                confidence_tier=tier,
                evidence=pattern.evidence,
                abstained=False,
            ))
        return insights

    # ── Private: Anomaly Insights ─────────────────────────────────────────────

    def _build_anomaly_insights(
        self, anomalies: list[Anomaly], udf: pd.DataFrame, user_id: str
    ) -> list[Insight]:
        insights = []
        for anomaly in anomalies:
            confidence, tier = self.confidence_engine.score_anomaly(anomaly)
            valid, fail_reason = self.evidence_validator.validate_anomaly(
                confidence, anomaly.z_score
            )

            ev = Evidence(
                description=anomaly.evidence,
                metric=anomaly.metric,
                value=anomaly.value,
                baseline=anomaly.baseline,
                delta=anomaly.value - anomaly.baseline,
            )

            if not valid:
                insights.append(Insight(
                    user_id=user_id,
                    insight_type="anomaly",
                    title=f"Anomaly: {anomaly.metric} on {anomaly.date}",
                    raw_description=self.evidence_validator.abstain_message(),
                    narration=self.evidence_validator.abstain_message(),
                    confidence=confidence,
                    confidence_tier=tier,
                    evidence=[ev],
                    abstained=True,
                    abstain_reason=fail_reason,
                ))
                continue

            raw_desc = anomaly.reason
            narration = self.groq.narrate_insight(
                f"Behavioral anomaly on {anomaly.date}",
                raw_desc,
                [anomaly.evidence],
                confidence,
                user_id,
            ) if self.groq.available else raw_desc

            insights.append(Insight(
                user_id=user_id,
                insight_type="anomaly",
                title=f"Anomaly: {anomaly.metric} on {anomaly.date}",
                raw_description=raw_desc,
                narration=narration,
                confidence=confidence,
                confidence_tier=tier,
                evidence=[ev],
                abstained=False,
            ))
        return insights

    # ── Private: Correlation Insights ─────────────────────────────────────────

    def _build_correlation_insights(
        self, correlations: list[CorrelationResult], udf: pd.DataFrame, user_id: str
    ) -> list[Insight]:
        insights = []
        for corr in correlations:
            confidence, tier = self.confidence_engine.score_correlation(
                corr.r_value, len(udf)
            )
            valid, fail_reason = self.evidence_validator.validate_correlation(
                confidence, corr.r_value, len(udf)
            )

            ev = Evidence(
                description=corr.interpretation,
                metric=f"{corr.metric_a}↔{corr.metric_b}",
                value=corr.r_value,
                baseline=0.0,
                delta=corr.r_value,
            )

            if not valid:
                continue   # silently drop weak correlations

            narration = self.groq.narrate_insight(
                f"Signal relationship: {corr.metric_a} ↔ {corr.metric_b}",
                corr.interpretation,
                [corr.interpretation],
                confidence,
                user_id,
            ) if self.groq.available else corr.interpretation

            insights.append(Insight(
                user_id=user_id,
                insight_type="correlation",
                title=f"Correlation: {corr.metric_a} ↔ {corr.metric_b}",
                raw_description=corr.interpretation,
                narration=narration,
                confidence=confidence,
                confidence_tier=tier,
                evidence=[ev],
                abstained=False,
            ))
        return insights