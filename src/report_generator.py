"""
report_generator.py
Generate weekly behavioral reports and JSON insight exports.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.groq_client import GroqClient
from src.utils import Insight, ensure_outputs_dir, get_logger

logger = get_logger(__name__)


@dataclass
class WeeklyReport:
    user_id: str
    week: int
    metrics_summary: dict[str, float]
    anomaly_count: int
    pattern_names: list[str]
    narrative: str
    insights_this_week: list[str]


class ReportGenerator:
    """Generate weekly summaries and exportable JSON artifacts."""

    def __init__(self, groq_client: GroqClient) -> None:
        self.groq = groq_client

    # ── Weekly Reports ────────────────────────────────────────────────────────

    def generate_weekly_reports(
        self,
        user_id: str,
        enriched_df: pd.DataFrame,
        all_results: dict,
    ) -> list[WeeklyReport]:
        """Generate one WeeklyReport per week for a user."""
        if "week" not in enriched_df.columns:
            return []

        reports = []
        for week_num in sorted(enriched_df["week"].unique()):
            wdf = enriched_df[enriched_df["week"] == week_num]
            report = self._build_report(user_id, week_num, wdf, all_results)
            reports.append(report)

        logger.info(f"User {user_id}: {len(reports)} weekly reports generated.")
        return reports

    def _build_report(
        self,
        user_id: str,
        week_num: int,
        wdf: pd.DataFrame,
        all_results: dict,
    ) -> WeeklyReport:
        """Build a single week's report."""
        # Metrics summary for this week
        metric_cols = [
            "steps", "sleep_hours", "screen_time_hours",
            "deep_work_hours", "exercise_minutes",
            "productivity_score", "recovery_score",
        ]
        metrics_summary = {
            col: round(float(wdf[col].mean()), 3)
            for col in metric_cols
            if col in wdf.columns
        }

        # Anomalies this week
        week_dates = set(wdf["date"].dt.strftime("%Y-%m-%d").tolist())
        anomalies_this_week = [
            a for a in all_results.get("anomalies", [])
            if a.date in week_dates
        ]

        # Patterns (not week-specific, include all)
        pattern_names = [p.name for p in all_results.get("patterns", [])]

        # Insights this week
        insights: list[Insight] = all_results.get("insights", [])
        insight_titles = [
            i.title for i in insights
            if i.week == week_num or i.week is None
        ][:5]  # cap display

        # Groq narrative
        narrative = self.groq.narrate_weekly_report(
            user_id=user_id,
            week_number=week_num,
            metrics_summary=metrics_summary,
            anomaly_count=len(anomalies_this_week),
            top_patterns=pattern_names[:3],
        )

        return WeeklyReport(
            user_id=user_id,
            week=week_num,
            metrics_summary=metrics_summary,
            anomaly_count=len(anomalies_this_week),
            pattern_names=pattern_names,
            narrative=narrative,
            insights_this_week=insight_titles,
        )

    # ── JSON Export ───────────────────────────────────────────────────────────

    def export_insights_json(
        self,
        all_user_results: dict[str, dict],
        filename: str = "insights_export.json",
    ) -> str:
        """
        Export all insights for all users as structured JSON.
        Returns path of written file.
        """
        output_dir = ensure_outputs_dir()
        output_path = os.path.join(output_dir, filename)

        export: dict[str, Any] = {}
        for user_id, results in all_user_results.items():
            insights: list[Insight] = results.get("insights", [])
            anomalies = results.get("anomalies", [])
            patterns = results.get("patterns", [])
            persona = results.get("persona")

            export[user_id] = {
                "persona": {
                    "label": persona.label if persona else "Unknown",
                    "description": persona.description if persona else "",
                },
                "feature_summary": results.get("feature_summary", {}),
                "insights": [
                    {
                        "type": ins.insight_type,
                        "title": ins.title,
                        "description": ins.narration or ins.raw_description,
                        "confidence": ins.confidence,
                        "confidence_tier": ins.confidence_tier,
                        "abstained": ins.abstained,
                        "evidence": [
                            {
                                "description": e.description,
                                "metric": e.metric,
                                "value": e.value,
                                "baseline": e.baseline,
                                "delta": e.delta,
                            }
                            for e in ins.evidence
                        ],
                    }
                    for ins in insights
                ],
                "anomaly_summary": [
                    {
                        "date": a.date,
                        "metric": a.metric,
                        "z_score": a.z_score,
                        "severity": a.severity,
                        "reason": a.reason,
                    }
                    for a in anomalies
                ],
                "pattern_summary": [
                    {
                        "name": p.name,
                        "metric": p.metric,
                        "direction": p.direction,
                        "r_squared": p.r_squared,
                        "week_delta_pct": p.week_delta,
                    }
                    for p in patterns
                ],
            }

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2, default=str)

        logger.info(f"Insights exported to: {output_path}")
        return output_path

    def export_user_report_json(
        self,
        user_id: str,
        weekly_reports: list[WeeklyReport],
        output_filename: str | None = None,
    ) -> str:
        """Export weekly reports for a single user as JSON."""
        output_dir = ensure_outputs_dir()
        fname = output_filename or f"{user_id}_weekly_report.json"
        output_path = os.path.join(output_dir, fname)

        data = [
            {
                "week": r.week,
                "metrics_summary": r.metrics_summary,
                "anomaly_count": r.anomaly_count,
                "pattern_names": r.pattern_names,
                "narrative": r.narrative,
            }
            for r in weekly_reports
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Weekly report exported to: {output_path}")
        return output_path