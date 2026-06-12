"""
dashboard/app.py
Streamlit Dashboard for the Behavioral Insight Engine.

Sections:
  1. Overview
  2. Behavioral Scores (Radar)
  3. Trend Charts
  4. Anomaly Timeline
  5. Weekly Reports
  6. AI Insights
  7. Persona Summary
  8. Correlation Analysis
  9. Conversational Query Box

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import DataLoader
from src.groq_client import GroqClient
from src.insight_generator import InsightGenerator
from src.report_generator import ReportGenerator
from src.utils import ensure_outputs_dir

DATA_PATH = os.path.join(ROOT, "data", "behavioral_data.csv")
CACHE_PATH = os.path.join(ROOT, "outputs", "pipeline_cache.pkl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Insight Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Cards */
  .metric-card {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    border-left: 4px solid #6366f1;
    margin-bottom: 0.5rem;
  }
  /* Confidence badges */
  .badge-high   { background:#15803d; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
  .badge-medium { background:#b45309; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
  .badge-low    { background:#dc2626; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
  /* Persona card */
  .persona-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
  }
  /* Section title */
  .section-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }
  /* Anomaly row */
  .anomaly-high   { border-left: 4px solid #dc2626; padding-left: 0.8rem; }
  .anomaly-medium { border-left: 4px solid #f59e0b; padding-left: 0.8rem; }
  .anomaly-low    { border-left: 4px solid #3b82f6; padding-left: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "steps": "#6366f1",
    "sleep_hours": "#06b6d4",
    "screen_time_hours": "#f59e0b",
    "deep_work_hours": "#10b981",
    "exercise_minutes": "#ec4899",
    "productivity_score": "#8b5cf6",
    "recovery_score": "#14b8a6",
}


# ── Data loading / caching ────────────────────────────────────────────────────

def _load_dotenv() -> None:
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@st.cache_resource(show_spinner=False)
def load_pipeline_results() -> dict:
    """Load cached results or run pipeline. Cached across reruns."""
    _load_dotenv()
    ensure_outputs_dir()

    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # Run pipeline if no cache
    groq = GroqClient()
    generator = InsightGenerator(groq_client=groq)
    reporter = ReportGenerator(groq_client=groq)

    loader = DataLoader(DATA_PATH)
    full_df = loader.load()
    all_results = generator.generate_all_users(full_df)

    for user_id, results in all_results.items():
        edf = results["enriched_df"]
        all_results[user_id]["weekly_reports"] = reporter.generate_weekly_reports(
            user_id, edf, results
        )

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_results, f)

    return all_results


def get_user_data(all_results: dict, user_id: str) -> dict:
    return all_results[user_id]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(all_results: dict) -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## 🧠 Behavioral Insight Engine")
        st.markdown("*Chronis AI/ML Assessment — Task A*")
        st.divider()

        users = sorted(all_results.keys())
        user_id = st.selectbox("👤 Select User", users, index=0)

        st.divider()
        section = st.radio(
            "📋 Navigate",
            [
                "📊 Overview",
                "🎯 Behavioral Scores",
                "📈 Trend Charts",
                "🚨 Anomaly Timeline",
                "📅 Weekly Reports",
                "💡 AI Insights",
                "🪪 Persona Summary",
                "🔗 Correlation Analysis",
                "💬 Query Interface",
            ],
        )

        st.divider()
        groq_ok = all_results[user_id].get("groq_available", False)
        if groq_ok:
            st.success("✓ Groq AI enabled")
        else:
            st.warning("⚠ Groq AI disabled\nSet GROQ_API_KEY in .env")

        st.divider()
        if st.button("🔄 Refresh Analysis"):
            if os.path.isfile(CACHE_PATH):
                os.remove(CACHE_PATH)
            st.cache_resource.clear()
            st.rerun()

    return user_id, section


# ── Overview ──────────────────────────────────────────────────────────────────

def render_overview(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 📊 Overview — {user_id}")

    edf: pd.DataFrame = user_data["enriched_df"]
    persona = user_data["persona"]
    anomalies = user_data["anomalies"]
    patterns = user_data["patterns"]
    fs = user_data["feature_summary"]

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📅 Days Tracked", len(edf))
    with col2:
        st.metric("⚡ Anomalies", len(anomalies))
    with col3:
        st.metric("📈 Patterns", len(patterns))
    with col4:
        prod = fs.get("productivity_score", 0)
        st.metric("🎯 Productivity", f"{prod:.2f}")
    with col5:
        rec = fs.get("recovery_score", 0)
        st.metric("💚 Recovery Score", f"{rec:.2f}")

    st.divider()

    # Mini trend spark chart — last 30 days
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("#### 📉 Signal Overview (30 days)")
        fig = _sparkline_chart(edf)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("#### 🪪 Current Persona")
        st.markdown(f"""
<div class="persona-card">
  <h3 style="margin:0 0 0.5rem 0;">{'🟢' if 'Consistent' in persona.label or 'Recovery' in persona.label else '🔴' if 'Burnout' in persona.label else '🟡'} {persona.label}</h3>
  <p style="margin:0;font-size:0.9rem;opacity:0.9;">{persona.groq_wording or persona.description}</p>
</div>
""", unsafe_allow_html=True)

        # Recent anomaly count this week
        weeks = sorted(edf["week"].unique()) if "week" in edf.columns else []
        if weeks:
            last_week = weeks[-1]
            last_week_dates = set(
                edf[edf["week"] == last_week]["date"].dt.strftime("%Y-%m-%d").tolist()
            )
            recent_anomalies = [a for a in anomalies if a.date in last_week_dates]
            if recent_anomalies:
                st.warning(f"⚠ {len(recent_anomalies)} anomalies detected this week")
            else:
                st.success("✓ No anomalies this week")


def _sparkline_chart(edf: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        "Daily Steps", "Sleep Hours",
        "Screen Time", "Deep Work",
        "Exercise (min)", "Productivity Score",
    ], shared_xaxes=False, vertical_spacing=0.12)

    pairs = [
        ("steps", 1, 1), ("sleep_hours", 1, 2),
        ("screen_time_hours", 2, 1), ("deep_work_hours", 2, 2),
        ("exercise_minutes", 3, 1), ("productivity_score", 3, 2),
    ]

    for metric, row, col in pairs:
        if metric not in edf.columns:
            continue
        color = COLORS.get(metric, "#6366f1")
        fig.add_trace(
            go.Scatter(
                x=edf["date"], y=edf[metric],
                mode="lines", line=dict(color=color, width=2),
                name=metric, showlegend=False,
            ),
            row=row, col=col,
        )
        # Rolling mean overlay
        roll_col = f"{metric}_roll_mean"
        if roll_col in edf.columns:
            fig.add_trace(
                go.Scatter(
                    x=edf["date"], y=edf[roll_col],
                    mode="lines", line=dict(color=color, width=1.5, dash="dot"),
                    name=f"{metric} avg", showlegend=False, opacity=0.5,
                ),
                row=row, col=col,
            )

    fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ── Behavioral Scores ─────────────────────────────────────────────────────────

def render_behavioral_scores(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 🎯 Behavioral Scores — {user_id}")
    fs = user_data["feature_summary"]

    if not fs:
        st.info("No feature data available.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        # Radar chart
        categories = list(fs.keys())
        values = list(fs.values())
        # Normalize 0-1 for radar
        vmin, vmax = min(values), max(values)
        if vmax != vmin:
            norm_values = [(v - vmin) / (vmax - vmin) for v in values]
        else:
            norm_values = [0.5] * len(values)

        fig = go.Figure(data=go.Scatterpolar(
            r=norm_values + [norm_values[0]],
            theta=[c.replace("_", " ").title() for c in categories] + [categories[0].replace("_", " ").title()],
            fill="toself",
            fillcolor="rgba(99,102,241,0.25)",
            line=dict(color="#6366f1", width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Behavioral Feature Radar",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Feature Breakdown")
        for feat, val in fs.items():
            label = feat.replace("_", " ").title()
            # Determine color based on normalized value
            norm = (val - min(fs.values())) / max((max(fs.values()) - min(fs.values())), 0.001)
            color = "#10b981" if norm > 0.6 else "#f59e0b" if norm > 0.35 else "#ef4444"
            bar_pct = int(norm * 100)
            st.markdown(f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
    <span style="font-size:0.85rem;font-weight:500;">{label}</span>
    <span style="font-size:0.85rem;color:{color};font-weight:600;">{val:.3f}</span>
  </div>
  <div style="background:rgba(128,128,128,0.15);border-radius:99px;height:6px;">
    <div style="width:{bar_pct}%;background:{color};height:6px;border-radius:99px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Trend Charts ──────────────────────────────────────────────────────────────

def render_trend_charts(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 📈 Trend Charts — {user_id}")
    edf: pd.DataFrame = user_data["enriched_df"]

    metric = st.selectbox(
        "Select metric",
        ["steps", "sleep_hours", "screen_time_hours", "deep_work_hours",
         "exercise_minutes", "productivity_score", "recovery_score",
         "behavioral_stability_score"],
        index=0,
    )

    if metric not in edf.columns:
        st.warning(f"Metric '{metric}' not available.")
        return

    color = COLORS.get(metric, "#6366f1")

    fig = go.Figure()

    # Raw values
    fig.add_trace(go.Scatter(
        x=edf["date"], y=edf[metric],
        mode="lines+markers", name=metric.replace("_", " ").title(),
        line=dict(color=color, width=2),
        marker=dict(size=5),
    ))

    # Rolling mean
    roll_col = f"{metric}_roll_mean"
    if roll_col in edf.columns:
        fig.add_trace(go.Scatter(
            x=edf["date"], y=edf[roll_col],
            mode="lines", name="7-day avg",
            line=dict(color=color, dash="dot", width=2),
            opacity=0.6,
        ))

    # Baseline
    baseline_col = f"{metric}_baseline"
    if baseline_col in edf.columns:
        fig.add_trace(go.Scatter(
            x=edf["date"], y=edf[baseline_col],
            mode="lines", name="Expanding baseline",
            line=dict(color="gray", dash="dash", width=1.5),
            opacity=0.4,
        ))

    # Anomaly markers
    anomalies = user_data.get("anomalies", [])
    anom_for_metric = [a for a in anomalies if a.metric == metric]
    if anom_for_metric:
        anom_dates = [pd.to_datetime(a.date) for a in anom_for_metric]
        anom_vals = []
        for d in anom_dates:
            row = edf[edf["date"] == d]
            anom_vals.append(float(row[metric].iloc[0]) if len(row) > 0 else None)

        fig.add_trace(go.Scatter(
            x=anom_dates, y=anom_vals,
            mode="markers", name="Anomaly",
            marker=dict(symbol="x", size=12, color="#ef4444", line=dict(width=2)),
        ))

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} over time",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, width="stretch")

    # Pattern summary for this metric
    patterns = [p for p in user_data.get("patterns", []) if p.metric == metric]
    if patterns:
        st.markdown("#### Detected Patterns")
        for p in patterns:
            icon = "📈" if p.direction == "increasing" else "📉" if p.direction == "decreasing" else "➡️"
            st.markdown(f"{icon} **{p.name}** — slope: `{p.slope:.4f}`, R²: `{p.r_squared:.2f}`, week Δ: `{p.week_delta:+.1f}%`")
            for ev in p.evidence:
                st.caption(f"Evidence: {ev.description}")


# ── Anomaly Timeline ──────────────────────────────────────────────────────────

def render_anomaly_timeline(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 🚨 Anomaly Timeline — {user_id}")

    anomalies = user_data.get("anomalies", [])
    if not anomalies:
        st.success("✅ No anomalies detected for this user.")
        return

    # Filter controls
    severity_filter = st.multiselect(
        "Filter by severity", ["high", "medium", "low"],
        default=["high", "medium", "low"]
    )
    filtered = [a for a in anomalies if a.severity in severity_filter]

    st.markdown(f"**{len(filtered)} anomalies** (of {len(anomalies)} total)")
    st.divider()

    for a in filtered:
        sev_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"}.get(a.severity, "#6366f1")
        badge_html = f'<span class="badge-{a.severity}">{a.severity.upper()}</span>'
        iforest_badge = ' <span style="background:#6366f1;color:#fff;padding:2px 8px;border-radius:99px;font-size:0.72rem;">+ IForest</span>' if a.confirmed_by_iforest else ""

        with st.expander(
            f"📅 {a.date} | {a.metric.replace('_', ' ').title()} | z={a.z_score:+.2f}",
            expanded=(a.severity == "high"),
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{badge_html}{iforest_badge}", unsafe_allow_html=True)
                st.markdown(f"**Reason:** {a.reason}")
                st.markdown(f"**Evidence:** {a.evidence}")
            with col2:
                st.metric("Observed", f"{a.value:.1f}")
                st.metric("Baseline", f"{a.baseline:.1f}")
                delta = a.value - a.baseline
                st.metric("Δ from baseline", f"{delta:+.1f}")

    # Timeline scatter
    if filtered:
        st.markdown("#### Anomaly Scatter Timeline")
        anom_df = pd.DataFrame([{
            "date": pd.to_datetime(a.date),
            "metric": a.metric,
            "z_score": a.z_score,
            "severity": a.severity,
            "value": a.value,
        } for a in filtered])

        fig = px.scatter(
            anom_df, x="date", y="z_score", color="severity",
            symbol="metric", size=anom_df["z_score"].abs(),
            color_discrete_map={"high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"},
            hover_data=["metric", "value"],
            title="Anomalies by date and z-score",
        )
        fig.add_hline(y=2.0, line_dash="dot", line_color="gray", annotation_text="z=+2")
        fig.add_hline(y=-2.0, line_dash="dot", line_color="gray", annotation_text="z=-2")
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")


# ── Weekly Reports ────────────────────────────────────────────────────────────

def render_weekly_reports(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 📅 Weekly Reports — {user_id}")

    weekly = user_data.get("weekly_reports", [])
    if not weekly:
        st.info("No weekly reports available.")
        return

    week_tabs = st.tabs([f"Week {r.week}" for r in weekly])
    for tab, report in zip(week_tabs, weekly):
        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📝 Weekly Narrative")
                st.info(report.narrative)

                if report.pattern_names:
                    st.markdown("**Detected patterns:**")
                    for pn in report.pattern_names:
                        st.markdown(f"- {pn}")

            with col2:
                st.markdown("#### 📊 Metrics")
                for k, v in report.metrics_summary.items():
                    st.metric(k.replace("_", " ").title(), f"{v:.2f}")
                if report.anomaly_count > 0:
                    st.warning(f"⚠ {report.anomaly_count} anomalies this week")
                else:
                    st.success("✓ No anomalies this week")


# ── AI Insights ───────────────────────────────────────────────────────────────

def render_ai_insights(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 💡 AI Insights — {user_id}")

    insights = user_data.get("insights", [])
    groq_on = user_data.get("groq_available", False)

    if not groq_on:
        st.warning("⚠ Groq API not configured. Showing raw analytical descriptions. Set GROQ_API_KEY to enable AI narration.")

    # Filter controls
    insight_type = st.multiselect(
        "Filter by type", ["pattern", "anomaly", "correlation"],
        default=["pattern", "anomaly", "correlation"]
    )
    show_abstained = st.checkbox("Show abstained (insufficient evidence) insights", value=False)

    filtered_insights = [
        i for i in insights
        if i.insight_type in insight_type
        and (show_abstained or not i.abstained)
    ]

    if not filtered_insights:
        st.info("No insights match your filters.")
        return

    st.markdown(f"**{len(filtered_insights)} insights**")
    st.divider()

    for ins in filtered_insights:
        tier_badge = f'<span class="badge-{ins.confidence_tier}">{ins.confidence_tier.upper()}</span>'
        type_icon = {"pattern": "📈", "anomaly": "🚨", "correlation": "🔗"}.get(ins.insight_type, "💡")

        with st.expander(
            f"{type_icon} {ins.title}",
            expanded=(ins.confidence_tier == "high" and not ins.abstained),
        ):
            st.markdown(tier_badge + f" confidence: `{ins.confidence:.2f}`", unsafe_allow_html=True)

            if ins.abstained:
                st.error(f"**Abstained:** {ins.abstain_reason}")
                st.caption(ins.raw_description)
            else:
                text = ins.narration if ins.narration else ins.raw_description
                st.markdown(text)
                if ins.evidence:
                    st.markdown("**Supporting Evidence:**")
                    for ev in ins.evidence:
                        st.caption(f"• {ev.description}")


# ── Persona Summary ───────────────────────────────────────────────────────────

def render_persona(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 🪪 Persona Summary — {user_id}")

    persona = user_data.get("persona")
    if not persona:
        st.info("No persona data available.")
        return

    emoji_map = {
        "Consistent Performer": "🏆",
        "Burnout Risk": "🔥",
        "Recovery Phase": "🌱",
        "Highly Structured User": "⚙️",
        "Digitally Fatigued": "📱",
        "Developing Pattern": "🔍",
    }
    icon = emoji_map.get(persona.label, "👤")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
<div class="persona-card">
  <h2 style="margin:0 0 0.5rem 0;">{icon} {persona.label}</h2>
  <p style="margin:0;opacity:0.9;line-height:1.6;">{persona.groq_wording or persona.description}</p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Supporting Metrics")
        metrics = persona.supporting_metrics
        if metrics:
            metric_df = pd.DataFrame(
                list(metrics.items()), columns=["Feature", "Score"]
            )
            metric_df["Feature"] = metric_df["Feature"].str.replace("_", " ").str.title()
            fig = px.bar(
                metric_df, x="Score", y="Feature", orientation="h",
                color="Score", color_continuous_scale="purples",
                range_x=[0, 1],
            )
            fig.update_layout(
                height=320, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width="stretch")


# ── Correlation Analysis ──────────────────────────────────────────────────────

def render_correlations(user_data: dict, user_id: str) -> None:
    st.markdown(f"## 🔗 Correlation Analysis — {user_id}")

    correlations = user_data.get("correlations", [])
    edf: pd.DataFrame = user_data["enriched_df"]

    col1, col2 = st.columns([1, 1])

    with col1:
        # Correlation heatmap
        numeric_cols = ["steps", "sleep_hours", "screen_time_hours",
                        "deep_work_hours", "exercise_minutes"]
        available = [c for c in numeric_cols if c in edf.columns]
        corr_matrix = edf[available].corr()
        corr_labels = [c.replace("_", " ").title() for c in available]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_labels, y=corr_labels,
            colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            title="Signal Correlation Matrix",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Significant Correlations")
        if correlations:
            for c in correlations:
                direction_icon = "↗️" if c.direction == "positive" else "↘️"
                sig_color = "#10b981" if c.direction == "positive" else "#ef4444"
                st.markdown(f"""
<div class="metric-card">
  <strong>{direction_icon} {c.metric_a.replace('_',' ').title()} ↔ {c.metric_b.replace('_',' ').title()}</strong><br>
  <span style="color:{sig_color};font-size:0.9rem;">r = {c.r_value:.3f}</span><br>
  <small style="opacity:0.75;">{c.interpretation}</small>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No significant correlations found (|r| < 0.40).")

    # Scatter pair explorer
    st.markdown("#### Scatter Explorer")
    sc1, sc2 = st.columns(2)
    signal_options = [c for c in numeric_cols if c in edf.columns]
    with sc1:
        xa = st.selectbox("X axis", signal_options, index=2)  # screen_time
    with sc2:
        ya = st.selectbox("Y axis", signal_options, index=1)  # sleep_hours

    if xa and ya and xa != ya:
        scatter_df = edf[[xa, ya]].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=scatter_df[xa], y=scatter_df[ya],
            mode="markers",
            marker=dict(color="#6366f1", size=7, opacity=0.75),
            name="observations",
        ))
        # Manual numpy trendline — no statsmodels required
        if len(scatter_df) >= 3:
            coeffs = np.polyfit(scatter_df[xa].values, scatter_df[ya].values, 1)
            x_line = np.linspace(scatter_df[xa].min(), scatter_df[xa].max(), 100)
            y_line = np.polyval(coeffs, x_line)
            fig2.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                name="trend",
            ))
        fig2.update_layout(
            title=f"{xa.replace('_',' ').title()} vs {ya.replace('_',' ').title()}",
            xaxis_title=xa.replace("_", " ").title(),
            yaxis_title=ya.replace("_", " ").title(),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, width="stretch")


# ── Query Interface ───────────────────────────────────────────────────────────

def render_query_interface(user_data: dict, user_id: str, all_results: dict) -> None:
    st.markdown(f"## 💬 Conversational Query Interface — {user_id}")

    groq_on = user_data.get("groq_available", False)
    if not groq_on:
        st.warning("⚠ Groq API not configured. Set GROQ_API_KEY to enable conversational queries.")

    st.markdown("Ask questions about the behavioral analysis:")
    st.caption("Examples: *Why was this anomaly flagged?* | *What changed this week?* | *Why did productivity decline?*")

    query = st.text_input("Your question", placeholder="e.g. Why did deep work decline this week?")

    if st.button("Ask", disabled=(not groq_on or not query)):
        if query:
            with st.spinner("Generating response..."):
                _load_dotenv()
                groq = GroqClient()
                generator = InsightGenerator(groq_client=groq)
                answer = generator.answer_query(query, user_data, user_id)
            st.markdown("#### Response")
            st.info(answer)

    # Quick question shortcuts
    st.markdown("#### Quick Questions")
    q_cols = st.columns(3)
    quick_qs = [
        "What is the main behavioral pattern detected?",
        "Were any anomalies flagged this week?",
        "What is the relationship between sleep and productivity?",
    ]
    for col, qq in zip(q_cols, quick_qs):
        with col:
            if st.button(qq, disabled=not groq_on, use_container_width=True):
                with st.spinner("Generating response..."):
                    _load_dotenv()
                    groq = GroqClient()
                    generator = InsightGenerator(groq_client=groq)
                    answer = generator.answer_query(qq, user_data, user_id)
                st.info(answer)


# ── Download section ──────────────────────────────────────────────────────────

def render_export(user_data: dict, user_id: str, all_results: dict) -> None:
    st.sidebar.divider()
    st.sidebar.markdown("#### 📥 Export")

    # JSON download
    export_data = {
        "persona": user_data["persona"].label,
        "feature_summary": user_data["feature_summary"],
        "anomalies": [
            {"date": a.date, "metric": a.metric, "severity": a.severity, "reason": a.reason}
            for a in user_data.get("anomalies", [])
        ],
        "patterns": [
            {"name": p.name, "direction": p.direction, "r_squared": p.r_squared}
            for p in user_data.get("patterns", [])
        ],
        "insights": [
            {"title": i.title, "confidence": i.confidence, "description": i.narration or i.raw_description}
            for i in user_data.get("insights", []) if not i.abstained
        ],
    }
    json_str = json.dumps(export_data, indent=2, default=str)
    st.sidebar.download_button(
        label="⬇ Download Insights JSON",
        data=json_str,
        file_name=f"{user_id}_insights.json",
        mime="application/json",
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    with st.spinner("Loading behavioral analysis..."):
        all_results = load_pipeline_results()

    user_id, section = render_sidebar(all_results)
    user_data = get_user_data(all_results, user_id)
    render_export(user_data, user_id, all_results)

    if section == "📊 Overview":
        render_overview(user_data, user_id)
    elif section == "🎯 Behavioral Scores":
        render_behavioral_scores(user_data, user_id)
    elif section == "📈 Trend Charts":
        render_trend_charts(user_data, user_id)
    elif section == "🚨 Anomaly Timeline":
        render_anomaly_timeline(user_data, user_id)
    elif section == "📅 Weekly Reports":
        render_weekly_reports(user_data, user_id)
    elif section == "💡 AI Insights":
        render_ai_insights(user_data, user_id)
    elif section == "🪪 Persona Summary":
        render_persona(user_data, user_id)
    elif section == "🔗 Correlation Analysis":
        render_correlations(user_data, user_id)
    elif section == "💬 Query Interface":
        render_query_interface(user_data, user_id, all_results)


if __name__ == "__main__":
    main()