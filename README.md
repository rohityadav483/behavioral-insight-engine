# 🧠 Behavioral Insight Engine

> **Chronis AI/ML Engineer Hiring Assessment — Task A**
>
> A production-quality behavioral analytics system that transforms raw daily observations into explainable, evidence-backed insights — with deterministic analysis, confidence scoring, and LLM-powered narration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Groq API Setup](#groq-api-setup)
- [Features](#features)
- [Methodology](#methodology)
- [Explainability Philosophy](#explainability-philosophy)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

The Behavioral Insight Engine analyzes daily behavioral data (steps, sleep, screen time, deep work, exercise) across multiple users and produces:

- **Detected patterns** — trends with direction, slope, and R² strength
- **Anomaly detection** — z-score flagged deviations with severity and evidence
- **Confidence scores** — every insight rated 0.0 → 1.0 with Low / Medium / High tier
- **Abstention** — system refuses to make claims when evidence is insufficient
- **Behavioral personas** — deterministic profile assignment per user
- **Multi-signal correlations** — hedged-language association findings
- **Weekly reports** — per-week behavioral health summaries
- **Groq AI narration** — LLM polishes analytical outputs without inventing data
- **Conversational queries** — ask natural language questions about any insight

---

## Architecture

```
CSV Data
    │
    ▼
┌─────────────────────┐
│   DataLoader        │  Schema validation, date parsing, user grouping
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│   Preprocessor      │  Rolling stats, baselines, z-scores, week tags
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│  FeatureEngineer    │  8 derived behavioral features per user per day
└─────────┬───────────┘
          │
    ┌─────┴──────┬──────────────┬─────────────────┐
    ▼            ▼              ▼                  ▼
PatternDetector  AnomalyDetector  CorrelationEngine  PersonaEngine
(OLS, rolling)   (z-score +       (Pearson r,        (rule-based,
                  IForest)         hedged language)    deterministic)
    └─────┬──────┴──────────────┴─────────────────┘
          │
    ▼
┌─────────────────────┐
│  ConfidenceEngine   │  0.0–1.0 score per insight (R², evidence volume, consistency)
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│  EvidenceValidator  │  Abstention gate — blocks weak/insufficient claims
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│   GroqClient        │  Evidence-constrained narration (never invents data)
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│  InsightGenerator   │  Assembles validated Insight objects
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│  ReportGenerator    │  Weekly summaries + JSON export
└─────────┬───────────┘
          │
    ▼
┌─────────────────────┐
│  Streamlit Dashboard│  9-section interactive UI
└─────────────────────┘
```

---

## Project Structure

```
behavioral-insight-engine/
│
├── data/
│   └── behavioral_data.csv              # Input dataset (150 rows, 5 users)
│
├── src/
│   ├── data_loader.py                   # CSV load, schema validation, date parsing
│   ├── preprocessing.py                 # Rolling stats, baselines, z-scores
│   ├── feature_engineering.py           # 8 derived behavioral features
│   ├── pattern_detector.py              # OLS trends, burnout/recovery patterns
│   ├── anomaly_detector.py              # Z-score + Isolation Forest
│   ├── confidence_engine.py             # 0.0–1.0 confidence scoring
│   ├── evidence_validator.py            # Abstention gates
│   ├── persona_engine.py                # Deterministic persona assignment
│   ├── correlation_engine.py            # Pearson r, hedged interpretation
│   ├── insight_generator.py             # Pipeline orchestrator
│   ├── groq_client.py                   # Safe Groq API wrapper
│   ├── report_generator.py              # Weekly reports + JSON export
│   └── utils.py                         # Shared types, constants, helpers
│
├── dashboard/
│   └── app.py                           # Full Streamlit dashboard
│
├── prompts/
│   ├── insight_prompt.txt               # Evidence-constrained narration prompt
│   ├── persona_prompt.txt               # Persona wording prompt
│   └── report_prompt.txt                # Weekly report prompt
│
├── outputs/                             # Auto-created — JSON exports, cache
│
├── tests/
│   ├── test_patterns.py                 # Pattern detection tests
│   ├── test_anomalies.py                # Anomaly detection tests
│   ├── test_confidence.py               # Confidence scoring tests
│   └── test_validator.py                # Evidence sufficiency tests
│
├── requirements.txt
├── README.md
├── decisions.md                         # Methodology and design rationale
├── .env.example
├── main.py                              # Unified entry point
├── Makefile
└── run.sh
```

---

## Quick Start

### 1. Clone

```bash
git clone <repo-url>
cd behavioral-insight-engine
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key (optional)

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

> The system runs fully without a Groq key — AI narration is disabled and raw analytical descriptions are shown instead. All pattern detection, anomaly detection, and confidence scoring work without any API key.

### 5. Run

```bash
streamlit run dashboard/app.py
```

Dashboard opens at **http://localhost:8501**

---

## Groq API Setup

1. Create a free account at [console.groq.com](https://console.groq.com)
2. Generate an API key
3. Add to `.env`:

```bash
GROQ_API_KEY=gsk_your_key_here
```

Model used: `llama-3.3-70b-versatile`

**What Groq is used for:**
- Narrating pre-computed insights in natural language
- Polishing behavioral persona descriptions
- Generating weekly report narratives
- Answering conversational queries about insights

**What Groq is never used for:**
- Computing statistics or metrics
- Detecting patterns or anomalies
- Generating confidence scores
- Making claims not backed by evidence

---

## Features

### Pattern Discovery

Detects meaningful behavioral trends using:
- OLS linear regression (slope + R² per metric)
- Rolling 7-day window averages
- Week-over-week percentage comparisons
- Variance analysis across time

Detected pattern types:
- Increasing / decreasing trends per metric
- Burnout-like patterns (declining deep work + rising screen time + poor sleep)
- Recovery phases (improving sleep + exercise after prior low period)
- Behavioral stabilization (low cross-signal variance)

### Anomaly Detection

Two-stage detection:
1. **Z-score** (primary) — flags |z| > 2.0 vs user's own rolling baseline
2. **Isolation Forest** (secondary) — multi-signal corroboration, contamination=0.10

Every anomaly includes:
- Date, metric, observed value
- Personal baseline at time of anomaly
- Z-score and deviation magnitude
- Reason and evidence strings
- Severity: Low / Medium / High
- IForest corroboration flag

### Confidence Scoring

```
confidence = 0.40 × trend_R²
           + 0.35 × min(evidence_days / 14, 1.0)
           + 0.25 × (1 - coefficient_of_variation)
```

Tiers: **High** (≥0.75) · **Medium** (0.50–0.74) · **Low** (<0.50)

### Evidence Sufficiency / Abstention

System abstains (outputs: *"Insufficient evidence for reliable behavioral interpretation."*) when:
- Confidence < 0.40
- Fewer than 5 data points
- Pattern R² < 0.05
- Anomaly |z| < 2.0
- Correlation |r| < 0.40

### Engineered Features

| Feature | Description |
|---|---|
| `sleep_consistency` | Inverted rolling std of sleep — high = consistent schedule |
| `activity_consistency` | Inverted rolling std of steps |
| `productivity_score` | 60% deep work + 40% steps, normalized 0–1 |
| `behavioral_volatility` | Mean absolute z-score across all signals |
| `exercise_adherence` | Fraction of rolling-window days with ≥30 min exercise |
| `screen_time_trend` | OLS slope of screen time over 7 days |
| `recovery_score` | 55% sleep quality + 45% exercise regularity |
| `behavioral_stability_score` | Inverse of mean rolling coefficient of variation |

### Behavioral Personas

Assigned deterministically from feature thresholds:

| Persona | Key indicators |
|---|---|
| Consistent Performer | High stability + high productivity + low volatility |
| Burnout Risk | Declining productivity + rising screen time + poor sleep |
| Recovery Phase | Improving sleep + exercise after prior low period |
| Highly Structured User | Low variance across sleep, steps, exercise |
| Digitally Fatigued | Screen time trend ↑ + recovery score ↓ + productivity ↓ |

### Multi-Signal Correlations

Computes Pearson r for behavioral hypothesis pairs. Language is always hedged:

> *"Patterns suggest screen time appears associated with lower sleep duration. (moderate association, r=−0.61)"*

Never claims causality.

---

## Methodology

### Why statistical methods over machine learning

The dataset contains 150 rows across 5 users (30 days per user). At this scale:

- Deep learning models overfit immediately — no meaningful validation split exists
- Trained models produce black-box outputs incompatible with the explainability requirement
- Statistical methods (OLS, z-scores, rolling windows) work correctly at small n and produce interpretable, auditable evidence chains

See `decisions.md` for full methodology rationale.

### Baseline computation

Uses expanding mean (all prior days for that user) rather than global or fixed baseline. This ensures the baseline updates as a user's behavior evolves — a gradual long-term improvement is not flagged as an anomaly.

### Z-score vs personal baseline

Each user's anomalies are computed relative to their own rolling history, not population averages. A step count of 3,500 may be normal for one user and severely anomalous for another.

---

## Explainability Philosophy

Every output in this system satisfies three properties:

**1. Evidence-backed** — no claim is made without a supporting data point. Insight objects carry `evidence` arrays with metric, value, baseline, and delta.

**2. Confidence-gated** — every claim has a numeric confidence score. Claims below the abstention threshold (0.40) are blocked entirely.

**3. Causality-free** — the system never asserts that A causes B. Correlation findings use hedged language: *"patterns suggest"*, *"appears associated with"*, *"may indicate"*.

The Groq LLM is constrained to narrate pre-computed evidence. It receives structured metrics and explicit instructions not to add claims. This prevents hallucination at the architectural level, not just at the prompt level.

---

## Dashboard

Nine sections accessible via sidebar navigation:

| Section | Contents |
|---|---|
| 📊 Overview | Summary cards, signal sparklines, persona badge, anomaly alert |
| 🎯 Behavioral Scores | Radar chart of 8 engineered features with bar breakdown |
| 📈 Trend Charts | Per-metric chart with rolling avg, baseline, anomaly markers |
| 🚨 Anomaly Timeline | Filterable list with expandable evidence panels + scatter plot |
| 📅 Weekly Reports | Per-week narrative, metrics summary, anomaly count |
| 💡 AI Insights | All insights with confidence badges, evidence expanders |
| 🪪 Persona Summary | Persona card with supporting metrics bar chart |
| 🔗 Correlation Analysis | Heatmap + significant findings + scatter pair explorer |
| 💬 Query Interface | Free-text question box + quick question shortcuts |

User selector and JSON download in sidebar. Refresh button clears cache and reruns pipeline.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run individual test modules
pytest tests/test_patterns.py -v
pytest tests/test_anomalies.py -v
pytest tests/test_confidence.py -v
pytest tests/test_validator.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Test coverage:
- **Pattern detection** — increasing/decreasing trends, burnout, recovery, stabilization, insufficient data, required fields
- **Anomaly detection** — sudden drops, spikes, stable signals, severity scaling, required fields, IForest flags
- **Confidence scoring** — range validation, tier classification, z-score scaling, IForest bonus, composite penalty
- **Evidence validation** — all abstention conditions, boundary cases, abstain message format

---

## Configuration

All thresholds configurable in `src/utils.py`:

```python
ZSCORE_THRESHOLD = 2.0           # anomaly flag threshold
CONFIDENCE_ABSTAIN_FLOOR = 0.40  # minimum confidence to report insight
ROLLING_WINDOW = 7               # days for rolling statistics
MIN_DAYS_FOR_INSIGHT = 5         # minimum data points required
ISOLATION_CONTAMINATION = 0.10   # IForest expected anomaly fraction
```

---

## Sample Output

```
Insight: Physical activity declined over the past two weeks.
Confidence: 0.81 (High)
Evidence:
  • 7-day average changed from 9,240 to 5,890 steps
  • Slope: -112.4 steps/day, R²: 0.74
  • Week-over-week delta: -36.2%

Anomaly: 2026-01-22 | steps | z = -3.14
Severity: high
Reason: Unusually low daily steps: 2,180 vs personal baseline of 8,340 (74% below average).
Evidence: z-score = -3.14 (threshold: ±2.0). Deviation: -6,160 steps (-73.9%). IForest corroborated.

Persona: Burnout Risk
Confidence: 0.65 (Medium)
Supporting: productivity_score=0.38, screen_time_trend=+0.12, sleep_consistency=0.51
```

---

## Limitations

- **Dataset size** — 30 days per user is sufficient for trend detection but limits statistical power for correlation analysis. Findings at n=30 should be treated as indicative, not conclusive.
- **No ground truth** — synthetic data means no external validation of insight accuracy is possible.
- **Single-user correlations** — correlation analysis runs per-user; cross-user patterns are not computed.
- **Forward-fill for missing values** — gaps of 1–2 days are filled; larger gaps would require different handling.
- **IForest at small n** — Isolation Forest is unreliable below ~20 samples; used as corroboration only.
- **Groq latency** — API calls add ~1–3s per insight when narration is enabled. Pipeline caches results after first run.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| Statistical analysis | scipy, scikit-learn |
| Visualization | plotly, streamlit |
| AI narration | Groq API (`llama-3.3-70b-versatile`) |
| Testing | pytest |
| Entry point | Python 3.10+ |

---
