# decisions.md — Behavioral Insight Engine

Design decisions, methodology rationale, assumptions, and failure modes.

---

## 1. Why Statistical Methods, Not Deep Learning

Dataset: 150 rows × 5 users × 30 days.

Deep learning requires thousands of samples minimum. Training a neural net on 30 rows per user produces overfitted noise, not insight. Statistical methods (linear regression, z-scores, rolling windows) work correctly at this scale, are fully interpretable, and produce explainable outputs — which is the core evaluation criterion.

Chosen methods and why:

| Method | Why chosen |
|---|---|
| OLS linear regression | Gives slope + R² → direction + strength in one shot |
| Z-score | Per-user deviation from personal baseline; intuitive unit |
| Rolling windows (7 days) | Smooths daily noise; standard in behavioral analytics |
| Isolation Forest | Multi-signal corroboration only; not primary signal |
| Pearson / Spearman correlation | Simple, interpretable, hedged-language safe |

---

## 2. Why No Model Training

Three reasons:

1. **Scale mismatch.** Any trained model (RF, XGBoost, LSTM) needs held-out validation sets. With 30 rows per user, no split produces a meaningful val set.
2. **Explainability requirement.** Trained models are black boxes. Every Chronis insight must carry evidence + confidence + reason. Z-scores and regression slopes do this naturally.
3. **Hallucination risk.** A trained model can output confident predictions with no interpretable evidence chain. This directly violates the abstention requirement.

---

## 3. Anomaly Detection — Threshold Decisions

**Z-score threshold: ±2.0**

- Industry standard for behavioral outlier detection.
- At |z| > 2.0: observation is 2+ standard deviations from user's own rolling baseline.
- Chosen over fixed absolute thresholds because each user has different baseline levels. A step count of 3,000 is normal for one user, anomalous for another.
- Threshold is configurable via `ZSCORE_THRESHOLD` in `utils.py`.

**Severity mapping:**

| |z-score| range | Severity |
|---|---|
| 2.0 – 2.5 | low |
| 2.5 – 3.5 | medium |
| ≥ 3.5 | high |

Rationale: mirrors standard deviation interpretation (2σ → notable, 3.5σ → extreme).

**Isolation Forest: contamination = 0.10**

Used as secondary corroboration only. Not primary flag. Why:
- IForest does not produce z-scores or interpretable deviations.
- Cannot produce evidence strings without additional work.
- At n=30, IForest is unstable; contamination=0.10 avoids over-flagging.
- When IForest agrees with z-score: confidence score gets +0.10 bonus.

**Baseline: expanding mean (not global mean)**

Uses `expanding().mean()` — average of all prior days for that user, growing daily. Rationale: captures behavioral evolution. A user who gradually increases steps should have their baseline update to reflect that, not flag the improvement as an anomaly.

---

## 4. Confidence Scoring Methodology

Formula:

```
confidence = 0.40 × trend_R²
           + 0.35 × min(evidence_days / 14, 1.0)
           + 0.25 × (1 - coefficient_of_variation)
```

Weight rationale:

- **Trend R² (0.40):** Strongest single signal of pattern reliability. Low R² = noisy trend = low confidence.
- **Evidence volume (0.35):** More supporting days = stronger claim. Saturates at 14 days to avoid over-rewarding long but flat signals.
- **Consistency (0.25):** Low CV metric = user behaves predictably = patterns are more trustworthy.

Composite patterns (burnout, recovery) receive 0.85× multiplier because they depend on 3+ signals aligning — more conditions = more failure points.

Anomaly confidence uses |z| / 4.0 (saturates at z=4) + 0.10 IForest bonus.

Tiers:
- High: ≥ 0.75
- Medium: 0.50 – 0.74
- Low: < 0.50

---

## 5. Abstention Logic

**Threshold: 0.40**

Any insight with confidence < 0.40 is blocked. System outputs:

> "Insufficient evidence for reliable behavioral interpretation."

Additional gates per insight type:

| Type | Additional gate |
|---|---|
| Pattern | R² < 0.05 → abstain (no directional signal) |
| Pattern | n_days < 5 → abstain (insufficient history) |
| Anomaly | \|z\| < 2.0 → abstain (below threshold) |
| Correlation | \|r\| < 0.40 → silently dropped |
| Correlation | n_samples < 5 → abstain |

Rationale: better to say nothing than make a wrong claim with false confidence. Abstention is not a failure state — it is correct behavior when evidence is weak.

---

## 6. Groq Integration — Hallucination Prevention Strategy

Core principle: **LLM narrates evidence; it does not discover it.**

Every Groq call receives:
1. A structured evidence block (pre-computed metrics, values, baselines)
2. A confidence score
3. An explicit instruction: "Only use the evidence provided. Do not add claims not in the evidence."

The LLM never sees raw data — it sees a constrained JSON-like summary. This prevents:
- Inventing statistics not in the data
- Extrapolating beyond observed patterns
- Claiming causality (prompts explicitly forbid causal language)

Temperature = 0.3: low randomness → consistent, factual tone.

Fallback: if Groq unavailable or API fails, system uses `raw_description` (deterministic string). Dashboard is fully functional without Groq.

---

## 7. Feature Engineering Decisions

**Sleep consistency = 1 / (1 + rolling_std)**

Inverted std gives high score for low variance. Rationale: consistency in sleep is independently valuable. A user sleeping exactly 7h every night scores higher than one sleeping 5h one night and 9h the next, even if the mean is identical.

**Productivity score = 0.60 × deep_work_norm + 0.40 × steps_norm**

Deep work weighted higher: it is a direct measure of cognitive output. Steps proxy for physical activity and overall engagement. Both normalized 0–1 before weighting.

**Behavioral volatility = mean(|z-scores| across all signals)**

High volatility = many signals deviating from personal baseline simultaneously. Useful for burnout detection and stability assessment. Computed from already-present z-score columns (no extra passes).

**Screen time trend = OLS slope over 7-day rolling window**

Not just level — direction matters. A user at 5h/day with rising slope is more concerning than one at 6h/day with flat or falling slope.

---

## 8. Persona Assignment Logic

Deterministic rules first — Groq wording second.

Why deterministic first:
- Personas must be reproducible. Same data → same persona every run.
- Auditable: evaluator can trace exactly which rule triggered which persona.
- Groq only polishes the language; it cannot change the assigned persona.

Full match → immediate assignment. Partial match (n-1 rules) → best partial match. No match → "Developing Pattern" fallback. This is the correct behavior for short behavioral histories.

---

## 9. Correlation Analysis — Hedged Language

System never claims causality. Exact wording constraints:

- Positive: *"Patterns suggest [A] may be associated with higher [B]."*
- Negative: *"Patterns suggest higher [A] appears associated with lower [B]."*

Why: Pearson r between 30 daily observations cannot establish causation. Confounders are plentiful (weekends, illness, travel). Hedged language is scientifically honest and protects against overclaiming.

Only |r| ≥ 0.40 reported. Below that, correlation is too weak to be actionable.

---

## 10. Failure Modes

| Failure mode | Risk | Mitigation |
|---|---|---|
| Z-score unstable early in series | High — rolling window needs min 3 points | `min_periods=3` on all rolling ops |
| IForest unstable at n=30 | Medium | Used as corroboration only; contamination=0.10 |
| Groq API down | Medium | Full fallback to raw_description; dashboard works without it |
| All users similar → no anomalies | Low | Per-user baselines ensure personal deviation is flagged |
| Weak trend misidentified | Medium | R² gate (0.05 min) + abstention at confidence < 0.40 |
| Persona mismatch for new users | Medium | "Developing Pattern" fallback; min_days=5 gate |
| Cache stale after data update | Low | `python main.py --refresh` clears cache |
| Forward-fill masking real gaps | Low | Logged as warning; acceptable for ≤1-2 missing days |

---

## 11. What Was Intentionally Left Out

- **TF-IDF / NLP on free text:** no text columns in dataset.
- **Time-series forecasting (ARIMA, Prophet):** 30 days insufficient; forecast horizon too short to be useful.
- **Clustering (k-means across users):** 5 users is not a clustering problem.
- **Causal inference (DoWhy, etc.):** requires experimental design; observational data only.
- **Deep learning (LSTM, Transformer):** see §1 and §2.

These exclusions are not limitations — they are correct engineering judgment for this dataset size and explainability requirement.

---

## 12. Reproducibility

- All random seeds fixed: `IsolationForest(random_state=42)`.
- No stochastic components in pattern detection or confidence scoring.
- Groq calls are the only non-deterministic step; all analytical logic produces identical output on identical input.
- Pipeline results cached to `outputs/pipeline_cache.pkl` after first run.