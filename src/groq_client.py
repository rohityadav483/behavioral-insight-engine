"""
groq_client.py
Groq API client with:
  - evidence-constrained prompting (LLM never invents data)
  - retry logic with exponential backoff
  - graceful degradation when API key absent
  - deterministic evidence passed in every prompt
"""

from __future__ import annotations

import os
import time
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 512
TEMPERATURE = 0.3    # low for factual narration
MAX_RETRIES = 3
RETRY_DELAY = 2.0    # seconds


class GroqClient:
    """
    Safe wrapper around Groq API.
    All calls are evidence-gated: structured metrics must be provided,
    the LLM is instructed only to narrate — not invent.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None
        self._available = False
        self._init_client()

    @property
    def available(self) -> bool:
        return self._available

    # ── Public ────────────────────────────────────────────────────────────────

    def narrate_insight(
        self,
        insight_title: str,
        raw_description: str,
        evidence_bullets: list[str],
        confidence: float,
        user_id: str,
    ) -> str:
        """
        Ask Groq to narrate a pre-computed insight.
        The LLM receives structured evidence and MUST NOT add claims beyond it.
        Returns polished natural language string, or falls back to raw_description.
        """
        if not self._available:
            return raw_description

        prompt = self._build_insight_prompt(
            insight_title, raw_description, evidence_bullets, confidence
        )
        return self._call(prompt) or raw_description

    def narrate_persona(
        self,
        persona_label: str,
        persona_description: str,
        supporting_metrics: dict[str, float],
        user_id: str,
    ) -> str:
        """
        Ask Groq for polished persona wording.
        Returns narration or raw description on failure.
        """
        if not self._available:
            return persona_description

        metrics_text = "\n".join(
            f"  - {k.replace('_', ' ')}: {v:.3f}" for k, v in supporting_metrics.items()
        )
        prompt = self._build_persona_prompt(persona_label, persona_description, metrics_text)
        return self._call(prompt) or persona_description

    def narrate_weekly_report(
        self,
        user_id: str,
        week_number: int,
        metrics_summary: dict[str, float],
        anomaly_count: int,
        top_patterns: list[str],
    ) -> str:
        """
        Ask Groq for a weekly behavioral health narrative.
        Returns narrative or fallback plain text.
        """
        if not self._available:
            return self._fallback_weekly(user_id, week_number, metrics_summary)

        prompt = self._build_report_prompt(
            user_id, week_number, metrics_summary, anomaly_count, top_patterns
        )
        return self._call(prompt) or self._fallback_weekly(user_id, week_number, metrics_summary)

    def answer_query(
        self,
        query: str,
        context_json: str,
        user_id: str,
    ) -> str:
        """
        Answer a conversational question using structured context.
        LLM may only reference data provided in context_json.
        """
        if not self._available:
            return "Groq API unavailable. Please set GROQ_API_KEY in your .env file."

        prompt = self._build_query_prompt(query, context_json, user_id)
        return self._call(prompt) or "Unable to generate a response at this time."

    # ── Prompt Builders ───────────────────────────────────────────────────────

    @staticmethod
    def _build_insight_prompt(
        title: str,
        raw: str,
        evidence: list[str],
        confidence: float,
    ) -> str:
        evidence_text = "\n".join(f"  • {e}" for e in evidence)
        return f"""You are a behavioral analysis assistant. Your task is to write a clear, \
professional, human-readable explanation of a behavioral insight.

STRICT RULES:
1. Only use the evidence provided below. Do NOT add claims, statistics, or interpretations \
not present in the evidence.
2. Never assert causality. Use hedged language: "appears associated with", "patterns suggest", \
"may indicate".
3. Keep the response to 2-3 sentences maximum.
4. Confidence score is {confidence:.2f}. If low (<0.5), acknowledge limited certainty.

INSIGHT TITLE: {title}

RAW FINDING: {raw}

EVIDENCE:
{evidence_text}

Write the narration now:"""

    @staticmethod
    def _build_persona_prompt(label: str, description: str, metrics_text: str) -> str:
        return f"""You are a behavioral analyst. Write a concise, empathetic 2-sentence summary \
for a behavioral persona.

STRICT RULES:
1. Only reference the metrics provided. Do not invent traits.
2. Avoid clinical or judgmental language.
3. Write in second person ("This user shows...").

PERSONA: {label}
BASE DESCRIPTION: {description}

SUPPORTING METRICS:
{metrics_text}

Write the persona summary now:"""

    @staticmethod
    def _build_report_prompt(
        user_id: str,
        week: int,
        metrics: dict[str, float],
        anomaly_count: int,
        patterns: list[str],
    ) -> str:
        metrics_text = "\n".join(f"  - {k}: {v:.3f}" for k, v in metrics.items())
        patterns_text = "\n".join(f"  - {p}" for p in patterns) if patterns else "  - No major patterns detected."
        return f"""You are writing a weekly behavioral health summary.

STRICT RULES:
1. Only reference the data below. No invented statistics.
2. 3-4 sentences maximum.
3. Hedged language: "patterns suggest", "appears to show".
4. If anomaly_count > 0, mention that unusual events were observed without fabricating details.

USER: {user_id} | WEEK: {week}

METRICS THIS WEEK:
{metrics_text}

PATTERNS DETECTED:
{patterns_text}

ANOMALY COUNT: {anomaly_count}

Write the weekly summary now:"""

    @staticmethod
    def _build_query_prompt(query: str, context_json: str, user_id: str) -> str:
        return f"""You are a behavioral data analyst. Answer the user's question using ONLY \
the structured context provided.

STRICT RULES:
1. Only use information in the context JSON below.
2. If the context does not contain enough information to answer, say so clearly.
3. Never invent data points, dates, or statistics.
4. Keep the answer to 3 sentences maximum.
5. Use hedged language where appropriate.

USER: {user_id}
QUESTION: {query}

CONTEXT:
{context_json}

Answer:"""

    # ── API Call ──────────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> Optional[str]:
        """Send prompt to Groq with retry logic. Returns text or None on failure."""
        if not self._available or self._client is None:
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                text = response.choices[0].message.content.strip()
                logger.debug(f"Groq response received ({len(text)} chars).")
                return text
            except Exception as e:
                logger.warning(f"Groq API attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        logger.error("All Groq API retries exhausted.")
        return None

    def _init_client(self) -> None:
        """Attempt to initialize Groq client. Gracefully degrade if unavailable."""
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. AI narration disabled — using raw descriptions.")
            self._available = False
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            self._available = True
            logger.info("Groq client initialized successfully.")
        except ImportError:
            logger.warning("groq package not installed. Run: pip install groq")
            self._available = False
        except Exception as e:
            logger.warning(f"Groq initialization failed: {e}")
            self._available = False

    # ── Fallbacks ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_weekly(
        user_id: str, week: int, metrics: dict[str, float]
    ) -> str:
        prod = metrics.get("productivity_score", 0.0)
        sleep = metrics.get("sleep_hours", 0.0)
        return (
            f"Week {week} summary for {user_id}: "
            f"average productivity score {prod:.2f}, "
            f"average sleep {sleep:.1f}h. "
            f"Full AI narration requires GROQ_API_KEY."
        )