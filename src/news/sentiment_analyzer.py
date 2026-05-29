"""Sentiment analysis for financial news articles.

Uses FinBERT (ProsusAI/finbert) for financial text sentiment analysis.
Provides a fallback rule-based scoring if the model is unavailable.

Validates: Requirements 23.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from src.config.constants import (
    NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS,
    SOURCE_CREDIBILITY_SOCIAL,
    SOURCE_CREDIBILITY_TIER1,
    SOURCE_CREDIBILITY_TIER2,
)

logger = logging.getLogger(__name__)


class ImpactLevel(str, Enum):
    """News impact classification levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SentimentAnalyzer:
    """NLP-based sentiment analyzer for financial news using FinBERT.

    Analyzes article text to produce a sentiment score in [-1.0, +1.0]
    and classifies impact level based on source credibility, corroboration,
    and sentiment magnitude.

    The model is loaded lazily on first use. If the model cannot be loaded
    (e.g., missing dependencies, no GPU, download failure), a rule-based
    fallback is used instead.

    The analysis must complete within NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS (5s).
    """

    # FinBERT model identifier on HuggingFace
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._model_loaded: bool = False
        self._model_available: bool = False
        self._pipeline: Any = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the FinBERT sentiment model on first use.

        If loading fails for any reason (missing packages, network issues,
        insufficient memory), falls back to rule-based scoring.
        """
        if self._model_loaded:
            return

        self._model_loaded = True
        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                truncation=True,
                max_length=512,
            )
            self._model_available = True
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self._model_available = False
            logger.warning(
                "FinBERT model unavailable, using rule-based fallback: %s", str(e)
            )

    def analyze(self, text: str) -> float:
        """Analyze text sentiment using FinBERT NLP model.

        Uses ProsusAI/finbert for financial-domain sentiment analysis.
        Falls back to keyword-based heuristic if model is unavailable.
        Must complete within 5 seconds.

        Args:
            text: The article text (headline + body) to analyze.

        Returns:
            Sentiment score in range [-1.0, +1.0] where:
            - -1.0 = extremely negative/bearish
            -  0.0 = neutral
            - +1.0 = extremely positive/bullish
        """
        if not text or not text.strip():
            return 0.0

        self._ensure_model_loaded()

        start_time = time.monotonic()

        if self._model_available and self._pipeline is not None:
            score = self._finbert_score(text)
        else:
            score = self._rule_based_sentiment(text)

        elapsed = time.monotonic() - start_time
        if elapsed > NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS:
            logger.warning(
                "Sentiment analysis took %.2fs (exceeds %ds timeout)",
                elapsed,
                NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS,
            )

        # Clamp to valid range
        return max(-1.0, min(1.0, score))

    async def analyze_async(self, text: str) -> float:
        """Async wrapper for sentiment analysis with timeout enforcement.

        Args:
            text: The article text to analyze.

        Returns:
            Sentiment score in range [-1.0, +1.0].

        Raises:
            asyncio.TimeoutError: If analysis exceeds the timeout.
        """
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self.analyze, text),
            timeout=NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS,
        )

    def classify_impact(
        self,
        sentiment_score: float,
        source_tier: float,
        corroboration: int,
    ) -> str:
        """Classify the impact level of a news article.

        Impact classification is based on:
        - Source credibility weight (tier-1: 1.0, tier-2: 0.7, social: 0.4)
        - Corroboration count (how many other sources reported similar news
          within a 5-minute window)
        - Sentiment magnitude (absolute value of sentiment score)

        Rules:
        - HIGH: Tier-1 source (weight >= 1.0) with |sentiment| > 0.7, OR
          any source with corroboration >= 2 and |sentiment| > 0.5
        - MEDIUM: Tier-2 source (weight >= 0.7) with |sentiment| > 0.5, OR
          tier-1 source with moderate sentiment (0.3 < |sentiment| <= 0.7)
        - LOW: Everything else (social media without corroboration, low
          sentiment magnitude)

        Args:
            sentiment_score: Sentiment score in [-1.0, +1.0].
            source_tier: Source credibility weight
                (tier-1: 1.0, tier-2: 0.7, social: 0.4).
            corroboration: Number of corroborating articles within
                a 5-minute window.

        Returns:
            Impact level string: "HIGH", "MEDIUM", or "LOW".
        """
        magnitude = abs(sentiment_score)

        # HIGH impact conditions:
        # 1. Tier-1 source with strong sentiment (|sentiment| > 0.7)
        if source_tier >= SOURCE_CREDIBILITY_TIER1 and magnitude > 0.7:
            return ImpactLevel.HIGH.value
        # 2. Any source with corroboration >= 2 and |sentiment| > 0.5
        if corroboration >= 2 and magnitude > 0.5:
            return ImpactLevel.HIGH.value

        # MEDIUM impact conditions:
        # 1. Tier-2 source (weight >= 0.7) with |sentiment| > 0.5
        if source_tier >= SOURCE_CREDIBILITY_TIER2 and magnitude > 0.5:
            return ImpactLevel.MEDIUM.value
        # 2. Tier-1 source with moderate sentiment (0.3 < |sentiment| <= 0.7)
        if source_tier >= SOURCE_CREDIBILITY_TIER1 and magnitude > 0.3:
            return ImpactLevel.MEDIUM.value

        # LOW impact — everything else
        return ImpactLevel.LOW.value

    def _finbert_score(self, text: str) -> float:
        """Score text using the FinBERT transformer model.

        FinBERT outputs labels: 'positive', 'negative', 'neutral'
        with associated confidence scores. We convert to [-1.0, +1.0].

        Args:
            text: The text to analyze.

        Returns:
            Sentiment score in [-1.0, +1.0].
        """
        try:
            result = self._pipeline(text[:512])  # Truncate to model max length
            if not result:
                return 0.0

            # Pipeline returns list of dicts: [{'label': ..., 'score': ...}]
            prediction = result[0]
            label = prediction["label"].lower()
            confidence = prediction["score"]

            if label == "positive":
                return confidence
            elif label == "negative":
                return -confidence
            else:  # neutral
                return 0.0
        except Exception as e:
            logger.warning("FinBERT inference failed, using fallback: %s", str(e))
            return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> float:
        """Fallback rule-based sentiment scoring using financial keywords.

        Used when FinBERT model is unavailable. Provides a reasonable
        approximation based on domain-specific keyword matching with
        weighted scoring.

        Args:
            text: The text to analyze.

        Returns:
            Sentiment score in [-1.0, +1.0].
        """
        text_lower = text.lower()

        # Weighted negative keywords (word, weight)
        negative_keywords = [
            ("crash", 1.0),
            ("crisis", 0.9),
            ("collapse", 0.9),
            ("bankruptcy", 0.9),
            ("default", 0.8),
            ("recession", 0.8),
            ("war", 0.8),
            ("sanctions", 0.7),
            ("downgrade", 0.7),
            ("loss", 0.5),
            ("plunge", 0.8),
            ("sell-off", 0.7),
            ("bearish", 0.6),
            ("decline", 0.5),
            ("layoffs", 0.6),
            ("inflation", 0.4),
            ("deficit", 0.4),
            ("fraud", 0.8),
            ("investigation", 0.4),
            ("warning", 0.4),
        ]

        # Weighted positive keywords (word, weight)
        positive_keywords = [
            ("surge", 0.8),
            ("rally", 0.7),
            ("growth", 0.6),
            ("profit", 0.6),
            ("upgrade", 0.7),
            ("bullish", 0.6),
            ("recovery", 0.6),
            ("expansion", 0.5),
            ("dividend", 0.5),
            ("breakthrough", 0.7),
            ("gain", 0.5),
            ("record high", 0.8),
            ("optimistic", 0.5),
            ("strong earnings", 0.7),
            ("beat expectations", 0.7),
            ("outperform", 0.6),
            ("acquisition", 0.4),
            ("innovation", 0.4),
        ]

        neg_score = sum(
            weight for keyword, weight in negative_keywords if keyword in text_lower
        )
        pos_score = sum(
            weight for keyword, weight in positive_keywords if keyword in text_lower
        )

        total = neg_score + pos_score
        if total == 0:
            return 0.0

        # Normalize to [-1, 1] range
        score = (pos_score - neg_score) / total
        return score
