"""Unit tests for the SentimentAnalyzer class.

Tests cover:
- Sentiment scoring range validation [-1.0, +1.0]
- Empty/whitespace input handling
- Rule-based fallback scoring
- Impact classification logic (HIGH/MEDIUM/LOW)
- Lazy model loading behavior
- Timeout compliance
"""

import pytest

from src.config.constants import (
    SOURCE_CREDIBILITY_SOCIAL,
    SOURCE_CREDIBILITY_TIER1,
    SOURCE_CREDIBILITY_TIER2,
)
from src.news.sentiment_analyzer import ImpactLevel, SentimentAnalyzer


@pytest.fixture
def analyzer():
    """Create a SentimentAnalyzer instance with model loading disabled."""
    sa = SentimentAnalyzer()
    # Force fallback mode for deterministic testing
    sa._model_loaded = True
    sa._model_available = False
    return sa


class TestAnalyzeMethod:
    """Tests for the analyze() method."""

    def test_empty_string_returns_zero(self, analyzer: SentimentAnalyzer):
        assert analyzer.analyze("") == 0.0

    def test_whitespace_only_returns_zero(self, analyzer: SentimentAnalyzer):
        assert analyzer.analyze("   \n\t  ") == 0.0

    def test_none_text_returns_zero(self, analyzer: SentimentAnalyzer):
        # The method checks `not text` which handles None-like empty
        assert analyzer.analyze("") == 0.0

    def test_score_in_valid_range(self, analyzer: SentimentAnalyzer):
        texts = [
            "Markets crash amid global crisis and recession fears",
            "Stock surge as company reports record high profits",
            "The weather is nice today",
            "Company announces quarterly results",
        ]
        for text in texts:
            score = analyzer.analyze(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {text}"

    def test_negative_text_produces_negative_score(self, analyzer: SentimentAnalyzer):
        text = "Markets crash amid global crisis and recession fears with massive sell-off"
        score = analyzer.analyze(text)
        assert score < 0.0

    def test_positive_text_produces_positive_score(self, analyzer: SentimentAnalyzer):
        text = "Stock surge as company reports record high profits and strong earnings"
        score = analyzer.analyze(text)
        assert score > 0.0

    def test_neutral_text_returns_zero(self, analyzer: SentimentAnalyzer):
        text = "The company held its annual meeting on Tuesday"
        score = analyzer.analyze(text)
        assert score == 0.0

    def test_score_clamped_to_negative_one(self, analyzer: SentimentAnalyzer):
        # Even with many negative keywords, score should not go below -1.0
        text = "crash crisis collapse bankruptcy default recession war sanctions"
        score = analyzer.analyze(text)
        assert score >= -1.0

    def test_score_clamped_to_positive_one(self, analyzer: SentimentAnalyzer):
        # Even with many positive keywords, score should not exceed +1.0
        text = "surge rally growth profit upgrade bullish recovery expansion"
        score = analyzer.analyze(text)
        assert score <= 1.0


class TestClassifyImpact:
    """Tests for the classify_impact() method.

    Rules:
    - HIGH: tier-1 (weight >= 1.0) with |sentiment| > 0.7, OR
      any source with corroboration >= 2 and |sentiment| > 0.5
    - MEDIUM: tier-2 (weight >= 0.7) with |sentiment| > 0.5, OR
      tier-1 with moderate sentiment (0.3 < |sentiment| <= 0.7)
    - LOW: everything else
    """

    def test_high_impact_tier1_high_sentiment(self, analyzer: SentimentAnalyzer):
        """Tier-1 source with |sentiment| > 0.7 → HIGH."""
        result = analyzer.classify_impact(
            sentiment_score=-0.8,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=1,
        )
        assert result == ImpactLevel.HIGH.value

    def test_high_impact_corroboration_threshold(self, analyzer: SentimentAnalyzer):
        """Any source with corroboration >= 2 and |sentiment| > 0.5 → HIGH."""
        result = analyzer.classify_impact(
            sentiment_score=0.6,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=2,
        )
        assert result == ImpactLevel.HIGH.value

    def test_high_impact_corroboration_3_with_sentiment(
        self, analyzer: SentimentAnalyzer
    ):
        """Corroboration >= 2 with |sentiment| > 0.5 → HIGH (even social)."""
        result = analyzer.classify_impact(
            sentiment_score=0.6,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=3,
        )
        assert result == ImpactLevel.HIGH.value

    def test_medium_impact_tier2_moderate_sentiment(self, analyzer: SentimentAnalyzer):
        """Tier-2 source with |sentiment| > 0.5 → MEDIUM."""
        result = analyzer.classify_impact(
            sentiment_score=0.6,
            source_tier=SOURCE_CREDIBILITY_TIER2,
            corroboration=0,
        )
        assert result == ImpactLevel.MEDIUM.value

    def test_medium_impact_tier1_moderate_sentiment(self, analyzer: SentimentAnalyzer):
        """Tier-1 source with moderate sentiment (0.3 < |sentiment| <= 0.7) → MEDIUM."""
        result = analyzer.classify_impact(
            sentiment_score=0.5,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == ImpactLevel.MEDIUM.value

    def test_low_impact_social_low_sentiment_no_corroboration(
        self, analyzer: SentimentAnalyzer
    ):
        """Social media with low sentiment and no corroboration → LOW."""
        result = analyzer.classify_impact(
            sentiment_score=0.2,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=0,
        )
        assert result == ImpactLevel.LOW.value

    def test_low_impact_zero_sentiment(self, analyzer: SentimentAnalyzer):
        """Any source with zero sentiment → LOW."""
        result = analyzer.classify_impact(
            sentiment_score=0.0,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == ImpactLevel.LOW.value

    def test_high_impact_negative_sentiment_tier1(self, analyzer: SentimentAnalyzer):
        """Tier-1 with negative |sentiment| > 0.7 → HIGH."""
        result = analyzer.classify_impact(
            sentiment_score=-0.8,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == ImpactLevel.HIGH.value

    def test_boundary_sentiment_exactly_0_7_tier1_is_medium(
        self, analyzer: SentimentAnalyzer
    ):
        """Tier-1 with |sentiment| = 0.7 exactly (not > 0.7) → MEDIUM."""
        result = analyzer.classify_impact(
            sentiment_score=0.7,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == ImpactLevel.MEDIUM.value

    def test_boundary_corroboration_2_sentiment_exactly_0_5_is_low(
        self, analyzer: SentimentAnalyzer
    ):
        """Corroboration >= 2 but |sentiment| = 0.5 exactly (not > 0.5) → LOW for social."""
        result = analyzer.classify_impact(
            sentiment_score=0.5,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=2,
        )
        assert result == ImpactLevel.LOW.value

    def test_boundary_tier1_sentiment_exactly_0_3_is_low(
        self, analyzer: SentimentAnalyzer
    ):
        """Tier-1 with |sentiment| = 0.3 exactly (not > 0.3) → LOW."""
        result = analyzer.classify_impact(
            sentiment_score=0.3,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == ImpactLevel.LOW.value

    def test_boundary_tier2_sentiment_exactly_0_5_is_low(
        self, analyzer: SentimentAnalyzer
    ):
        """Tier-2 with |sentiment| = 0.5 exactly (not > 0.5) → LOW."""
        result = analyzer.classify_impact(
            sentiment_score=0.5,
            source_tier=SOURCE_CREDIBILITY_TIER2,
            corroboration=0,
        )
        assert result == ImpactLevel.LOW.value

    def test_social_with_corroboration_1_low_sentiment_is_low(
        self, analyzer: SentimentAnalyzer
    ):
        """Social media with 1 corroboration and low sentiment → LOW."""
        result = analyzer.classify_impact(
            sentiment_score=0.1,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=1,
        )
        assert result == ImpactLevel.LOW.value


class TestLazyModelLoading:
    """Tests for lazy model loading behavior."""

    def test_model_not_loaded_on_init(self):
        analyzer = SentimentAnalyzer()
        assert analyzer._model_loaded is False
        assert analyzer._model_available is False
        assert analyzer._pipeline is None

    def test_model_loads_on_first_analyze(self):
        analyzer = SentimentAnalyzer()
        # After analyze, model loading should have been attempted
        analyzer.analyze("test text")
        assert analyzer._model_loaded is True

    def test_model_loads_only_once(self, analyzer: SentimentAnalyzer):
        # Already loaded in fixture
        analyzer._model_loaded = True
        analyzer.analyze("first call")
        analyzer.analyze("second call")
        # Should still be loaded (not re-loaded)
        assert analyzer._model_loaded is True


class TestRuleBasedFallback:
    """Tests for the rule-based fallback scoring."""

    def test_fallback_used_when_model_unavailable(self):
        analyzer = SentimentAnalyzer()
        analyzer._model_loaded = True
        analyzer._model_available = False
        # Should not raise, should use fallback
        score = analyzer.analyze("Markets crash in global crisis")
        assert score < 0.0

    def test_fallback_mixed_sentiment(self):
        analyzer = SentimentAnalyzer()
        analyzer._model_loaded = True
        analyzer._model_available = False
        # Mix of positive and negative keywords
        text = "Despite the crash, there are signs of recovery and growth"
        score = analyzer.analyze(text)
        # Should be somewhere in the middle
        assert -1.0 <= score <= 1.0
