"""Unit tests for the ConfidencePipeline module.

Tests cover the end-to-end confidence penalty pipeline:
  base confidence → Mistake_Pattern penalty → High-impact news penalty → rejection.

Validates: Cross-Cutting Rule 4, Requirements 8.5, 21.4, 23.12
"""

import pytest

from src.config.constants import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY,
)
from src.strategy.confidence_pipeline import (
    ConfidencePipeline,
    PenaltyInput,
    PipelineResult,
)


@pytest.fixture
def pipeline() -> ConfidencePipeline:
    """Create a fresh ConfidencePipeline instance with default threshold."""
    return ConfidencePipeline()


# =============================================================================
# Base confidence passes through unchanged when no penalties
# =============================================================================


class TestNoPenalties:
    """Tests that base confidence passes through unchanged when no penalties apply."""

    def test_base_confidence_unchanged_no_penalties(self, pipeline: ConfidencePipeline) -> None:
        """Base confidence should pass through unchanged when no penalties are active."""
        penalties = PenaltyInput(
            mistake_pattern_active=False,
            mistake_pattern_reactivated=False,
            high_impact_news_active=False,
        )
        result = pipeline.evaluate(base_confidence=85, penalties=penalties)

        assert result.final_confidence == 85
        assert result.penalties_applied == []
        assert result.rejected is False
        assert result.rejection_reason is None

    def test_high_confidence_no_penalties_accepted(self, pipeline: ConfidencePipeline) -> None:
        """A high confidence score with no penalties should be accepted."""
        penalties = PenaltyInput()
        result = pipeline.evaluate(base_confidence=100, penalties=penalties)

        assert result.final_confidence == 100
        assert result.rejected is False

    def test_exactly_at_threshold_no_penalties_accepted(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Confidence exactly at threshold (60) with no penalties should be accepted."""
        penalties = PenaltyInput()
        result = pipeline.evaluate(base_confidence=60, penalties=penalties)

        assert result.final_confidence == 60
        assert result.rejected is False


# =============================================================================
# Mistake pattern penalty reduces confidence by 20
# =============================================================================


class TestMistakePatternPenalty:
    """Tests for active mistake pattern penalty (-20)."""

    def test_mistake_pattern_reduces_by_20(self, pipeline: ConfidencePipeline) -> None:
        """Active mistake pattern should reduce confidence by 20."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=False,
            high_impact_news_active=False,
        )
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        assert result.final_confidence == 80 - MISTAKE_BASE_CONFIDENCE_PENALTY
        assert result.final_confidence == 60

    def test_mistake_pattern_penalty_tracked(self, pipeline: ConfidencePipeline) -> None:
        """Mistake pattern penalty should be tracked in penalties_applied."""
        penalties = PenaltyInput(mistake_pattern_active=True)
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        assert len(result.penalties_applied) == 1
        assert result.penalties_applied[0] == ("mistake_pattern", 20)


# =============================================================================
# Reactivated pattern penalty reduces confidence by 30
# =============================================================================


class TestReactivatedPatternPenalty:
    """Tests for reactivated mistake pattern penalty (-30)."""

    def test_reactivated_pattern_reduces_by_30(self, pipeline: ConfidencePipeline) -> None:
        """Reactivated mistake pattern should reduce confidence by 30."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=False,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        assert result.final_confidence == 90 - MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
        assert result.final_confidence == 60

    def test_reactivated_penalty_tracked(self, pipeline: ConfidencePipeline) -> None:
        """Reactivated pattern penalty should be tracked distinctly."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        assert len(result.penalties_applied) == 1
        assert result.penalties_applied[0] == ("mistake_pattern_reactivated", 30)


# =============================================================================
# High-impact news penalty reduces confidence by 25
# =============================================================================


class TestHighImpactNewsPenalty:
    """Tests for high-impact news penalty (-25)."""

    def test_news_penalty_reduces_by_25(self, pipeline: ConfidencePipeline) -> None:
        """High-impact news should reduce confidence by 25."""
        penalties = PenaltyInput(
            mistake_pattern_active=False,
            mistake_pattern_reactivated=False,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        assert result.final_confidence == 80 - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        assert result.final_confidence == 55

    def test_news_penalty_tracked(self, pipeline: ConfidencePipeline) -> None:
        """News penalty should be tracked in penalties_applied."""
        penalties = PenaltyInput(high_impact_news_active=True)
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        assert len(result.penalties_applied) == 1
        assert result.penalties_applied[0] == ("high_impact_news", 25)


# =============================================================================
# Cumulative penalties stack (mistake + news)
# =============================================================================


class TestCumulativePenalties:
    """Tests that penalties stack cumulatively per Cross-Cutting Rule 4."""

    def test_mistake_and_news_stack(self, pipeline: ConfidencePipeline) -> None:
        """Mistake pattern (-20) and news (-25) should stack to -45."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=False,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        expected = 90 - MISTAKE_BASE_CONFIDENCE_PENALTY - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        assert result.final_confidence == expected
        assert result.final_confidence == 45

    def test_reactivated_and_news_stack(self, pipeline: ConfidencePipeline) -> None:
        """Reactivated pattern (-30) and news (-25) should stack to -55."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        expected = (
            90 - MISTAKE_REACTIVATED_CONFIDENCE_PENALTY - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        )
        assert result.final_confidence == expected
        assert result.final_confidence == 35

    def test_cumulative_penalties_tracked(self, pipeline: ConfidencePipeline) -> None:
        """Both penalties should be tracked in penalties_applied list."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        assert len(result.penalties_applied) == 2
        sources = [p[0] for p in result.penalties_applied]
        assert "mistake_pattern" in sources
        assert "high_impact_news" in sources

    def test_spec_example_90_minus_20_minus_25_equals_45(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Cross-Cutting Rule 4 example: 90 - 20 - 25 = 45, rejected."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        assert result.final_confidence == 45
        assert result.rejected is True


# =============================================================================
# Signal rejected when final confidence < 60
# =============================================================================


class TestRejection:
    """Tests that signals are rejected when final confidence < 60."""

    def test_below_threshold_rejected(self, pipeline: ConfidencePipeline) -> None:
        """Final confidence below 60 should be rejected."""
        penalties = PenaltyInput(high_impact_news_active=True)
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        # 80 - 25 = 55, below 60
        assert result.final_confidence == 55
        assert result.rejected is True
        assert result.rejection_reason is not None

    def test_rejection_reason_contains_score(self, pipeline: ConfidencePipeline) -> None:
        """Rejection reason should mention the final confidence score."""
        penalties = PenaltyInput(high_impact_news_active=True)
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        assert "55" in result.rejection_reason
        assert "60" in result.rejection_reason

    def test_low_base_confidence_rejected_without_penalties(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Base confidence below threshold should be rejected even without penalties."""
        penalties = PenaltyInput()
        result = pipeline.evaluate(base_confidence=59, penalties=penalties)

        assert result.rejected is True
        assert result.final_confidence == 59

    def test_penalties_can_push_below_zero(self, pipeline: ConfidencePipeline) -> None:
        """Penalties can push confidence below zero."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=30, penalties=penalties)

        # 30 - 30 - 25 = -25
        assert result.final_confidence == -25
        assert result.rejected is True


# =============================================================================
# Signal allowed when final confidence >= 60 after penalties
# =============================================================================


class TestAcceptance:
    """Tests that signals are accepted when final confidence >= 60 after penalties."""

    def test_exactly_at_threshold_after_penalty_accepted(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Final confidence exactly at 60 after penalty should be accepted."""
        penalties = PenaltyInput(mistake_pattern_active=True)
        result = pipeline.evaluate(base_confidence=80, penalties=penalties)

        # 80 - 20 = 60, exactly at threshold
        assert result.final_confidence == 60
        assert result.rejected is False
        assert result.rejection_reason is None

    def test_above_threshold_after_penalty_accepted(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Final confidence above 60 after penalty should be accepted."""
        penalties = PenaltyInput(mistake_pattern_active=True)
        result = pipeline.evaluate(base_confidence=85, penalties=penalties)

        # 85 - 20 = 65, above threshold
        assert result.final_confidence == 65
        assert result.rejected is False

    def test_high_confidence_survives_all_penalties(
        self, pipeline: ConfidencePipeline
    ) -> None:
        """Very high confidence can survive all penalties and still be accepted."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            high_impact_news_active=True,
        )
        # Need base >= 60 + 20 + 25 = 105, but max is 100
        # So with both penalties, max surviving score is 100 - 45 = 55 (rejected)
        # With just mistake: 85 - 20 = 65 (accepted)
        result = pipeline.evaluate(base_confidence=85, penalties=PenaltyInput(
            mistake_pattern_active=True,
        ))
        assert result.final_confidence == 65
        assert result.rejected is False


# =============================================================================
# Pipeline result structure
# =============================================================================


class TestPipelineResult:
    """Tests for the PipelineResult dataclass."""

    def test_result_has_all_fields(self, pipeline: ConfidencePipeline) -> None:
        """PipelineResult should have all expected fields."""
        penalties = PenaltyInput(
            mistake_pattern_active=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalties)

        assert isinstance(result, PipelineResult)
        assert isinstance(result.final_confidence, int)
        assert isinstance(result.penalties_applied, list)
        assert isinstance(result.rejected, bool)

    def test_custom_threshold(self) -> None:
        """Pipeline should respect a custom threshold."""
        pipeline = ConfidencePipeline(threshold=75)
        penalties = PenaltyInput()
        result = pipeline.evaluate(base_confidence=70, penalties=penalties)

        assert result.rejected is True
        assert result.final_confidence == 70

    def test_default_threshold_is_60(self) -> None:
        """Default threshold should be 60 per Cross-Cutting Rule 4."""
        pipeline = ConfidencePipeline()
        assert pipeline.threshold == CONFIDENCE_THRESHOLD_DEFAULT
        assert pipeline.threshold == 60
