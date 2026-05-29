"""Unit tests for the ConfidenceScorer module.

Tests cover base score calculation (weighted components), penalty application,
threshold rejection, and the full evaluate pipeline.

Validates: Requirements 8.4, 8.5, 21.4, 23.12, Cross-Cutting Rule 4
"""

import pytest

from src.config.constants import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    CONFIDENCE_THRESHOLD_ELEVATED,
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY,
)
from src.strategy.confidence_scorer import ConfidenceResult, ConfidenceScorer


@pytest.fixture
def scorer() -> ConfidenceScorer:
    """Create a fresh ConfidenceScorer instance."""
    return ConfidenceScorer()


# =============================================================================
# Task 16.1: Base score calculation (indicators 40%, Sharpe 30%, regime 30%)
# =============================================================================


class TestCalculateBaseScore:
    """Tests for ConfidenceScorer.calculate() — weighted score calculation."""

    def test_perfect_inputs_return_100(self, scorer: ConfidenceScorer) -> None:
        """All components at maximum should yield 100."""
        result = scorer.calculate(
            confirming_indicators=10,
            total_indicators=10,
            strategy_backtest_sharpe=3.0,
            regime_alignment_score=1.0,
        )
        assert result == 100

    def test_zero_inputs_return_0(self, scorer: ConfidenceScorer) -> None:
        """All components at zero should yield 0."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=10,
            strategy_backtest_sharpe=0.0,
            regime_alignment_score=0.0,
        )
        assert result == 0

    def test_only_indicators_contributing(self, scorer: ConfidenceScorer) -> None:
        """Only indicator agreement at 100%, others at 0 → 40."""
        result = scorer.calculate(
            confirming_indicators=5,
            total_indicators=5,
            strategy_backtest_sharpe=0.0,
            regime_alignment_score=0.0,
        )
        assert result == 40

    def test_only_sharpe_contributing(self, scorer: ConfidenceScorer) -> None:
        """Only Sharpe at max (3.0), others at 0 → 30."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=5,
            strategy_backtest_sharpe=3.0,
            regime_alignment_score=0.0,
        )
        assert result == 30

    def test_only_regime_contributing(self, scorer: ConfidenceScorer) -> None:
        """Only regime alignment at 1.0, others at 0 → 30."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=5,
            strategy_backtest_sharpe=0.0,
            regime_alignment_score=1.0,
        )
        assert result == 30

    def test_half_indicators_half_sharpe_half_regime(self, scorer: ConfidenceScorer) -> None:
        """50% indicators, Sharpe 1.5, regime 0.5 → 50."""
        result = scorer.calculate(
            confirming_indicators=5,
            total_indicators=10,
            strategy_backtest_sharpe=1.5,
            regime_alignment_score=0.5,
        )
        # indicators: 50 * 0.4 = 20
        # sharpe: (1.5/3.0)*100 * 0.3 = 15
        # regime: 50 * 0.3 = 15
        # total = 50
        assert result == 50

    def test_sharpe_clamped_at_3(self, scorer: ConfidenceScorer) -> None:
        """Sharpe above 3.0 should be clamped to 3.0 (max 30 contribution)."""
        result = scorer.calculate(
            confirming_indicators=10,
            total_indicators=10,
            strategy_backtest_sharpe=5.0,
            regime_alignment_score=1.0,
        )
        assert result == 100

    def test_negative_sharpe_treated_as_zero(self, scorer: ConfidenceScorer) -> None:
        """Negative Sharpe should contribute 0 to the score."""
        result = scorer.calculate(
            confirming_indicators=10,
            total_indicators=10,
            strategy_backtest_sharpe=-1.0,
            regime_alignment_score=1.0,
        )
        # indicators: 100 * 0.4 = 40
        # sharpe: 0 * 0.3 = 0
        # regime: 100 * 0.3 = 30
        # total = 70
        assert result == 70

    def test_regime_alignment_clamped_above_1(self, scorer: ConfidenceScorer) -> None:
        """Regime alignment above 1.0 should be clamped to 1.0."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=5,
            strategy_backtest_sharpe=0.0,
            regime_alignment_score=1.5,
        )
        assert result == 30

    def test_regime_alignment_clamped_below_0(self, scorer: ConfidenceScorer) -> None:
        """Negative regime alignment should be clamped to 0."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=5,
            strategy_backtest_sharpe=0.0,
            regime_alignment_score=-0.5,
        )
        assert result == 0

    def test_zero_total_indicators_gives_zero_indicator_score(
        self, scorer: ConfidenceScorer
    ) -> None:
        """When total_indicators is 0, indicator component should be 0."""
        result = scorer.calculate(
            confirming_indicators=0,
            total_indicators=0,
            strategy_backtest_sharpe=3.0,
            regime_alignment_score=1.0,
        )
        # indicators: 0
        # sharpe: 30
        # regime: 30
        # total = 60
        assert result == 60

    def test_result_is_integer(self, scorer: ConfidenceScorer) -> None:
        """Result should always be an integer."""
        result = scorer.calculate(
            confirming_indicators=3,
            total_indicators=7,
            strategy_backtest_sharpe=1.2,
            regime_alignment_score=0.65,
        )
        assert isinstance(result, int)

    def test_result_bounded_0_to_100(self, scorer: ConfidenceScorer) -> None:
        """Result should always be in [0, 100]."""
        result = scorer.calculate(
            confirming_indicators=3,
            total_indicators=7,
            strategy_backtest_sharpe=1.2,
            regime_alignment_score=0.65,
        )
        assert 0 <= result <= 100


# =============================================================================
# Task 16.2: Confidence threshold enforcement
# =============================================================================


class TestShouldReject:
    """Tests for ConfidenceScorer.should_reject() — threshold enforcement."""

    def test_score_below_default_threshold_rejected(self, scorer: ConfidenceScorer) -> None:
        """Score below 60 should be rejected."""
        assert scorer.should_reject(59) is True

    def test_score_at_default_threshold_not_rejected(self, scorer: ConfidenceScorer) -> None:
        """Score exactly at 60 should NOT be rejected."""
        assert scorer.should_reject(60) is False

    def test_score_above_default_threshold_not_rejected(self, scorer: ConfidenceScorer) -> None:
        """Score above 60 should NOT be rejected."""
        assert scorer.should_reject(80) is False

    def test_custom_threshold(self, scorer: ConfidenceScorer) -> None:
        """Custom threshold should be respected."""
        assert scorer.should_reject(74, threshold=75) is True
        assert scorer.should_reject(75, threshold=75) is False

    def test_zero_score_rejected(self, scorer: ConfidenceScorer) -> None:
        """Score of 0 should be rejected."""
        assert scorer.should_reject(0) is True

    def test_negative_score_rejected(self, scorer: ConfidenceScorer) -> None:
        """Negative score should be rejected."""
        assert scorer.should_reject(-10) is True

    def test_score_100_not_rejected(self, scorer: ConfidenceScorer) -> None:
        """Score of 100 should NOT be rejected."""
        assert scorer.should_reject(100) is False

    def test_default_threshold_matches_constant(self, scorer: ConfidenceScorer) -> None:
        """Default threshold should match CONFIDENCE_THRESHOLD_DEFAULT (60)."""
        assert CONFIDENCE_THRESHOLD_DEFAULT == 60
        assert scorer.should_reject(CONFIDENCE_THRESHOLD_DEFAULT) is False
        assert scorer.should_reject(CONFIDENCE_THRESHOLD_DEFAULT - 1) is True


# =============================================================================
# Task 16.3: Cumulative penalty application (Cross-Cutting Rule 4)
# =============================================================================


class TestApplyPenalties:
    """Tests for ConfidenceScorer.apply_penalties() — cumulative penalties."""

    def test_no_penalties_returns_base_score(self, scorer: ConfidenceScorer) -> None:
        """No penalties active → score unchanged."""
        result = scorer.apply_penalties(
            base_score=80,
            mistake_pattern_match=False,
            mistake_pattern_reactivated=False,
            high_impact_news_active=False,
        )
        assert result == 80

    def test_mistake_pattern_penalty_minus_20(self, scorer: ConfidenceScorer) -> None:
        """Active mistake pattern → -20 penalty."""
        result = scorer.apply_penalties(
            base_score=80,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=False,
            high_impact_news_active=False,
        )
        assert result == 80 - MISTAKE_BASE_CONFIDENCE_PENALTY
        assert result == 60

    def test_mistake_reactivated_penalty_minus_30(self, scorer: ConfidenceScorer) -> None:
        """Reactivated mistake pattern → -30 penalty (not -20)."""
        result = scorer.apply_penalties(
            base_score=80,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=False,
        )
        assert result == 80 - MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
        assert result == 50

    def test_news_penalty_minus_25(self, scorer: ConfidenceScorer) -> None:
        """High-impact news active → -25 penalty."""
        result = scorer.apply_penalties(
            base_score=80,
            mistake_pattern_match=False,
            mistake_pattern_reactivated=False,
            high_impact_news_active=True,
        )
        assert result == 80 - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        assert result == 55

    def test_cumulative_mistake_and_news(self, scorer: ConfidenceScorer) -> None:
        """Both mistake pattern and news → cumulative -20 + -25 = -45."""
        result = scorer.apply_penalties(
            base_score=90,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=False,
            high_impact_news_active=True,
        )
        expected = 90 - MISTAKE_BASE_CONFIDENCE_PENALTY - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        assert result == expected
        assert result == 45

    def test_cumulative_reactivated_and_news(self, scorer: ConfidenceScorer) -> None:
        """Reactivated pattern + news → cumulative -30 + -25 = -55."""
        result = scorer.apply_penalties(
            base_score=90,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=True,
        )
        expected = 90 - MISTAKE_REACTIVATED_CONFIDENCE_PENALTY - NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
        assert result == expected
        assert result == 35

    def test_penalties_can_go_below_zero(self, scorer: ConfidenceScorer) -> None:
        """Penalties can push score below 0."""
        result = scorer.apply_penalties(
            base_score=30,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=True,
            high_impact_news_active=True,
        )
        assert result < 0
        assert result == 30 - 30 - 25

    def test_penalty_constants_match_spec(self) -> None:
        """Verify penalty constants match the specification."""
        assert MISTAKE_BASE_CONFIDENCE_PENALTY == 20
        assert MISTAKE_REACTIVATED_CONFIDENCE_PENALTY == 30
        assert NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY == 25


# =============================================================================
# Task 16.4: Full evaluate pipeline
# =============================================================================


class TestEvaluate:
    """Tests for ConfidenceScorer.evaluate() — full pipeline."""

    def test_high_confidence_no_penalties_accepted(self, scorer: ConfidenceScorer) -> None:
        """High base score with no penalties → accepted."""
        result = scorer.evaluate(
            confirming_indicators=8,
            total_indicators=10,
            strategy_backtest_sharpe=2.0,
            regime_alignment_score=0.8,
        )
        assert isinstance(result, ConfidenceResult)
        assert result.rejected is False
        assert result.base_score == result.final_score
        assert result.penalties_applied == []
        assert result.rejection_reason is None

    def test_low_confidence_rejected(self, scorer: ConfidenceScorer) -> None:
        """Low base score → rejected."""
        result = scorer.evaluate(
            confirming_indicators=2,
            total_indicators=10,
            strategy_backtest_sharpe=0.5,
            regime_alignment_score=0.2,
        )
        assert result.rejected is True
        assert result.rejection_reason is not None
        assert "below threshold" in result.rejection_reason.lower()

    def test_penalty_causes_rejection(self, scorer: ConfidenceScorer) -> None:
        """Base score above threshold but penalties push below → rejected."""
        # Base score should be around 65 (above 60)
        result = scorer.evaluate(
            confirming_indicators=6,
            total_indicators=10,
            strategy_backtest_sharpe=1.5,
            regime_alignment_score=0.5,
            mistake_pattern_match=True,
            high_impact_news_active=True,
        )
        # Base: 24 + 15 + 15 = 54... let's use higher values
        result = scorer.evaluate(
            confirming_indicators=8,
            total_indicators=10,
            strategy_backtest_sharpe=1.5,
            regime_alignment_score=0.7,
            mistake_pattern_match=True,
            high_impact_news_active=True,
        )
        # Base: 32 + 15 + 21 = 68
        # Penalties: -20 -25 = -45
        # Final: 68 - 45 = 23
        assert result.base_score == 68
        assert result.final_score == 23
        assert result.rejected is True

    def test_penalties_tracked_in_result(self, scorer: ConfidenceScorer) -> None:
        """Penalties applied should be tracked in the result."""
        result = scorer.evaluate(
            confirming_indicators=10,
            total_indicators=10,
            strategy_backtest_sharpe=3.0,
            regime_alignment_score=1.0,
            mistake_pattern_match=True,
            high_impact_news_active=True,
        )
        assert len(result.penalties_applied) == 2
        penalty_sources = [p[0] for p in result.penalties_applied]
        assert "mistake_pattern" in penalty_sources
        assert "high_impact_news" in penalty_sources

    def test_reactivated_penalty_tracked(self, scorer: ConfidenceScorer) -> None:
        """Reactivated pattern penalty should be tracked distinctly."""
        result = scorer.evaluate(
            confirming_indicators=10,
            total_indicators=10,
            strategy_backtest_sharpe=3.0,
            regime_alignment_score=1.0,
            mistake_pattern_match=True,
            mistake_pattern_reactivated=True,
        )
        penalty_sources = [p[0] for p in result.penalties_applied]
        assert "mistake_pattern_reactivated" in penalty_sources
        penalty_amounts = [p[1] for p in result.penalties_applied]
        assert MISTAKE_REACTIVATED_CONFIDENCE_PENALTY in penalty_amounts

    def test_custom_threshold_in_evaluate(self, scorer: ConfidenceScorer) -> None:
        """Custom threshold should be used in evaluate."""
        result = scorer.evaluate(
            confirming_indicators=6,
            total_indicators=10,
            strategy_backtest_sharpe=1.5,
            regime_alignment_score=0.5,
            threshold=CONFIDENCE_THRESHOLD_ELEVATED,
        )
        # Base: 24 + 15 + 15 = 54, below 75
        assert result.rejected is True

    def test_evaluate_base_score_matches_calculate(self, scorer: ConfidenceScorer) -> None:
        """evaluate().base_score should match calculate() for same inputs."""
        base = scorer.calculate(
            confirming_indicators=7,
            total_indicators=10,
            strategy_backtest_sharpe=2.0,
            regime_alignment_score=0.8,
        )
        result = scorer.evaluate(
            confirming_indicators=7,
            total_indicators=10,
            strategy_backtest_sharpe=2.0,
            regime_alignment_score=0.8,
        )
        assert result.base_score == base


class TestConfidenceResult:
    """Tests for the ConfidenceResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """ConfidenceResult should have all expected fields."""
        result = ConfidenceResult(
            base_score=80,
            final_score=55,
            penalties_applied=[("mistake_pattern", 20), ("high_impact_news", 25)],
            rejected=True,
            rejection_reason="Confidence score 55 below threshold 60",
        )
        assert result.base_score == 80
        assert result.final_score == 55
        assert len(result.penalties_applied) == 2
        assert result.rejected is True
        assert result.rejection_reason is not None

    def test_default_rejection_reason_is_none(self) -> None:
        """rejection_reason defaults to None."""
        result = ConfidenceResult(
            base_score=80,
            final_score=80,
            penalties_applied=[],
            rejected=False,
        )
        assert result.rejection_reason is None
