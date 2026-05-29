"""Confidence scoring for trade signals.

Calculates a composite confidence score based on indicator agreement,
strategy backtest performance, and regime alignment. Applies cumulative
penalties for mistake patterns and high-impact news events per Cross-Cutting Rule 4.

Validates: Requirements 8.4, 8.5, 21.4, 23.12, Cross-Cutting Rule 4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config.constants import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence score calculation.

    Attributes:
        base_score: Score before penalties (0-100).
        final_score: Score after all penalties applied (can be < 0).
        penalties_applied: List of (source, amount) tuples for applied penalties.
        rejected: Whether the signal should be rejected (final_score < threshold).
        rejection_reason: Human-readable reason if rejected.
    """

    base_score: int
    final_score: int
    penalties_applied: list[tuple[str, int]]
    rejected: bool
    rejection_reason: str | None = None


class ConfidenceScorer:
    """Calculates and validates trade signal confidence scores.

    The confidence score is a weighted combination of:
      - Indicator agreement: 40% weight
      - Strategy backtest Sharpe ratio: 30% weight (normalized to 0-100)
      - Regime alignment score: 30% weight

    Penalties are applied cumulatively per Cross-Cutting Rule 4:
      - Mistake pattern match: -20 (or -30 if reactivated)
      - High-impact news active: -25

    Signals with final score below the threshold (default 60) are rejected.
    """

    # Sharpe ratio normalization: Sharpe of 3.0 maps to 100
    SHARPE_NORMALIZATION_MAX: float = 3.0

    def calculate(
        self,
        confirming_indicators: int,
        total_indicators: int,
        strategy_backtest_sharpe: float,
        regime_alignment_score: float,
    ) -> int:
        """Calculate the base confidence score from component metrics.

        Args:
            confirming_indicators: Number of indicators confirming the signal.
            total_indicators: Total number of indicators evaluated.
            strategy_backtest_sharpe: Strategy's Sharpe ratio over prior 30-day
                backtest period.
            regime_alignment_score: Alignment score between signal direction and
                current regime classification (0.0 to 1.0).

        Returns:
            Base confidence score as an integer in [0, 100].
        """
        # Indicator agreement component (40% weight)
        if total_indicators > 0:
            indicator_score = (confirming_indicators / total_indicators) * 100.0
        else:
            indicator_score = 0.0

        # Backtest Sharpe component (30% weight), normalized to 0-100
        # Sharpe of 0 or below → 0, Sharpe of 3.0+ → 100
        normalized_sharpe = max(0.0, min(strategy_backtest_sharpe, self.SHARPE_NORMALIZATION_MAX))
        sharpe_score = (normalized_sharpe / self.SHARPE_NORMALIZATION_MAX) * 100.0

        # Regime alignment component (30% weight)
        # Clamp to [0, 1] then scale to 0-100
        alignment_clamped = max(0.0, min(regime_alignment_score, 1.0))
        alignment_score = alignment_clamped * 100.0

        # Weighted combination
        base_score = (
            indicator_score * 0.40
            + sharpe_score * 0.30
            + alignment_score * 0.30
        )

        # Clamp to [0, 100] and round to integer
        return int(round(max(0.0, min(base_score, 100.0))))

    def apply_penalties(
        self,
        base_score: int,
        mistake_pattern_match: bool,
        mistake_pattern_reactivated: bool,
        high_impact_news_active: bool,
    ) -> int:
        """Apply cumulative confidence penalties per Cross-Cutting Rule 4.

        Penalties are applied cumulatively:
          - Mistake pattern match: -20 points (or -30 if reactivated)
          - High-impact news active: -25 points

        A signal with base confidence 90 matching both a mistake pattern and
        high-impact news would be reduced to 45 (90 - 20 - 25).

        Args:
            base_score: The base confidence score before penalties.
            mistake_pattern_match: Whether the signal matches an active mistake pattern.
            mistake_pattern_reactivated: Whether the matched pattern was reactivated
                (implies mistake_pattern_match is True).
            high_impact_news_active: Whether high-impact news is active for the
                instrument.

        Returns:
            Penalized confidence score (can go below 0).
        """
        score = base_score

        if mistake_pattern_match:
            if mistake_pattern_reactivated:
                score -= MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
            else:
                score -= MISTAKE_BASE_CONFIDENCE_PENALTY

        if high_impact_news_active:
            score -= NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY

        return score

    def should_reject(self, final_score: int, threshold: int = CONFIDENCE_THRESHOLD_DEFAULT) -> bool:
        """Determine whether a signal should be rejected based on confidence score.

        Args:
            final_score: The final confidence score after all penalties.
            threshold: Minimum confidence threshold (default 60).

        Returns:
            True if the signal should be rejected (score < threshold).
        """
        return final_score < threshold

    def evaluate(
        self,
        confirming_indicators: int,
        total_indicators: int,
        strategy_backtest_sharpe: float,
        regime_alignment_score: float,
        mistake_pattern_match: bool = False,
        mistake_pattern_reactivated: bool = False,
        high_impact_news_active: bool = False,
        threshold: int = CONFIDENCE_THRESHOLD_DEFAULT,
    ) -> ConfidenceResult:
        """Full confidence evaluation: calculate, penalize, and decide.

        Convenience method that runs the full pipeline: base score calculation,
        penalty application, and rejection decision.

        Args:
            confirming_indicators: Number of confirming indicators.
            total_indicators: Total indicators evaluated.
            strategy_backtest_sharpe: Strategy's 30-day backtest Sharpe ratio.
            regime_alignment_score: Regime alignment score (0.0 to 1.0).
            mistake_pattern_match: Whether signal matches a mistake pattern.
            mistake_pattern_reactivated: Whether the pattern is reactivated.
            high_impact_news_active: Whether high-impact news is active.
            threshold: Minimum confidence threshold.

        Returns:
            ConfidenceResult with full scoring details.
        """
        base_score = self.calculate(
            confirming_indicators=confirming_indicators,
            total_indicators=total_indicators,
            strategy_backtest_sharpe=strategy_backtest_sharpe,
            regime_alignment_score=regime_alignment_score,
        )

        final_score = self.apply_penalties(
            base_score=base_score,
            mistake_pattern_match=mistake_pattern_match,
            mistake_pattern_reactivated=mistake_pattern_reactivated,
            high_impact_news_active=high_impact_news_active,
        )

        # Track penalties applied
        penalties: list[tuple[str, int]] = []
        if mistake_pattern_match:
            if mistake_pattern_reactivated:
                penalties.append(("mistake_pattern_reactivated", MISTAKE_REACTIVATED_CONFIDENCE_PENALTY))
            else:
                penalties.append(("mistake_pattern", MISTAKE_BASE_CONFIDENCE_PENALTY))
        if high_impact_news_active:
            penalties.append(("high_impact_news", NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY))

        rejected = self.should_reject(final_score, threshold)
        rejection_reason = None
        if rejected:
            rejection_reason = (
                f"Confidence score {final_score} below threshold {threshold}"
            )

        logger.info(
            "Confidence evaluation: base=%d final=%d penalties=%s rejected=%s",
            base_score,
            final_score,
            penalties,
            rejected,
        )

        return ConfidenceResult(
            base_score=base_score,
            final_score=final_score,
            penalties_applied=penalties,
            rejected=rejected,
            rejection_reason=rejection_reason,
        )
