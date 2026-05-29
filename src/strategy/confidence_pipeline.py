"""End-to-end confidence penalty pipeline.

Implements the full confidence penalty pipeline per Cross-Cutting Rule 4:
  base confidence → apply Mistake_Pattern penalty → apply High-impact news penalty
  → reject if final confidence < 60.

The pipeline accepts penalty inputs from the MistakeAnalyzer and NewsEngine,
and is called by the Strategy Engine before passing signals to the Risk Engine.

Validates: Cross-Cutting Rule 4, Requirements 8.5, 21.4, 23.12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config.constants import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY,
)

logger = logging.getLogger(__name__)


@dataclass
class PenaltyInput:
    """Input describing penalties to apply in the confidence pipeline.

    Attributes:
        mistake_pattern_active: Whether an active mistake pattern matches the signal.
        mistake_pattern_reactivated: Whether the matched pattern was previously
            resolved and has been reactivated (implies mistake_pattern_active is True).
        high_impact_news_active: Whether high-impact news affects the instrument.
    """

    mistake_pattern_active: bool = False
    mistake_pattern_reactivated: bool = False
    high_impact_news_active: bool = False


@dataclass
class PipelineResult:
    """Result of the confidence penalty pipeline evaluation.

    Attributes:
        final_confidence: The confidence score after all penalties are applied.
        penalties_applied: List of (penalty_source, penalty_amount) tuples.
        rejected: Whether the signal was rejected (final_confidence < threshold).
        rejection_reason: Human-readable reason if rejected, None otherwise.
    """

    final_confidence: int
    penalties_applied: list[tuple[str, int]] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None


class ConfidencePipeline:
    """End-to-end confidence penalty pipeline.

    Processes a trade signal's base confidence through the penalty stack:
      1. Start with base confidence score (0-100) from the Strategy Engine
      2. Apply Mistake_Pattern penalty: -20 for active pattern, -30 for reactivated
      3. Apply High-impact news penalty: -25 when high-impact news affects instrument
      4. Reject the signal if final confidence < 60

    The pipeline logs all penalty applications and rejections for audit purposes.

    Usage:
        pipeline = ConfidencePipeline()
        penalty_input = PenaltyInput(
            mistake_pattern_active=True,
            high_impact_news_active=True,
        )
        result = pipeline.evaluate(base_confidence=90, penalties=penalty_input)
        if result.rejected:
            # Signal should not be passed to Risk Engine
            ...
    """

    def __init__(self, threshold: int = CONFIDENCE_THRESHOLD_DEFAULT) -> None:
        """Initialize the confidence pipeline.

        Args:
            threshold: Minimum confidence score for signal acceptance (default 60).
        """
        self.threshold = threshold

    def evaluate(
        self,
        base_confidence: int,
        penalties: PenaltyInput,
    ) -> PipelineResult:
        """Run the full confidence penalty pipeline.

        Applies penalties in order:
          1. Mistake_Pattern penalty (-20 or -30 if reactivated)
          2. High-impact news penalty (-25)

        Then checks if the final confidence meets the threshold.

        Args:
            base_confidence: Base confidence score (0-100) from the Strategy Engine.
            penalties: PenaltyInput describing which penalties to apply.

        Returns:
            PipelineResult with final confidence, penalties applied, and rejection status.
        """
        current_confidence = base_confidence
        penalties_applied: list[tuple[str, int]] = []

        # Step 1: Apply Mistake_Pattern penalty
        if penalties.mistake_pattern_active:
            if penalties.mistake_pattern_reactivated:
                penalty_amount = MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
                penalty_source = "mistake_pattern_reactivated"
            else:
                penalty_amount = MISTAKE_BASE_CONFIDENCE_PENALTY
                penalty_source = "mistake_pattern"

            current_confidence -= penalty_amount
            penalties_applied.append((penalty_source, penalty_amount))

            logger.info(
                "Confidence penalty applied: source=%s amount=-%d confidence_after=%d",
                penalty_source,
                penalty_amount,
                current_confidence,
            )

        # Step 2: Apply High-impact news penalty
        if penalties.high_impact_news_active:
            penalty_amount = NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY
            penalty_source = "high_impact_news"

            current_confidence -= penalty_amount
            penalties_applied.append((penalty_source, penalty_amount))

            logger.info(
                "Confidence penalty applied: source=%s amount=-%d confidence_after=%d",
                penalty_source,
                penalty_amount,
                current_confidence,
            )

        # Step 3: Determine rejection
        rejected = current_confidence < self.threshold
        rejection_reason: str | None = None

        if rejected:
            rejection_reason = (
                f"Signal rejected: final confidence {current_confidence} "
                f"is below threshold {self.threshold}"
            )
            logger.warning(
                "Signal rejected by confidence pipeline: "
                "base=%d final=%d threshold=%d penalties=%s reason='%s'",
                base_confidence,
                current_confidence,
                self.threshold,
                penalties_applied,
                rejection_reason,
            )
        else:
            logger.info(
                "Signal accepted by confidence pipeline: "
                "base=%d final=%d threshold=%d penalties=%s",
                base_confidence,
                current_confidence,
                self.threshold,
                penalties_applied,
            )

        return PipelineResult(
            final_confidence=current_confidence,
            penalties_applied=penalties_applied,
            rejected=rejected,
            rejection_reason=rejection_reason,
        )
