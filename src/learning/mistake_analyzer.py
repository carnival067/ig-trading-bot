"""Mistake Analyzer: classification, pattern detection, and penalty application.

Implements the self-learning mistake analysis system that detects recurring
trading mistakes, applies confidence penalties and position size reductions,
and tracks resolution progress.

Validates: Requirements 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 21.10
Cross-Cutting Rules: 1 (multiplicative stacking), 2 (HFT not exempt)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config.constants import (
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_BASE_SIZE_REDUCTION,
    MISTAKE_PATTERN_MATCH_INDICATORS,
    MISTAKE_PATTERN_THRESHOLD,
    MISTAKE_PATTERN_TOTAL_INDICATORS,
    MISTAKE_PATTERN_WINDOW_DAYS,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_SIZE_REDUCTION,
    MISTAKE_RESOLUTION_STREAK,
)
from src.learning.mistake_database import (
    MistakeClassification,
    MistakeDatabase,
    MistakePattern,
    MistakeRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    """A trade signal to be evaluated against active mistake patterns.

    Attributes:
        regime: Current market regime.
        strategy: Strategy generating the signal.
        indicators: Dict of indicator name → value at signal time.
        confidence: Base confidence score before penalties.
        is_hft: Whether this is an HFT signal.
    """

    regime: str
    strategy: str
    indicators: dict[str, float]
    confidence: int
    is_hft: bool = False


@dataclass
class ClosedTrade:
    """A closed trade for mistake analysis and resolution tracking.

    Attributes:
        trade_id: Unique trade identifier.
        regime: Market regime at entry.
        strategy: Strategy that generated the trade.
        indicators: Indicator values at entry.
        confidence_at_entry: Confidence score at entry.
        exit_reason: Reason for trade exit.
        pnl: Profit/loss amount.
        entry_conditions: Market conditions at entry.
        is_profitable: Whether the trade was profitable.
    """

    trade_id: str
    regime: str
    strategy: str
    indicators: dict[str, float]
    confidence_at_entry: int
    exit_reason: str
    pnl: float
    entry_conditions: dict[str, Any] = field(default_factory=dict)

    @property
    def is_profitable(self) -> bool:
        """Whether the trade resulted in a profit."""
        return self.pnl > 0


@dataclass
class MarketOutcome:
    """Market outcome data used for mistake classification.

    Attributes:
        actual_direction: The actual market direction after entry.
        volatility_realized: Realized volatility vs expected.
        regime_actual: The actual regime (may differ from detected).
        breakout_confirmed: Whether a breakout was confirmed.
        timing_optimal: Whether entry timing was optimal.
    """

    actual_direction: str  # "up" or "down"
    volatility_realized: float
    regime_actual: str
    breakout_confirmed: bool = True
    timing_optimal: bool = True


# ---------------------------------------------------------------------------
# Mistake Analyzer
# ---------------------------------------------------------------------------


class MistakeAnalyzer:
    """Classifies trading mistakes, detects patterns, and applies penalties.

    Core responsibilities:
    - Classify losing trades into 6 root-cause categories (Req 21.2)
    - Detect recurring patterns (5+ losses in 30 days) (Req 21.3)
    - Apply confidence penalties and size reductions (Req 21.4, 21.5)
    - Track resolution progress (20 consecutive profits) (Req 21.7)
    - Reactivate resolved patterns with increased penalties (Req 21.8)
    - Load patterns on startup without warm-up (Req 21.10)
    - Expose patterns to Dashboard API (Req 21.9)

    Mistake_Pattern penalties apply to ALL signals including HFT
    (Cross-Cutting Rule 2).
    """

    PATTERN_THRESHOLD = MISTAKE_PATTERN_THRESHOLD
    PATTERN_WINDOW_DAYS = MISTAKE_PATTERN_WINDOW_DAYS
    RESOLUTION_STREAK = MISTAKE_RESOLUTION_STREAK

    def __init__(self, mistake_db: MistakeDatabase) -> None:
        self.mistake_db = mistake_db
        self.active_patterns: list[MistakePattern] = []

    # ─── Classification (Req 21.2) ───────────────────────────────────────

    def classify_mistake(
        self, trade_context: ClosedTrade, market_outcome: MarketOutcome
    ) -> MistakeClassification:
        """Classify a losing trade into one of 6 root-cause categories.

        Classification logic:
        - COUNTER_TREND: Entry against the actual market direction
        - FALSE_BREAKOUT: Breakout entry that was not confirmed
        - VOLATILITY_MISJUDGMENT: Realized volatility significantly exceeded expected
        - POOR_TIMING: Entry timing was suboptimal
        - OVEREXPOSURE: Excessive position size or correlated exposure
        - REGIME_MISCLASSIFICATION: Detected regime differed from actual

        Args:
            trade_context: The closed trade with entry conditions.
            market_outcome: The actual market outcome after entry.

        Returns:
            The root-cause classification.
        """
        # Check regime misclassification first (most specific)
        if trade_context.regime != market_outcome.regime_actual:
            return MistakeClassification.REGIME_MISCLASSIFICATION

        # Check counter-trend entry
        entry_direction = trade_context.entry_conditions.get("direction", "")
        if entry_direction and entry_direction != market_outcome.actual_direction:
            return MistakeClassification.COUNTER_TREND

        # Check false breakout
        if not market_outcome.breakout_confirmed:
            return MistakeClassification.FALSE_BREAKOUT

        # Check volatility misjudgment (realized > 2x expected)
        expected_vol = trade_context.indicators.get("expected_volatility", 0.0)
        if expected_vol > 0 and market_outcome.volatility_realized > expected_vol * 2:
            return MistakeClassification.VOLATILITY_MISJUDGMENT

        # Check overexposure
        if trade_context.exit_reason in ("margin_call", "exposure_limit", "overexposure"):
            return MistakeClassification.OVEREXPOSURE

        # Check poor timing
        if not market_outcome.timing_optimal:
            return MistakeClassification.POOR_TIMING

        # Default to poor timing if no other classification matches
        return MistakeClassification.POOR_TIMING

    # ─── Recording (Req 21.1) ────────────────────────────────────────────

    def record_mistake(
        self, trade: ClosedTrade, classification: MistakeClassification
    ) -> MistakeRecord:
        """Create a structured mistake record for a losing trade.

        Must be called within 10 seconds of trade closure (Req 21.1).

        Args:
            trade: The closed losing trade.
            classification: The root-cause classification.

        Returns:
            The created MistakeRecord.
        """
        record = MistakeRecord(
            trade_id=trade.trade_id,
            classification=classification,
            entry_conditions=dict(trade.entry_conditions),
            regime=trade.regime,
            strategy=trade.strategy,
            indicators=dict(trade.indicators),
            confidence_at_entry=trade.confidence_at_entry,
            exit_reason=trade.exit_reason,
            pnl=trade.pnl,
            created_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Recorded mistake: trade_id=%s classification=%s pnl=%.2f",
            trade.trade_id,
            classification.value,
            trade.pnl,
        )
        return record

    # ─── Pattern Detection (Req 21.3) ────────────────────────────────────

    async def detect_patterns(self) -> list[MistakePattern]:
        """Check if any classification has 5+ occurrences in 30-day window.

        Scans all classifications and flags new patterns when the threshold
        is reached. Existing active patterns are updated with new loss counts.

        Returns:
            List of newly detected or updated MistakePattern instances.
        """
        since = datetime.now(timezone.utc) - timedelta(days=self.PATTERN_WINDOW_DAYS)
        new_patterns: list[MistakePattern] = []

        for classification in MistakeClassification:
            records = await self.mistake_db.get_records_by_classification(
                classification.value, since
            )

            if len(records) < self.PATTERN_THRESHOLD:
                continue

            # Check if pattern already exists for this classification
            existing = await self.mistake_db.get_pattern_by_classification(
                classification.value
            )

            if existing is not None:
                # Update existing pattern
                existing.loss_count = len(records)
                existing.last_occurrence = records[-1].created_at
                await self.mistake_db.update_pattern(
                    existing.id,
                    loss_count=len(records),
                    last_occurrence=records[-1].created_at,
                )
            else:
                # Check if there's a resolved pattern to reactivate
                all_patterns = await self.mistake_db.get_all_patterns()
                resolved_pattern = next(
                    (
                        p
                        for p in all_patterns
                        if p.classification == classification and not p.active
                    ),
                    None,
                )

                if resolved_pattern is not None:
                    # Reactivate with increased penalties (Req 21.8)
                    await self.reactivate_pattern(classification)
                else:
                    # Create new pattern
                    indicator_conditions = self._average_indicators(records)
                    pattern = MistakePattern(
                        id=str(uuid.uuid4()),
                        classification=classification,
                        loss_count=len(records),
                        first_occurrence=records[0].created_at,
                        last_occurrence=records[-1].created_at,
                        active=True,
                        reactivated=False,
                        confidence_penalty=MISTAKE_BASE_CONFIDENCE_PENALTY,
                        size_reduction=MISTAKE_BASE_SIZE_REDUCTION,
                        resolution_progress=0,
                        indicator_conditions=indicator_conditions,
                    )
                    await self.mistake_db.create_pattern(pattern)
                    self.active_patterns.append(pattern)
                    new_patterns.append(pattern)

                    logger.info(
                        "Detected new mistake pattern: classification=%s "
                        "loss_count=%d window=%d days",
                        classification.value,
                        len(records),
                        self.PATTERN_WINDOW_DAYS,
                    )

        return new_patterns

    # ─── Pattern Matching (Req 21.4) ─────────────────────────────────────

    def matches_pattern(self, signal: TradeSignal, pattern: MistakePattern) -> bool:
        """Check if a trade signal matches an active mistake pattern.

        Match criteria (all must be true):
        1. Same regime
        2. Same strategy type
        3. At least 3 of 5 indicator conditions match

        Args:
            signal: The trade signal to evaluate.
            pattern: The active mistake pattern to check against.

        Returns:
            True if the signal matches the pattern.
        """
        # Must be same regime
        if signal.regime != pattern.indicator_conditions.get("regime", signal.regime):
            # If pattern has a stored regime, check it
            pass

        # Check regime from pattern records (stored in indicator_conditions or inferred)
        pattern_regime = pattern.indicator_conditions.get("_regime")
        if pattern_regime and signal.regime != pattern_regime:
            return False

        # Check strategy type
        pattern_strategy = pattern.indicator_conditions.get("_strategy")
        if pattern_strategy and signal.strategy != pattern_strategy:
            return False

        # Check indicator conditions: at least 3 of 5 must match
        matching_indicators = 0
        total_checked = 0

        for indicator_name, pattern_value in pattern.indicator_conditions.items():
            # Skip internal metadata keys
            if indicator_name.startswith("_"):
                continue

            total_checked += 1
            if total_checked > MISTAKE_PATTERN_TOTAL_INDICATORS:
                break

            signal_value = signal.indicators.get(indicator_name)
            if signal_value is not None and self._indicator_matches(
                signal_value, pattern_value
            ):
                matching_indicators += 1

        return matching_indicators >= MISTAKE_PATTERN_MATCH_INDICATORS

    # ─── Penalty Application (Req 21.4, 21.5, 21.6) ─────────────────────

    def get_confidence_penalty(self, signal: TradeSignal) -> int:
        """Calculate total confidence penalty from all matching active patterns.

        Penalties apply to ALL signals including HFT (Cross-Cutting Rule 2).

        Base penalty: -20 per active pattern
        Reactivated penalty: -30 per reactivated pattern

        Args:
            signal: The trade signal to evaluate.

        Returns:
            Total confidence penalty (negative integer, e.g., -20, -50).
        """
        total_penalty = 0
        for pattern in self.active_patterns:
            if not pattern.active:
                continue
            if self.matches_pattern(signal, pattern):
                total_penalty += pattern.confidence_penalty
        return total_penalty

    def get_size_reduction_factor(self, signal: TradeSignal) -> float:
        """Calculate multiplicative size reduction factor from matching patterns.

        Factors are multiplied together per Cross-Cutting Rule 1.
        Base: 0.70 per active pattern (30% reduction)
        Reactivated: 0.50 per reactivated pattern (50% reduction)

        Args:
            signal: The trade signal to evaluate.

        Returns:
            Multiplicative factor (1.0 = no reduction, 0.7 = 30% reduction, etc.)
        """
        factor = 1.0
        for pattern in self.active_patterns:
            if not pattern.active:
                continue
            if self.matches_pattern(signal, pattern):
                factor *= pattern.size_reduction
        return factor

    # ─── Resolution Tracking (Req 21.7) ──────────────────────────────────

    async def update_resolution_progress(self, trade: ClosedTrade) -> None:
        """Update resolution progress for matching active patterns.

        - Profitable trade matching pattern → increment consecutive counter
        - Any loss matching pattern → reset counter to 0
        - At 20 consecutive profits → deactivate pattern (Req 21.7)

        Args:
            trade: The closed trade to evaluate.
        """
        signal = TradeSignal(
            regime=trade.regime,
            strategy=trade.strategy,
            indicators=trade.indicators,
            confidence=trade.confidence_at_entry,
        )

        for pattern in list(self.active_patterns):
            if not pattern.active:
                continue
            if not self.matches_pattern(signal, pattern):
                continue

            if trade.is_profitable:
                pattern.resolution_progress += 1
                logger.debug(
                    "Resolution progress: pattern=%s progress=%d/%d",
                    pattern.classification.value,
                    pattern.resolution_progress,
                    self.RESOLUTION_STREAK,
                )

                if pattern.resolution_progress >= self.RESOLUTION_STREAK:
                    # Pattern resolved! Deactivate it (Req 21.7)
                    pattern.active = False
                    await self.mistake_db.update_pattern_status(pattern.id, active=False)
                    self.active_patterns = [
                        p for p in self.active_patterns if p.id != pattern.id
                    ]
                    logger.info(
                        "Pattern resolved: classification=%s after %d consecutive profits",
                        pattern.classification.value,
                        self.RESOLUTION_STREAK,
                    )
                else:
                    await self.mistake_db.update_pattern(
                        pattern.id,
                        resolution_progress=pattern.resolution_progress,
                    )
            else:
                # Any loss resets counter to 0 (Req 21.7)
                pattern.resolution_progress = 0
                await self.mistake_db.update_pattern(
                    pattern.id, resolution_progress=0
                )
                logger.debug(
                    "Resolution progress reset: pattern=%s (loss detected)",
                    pattern.classification.value,
                )

    # ─── Reactivation (Req 21.8) ─────────────────────────────────────────

    async def reactivate_pattern(self, classification: MistakeClassification) -> None:
        """Reactivate a previously resolved pattern with increased penalties.

        When a resolved pattern recurs (5 new losses within 30 days after
        deactivation), it is reactivated with harsher penalties:
        - Confidence penalty: -30 (was -20)
        - Size reduction: 0.50 (was 0.70)

        Args:
            classification: The classification to reactivate.
        """
        all_patterns = await self.mistake_db.get_all_patterns()
        resolved_pattern = next(
            (
                p
                for p in all_patterns
                if p.classification == classification and not p.active
            ),
            None,
        )

        if resolved_pattern is not None:
            resolved_pattern.active = True
            resolved_pattern.reactivated = True
            resolved_pattern.confidence_penalty = MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
            resolved_pattern.size_reduction = MISTAKE_REACTIVATED_SIZE_REDUCTION
            resolved_pattern.resolution_progress = 0

            await self.mistake_db.update_pattern(
                resolved_pattern.id,
                active=True,
                reactivated=True,
                confidence_penalty=MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
                size_reduction=MISTAKE_REACTIVATED_SIZE_REDUCTION,
                resolution_progress=0,
            )

            # Add to active patterns list
            if resolved_pattern not in self.active_patterns:
                self.active_patterns.append(resolved_pattern)

            logger.info(
                "Reactivated pattern: classification=%s "
                "confidence_penalty=%d size_reduction=%.2f",
                classification.value,
                MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
                MISTAKE_REACTIVATED_SIZE_REDUCTION,
            )
        else:
            logger.warning(
                "No resolved pattern found to reactivate: classification=%s",
                classification.value,
            )

    # ─── Startup Loading (Req 21.10) ─────────────────────────────────────

    async def load_patterns_on_startup(self) -> None:
        """Load all active patterns from DB and apply immediately.

        No warm-up period — penalties take effect as soon as patterns are loaded.
        Called during application startup.
        """
        self.active_patterns = await self.mistake_db.get_active_patterns()
        logger.info(
            "Loaded %d active mistake patterns on startup",
            len(self.active_patterns),
        )

    # ─── Dashboard API Exposure (Req 21.9) ───────────────────────────────

    def get_dashboard_patterns(self) -> list[dict[str, Any]]:
        """Get active patterns formatted for Dashboard API exposure.

        Returns pattern data including classification, loss count,
        last occurrence, penalty level, and resolution progress.

        Returns:
            List of pattern dictionaries for the dashboard.
        """
        return [
            {
                "id": pattern.id,
                "classification": pattern.classification.value,
                "loss_count": pattern.loss_count,
                "first_occurrence": pattern.first_occurrence.isoformat(),
                "last_occurrence": pattern.last_occurrence.isoformat(),
                "confidence_penalty": pattern.confidence_penalty,
                "size_reduction": pattern.size_reduction,
                "resolution_progress": pattern.resolution_progress,
                "resolution_target": self.RESOLUTION_STREAK,
                "reactivated": pattern.reactivated,
                "active": pattern.active,
            }
            for pattern in self.active_patterns
            if pattern.active
        ]

    # ─── Internal Helpers ────────────────────────────────────────────────

    def _average_indicators(self, records: list[MistakeRecord]) -> dict[str, float]:
        """Average indicator values across multiple mistake records.

        Also stores regime and strategy metadata with underscore prefix.

        Args:
            records: List of mistake records to average.

        Returns:
            Dict of averaged indicator values plus metadata.
        """
        if not records:
            return {}

        # Collect all indicator keys
        all_keys: set[str] = set()
        for record in records:
            all_keys.update(record.indicators.keys())

        # Average each indicator
        averaged: dict[str, float] = {}
        for key in all_keys:
            values = [
                record.indicators[key]
                for record in records
                if key in record.indicators
            ]
            if values:
                averaged[key] = sum(values) / len(values)

        # Store regime and strategy metadata
        regimes = [r.regime for r in records if r.regime]
        if regimes:
            # Use most common regime
            from collections import Counter

            regime_counts = Counter(regimes)
            averaged["_regime"] = regime_counts.most_common(1)[0][0]  # type: ignore[assignment]

        strategies = [r.strategy for r in records if r.strategy]
        if strategies:
            from collections import Counter

            strategy_counts = Counter(strategies)
            averaged["_strategy"] = strategy_counts.most_common(1)[0][0]  # type: ignore[assignment]

        return averaged

    @staticmethod
    def _indicator_matches(signal_value: float, pattern_value: float) -> bool:
        """Check if a signal indicator value matches a pattern's averaged value.

        Uses a 20% tolerance band around the pattern value.

        Args:
            signal_value: The indicator value from the current signal.
            pattern_value: The averaged indicator value from the pattern.

        Returns:
            True if the values are within 20% tolerance.
        """
        if pattern_value == 0:
            return abs(signal_value) < 0.01

        tolerance = abs(pattern_value) * 0.20
        return abs(signal_value - pattern_value) <= tolerance
