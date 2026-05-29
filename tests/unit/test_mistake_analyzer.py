"""Unit tests for MistakeDatabase, MistakeAnalyzer, and pattern lifecycle.

Tests cover:
- Tasks 27.1-27.3: Mistake Database and Recording
- Tasks 28.1-28.5: Mistake Pattern Detection and Penalties
- Tasks 29.1-29.5: Pattern Lifecycle and Resolution
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

# Set required env vars before importing settings-dependent modules
os.environ.setdefault("IG_API_KEY", "test_key")
os.environ.setdefault("IG_USERNAME", "test_user")
os.environ.setdefault("IG_PASSWORD", "test_pass")

import pytest

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
from src.learning.mistake_analyzer import (
    ClosedTrade,
    MarketOutcome,
    MistakeAnalyzer,
    TradeSignal,
)
from src.learning.mistake_database import (
    MistakeClassification,
    MistakeDatabase,
    MistakePattern,
    MistakeRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mistake_db() -> MistakeDatabase:
    """Create a fresh MistakeDatabase instance."""
    return MistakeDatabase()


@pytest.fixture
def analyzer(mistake_db: MistakeDatabase) -> MistakeAnalyzer:
    """Create a MistakeAnalyzer with a fresh database."""
    return MistakeAnalyzer(mistake_db)


def _make_record(
    classification: MistakeClassification = MistakeClassification.COUNTER_TREND,
    regime: str = "trending",
    strategy: str = "trend_following",
    pnl: float = -50.0,
    indicators: dict | None = None,
    created_at: datetime | None = None,
) -> MistakeRecord:
    """Helper to create a MistakeRecord with defaults."""
    return MistakeRecord(
        trade_id=str(uuid.uuid4()),
        classification=classification,
        entry_conditions={"direction": "up", "ma_cross": "bearish"},
        regime=regime,
        strategy=strategy,
        indicators=indicators or {"rsi": 72.0, "adx": 35.0, "atr": 1.5, "macd": 0.5, "bb_width": 2.0},
        confidence_at_entry=75,
        exit_reason="stop_loss_hit",
        pnl=pnl,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _make_pattern(
    classification: MistakeClassification = MistakeClassification.COUNTER_TREND,
    active: bool = True,
    reactivated: bool = False,
    confidence_penalty: int = MISTAKE_BASE_CONFIDENCE_PENALTY,
    size_reduction: float = MISTAKE_BASE_SIZE_REDUCTION,
    resolution_progress: int = 0,
    regime: str = "trending",
    strategy: str = "trend_following",
) -> MistakePattern:
    """Helper to create a MistakePattern with defaults."""
    return MistakePattern(
        id=str(uuid.uuid4()),
        classification=classification,
        loss_count=5,
        first_occurrence=datetime.now(timezone.utc) - timedelta(days=20),
        last_occurrence=datetime.now(timezone.utc),
        active=active,
        reactivated=reactivated,
        confidence_penalty=confidence_penalty,
        size_reduction=size_reduction,
        resolution_progress=resolution_progress,
        indicator_conditions={
            "rsi": 72.0,
            "adx": 35.0,
            "atr": 1.5,
            "macd": 0.5,
            "bb_width": 2.0,
            "_regime": regime,
            "_strategy": strategy,
        },
    )


def _make_signal(
    regime: str = "trending",
    strategy: str = "trend_following",
    indicators: dict | None = None,
    confidence: int = 80,
    is_hft: bool = False,
) -> TradeSignal:
    """Helper to create a TradeSignal with defaults."""
    return TradeSignal(
        regime=regime,
        strategy=strategy,
        indicators=indicators or {"rsi": 72.0, "adx": 35.0, "atr": 1.5, "macd": 0.5, "bb_width": 2.0},
        confidence=confidence,
        is_hft=is_hft,
    )


def _make_closed_trade(
    regime: str = "trending",
    strategy: str = "trend_following",
    pnl: float = 100.0,
    indicators: dict | None = None,
) -> ClosedTrade:
    """Helper to create a ClosedTrade with defaults."""
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        regime=regime,
        strategy=strategy,
        indicators=indicators or {"rsi": 72.0, "adx": 35.0, "atr": 1.5, "macd": 0.5, "bb_width": 2.0},
        confidence_at_entry=75,
        exit_reason="take_profit",
        pnl=pnl,
        entry_conditions={"direction": "up"},
    )


# ===========================================================================
# Task 27.1-27.3: Mistake Database and Recording
# ===========================================================================


class TestMistakeDatabase:
    """Tests for MistakeDatabase async storage and retrieval."""

    async def test_store_record(self, mistake_db: MistakeDatabase):
        record = _make_record()
        await mistake_db.store_record(record)
        assert len(mistake_db.records) == 1
        assert mistake_db.records[0].trade_id == record.trade_id

    async def test_get_records_by_classification(self, mistake_db: MistakeDatabase):
        record = _make_record(classification=MistakeClassification.FALSE_BREAKOUT)
        await mistake_db.store_record(record)
        since = datetime.now(timezone.utc) - timedelta(days=30)
        results = await mistake_db.get_records_by_classification("false_breakout", since)
        assert len(results) == 1
        assert results[0].classification == MistakeClassification.FALSE_BREAKOUT

    async def test_get_records_excludes_other_classifications(self, mistake_db: MistakeDatabase):
        await mistake_db.store_record(_make_record(classification=MistakeClassification.COUNTER_TREND))
        since = datetime.now(timezone.utc) - timedelta(days=30)
        results = await mistake_db.get_records_by_classification("false_breakout", since)
        assert len(results) == 0

    async def test_get_records_excludes_old_records(self, mistake_db: MistakeDatabase):
        old_record = _make_record(
            created_at=datetime.now(timezone.utc) - timedelta(days=60)
        )
        await mistake_db.store_record(old_record)
        since = datetime.now(timezone.utc) - timedelta(days=30)
        results = await mistake_db.get_records_by_classification("counter_trend_entry", since)
        assert len(results) == 0

    async def test_get_active_patterns(self, mistake_db: MistakeDatabase):
        pattern = _make_pattern(active=True)
        await mistake_db.create_pattern(pattern)
        active = await mistake_db.get_active_patterns()
        assert len(active) == 1
        assert active[0].id == pattern.id

    async def test_get_active_patterns_excludes_inactive(self, mistake_db: MistakeDatabase):
        pattern = _make_pattern(active=False)
        await mistake_db.create_pattern(pattern)
        active = await mistake_db.get_active_patterns()
        assert len(active) == 0

    async def test_update_pattern_status(self, mistake_db: MistakeDatabase):
        pattern = _make_pattern(active=True)
        await mistake_db.create_pattern(pattern)
        await mistake_db.update_pattern_status(pattern.id, active=False)
        active = await mistake_db.get_active_patterns()
        assert len(active) == 0


class TestMistakeClassification:
    """Tests for root-cause classification (Task 27.3)."""

    def test_classify_counter_trend(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-50.0)
        trade.entry_conditions = {"direction": "up"}
        outcome = MarketOutcome(
            actual_direction="down",
            volatility_realized=1.0,
            regime_actual="trending",
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.COUNTER_TREND

    def test_classify_false_breakout(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-30.0)
        trade.entry_conditions = {"direction": "up"}
        outcome = MarketOutcome(
            actual_direction="up",
            volatility_realized=1.0,
            regime_actual="trending",
            breakout_confirmed=False,
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.FALSE_BREAKOUT

    def test_classify_volatility_misjudgment(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-80.0)
        trade.indicators = {"expected_volatility": 1.0, "rsi": 50.0, "adx": 25.0, "atr": 1.0, "macd": 0.0}
        trade.entry_conditions = {"direction": "up"}
        outcome = MarketOutcome(
            actual_direction="up",
            volatility_realized=3.0,  # > 2x expected
            regime_actual="trending",
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.VOLATILITY_MISJUDGMENT

    def test_classify_regime_misclassification(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(regime="trending", pnl=-40.0)
        outcome = MarketOutcome(
            actual_direction="up",
            volatility_realized=1.0,
            regime_actual="ranging",  # Different from trade's regime
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.REGIME_MISCLASSIFICATION

    def test_classify_overexposure(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-100.0)
        trade.exit_reason = "exposure_limit"
        trade.entry_conditions = {"direction": "up"}
        outcome = MarketOutcome(
            actual_direction="up",
            volatility_realized=1.0,
            regime_actual="trending",
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.OVEREXPOSURE

    def test_classify_poor_timing(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-20.0)
        trade.entry_conditions = {"direction": "up"}
        outcome = MarketOutcome(
            actual_direction="up",
            volatility_realized=1.0,
            regime_actual="trending",
            timing_optimal=False,
        )
        result = analyzer.classify_mistake(trade, outcome)
        assert result == MistakeClassification.POOR_TIMING

    def test_all_six_classifications_exist(self):
        assert len(MistakeClassification) == 6
        values = {c.value for c in MistakeClassification}
        expected = {
            "counter_trend_entry",
            "false_breakout",
            "volatility_misjudgment",
            "poor_timing",
            "overexposure",
            "regime_misclassification",
        }
        assert values == expected


class TestRecordMistake:
    """Tests for mistake record creation (Task 27.2)."""

    def test_record_mistake_creates_record(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-50.0)
        record = analyzer.record_mistake(trade, MistakeClassification.COUNTER_TREND)
        assert record.trade_id == trade.trade_id
        assert record.classification == MistakeClassification.COUNTER_TREND
        assert record.pnl == -50.0
        assert record.regime == "trending"
        assert record.strategy == "trend_following"

    def test_record_mistake_captures_indicators(self, analyzer: MistakeAnalyzer):
        trade = _make_closed_trade(pnl=-30.0)
        record = analyzer.record_mistake(trade, MistakeClassification.FALSE_BREAKOUT)
        assert "rsi" in record.indicators
        assert record.indicators["rsi"] == 72.0


# ===========================================================================
# Task 28.1-28.5: Mistake Pattern Detection and Penalties
# ===========================================================================


class TestPatternDetection:
    """Tests for pattern detection logic (Task 28.1)."""

    async def test_detect_pattern_at_threshold(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """5 losses with same classification in 30 days → pattern flagged."""
        for _ in range(MISTAKE_PATTERN_THRESHOLD):
            await mistake_db.store_record(_make_record())

        patterns = await analyzer.detect_patterns()
        assert len(patterns) == 1
        assert patterns[0].classification == MistakeClassification.COUNTER_TREND
        assert patterns[0].loss_count == MISTAKE_PATTERN_THRESHOLD
        assert patterns[0].active is True

    async def test_no_pattern_below_threshold(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """4 losses should not trigger a pattern."""
        for _ in range(MISTAKE_PATTERN_THRESHOLD - 1):
            await mistake_db.store_record(_make_record())

        patterns = await analyzer.detect_patterns()
        assert len(patterns) == 0

    async def test_pattern_uses_30_day_window(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Records older than 30 days should not count."""
        # 3 recent records
        for _ in range(3):
            await mistake_db.store_record(_make_record())
        # 3 old records (outside window)
        for _ in range(3):
            await mistake_db.store_record(
                _make_record(created_at=datetime.now(timezone.utc) - timedelta(days=45))
            )

        patterns = await analyzer.detect_patterns()
        assert len(patterns) == 0

    async def test_pattern_stores_averaged_indicators(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Pattern should store averaged indicator conditions."""
        for _ in range(MISTAKE_PATTERN_THRESHOLD):
            await mistake_db.store_record(_make_record())

        patterns = await analyzer.detect_patterns()
        assert len(patterns) == 1
        assert "rsi" in patterns[0].indicator_conditions
        assert patterns[0].indicator_conditions["rsi"] == pytest.approx(72.0)


class TestPatternMatching:
    """Tests for pattern matching logic (Task 28.2)."""

    def test_matches_same_regime_strategy_indicators(self, analyzer: MistakeAnalyzer):
        """Signal with same regime, strategy, and 3+ matching indicators → match."""
        pattern = _make_pattern()
        analyzer.active_patterns = [pattern]
        signal = _make_signal()
        assert analyzer.matches_pattern(signal, pattern) is True

    def test_no_match_different_regime(self, analyzer: MistakeAnalyzer):
        """Different regime → no match."""
        pattern = _make_pattern(regime="trending")
        signal = _make_signal(regime="ranging")
        assert analyzer.matches_pattern(signal, pattern) is False

    def test_no_match_different_strategy(self, analyzer: MistakeAnalyzer):
        """Different strategy → no match."""
        pattern = _make_pattern(strategy="trend_following")
        signal = _make_signal(strategy="mean_reversion")
        assert analyzer.matches_pattern(signal, pattern) is False

    def test_no_match_fewer_than_3_indicators(self, analyzer: MistakeAnalyzer):
        """Fewer than 3 matching indicators → no match."""
        pattern = _make_pattern()
        # Signal with completely different indicator values
        signal = _make_signal(indicators={
            "rsi": 20.0,  # very different from 72.0
            "adx": 10.0,  # very different from 35.0
            "atr": 10.0,  # very different from 1.5
            "macd": 5.0,  # very different from 0.5
            "bb_width": 10.0,  # very different from 2.0
        })
        assert analyzer.matches_pattern(signal, pattern) is False

    def test_matches_with_exactly_3_indicators(self, analyzer: MistakeAnalyzer):
        """Exactly 3 matching indicators → match."""
        pattern = _make_pattern()
        # 3 matching, 2 not matching
        signal = _make_signal(indicators={
            "rsi": 72.0,  # match
            "adx": 35.0,  # match
            "atr": 1.5,   # match
            "macd": 5.0,  # no match
            "bb_width": 10.0,  # no match
        })
        assert analyzer.matches_pattern(signal, pattern) is True


class TestConfidencePenalty:
    """Tests for confidence penalty application (Task 28.3)."""

    def test_base_penalty_for_active_pattern(self, analyzer: MistakeAnalyzer):
        """Active pattern applies -20 confidence penalty."""
        pattern = _make_pattern(confidence_penalty=MISTAKE_BASE_CONFIDENCE_PENALTY)
        analyzer.active_patterns = [pattern]
        signal = _make_signal()
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == MISTAKE_BASE_CONFIDENCE_PENALTY

    def test_reactivated_penalty(self, analyzer: MistakeAnalyzer):
        """Reactivated pattern applies -30 confidence penalty."""
        pattern = _make_pattern(
            reactivated=True,
            confidence_penalty=MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
        )
        analyzer.active_patterns = [pattern]
        signal = _make_signal()
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == MISTAKE_REACTIVATED_CONFIDENCE_PENALTY

    def test_multiple_patterns_stack_penalties(self, analyzer: MistakeAnalyzer):
        """Multiple matching patterns stack their penalties."""
        pattern1 = _make_pattern(
            classification=MistakeClassification.COUNTER_TREND,
            confidence_penalty=MISTAKE_BASE_CONFIDENCE_PENALTY,
        )
        pattern2 = _make_pattern(
            classification=MistakeClassification.FALSE_BREAKOUT,
            confidence_penalty=MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
        )
        # Both patterns match the same signal conditions
        analyzer.active_patterns = [pattern1, pattern2]
        signal = _make_signal()
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == MISTAKE_BASE_CONFIDENCE_PENALTY + MISTAKE_REACTIVATED_CONFIDENCE_PENALTY

    def test_no_penalty_when_no_match(self, analyzer: MistakeAnalyzer):
        """No penalty when signal doesn't match any pattern."""
        pattern = _make_pattern(regime="trending")
        analyzer.active_patterns = [pattern]
        signal = _make_signal(regime="ranging")
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == 0

    def test_penalty_applies_to_hft_signals(self, analyzer: MistakeAnalyzer):
        """HFT signals are NOT exempt from mistake penalties (Cross-Cutting Rule 2)."""
        pattern = _make_pattern()
        analyzer.active_patterns = [pattern]
        signal = _make_signal(is_hft=True)
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == MISTAKE_BASE_CONFIDENCE_PENALTY


class TestSizeReduction:
    """Tests for position size reduction factor (Task 28.4)."""

    def test_base_size_reduction(self, analyzer: MistakeAnalyzer):
        """Active pattern applies 0.70 size reduction."""
        pattern = _make_pattern(size_reduction=MISTAKE_BASE_SIZE_REDUCTION)
        analyzer.active_patterns = [pattern]
        signal = _make_signal()
        factor = analyzer.get_size_reduction_factor(signal)
        assert factor == pytest.approx(MISTAKE_BASE_SIZE_REDUCTION)

    def test_reactivated_size_reduction(self, analyzer: MistakeAnalyzer):
        """Reactivated pattern applies 0.50 size reduction."""
        pattern = _make_pattern(
            reactivated=True,
            size_reduction=MISTAKE_REACTIVATED_SIZE_REDUCTION,
        )
        analyzer.active_patterns = [pattern]
        signal = _make_signal()
        factor = analyzer.get_size_reduction_factor(signal)
        assert factor == pytest.approx(MISTAKE_REACTIVATED_SIZE_REDUCTION)

    def test_multiple_patterns_multiply_reductions(self, analyzer: MistakeAnalyzer):
        """Multiple matching patterns multiply their size reductions."""
        pattern1 = _make_pattern(
            classification=MistakeClassification.COUNTER_TREND,
            size_reduction=MISTAKE_BASE_SIZE_REDUCTION,
        )
        pattern2 = _make_pattern(
            classification=MistakeClassification.FALSE_BREAKOUT,
            size_reduction=MISTAKE_REACTIVATED_SIZE_REDUCTION,
        )
        analyzer.active_patterns = [pattern1, pattern2]
        signal = _make_signal()
        factor = analyzer.get_size_reduction_factor(signal)
        expected = MISTAKE_BASE_SIZE_REDUCTION * MISTAKE_REACTIVATED_SIZE_REDUCTION
        assert factor == pytest.approx(expected)

    def test_no_reduction_when_no_match(self, analyzer: MistakeAnalyzer):
        """No reduction (factor=1.0) when signal doesn't match any pattern."""
        pattern = _make_pattern(regime="trending")
        analyzer.active_patterns = [pattern]
        signal = _make_signal(regime="ranging")
        factor = analyzer.get_size_reduction_factor(signal)
        assert factor == pytest.approx(1.0)

    def test_size_reduction_applies_to_hft(self, analyzer: MistakeAnalyzer):
        """HFT signals are NOT exempt from size reduction (Cross-Cutting Rule 2)."""
        pattern = _make_pattern()
        analyzer.active_patterns = [pattern]
        signal = _make_signal(is_hft=True)
        factor = analyzer.get_size_reduction_factor(signal)
        assert factor == pytest.approx(MISTAKE_BASE_SIZE_REDUCTION)


# ===========================================================================
# Task 29.1-29.5: Pattern Lifecycle and Resolution
# ===========================================================================


class TestResolutionTracking:
    """Tests for resolution tracking (Task 29.1, 29.2)."""

    async def test_profitable_trade_increments_progress(self, analyzer: MistakeAnalyzer):
        """Profitable trade matching pattern increments resolution counter."""
        pattern = _make_pattern(resolution_progress=0)
        analyzer.active_patterns = [pattern]
        await analyzer.mistake_db.create_pattern(pattern)

        trade = _make_closed_trade(pnl=100.0)
        await analyzer.update_resolution_progress(trade)
        assert pattern.resolution_progress == 1

    async def test_20_consecutive_profits_deactivates_pattern(self, analyzer: MistakeAnalyzer):
        """20 consecutive profitable trades → deactivate pattern (Task 29.1)."""
        pattern = _make_pattern(resolution_progress=MISTAKE_RESOLUTION_STREAK - 1)
        analyzer.active_patterns = [pattern]
        await analyzer.mistake_db.create_pattern(pattern)

        trade = _make_closed_trade(pnl=50.0)
        await analyzer.update_resolution_progress(trade)

        # Pattern should be deactivated
        assert pattern.active is False
        assert len(analyzer.active_patterns) == 0

    async def test_loss_resets_counter_to_zero(self, analyzer: MistakeAnalyzer):
        """Any loss resets resolution counter to 0 (Task 29.2)."""
        pattern = _make_pattern(resolution_progress=15)
        analyzer.active_patterns = [pattern]
        await analyzer.mistake_db.create_pattern(pattern)

        trade = _make_closed_trade(pnl=-20.0)  # Loss
        await analyzer.update_resolution_progress(trade)
        assert pattern.resolution_progress == 0

    async def test_non_matching_trade_does_not_affect_progress(self, analyzer: MistakeAnalyzer):
        """Trade that doesn't match pattern conditions has no effect."""
        pattern = _make_pattern(regime="trending", resolution_progress=5)
        analyzer.active_patterns = [pattern]
        await analyzer.mistake_db.create_pattern(pattern)

        # Trade with different regime
        trade = _make_closed_trade(regime="ranging", pnl=100.0)
        await analyzer.update_resolution_progress(trade)
        assert pattern.resolution_progress == 5  # Unchanged


class TestPatternReactivation:
    """Tests for pattern reactivation (Task 29.3)."""

    async def test_reactivation_with_increased_penalties(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Reactivated pattern gets -30 confidence and 0.50 size reduction."""
        # Create a resolved pattern
        pattern = _make_pattern(
            active=False,
            confidence_penalty=MISTAKE_BASE_CONFIDENCE_PENALTY,
            size_reduction=MISTAKE_BASE_SIZE_REDUCTION,
        )
        await mistake_db.create_pattern(pattern)

        await analyzer.reactivate_pattern(MistakeClassification.COUNTER_TREND)

        # Check the pattern was reactivated with increased penalties
        reactivated = await mistake_db.get_pattern_by_classification("counter_trend_entry")
        assert reactivated is not None
        assert reactivated.active is True
        assert reactivated.reactivated is True
        assert reactivated.confidence_penalty == MISTAKE_REACTIVATED_CONFIDENCE_PENALTY
        assert reactivated.size_reduction == MISTAKE_REACTIVATED_SIZE_REDUCTION
        assert reactivated.resolution_progress == 0

    async def test_reactivation_adds_to_active_patterns(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Reactivated pattern is added to active_patterns list."""
        pattern = _make_pattern(active=False)
        await mistake_db.create_pattern(pattern)

        await analyzer.reactivate_pattern(MistakeClassification.COUNTER_TREND)
        assert len(analyzer.active_patterns) == 1
        assert analyzer.active_patterns[0].reactivated is True


class TestStartupLoading:
    """Tests for startup pattern loading (Task 29.4)."""

    async def test_load_patterns_on_startup(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Startup loads all active patterns immediately."""
        pattern1 = _make_pattern(classification=MistakeClassification.COUNTER_TREND)
        pattern2 = _make_pattern(classification=MistakeClassification.FALSE_BREAKOUT)
        await mistake_db.create_pattern(pattern1)
        await mistake_db.create_pattern(pattern2)

        await analyzer.load_patterns_on_startup()
        assert len(analyzer.active_patterns) == 2

    async def test_startup_excludes_inactive_patterns(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Startup does not load inactive patterns."""
        active = _make_pattern(active=True)
        inactive = _make_pattern(
            classification=MistakeClassification.FALSE_BREAKOUT,
            active=False,
        )
        await mistake_db.create_pattern(active)
        await mistake_db.create_pattern(inactive)

        await analyzer.load_patterns_on_startup()
        assert len(analyzer.active_patterns) == 1
        assert analyzer.active_patterns[0].classification == MistakeClassification.COUNTER_TREND

    async def test_startup_applies_immediately_no_warmup(self, analyzer: MistakeAnalyzer, mistake_db: MistakeDatabase):
        """Loaded patterns apply penalties immediately (no warm-up)."""
        pattern = _make_pattern()
        await mistake_db.create_pattern(pattern)

        await analyzer.load_patterns_on_startup()

        # Penalty should be active immediately
        signal = _make_signal()
        penalty = analyzer.get_confidence_penalty(signal)
        assert penalty == MISTAKE_BASE_CONFIDENCE_PENALTY


class TestDashboardExposure:
    """Tests for Dashboard API exposure (Task 29.5)."""

    def test_get_dashboard_patterns_returns_active(self, analyzer: MistakeAnalyzer):
        """Dashboard returns active patterns with required fields."""
        pattern = _make_pattern(resolution_progress=5)
        analyzer.active_patterns = [pattern]

        dashboard_data = analyzer.get_dashboard_patterns()
        assert len(dashboard_data) == 1
        data = dashboard_data[0]
        assert data["classification"] == "counter_trend_entry"
        assert data["loss_count"] == 5
        assert data["confidence_penalty"] == MISTAKE_BASE_CONFIDENCE_PENALTY
        assert data["size_reduction"] == MISTAKE_BASE_SIZE_REDUCTION
        assert data["resolution_progress"] == 5
        assert data["resolution_target"] == MISTAKE_RESOLUTION_STREAK
        assert data["active"] is True
        assert "last_occurrence" in data
        assert "first_occurrence" in data

    def test_get_dashboard_patterns_excludes_inactive(self, analyzer: MistakeAnalyzer):
        """Dashboard does not expose inactive patterns."""
        pattern = _make_pattern(active=False)
        analyzer.active_patterns = [pattern]

        dashboard_data = analyzer.get_dashboard_patterns()
        assert len(dashboard_data) == 0

    def test_get_dashboard_patterns_empty_when_no_patterns(self, analyzer: MistakeAnalyzer):
        """Dashboard returns empty list when no active patterns."""
        dashboard_data = analyzer.get_dashboard_patterns()
        assert dashboard_data == []


class TestConstants:
    """Verify constants are correctly defined."""

    def test_pattern_threshold(self):
        assert MISTAKE_PATTERN_THRESHOLD == 5

    def test_pattern_window_days(self):
        assert MISTAKE_PATTERN_WINDOW_DAYS == 30

    def test_resolution_streak(self):
        assert MISTAKE_RESOLUTION_STREAK == 20

    def test_base_confidence_penalty(self):
        assert MISTAKE_BASE_CONFIDENCE_PENALTY == 20

    def test_reactivated_confidence_penalty(self):
        assert MISTAKE_REACTIVATED_CONFIDENCE_PENALTY == 30

    def test_base_size_reduction(self):
        assert MISTAKE_BASE_SIZE_REDUCTION == 0.70

    def test_reactivated_size_reduction(self):
        assert MISTAKE_REACTIVATED_SIZE_REDUCTION == 0.50

    def test_pattern_match_indicators(self):
        assert MISTAKE_PATTERN_MATCH_INDICATORS == 3

    def test_pattern_total_indicators(self):
        assert MISTAKE_PATTERN_TOTAL_INDICATORS == 5
