"""Unit tests for the OvertradingGuard module.

Tests cover daily trade limits, minimum time intervals, consecutive loss
cooldowns, win rate throttling, and HFT bypass.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, Cross-Cutting Rule 2
"""

from datetime import datetime, timedelta

import pytest

from src.config.constants import (
    CONFIDENCE_THRESHOLD_ELEVATED,
    CONSECUTIVE_LOSS_COOLDOWN_HOURS,
    CONSECUTIVE_LOSS_THRESHOLD,
    MAX_TRADES_PER_DAY_DEFAULT,
    MIN_TRADE_INTERVAL_MINUTES,
    WIN_RATE_THROTTLE_THRESHOLD,
)
from src.strategy.overtrading_guard import OvertradingGuard, TradeDecision


@pytest.fixture
def guard() -> OvertradingGuard:
    """Create a fresh OvertradingGuard with default settings."""
    return OvertradingGuard()


@pytest.fixture
def base_time() -> datetime:
    """A fixed base time for testing."""
    return datetime(2024, 1, 15, 10, 0, 0)


# =============================================================================
# Task 17.1: Daily trade count limits (max 10, configurable 1-100)
# =============================================================================


class TestDailyTradeLimit:
    """Tests for daily trade count enforcement (Task 17.1)."""

    def test_default_max_trades_is_10(self, guard: OvertradingGuard) -> None:
        """Default max trades per day should be 10."""
        assert guard.max_trades_per_day == MAX_TRADES_PER_DAY_DEFAULT
        assert guard.max_trades_per_day == 10

    def test_first_trade_allowed(self, guard: OvertradingGuard, base_time: datetime) -> None:
        """First trade should always be allowed."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_trade_blocked_at_limit(self, guard: OvertradingGuard, base_time: datetime) -> None:
        """Trade should be blocked when daily limit is reached."""
        # Record 10 trades (at different times to avoid interval check)
        for i in range(10):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # 11th trade should be blocked
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=66),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False
        assert decision.reason is not None
        assert "limit" in decision.reason.lower()

    def test_different_instruments_have_separate_counts(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Trade counts are per-instrument per-strategy."""
        for i in range(10):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # Different instrument should still be allowed
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="GBPUSD",
            current_time=base_time + timedelta(minutes=66),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_different_strategies_have_separate_counts(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Trade counts are per-strategy."""
        for i in range(10):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # Different strategy should still be allowed
        decision = guard.can_trade(
            strategy_name="mean_reversion",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=66),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_configurable_max_trades(self, base_time: datetime) -> None:
        """Max trades should be configurable."""
        guard = OvertradingGuard(max_trades_per_day=5)
        assert guard.max_trades_per_day == 5

        for i in range(5):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=36),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False

    def test_max_trades_clamped_to_1_minimum(self) -> None:
        """Max trades should be at least 1."""
        guard = OvertradingGuard(max_trades_per_day=0)
        assert guard.max_trades_per_day == 1

    def test_max_trades_clamped_to_100_maximum(self) -> None:
        """Max trades should be at most 100."""
        guard = OvertradingGuard(max_trades_per_day=200)
        assert guard.max_trades_per_day == 100

    def test_reset_daily_clears_counts(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """reset_daily should clear all trade counts."""
        for i in range(10):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        guard.reset_daily()

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(days=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_get_trade_count(self, guard: OvertradingGuard, base_time: datetime) -> None:
        """get_trade_count should return current count."""
        assert guard.get_trade_count("momentum", "EURUSD") == 0
        guard.record_trade("momentum", "EURUSD", base_time)
        assert guard.get_trade_count("momentum", "EURUSD") == 1


# =============================================================================
# Task 17.2: Minimum time interval (5 minutes between trades on same instrument)
# =============================================================================


class TestMinimumTimeInterval:
    """Tests for minimum 5-minute interval enforcement (Task 17.2)."""

    def test_trade_blocked_within_5_minutes(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Trade within 5 minutes of last trade on same instrument should be blocked."""
        guard.record_trade("momentum", "EURUSD", base_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=3),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False
        assert decision.reason is not None
        assert "interval" in decision.reason.lower()

    def test_trade_allowed_after_5_minutes(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Trade after 5 minutes should be allowed."""
        guard.record_trade("momentum", "EURUSD", base_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=5),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_trade_allowed_at_exactly_5_minutes(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Trade at exactly 5 minutes should be allowed (elapsed >= interval)."""
        guard.record_trade("momentum", "EURUSD", base_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=MIN_TRADE_INTERVAL_MINUTES),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_interval_is_per_instrument(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Interval check is per-instrument, not global."""
        guard.record_trade("momentum", "EURUSD", base_time)

        # Different instrument should not be blocked
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="GBPUSD",
            current_time=base_time + timedelta(minutes=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_interval_shared_across_strategies(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Interval is per-instrument regardless of strategy."""
        guard.record_trade("momentum", "EURUSD", base_time)

        # Same instrument, different strategy should still be blocked
        decision = guard.can_trade(
            strategy_name="mean_reversion",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=2),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False

    def test_remaining_time_in_reason(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Reason should include remaining time."""
        guard.record_trade("momentum", "EURUSD", base_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=2),
            recent_win_rate=0.5,
        )
        assert decision.reason is not None
        assert "remaining" in decision.reason.lower()


# =============================================================================
# Task 17.3: Consecutive loss cooldown (3 losses → 1-hour cooldown)
# =============================================================================


class TestConsecutiveLossCooldown:
    """Tests for consecutive loss cooldown (Task 17.3)."""

    def test_no_cooldown_with_fewer_than_3_losses(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Fewer than 3 consecutive losses should not trigger cooldown."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_cooldown_triggered_after_3_losses(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """3 consecutive losses should trigger 1-hour cooldown."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        # Use a time shortly after the cooldown was set (record_loss uses utcnow)
        now = datetime.utcnow()
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=now + timedelta(seconds=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False
        assert decision.reason is not None
        assert "cooldown" in decision.reason.lower()

    def test_cooldown_expires_after_1_hour(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Cooldown should expire after 1 hour."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        # After 1 hour from now, cooldown should have expired
        now = datetime.utcnow()
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=now + timedelta(hours=CONSECUTIVE_LOSS_COOLDOWN_HOURS, seconds=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_cooldown_still_active_before_1_hour(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Cooldown should still be active before 1 hour."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        now = datetime.utcnow()
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=now + timedelta(minutes=30),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False

    def test_win_resets_consecutive_losses(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """A win should reset the consecutive loss counter."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_win("EURUSD")
        guard.record_loss("EURUSD")

        # Only 1 loss after the win, should be allowed
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_cooldown_is_per_instrument(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Cooldown is per-instrument."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        now = datetime.utcnow()
        # Different instrument should not be in cooldown
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="GBPUSD",
            current_time=now + timedelta(seconds=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_is_in_cooldown_method(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """is_in_cooldown should correctly report cooldown state."""
        now = datetime.utcnow()
        assert guard.is_in_cooldown("EURUSD", now) is False

        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        now = datetime.utcnow()
        assert guard.is_in_cooldown("EURUSD", now + timedelta(seconds=1)) is True
        assert guard.is_in_cooldown(
            "EURUSD", now + timedelta(hours=2)
        ) is False

    def test_consecutive_loss_threshold_is_3(self) -> None:
        """Verify the threshold constant is 3."""
        assert CONSECUTIVE_LOSS_THRESHOLD == 3


# =============================================================================
# Task 17.4: Win rate throttling (below 40% → halve frequency, raise threshold)
# =============================================================================


class TestWinRateThrottling:
    """Tests for win rate throttling (Task 17.4)."""

    def test_normal_win_rate_full_limit(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Win rate >= 40% should use full daily limit."""
        # Record 9 trades
        for i in range(9):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # 10th trade should be allowed with normal win rate
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=60),
            recent_win_rate=0.5,
        )
        assert decision.allowed is True

    def test_low_win_rate_halves_limit(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Win rate < 40% should halve the daily limit (10 → 5)."""
        # Record 5 trades
        for i in range(5):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # 6th trade should be blocked with low win rate
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=36),
            recent_win_rate=0.3,
        )
        assert decision.allowed is False

    def test_low_win_rate_returns_elevated_threshold(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Low win rate should signal elevated confidence threshold (75)."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.3,
        )
        assert decision.allowed is True
        assert decision.elevated_threshold == CONFIDENCE_THRESHOLD_ELEVATED
        assert decision.elevated_threshold == 75

    def test_normal_win_rate_no_elevated_threshold(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Normal win rate should not set elevated threshold."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.5,
        )
        assert decision.elevated_threshold is None

    def test_win_rate_at_threshold_not_throttled(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Win rate exactly at 40% should NOT be throttled (< not <=)."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=WIN_RATE_THROTTLE_THRESHOLD,
        )
        assert decision.elevated_threshold is None

    def test_win_rate_just_below_threshold_throttled(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Win rate just below 40% should be throttled."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.39,
        )
        assert decision.elevated_threshold == CONFIDENCE_THRESHOLD_ELEVATED

    def test_halved_limit_minimum_is_1(self, base_time: datetime) -> None:
        """Halved limit should be at least 1."""
        guard = OvertradingGuard(max_trades_per_day=1)
        # With max=1, halved = max(1, 1//2) = max(1, 0) = 1
        # So 1 trade should still be allowed
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.3,
        )
        assert decision.allowed is True


# =============================================================================
# Task 17.5: HFT bypass (is_hft_signal=True skips ALL rules)
# =============================================================================


class TestHFTBypass:
    """Tests for HFT bypass of all overtrading rules (Task 17.5, Cross-Cutting Rule 2)."""

    def test_hft_bypasses_daily_limit(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """HFT signal should bypass daily trade limit."""
        # Fill up the daily limit
        for i in range(10):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # HFT should still be allowed
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=66),
            recent_win_rate=0.5,
            is_hft_signal=True,
        )
        assert decision.allowed is True

    def test_hft_bypasses_time_interval(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """HFT signal should bypass minimum time interval."""
        guard.record_trade("momentum", "EURUSD", base_time)

        # Immediately after, HFT should be allowed
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(seconds=1),
            recent_win_rate=0.5,
            is_hft_signal=True,
        )
        assert decision.allowed is True

    def test_hft_bypasses_consecutive_loss_cooldown(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """HFT signal should bypass consecutive loss cooldown."""
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")

        now = datetime.utcnow()
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=now + timedelta(seconds=1),
            recent_win_rate=0.5,
            is_hft_signal=True,
        )
        assert decision.allowed is True

    def test_hft_bypasses_win_rate_throttle(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """HFT signal should bypass win rate throttling."""
        # Fill up the halved limit (5 trades)
        for i in range(5):
            trade_time = base_time + timedelta(minutes=i * 6)
            guard.record_trade("momentum", "EURUSD", trade_time)

        # Non-HFT with low win rate would be blocked
        decision_normal = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=36),
            recent_win_rate=0.3,
            is_hft_signal=False,
        )
        assert decision_normal.allowed is False

        # HFT should still be allowed
        decision_hft = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=36),
            recent_win_rate=0.3,
            is_hft_signal=True,
        )
        assert decision_hft.allowed is True

    def test_hft_no_elevated_threshold(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """HFT signal should not have elevated threshold set."""
        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time,
            recent_win_rate=0.3,
            is_hft_signal=True,
        )
        assert decision.allowed is True
        assert decision.elevated_threshold is None

    def test_non_hft_default_is_false(
        self, guard: OvertradingGuard, base_time: datetime
    ) -> None:
        """Default is_hft_signal should be False (rules apply)."""
        guard.record_trade("momentum", "EURUSD", base_time)

        decision = guard.can_trade(
            strategy_name="momentum",
            instrument="EURUSD",
            current_time=base_time + timedelta(minutes=1),
            recent_win_rate=0.5,
        )
        assert decision.allowed is False


# =============================================================================
# Additional tests: TradeDecision dataclass and record methods
# =============================================================================


class TestTradeDecision:
    """Tests for the TradeDecision dataclass."""

    def test_allowed_decision(self) -> None:
        """Allowed decision should have no reason."""
        decision = TradeDecision(allowed=True)
        assert decision.allowed is True
        assert decision.reason is None
        assert decision.elevated_threshold is None

    def test_blocked_decision_with_reason(self) -> None:
        """Blocked decision should have a reason."""
        decision = TradeDecision(allowed=False, reason="Daily limit reached")
        assert decision.allowed is False
        assert decision.reason == "Daily limit reached"

    def test_decision_with_elevated_threshold(self) -> None:
        """Decision can include elevated threshold."""
        decision = TradeDecision(allowed=True, elevated_threshold=75)
        assert decision.elevated_threshold == 75


class TestRecordMethods:
    """Tests for record_trade, record_loss, record_win methods."""

    def test_record_trade_increments_count(self) -> None:
        guard = OvertradingGuard()
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        guard.record_trade("momentum", "EURUSD", base_time)
        assert guard.get_trade_count("momentum", "EURUSD") == 1

        guard.record_trade("momentum", "EURUSD", base_time + timedelta(minutes=6))
        assert guard.get_trade_count("momentum", "EURUSD") == 2

    def test_record_trade_updates_last_time(self) -> None:
        guard = OvertradingGuard()
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        guard.record_trade("momentum", "EURUSD", base_time)
        assert guard.last_trade_times["EURUSD"] == base_time

    def test_record_loss_increments_counter(self) -> None:
        guard = OvertradingGuard()
        guard.record_loss("EURUSD")
        assert guard.consecutive_losses["EURUSD"] == 1
        guard.record_loss("EURUSD")
        assert guard.consecutive_losses["EURUSD"] == 2

    def test_record_win_resets_counter(self) -> None:
        guard = OvertradingGuard()
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_win("EURUSD")
        assert guard.consecutive_losses["EURUSD"] == 0

    def test_record_win_clears_cooldown(self) -> None:
        guard = OvertradingGuard()
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        guard.record_loss("EURUSD")
        # Cooldown should be set
        assert "EURUSD" in guard._cooldown_start

        guard.record_win("EURUSD")
        assert "EURUSD" not in guard._cooldown_start
