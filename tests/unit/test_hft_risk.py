"""Unit tests for the HFT Risk Manager module.

Tests cover trade validation, PnL tracking, circuit breaker activation and
escalation, rate limiting (global and per-instrument), rejection rate tracking,
and throttle detection.

Validates: Requirements 22.3, 22.6, 22.7, 22.8, 22.9, 22.10
"""

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.risk.hft_risk import (
    CircuitBreakerEvent,
    HFTRiskManager,
    RateLimitStatus,
    _SlidingWindowCounter,
)


# =============================================================================
# Task 32.1: HFTRiskManager initialization and trade size limits
# =============================================================================


class TestHFTRiskManagerInit:
    """Tests for HFTRiskManager initialization."""

    def test_default_initialization(self) -> None:
        rm = HFTRiskManager()
        assert rm.circuit_breaker_active is False
        assert rm._hft_disabled is False
        assert rm._max_order_rate == 100
        assert rm._max_per_instrument_rate == 50

    def test_custom_order_rate(self) -> None:
        rm = HFTRiskManager(max_order_rate=200)
        assert rm._max_order_rate == 200

    def test_order_rate_clamped_to_min_10(self) -> None:
        rm = HFTRiskManager(max_order_rate=5)
        assert rm._max_order_rate == 10

    def test_order_rate_clamped_to_max_500(self) -> None:
        rm = HFTRiskManager(max_order_rate=1000)
        assert rm._max_order_rate == 500

    def test_max_trade_size_pct(self) -> None:
        assert HFTRiskManager.MAX_TRADE_SIZE_PCT == Decimal("0.005")

    def test_max_hft_exposure_pct(self) -> None:
        assert HFTRiskManager.MAX_HFT_EXPOSURE_PCT == Decimal("0.15")


# =============================================================================
# Task 32.6: HFT trade validation
# =============================================================================


class TestValidateHFTTrade:
    """Tests for HFT trade size and exposure validation."""

    def test_valid_trade_within_limits(self) -> None:
        rm = HFTRiskManager()
        # 0.5% of 100000 = 500, trade is 400
        result = rm.validate_hft_trade(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("10000"),
        )
        assert result is True

    def test_rejects_trade_exceeding_size_limit(self) -> None:
        rm = HFTRiskManager()
        # 0.5% of 100000 = 500, trade is 600
        result = rm.validate_hft_trade(
            trade_size=Decimal("600"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result is False

    def test_rejects_trade_at_exact_size_limit(self) -> None:
        rm = HFTRiskManager()
        # 0.5% of 100000 = 500, trade is exactly 500.01
        result = rm.validate_hft_trade(
            trade_size=Decimal("500.01"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result is False

    def test_allows_trade_at_exact_size_limit(self) -> None:
        rm = HFTRiskManager()
        # 0.5% of 100000 = 500, trade is exactly 500
        result = rm.validate_hft_trade(
            trade_size=Decimal("500"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result is True

    def test_rejects_trade_exceeding_exposure_limit(self) -> None:
        rm = HFTRiskManager()
        # 15% of 100000 = 15000, current exposure 14800, trade 300 → total 15100
        result = rm.validate_hft_trade(
            trade_size=Decimal("300"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("14800"),
        )
        assert result is False

    def test_allows_trade_within_exposure_limit(self) -> None:
        rm = HFTRiskManager()
        # 15% of 100000 = 15000, current exposure 14000, trade 500 → total 14500
        result = rm.validate_hft_trade(
            trade_size=Decimal("500"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("14000"),
        )
        assert result is True

    def test_rejects_when_equity_is_zero(self) -> None:
        rm = HFTRiskManager()
        result = rm.validate_hft_trade(
            trade_size=Decimal("100"),
            account_equity=Decimal("0"),
            current_hft_exposure=Decimal("0"),
        )
        assert result is False

    def test_rejects_when_equity_is_negative(self) -> None:
        rm = HFTRiskManager()
        result = rm.validate_hft_trade(
            trade_size=Decimal("100"),
            account_equity=Decimal("-1000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result is False


# =============================================================================
# Task 32.2: Rate limiting (global and per-instrument)
# =============================================================================


class TestRateLimiting:
    """Tests for global and per-instrument rate limiting."""

    def test_global_rate_allows_under_limit(self) -> None:
        rm = HFTRiskManager(max_order_rate=100)
        assert rm.check_global_rate() is True

    def test_global_rate_blocks_at_limit(self) -> None:
        rm = HFTRiskManager(max_order_rate=10)
        for _ in range(10):
            rm._global_rate_counter.record()
        assert rm.check_global_rate() is False

    def test_instrument_rate_allows_under_limit(self) -> None:
        rm = HFTRiskManager(max_per_instrument_rate=50)
        assert rm.check_instrument_rate("EUR/USD") is True

    def test_instrument_rate_blocks_at_limit(self) -> None:
        rm = HFTRiskManager(max_per_instrument_rate=3)
        rm._instrument_rate_counters["EUR/USD"] = _SlidingWindowCounter()
        for _ in range(3):
            rm._instrument_rate_counters["EUR/USD"].record()
        assert rm.check_instrument_rate("EUR/USD") is False

    def test_instrument_rate_independent_per_instrument(self) -> None:
        rm = HFTRiskManager(max_per_instrument_rate=3)
        rm._instrument_rate_counters["EUR/USD"] = _SlidingWindowCounter()
        for _ in range(3):
            rm._instrument_rate_counters["EUR/USD"].record()
        # GBP/USD should still be allowed
        assert rm.check_instrument_rate("GBP/USD") is True

    def test_record_order_increments_counters(self) -> None:
        rm = HFTRiskManager()
        rm.record_order("EUR/USD")
        assert rm._global_rate_counter.count() == 1
        assert rm._instrument_rate_counters["EUR/USD"].count() == 1

    def test_combined_rate_check_allows(self) -> None:
        rm = HFTRiskManager()
        status = rm.check_rate_limit("EUR/USD")
        assert status.allowed is True
        assert status.reason == ""

    def test_combined_rate_check_blocks_global(self) -> None:
        rm = HFTRiskManager(max_order_rate=10)
        for _ in range(10):
            rm._global_rate_counter.record()
        status = rm.check_rate_limit("EUR/USD")
        assert status.allowed is False
        assert "Global rate limit" in status.reason

    def test_combined_rate_check_blocks_instrument(self) -> None:
        rm = HFTRiskManager(max_per_instrument_rate=5)
        rm._instrument_rate_counters["EUR/USD"] = _SlidingWindowCounter()
        for _ in range(5):
            rm._instrument_rate_counters["EUR/USD"].record()
        status = rm.check_rate_limit("EUR/USD")
        assert status.allowed is False
        assert "Per-instrument rate limit" in status.reason


# =============================================================================
# Task 32.3: Rejection rate tracking and throttle signal
# =============================================================================


class TestRejectionAndThrottle:
    """Tests for rejection rate tracking and throttle detection."""

    def test_initial_rejection_rate_is_zero(self) -> None:
        rm = HFTRiskManager()
        assert rm.get_rejection_rate() == 0.0

    def test_rejection_rate_calculation(self) -> None:
        rm = HFTRiskManager()
        # 10 total orders, 3 rejections → 30%
        for _ in range(10):
            rm._total_order_counter.record()
        for _ in range(3):
            rm._rejection_counter.record()
        rate = rm.get_rejection_rate()
        assert abs(rate - 0.3) < 0.01

    def test_should_throttle_false_below_threshold(self) -> None:
        rm = HFTRiskManager()
        # 10 total, 1 rejection → 10% < 20%
        for _ in range(10):
            rm._total_order_counter.record()
        rm._rejection_counter.record()
        assert rm.should_throttle() is False

    def test_should_throttle_true_above_threshold(self) -> None:
        rm = HFTRiskManager()
        # 10 total, 3 rejections → 30% > 20%
        for _ in range(10):
            rm._total_order_counter.record()
        for _ in range(3):
            rm._rejection_counter.record()
        assert rm.should_throttle() is True

    def test_should_throttle_false_when_no_orders(self) -> None:
        rm = HFTRiskManager()
        assert rm.should_throttle() is False

    def test_record_rejection_increments_counter(self) -> None:
        rm = HFTRiskManager()
        rm.record_rejection()
        rm.record_rejection()
        assert rm._rejection_counter.count() == 2


# =============================================================================
# Task 32.4: 1-minute rolling PnL and circuit breaker activation
# =============================================================================


class TestPnLAndCircuitBreaker:
    """Tests for PnL tracking and circuit breaker activation."""

    def test_update_pnl_records_entry(self) -> None:
        rm = HFTRiskManager()
        rm.update_pnl(Decimal("-10"), account_equity=Decimal("100000"))
        assert len(rm._pnl_window) == 1

    def test_circuit_breaker_activates_on_threshold(self) -> None:
        rm = HFTRiskManager()
        # -0.5% of 100000 = -500
        # Single large loss that exceeds threshold
        rm.update_pnl(Decimal("-600"), account_equity=Decimal("100000"))
        assert rm.circuit_breaker_active is True

    def test_circuit_breaker_does_not_activate_above_threshold(self) -> None:
        rm = HFTRiskManager()
        # -0.5% of 100000 = -500, loss is only -400
        rm.update_pnl(Decimal("-400"), account_equity=Decimal("100000"))
        assert rm.circuit_breaker_active is False

    def test_circuit_breaker_activates_on_cumulative_loss(self) -> None:
        rm = HFTRiskManager()
        # Multiple small losses that cumulatively exceed threshold
        # -0.5% of 100000 = -500
        rm.update_pnl(Decimal("-200"), account_equity=Decimal("100000"))
        rm.update_pnl(Decimal("-200"), account_equity=Decimal("100000"))
        assert rm.circuit_breaker_active is False
        rm.update_pnl(Decimal("-200"), account_equity=Decimal("100000"))
        # Cumulative: -600 < -500 threshold
        assert rm.circuit_breaker_active is True

    def test_pnl_window_prunes_old_entries(self) -> None:
        rm = HFTRiskManager()
        # Add an old entry
        old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        rm._pnl_window.append((old_time, Decimal("-1000")))
        # Update with a small loss — old entry should be pruned
        rm.update_pnl(Decimal("-10"), account_equity=Decimal("100000"))
        # Only the new entry should remain
        assert len(rm._pnl_window) == 1
        assert rm.circuit_breaker_active is False

    def test_is_hft_allowed_false_when_circuit_breaker_active(self) -> None:
        rm = HFTRiskManager()
        rm.circuit_breaker_active = True
        rm._circuit_breaker_activated_at = datetime.now(timezone.utc)
        assert rm.is_hft_allowed() is False

    def test_is_hft_allowed_true_after_duration_expires(self) -> None:
        rm = HFTRiskManager()
        rm.circuit_breaker_active = True
        # Set activation time to 61 seconds ago
        rm._circuit_breaker_activated_at = datetime.now(timezone.utc) - timedelta(
            seconds=61
        )
        assert rm.is_hft_allowed() is True
        # Should auto-deactivate
        assert rm.circuit_breaker_active is False


# =============================================================================
# Task 32.5: Circuit breaker escalation
# =============================================================================


class TestCircuitBreakerEscalation:
    """Tests for circuit breaker escalation (3 activations → disable HFT)."""

    def test_first_activation_does_not_disable(self) -> None:
        rm = HFTRiskManager()
        rm.activate_circuit_breaker()
        assert rm._hft_disabled is False
        assert rm._circuit_breaker_count == 1

    def test_second_activation_does_not_disable(self) -> None:
        rm = HFTRiskManager()
        rm.activate_circuit_breaker()
        rm.deactivate_circuit_breaker()
        rm.activate_circuit_breaker()
        assert rm._hft_disabled is False
        assert rm._circuit_breaker_count == 2

    def test_third_activation_disables_hft(self) -> None:
        rm = HFTRiskManager()
        rm.activate_circuit_breaker()
        rm.deactivate_circuit_breaker()
        rm.activate_circuit_breaker()
        rm.deactivate_circuit_breaker()
        rm.activate_circuit_breaker()
        assert rm._hft_disabled is True
        assert rm._circuit_breaker_count == 3

    def test_hft_not_allowed_when_disabled(self) -> None:
        rm = HFTRiskManager()
        rm._hft_disabled = True
        assert rm.is_hft_allowed() is False

    def test_re_enable_hft_after_disable(self) -> None:
        rm = HFTRiskManager()
        rm._hft_disabled = True
        rm.re_enable_hft()
        assert rm._hft_disabled is False
        assert rm.is_hft_allowed() is True
        assert rm._circuit_breaker_count == 0

    def test_escalation_only_counts_within_window(self) -> None:
        rm = HFTRiskManager()
        # Add 2 old events outside the 1-hour window
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        rm._circuit_breaker_events.append(
            CircuitBreakerEvent(
                timestamp=old_time,
                pnl_at_activation=Decimal("-500"),
                equity_at_activation=Decimal("100000"),
            )
        )
        rm._circuit_breaker_events.append(
            CircuitBreakerEvent(
                timestamp=old_time,
                pnl_at_activation=Decimal("-500"),
                equity_at_activation=Decimal("100000"),
            )
        )
        # This should be the first activation in the current window
        rm.activate_circuit_breaker()
        assert rm._circuit_breaker_count == 1
        assert rm._hft_disabled is False


# =============================================================================
# Status
# =============================================================================


class TestGetStatus:
    """Tests for the get_status method."""

    def test_status_when_healthy(self) -> None:
        rm = HFTRiskManager()
        status = rm.get_status()
        assert status["hft_allowed"] is True
        assert status["hft_disabled"] is False
        assert status["circuit_breaker_active"] is False
        assert status["circuit_breaker_count"] == 0
        assert status["rejection_rate"] == 0.0
        assert status["should_throttle"] is False

    def test_status_when_circuit_breaker_active(self) -> None:
        rm = HFTRiskManager()
        rm.circuit_breaker_active = True
        rm._circuit_breaker_activated_at = datetime.now(timezone.utc)
        rm._circuit_breaker_count = 2
        status = rm.get_status()
        assert status["hft_allowed"] is False
        assert status["circuit_breaker_active"] is True
        assert status["circuit_breaker_count"] == 2

    def test_status_when_hft_disabled(self) -> None:
        rm = HFTRiskManager()
        rm._hft_disabled = True
        status = rm.get_status()
        assert status["hft_allowed"] is False
        assert status["hft_disabled"] is True


# =============================================================================
# _SlidingWindowCounter
# =============================================================================


class TestSlidingWindowCounterInternal:
    """Tests for the internal _SlidingWindowCounter."""

    def test_initial_count_zero(self) -> None:
        counter = _SlidingWindowCounter()
        assert counter.count() == 0

    def test_records_increment(self) -> None:
        counter = _SlidingWindowCounter()
        counter.record()
        counter.record()
        assert counter.count() == 2

    def test_prunes_expired_entries(self) -> None:
        counter = _SlidingWindowCounter(window_seconds=0.01)
        counter.record()
        time.sleep(0.02)
        assert counter.count() == 0


# =============================================================================
# RateLimitStatus dataclass
# =============================================================================


class TestRateLimitStatusDataclass:
    """Tests for the RateLimitStatus dataclass."""

    def test_allowed_status(self) -> None:
        status = RateLimitStatus(allowed=True, global_count=5, instrument_count=2)
        assert status.allowed is True
        assert status.reason == ""
        assert status.global_count == 5
        assert status.instrument_count == 2

    def test_rejected_status(self) -> None:
        status = RateLimitStatus(
            allowed=False,
            reason="Global rate limit exceeded",
            global_count=100,
            instrument_count=10,
        )
        assert status.allowed is False
        assert "Global rate limit" in status.reason
