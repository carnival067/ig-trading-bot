"""HFT Risk Manager with circuit breaker and rate limiting.

Provides the HFTRiskManager class that enforces HFT-specific risk controls:
trade size limits, exposure limits, per-instrument and global rate limiting,
rolling PnL tracking, and circuit breaker logic with escalation.

Validates: Requirements 22.3, 22.6, 22.7, 22.8, 22.9, 22.10
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.config.constants import (
    HFT_CIRCUIT_BREAKER_DURATION_SECONDS,
    HFT_CIRCUIT_BREAKER_MAX_ACTIVATIONS,
    HFT_CIRCUIT_BREAKER_PNL_PCT,
    HFT_CIRCUIT_BREAKER_WINDOW_HOURS,
    HFT_MAX_EXPOSURE_PCT,
    HFT_MAX_ORDER_RATE_DEFAULT,
    HFT_MAX_PER_INSTRUMENT_RATE,
    HFT_MAX_TRADE_SIZE_PCT,
    HFT_THROTTLE_REJECTION_RATE_PCT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RateLimitStatus:
    """Status of rate limiting checks.

    Attributes:
        allowed: Whether the order is allowed.
        reason: Reason for rejection if not allowed.
        global_count: Current global order count in window.
        instrument_count: Current per-instrument order count in window.
    """

    allowed: bool
    reason: str = ""
    global_count: int = 0
    instrument_count: int = 0


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker activation.

    Attributes:
        timestamp: When the circuit breaker was activated.
        pnl_at_activation: The PnL value that triggered activation.
        equity_at_activation: Account equity at time of activation.
    """

    timestamp: datetime
    pnl_at_activation: Decimal
    equity_at_activation: Decimal


# ---------------------------------------------------------------------------
# Sliding Window Counter (for rate limiting)
# ---------------------------------------------------------------------------


class _SlidingWindowCounter:
    """Counts events within a sliding time window for rate limiting."""

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []

    def record(self) -> None:
        """Record an event at the current time."""
        self._timestamps.append(time.monotonic())
        self._prune()

    def count(self) -> int:
        """Return the number of events within the current window."""
        self._prune()
        return len(self._timestamps)

    def _prune(self) -> None:
        """Remove timestamps outside the sliding window."""
        cutoff = time.monotonic() - self._window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]


# ---------------------------------------------------------------------------
# HFT Risk Manager
# ---------------------------------------------------------------------------


class HFTRiskManager:
    """Manages HFT-specific risk controls including rate limiting and circuit breaker.

    Enforces:
    - Trade size limit: max 0.5% of equity per trade
    - Exposure limit: max 15% of equity total HFT exposure
    - Global rate limit: 100 orders/sec (configurable 10-500)
    - Per-instrument rate limit: 50 orders/sec
    - Circuit breaker: halts HFT for 60s when 1-min PnL < -0.5% equity
    - Escalation: 3 activations in 1 hour → disable HFT entirely

    Args:
        max_order_rate: Maximum global orders per second (default 100, range 10-500).
        max_per_instrument_rate: Maximum orders per second per instrument (default 50).
    """

    MAX_TRADE_SIZE_PCT = Decimal(str(HFT_MAX_TRADE_SIZE_PCT))
    MAX_HFT_EXPOSURE_PCT = Decimal(str(HFT_MAX_EXPOSURE_PCT))

    def __init__(
        self,
        max_order_rate: int = HFT_MAX_ORDER_RATE_DEFAULT,
        max_per_instrument_rate: int = HFT_MAX_PER_INSTRUMENT_RATE,
    ) -> None:
        # Clamp max_order_rate to valid range
        self._max_order_rate = max(10, min(500, max_order_rate))
        self._max_per_instrument_rate = max_per_instrument_rate

        # Circuit breaker state
        self.circuit_breaker_active: bool = False
        self._circuit_breaker_activated_at: datetime | None = None
        self._circuit_breaker_count: int = 0
        self._circuit_breaker_events: list[CircuitBreakerEvent] = []
        self._hft_disabled: bool = False

        # PnL tracking (1-minute rolling window)
        self._pnl_window: list[tuple[datetime, Decimal]] = []

        # Rate limiting
        self._global_rate_counter = _SlidingWindowCounter(window_seconds=1.0)
        self._instrument_rate_counters: dict[str, _SlidingWindowCounter] = {}

        # Rejection tracking (10-second window for throttle detection)
        self._rejection_counter = _SlidingWindowCounter(window_seconds=10.0)
        self._total_order_counter = _SlidingWindowCounter(window_seconds=10.0)

    # -------------------------------------------------------------------------
    # Trade Validation
    # -------------------------------------------------------------------------

    def validate_hft_trade(
        self,
        trade_size: Decimal,
        account_equity: Decimal,
        current_hft_exposure: Decimal,
    ) -> bool:
        """Validate an HFT trade against size and exposure limits.

        A trade is valid if:
        - trade_size <= 0.5% of account_equity
        - current_hft_exposure + trade_size <= 15% of account_equity

        Args:
            trade_size: The notional size of the proposed trade.
            account_equity: Current account equity.
            current_hft_exposure: Current total HFT exposure.

        Returns:
            True if the trade passes validation, False otherwise.
        """
        if account_equity <= 0:
            return False

        max_trade_size = account_equity * self.MAX_TRADE_SIZE_PCT
        max_exposure = account_equity * self.MAX_HFT_EXPOSURE_PCT

        # Check individual trade size
        if trade_size > max_trade_size:
            logger.warning(
                "HFT trade rejected: size %s > max %s (0.5%% of equity %s)",
                trade_size,
                max_trade_size,
                account_equity,
            )
            return False

        # Check total exposure
        if current_hft_exposure + trade_size > max_exposure:
            logger.warning(
                "HFT trade rejected: exposure %s + %s > max %s (15%% of equity %s)",
                current_hft_exposure,
                trade_size,
                max_exposure,
                account_equity,
            )
            return False

        return True

    # -------------------------------------------------------------------------
    # PnL Tracking and Circuit Breaker
    # -------------------------------------------------------------------------

    def update_pnl(self, pnl: Decimal, account_equity: Decimal) -> None:
        """Update the 1-minute rolling PnL window and check circuit breaker.

        If cumulative PnL in the 1-minute window drops below -0.5% of equity,
        the circuit breaker is activated for 60 seconds.

        Args:
            pnl: The PnL value to record (positive or negative).
            account_equity: Current account equity for threshold calculation.
        """
        now = datetime.now(timezone.utc)
        self._pnl_window.append((now, pnl))

        # Prune entries older than 1 minute
        cutoff = now - timedelta(minutes=1)
        self._pnl_window = [(t, p) for t, p in self._pnl_window if t > cutoff]

        # Calculate cumulative PnL in window
        cumulative_pnl = sum(p for _, p in self._pnl_window)

        # Check circuit breaker threshold
        threshold = account_equity * Decimal(str(HFT_CIRCUIT_BREAKER_PNL_PCT))
        if cumulative_pnl < threshold and not self.circuit_breaker_active:
            logger.warning(
                "HFT circuit breaker triggered: PnL %s < threshold %s",
                cumulative_pnl,
                threshold,
            )
            self.activate_circuit_breaker(
                pnl_at_activation=cumulative_pnl,
                equity_at_activation=account_equity,
            )

    def activate_circuit_breaker(
        self,
        pnl_at_activation: Decimal = Decimal("0"),
        equity_at_activation: Decimal = Decimal("0"),
    ) -> None:
        """Activate the circuit breaker, halting HFT for 60 seconds.

        If this is the 3rd activation within a 1-hour window, HFT is
        disabled entirely and requires manual re-enablement.

        Args:
            pnl_at_activation: The PnL value that triggered activation.
            equity_at_activation: Account equity at time of activation.
        """
        now = datetime.now(timezone.utc)
        self.circuit_breaker_active = True
        self._circuit_breaker_activated_at = now

        # Record the event
        event = CircuitBreakerEvent(
            timestamp=now,
            pnl_at_activation=pnl_at_activation,
            equity_at_activation=equity_at_activation,
        )
        self._circuit_breaker_events.append(event)

        # Count activations within the 1-hour window
        window_start = now - timedelta(hours=HFT_CIRCUIT_BREAKER_WINDOW_HOURS)
        recent_events = [
            e for e in self._circuit_breaker_events if e.timestamp > window_start
        ]
        self._circuit_breaker_count = len(recent_events)

        logger.info(
            "Circuit breaker activated (#%d in window). HFT halted for %ds.",
            self._circuit_breaker_count,
            HFT_CIRCUIT_BREAKER_DURATION_SECONDS,
        )

        # Escalation: 3 activations in 1 hour → disable HFT entirely
        if self._circuit_breaker_count >= HFT_CIRCUIT_BREAKER_MAX_ACTIVATIONS:
            self._hft_disabled = True
            logger.critical(
                "HFT DISABLED: %d circuit breaker activations within %d hour(s). "
                "Manual re-enablement required.",
                self._circuit_breaker_count,
                HFT_CIRCUIT_BREAKER_WINDOW_HOURS,
            )

    def deactivate_circuit_breaker(self) -> None:
        """Manually deactivate the circuit breaker (for testing or after timeout)."""
        self.circuit_breaker_active = False
        self._circuit_breaker_activated_at = None

    def is_hft_allowed(self) -> bool:
        """Check if HFT trading is currently allowed.

        Returns False if:
        - Circuit breaker is active (and duration hasn't expired)
        - HFT has been disabled due to escalation

        Returns:
            True if HFT is allowed, False otherwise.
        """
        if self._hft_disabled:
            return False

        if self.circuit_breaker_active:
            # Check if circuit breaker duration has expired
            if self._circuit_breaker_activated_at is not None:
                elapsed = (
                    datetime.now(timezone.utc) - self._circuit_breaker_activated_at
                )
                if elapsed.total_seconds() >= HFT_CIRCUIT_BREAKER_DURATION_SECONDS:
                    # Auto-deactivate after duration expires
                    self.circuit_breaker_active = False
                    self._circuit_breaker_activated_at = None
                    logger.info("Circuit breaker expired, HFT resumed.")
                    return True
                return False

        return True

    def re_enable_hft(self) -> None:
        """Manually re-enable HFT after escalation-based disablement."""
        self._hft_disabled = False
        self.circuit_breaker_active = False
        self._circuit_breaker_activated_at = None
        self._circuit_breaker_count = 0
        logger.info("HFT manually re-enabled.")

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------

    def check_global_rate(self) -> bool:
        """Check if the global order rate is within limits.

        Returns:
            True if under the global rate limit (100 orders/sec default).
        """
        return self._global_rate_counter.count() < self._max_order_rate

    def check_instrument_rate(self, instrument: str) -> bool:
        """Check if the per-instrument order rate is within limits.

        Args:
            instrument: The instrument identifier.

        Returns:
            True if under the per-instrument rate limit (50 orders/sec).
        """
        if instrument not in self._instrument_rate_counters:
            return True
        return (
            self._instrument_rate_counters[instrument].count()
            < self._max_per_instrument_rate
        )

    def record_order(self, instrument: str) -> None:
        """Record an order submission for rate tracking.

        Args:
            instrument: The instrument the order was submitted for.
        """
        self._global_rate_counter.record()
        if instrument not in self._instrument_rate_counters:
            self._instrument_rate_counters[instrument] = _SlidingWindowCounter()
        self._instrument_rate_counters[instrument].record()
        self._total_order_counter.record()

    def record_rejection(self) -> None:
        """Record an order rejection for throttle detection."""
        self._rejection_counter.record()

    def get_rejection_rate(self) -> float:
        """Calculate the current rejection rate in the 10-second window.

        Returns:
            Rejection rate as a float between 0.0 and 1.0.
        """
        total = self._total_order_counter.count()
        if total == 0:
            return 0.0
        rejections = self._rejection_counter.count()
        return rejections / total

    def should_throttle(self) -> bool:
        """Check if HFT should be throttled due to high rejection rate.

        Returns True if the rejection rate exceeds 20% in the 10-second window.

        Returns:
            True if throttling should be applied.
        """
        return self.get_rejection_rate() > HFT_THROTTLE_REJECTION_RATE_PCT

    # -------------------------------------------------------------------------
    # Combined Rate Check
    # -------------------------------------------------------------------------

    def check_rate_limit(self, instrument: str) -> RateLimitStatus:
        """Perform a combined rate limit check (global + per-instrument).

        Args:
            instrument: The instrument to check.

        Returns:
            RateLimitStatus with the result.
        """
        global_count = self._global_rate_counter.count()
        instrument_count = 0
        if instrument in self._instrument_rate_counters:
            instrument_count = self._instrument_rate_counters[instrument].count()

        if global_count >= self._max_order_rate:
            self.record_rejection()
            return RateLimitStatus(
                allowed=False,
                reason=f"Global rate limit exceeded: {global_count}/{self._max_order_rate}",
                global_count=global_count,
                instrument_count=instrument_count,
            )

        if instrument_count >= self._max_per_instrument_rate:
            self.record_rejection()
            return RateLimitStatus(
                allowed=False,
                reason=(
                    f"Per-instrument rate limit exceeded for {instrument}: "
                    f"{instrument_count}/{self._max_per_instrument_rate}"
                ),
                global_count=global_count,
                instrument_count=instrument_count,
            )

        return RateLimitStatus(
            allowed=True,
            global_count=global_count,
            instrument_count=instrument_count,
        )

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return the current HFT risk manager status.

        Returns:
            Dict with circuit breaker state, rate info, and HFT status.
        """
        return {
            "hft_allowed": self.is_hft_allowed(),
            "hft_disabled": self._hft_disabled,
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_count": self._circuit_breaker_count,
            "global_order_rate": self._global_rate_counter.count(),
            "rejection_rate": self.get_rejection_rate(),
            "should_throttle": self.should_throttle(),
        }
