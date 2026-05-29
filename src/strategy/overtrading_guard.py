"""Overtrading prevention guard for the Strategy Engine.

Enforces trade frequency limits including daily maximums, minimum time intervals,
consecutive loss cooldowns, and win rate throttling. HFT signals bypass all
overtrading rules per Cross-Cutting Rule 2.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, Cross-Cutting Rule 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.config.constants import (
    CONFIDENCE_THRESHOLD_ELEVATED,
    CONSECUTIVE_LOSS_COOLDOWN_HOURS,
    CONSECUTIVE_LOSS_THRESHOLD,
    MAX_TRADES_PER_DAY_DEFAULT,
    MIN_TRADE_INTERVAL_MINUTES,
    WIN_RATE_THROTTLE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Result of an overtrading guard check.

    Attributes:
        allowed: Whether the trade is permitted.
        reason: Human-readable reason if blocked.
        elevated_threshold: If win rate is low, the elevated confidence threshold
            that should be applied (or None if normal threshold applies).
    """

    allowed: bool
    reason: str | None = None
    elevated_threshold: int | None = None


class OvertradingGuard:
    """Prevents overtrading by enforcing frequency and cooldown rules.

    Rules enforced (in order):
    1. Daily trade limit per strategy per instrument (default 10, configurable 1-100)
    2. Minimum 5-minute interval between trades on the same instrument
    3. Consecutive loss cooldown (3 losses → 1-hour cooldown per instrument)
    4. Win rate throttle (below 40% → halve frequency, raise threshold to 75)

    Cross-Cutting Rule 2: HFT signals bypass ALL overtrading rules.

    Args:
        max_trades_per_day: Maximum trades per day per strategy (1-100).
        min_interval_minutes: Minimum minutes between trades on same instrument.
        consecutive_loss_cooldown_hours: Hours of cooldown after consecutive losses.
    """

    def __init__(
        self,
        max_trades_per_day: int = MAX_TRADES_PER_DAY_DEFAULT,
        min_interval_minutes: int = MIN_TRADE_INTERVAL_MINUTES,
        consecutive_loss_cooldown_hours: int = CONSECUTIVE_LOSS_COOLDOWN_HOURS,
    ) -> None:
        # Clamp max_trades_per_day to valid range [1, 100]
        self._max_trades_per_day = max(1, min(max_trades_per_day, 100))
        self._min_interval = timedelta(minutes=min_interval_minutes)
        self._cooldown_duration = timedelta(hours=consecutive_loss_cooldown_hours)

        # strategy_name → instrument → count
        self.trade_counts: dict[str, dict[str, int]] = {}

        # instrument → last trade time
        self.last_trade_times: dict[str, datetime] = {}

        # instrument → consecutive loss count
        self.consecutive_losses: dict[str, int] = {}

        # instrument → cooldown start time
        self._cooldown_start: dict[str, datetime] = {}

    @property
    def max_trades_per_day(self) -> int:
        """Configured maximum trades per day per strategy."""
        return self._max_trades_per_day

    def can_trade(
        self,
        strategy_name: str,
        instrument: str,
        current_time: datetime,
        recent_win_rate: float,
        is_hft_signal: bool = False,
    ) -> TradeDecision:
        """Check whether a trade is allowed by overtrading rules.

        Cross-Cutting Rule 2: If is_hft_signal=True, bypass ALL overtrading
        rules. HFT has its own dedicated safeguards (rate limiting, circuit
        breaker, exposure cap) defined in the HFT Risk Manager.

        Args:
            strategy_name: Name of the strategy requesting the trade.
            instrument: Instrument identifier.
            current_time: Current timestamp for interval checks.
            recent_win_rate: Win rate over the last 20 trades (0.0 to 1.0).
            is_hft_signal: Whether this is an HFT-generated signal.

        Returns:
            TradeDecision indicating whether the trade is allowed.
        """
        # Cross-Cutting Rule 2: HFT bypasses all overtrading rules
        if is_hft_signal:
            return TradeDecision(allowed=True)

        # Rule 1: Daily trade limit per strategy
        strategy_counts = self.trade_counts.get(strategy_name, {})
        instrument_count = strategy_counts.get(instrument, 0)

        # Win rate throttle affects the effective daily limit
        effective_max = self._max_trades_per_day
        elevated_threshold: int | None = None

        if recent_win_rate < WIN_RATE_THROTTLE_THRESHOLD:
            # Halve the frequency
            effective_max = max(1, self._max_trades_per_day // 2)
            elevated_threshold = CONFIDENCE_THRESHOLD_ELEVATED

        if instrument_count >= effective_max:
            reason = (
                f"Daily trade limit reached: {instrument_count}/{effective_max} "
                f"trades for strategy '{strategy_name}' on {instrument}"
            )
            logger.info("Overtrading blocked: %s", reason)
            return TradeDecision(allowed=False, reason=reason)

        # Rule 2: Minimum time interval (5 minutes between trades on same instrument)
        last_time = self.last_trade_times.get(instrument)
        if last_time is not None:
            elapsed = current_time - last_time
            if elapsed < self._min_interval:
                remaining = self._min_interval - elapsed
                reason = (
                    f"Minimum interval not met: {remaining.total_seconds():.0f}s "
                    f"remaining for {instrument}"
                )
                logger.info("Overtrading blocked: %s", reason)
                return TradeDecision(allowed=False, reason=reason)

        # Rule 3: Consecutive loss cooldown
        losses = self.consecutive_losses.get(instrument, 0)
        if losses >= CONSECUTIVE_LOSS_THRESHOLD:
            cooldown_start = self._cooldown_start.get(instrument)
            if cooldown_start is not None:
                cooldown_end = cooldown_start + self._cooldown_duration
                if current_time < cooldown_end:
                    remaining = cooldown_end - current_time
                    reason = (
                        f"Consecutive loss cooldown active: {losses} losses on "
                        f"{instrument}, {remaining.total_seconds():.0f}s remaining"
                    )
                    logger.info("Overtrading blocked: %s", reason)
                    return TradeDecision(allowed=False, reason=reason)
                else:
                    # Cooldown expired, reset
                    self.consecutive_losses[instrument] = 0
                    del self._cooldown_start[instrument]

        # Rule 4: Win rate throttle (already applied to effective_max above)
        # If win rate is low, signal that elevated threshold should be used
        return TradeDecision(allowed=True, elevated_threshold=elevated_threshold)

    def record_trade(
        self, strategy_name: str, instrument: str, time: datetime
    ) -> None:
        """Record that a trade was executed.

        Updates the daily trade count and last trade time for the instrument.

        Args:
            strategy_name: Name of the strategy that traded.
            instrument: Instrument that was traded.
            time: Time of the trade execution.
        """
        # Update trade count
        if strategy_name not in self.trade_counts:
            self.trade_counts[strategy_name] = {}
        strategy_counts = self.trade_counts[strategy_name]
        strategy_counts[instrument] = strategy_counts.get(instrument, 0) + 1

        # Update last trade time
        self.last_trade_times[instrument] = time

        logger.debug(
            "Trade recorded: strategy=%s instrument=%s count=%d",
            strategy_name,
            instrument,
            strategy_counts[instrument],
        )

    def record_loss(self, instrument: str) -> None:
        """Record a losing trade on an instrument.

        Increments the consecutive loss counter. If the threshold is reached,
        starts the cooldown period.

        Args:
            instrument: Instrument that had a losing trade.
        """
        current = self.consecutive_losses.get(instrument, 0)
        current += 1
        self.consecutive_losses[instrument] = current

        if current >= CONSECUTIVE_LOSS_THRESHOLD and instrument not in self._cooldown_start:
            self._cooldown_start[instrument] = datetime.utcnow()
            logger.info(
                "Cooldown triggered: %d consecutive losses on %s",
                current,
                instrument,
            )

    def record_win(self, instrument: str) -> None:
        """Record a winning trade on an instrument.

        Resets the consecutive loss counter for the instrument.

        Args:
            instrument: Instrument that had a winning trade.
        """
        self.consecutive_losses[instrument] = 0
        if instrument in self._cooldown_start:
            del self._cooldown_start[instrument]

    def reset_daily(self) -> None:
        """Reset daily trade counts at 00:00 UTC.

        Clears all strategy trade counts for the new trading day.
        Does NOT reset consecutive loss tracking or cooldowns.
        """
        self.trade_counts.clear()
        logger.info("Daily trade counts reset")

    def get_trade_count(self, strategy_name: str, instrument: str) -> int:
        """Get the current daily trade count for a strategy/instrument pair.

        Args:
            strategy_name: Strategy name.
            instrument: Instrument identifier.

        Returns:
            Number of trades executed today.
        """
        return self.trade_counts.get(strategy_name, {}).get(instrument, 0)

    def is_in_cooldown(self, instrument: str, current_time: datetime) -> bool:
        """Check if an instrument is currently in cooldown.

        Args:
            instrument: Instrument to check.
            current_time: Current time for comparison.

        Returns:
            True if the instrument is in an active cooldown period.
        """
        losses = self.consecutive_losses.get(instrument, 0)
        if losses < CONSECUTIVE_LOSS_THRESHOLD:
            return False

        cooldown_start = self._cooldown_start.get(instrument)
        if cooldown_start is None:
            return False

        cooldown_end = cooldown_start + self._cooldown_duration
        return current_time < cooldown_end
