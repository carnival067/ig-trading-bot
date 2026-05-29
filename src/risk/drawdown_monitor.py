"""Drawdown monitoring and daily loss protection for the Risk Engine.

Tracks peak equity, calculates drawdown percentages, enforces daily max loss
limits, applies drawdown-based position size reductions, and triggers the kill
switch when drawdown exceeds critical thresholds.

Validates: Requirements 5.1, 5.2, 5.3
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from src.config.constants import (
    DAILY_MAX_LOSS_PCT,
    DRAWDOWN_REDUCTION_PCT,
    DRAWDOWN_SIZE_REDUCTION_FACTOR,
    KILL_SWITCH_DRAWDOWN_PCT,
)


class TradeDecision(Enum):
    """Possible outcomes from the drawdown monitor's trade evaluation."""

    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    KILL_SWITCH = "kill_switch"


@dataclass
class ReductionFactor:
    """A multiplicative position size reduction factor with a reason.

    Attributes:
        factor: Multiplier applied to position size (e.g. 0.25 means 75% reduction).
        reason: Human-readable explanation for the reduction.
    """

    factor: Decimal
    reason: str


@dataclass
class DrawdownCheckResult:
    """Result of a drawdown/daily-loss check for a trade signal.

    Attributes:
        decision: The trade decision (ALLOW, REDUCE_SIZE, REJECT, KILL_SWITCH).
        reason: Human-readable explanation for the decision, or None if ALLOW.
        reduction_factor: The ReductionFactor to apply when decision is REDUCE_SIZE.
        drawdown_pct: Current drawdown as a Decimal percentage (e.g. Decimal("0.12") = 12%).
    """

    decision: TradeDecision
    reason: str | None
    reduction_factor: ReductionFactor | None
    drawdown_pct: Decimal


class DrawdownMonitor:
    """Monitors account drawdown and daily losses to protect capital.

    Tracks peak equity, calculates current drawdown, enforces daily max loss
    limits, applies position size reductions at configurable thresholds, and
    triggers the kill switch at critical drawdown levels.

    Args:
        initial_equity: Starting account equity value.
        daily_max_loss_pct: Maximum daily realized loss as fraction of equity (default 3%).
        drawdown_reduction_pct: Drawdown threshold for size reduction (default 10%).
        kill_switch_pct: Drawdown threshold for kill switch activation (default 15%).
    """

    def __init__(
        self,
        initial_equity: Decimal,
        daily_max_loss_pct: Decimal | None = None,
        drawdown_reduction_pct: Decimal | None = None,
        kill_switch_pct: Decimal | None = None,
    ) -> None:
        self.peak_equity: Decimal = initial_equity
        self.daily_max_loss_pct: Decimal = (
            daily_max_loss_pct
            if daily_max_loss_pct is not None
            else Decimal(str(DAILY_MAX_LOSS_PCT))
        )
        self.drawdown_reduction_pct: Decimal = (
            drawdown_reduction_pct
            if drawdown_reduction_pct is not None
            else Decimal(str(DRAWDOWN_REDUCTION_PCT))
        )
        self.kill_switch_pct: Decimal = (
            kill_switch_pct
            if kill_switch_pct is not None
            else Decimal(str(KILL_SWITCH_DRAWDOWN_PCT))
        )

        # Daily loss tracking
        self._daily_realized_loss: Decimal = Decimal("0")
        self._daily_start_equity: Decimal = initial_equity
        self._current_day: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._daily_loss_limit_hit: bool = False

    @property
    def daily_realized_loss(self) -> Decimal:
        """Sum of losses from closed positions since start of current trading day."""
        return self._daily_realized_loss

    @property
    def daily_loss_limit_hit(self) -> bool:
        """Whether the daily loss limit has been breached today."""
        return self._daily_loss_limit_hit

    def update_equity(self, equity: Decimal) -> None:
        """Update peak equity if a new high is reached.

        Args:
            equity: Current account equity value.
        """
        if equity > self.peak_equity:
            self.peak_equity = equity

    def get_drawdown(self, current_equity: Decimal) -> Decimal:
        """Calculate current drawdown as a fraction of peak equity.

        Returns:
            Drawdown as a Decimal fraction (e.g. Decimal("0.12") means 12% drawdown).
            Returns Decimal("0") if current equity is at or above peak.
        """
        if self.peak_equity <= Decimal("0"):
            return Decimal("0")
        if current_equity >= self.peak_equity:
            return Decimal("0")
        return (self.peak_equity - current_equity) / self.peak_equity

    def _reset_daily_if_needed(self) -> None:
        """Reset daily loss tracking if a new UTC day has started."""
        now = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if now > self._current_day:
            self._daily_realized_loss = Decimal("0")
            self._daily_loss_limit_hit = False
            self._current_day = now

    def reset_daily(self, start_of_day_equity: Decimal) -> None:
        """Manually reset daily loss tracking for a new trading day.

        This is useful for explicit day boundary handling and testing.

        Args:
            start_of_day_equity: Account equity at the start of the new day.
        """
        self._daily_realized_loss = Decimal("0")
        self._daily_start_equity = start_of_day_equity
        self._daily_loss_limit_hit = False
        self._current_day = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def update_on_trade_close(self, pnl: Decimal) -> None:
        """Update daily loss tracking when a trade is closed.

        Only negative PnL (losses) contribute to the daily realized loss total.

        Args:
            pnl: Profit or loss from the closed trade. Negative values are losses.
        """
        self._reset_daily_if_needed()
        if pnl < Decimal("0"):
            self._daily_realized_loss += abs(pnl)
        # Check if daily loss limit is now breached
        daily_limit = self._daily_start_equity * self.daily_max_loss_pct
        if self._daily_realized_loss >= daily_limit:
            self._daily_loss_limit_hit = True

    def check_daily_loss_limit(self, current_equity: Decimal) -> bool:
        """Check if the daily max loss limit has been breached.

        Args:
            current_equity: Current account equity (used for context, not calculation).

        Returns:
            True if the daily loss limit has been breached, False otherwise.
        """
        self._reset_daily_if_needed()
        daily_limit = self._daily_start_equity * self.daily_max_loss_pct
        return self._daily_realized_loss >= daily_limit

    def get_drawdown_pct(self, current_equity: Decimal) -> float:
        """Return the current drawdown percentage as a float.

        Convenience method that returns the drawdown as a Python float
        for use in logging, metrics, and display contexts.

        Args:
            current_equity: Current account equity value.

        Returns:
            Current drawdown as a float (e.g. 0.12 means 12% drawdown).
        """
        return float(self.get_drawdown(current_equity))

    def should_trigger_kill_switch(self, current_equity: Decimal) -> bool:
        """Check if the kill switch should be triggered based on current equity.

        Returns True if the drawdown from peak equity exceeds the kill switch
        threshold (default 15%).

        Args:
            current_equity: Current account equity value.

        Returns:
            True if drawdown >= kill_switch_pct, False otherwise.
        """
        return self.get_drawdown(current_equity) >= self.kill_switch_pct

    def check_trade_allowed(self, current_equity: Decimal) -> DrawdownCheckResult:
        """Evaluate whether a new trade is allowed given current drawdown and daily loss.

        Checks are evaluated in priority order:
        1. Kill switch trigger (drawdown > 15% from peak)
        2. Daily max loss limit (reject all signals until next day)
        3. Drawdown-based size reduction (drawdown > 10% from peak)
        4. Allow trade

        Args:
            current_equity: Current account equity value.

        Returns:
            DrawdownCheckResult with the decision, reason, optional reduction factor,
            and current drawdown percentage.
        """
        self._reset_daily_if_needed()

        # Update peak equity
        self.update_equity(current_equity)

        # Calculate current drawdown
        drawdown_pct = self.get_drawdown(current_equity)

        # Priority 1: Kill switch at 15% drawdown
        if drawdown_pct >= self.kill_switch_pct:
            return DrawdownCheckResult(
                decision=TradeDecision.KILL_SWITCH,
                reason=(
                    f"Drawdown {drawdown_pct:.4f} exceeds kill switch threshold "
                    f"{self.kill_switch_pct:.4f}"
                ),
                reduction_factor=None,
                drawdown_pct=drawdown_pct,
            )

        # Priority 2: Daily loss limit
        if self._daily_loss_limit_hit or self.check_daily_loss_limit(current_equity):
            return DrawdownCheckResult(
                decision=TradeDecision.REJECT,
                reason=(
                    f"Daily realized loss {self._daily_realized_loss} exceeds limit "
                    f"({self.daily_max_loss_pct * 100}% of start-of-day equity "
                    f"{self._daily_start_equity})"
                ),
                reduction_factor=None,
                drawdown_pct=drawdown_pct,
            )

        # Priority 3: Drawdown-based size reduction at 10%
        if drawdown_pct >= self.drawdown_reduction_pct:
            reduction = ReductionFactor(
                factor=Decimal(str(DRAWDOWN_SIZE_REDUCTION_FACTOR)),
                reason=(
                    f"Drawdown {drawdown_pct:.4f} exceeds reduction threshold "
                    f"{self.drawdown_reduction_pct:.4f} — applying "
                    f"{DRAWDOWN_SIZE_REDUCTION_FACTOR} size factor (75% reduction)"
                ),
            )
            return DrawdownCheckResult(
                decision=TradeDecision.REDUCE_SIZE,
                reason=reduction.reason,
                reduction_factor=reduction,
                drawdown_pct=drawdown_pct,
            )

        # All clear
        return DrawdownCheckResult(
            decision=TradeDecision.ALLOW,
            reason=None,
            reduction_factor=None,
            drawdown_pct=drawdown_pct,
        )
