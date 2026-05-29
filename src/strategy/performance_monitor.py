"""Strategy Performance Monitoring and Auto-Disable.

Tracks rolling 30-day performance per strategy, auto-disables underperforming
strategies, handles forced liquidation escalation, weekly re-evaluation of
disabled strategies, and suspension logic.

Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine

from src.config.constants import (
    STRATEGY_DISABLE_SHARPE_THRESHOLD,
    STRATEGY_ENABLE_SHARPE_THRESHOLD,
    STRATEGY_SUSPENSION_DAYS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class StrategyStatus(Enum):
    """Status of a strategy in the performance monitoring system."""

    ACTIVE = "active"
    DISABLED = "disabled"
    SUSPENDED = "suspended"


@dataclass
class PerformanceMetrics:
    """Rolling 30-day performance metrics for a strategy.

    Attributes:
        sharpe_ratio: Rolling 30-day Sharpe ratio.
        win_rate: Fraction of winning trades (0.0 to 1.0).
        profit_factor: Gross profit / gross loss.
        trade_count: Number of trades in the window.
        calculated_at: Timestamp of calculation.
    """

    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    trade_count: int = 0
    calculated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyState:
    """Internal state for a monitored strategy.

    Attributes:
        strategy_name: Identifier for the strategy.
        status: Current status (active, disabled, suspended).
        metrics_history: List of historical performance evaluations.
        disabled_at: When the strategy was disabled.
        re_enabled_at: When the strategy was last re-enabled.
        consecutive_low_sharpe: Count of consecutive evaluations below threshold.
        last_evaluation_at: Timestamp of last evaluation.
    """

    strategy_name: str
    status: StrategyStatus = StrategyStatus.ACTIVE
    metrics_history: list[PerformanceMetrics] = field(default_factory=list)
    disabled_at: datetime | None = None
    re_enabled_at: datetime | None = None
    consecutive_low_sharpe: int = 0
    last_evaluation_at: datetime | None = None


@dataclass
class TradeRecord:
    """A closed trade record for performance calculation.

    Attributes:
        strategy_name: Strategy that generated the trade.
        pnl: Profit/loss of the trade.
        closed_at: When the trade was closed.
        is_winner: Whether the trade was profitable.
    """

    strategy_name: str
    pnl: float
    closed_at: datetime
    is_winner: bool


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Monitors strategy performance and manages auto-disable/re-enable lifecycle.

    Tracks rolling 30-day performance metrics per strategy, auto-disables
    strategies with Sharpe < 0.5 on two consecutive evaluations, handles
    forced liquidation escalation, weekly re-evaluation, and suspension logic.

    Args:
        disable_sharpe_threshold: Sharpe below which strategy is flagged.
        enable_sharpe_threshold: OOS Sharpe above which disabled strategy is re-enabled.
        suspension_days: Days within which re-disable triggers suspension.
        evaluation_interval_hours: Hours between performance evaluations.
        close_positions_timeout_seconds: Seconds allowed to close positions.
        notify_callback: Async callback for notifications.
        close_positions_callback: Async callback to close positions for a strategy.
        force_liquidation_callback: Async callback for forced liquidation escalation.
        backtest_callback: Async callback to run OOS backtest for a strategy.
    """

    def __init__(
        self,
        disable_sharpe_threshold: float = STRATEGY_DISABLE_SHARPE_THRESHOLD,
        enable_sharpe_threshold: float = STRATEGY_ENABLE_SHARPE_THRESHOLD,
        suspension_days: int = STRATEGY_SUSPENSION_DAYS,
        evaluation_interval_hours: int = 24,
        close_positions_timeout_seconds: int = 60,
        notify_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        close_positions_callback: Callable[..., Coroutine[Any, Any, bool]] | None = None,
        force_liquidation_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        backtest_callback: Callable[..., Coroutine[Any, Any, float]] | None = None,
    ) -> None:
        self._disable_sharpe_threshold = disable_sharpe_threshold
        self._enable_sharpe_threshold = enable_sharpe_threshold
        self._suspension_days = suspension_days
        self._evaluation_interval_hours = evaluation_interval_hours
        self._close_positions_timeout_seconds = close_positions_timeout_seconds
        self._notify_callback = notify_callback
        self._close_positions_callback = close_positions_callback
        self._force_liquidation_callback = force_liquidation_callback
        self._backtest_callback = backtest_callback

        self._strategies: dict[str, StrategyState] = {}
        self._trade_history: list[TradeRecord] = []

    @property
    def strategies(self) -> dict[str, StrategyState]:
        """Current strategy states."""
        return dict(self._strategies)

    def register_strategy(self, strategy_name: str) -> None:
        """Register a strategy for performance monitoring.

        Args:
            strategy_name: Unique identifier for the strategy.
        """
        if strategy_name not in self._strategies:
            self._strategies[strategy_name] = StrategyState(strategy_name=strategy_name)
            logger.info("Registered strategy for monitoring: %s", strategy_name)

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a closed trade for performance tracking.

        Args:
            trade: The closed trade record.
        """
        self._trade_history.append(trade)

    def get_strategy_status(self, strategy_name: str) -> StrategyStatus | None:
        """Get the current status of a strategy.

        Args:
            strategy_name: Strategy identifier.

        Returns:
            The strategy's status, or None if not registered.
        """
        state = self._strategies.get(strategy_name)
        return state.status if state else None

    # -----------------------------------------------------------------------
    # Task 25.1: Rolling 30-day performance tracking
    # -----------------------------------------------------------------------

    def calculate_metrics(
        self, strategy_name: str, as_of: datetime | None = None
    ) -> PerformanceMetrics | None:
        """Calculate rolling 30-day performance metrics for a strategy.

        Computes Sharpe ratio, win rate, and profit factor from the trade
        history within the 30-day window ending at `as_of`.

        Args:
            strategy_name: Strategy to calculate metrics for.
            as_of: Reference time for the 30-day window (defaults to now).

        Returns:
            PerformanceMetrics if there are trades, None otherwise.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        window_start = as_of - timedelta(days=30)

        # Filter trades for this strategy within the window
        trades = [
            t
            for t in self._trade_history
            if t.strategy_name == strategy_name
            and window_start <= t.closed_at <= as_of
        ]

        if not trades:
            return None

        # Win rate
        winners = sum(1 for t in trades if t.is_winner)
        win_rate = winners / len(trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio (simplified: mean return / std of returns)
        returns = [t.pnl for t in trades]
        mean_return = sum(returns) / len(returns)
        if len(returns) > 1:
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_return = variance**0.5
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        metrics = PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=len(trades),
            calculated_at=as_of,
        )

        return metrics

    async def evaluate_strategy(
        self, strategy_name: str, as_of: datetime | None = None
    ) -> PerformanceMetrics | None:
        """Evaluate a strategy's performance and update its state.

        Recalculates metrics and checks if auto-disable should trigger.
        This is called every 24 hours per strategy.

        Args:
            strategy_name: Strategy to evaluate.
            as_of: Reference time for evaluation.

        Returns:
            The calculated metrics, or None if no trades.
        """
        state = self._strategies.get(strategy_name)
        if state is None:
            logger.warning("Strategy not registered: %s", strategy_name)
            return None

        if state.status == StrategyStatus.SUSPENDED:
            logger.debug("Skipping evaluation for suspended strategy: %s", strategy_name)
            return None

        metrics = self.calculate_metrics(strategy_name, as_of)
        if metrics is None:
            state.last_evaluation_at = as_of or datetime.utcnow()
            return None

        # Store metrics in history
        state.metrics_history.append(metrics)
        state.last_evaluation_at = metrics.calculated_at

        # Check auto-disable condition (only for active strategies)
        if state.status == StrategyStatus.ACTIVE:
            if metrics.sharpe_ratio < self._disable_sharpe_threshold:
                state.consecutive_low_sharpe += 1
            else:
                state.consecutive_low_sharpe = 0

            # Two consecutive evaluations below threshold → disable
            if state.consecutive_low_sharpe >= 2:
                await self._disable_strategy(state, metrics)

        logger.info(
            "Evaluated strategy %s: sharpe=%.3f win_rate=%.3f profit_factor=%.3f "
            "consecutive_low=%d status=%s",
            strategy_name,
            metrics.sharpe_ratio,
            metrics.win_rate,
            metrics.profit_factor,
            state.consecutive_low_sharpe,
            state.status.value,
        )

        return metrics

    # -----------------------------------------------------------------------
    # Task 25.2: Auto-disable trigger
    # -----------------------------------------------------------------------

    async def _disable_strategy(
        self, state: StrategyState, metrics: PerformanceMetrics
    ) -> None:
        """Disable a strategy and close its positions.

        Args:
            state: The strategy state to disable.
            metrics: The metrics at time of disabling.
        """
        state.status = StrategyStatus.DISABLED
        state.disabled_at = datetime.utcnow()
        state.consecutive_low_sharpe = 0

        logger.warning(
            "Auto-disabling strategy %s: sharpe=%.3f win_rate=%.3f profit_factor=%.3f",
            state.strategy_name,
            metrics.sharpe_ratio,
            metrics.win_rate,
            metrics.profit_factor,
        )

        # Notify
        if self._notify_callback:
            await self._notify_callback(
                event="strategy_disabled",
                strategy_name=state.strategy_name,
                sharpe_ratio=metrics.sharpe_ratio,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                timestamp=state.disabled_at.isoformat(),
            )

        # Close positions within timeout
        await self._close_strategy_positions(state)

    async def _close_strategy_positions(self, state: StrategyState) -> None:
        """Close all positions for a disabled strategy within timeout.

        If positions cannot be closed within 60 seconds, escalates to
        forced liquidation.

        Args:
            state: The strategy state whose positions need closing.
        """
        if self._close_positions_callback is None:
            logger.debug("No close_positions_callback configured, skipping position close")
            return

        try:
            closed = await asyncio.wait_for(
                self._close_positions_callback(state.strategy_name),
                timeout=self._close_positions_timeout_seconds,
            )
            if closed:
                logger.info(
                    "Successfully closed positions for strategy %s",
                    state.strategy_name,
                )
            else:
                # Positions could not be closed within the callback
                await self._escalate_forced_liquidation(state)
        except asyncio.TimeoutError:
            # Task 25.3: Escalate to forced liquidation
            logger.error(
                "Timeout closing positions for strategy %s, escalating to forced liquidation",
                state.strategy_name,
            )
            await self._escalate_forced_liquidation(state)

    # -----------------------------------------------------------------------
    # Task 25.3: Forced liquidation escalation
    # -----------------------------------------------------------------------

    async def _escalate_forced_liquidation(self, state: StrategyState) -> None:
        """Escalate to forced liquidation when positions cannot close in time.

        Args:
            state: The strategy state requiring forced liquidation.
        """
        logger.warning(
            "Escalating to forced liquidation for strategy %s",
            state.strategy_name,
        )

        if self._force_liquidation_callback:
            await self._force_liquidation_callback(state.strategy_name)

        if self._notify_callback:
            await self._notify_callback(
                event="forced_liquidation_escalation",
                strategy_name=state.strategy_name,
                timestamp=datetime.utcnow().isoformat(),
            )

    # -----------------------------------------------------------------------
    # Task 25.4: Weekly re-evaluation of disabled strategies
    # -----------------------------------------------------------------------

    async def weekly_re_evaluate(self, current_time: datetime | None = None) -> list[str]:
        """Re-evaluate disabled strategies for potential re-enablement.

        Runs OOS backtest on each disabled (non-suspended) strategy.
        Re-enables if OOS Sharpe > 1.0.

        Args:
            current_time: Current time for evaluation context.

        Returns:
            List of strategy names that were re-enabled.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        re_enabled: list[str] = []

        for state in self._strategies.values():
            if state.status != StrategyStatus.DISABLED:
                continue

            # Run OOS backtest
            oos_sharpe = await self._run_oos_backtest(state.strategy_name)

            if oos_sharpe is not None and oos_sharpe > self._enable_sharpe_threshold:
                state.status = StrategyStatus.ACTIVE
                state.re_enabled_at = current_time
                state.consecutive_low_sharpe = 0
                re_enabled.append(state.strategy_name)

                logger.info(
                    "Re-enabled strategy %s: OOS Sharpe=%.3f (threshold=%.3f)",
                    state.strategy_name,
                    oos_sharpe,
                    self._enable_sharpe_threshold,
                )

                if self._notify_callback:
                    await self._notify_callback(
                        event="strategy_re_enabled",
                        strategy_name=state.strategy_name,
                        oos_sharpe=oos_sharpe,
                        timestamp=current_time.isoformat(),
                    )
            else:
                logger.info(
                    "Strategy %s remains disabled: OOS Sharpe=%.3f (need > %.3f)",
                    state.strategy_name,
                    oos_sharpe if oos_sharpe is not None else 0.0,
                    self._enable_sharpe_threshold,
                )

        return re_enabled

    async def _run_oos_backtest(self, strategy_name: str) -> float | None:
        """Run out-of-sample backtest for a strategy.

        Args:
            strategy_name: Strategy to backtest.

        Returns:
            OOS Sharpe ratio, or None if backtest unavailable.
        """
        if self._backtest_callback:
            return await self._backtest_callback(strategy_name)
        return None

    # -----------------------------------------------------------------------
    # Task 25.5: Suspension logic
    # -----------------------------------------------------------------------

    async def check_suspension(
        self, state: StrategyState, current_time: datetime | None = None
    ) -> bool:
        """Check if a strategy should be suspended after being re-disabled.

        A strategy is suspended if it was re-enabled and then disabled again
        within STRATEGY_SUSPENSION_DAYS days.

        Args:
            state: The strategy state to check.
            current_time: Current time for comparison.

        Returns:
            True if the strategy was suspended.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if state.re_enabled_at is None:
            return False

        days_since_re_enable = (current_time - state.re_enabled_at).days

        if days_since_re_enable <= self._suspension_days:
            state.status = StrategyStatus.SUSPENDED
            logger.warning(
                "Strategy %s suspended: re-disabled within %d days of re-enablement "
                "(re-enabled at %s, disabled again at %s)",
                state.strategy_name,
                self._suspension_days,
                state.re_enabled_at.isoformat(),
                current_time.isoformat(),
            )

            if self._notify_callback:
                await self._notify_callback(
                    event="strategy_suspended",
                    strategy_name=state.strategy_name,
                    days_since_re_enable=days_since_re_enable,
                    timestamp=current_time.isoformat(),
                )
            return True

        return False

    async def disable_strategy_with_suspension_check(
        self, strategy_name: str, metrics: PerformanceMetrics
    ) -> None:
        """Disable a strategy and check if it should be suspended.

        This is the full disable flow that includes suspension logic.

        Args:
            strategy_name: Strategy to disable.
            metrics: Current metrics at time of disabling.
        """
        state = self._strategies.get(strategy_name)
        if state is None:
            return

        await self._disable_strategy(state, metrics)

        # Check suspension: if re-enabled recently, mark as suspended
        if state.re_enabled_at is not None:
            await self.check_suspension(state, state.disabled_at)

    async def evaluate_all_strategies(
        self, as_of: datetime | None = None
    ) -> dict[str, PerformanceMetrics | None]:
        """Evaluate all registered strategies.

        Called periodically (every 24 hours) to recalculate metrics
        and check auto-disable conditions.

        Args:
            as_of: Reference time for evaluation.

        Returns:
            Dict mapping strategy name to its metrics (or None).
        """
        results: dict[str, PerformanceMetrics | None] = {}

        for strategy_name in list(self._strategies.keys()):
            metrics = await self.evaluate_strategy(strategy_name, as_of)
            results[strategy_name] = metrics

        return results

    def is_strategy_active(self, strategy_name: str) -> bool:
        """Check if a strategy is currently active (not disabled or suspended).

        Args:
            strategy_name: Strategy identifier.

        Returns:
            True if the strategy is active.
        """
        state = self._strategies.get(strategy_name)
        if state is None:
            return False
        return state.status == StrategyStatus.ACTIVE

    def get_metrics_history(
        self, strategy_name: str, limit: int = 10
    ) -> list[PerformanceMetrics]:
        """Get recent metrics history for a strategy.

        Args:
            strategy_name: Strategy identifier.
            limit: Maximum number of entries to return.

        Returns:
            List of recent performance metrics.
        """
        state = self._strategies.get(strategy_name)
        if state is None:
            return []
        return state.metrics_history[-limit:]
