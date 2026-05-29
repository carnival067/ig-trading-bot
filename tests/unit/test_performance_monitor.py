"""Unit tests for the Strategy Performance Monitor.

Tests cover rolling 30-day performance tracking, auto-disable trigger,
forced liquidation escalation, weekly re-evaluation, and suspension logic.

Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from src.strategy.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    StrategyState,
    StrategyStatus,
    TradeRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> PerformanceMonitor:
    """Create a basic performance monitor."""
    return PerformanceMonitor()


@pytest.fixture
def monitor_with_callbacks() -> tuple[PerformanceMonitor, dict]:
    """Create a monitor with tracking callbacks."""
    events: dict = {"notifications": [], "closed": [], "liquidated": [], "backtests": {}}

    async def notify_cb(**kwargs):
        events["notifications"].append(kwargs)

    async def close_cb(strategy_name: str) -> bool:
        events["closed"].append(strategy_name)
        return True

    async def force_liquidation_cb(strategy_name: str):
        events["liquidated"].append(strategy_name)

    async def backtest_cb(strategy_name: str) -> float:
        return events["backtests"].get(strategy_name, 0.5)

    m = PerformanceMonitor(
        notify_callback=notify_cb,
        close_positions_callback=close_cb,
        force_liquidation_callback=force_liquidation_cb,
        backtest_callback=backtest_cb,
    )
    return m, events


def _make_trades(
    strategy_name: str,
    count: int,
    pnl_values: list[float] | None = None,
    base_time: datetime | None = None,
) -> list[TradeRecord]:
    """Helper to create trade records."""
    if base_time is None:
        base_time = datetime.utcnow() - timedelta(days=15)

    if pnl_values is None:
        pnl_values = [100.0] * count

    trades = []
    for i, pnl in enumerate(pnl_values):
        trades.append(
            TradeRecord(
                strategy_name=strategy_name,
                pnl=pnl,
                closed_at=base_time + timedelta(hours=i),
                is_winner=pnl > 0,
            )
        )
    return trades


# ---------------------------------------------------------------------------
# Task 25.1: Rolling 30-day performance tracking
# ---------------------------------------------------------------------------


class TestRolling30DayPerformanceTracking:
    """Tests for rolling 30-day performance metric calculation."""

    def test_calculate_metrics_no_trades_returns_none(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        result = monitor.calculate_metrics("test_strategy")
        assert result is None

    def test_calculate_metrics_with_trades(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        trades = _make_trades("test_strategy", 10, [100, -50, 200, -30, 150, 80, -20, 300, -10, 50])
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.trade_count == 10
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_win_rate_calculation(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        # 7 winners, 3 losers
        pnls = [100, 200, 300, -50, 100, -30, 200, -10, 100, 50]
        trades = _make_trades("test_strategy", 10, pnls)
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.win_rate == 0.7

    def test_profit_factor_calculation(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        # Gross profit = 300, Gross loss = 100
        pnls = [100, -50, 200, -50]
        trades = _make_trades("test_strategy", 4, pnls)
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.profit_factor == 3.0

    def test_profit_factor_no_losses(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        pnls = [100, 200, 300]
        trades = _make_trades("test_strategy", 3, pnls)
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.profit_factor == float("inf")

    def test_only_trades_within_30_day_window(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        now = datetime.utcnow()

        # Old trade (outside window)
        old_trade = TradeRecord(
            strategy_name="test_strategy",
            pnl=-1000,
            closed_at=now - timedelta(days=35),
            is_winner=False,
        )
        # Recent trade (inside window)
        recent_trade = TradeRecord(
            strategy_name="test_strategy",
            pnl=100,
            closed_at=now - timedelta(days=5),
            is_winner=True,
        )
        monitor.record_trade(old_trade)
        monitor.record_trade(recent_trade)

        metrics = monitor.calculate_metrics("test_strategy", as_of=now)
        assert metrics is not None
        assert metrics.trade_count == 1
        assert metrics.win_rate == 1.0

    def test_sharpe_ratio_positive_for_consistent_profits(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        pnls = [100, 110, 105, 95, 100, 108, 102, 98, 103, 97]
        trades = _make_trades("test_strategy", 10, pnls)
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.sharpe_ratio > 0

    def test_sharpe_ratio_negative_for_consistent_losses(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        pnls = [-100, -110, -105, -95, -100, -108, -102, -98, -103, -97]
        trades = _make_trades("test_strategy", 10, pnls)
        for t in trades:
            monitor.record_trade(t)

        metrics = monitor.calculate_metrics("test_strategy")
        assert metrics is not None
        assert metrics.sharpe_ratio < 0


# ---------------------------------------------------------------------------
# Task 25.2: Auto-disable trigger
# ---------------------------------------------------------------------------


class TestAutoDisableTrigger:
    """Tests for auto-disable on two consecutive low Sharpe evaluations."""

    @pytest.mark.asyncio
    async def test_single_low_sharpe_does_not_disable(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        # Create trades with negative Sharpe
        pnls = [-100, -50, -80, -60, -90, 10, -70, -40, -110, -30]
        trades = _make_trades("test_strategy", 10, pnls)
        for t in trades:
            monitor.record_trade(t)

        await monitor.evaluate_strategy("test_strategy")
        assert monitor.get_strategy_status("test_strategy") == StrategyStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_two_consecutive_low_sharpe_disables(self):
        events: dict = {"notifications": [], "closed": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def close_cb(strategy_name: str) -> bool:
            events["closed"].append(strategy_name)
            return True

        m = PerformanceMonitor(
            notify_callback=notify_cb,
            close_positions_callback=close_cb,
        )
        m.register_strategy("test_strategy")

        # Create trades with very negative Sharpe (consistent losses)
        pnls = [-100, -50, -80, -60, -90, 10, -70, -40, -110, -30]
        base_time = datetime.utcnow() - timedelta(days=15)
        trades = _make_trades("test_strategy", 10, pnls, base_time=base_time)
        for t in trades:
            m.record_trade(t)

        # First evaluation
        await m.evaluate_strategy("test_strategy")
        assert m.get_strategy_status("test_strategy") == StrategyStatus.ACTIVE

        # Second evaluation (consecutive)
        await m.evaluate_strategy("test_strategy")
        assert m.get_strategy_status("test_strategy") == StrategyStatus.DISABLED
        assert len(events["notifications"]) == 1
        assert events["notifications"][0]["event"] == "strategy_disabled"
        assert len(events["closed"]) == 1

    @pytest.mark.asyncio
    async def test_good_sharpe_resets_consecutive_count(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")

        # First: bad trades (low Sharpe)
        bad_pnls = [-100, -50, -80, -60, -90, 10, -70, -40, -110, -30]
        base_time = datetime.utcnow() - timedelta(days=15)
        trades = _make_trades("test_strategy", 10, bad_pnls, base_time=base_time)
        for t in trades:
            monitor.record_trade(t)

        await monitor.evaluate_strategy("test_strategy")
        state = monitor._strategies["test_strategy"]
        assert state.consecutive_low_sharpe == 1

        # Now add good trades to improve Sharpe
        monitor._trade_history.clear()
        good_pnls = [200, 180, 190, 210, 195, 205, 185, 200, 215, 190]
        trades = _make_trades("test_strategy", 10, good_pnls, base_time=base_time)
        for t in trades:
            monitor.record_trade(t)

        await monitor.evaluate_strategy("test_strategy")
        assert state.consecutive_low_sharpe == 0
        assert state.status == StrategyStatus.ACTIVE


# ---------------------------------------------------------------------------
# Task 25.3: Forced liquidation escalation
# ---------------------------------------------------------------------------


class TestForcedLiquidationEscalation:
    """Tests for forced liquidation when positions can't close in time."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_forced_liquidation(self):
        events: dict = {"notifications": [], "liquidated": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def slow_close_cb(strategy_name: str) -> bool:
            # Simulate a slow close that will timeout
            await asyncio.sleep(5)
            return True

        async def force_liquidation_cb(strategy_name: str):
            events["liquidated"].append(strategy_name)

        m = PerformanceMonitor(
            close_positions_timeout_seconds=1,  # Very short timeout for testing
            notify_callback=notify_cb,
            close_positions_callback=slow_close_cb,
            force_liquidation_callback=force_liquidation_cb,
        )
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        metrics = PerformanceMetrics(sharpe_ratio=0.3, win_rate=0.4, profit_factor=0.8)

        await m._disable_strategy(state, metrics)

        assert "test_strategy" in events["liquidated"]

    @pytest.mark.asyncio
    async def test_close_failure_triggers_forced_liquidation(self):
        events: dict = {"notifications": [], "liquidated": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def failing_close_cb(strategy_name: str) -> bool:
            return False  # Positions could not be closed

        async def force_liquidation_cb(strategy_name: str):
            events["liquidated"].append(strategy_name)

        m = PerformanceMonitor(
            notify_callback=notify_cb,
            close_positions_callback=failing_close_cb,
            force_liquidation_callback=force_liquidation_cb,
        )
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        metrics = PerformanceMetrics(sharpe_ratio=0.3, win_rate=0.4, profit_factor=0.8)

        await m._disable_strategy(state, metrics)

        assert "test_strategy" in events["liquidated"]

    @pytest.mark.asyncio
    async def test_successful_close_no_escalation(self):
        events: dict = {"notifications": [], "closed": [], "liquidated": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def close_cb(strategy_name: str) -> bool:
            events["closed"].append(strategy_name)
            return True

        async def force_liquidation_cb(strategy_name: str):
            events["liquidated"].append(strategy_name)

        m = PerformanceMonitor(
            notify_callback=notify_cb,
            close_positions_callback=close_cb,
            force_liquidation_callback=force_liquidation_cb,
        )
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        metrics = PerformanceMetrics(sharpe_ratio=0.3, win_rate=0.4, profit_factor=0.8)

        await m._disable_strategy(state, metrics)

        assert "test_strategy" in events["closed"]
        assert "test_strategy" not in events["liquidated"]


# ---------------------------------------------------------------------------
# Task 25.4: Weekly re-evaluation of disabled strategies
# ---------------------------------------------------------------------------


class TestWeeklyReEvaluation:
    """Tests for weekly re-evaluation of disabled strategies."""

    @pytest.mark.asyncio
    async def test_re_enable_when_oos_sharpe_above_threshold(self):
        events: dict = {"notifications": [], "backtests": {"test_strategy": 1.5}}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def backtest_cb(strategy_name: str) -> float:
            return events["backtests"].get(strategy_name, 0.5)

        m = PerformanceMonitor(
            notify_callback=notify_cb,
            backtest_callback=backtest_cb,
        )
        m.register_strategy("test_strategy")
        m._strategies["test_strategy"].status = StrategyStatus.DISABLED

        re_enabled = await m.weekly_re_evaluate()

        assert "test_strategy" in re_enabled
        assert m.get_strategy_status("test_strategy") == StrategyStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_remain_disabled_when_oos_sharpe_below_threshold(self):
        events: dict = {"backtests": {"test_strategy": 0.7}}

        async def backtest_cb(strategy_name: str) -> float:
            return events["backtests"].get(strategy_name, 0.5)

        m = PerformanceMonitor(backtest_callback=backtest_cb)
        m.register_strategy("test_strategy")
        m._strategies["test_strategy"].status = StrategyStatus.DISABLED

        re_enabled = await m.weekly_re_evaluate()

        assert "test_strategy" not in re_enabled
        assert m.get_strategy_status("test_strategy") == StrategyStatus.DISABLED

    @pytest.mark.asyncio
    async def test_suspended_strategies_not_re_evaluated(self):
        events: dict = {"backtests": {"test_strategy": 2.0}}

        async def backtest_cb(strategy_name: str) -> float:
            events["backtests"]["called"] = True
            return 2.0

        m = PerformanceMonitor(backtest_callback=backtest_cb)
        m.register_strategy("test_strategy")
        m._strategies["test_strategy"].status = StrategyStatus.SUSPENDED

        re_enabled = await m.weekly_re_evaluate()

        assert "test_strategy" not in re_enabled
        assert m.get_strategy_status("test_strategy") == StrategyStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_re_enable_sets_re_enabled_at(self):
        async def backtest_cb(strategy_name: str) -> float:
            return 1.5

        m = PerformanceMonitor(backtest_callback=backtest_cb)
        m.register_strategy("test_strategy")
        m._strategies["test_strategy"].status = StrategyStatus.DISABLED

        now = datetime.utcnow()
        await m.weekly_re_evaluate(current_time=now)

        state = m._strategies["test_strategy"]
        assert state.re_enabled_at == now


# ---------------------------------------------------------------------------
# Task 25.5: Suspension logic
# ---------------------------------------------------------------------------


class TestSuspensionLogic:
    """Tests for suspension when re-disabled within 14 days."""

    @pytest.mark.asyncio
    async def test_re_disabled_within_14_days_triggers_suspension(self):
        events: dict = {"notifications": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        m = PerformanceMonitor(notify_callback=notify_cb)
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        state.re_enabled_at = datetime.utcnow() - timedelta(days=10)

        # Simulate re-disable
        suspended = await m.check_suspension(state)

        assert suspended is True
        assert state.status == StrategyStatus.SUSPENDED
        assert len(events["notifications"]) == 1
        assert events["notifications"][0]["event"] == "strategy_suspended"

    @pytest.mark.asyncio
    async def test_re_disabled_after_14_days_no_suspension(self):
        m = PerformanceMonitor()
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        state.status = StrategyStatus.DISABLED
        state.re_enabled_at = datetime.utcnow() - timedelta(days=20)

        suspended = await m.check_suspension(state)

        assert suspended is False
        assert state.status == StrategyStatus.DISABLED

    @pytest.mark.asyncio
    async def test_no_re_enabled_at_no_suspension(self):
        m = PerformanceMonitor()
        m.register_strategy("test_strategy")

        state = m._strategies["test_strategy"]
        state.status = StrategyStatus.DISABLED
        state.re_enabled_at = None

        suspended = await m.check_suspension(state)

        assert suspended is False

    @pytest.mark.asyncio
    async def test_full_disable_with_suspension_check(self):
        events: dict = {"notifications": [], "closed": []}

        async def notify_cb(**kwargs):
            events["notifications"].append(kwargs)

        async def close_cb(strategy_name: str) -> bool:
            events["closed"].append(strategy_name)
            return True

        m = PerformanceMonitor(
            notify_callback=notify_cb,
            close_positions_callback=close_cb,
        )
        m.register_strategy("test_strategy")

        # Simulate previous re-enablement 5 days ago
        state = m._strategies["test_strategy"]
        state.re_enabled_at = datetime.utcnow() - timedelta(days=5)

        metrics = PerformanceMetrics(sharpe_ratio=0.3, win_rate=0.4, profit_factor=0.8)
        await m.disable_strategy_with_suspension_check("test_strategy", metrics)

        assert state.status == StrategyStatus.SUSPENDED


# ---------------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------------


class TestPerformanceMonitorGeneral:
    """General tests for the performance monitor."""

    def test_register_strategy(self, monitor: PerformanceMonitor):
        monitor.register_strategy("my_strategy")
        assert "my_strategy" in monitor.strategies
        assert monitor.get_strategy_status("my_strategy") == StrategyStatus.ACTIVE

    def test_register_duplicate_strategy_no_error(self, monitor: PerformanceMonitor):
        monitor.register_strategy("my_strategy")
        monitor.register_strategy("my_strategy")
        assert len(monitor.strategies) == 1

    def test_get_status_unregistered_returns_none(self, monitor: PerformanceMonitor):
        assert monitor.get_strategy_status("unknown") is None

    def test_is_strategy_active(self, monitor: PerformanceMonitor):
        monitor.register_strategy("active_one")
        assert monitor.is_strategy_active("active_one") is True
        assert monitor.is_strategy_active("unknown") is False

    def test_record_trade(self, monitor: PerformanceMonitor):
        trade = TradeRecord(
            strategy_name="test",
            pnl=100.0,
            closed_at=datetime.utcnow(),
            is_winner=True,
        )
        monitor.record_trade(trade)
        assert len(monitor._trade_history) == 1

    @pytest.mark.asyncio
    async def test_evaluate_all_strategies(self, monitor: PerformanceMonitor):
        monitor.register_strategy("s1")
        monitor.register_strategy("s2")

        trades = _make_trades("s1", 5, [100, 200, -50, 150, 80])
        for t in trades:
            monitor.record_trade(t)

        results = await monitor.evaluate_all_strategies()
        assert "s1" in results
        assert "s2" in results
        assert results["s1"] is not None
        assert results["s2"] is None  # No trades for s2

    def test_get_metrics_history(self, monitor: PerformanceMonitor):
        monitor.register_strategy("test_strategy")
        state = monitor._strategies["test_strategy"]
        state.metrics_history.append(
            PerformanceMetrics(sharpe_ratio=1.0, win_rate=0.6, profit_factor=2.0)
        )
        state.metrics_history.append(
            PerformanceMetrics(sharpe_ratio=0.8, win_rate=0.55, profit_factor=1.5)
        )

        history = monitor.get_metrics_history("test_strategy")
        assert len(history) == 2
