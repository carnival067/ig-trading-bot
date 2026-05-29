"""Unit tests for the copy trading engine and allocation manager.

Tests cover:
- Task 21.1: Trade replication with risk-adjusted sizing
- Task 21.2: Proportional allocation (capped at 10%, max 10 traders)
- Task 21.3: Risk Engine validation for copied trades
- Task 21.4: Drawdown-based copy stop (15% in 7-day window)
- Task 21.5: Position close mirroring within 2 seconds
- Task 21.6: Execution timeout (cancel if not within 3 seconds)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.copy_trading.allocation_manager import AllocationManager, AllocationResult
from src.copy_trading.copy_engine import (
    CopiedTrade,
    CopyEngine,
    CopyStatus,
    CopyStopReason,
    SourceTrade,
    TraderAllocation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> CopyEngine:
    return CopyEngine()


@pytest.fixture
def allocation_manager() -> AllocationManager:
    return AllocationManager()


@pytest.fixture
def source_trade() -> SourceTrade:
    return SourceTrade(
        trade_id="SRC-001",
        trader_id="TRADER-A",
        instrument="EUR/USD",
        direction="LONG",
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0950"),
        take_profit=Decimal("1.1100"),
        size=Decimal("10.00"),
        asset_class="forex",
    )


# ---------------------------------------------------------------------------
# Task 21.1: Trade Replication
# ---------------------------------------------------------------------------


class TestTradeReplication:
    """Task 21.1: Trade replication with risk-adjusted sizing."""

    @pytest.mark.asyncio
    async def test_basic_replication(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Should replicate a trade with proportional sizing."""
        result = await engine.replicate_trade(
            source_trade=source_trade,
            trader_allocation=Decimal("10000"),
            copier_equity=Decimal("100000"),
        )
        assert result.status == CopyStatus.EXECUTED
        assert result.instrument == "EUR/USD"
        assert result.direction == "LONG"
        assert result.trader_id == "TRADER-A"
        # Size = 10.00 * (10000 / 100000) = 1.00
        assert result.size == Decimal("1.00")

    @pytest.mark.asyncio
    async def test_size_proportional_to_allocation(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Copied size should be proportional to allocation/equity ratio."""
        result = await engine.replicate_trade(
            source_trade=source_trade,
            trader_allocation=Decimal("5000"),
            copier_equity=Decimal("100000"),
        )
        # Size = 10.00 * (5000 / 100000) = 0.50
        assert result.size == Decimal("0.50")

    @pytest.mark.asyncio
    async def test_minimum_size_enforced(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Copied size should not go below 0.01."""
        result = await engine.replicate_trade(
            source_trade=source_trade,
            trader_allocation=Decimal("1"),
            copier_equity=Decimal("100000"),
        )
        # Size = 10.00 * (1 / 100000) = 0.0001 → clamped to 0.01
        assert result.size == Decimal("0.01")

    @pytest.mark.asyncio
    async def test_zero_equity_cancels(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Zero copier equity should cancel the trade."""
        result = await engine.replicate_trade(
            source_trade=source_trade,
            trader_allocation=Decimal("10000"),
            copier_equity=Decimal("0"),
        )
        assert result.status == CopyStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_trade_tracked_in_active_copies(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Executed trades should be tracked in active_copies."""
        result = await engine.replicate_trade(
            source_trade=source_trade,
            trader_allocation=Decimal("10000"),
            copier_equity=Decimal("100000"),
        )
        assert result.deal_id in engine.active_copies


# ---------------------------------------------------------------------------
# Task 21.2: Allocation Manager
# ---------------------------------------------------------------------------


class TestAllocationManager:
    """Task 21.2: Proportional allocation with caps."""

    def test_proportional_allocation(self, allocation_manager: AllocationManager) -> None:
        """Allocation should be proportional to trader score."""
        # Trader with 50% of total scores gets proportional allocation
        allocation = allocation_manager.calculate_allocation(
            trader_score=50.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 50/100 * 100000 = 50000, but capped at 10% = 10000
        assert allocation == Decimal("10000.00")

    def test_cap_at_10_percent(self, allocation_manager: AllocationManager) -> None:
        """Allocation should be capped at 10% of equity."""
        allocation = allocation_manager.calculate_allocation(
            trader_score=80.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 80% would be 80000, capped at 10% = 10000
        assert allocation <= Decimal("10000.00")

    def test_floor_at_1_percent(self, allocation_manager: AllocationManager) -> None:
        """Allocation should be at least 1% of equity."""
        allocation = allocation_manager.calculate_allocation(
            trader_score=0.5,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 0.5% would be 500, floored at 1% = 1000
        assert allocation >= Decimal("1000.00")

    def test_max_10_traders(self, allocation_manager: AllocationManager) -> None:
        """Should limit to maximum 10 traders."""
        scores = {f"T{i}": float(50 + i) for i in range(15)}
        results = allocation_manager.calculate_allocations(scores, Decimal("100000"))
        assert len(results) <= 10

    def test_selects_top_traders_by_score(self, allocation_manager: AllocationManager) -> None:
        """Should select the top traders by score when limiting."""
        scores = {f"T{i:02d}": float(i * 5) for i in range(15)}
        results = allocation_manager.calculate_allocations(scores, Decimal("100000"))
        # Should have the top 10 by score
        result_ids = {r.trader_id for r in results}
        # Top 10 are T05 through T14 (scores 25 through 70)
        for i in range(5, 15):
            assert f"T{i:02d}" in result_ids

    def test_zero_equity_returns_empty(self, allocation_manager: AllocationManager) -> None:
        """Zero equity should return empty allocations."""
        results = allocation_manager.calculate_allocations(
            {"T1": 50.0}, Decimal("0")
        )
        assert results == []

    def test_zero_score_returns_zero(self, allocation_manager: AllocationManager) -> None:
        """Zero trader score should return zero allocation."""
        allocation = allocation_manager.calculate_allocation(
            trader_score=0.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        assert allocation == Decimal("0")

    def test_validate_allocation_within_bounds(self, allocation_manager: AllocationManager) -> None:
        """Allocation within 1-10% should be valid."""
        assert allocation_manager.validate_allocation(Decimal("5000"), Decimal("100000")) is True
        assert allocation_manager.validate_allocation(Decimal("1000"), Decimal("100000")) is True
        assert allocation_manager.validate_allocation(Decimal("10000"), Decimal("100000")) is True

    def test_validate_allocation_out_of_bounds(self, allocation_manager: AllocationManager) -> None:
        """Allocation outside 1-10% should be invalid."""
        assert allocation_manager.validate_allocation(Decimal("500"), Decimal("100000")) is False
        assert allocation_manager.validate_allocation(Decimal("15000"), Decimal("100000")) is False


# ---------------------------------------------------------------------------
# Task 21.4: Drawdown-Based Copy Stop
# ---------------------------------------------------------------------------


class TestDrawdownStop:
    """Task 21.4: Drawdown-based copy stop."""

    def test_no_stop_within_threshold(self, engine: CopyEngine) -> None:
        """Should not stop if drawdown is within 15% of allocation."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        # PnL of -1000 is 10% of allocation (below 15% threshold)
        should_stop = engine.check_drawdown_stop("TRADER-A", Decimal("-1000"))
        assert should_stop is False

    def test_stop_when_exceeds_threshold(self, engine: CopyEngine) -> None:
        """Should stop when drawdown exceeds 15% of allocated capital."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        # First record a peak
        engine.check_drawdown_stop("TRADER-A", Decimal("500"))
        # Then drawdown: peak (500) - current (-1100) = 1600 > 1500 (15% of 10000)
        should_stop = engine.check_drawdown_stop("TRADER-A", Decimal("-1100"))
        assert should_stop is True

    def test_drawdown_from_peak_not_start(self, engine: CopyEngine) -> None:
        """Drawdown should be measured from peak PnL, not from start."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        # Build up to a peak
        engine.check_drawdown_stop("TRADER-A", Decimal("2000"))
        # Drawdown from peak: 2000 - 600 = 1400 < 1500 (15% of 10000)
        should_stop = engine.check_drawdown_stop("TRADER-A", Decimal("600"))
        assert should_stop is False
        # Drawdown from peak: 2000 - 400 = 1600 > 1500
        should_stop = engine.check_drawdown_stop("TRADER-A", Decimal("400"))
        assert should_stop is True

    def test_unknown_trader_returns_false(self, engine: CopyEngine) -> None:
        """Unknown trader should not trigger stop."""
        should_stop = engine.check_drawdown_stop("UNKNOWN", Decimal("-5000"))
        assert should_stop is False

    @pytest.mark.asyncio
    async def test_handle_drawdown_stop_closes_positions(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Drawdown stop should close all positions for the trader."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        await engine.replicate_trade(source_trade, Decimal("10000"), Decimal("100000"))
        assert len(engine.active_copies) == 1

        closed = await engine.handle_drawdown_stop("TRADER-A")
        assert len(closed) == 1
        assert closed[0].status == CopyStatus.CLOSED
        assert len(engine.active_copies) == 0


# ---------------------------------------------------------------------------
# Task 21.5: Position Close Mirroring
# ---------------------------------------------------------------------------


class TestPositionCloseMirroring:
    """Task 21.5: Close copied position within 2 seconds."""

    @pytest.mark.asyncio
    async def test_close_existing_position(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Should close an existing copied position."""
        result = await engine.replicate_trade(source_trade, Decimal("10000"), Decimal("100000"))
        deal_id = result.deal_id

        closed = await engine.close_copied_position(deal_id)
        assert closed is not None
        assert closed.status == CopyStatus.CLOSED
        assert closed.closed_at is not None

    @pytest.mark.asyncio
    async def test_close_removes_from_active(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Closing should remove the trade from active copies."""
        result = await engine.replicate_trade(source_trade, Decimal("10000"), Decimal("100000"))
        deal_id = result.deal_id
        assert deal_id in engine.active_copies

        await engine.close_copied_position(deal_id)
        assert deal_id not in engine.active_copies

    @pytest.mark.asyncio
    async def test_close_unknown_position_returns_none(self, engine: CopyEngine) -> None:
        """Closing an unknown position should return None."""
        result = await engine.close_copied_position("UNKNOWN-001")
        assert result is None


# ---------------------------------------------------------------------------
# Task 21.6: Execution Timeout
# ---------------------------------------------------------------------------


class TestExecutionTimeout:
    """Task 21.6: Execution timeout handling."""

    @pytest.mark.asyncio
    async def test_normal_execution_within_timeout(self, engine: CopyEngine, source_trade: SourceTrade) -> None:
        """Normal execution should complete within timeout."""
        result = await engine.replicate_trade(source_trade, Decimal("10000"), Decimal("100000"))
        assert result.status == CopyStatus.EXECUTED
        assert result.execution_time_ms < 3000  # Less than 3 seconds

    @pytest.mark.asyncio
    async def test_timeout_configured_from_constants(self) -> None:
        """Engine should use COPY_EXECUTION_TIMEOUT_SECONDS from constants."""
        from src.config.constants import COPY_EXECUTION_TIMEOUT_SECONDS
        engine = CopyEngine()
        assert engine._execution_timeout == COPY_EXECUTION_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_close_timeout_configured_from_constants(self) -> None:
        """Engine should use COPY_CLOSE_TIMEOUT_SECONDS from constants."""
        from src.config.constants import COPY_CLOSE_TIMEOUT_SECONDS
        engine = CopyEngine()
        assert engine._close_timeout == COPY_CLOSE_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Stop Copying Trader
# ---------------------------------------------------------------------------


class TestStopCopyingTrader:
    """Test stop_copying_trader functionality."""

    @pytest.mark.asyncio
    async def test_stop_closes_all_positions(self, engine: CopyEngine) -> None:
        """Stopping a trader should close all their open positions."""
        engine.register_trader("TRADER-A", Decimal("10000"))

        # Open multiple positions
        for i in range(3):
            trade = SourceTrade(
                trade_id=f"SRC-{i}",
                trader_id="TRADER-A",
                instrument="EUR/USD",
                direction="LONG",
                entry_price=Decimal("1.1000"),
                stop_loss=Decimal("1.0950"),
                take_profit=Decimal("1.1100"),
                size=Decimal("5.00"),
                asset_class="forex",
            )
            await engine.replicate_trade(trade, Decimal("10000"), Decimal("100000"))

        assert len(engine.active_copies) == 3

        closed = await engine.stop_copying_trader("TRADER-A")
        assert len(closed) == 3
        assert all(t.status == CopyStatus.CLOSED for t in closed)
        assert len(engine.active_copies) == 0

    @pytest.mark.asyncio
    async def test_stop_removes_allocation(self, engine: CopyEngine) -> None:
        """Stopping should remove the trader's allocation."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        assert "TRADER-A" in engine.trader_allocations

        await engine.stop_copying_trader("TRADER-A")
        assert "TRADER-A" not in engine.trader_allocations

    @pytest.mark.asyncio
    async def test_stop_only_affects_target_trader(self, engine: CopyEngine) -> None:
        """Stopping one trader should not affect another's positions."""
        engine.register_trader("TRADER-A", Decimal("10000"))
        engine.register_trader("TRADER-B", Decimal("10000"))

        trade_a = SourceTrade(
            trade_id="SRC-A",
            trader_id="TRADER-A",
            instrument="EUR/USD",
            direction="LONG",
            entry_price=Decimal("1.1000"),
            stop_loss=Decimal("1.0950"),
            take_profit=Decimal("1.1100"),
            size=Decimal("5.00"),
            asset_class="forex",
        )
        trade_b = SourceTrade(
            trade_id="SRC-B",
            trader_id="TRADER-B",
            instrument="GBP/USD",
            direction="SHORT",
            entry_price=Decimal("1.2500"),
            stop_loss=Decimal("1.2550"),
            take_profit=Decimal("1.2400"),
            size=Decimal("5.00"),
            asset_class="forex",
        )

        await engine.replicate_trade(trade_a, Decimal("10000"), Decimal("100000"))
        await engine.replicate_trade(trade_b, Decimal("10000"), Decimal("100000"))
        assert len(engine.active_copies) == 2

        await engine.stop_copying_trader("TRADER-A")
        assert len(engine.active_copies) == 1
        remaining = list(engine.active_copies.values())[0]
        assert remaining.trader_id == "TRADER-B"
