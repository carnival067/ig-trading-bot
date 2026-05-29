"""Unit tests for the HFT Pipeline module.

Tests cover tick-by-tick microstructure analysis, order batching, connection pool
warming, HFT mode enable/disable with logging, rate limiting, and metrics.

Validates: Requirements 22.1, 22.2, 22.4, 22.5, 22.11, 22.12, 22.13
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.trading.hft_pipeline import (
    HFTMetrics,
    HFTPipeline,
    HFTSignal,
    SlidingWindowCounter,
)


# =============================================================================
# Task 31.1: HFTPipeline class structure
# =============================================================================


class TestHFTPipelineInit:
    """Tests for HFTPipeline initialization."""

    def test_default_initialization(self) -> None:
        pipeline = HFTPipeline()
        assert pipeline.active is False
        assert pipeline._max_order_rate == 100
        assert pipeline._max_per_instrument_rate == 50
        assert pipeline._batch_window_ms == 100

    def test_custom_initialization(self) -> None:
        pipeline = HFTPipeline(
            max_order_rate=200,
            max_per_instrument_rate=75,
            batch_window_ms=50,
        )
        assert pipeline._max_order_rate == 200
        assert pipeline._max_per_instrument_rate == 75
        assert pipeline._batch_window_ms == 50

    def test_connection_pool_initially_empty(self) -> None:
        pipeline = HFTPipeline()
        assert pipeline._connection_pool == []

    def test_batch_buffer_initially_empty(self) -> None:
        pipeline = HFTPipeline()
        assert pipeline._batch_buffer == []


# =============================================================================
# Task 31.2: Tick-by-tick microstructure analysis
# =============================================================================


class TestProcessTick:
    """Tests for tick processing and microstructure analysis."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_inactive(self) -> None:
        pipeline = HFTPipeline()
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 100,
            "ask_volume": 10,
        }
        signals = await pipeline.process_tick(tick)
        assert signals == []

    @pytest.mark.asyncio
    async def test_detects_order_flow_imbalance_buy(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 90,
            "ask_volume": 10,
            "avg_spread": 0.0010,
            "price_change_pct": 0.0,
        }
        signals = await pipeline.process_tick(tick)
        imbalance_signals = [s for s in signals if s.signal_type == "order_flow_imbalance"]
        assert len(imbalance_signals) == 1
        assert imbalance_signals[0].direction == "BUY"
        assert imbalance_signals[0].instrument == "EUR/USD"

    @pytest.mark.asyncio
    async def test_detects_order_flow_imbalance_sell(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "GBP/USD",
            "bid": 1.2500,
            "ask": 1.2502,
            "bid_volume": 10,
            "ask_volume": 90,
            "avg_spread": 0.0010,
            "price_change_pct": 0.0,
        }
        signals = await pipeline.process_tick(tick)
        imbalance_signals = [s for s in signals if s.signal_type == "order_flow_imbalance"]
        assert len(imbalance_signals) == 1
        assert imbalance_signals[0].direction == "SELL"

    @pytest.mark.asyncio
    async def test_no_imbalance_when_balanced(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 50,
            "ask_volume": 50,
            "avg_spread": 0.0010,
            "price_change_pct": 0.0,
        }
        signals = await pipeline.process_tick(tick)
        imbalance_signals = [s for s in signals if s.signal_type == "order_flow_imbalance"]
        assert len(imbalance_signals) == 0

    @pytest.mark.asyncio
    async def test_detects_spread_compression(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.12340,
            "ask": 1.12343,  # spread = 0.00003
            "avg_spread": 0.0002,  # avg spread much larger
            "bid_volume": 50,
            "ask_volume": 50,
            "price_change_pct": 0.0,
            "last_mid": 1.12340,
        }
        signals = await pipeline.process_tick(tick)
        spread_signals = [s for s in signals if s.signal_type == "spread_compression"]
        assert len(spread_signals) == 1

    @pytest.mark.asyncio
    async def test_detects_momentum_burst_buy(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 50,
            "ask_volume": 50,
            "avg_spread": 0.0010,
            "price_change_pct": 0.005,  # 0.5% move
        }
        signals = await pipeline.process_tick(tick)
        momentum_signals = [s for s in signals if s.signal_type == "momentum_burst"]
        assert len(momentum_signals) == 1
        assert momentum_signals[0].direction == "BUY"

    @pytest.mark.asyncio
    async def test_detects_momentum_burst_sell(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 50,
            "ask_volume": 50,
            "avg_spread": 0.0010,
            "price_change_pct": -0.003,  # -0.3% move
        }
        signals = await pipeline.process_tick(tick)
        momentum_signals = [s for s in signals if s.signal_type == "momentum_burst"]
        assert len(momentum_signals) == 1
        assert momentum_signals[0].direction == "SELL"

    @pytest.mark.asyncio
    async def test_records_latency(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 50,
            "ask_volume": 50,
            "avg_spread": 0.0010,
            "price_change_pct": 0.005,
        }
        signals = await pipeline.process_tick(tick)
        # Signals should have latency recorded
        for signal in signals:
            assert signal.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_no_signal_on_zero_volumes(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 0,
            "ask_volume": 0,
            "avg_spread": 0.0010,
            "price_change_pct": 0.0,
        }
        signals = await pipeline.process_tick(tick)
        imbalance_signals = [s for s in signals if s.signal_type == "order_flow_imbalance"]
        assert len(imbalance_signals) == 0


# =============================================================================
# Task 31.3: Order batching and parallel submission
# =============================================================================


class TestBatchAndSubmit:
    """Tests for order batching within 100ms window and parallel submission."""

    @pytest.mark.asyncio
    async def test_empty_signals_returns_empty(self) -> None:
        pipeline = HFTPipeline(batch_window_ms=10)
        pipeline.active = True
        results = await pipeline.batch_and_submit([])
        assert results == []

    @pytest.mark.asyncio
    async def test_submits_signals_in_batch(self) -> None:
        pipeline = HFTPipeline(batch_window_ms=10)
        pipeline.active = True
        signals = [
            HFTSignal(
                instrument="EUR/USD",
                signal_type="momentum_burst",
                direction="BUY",
                strength=0.8,
            ),
            HFTSignal(
                instrument="GBP/USD",
                signal_type="order_flow_imbalance",
                direction="SELL",
                strength=0.7,
            ),
        ]
        results = await pipeline.batch_and_submit(signals)
        assert len(results) == 2
        assert all(r["status"] == "submitted" for r in results)

    @pytest.mark.asyncio
    async def test_records_order_rate_on_submission(self) -> None:
        pipeline = HFTPipeline(batch_window_ms=10)
        pipeline.active = True
        signals = [
            HFTSignal(
                instrument="EUR/USD",
                signal_type="momentum_burst",
                direction="BUY",
                strength=0.8,
            ),
        ]
        await pipeline.batch_and_submit(signals)
        assert pipeline._order_rate_tracker.count() == 1


# =============================================================================
# Task 31.4: Pre-warmed connection pool
# =============================================================================


class TestConnectionPool:
    """Tests for pre-warmed connection pool."""

    @pytest.mark.asyncio
    async def test_warm_connection_pool_creates_min_5(self) -> None:
        pipeline = HFTPipeline()
        await pipeline._warm_connection_pool()
        assert len(pipeline._connection_pool) >= 5

    @pytest.mark.asyncio
    async def test_connection_pool_entries_have_ready_status(self) -> None:
        pipeline = HFTPipeline()
        await pipeline._warm_connection_pool()
        for conn in pipeline._connection_pool:
            assert conn["status"] == "ready"
            assert "id" in conn
            assert "created_at" in conn

    @pytest.mark.asyncio
    async def test_enable_warms_pool(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="test_user", equity=Decimal("100000"))
        assert len(pipeline._connection_pool) >= 5


# =============================================================================
# Task 31.5: HFT mode enable/disable with logging
# =============================================================================


class TestHFTModeToggle:
    """Tests for HFT mode enable/disable with logging."""

    @pytest.mark.asyncio
    async def test_enable_sets_active(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        assert pipeline.active is True

    @pytest.mark.asyncio
    async def test_enable_logs_mode_change(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        assert len(pipeline.mode_changes) == 1
        change = pipeline.mode_changes[0]
        assert change["action"] == "enable"
        assert change["user"] == "trader1"
        assert change["equity"] == "50000"
        assert "timestamp" in change

    @pytest.mark.asyncio
    async def test_enable_when_already_active_is_noop(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        await pipeline.enable(user="trader2", equity=Decimal("60000"))
        assert len(pipeline.mode_changes) == 1  # Only first enable logged

    @pytest.mark.asyncio
    async def test_disable_sets_inactive(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        await pipeline.disable(
            reason="manual stop", user="trader1", equity=Decimal("49000")
        )
        assert pipeline.active is False

    @pytest.mark.asyncio
    async def test_disable_logs_mode_change(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        await pipeline.disable(
            reason="circuit breaker", user="system", equity=Decimal("48000")
        )
        assert len(pipeline.mode_changes) == 2
        change = pipeline.mode_changes[1]
        assert change["action"] == "disable"
        assert change["user"] == "system"
        assert change["reason"] == "circuit breaker"
        assert change["equity"] == "48000"

    @pytest.mark.asyncio
    async def test_disable_drains_batch_buffer(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.enable(user="trader1", equity=Decimal("50000"))
        # Add signals to buffer
        pipeline._batch_buffer.append(
            HFTSignal(
                instrument="EUR/USD",
                signal_type="test",
                direction="BUY",
                strength=0.5,
            )
        )
        await pipeline.disable(
            reason="shutdown", user="system", equity=Decimal("50000")
        )
        assert len(pipeline._batch_buffer) == 0
        assert pipeline.mode_changes[1]["pending_orders_drained"] == 1

    @pytest.mark.asyncio
    async def test_disable_when_already_inactive_is_noop(self) -> None:
        pipeline = HFTPipeline()
        await pipeline.disable(
            reason="test", user="system", equity=Decimal("50000")
        )
        assert len(pipeline.mode_changes) == 0


# =============================================================================
# Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limit checking."""

    def test_rate_limit_allows_when_under_limit(self) -> None:
        pipeline = HFTPipeline(max_order_rate=100, max_per_instrument_rate=50)
        assert pipeline.check_rate_limit("EUR/USD") is True

    def test_rate_limit_blocks_when_global_exceeded(self) -> None:
        pipeline = HFTPipeline(max_order_rate=5, max_per_instrument_rate=50)
        # Simulate 5 orders
        for _ in range(5):
            pipeline._order_rate_tracker.record()
        assert pipeline.check_rate_limit("EUR/USD") is False

    def test_rate_limit_blocks_when_instrument_exceeded(self) -> None:
        pipeline = HFTPipeline(max_order_rate=100, max_per_instrument_rate=3)
        # Simulate 3 orders for EUR/USD
        pipeline._instrument_rate_trackers["EUR/USD"] = SlidingWindowCounter()
        for _ in range(3):
            pipeline._instrument_rate_trackers["EUR/USD"].record()
        assert pipeline.check_rate_limit("EUR/USD") is False

    def test_rate_limit_allows_different_instrument(self) -> None:
        pipeline = HFTPipeline(max_order_rate=100, max_per_instrument_rate=3)
        pipeline._instrument_rate_trackers["EUR/USD"] = SlidingWindowCounter()
        for _ in range(3):
            pipeline._instrument_rate_trackers["EUR/USD"].record()
        # GBP/USD should still be allowed
        assert pipeline.check_rate_limit("GBP/USD") is True


# =============================================================================
# Metrics
# =============================================================================


class TestHFTMetrics:
    """Tests for HFT metrics collection."""

    def test_initial_metrics(self) -> None:
        pipeline = HFTPipeline()
        metrics = pipeline.get_metrics()
        assert metrics.orders_per_second == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.net_pnl_1min == Decimal("0")
        assert metrics.circuit_breaker_active is False

    def test_update_metrics(self) -> None:
        pipeline = HFTPipeline()
        pipeline.update_metrics(
            net_pnl_1min=Decimal("100"),
            net_pnl_5min=Decimal("500"),
            net_pnl_daily=Decimal("2000"),
            circuit_breaker_active=True,
            total_exposure_pct=0.08,
        )
        metrics = pipeline.get_metrics()
        assert metrics.net_pnl_1min == Decimal("100")
        assert metrics.net_pnl_5min == Decimal("500")
        assert metrics.net_pnl_daily == Decimal("2000")
        assert metrics.circuit_breaker_active is True
        assert metrics.total_exposure_pct == 0.08

    @pytest.mark.asyncio
    async def test_latency_tracking(self) -> None:
        pipeline = HFTPipeline()
        pipeline.active = True
        tick = {
            "instrument": "EUR/USD",
            "bid": 1.1234,
            "ask": 1.1236,
            "bid_volume": 90,
            "ask_volume": 10,
            "avg_spread": 0.0010,
            "price_change_pct": 0.0,
        }
        await pipeline.process_tick(tick)
        metrics = pipeline.get_metrics()
        assert metrics.avg_latency_ms > 0


# =============================================================================
# SlidingWindowCounter
# =============================================================================


class TestSlidingWindowCounter:
    """Tests for the SlidingWindowCounter utility."""

    def test_initial_count_is_zero(self) -> None:
        counter = SlidingWindowCounter()
        assert counter.count() == 0

    def test_records_increment_count(self) -> None:
        counter = SlidingWindowCounter()
        counter.record()
        counter.record()
        counter.record()
        assert counter.count() == 3

    def test_prunes_old_entries(self) -> None:
        counter = SlidingWindowCounter(window_seconds=0.01)
        counter.record()
        import time

        time.sleep(0.02)
        assert counter.count() == 0


# =============================================================================
# HFTSignal dataclass
# =============================================================================


class TestHFTSignal:
    """Tests for the HFTSignal dataclass."""

    def test_creation(self) -> None:
        signal = HFTSignal(
            instrument="EUR/USD",
            signal_type="order_flow_imbalance",
            direction="BUY",
            strength=0.85,
        )
        assert signal.instrument == "EUR/USD"
        assert signal.signal_type == "order_flow_imbalance"
        assert signal.direction == "BUY"
        assert signal.strength == 0.85
        assert signal.latency_ms == 0.0

    def test_timestamp_auto_generated(self) -> None:
        signal = HFTSignal(
            instrument="EUR/USD",
            signal_type="test",
            direction="BUY",
            strength=0.5,
        )
        assert signal.timestamp is not None
        assert signal.timestamp.tzinfo == timezone.utc


# =============================================================================
# HFTMetrics dataclass
# =============================================================================


class TestHFTMetricsDataclass:
    """Tests for the HFTMetrics dataclass."""

    def test_default_values(self) -> None:
        metrics = HFTMetrics()
        assert metrics.orders_per_second == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.net_pnl_1min == Decimal("0")
        assert metrics.net_pnl_5min == Decimal("0")
        assert metrics.net_pnl_daily == Decimal("0")
        assert metrics.circuit_breaker_active is False
        assert metrics.total_exposure_pct == 0.0

    def test_custom_values(self) -> None:
        metrics = HFTMetrics(
            orders_per_second=45.0,
            avg_latency_ms=3.2,
            net_pnl_1min=Decimal("-50"),
            net_pnl_5min=Decimal("200"),
            net_pnl_daily=Decimal("1500"),
            circuit_breaker_active=True,
            total_exposure_pct=0.12,
        )
        assert metrics.orders_per_second == 45.0
        assert metrics.avg_latency_ms == 3.2
        assert metrics.net_pnl_1min == Decimal("-50")
        assert metrics.circuit_breaker_active is True
