"""Unit tests for the HFT Rate Limiter module.

Tests cover rate limiting, order queuing, latency rejection (>500ms),
logging of cancelled orders, and metrics tracking.

Validates: Cross-Cutting Rule 7
"""

import asyncio
import logging
import time
from unittest.mock import patch

import pytest

from src.trading.hft_rate_limiter import (
    HFTRateLimiter,
    LatencyRejection,
    QueuedOrder,
    RateLimiterMetrics,
    TokenBucket,
)


# =============================================================================
# TokenBucket Tests
# =============================================================================


class TestTokenBucket:
    """Tests for the TokenBucket rate limiting algorithm."""

    def test_initial_tokens_at_capacity(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        assert bucket.available_tokens == pytest.approx(10.0, abs=0.1)

    def test_consume_reduces_tokens(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        assert bucket.consume() is True
        assert bucket.available_tokens == pytest.approx(9.0, abs=0.1)

    def test_consume_fails_when_empty(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=2.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_tokens_refill_over_time(self) -> None:
        bucket = TokenBucket(rate=1000.0, capacity=1000.0)
        # Consume all tokens
        for _ in range(1000):
            bucket.consume()
        assert bucket.available_tokens < 1.0
        # Wait for refill
        time.sleep(0.01)  # 10ms at 1000/sec = ~10 tokens
        assert bucket.available_tokens >= 5.0

    def test_tokens_capped_at_capacity(self) -> None:
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        time.sleep(0.1)  # Would add 10 tokens but capped at 5
        assert bucket.available_tokens <= 5.0


# =============================================================================
# Orders Within Rate Limit Are Submitted Immediately
# =============================================================================


class TestImmediateSubmission:
    """Tests that orders within rate limit are submitted immediately."""

    @pytest.mark.asyncio
    async def test_order_submitted_immediately_when_under_limit(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=100)
        order = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        result = await limiter.submit_order(order)
        assert result["status"] == "submitted"
        assert result["order_id"] == "order-001"
        assert result["queue_time_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_orders_submitted_within_limit(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=100)
        results = []
        for i in range(10):
            order = QueuedOrder(
                order_id=f"order-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            result = await limiter.submit_order(order)
            results.append(result)

        assert all(r["status"] == "submitted" for r in results)

    @pytest.mark.asyncio
    async def test_immediate_submission_records_zero_queue_time(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=100)
        order = QueuedOrder(
            order_id="order-001",
            instrument="GBP/USD",
            direction="SELL",
            size=0.5,
        )
        await limiter.submit_order(order)
        metrics = limiter.get_metrics()
        # avg_queue_time should be 0 since order was submitted immediately
        assert metrics.avg_queue_time_ms == 0.0


# =============================================================================
# Orders Exceeding Rate Limit Are Queued
# =============================================================================


class TestOrderQueuing:
    """Tests that orders exceeding rate limit are queued."""

    @pytest.mark.asyncio
    async def test_order_queued_when_rate_exceeded(self) -> None:
        # Very low rate to force queuing
        limiter = HFTRateLimiter(max_orders_per_second=1)
        # First order consumes the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        result1 = await limiter.submit_order(order1)
        assert result1["status"] == "submitted"

        # Second order should be queued (no tokens left)
        order2 = QueuedOrder(
            order_id="order-002",
            instrument="EUR/USD",
            direction="SELL",
            size=1.0,
        )
        result2 = await limiter.submit_order(order2)
        assert result2["status"] == "queued"
        assert result2["order_id"] == "order-002"

    @pytest.mark.asyncio
    async def test_queue_depth_increases_on_queuing(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=1)
        # Consume the single token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue additional orders
        for i in range(3):
            order = QueuedOrder(
                order_id=f"order-{i+2:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        assert limiter.queue_depth == 3

    @pytest.mark.asyncio
    async def test_queued_orders_increment_metric(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=1)
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue one more
        order2 = QueuedOrder(
            order_id="order-002",
            instrument="EUR/USD",
            direction="SELL",
            size=1.0,
        )
        await limiter.submit_order(order2)

        metrics = limiter.get_metrics()
        assert metrics.orders_queued == 1


# =============================================================================
# Queued Orders Cancelled After 500ms
# =============================================================================


class TestLatencyCancellation:
    """Tests that queued orders are cancelled after 500ms."""

    @pytest.mark.asyncio
    async def test_stale_order_cancelled_on_sweep(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=100,  # Use 100ms for faster testing
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue an order with a past enqueue time (simulating 200ms wait)
        stale_order = QueuedOrder(
            order_id="order-stale",
            instrument="EUR/USD",
            direction="SELL",
            size=1.0,
            enqueue_time=time.monotonic() - 0.2,  # 200ms ago
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        # Sweep should cancel the stale order
        results = await limiter.sweep_stale_orders()
        assert len(results) == 1
        assert results[0]["status"] == "cancelled"
        assert results[0]["reason"] == "latency_rejection"
        assert results[0]["order_id"] == "order-stale"

    @pytest.mark.asyncio
    async def test_order_not_cancelled_within_threshold(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=500,
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue a fresh order (just now)
        fresh_order = QueuedOrder(
            order_id="order-fresh",
            instrument="EUR/USD",
            direction="SELL",
            size=1.0,
        )
        async with limiter._queue_lock:
            limiter._queue.append(fresh_order)

        # Sweep should not cancel the fresh order
        results = await limiter.sweep_stale_orders()
        assert len(results) == 0
        assert limiter.queue_depth == 1

    @pytest.mark.asyncio
    async def test_process_queue_cancels_stale_orders(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=50,
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue an order that's already stale
        stale_order = QueuedOrder(
            order_id="order-stale",
            instrument="GBP/USD",
            direction="BUY",
            size=2.0,
            enqueue_time=time.monotonic() - 0.1,  # 100ms ago, threshold is 50ms
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        results = await limiter.process_queue()
        cancelled = [r for r in results if r["status"] == "cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0]["reason"] == "latency_rejection"

    @pytest.mark.asyncio
    async def test_default_threshold_is_500ms(self) -> None:
        limiter = HFTRateLimiter()
        assert limiter.latency_threshold_ms == 500


# =============================================================================
# Latency Rejection Is Logged
# =============================================================================


class TestLatencyRejectionLogging:
    """Tests that latency rejections are properly logged."""

    @pytest.mark.asyncio
    async def test_rejection_logged_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=50,
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Add a stale order
        stale_order = QueuedOrder(
            order_id="order-stale",
            instrument="USD/JPY",
            direction="SELL",
            size=3.0,
            enqueue_time=time.monotonic() - 0.1,
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        with caplog.at_level(logging.WARNING):
            await limiter.sweep_stale_orders()

        assert "latency_rejection" in caplog.text
        assert "order-stale" in caplog.text
        assert "USD/JPY" in caplog.text

    @pytest.mark.asyncio
    async def test_rejection_stored_in_rejections_list(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=50,
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Add a stale order
        stale_order = QueuedOrder(
            order_id="order-rejected",
            instrument="EUR/GBP",
            direction="BUY",
            size=2.5,
            enqueue_time=time.monotonic() - 0.1,
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        await limiter.sweep_stale_orders()

        rejections = limiter.rejections
        assert len(rejections) == 1
        assert rejections[0].order_id == "order-rejected"
        assert rejections[0].instrument == "EUR/GBP"
        assert rejections[0].direction == "BUY"
        assert rejections[0].size == 2.5
        assert rejections[0].reason == "latency_rejection"
        assert rejections[0].queue_time_ms > 50.0
        assert rejections[0].rejection_timestamp is not None

    @pytest.mark.asyncio
    async def test_rejection_includes_timestamp_and_order_details(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=50,
        )
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        stale_order = QueuedOrder(
            order_id="order-detail-test",
            instrument="AUD/USD",
            direction="SELL",
            size=5.0,
            enqueue_time=time.monotonic() - 0.2,
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        await limiter.sweep_stale_orders()

        rejection = limiter.rejections[0]
        assert rejection.order_id == "order-detail-test"
        assert rejection.instrument == "AUD/USD"
        assert rejection.direction == "SELL"
        assert rejection.size == 5.0
        assert rejection.queue_time_ms >= 200.0


# =============================================================================
# Metrics Are Tracked Correctly
# =============================================================================


class TestMetricsTracking:
    """Tests that metrics are tracked correctly."""

    @pytest.mark.asyncio
    async def test_initial_metrics_are_zero(self) -> None:
        limiter = HFTRateLimiter()
        metrics = limiter.get_metrics()
        assert metrics.orders_queued == 0
        assert metrics.orders_cancelled_latency == 0
        assert metrics.avg_queue_time_ms == 0.0
        assert metrics.orders_submitted == 0
        assert metrics.current_queue_depth == 0

    @pytest.mark.asyncio
    async def test_submitted_orders_counted(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=100)
        for i in range(5):
            order = QueuedOrder(
                order_id=f"order-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        metrics = limiter.get_metrics()
        assert metrics.orders_submitted == 5

    @pytest.mark.asyncio
    async def test_queued_orders_counted(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=2)
        # Submit 2 to consume tokens
        for i in range(2):
            order = QueuedOrder(
                order_id=f"order-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        # Next orders should be queued
        for i in range(3):
            order = QueuedOrder(
                order_id=f"order-queued-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        metrics = limiter.get_metrics()
        assert metrics.orders_queued == 3
        assert metrics.orders_submitted == 2

    @pytest.mark.asyncio
    async def test_cancelled_orders_counted(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=50,
        )
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Add stale orders
        for i in range(3):
            stale_order = QueuedOrder(
                order_id=f"order-stale-{i}",
                instrument="EUR/USD",
                direction="SELL",
                size=1.0,
                enqueue_time=time.monotonic() - 0.1,
            )
            async with limiter._queue_lock:
                limiter._queue.append(stale_order)
                limiter._orders_queued += 1

        await limiter.sweep_stale_orders()

        metrics = limiter.get_metrics()
        assert metrics.orders_cancelled_latency == 3

    @pytest.mark.asyncio
    async def test_avg_queue_time_calculated(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=100)
        # Submit orders immediately (queue_time = 0)
        for i in range(5):
            order = QueuedOrder(
                order_id=f"order-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        metrics = limiter.get_metrics()
        assert metrics.avg_queue_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_current_queue_depth_reflects_queue_size(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=1)
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Queue 2 orders
        for i in range(2):
            order = QueuedOrder(
                order_id=f"order-q-{i}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            await limiter.submit_order(order)

        metrics = limiter.get_metrics()
        assert metrics.current_queue_depth == 2


# =============================================================================
# Sweep Loop Tests
# =============================================================================


class TestSweepLoop:
    """Tests for the periodic sweep loop."""

    @pytest.mark.asyncio
    async def test_start_and_stop_sweep_loop(self) -> None:
        limiter = HFTRateLimiter(sweep_interval_ms=10)
        await limiter.start_sweep_loop()
        assert limiter._running is True
        assert limiter._sweep_task is not None

        await limiter.stop_sweep_loop()
        assert limiter._running is False

    @pytest.mark.asyncio
    async def test_sweep_loop_cancels_stale_orders(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=30,
            sweep_interval_ms=10,
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Add a stale order
        stale_order = QueuedOrder(
            order_id="order-sweep-test",
            instrument="EUR/USD",
            direction="SELL",
            size=1.0,
            enqueue_time=time.monotonic() - 0.05,  # 50ms ago, threshold is 30ms
        )
        async with limiter._queue_lock:
            limiter._queue.append(stale_order)
            limiter._orders_queued += 1

        await limiter.start_sweep_loop()
        # Wait for at least one sweep cycle
        await asyncio.sleep(0.05)
        await limiter.stop_sweep_loop()

        metrics = limiter.get_metrics()
        assert metrics.orders_cancelled_latency >= 1

    @pytest.mark.asyncio
    async def test_start_sweep_loop_idempotent(self) -> None:
        limiter = HFTRateLimiter(sweep_interval_ms=10)
        await limiter.start_sweep_loop()
        task1 = limiter._sweep_task
        await limiter.start_sweep_loop()
        task2 = limiter._sweep_task
        assert task1 is task2
        await limiter.stop_sweep_loop()


# =============================================================================
# Process Queue Tests
# =============================================================================


class TestProcessQueue:
    """Tests for processing the order queue."""

    @pytest.mark.asyncio
    async def test_process_queue_submits_when_tokens_available(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=100,
            latency_threshold_ms=500,
        )
        # Manually add an order to the queue
        order = QueuedOrder(
            order_id="order-queued",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        async with limiter._queue_lock:
            limiter._queue.append(order)

        results = await limiter.process_queue()
        assert len(results) == 1
        assert results[0]["status"] == "submitted"
        assert results[0]["order_id"] == "order-queued"

    @pytest.mark.asyncio
    async def test_process_queue_keeps_orders_when_still_rate_limited(self) -> None:
        limiter = HFTRateLimiter(
            max_orders_per_second=1,
            latency_threshold_ms=5000,  # High threshold so orders don't get cancelled
        )
        # Consume the token
        order1 = QueuedOrder(
            order_id="order-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.0,
        )
        await limiter.submit_order(order1)

        # Add fresh orders to queue
        for i in range(2):
            order = QueuedOrder(
                order_id=f"order-q-{i}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            async with limiter._queue_lock:
                limiter._queue.append(order)

        results = await limiter.process_queue()
        # Orders should remain in queue since no tokens available
        # and they haven't exceeded latency threshold
        submitted = [r for r in results if r["status"] == "submitted"]
        assert len(submitted) == 0
        assert limiter.queue_depth == 2


# =============================================================================
# Integration with HFT Pipeline
# =============================================================================


class TestHFTIntegration:
    """Tests for integration with the HFT pipeline order submission path."""

    @pytest.mark.asyncio
    async def test_rate_limiter_properties(self) -> None:
        limiter = HFTRateLimiter(max_orders_per_second=50)
        assert limiter.max_orders_per_second == 50
        assert limiter.latency_threshold_ms == 500

    @pytest.mark.asyncio
    async def test_queued_order_dataclass(self) -> None:
        order = QueuedOrder(
            order_id="test-001",
            instrument="EUR/USD",
            direction="BUY",
            size=1.5,
            order_data={"strategy": "momentum"},
        )
        assert order.order_id == "test-001"
        assert order.instrument == "EUR/USD"
        assert order.direction == "BUY"
        assert order.size == 1.5
        assert order.order_data == {"strategy": "momentum"}
        assert order.enqueue_time > 0
        assert order.enqueue_timestamp is not None

    @pytest.mark.asyncio
    async def test_full_lifecycle_submit_queue_cancel(self) -> None:
        """Test the full lifecycle: submit → queue → cancel due to latency."""
        limiter = HFTRateLimiter(
            max_orders_per_second=2,
            latency_threshold_ms=50,
        )

        # Submit orders within limit
        results_submitted = []
        for i in range(2):
            order = QueuedOrder(
                order_id=f"order-{i:03d}",
                instrument="EUR/USD",
                direction="BUY",
                size=1.0,
            )
            result = await limiter.submit_order(order)
            results_submitted.append(result)

        assert all(r["status"] == "submitted" for r in results_submitted)

        # Submit orders that exceed limit (will be queued)
        for i in range(3):
            order = QueuedOrder(
                order_id=f"order-excess-{i}",
                instrument="EUR/USD",
                direction="SELL",
                size=1.0,
            )
            await limiter.submit_order(order)

        assert limiter.queue_depth == 3

        # Simulate time passing beyond threshold
        async with limiter._queue_lock:
            now = time.monotonic()
            for order in limiter._queue:
                order.enqueue_time = now - 0.1  # 100ms ago, threshold is 50ms

        # Sweep should cancel all stale orders
        cancelled = await limiter.sweep_stale_orders()
        assert len(cancelled) == 3
        assert all(r["reason"] == "latency_rejection" for r in cancelled)

        # Verify metrics
        metrics = limiter.get_metrics()
        assert metrics.orders_submitted == 2
        assert metrics.orders_queued == 3
        assert metrics.orders_cancelled_latency == 3
        assert metrics.current_queue_depth == 0
