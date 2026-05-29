"""HFT Rate Limiter implementing IG API rate limit constraint.

Queues orders that exceed the IG API rate limit, cancels any order queued
for more than 500ms (latency rejection), and provides metrics on queue
performance.

Validates: Cross-Cutting Rule 7
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.config.constants import (
    HFT_LATENCY_REJECTION_MS,
    HFT_MAX_ORDER_RATE_DEFAULT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class QueuedOrder:
    """An order waiting in the rate limit queue.

    Attributes:
        order_id: Unique identifier for the order.
        instrument: The instrument identifier.
        direction: Trade direction ("BUY" or "SELL").
        size: Order size.
        order_data: Full order details.
        enqueue_time: Monotonic time when the order was queued.
        enqueue_timestamp: UTC datetime when the order was queued.
    """

    order_id: str
    instrument: str
    direction: str
    size: float
    order_data: dict[str, Any] = field(default_factory=dict)
    enqueue_time: float = field(default_factory=time.monotonic)
    enqueue_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class LatencyRejection:
    """Record of an order cancelled due to latency rejection.

    Attributes:
        order_id: The rejected order's identifier.
        instrument: The instrument identifier.
        direction: Trade direction.
        size: Order size.
        queue_time_ms: How long the order was queued before cancellation.
        rejection_timestamp: UTC datetime of the rejection.
        reason: Always "latency_rejection".
    """

    order_id: str
    instrument: str
    direction: str
    size: float
    queue_time_ms: float
    rejection_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reason: str = "latency_rejection"


@dataclass
class RateLimiterMetrics:
    """Metrics for the HFT rate limiter.

    Attributes:
        orders_queued: Total number of orders that have been queued.
        orders_cancelled_latency: Total orders cancelled due to latency rejection.
        avg_queue_time_ms: Average time orders spend in the queue.
        orders_submitted: Total orders successfully submitted.
        current_queue_depth: Current number of orders in the queue.
    """

    orders_queued: int = 0
    orders_cancelled_latency: int = 0
    avg_queue_time_ms: float = 0.0
    orders_submitted: int = 0
    current_queue_depth: int = 0


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------


class TokenBucket:
    """Token bucket algorithm for rate limiting.

    Tokens are added at a fixed rate up to a maximum capacity.
    Each order consumes one token. If no tokens are available,
    the order must be queued.

    Args:
        rate: Tokens added per second (max orders per second).
        capacity: Maximum token capacity (burst allowance).
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        self._tokens = self._capacity
        self._last_refill: float = time.monotonic()

    def consume(self) -> bool:
        """Try to consume a token.

        Returns:
            True if a token was available and consumed, False otherwise.
        """
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# HFT Rate Limiter
# ---------------------------------------------------------------------------


class HFTRateLimiter:
    """Rate limiter for HFT orders constrained by IG API rate limits.

    Queues orders that exceed the IG API rate limit and cancels any order
    that has been queued for more than 500ms (latency rejection). Provides
    metrics on queue performance and integrates with the HFT pipeline's
    order submission path.

    Args:
        max_orders_per_second: Maximum orders per second allowed by IG API.
            Defaults to HFT_MAX_ORDER_RATE_DEFAULT from constants.
        latency_threshold_ms: Maximum acceptable queue time in milliseconds
            before an order is cancelled. Defaults to HFT_LATENCY_REJECTION_MS.
        sweep_interval_ms: Interval in milliseconds between queue sweeps
            for stale orders. Defaults to 50ms.
    """

    def __init__(
        self,
        max_orders_per_second: int = HFT_MAX_ORDER_RATE_DEFAULT,
        latency_threshold_ms: int = HFT_LATENCY_REJECTION_MS,
        sweep_interval_ms: int = 50,
    ) -> None:
        self._max_orders_per_second = max_orders_per_second
        self._latency_threshold_ms = latency_threshold_ms
        self._sweep_interval_ms = sweep_interval_ms

        # Token bucket for rate limiting
        self._token_bucket = TokenBucket(
            rate=float(max_orders_per_second),
            capacity=float(max_orders_per_second),
        )

        # Order queue
        self._queue: deque[QueuedOrder] = deque()
        self._queue_lock = asyncio.Lock()

        # Metrics tracking
        self._orders_queued: int = 0
        self._orders_cancelled_latency: int = 0
        self._orders_submitted: int = 0
        self._queue_times: list[float] = []
        self._max_queue_time_samples: int = 1000

        # Rejection log
        self._rejections: list[LatencyRejection] = []
        self._max_rejection_log: int = 10000

        # Sweep task handle
        self._sweep_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def submit_order(self, order: QueuedOrder) -> dict[str, Any]:
        """Submit an order through the rate limiter.

        If the rate limit allows, the order is submitted immediately.
        Otherwise, it is queued for later submission. If the order
        remains queued for more than 500ms, it is cancelled as a
        latency rejection.

        Args:
            order: The order to submit.

        Returns:
            Dict with submission result:
            - status: "submitted", "queued", or "cancelled"
            - order_id: The order identifier
            - queue_time_ms: Time spent in queue (0 if immediate)
            - reason: Reason for cancellation if applicable
        """
        if self._token_bucket.consume():
            # Token available - submit immediately
            self._orders_submitted += 1
            self._record_queue_time(0.0)
            return {
                "status": "submitted",
                "order_id": order.order_id,
                "instrument": order.instrument,
                "direction": order.direction,
                "queue_time_ms": 0.0,
            }

        # Rate limit exceeded - queue the order
        async with self._queue_lock:
            self._queue.append(order)
            self._orders_queued += 1

        logger.info(
            "Order queued due to rate limit: order_id=%s instrument=%s direction=%s",
            order.order_id,
            order.instrument,
            order.direction,
        )

        return {
            "status": "queued",
            "order_id": order.order_id,
            "instrument": order.instrument,
            "direction": order.direction,
            "queue_time_ms": 0.0,
        }

    async def process_queue(self) -> list[dict[str, Any]]:
        """Process queued orders: submit those within latency threshold, cancel stale ones.

        Returns:
            List of results for each processed order.
        """
        results: list[dict[str, Any]] = []

        async with self._queue_lock:
            remaining: deque[QueuedOrder] = deque()
            now = time.monotonic()

            while self._queue:
                order = self._queue.popleft()
                queue_time_ms = (now - order.enqueue_time) * 1000

                if queue_time_ms > self._latency_threshold_ms:
                    # Cancel - latency rejection
                    result = self._cancel_order_latency(order, queue_time_ms)
                    results.append(result)
                elif self._token_bucket.consume():
                    # Token available - submit
                    self._orders_submitted += 1
                    self._record_queue_time(queue_time_ms)
                    results.append({
                        "status": "submitted",
                        "order_id": order.order_id,
                        "instrument": order.instrument,
                        "direction": order.direction,
                        "queue_time_ms": queue_time_ms,
                    })
                else:
                    # Still rate limited - keep in queue
                    remaining.append(order)

            self._queue = remaining

        return results

    async def sweep_stale_orders(self) -> list[dict[str, Any]]:
        """Sweep the queue and cancel any orders that have exceeded the latency threshold.

        This is called periodically to ensure stale orders are cancelled promptly.

        Returns:
            List of cancellation results for stale orders.
        """
        results: list[dict[str, Any]] = []

        async with self._queue_lock:
            remaining: deque[QueuedOrder] = deque()
            now = time.monotonic()

            while self._queue:
                order = self._queue.popleft()
                queue_time_ms = (now - order.enqueue_time) * 1000

                if queue_time_ms > self._latency_threshold_ms:
                    result = self._cancel_order_latency(order, queue_time_ms)
                    results.append(result)
                else:
                    remaining.append(order)

            self._queue = remaining

        return results

    async def start_sweep_loop(self) -> None:
        """Start the periodic sweep loop for cancelling stale orders."""
        if self._running:
            return
        self._running = True
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def stop_sweep_loop(self) -> None:
        """Stop the periodic sweep loop."""
        self._running = False
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

    def get_metrics(self) -> RateLimiterMetrics:
        """Return current rate limiter metrics.

        Returns:
            RateLimiterMetrics with current performance data.
        """
        return RateLimiterMetrics(
            orders_queued=self._orders_queued,
            orders_cancelled_latency=self._orders_cancelled_latency,
            avg_queue_time_ms=self._calculate_avg_queue_time(),
            orders_submitted=self._orders_submitted,
            current_queue_depth=len(self._queue),
        )

    @property
    def rejections(self) -> list[LatencyRejection]:
        """Return the log of latency rejections."""
        return self._rejections.copy()

    @property
    def queue_depth(self) -> int:
        """Return the current number of orders in the queue."""
        return len(self._queue)

    @property
    def max_orders_per_second(self) -> int:
        """Return the configured max orders per second."""
        return self._max_orders_per_second

    @property
    def latency_threshold_ms(self) -> int:
        """Return the configured latency threshold in milliseconds."""
        return self._latency_threshold_ms

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _cancel_order_latency(
        self, order: QueuedOrder, queue_time_ms: float
    ) -> dict[str, Any]:
        """Cancel an order due to latency rejection and log it.

        Args:
            order: The order to cancel.
            queue_time_ms: How long the order was queued.

        Returns:
            Dict with cancellation details.
        """
        self._orders_cancelled_latency += 1
        self._record_queue_time(queue_time_ms)

        rejection = LatencyRejection(
            order_id=order.order_id,
            instrument=order.instrument,
            direction=order.direction,
            size=order.size,
            queue_time_ms=queue_time_ms,
        )
        self._rejections.append(rejection)

        # Trim rejection log if needed
        if len(self._rejections) > self._max_rejection_log:
            self._rejections = self._rejections[-self._max_rejection_log:]

        logger.warning(
            "latency_rejection: order_id=%s instrument=%s direction=%s "
            "size=%s queue_time_ms=%.2f timestamp=%s",
            order.order_id,
            order.instrument,
            order.direction,
            order.size,
            queue_time_ms,
            rejection.rejection_timestamp.isoformat(),
        )

        return {
            "status": "cancelled",
            "order_id": order.order_id,
            "instrument": order.instrument,
            "direction": order.direction,
            "queue_time_ms": queue_time_ms,
            "reason": "latency_rejection",
        }

    def _record_queue_time(self, queue_time_ms: float) -> None:
        """Record a queue time sample for metrics."""
        self._queue_times.append(queue_time_ms)
        if len(self._queue_times) > self._max_queue_time_samples:
            self._queue_times = self._queue_times[-self._max_queue_time_samples:]

    def _calculate_avg_queue_time(self) -> float:
        """Calculate average queue time from recent samples."""
        if not self._queue_times:
            return 0.0
        return sum(self._queue_times) / len(self._queue_times)

    async def _sweep_loop(self) -> None:
        """Periodically sweep the queue to cancel stale orders."""
        while self._running:
            try:
                await asyncio.sleep(self._sweep_interval_ms / 1000.0)
                await self.sweep_stale_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in sweep loop: %s", str(e))
