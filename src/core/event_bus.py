"""Redis Pub/Sub event bus for async event distribution.

Provides an EventBus class that wraps Redis Pub/Sub for publishing and
subscribing to events across system components. Uses orjson for fast
JSON serialization and redis.asyncio for non-blocking operations.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import orjson
import redis.asyncio as aioredis


# ---------------------------------------------------------------------------
# Channel Constants
# ---------------------------------------------------------------------------

# Market data channels
MARKET_TICK = "market.tick.{instrument}"

# Signal channels
SIGNAL_GENERATED = "signal.generated"
SIGNAL_VALIDATED = "signal.validated"

# Order channels
ORDER_SUBMITTED = "order.submitted"
ORDER_FILLED = "order.filled"
ORDER_REJECTED = "order.rejected"

# Risk channels
RISK_ALERT = "risk.alert"
KILL_SWITCH_ACTIVATED = "kill_switch.activated"
KILL_SWITCH_DEACTIVATED = "kill_switch.deactivated"

# Strategy channels
STRATEGY_DISABLED = "strategy.disabled"
STRATEGY_ENABLED = "strategy.enabled"

# News channels
NEWS_ARTICLE_RECEIVED = "news.article_received"
NEWS_HIGH_IMPACT = "news.high_impact"
NEWS_CRISIS_ALERT = "news.crisis_alert"
NEWS_ECONOMIC_EVENT = "news.economic_event_approaching"
NEWS_ALL_SOURCES_DOWN = "news.all_sources_down"
NEWS_SOURCES_RESTORED = "news.sources_restored"

# HFT channels
HFT_CIRCUIT_BREAKER_ACTIVATED = "hft.circuit_breaker.activated"
HFT_MODE_CHANGED = "hft.mode_changed"
HFT_METRICS_UPDATE = "hft.metrics_update"

# Mistake channels
MISTAKE_PATTERN_DETECTED = "mistake.pattern_detected"
MISTAKE_PATTERN_RESOLVED = "mistake.pattern_resolved"


# ---------------------------------------------------------------------------
# Event Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Base event structure for all messages on the event bus.

    Attributes:
        event_type: Identifier for the type of event (e.g. "market.tick").
        timestamp: UTC timestamp when the event was created.
        payload: Arbitrary data associated with the event.
        correlation_id: Optional ID for tracing related events across components.
    """

    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a dictionary suitable for JSON encoding."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize an event from a dictionary."""
        return cls(
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
        )


# Type alias for event handler callbacks
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Async event bus backed by Redis Pub/Sub.

    Usage::

        bus = EventBus(redis_url="redis://localhost:6379")
        await bus.start()

        async def on_tick(event: Event):
            print(event.payload)

        await bus.subscribe("market.tick.EURUSD", on_tick)
        await bus.publish("market.tick.EURUSD", Event(
            event_type="market.tick",
            payload={"bid": 1.1234, "ask": 1.1236},
        ))

        await bus.stop()
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the event bus is currently active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the Redis connection and start the message listener."""
        if self._running:
            return

        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,
        )
        self._pubsub = self._redis.pubsub()
        self._running = True
        self._listener_task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        """Unsubscribe from all channels, cancel the listener, and close Redis."""
        self._running = False

        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None

        if self._redis is not None:
            await self._redis.close()
            self._redis = None

        self._handlers.clear()

    # ------------------------------------------------------------------
    # Publish / Subscribe
    # ------------------------------------------------------------------

    async def publish(self, channel: str, event: Event) -> int:
        """Serialize an event to JSON and publish it to a Redis channel.

        Args:
            channel: The Redis Pub/Sub channel name.
            event: The Event instance to publish.

        Returns:
            The number of subscribers that received the message.

        Raises:
            RuntimeError: If the event bus has not been started.
        """
        if self._redis is None:
            raise RuntimeError("EventBus is not started. Call start() first.")

        data = orjson.dumps(event.to_dict())
        result: int = await self._redis.publish(channel, data)
        return result

    async def subscribe(self, channel: str, handler: EventHandler) -> None:
        """Subscribe to a channel and register a handler for incoming events.

        Multiple handlers can be registered for the same channel.

        Args:
            channel: The Redis Pub/Sub channel to subscribe to.
            handler: An async callable that receives an Event.

        Raises:
            RuntimeError: If the event bus has not been started.
        """
        if self._pubsub is None:
            raise RuntimeError("EventBus is not started. Call start() first.")

        if channel not in self._handlers:
            self._handlers[channel] = []
            await self._pubsub.subscribe(channel)

        self._handlers[channel].append(handler)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel and remove all associated handlers.

        Args:
            channel: The Redis Pub/Sub channel to unsubscribe from.

        Raises:
            RuntimeError: If the event bus has not been started.
        """
        if self._pubsub is None:
            raise RuntimeError("EventBus is not started. Call start() first.")

        if channel in self._handlers:
            del self._handlers[channel]
            await self._pubsub.unsubscribe(channel)

    # ------------------------------------------------------------------
    # Internal Listener
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Background task that reads messages from Redis and dispatches to handlers."""
        if self._pubsub is None:
            return

        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message is not None and message["type"] == "message":
                    channel: str = (
                        message["channel"].decode()
                        if isinstance(message["channel"], bytes)
                        else message["channel"]
                    )
                    data = orjson.loads(message["data"])
                    event = Event.from_dict(data)
                    await self._dispatch(channel, event)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log and continue — don't let a single bad message kill the listener
                await asyncio.sleep(0.1)

    async def _dispatch(self, channel: str, event: Event) -> None:
        """Invoke all registered handlers for a channel."""
        handlers = self._handlers.get(channel, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                # Individual handler failures should not affect other handlers
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_event(
        event_type: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Event:
        """Convenience factory for creating events with auto-generated correlation IDs.

        Args:
            event_type: The type identifier for the event.
            payload: Optional data payload.
            correlation_id: Optional correlation ID. If None, one is generated.

        Returns:
            A new Event instance.
        """
        return Event(
            event_type=event_type,
            payload=payload or {},
            correlation_id=correlation_id or str(uuid.uuid4()),
        )
