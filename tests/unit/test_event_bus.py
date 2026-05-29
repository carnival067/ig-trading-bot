"""Unit tests for src/core/event_bus.py."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from src.core.event_bus import (
    Event,
    EventBus,
    HFT_CIRCUIT_BREAKER_ACTIVATED,
    HFT_MODE_CHANGED,
    KILL_SWITCH_ACTIVATED,
    KILL_SWITCH_DEACTIVATED,
    MARKET_TICK,
    MISTAKE_PATTERN_DETECTED,
    MISTAKE_PATTERN_RESOLVED,
    NEWS_ARTICLE_RECEIVED,
    NEWS_CRISIS_ALERT,
    NEWS_ECONOMIC_EVENT,
    ORDER_FILLED,
    ORDER_REJECTED,
    ORDER_SUBMITTED,
    RISK_ALERT,
    SIGNAL_GENERATED,
    SIGNAL_VALIDATED,
    STRATEGY_DISABLED,
    STRATEGY_ENABLED,
)


# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


class TestEvent:
    """Tests for the Event dataclass."""

    def test_event_creation_defaults(self) -> None:
        event = Event(event_type="test.event")
        assert event.event_type == "test.event"
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == timezone.utc
        assert event.payload == {}
        assert event.correlation_id is None

    def test_event_creation_with_all_fields(self) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = Event(
            event_type="order.filled",
            timestamp=ts,
            payload={"order_id": "123", "price": 1.1234},
            correlation_id="corr-abc",
        )
        assert event.event_type == "order.filled"
        assert event.timestamp == ts
        assert event.payload == {"order_id": "123", "price": 1.1234}
        assert event.correlation_id == "corr-abc"

    def test_to_dict(self) -> None:
        ts = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        event = Event(
            event_type="signal.generated",
            timestamp=ts,
            payload={"direction": "long"},
            correlation_id="id-1",
        )
        d = event.to_dict()
        assert d["event_type"] == "signal.generated"
        assert d["timestamp"] == "2024-06-15T10:30:00+00:00"
        assert d["payload"] == {"direction": "long"}
        assert d["correlation_id"] == "id-1"

    def test_from_dict(self) -> None:
        data = {
            "event_type": "risk.alert",
            "timestamp": "2024-06-15T10:30:00+00:00",
            "payload": {"level": "critical"},
            "correlation_id": "corr-xyz",
        }
        event = Event.from_dict(data)
        assert event.event_type == "risk.alert"
        assert event.timestamp == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert event.payload == {"level": "critical"}
        assert event.correlation_id == "corr-xyz"

    def test_from_dict_missing_optional_fields(self) -> None:
        data = {
            "event_type": "test",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        event = Event.from_dict(data)
        assert event.payload == {}
        assert event.correlation_id is None

    def test_roundtrip_serialization(self) -> None:
        original = Event(
            event_type="market.tick",
            payload={"bid": 1.2345, "ask": 1.2347, "instrument": "EURUSD"},
            correlation_id="rt-001",
        )
        serialized = orjson.dumps(original.to_dict())
        deserialized = Event.from_dict(orjson.loads(serialized))
        assert deserialized.event_type == original.event_type
        assert deserialized.payload == original.payload
        assert deserialized.correlation_id == original.correlation_id


# ---------------------------------------------------------------------------
# Channel constants tests
# ---------------------------------------------------------------------------


class TestChannelConstants:
    """Tests for pre-defined channel constants."""

    def test_market_tick_template(self) -> None:
        assert MARKET_TICK == "market.tick.{instrument}"
        assert MARKET_TICK.format(instrument="EURUSD") == "market.tick.EURUSD"

    def test_signal_channels(self) -> None:
        assert SIGNAL_GENERATED == "signal.generated"
        assert SIGNAL_VALIDATED == "signal.validated"

    def test_order_channels(self) -> None:
        assert ORDER_SUBMITTED == "order.submitted"
        assert ORDER_FILLED == "order.filled"
        assert ORDER_REJECTED == "order.rejected"

    def test_risk_channels(self) -> None:
        assert RISK_ALERT == "risk.alert"
        assert KILL_SWITCH_ACTIVATED == "kill_switch.activated"
        assert KILL_SWITCH_DEACTIVATED == "kill_switch.deactivated"

    def test_strategy_channels(self) -> None:
        assert STRATEGY_DISABLED == "strategy.disabled"
        assert STRATEGY_ENABLED == "strategy.enabled"

    def test_news_channels(self) -> None:
        assert NEWS_ARTICLE_RECEIVED == "news.article_received"
        assert NEWS_CRISIS_ALERT == "news.crisis_alert"
        assert NEWS_ECONOMIC_EVENT == "news.economic_event_approaching"

    def test_hft_channels(self) -> None:
        assert HFT_CIRCUIT_BREAKER_ACTIVATED == "hft.circuit_breaker.activated"
        assert HFT_MODE_CHANGED == "hft.mode_changed"

    def test_mistake_channels(self) -> None:
        assert MISTAKE_PATTERN_DETECTED == "mistake.pattern_detected"
        assert MISTAKE_PATTERN_RESOLVED == "mistake.pattern_resolved"


# ---------------------------------------------------------------------------
# EventBus tests (mocked Redis)
# ---------------------------------------------------------------------------


class TestEventBus:
    """Tests for the EventBus class with mocked Redis."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        redis_mock = AsyncMock()
        redis_mock.publish = AsyncMock(return_value=1)
        redis_mock.close = AsyncMock()
        return redis_mock

    @pytest.fixture
    def mock_pubsub(self) -> MagicMock:
        pubsub_mock = AsyncMock()
        pubsub_mock.subscribe = AsyncMock()
        pubsub_mock.unsubscribe = AsyncMock()
        pubsub_mock.close = AsyncMock()

        async def _get_message(**kwargs):
            await asyncio.sleep(0.01)
            return None

        pubsub_mock.get_message = AsyncMock(side_effect=_get_message)
        return pubsub_mock

    @pytest.fixture
    async def bus(self, mock_redis: MagicMock, mock_pubsub: MagicMock) -> EventBus:
        """Create an EventBus with mocked Redis internals."""
        with patch("src.core.event_bus.aioredis.from_url", return_value=mock_redis):
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
            bus = EventBus(redis_url="redis://localhost:6379")
            await bus.start()
            yield bus
            await bus.stop()

    async def test_start_sets_running(self, bus: EventBus) -> None:
        assert bus.is_running is True

    async def test_stop_clears_state(self, mock_redis: MagicMock, mock_pubsub: MagicMock) -> None:
        with patch("src.core.event_bus.aioredis.from_url", return_value=mock_redis):
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
            bus = EventBus()
            await bus.start()
            assert bus.is_running is True
            await bus.stop()
            assert bus.is_running is False

    async def test_publish_serializes_event(
        self, bus: EventBus, mock_redis: MagicMock
    ) -> None:
        event = Event(event_type="test.publish", payload={"key": "value"})
        result = await bus.publish("test.channel", event)
        assert result == 1
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "test.channel"
        # Verify the data is valid orjson
        data = orjson.loads(call_args[0][1])
        assert data["event_type"] == "test.publish"
        assert data["payload"] == {"key": "value"}

    async def test_publish_raises_if_not_started(self) -> None:
        bus = EventBus()
        event = Event(event_type="test")
        with pytest.raises(RuntimeError, match="not started"):
            await bus.publish("channel", event)

    async def test_subscribe_registers_handler(
        self, bus: EventBus, mock_pubsub: MagicMock
    ) -> None:
        handler = AsyncMock()
        await bus.subscribe("test.channel", handler)
        mock_pubsub.subscribe.assert_called_once_with("test.channel")
        assert "test.channel" in bus._handlers
        assert handler in bus._handlers["test.channel"]

    async def test_subscribe_multiple_handlers(
        self, bus: EventBus, mock_pubsub: MagicMock
    ) -> None:
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        await bus.subscribe("ch", handler1)
        await bus.subscribe("ch", handler2)
        # subscribe to Redis only called once for the channel
        mock_pubsub.subscribe.assert_called_once_with("ch")
        assert len(bus._handlers["ch"]) == 2

    async def test_subscribe_raises_if_not_started(self) -> None:
        bus = EventBus()
        with pytest.raises(RuntimeError, match="not started"):
            await bus.subscribe("channel", AsyncMock())

    async def test_unsubscribe_removes_handlers(
        self, bus: EventBus, mock_pubsub: MagicMock
    ) -> None:
        handler = AsyncMock()
        await bus.subscribe("ch", handler)
        await bus.unsubscribe("ch")
        assert "ch" not in bus._handlers
        mock_pubsub.unsubscribe.assert_called_once_with("ch")

    async def test_unsubscribe_raises_if_not_started(self) -> None:
        bus = EventBus()
        with pytest.raises(RuntimeError, match="not started"):
            await bus.unsubscribe("channel")

    async def test_create_event_helper(self) -> None:
        event = EventBus.create_event(
            event_type="order.submitted",
            payload={"order_id": "o1"},
        )
        assert event.event_type == "order.submitted"
        assert event.payload == {"order_id": "o1"}
        assert event.correlation_id is not None  # auto-generated

    async def test_create_event_with_correlation_id(self) -> None:
        event = EventBus.create_event(
            event_type="test",
            correlation_id="my-id",
        )
        assert event.correlation_id == "my-id"

    async def test_dispatch_calls_handlers(self, bus: EventBus) -> None:
        handler = AsyncMock()
        await bus.subscribe("ch", handler)
        event = Event(event_type="test")
        await bus._dispatch("ch", event)
        handler.assert_called_once_with(event)

    async def test_dispatch_handler_error_does_not_propagate(self, bus: EventBus) -> None:
        failing_handler = AsyncMock(side_effect=ValueError("boom"))
        good_handler = AsyncMock()
        await bus.subscribe("ch", failing_handler)
        await bus.subscribe("ch", good_handler)
        event = Event(event_type="test")
        # Should not raise
        await bus._dispatch("ch", event)
        # Good handler still called
        good_handler.assert_called_once_with(event)

    async def test_start_is_idempotent(
        self, mock_redis: MagicMock, mock_pubsub: MagicMock
    ) -> None:
        with patch("src.core.event_bus.aioredis.from_url", return_value=mock_redis):
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
            bus = EventBus()
            await bus.start()
            await bus.start()  # second call should be no-op
            assert bus.is_running is True
            await bus.stop()
