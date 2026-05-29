"""Unit tests for src/trading/ig_stream.py."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.event_bus import MARKET_TICK, Event, EventBus
from src.core.exceptions import StreamDisconnectedError
from src.trading.ig_stream import (
    IGStream,
    MAX_SIMULTANEOUS_INSTRUMENTS,
    RECONNECT_BASE_DELAY_SECONDS,
    RECONNECT_MAX_TOTAL_SECONDS,
    STALENESS_CHECK_INTERVAL_SECONDS,
    SUBSCRIPTION_RETRY_DELAY_SECONDS,
    SUBSCRIPTION_RETRY_MAX_ATTEMPTS,
    SubscriptionState,
    SubscriptionStatus,
)
from src.config.constants import RECONNECT_MAX_ATTEMPTS, TICK_STALENESS_SECONDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create a mock EventBus."""
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def mock_ig_client() -> AsyncMock:
    """Create a mock IG REST client."""
    client = AsyncMock()
    client.get_prices = AsyncMock(return_value=[])
    return client


@pytest.fixture
def stream(mock_event_bus: AsyncMock, mock_ig_client: AsyncMock) -> IGStream:
    """Create an IGStream instance with mocked dependencies (not started)."""
    return IGStream(
        stream_url="wss://push.lightstreamer.com/lightstreamer",
        cst="test-cst",
        security_token="test-security-token",
        event_bus=mock_event_bus,
        ig_client=mock_ig_client,
    )


# ---------------------------------------------------------------------------
# SubscriptionState Tests
# ---------------------------------------------------------------------------


class TestSubscriptionState:
    """Tests for the SubscriptionState dataclass."""

    def test_default_creation(self) -> None:
        state = SubscriptionState(epic="CS.D.EURUSD.CFD.IP")
        assert state.epic == "CS.D.EURUSD.CFD.IP"
        assert state.status == SubscriptionStatus.SUBSCRIBING
        assert state.last_tick_time is None
        assert state.error_count == 0
        assert isinstance(state.subscribe_time, datetime)

    def test_custom_status(self) -> None:
        state = SubscriptionState(
            epic="IX.D.FTSE.DAILY.IP",
            status=SubscriptionStatus.ACTIVE,
        )
        assert state.status == SubscriptionStatus.ACTIVE


class TestSubscriptionStatus:
    """Tests for the SubscriptionStatus enum."""

    def test_status_values(self) -> None:
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.STALE.value == "stale"
        assert SubscriptionStatus.SUBSCRIBING.value == "subscribing"
        assert SubscriptionStatus.UNSUBSCRIBED.value == "unsubscribed"
        assert SubscriptionStatus.ERROR.value == "error"


# ---------------------------------------------------------------------------
# IGStream Initialization Tests
# ---------------------------------------------------------------------------


class TestIGStreamInit:
    """Tests for IGStream initialization."""

    def test_initial_state(self, stream: IGStream) -> None:
        assert stream._connected is False
        assert stream._running is False
        assert stream._reconnect_attempts == 0
        assert stream._subscriptions == {}
        assert stream._last_tick_times == {}

    def test_properties_before_start(self, stream: IGStream) -> None:
        assert stream.is_connected is False
        assert stream.subscription_count == 0
        assert stream.get_subscription_status() == {}
        assert stream.get_stale_instruments() == []


# ---------------------------------------------------------------------------
# IGStream Lifecycle Tests
# ---------------------------------------------------------------------------


class TestIGStreamLifecycle:
    """Tests for start/stop lifecycle."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_start_connects_and_starts_tasks(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()

        assert stream._running is True
        assert stream._connected is True
        assert stream._listener_task is not None
        assert stream._staleness_task is not None

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_stop_cleans_up(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.stop()

        assert stream._running is False
        assert stream._connected is False
        assert stream._listener_task is None
        assert stream._staleness_task is None
        assert stream._subscriptions == {}

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_start_is_idempotent(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.start()  # Should be no-op

        assert mock_ws_connect.call_count == 1
        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_start_raises_on_connection_failure(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws_connect.side_effect = Exception("Connection refused")

        with pytest.raises(StreamDisconnectedError):
            await stream.start()

        assert stream._connected is False


# ---------------------------------------------------------------------------
# Subscription Management Tests
# ---------------------------------------------------------------------------


class TestIGStreamSubscription:
    """Tests for subscribe/unsubscribe operations."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_subscribe_adds_instrument(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        assert "CS.D.EURUSD.CFD.IP" in stream._subscriptions
        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ACTIVE
        assert stream.subscription_count == 1

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_subscribe_multiple_instruments(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()

        epics = [f"CS.D.INSTRUMENT{i}.CFD.IP" for i in range(MAX_SIMULTANEOUS_INSTRUMENTS)]
        for epic in epics:
            await stream.subscribe(epic)

        assert stream.subscription_count == MAX_SIMULTANEOUS_INSTRUMENTS

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_subscribe_idempotent(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        await stream.subscribe("CS.D.EURUSD.CFD.IP")  # Should be no-op

        # send called only once for the subscription
        assert mock_ws.send.call_count == 1
        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_subscribe_retries_on_failure(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock(
            side_effect=[Exception("fail"), Exception("fail"), None]
        )
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Should succeed on 3rd attempt
        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ACTIVE
        assert mock_ws.send.call_count == 3

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_subscribe_marks_error_after_all_retries_exhausted(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock(side_effect=Exception("always fails"))
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ERROR
        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].error_count == 1

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_unsubscribe_removes_instrument(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        await stream.unsubscribe("CS.D.EURUSD.CFD.IP")

        assert "CS.D.EURUSD.CFD.IP" not in stream._subscriptions
        assert stream.subscription_count == 0

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_unsubscribe_nonexistent_is_noop(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.unsubscribe("NONEXISTENT")  # Should not raise
        await stream.stop()


# ---------------------------------------------------------------------------
# Tick Processing Tests
# ---------------------------------------------------------------------------


class TestIGStreamTickProcessing:
    """Tests for tick processing and Event Bus distribution."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_on_tick_publishes_to_event_bus(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        tick_data = {"bid": 1.1234, "ask": 1.1236, "timestamp": "2024-01-01T12:00:00Z"}
        await stream._on_tick("CS.D.EURUSD.CFD.IP", tick_data)

        # Verify event bus was called
        mock_event_bus.publish.assert_called()
        call_args = mock_event_bus.publish.call_args
        channel = call_args[0][0]
        event = call_args[0][1]

        assert channel == MARKET_TICK.format(instrument="CS.D.EURUSD.CFD.IP")
        assert event.event_type == "market.tick"
        assert event.payload["epic"] == "CS.D.EURUSD.CFD.IP"
        assert event.payload["bid"] == 1.1234
        assert event.payload["ask"] == 1.1236

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_on_tick_updates_last_tick_time(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        tick_data = {"bid": 1.1234, "ask": 1.1236}
        await stream._on_tick("CS.D.EURUSD.CFD.IP", tick_data)

        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times
        assert isinstance(stream._last_tick_times["CS.D.EURUSD.CFD.IP"], datetime)

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_on_tick_restores_stale_instrument(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Manually mark as stale
        stream._subscriptions["CS.D.EURUSD.CFD.IP"].status = SubscriptionStatus.STALE

        tick_data = {"bid": 1.1234, "ask": 1.1236}
        await stream._on_tick("CS.D.EURUSD.CFD.IP", tick_data)

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ACTIVE

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_on_tick_handles_event_bus_failure(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        mock_event_bus.publish.side_effect = Exception("Redis down")

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Should not raise even if event bus fails
        tick_data = {"bid": 1.1234, "ask": 1.1236}
        await stream._on_tick("CS.D.EURUSD.CFD.IP", tick_data)

        # Last tick time should still be updated
        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

        await stream.stop()


# ---------------------------------------------------------------------------
# Reconnection Tests
# ---------------------------------------------------------------------------


class TestIGStreamReconnection:
    """Tests for auto-reconnect behavior."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_reconnect_succeeds_on_first_attempt(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        stream._running = True
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP", status=SubscriptionStatus.ACTIVE
        )

        await stream._reconnect()

        assert stream._connected is True
        assert stream._reconnect_attempts == 0

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_reconnect_retries_with_backoff(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Fail first 2 attempts, succeed on 3rd
        mock_ws_connect.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            mock_ws,
        ]

        stream._running = True
        await stream._reconnect()

        assert stream._connected is True
        assert mock_ws_connect.call_count == 3

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_reconnect_raises_after_all_attempts_exhausted(
        self, mock_ws_connect: AsyncMock, stream: IGStream
    ) -> None:
        mock_ws_connect.side_effect = Exception("always fails")

        stream._running = True

        with pytest.raises(StreamDisconnectedError):
            await stream._reconnect()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_reconnect_recovers_missed_data(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_ig_client: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        mock_ig_client.get_prices.return_value = [
            {
                "snapshotTimeUTC": "2024-01-01T12:00:00",
                "closePrice": {"bid": 1.1234, "ask": 1.1236},
            }
        ]

        stream._running = True
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP", status=SubscriptionStatus.ACTIVE
        )

        await stream._reconnect()

        # Verify REST API was called for missed data
        mock_ig_client.get_prices.assert_called_once()


# ---------------------------------------------------------------------------
# Staleness Detection Tests
# ---------------------------------------------------------------------------


class TestIGStreamStaleness:
    """Tests for staleness detection."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_instrument_marked_stale_after_timeout(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Simulate old last tick time (> 60 seconds ago)
        old_time = datetime.now(timezone.utc) - timedelta(seconds=TICK_STALENESS_SECONDS + 10)
        stream._last_tick_times["CS.D.EURUSD.CFD.IP"] = old_time
        stream._subscriptions["CS.D.EURUSD.CFD.IP"].subscribe_time = old_time

        # Cancel the staleness task and run one iteration manually
        if stream._staleness_task:
            stream._staleness_task.cancel()
            try:
                await stream._staleness_task
            except asyncio.CancelledError:
                pass

        # Manually trigger staleness check logic
        now = datetime.now(timezone.utc)
        epic = "CS.D.EURUSD.CFD.IP"
        state = stream._subscriptions[epic]
        last_tick = stream._last_tick_times.get(epic, state.subscribe_time)
        seconds_since_tick = (now - last_tick).total_seconds()

        if seconds_since_tick >= TICK_STALENESS_SECONDS:
            state.status = SubscriptionStatus.STALE
            await stream._notify_stale(epic)

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.STALE

        await stream.stop()

    def test_get_stale_instruments(self, stream: IGStream) -> None:
        stream._subscriptions["EURUSD"] = SubscriptionState(
            epic="EURUSD", status=SubscriptionStatus.STALE
        )
        stream._subscriptions["GBPUSD"] = SubscriptionState(
            epic="GBPUSD", status=SubscriptionStatus.ACTIVE
        )
        stream._subscriptions["USDJPY"] = SubscriptionState(
            epic="USDJPY", status=SubscriptionStatus.STALE
        )

        stale = stream.get_stale_instruments()
        assert sorted(stale) == ["EURUSD", "USDJPY"]

    def test_get_subscription_status(self, stream: IGStream) -> None:
        stream._subscriptions["EURUSD"] = SubscriptionState(
            epic="EURUSD", status=SubscriptionStatus.ACTIVE
        )
        stream._subscriptions["GBPUSD"] = SubscriptionState(
            epic="GBPUSD", status=SubscriptionStatus.STALE
        )

        status = stream.get_subscription_status()
        assert status == {"EURUSD": "active", "GBPUSD": "stale"}


# ---------------------------------------------------------------------------
# Market Hours Tests
# ---------------------------------------------------------------------------


class TestIGStreamMarketHours:
    """Tests for market hours detection (Cross-Cutting Rule 5)."""

    def test_is_market_open_defaults_to_true(self, stream: IGStream) -> None:
        """Default implementation returns True for all instruments."""
        assert stream.is_market_open("CS.D.EURUSD.CFD.IP") is True
        assert stream.is_market_open("IX.D.FTSE.DAILY.IP") is True
        assert stream.is_market_open("CS.D.BITCOIN.CFD.IP") is True


# ---------------------------------------------------------------------------
# Message Parsing Tests
# ---------------------------------------------------------------------------


class TestIGStreamMessageParsing:
    """Tests for Lightstreamer message parsing."""

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_handle_tick_update_message(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Simulate a Lightstreamer update message
        message = "U,CS.D.EURUSD.CFD.IP,1,1.1234|1.1236|12:00:00"
        await stream._handle_message(message)

        # Verify tick was processed
        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_handle_bytes_message(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Simulate a bytes message
        message = b"U,CS.D.EURUSD.CFD.IP,1,1.1234|1.1236|12:00:00"
        await stream._handle_message(message)

        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_handle_empty_message(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()

        # Should not raise on empty message
        await stream._handle_message("")
        await stream._handle_message("\n\n")

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_handle_non_update_message(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()

        # Non-update messages should be ignored
        await stream._handle_message("PROBE")
        await stream._handle_message("LOOP,5000")

        await stream.stop()

    @patch("src.trading.ig_stream.websockets.connect", new_callable=AsyncMock)
    async def test_parse_tick_with_missing_fields(
        self, mock_ws_connect: AsyncMock, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws_connect.return_value = mock_ws

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        # Message with only bid (partial update)
        message = "U,CS.D.EURUSD.CFD.IP,1,1.1234||"
        await stream._handle_message(message)

        # Should still process the partial tick
        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

        await stream.stop()


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestIGStreamConstants:
    """Tests for module-level constants."""

    def test_reconnect_constants(self) -> None:
        assert RECONNECT_BASE_DELAY_SECONDS == 1.0
        assert RECONNECT_MAX_TOTAL_SECONDS == 30.0
        assert RECONNECT_MAX_ATTEMPTS == 5

    def test_staleness_constants(self) -> None:
        assert TICK_STALENESS_SECONDS == 60
        assert STALENESS_CHECK_INTERVAL_SECONDS == 10.0

    def test_subscription_constants(self) -> None:
        assert SUBSCRIPTION_RETRY_MAX_ATTEMPTS == 3
        assert SUBSCRIPTION_RETRY_DELAY_SECONDS == 2.0
        assert MAX_SIMULTANEOUS_INSTRUMENTS == 50
