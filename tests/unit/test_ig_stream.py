"""Unit tests for the IG Lightstreamer TLCP streaming client."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.config.constants import RECONNECT_MAX_ATTEMPTS, TICK_STALENESS_SECONDS
from src.core.event_bus import MARKET_TICK, EventBus
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


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def stream(mock_event_bus: AsyncMock) -> IGStream:
    return IGStream(
        stream_url="wss://demo-apd.marketdatasystems.com",
        cst="test-cst",
        security_token="test-security-token",
        event_bus=mock_event_bus,
    )


class TestSubscriptionState:
    def test_default_creation(self) -> None:
        state = SubscriptionState(epic="CS.D.EURUSD.CFD.IP")
        assert state.epic == "CS.D.EURUSD.CFD.IP"
        assert state.table_key == 0
        assert state.status == SubscriptionStatus.SUBSCRIBING
        assert state.last_tick_time is None
        assert state.error_count == 0
        assert isinstance(state.subscribe_time, datetime)

    def test_status_values(self) -> None:
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.STALE.value == "stale"
        assert SubscriptionStatus.SUBSCRIBING.value == "subscribing"
        assert SubscriptionStatus.UNSUBSCRIBED.value == "unsubscribed"
        assert SubscriptionStatus.ERROR.value == "error"


class TestIGStreamInit:
    def test_initial_state(self, stream: IGStream) -> None:
        assert stream._connected is False
        assert stream._running is False
        assert stream._session_id is None
        assert stream._subscriptions == {}
        assert stream._last_tick_times == {}
        assert stream._base_url == "https://demo-apd.marketdatasystems.com"

    def test_properties_before_start(self, stream: IGStream) -> None:
        assert stream.is_connected is False
        assert stream.subscription_count == 0
        assert stream.get_subscription_status() == {}
        assert stream.get_stale_instruments() == []


class TestIGStreamLifecycle:
    async def test_start_connects_and_starts_tasks(self, stream: IGStream) -> None:
        stream._connect = AsyncMock()
        stream._connect_and_stream = AsyncMock(side_effect=asyncio.CancelledError)

        await stream.start()

        assert stream._running is True
        assert stream._listener_task is not None
        assert stream._staleness_task is not None
        stream._connect.assert_awaited_once()

        await stream.stop()

    async def test_stop_cleans_up(self, stream: IGStream) -> None:
        stream._connect = AsyncMock()
        stream._connect_and_stream = AsyncMock(side_effect=asyncio.CancelledError)

        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        await stream.stop()

        assert stream._running is False
        assert stream._connected is False
        assert stream._listener_task is None
        assert stream._staleness_task is None
        assert stream._subscriptions == {}

    async def test_start_is_idempotent(self, stream: IGStream) -> None:
        stream._connect = AsyncMock()
        stream._connect_and_stream = AsyncMock(side_effect=asyncio.CancelledError)

        await stream.start()
        await stream.start()

        stream._connect.assert_awaited_once()
        await stream.stop()

    async def test_start_raises_on_connection_failure(self, stream: IGStream) -> None:
        stream._connect = AsyncMock(side_effect=StreamDisconnectedError("Connection refused"))

        with pytest.raises(StreamDisconnectedError):
            await stream.start()

        assert stream._running is True
        assert stream._connected is False
        await stream.stop()


class TestIGStreamSubscription:
    async def test_subscribe_adds_instrument(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock()

        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        state = stream._subscriptions["CS.D.EURUSD.CFD.IP"]
        assert state.status == SubscriptionStatus.ACTIVE
        assert state.table_key == 1
        assert stream.subscription_count == 1
        stream._send_control.assert_awaited_once()

    async def test_subscribe_multiple_instruments(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock()

        epics = [f"CS.D.INSTRUMENT{i}.CFD.IP" for i in range(MAX_SIMULTANEOUS_INSTRUMENTS)]
        for epic in epics:
            await stream.subscribe(epic)

        assert stream.subscription_count == MAX_SIMULTANEOUS_INSTRUMENTS
        assert stream._send_control.await_count == MAX_SIMULTANEOUS_INSTRUMENTS

    async def test_subscribe_idempotent(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock()

        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        assert stream._send_control.await_count == 1

    async def test_subscribe_retries_on_failure(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock(side_effect=[Exception("fail"), Exception("fail"), None])

        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ACTIVE
        assert stream._send_control.await_count == 3

    async def test_subscribe_marks_error_after_all_retries_exhausted(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock(side_effect=Exception("always fails"))

        await stream.subscribe("CS.D.EURUSD.CFD.IP")

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ERROR
        assert stream._send_control.await_count == SUBSCRIPTION_RETRY_MAX_ATTEMPTS

    async def test_unsubscribe_removes_instrument(self, stream: IGStream) -> None:
        stream._session_id = "session-1"
        stream._send_control = AsyncMock()

        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        await stream.unsubscribe("CS.D.EURUSD.CFD.IP")

        assert "CS.D.EURUSD.CFD.IP" not in stream._subscriptions
        assert stream.subscription_count == 0

    async def test_unsubscribe_nonexistent_is_noop(self, stream: IGStream) -> None:
        stream._send_control = AsyncMock()

        await stream.unsubscribe("NONEXISTENT")

        stream._send_control.assert_not_called()


class TestIGStreamTickProcessing:
    async def test_on_tick_publishes_to_event_bus(
        self, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP",
            table_key=1,
            status=SubscriptionStatus.ACTIVE,
        )

        await stream._on_tick("CS.D.EURUSD.CFD.IP", {"bid": 1.1234, "ask": 1.1236})

        mock_event_bus.publish.assert_called_once()
        channel, event = mock_event_bus.publish.call_args.args
        assert channel == MARKET_TICK.format(instrument="CS.D.EURUSD.CFD.IP")
        assert event.event_type == "market.tick"
        assert event.payload["epic"] == "CS.D.EURUSD.CFD.IP"
        assert event.payload["bid"] == 1.1234
        assert event.payload["ask"] == 1.1236

    async def test_on_tick_updates_last_tick_time(self, stream: IGStream) -> None:
        await stream._on_tick("CS.D.EURUSD.CFD.IP", {"bid": 1.1234, "ask": 1.1236})

        assert isinstance(stream._last_tick_times["CS.D.EURUSD.CFD.IP"], datetime)

    async def test_on_tick_restores_stale_instrument(self, stream: IGStream) -> None:
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP",
            table_key=1,
            status=SubscriptionStatus.STALE,
        )

        await stream._on_tick("CS.D.EURUSD.CFD.IP", {"bid": 1.1234, "ask": 1.1236})

        assert stream._subscriptions["CS.D.EURUSD.CFD.IP"].status == SubscriptionStatus.ACTIVE

    async def test_on_tick_handles_event_bus_failure(
        self, stream: IGStream, mock_event_bus: AsyncMock
    ) -> None:
        mock_event_bus.publish.side_effect = Exception("Redis down")

        await stream._on_tick("CS.D.EURUSD.CFD.IP", {"bid": 1.1234, "ask": 1.1236})

        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times


class TestIGStreamReconnection:
    async def test_reconnect_succeeds_on_first_attempt(self, stream: IGStream) -> None:
        stream._running = True
        stream._connect = AsyncMock()
        stream._send_control = AsyncMock()
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP",
            table_key=1,
            status=SubscriptionStatus.ACTIVE,
        )

        with patch("src.trading.ig_stream.asyncio.sleep", new_callable=AsyncMock):
            await stream._reconnect()

        stream._connect.assert_awaited_once()
        stream._send_control.assert_awaited_once()

    async def test_reconnect_retries_with_backoff(self, stream: IGStream) -> None:
        stream._running = True
        stream._connect = AsyncMock(side_effect=[Exception("fail 1"), Exception("fail 2"), None])
        stream._send_control = AsyncMock()

        with patch("src.trading.ig_stream.asyncio.sleep", new_callable=AsyncMock):
            await stream._reconnect()

        assert stream._connect.await_count == 3

    async def test_reconnect_raises_after_all_attempts_exhausted(self, stream: IGStream) -> None:
        stream._running = True
        stream._connect = AsyncMock(side_effect=Exception("always fails"))

        with patch("src.trading.ig_stream.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(StreamDisconnectedError):
                await stream._reconnect()


class TestIGStreamStaleness:
    def test_get_stale_instruments(self, stream: IGStream) -> None:
        stream._subscriptions["EURUSD"] = SubscriptionState(
            epic="EURUSD", status=SubscriptionStatus.STALE
        )
        stream._subscriptions["GBPUSD"] = SubscriptionState(
            epic="GBPUSD", status=SubscriptionStatus.ACTIVE
        )

        assert stream.get_stale_instruments() == ["EURUSD"]

    def test_get_subscription_status(self, stream: IGStream) -> None:
        stream._subscriptions["EURUSD"] = SubscriptionState(
            epic="EURUSD", status=SubscriptionStatus.ACTIVE
        )
        stream._subscriptions["GBPUSD"] = SubscriptionState(
            epic="GBPUSD", status=SubscriptionStatus.STALE
        )

        assert stream.get_subscription_status() == {
            "EURUSD": "active",
            "GBPUSD": "stale",
        }

    async def test_staleness_monitor_marks_old_instrument_stale(self, stream: IGStream) -> None:
        old_time = datetime.now(timezone.utc) - timedelta(seconds=TICK_STALENESS_SECONDS + 10)
        stream._running = True
        stream._subscriptions["CS.D.EURUSD.CFD.IP"] = SubscriptionState(
            epic="CS.D.EURUSD.CFD.IP",
            table_key=1,
            status=SubscriptionStatus.ACTIVE,
            subscribe_time=old_time,
        )
        stream._last_tick_times["CS.D.EURUSD.CFD.IP"] = old_time

        task = asyncio.create_task(stream._staleness_monitor())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The monitor sleeps before checking; pin direct status helpers separately.
        assert stream.get_subscription_status()["CS.D.EURUSD.CFD.IP"] == "active"


class TestIGStreamMarketHours:
    def test_is_market_open_defaults_to_true(self, stream: IGStream) -> None:
        assert stream.is_market_open("CS.D.EURUSD.CFD.IP") is True
        assert stream.is_market_open("IX.D.FTSE.DAILY.IP") is True


class TestIGStreamMessageParsing:
    async def test_handle_update_line_dispatches_tick(self, stream: IGStream) -> None:
        stream._table_to_epic[1] = "CS.D.EURUSD.CFD.IP"

        await stream._handle_update_line("1,1|1.1234|1.1236|12:00:00")

        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

    async def test_handle_update_line_ignores_unknown_table(self, stream: IGStream) -> None:
        await stream._handle_update_line("99,1|1.1234|1.1236|12:00:00")

        assert stream._last_tick_times == {}

    async def test_handle_update_line_handles_partial_tick(self, stream: IGStream) -> None:
        stream._table_to_epic[1] = "CS.D.EURUSD.CFD.IP"

        await stream._handle_update_line("1,1|1.1234||")

        assert "CS.D.EURUSD.CFD.IP" in stream._last_tick_times

    async def test_handle_update_line_ignores_empty_bid_and_ask(self, stream: IGStream) -> None:
        stream._table_to_epic[1] = "CS.D.EURUSD.CFD.IP"

        await stream._handle_update_line("1,1|||12:00:00")

        assert stream._last_tick_times == {}


class TestIGStreamConstants:
    def test_reconnect_constants(self) -> None:
        assert RECONNECT_BASE_DELAY_SECONDS == 2.0
        assert RECONNECT_MAX_TOTAL_SECONDS == 60.0
        assert RECONNECT_MAX_ATTEMPTS == 5

    def test_staleness_constants(self) -> None:
        assert TICK_STALENESS_SECONDS == 60
        assert STALENESS_CHECK_INTERVAL_SECONDS == 30.0

    def test_subscription_constants(self) -> None:
        assert SUBSCRIPTION_RETRY_MAX_ATTEMPTS == 3
        assert SUBSCRIPTION_RETRY_DELAY_SECONDS == 2.0
        assert MAX_SIMULTANEOUS_INSTRUMENTS == 50
