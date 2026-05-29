"""Unit tests for the EconomicCalendar module.

Tests the new functionality added in task 38.1:
- Daily event fetching (00:00 UTC)
- High-impact event storage (NFP, CPI, rate decisions, GDP, central bank speeches)
- Event data model with event_type, scheduled_time, currency/region, expected_impact
- get_upcoming_events(hours_ahead=24)
- get_events_within(minutes)
- is_high_impact_event_near(instrument, minutes=15)
- Configurable event source (API endpoint or static schedule)
- NEWS_ECONOMIC_EVENT publishing to Event Bus (15 min before)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.news.economic_calendar import (
    APIEndpointProvider,
    EconomicCalendar,
    EconomicEventData,
    EventImpact,
    HIGH_IMPACT_EVENT_TYPES,
    HighImpactEventType,
    PRE_EVENT_NOTIFICATION_MINUTES,
    PreEventAdjustment,
    StaticScheduleProvider,
)
from src.config.constants import (
    NEWS_EVENT_SIZE_REDUCTION_FACTOR,
    NEWS_PRE_EVENT_RISK_WINDOW_MINUTES,
    NEWS_PRE_EVENT_SIGNAL_PAUSE_MINUTES,
)
from src.core.event_bus import NEWS_ECONOMIC_EVENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(
    minutes_ahead: int = 10,
    impact: str = "HIGH",
    event_type: str = "NFP",
    instruments: list[str] | None = None,
    currency_region: str = "USD",
    event_name: str = "US Non-Farm Payrolls",
) -> EconomicEventData:
    """Create a test economic event."""
    return EconomicEventData(
        id=str(uuid.uuid4()),
        event_name=event_name,
        event_type=event_type,
        scheduled_at=datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead),
        impact_level=impact,
        currency_region=currency_region,
        correlated_instruments=instruments or ["EURUSD", "GBPUSD", "US30"],
    )


# ---------------------------------------------------------------------------
# EconomicEventData Tests
# ---------------------------------------------------------------------------


class TestEconomicEventData:
    """Tests for the EconomicEventData dataclass."""

    def test_create_event_with_all_fields(self) -> None:
        """Event can be created with all required fields."""
        scheduled = datetime.now(timezone.utc) + timedelta(hours=2)
        event = EconomicEventData(
            id="evt-001",
            event_name="US CPI Release",
            event_type="CPI",
            scheduled_at=scheduled,
            impact_level="HIGH",
            currency_region="USD",
            correlated_instruments=["EURUSD", "USDJPY"],
            forecast_value="3.2%",
            previous_value="3.1%",
        )

        assert event.id == "evt-001"
        assert event.event_name == "US CPI Release"
        assert event.event_type == "CPI"
        assert event.scheduled_at == scheduled
        assert event.impact_level == "HIGH"
        assert event.currency_region == "USD"
        assert event.correlated_instruments == ["EURUSD", "USDJPY"]
        assert event.forecast_value == "3.2%"
        assert event.previous_value == "3.1%"

    def test_backward_compat_scheduled_time_property(self) -> None:
        """scheduled_time property returns scheduled_at value."""
        event = make_event()
        assert event.scheduled_time == event.scheduled_at

    def test_backward_compat_expected_impact_property(self) -> None:
        """expected_impact property returns impact_level value."""
        event = make_event(impact="MEDIUM")
        assert event.expected_impact == "MEDIUM"

    def test_default_impact_is_high(self) -> None:
        """Default impact level is HIGH."""
        event = EconomicEventData(
            id="x",
            event_name="Test",
            event_type="NFP",
            scheduled_at=datetime.now(timezone.utc),
        )
        assert event.impact_level == "HIGH"

    def test_notified_defaults_to_false(self) -> None:
        """Notified flag defaults to False."""
        event = make_event()
        assert event.notified is False


# ---------------------------------------------------------------------------
# HighImpactEventType Tests
# ---------------------------------------------------------------------------


class TestHighImpactEventTypes:
    """Tests for high-impact event type constants."""

    def test_all_required_types_present(self) -> None:
        """All required high-impact event types are defined."""
        assert "NFP" in HIGH_IMPACT_EVENT_TYPES
        assert "CPI" in HIGH_IMPACT_EVENT_TYPES
        assert "RATE_DECISION" in HIGH_IMPACT_EVENT_TYPES
        assert "GDP" in HIGH_IMPACT_EVENT_TYPES
        assert "CENTRAL_BANK_SPEECH" in HIGH_IMPACT_EVENT_TYPES

    def test_enum_values_match_set(self) -> None:
        """Enum values match the HIGH_IMPACT_EVENT_TYPES set."""
        for member in HighImpactEventType:
            assert member.value in HIGH_IMPACT_EVENT_TYPES


# ---------------------------------------------------------------------------
# StaticScheduleProvider Tests
# ---------------------------------------------------------------------------


class TestStaticScheduleProvider:
    """Tests for the StaticScheduleProvider."""

    @pytest.mark.asyncio
    async def test_fetch_events_returns_matching_date(self) -> None:
        """Returns events matching the requested date."""
        today = datetime.now(timezone.utc)
        events = [
            {
                "id": "1",
                "event_name": "NFP",
                "event_type": "NFP",
                "scheduled_at": today.replace(hour=13, minute=30),
                "impact_level": "HIGH",
            },
            {
                "id": "2",
                "event_name": "CPI",
                "event_type": "CPI",
                "scheduled_at": today + timedelta(days=1),
                "impact_level": "HIGH",
            },
        ]
        provider = StaticScheduleProvider(events=events)
        result = await provider.fetch_events(today)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_fetch_events_empty_when_no_match(self) -> None:
        """Returns empty list when no events match the date."""
        provider = StaticScheduleProvider(events=[
            {
                "id": "1",
                "event_name": "NFP",
                "event_type": "NFP",
                "scheduled_at": datetime.now(timezone.utc) + timedelta(days=5),
                "impact_level": "HIGH",
            },
        ])
        result = await provider.fetch_events(datetime.now(timezone.utc))
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_add_events(self) -> None:
        """Events can be added to the static schedule."""
        provider = StaticScheduleProvider()
        today = datetime.now(timezone.utc)
        provider.add_events([
            {"id": "1", "scheduled_at": today, "event_name": "Test", "event_type": "NFP"},
        ])
        result = await provider.fetch_events(today)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_clear_events(self) -> None:
        """Clear removes all events."""
        today = datetime.now(timezone.utc)
        provider = StaticScheduleProvider(events=[
            {"id": "1", "scheduled_at": today, "event_name": "Test", "event_type": "NFP"},
        ])
        provider.clear()
        result = await provider.fetch_events(today)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# EconomicCalendar - Query Methods Tests
# ---------------------------------------------------------------------------


class TestEconomicCalendarQueries:
    """Tests for EconomicCalendar query methods."""

    def setup_method(self) -> None:
        self.calendar = EconomicCalendar()

    def test_get_upcoming_events_hours_ahead(self) -> None:
        """get_upcoming_events with hours_ahead returns events within window."""
        event = make_event(minutes_ahead=60)  # 1 hour ahead
        self.calendar.add_event(event)

        result = self.calendar.get_upcoming_events(hours_ahead=2)
        assert len(result) == 1
        assert result[0].id == event.id

    def test_get_upcoming_events_hours_ahead_excludes_far(self) -> None:
        """get_upcoming_events excludes events beyond the window."""
        event = make_event(minutes_ahead=180)  # 3 hours ahead
        self.calendar.add_event(event)

        result = self.calendar.get_upcoming_events(hours_ahead=2)
        assert len(result) == 0

    def test_get_upcoming_events_default_24h(self) -> None:
        """get_upcoming_events defaults to 24 hours when no args given."""
        event = make_event(minutes_ahead=600)  # 10 hours ahead
        self.calendar.add_event(event)

        result = self.calendar.get_upcoming_events()
        assert len(result) == 1

    def test_get_upcoming_events_within_minutes_backward_compat(self) -> None:
        """get_upcoming_events with within_minutes works for backward compat."""
        event = make_event(minutes_ahead=10)
        self.calendar.add_event(event)

        result = self.calendar.get_upcoming_events(within_minutes=15)
        assert len(result) == 1

    def test_get_upcoming_events_sorted_by_time(self) -> None:
        """Events are returned sorted by scheduled time."""
        event1 = make_event(minutes_ahead=30)
        event2 = make_event(minutes_ahead=10)
        event3 = make_event(minutes_ahead=20)
        self.calendar.add_event(event1)
        self.calendar.add_event(event2)
        self.calendar.add_event(event3)

        result = self.calendar.get_upcoming_events(hours_ahead=1)
        assert result[0].id == event2.id
        assert result[1].id == event3.id
        assert result[2].id == event1.id

    def test_get_events_within(self) -> None:
        """get_events_within returns events within the minutes window."""
        event_near = make_event(minutes_ahead=5)
        event_far = make_event(minutes_ahead=20)
        self.calendar.add_event(event_near)
        self.calendar.add_event(event_far)

        result = self.calendar.get_events_within(10)
        assert len(result) == 1
        assert result[0].id == event_near.id

    def test_get_events_within_excludes_past(self) -> None:
        """get_events_within excludes past events."""
        event = make_event(minutes_ahead=-5)
        self.calendar.add_event(event)

        result = self.calendar.get_events_within(10)
        assert len(result) == 0

    def test_is_high_impact_event_near_true(self) -> None:
        """Returns True when high-impact event is near for instrument."""
        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.is_high_impact_event_near("EURUSD") is True

    def test_is_high_impact_event_near_false_too_far(self) -> None:
        """Returns False when event is beyond the window."""
        event = make_event(minutes_ahead=20, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.is_high_impact_event_near("EURUSD", minutes=15) is False

    def test_is_high_impact_event_near_false_uncorrelated(self) -> None:
        """Returns False for uncorrelated instruments."""
        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.is_high_impact_event_near("USDJPY") is False

    def test_is_high_impact_event_near_false_low_impact(self) -> None:
        """Returns False for low-impact events."""
        event = make_event(minutes_ahead=10, impact="LOW", instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.is_high_impact_event_near("EURUSD") is False

    def test_is_high_impact_event_near_custom_window(self) -> None:
        """Custom minutes window is respected."""
        event = make_event(minutes_ahead=25, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.is_high_impact_event_near("EURUSD", minutes=30) is True
        assert self.calendar.is_high_impact_event_near("EURUSD", minutes=20) is False

    def test_get_high_impact_events(self) -> None:
        """get_high_impact_events returns only HIGH impact events."""
        high = make_event(impact="HIGH")
        medium = make_event(impact="MEDIUM")
        low = make_event(impact="LOW")
        self.calendar.add_event(high)
        self.calendar.add_event(medium)
        self.calendar.add_event(low)

        result = self.calendar.get_high_impact_events()
        assert len(result) == 1
        assert result[0].id == high.id


# ---------------------------------------------------------------------------
# EconomicCalendar - Event Source Integration Tests
# ---------------------------------------------------------------------------


class TestEconomicCalendarEventSource:
    """Tests for configurable event source integration."""

    @pytest.mark.asyncio
    async def test_update_daily_with_static_provider(self) -> None:
        """update_daily fetches from static provider."""
        today = datetime.now(timezone.utc)
        provider = StaticScheduleProvider(events=[
            {
                "id": "nfp-1",
                "event_name": "US NFP",
                "event_type": "NFP",
                "scheduled_at": today.replace(hour=13, minute=30),
                "impact_level": "HIGH",
                "currency_region": "USD",
                "correlated_instruments": ["EURUSD", "GBPUSD"],
            },
        ])
        calendar = EconomicCalendar(event_source=provider)
        await calendar.update_daily()

        assert len(calendar.events) == 1
        assert calendar.events[0].event_name == "US NFP"
        assert calendar.events[0].currency_region == "USD"
        assert calendar.last_update is not None

    @pytest.mark.asyncio
    async def test_update_daily_without_provider(self) -> None:
        """update_daily works without a provider (no events fetched)."""
        calendar = EconomicCalendar()
        await calendar.update_daily()

        assert calendar.last_update is not None
        assert len(calendar.events) == 0

    @pytest.mark.asyncio
    async def test_update_daily_replaces_events(self) -> None:
        """update_daily replaces existing events with fresh data."""
        today = datetime.now(timezone.utc)
        provider = StaticScheduleProvider(events=[
            {
                "id": "1",
                "event_name": "Event 1",
                "event_type": "CPI",
                "scheduled_at": today.replace(hour=10),
                "impact_level": "HIGH",
            },
        ])
        calendar = EconomicCalendar(event_source=provider)

        # Add a manual event first
        calendar.add_event(make_event())
        assert len(calendar.events) == 1

        # Update should replace with provider data
        await calendar.update_daily()
        assert len(calendar.events) == 1
        assert calendar.events[0].event_name == "Event 1"

    @pytest.mark.asyncio
    async def test_update_daily_handles_provider_error(self) -> None:
        """update_daily handles provider errors gracefully."""
        class FailingProvider:
            async def fetch_events(self, date: datetime) -> list[dict[str, Any]]:
                raise RuntimeError("API unavailable")

        calendar = EconomicCalendar(event_source=FailingProvider())
        calendar.add_event(make_event())

        # Should not crash, keeps existing events
        await calendar.update_daily()
        assert calendar.last_update is not None

    @pytest.mark.asyncio
    async def test_parse_event_with_string_datetime(self) -> None:
        """Events with ISO string datetimes are parsed correctly."""
        today = datetime.now(timezone.utc)
        provider = StaticScheduleProvider(events=[
            {
                "id": "1",
                "event_name": "GDP",
                "event_type": "GDP",
                "scheduled_at": today.replace(hour=14).isoformat(),
                "impact_level": "HIGH",
                "currency_region": "EUR",
            },
        ])
        # Patch the provider to return string datetime
        calendar = EconomicCalendar(event_source=provider)

        # Manually test _parse_event with string
        result = calendar._parse_event({
            "id": "1",
            "event_name": "GDP",
            "event_type": "GDP",
            "scheduled_at": today.replace(hour=14).isoformat(),
            "impact_level": "HIGH",
            "currency_region": "EUR",
        })
        assert result is not None
        assert result.currency_region == "EUR"


# ---------------------------------------------------------------------------
# EconomicCalendar - Event Bus Publishing Tests
# ---------------------------------------------------------------------------


class TestEconomicCalendarEventBus:
    """Tests for NEWS_ECONOMIC_EVENT publishing."""

    @pytest.mark.asyncio
    async def test_approaching_event_published(self) -> None:
        """Approaching high-impact events are published to event bus."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        calendar.add_event(event)

        await calendar._check_approaching_events()

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == NEWS_ECONOMIC_EVENT
        payload = call_args[0][1].payload
        assert payload["event_id"] == event.id
        assert payload["notification_type"] == "approaching"
        assert payload["correlated_instruments"] == ["EURUSD"]

    @pytest.mark.asyncio
    async def test_approaching_event_not_published_twice(self) -> None:
        """Same event is not published twice."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        calendar.add_event(event)

        await calendar._check_approaching_events()
        await calendar._check_approaching_events()

        assert mock_bus.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_low_impact_event_not_published(self) -> None:
        """Low-impact events are not published as approaching."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event = make_event(minutes_ahead=10, impact="LOW", instruments=["EURUSD"])
        calendar.add_event(event)

        await calendar._check_approaching_events()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_far_event_not_published(self) -> None:
        """Events beyond 15 minutes are not published."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event = make_event(minutes_ahead=20, instruments=["EURUSD"])
        calendar.add_event(event)

        await calendar._check_approaching_events()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_event_bus_no_error(self) -> None:
        """No error when event bus is not configured."""
        calendar = EconomicCalendar(event_bus=None)

        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        calendar.add_event(event)

        # Should not raise
        await calendar._check_approaching_events()
        assert event.notified is True

    @pytest.mark.asyncio
    async def test_event_bus_publish_error_handled(self) -> None:
        """Errors during event bus publish are handled gracefully."""
        mock_bus = AsyncMock()
        mock_bus.publish.side_effect = RuntimeError("Redis down")
        calendar = EconomicCalendar(event_bus=mock_bus)

        event = make_event(minutes_ahead=10, instruments=["EURUSD"])
        calendar.add_event(event)

        # Should not raise
        await calendar._check_approaching_events()


# ---------------------------------------------------------------------------
# EconomicCalendar - Lifecycle Tests
# ---------------------------------------------------------------------------


class TestEconomicCalendarLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_performs_initial_fetch(self) -> None:
        """Starting the calendar performs an initial daily update."""
        today = datetime.now(timezone.utc)
        provider = StaticScheduleProvider(events=[
            {
                "id": "1",
                "event_name": "NFP",
                "event_type": "NFP",
                "scheduled_at": today.replace(hour=13, minute=30),
                "impact_level": "HIGH",
            },
        ])
        calendar = EconomicCalendar(event_source=provider)

        await calendar.start()
        try:
            assert calendar.is_running is True
            assert len(calendar.events) == 1
            assert calendar.last_update is not None
        finally:
            await calendar.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self) -> None:
        """Stopping the calendar cleans up tasks."""
        calendar = EconomicCalendar()
        await calendar.start()
        await calendar.stop()

        assert calendar.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        """Starting twice does not create duplicate tasks."""
        calendar = EconomicCalendar()
        await calendar.start()
        await calendar.start()  # Should be no-op

        assert calendar.is_running is True
        await calendar.stop()


# ---------------------------------------------------------------------------
# Tests: get_news_factor (Task 38.4)
# ---------------------------------------------------------------------------


class TestGetNewsFactor:
    """Tests for get_news_factor method.

    Validates: Requirement 23.4, Cross-Cutting Rule 1

    get_news_factor returns 0.5 when a high-impact economic event is within
    15 minutes for a correlated instrument, and 1.0 otherwise.
    """

    def test_returns_0_5_when_high_impact_event_within_15_minutes(self) -> None:
        """Should return 0.5 when high-impact event is within 15 min."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-test",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD", "GBPUSD"],
        )
        calendar.add_event(event)
        factor = calendar.get_news_factor("EURUSD")
        assert factor == 0.5

    def test_returns_1_0_when_no_event_near(self) -> None:
        """Should return 1.0 when no high-impact event is near."""
        calendar = EconomicCalendar()
        factor = calendar.get_news_factor("EURUSD")
        assert factor == 1.0

    def test_returns_1_0_for_uncorrelated_instrument(self) -> None:
        """Should return 1.0 for instruments not correlated with the event."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-test",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD", "GBPUSD"],
        )
        calendar.add_event(event)
        factor = calendar.get_news_factor("AUDJPY")
        assert factor == 1.0

    def test_returns_1_0_for_low_impact_event(self) -> None:
        """Should return 1.0 for low-impact events."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="minor-test",
            event_name="Minor Report",
            event_type="earnings",
            scheduled_at=event_time,
            impact_level=EventImpact.LOW.value,
            correlated_instruments=["SPX500"],
        )
        calendar.add_event(event)
        factor = calendar.get_news_factor("SPX500")
        assert factor == 1.0

    def test_returns_1_0_when_event_outside_window(self) -> None:
        """Should return 1.0 when event is more than 15 minutes away."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=20)
        event = EconomicEventData(
            id="far-event",
            event_name="GDP Release",
            event_type="GDP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        factor = calendar.get_news_factor("EURUSD")
        assert factor == 1.0

    def test_returns_1_0_for_past_event(self) -> None:
        """Should return 1.0 for events that have already passed."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        event = EconomicEventData(
            id="past-event",
            event_name="Past CPI",
            event_type="CPI",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        factor = calendar.get_news_factor("EURUSD")
        assert factor == 1.0

    def test_with_explicit_current_time(self) -> None:
        """Should work correctly with explicit current_time parameter."""
        calendar = EconomicCalendar()
        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="timed-event",
            event_name="Rate Decision",
            event_type="RATE_DECISION",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["USDJPY"],
        )
        calendar.add_event(event)

        # 10 minutes before event — should return 0.5
        check_time = event_time - timedelta(minutes=10)
        assert calendar.get_news_factor("USDJPY", current_time=check_time) == 0.5

        # 20 minutes before event — should return 1.0
        check_time = event_time - timedelta(minutes=20)
        assert calendar.get_news_factor("USDJPY", current_time=check_time) == 1.0

    def test_consistent_with_get_size_reduction_factor(self) -> None:
        """get_news_factor should return the same value as get_size_reduction_factor."""
        calendar = EconomicCalendar()
        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="consistency-test",
            event_name="NFP",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        check_time = event_time - timedelta(minutes=10)
        assert calendar.get_news_factor("EURUSD", current_time=check_time) == \
            calendar.get_size_reduction_factor("EURUSD", current_time=check_time)


# ---------------------------------------------------------------------------
# Tests: get_news_reduction_factor (Task 38.4)
# ---------------------------------------------------------------------------


class TestGetNewsReductionFactor:
    """Tests for get_news_reduction_factor method.

    Validates: Requirement 23.4, Cross-Cutting Rule 1

    get_news_reduction_factor returns a ReductionFactor with factor=0.5
    when a high-impact event is near, or None when no reduction is needed.
    The ReductionFactor integrates with the position sizer's multiplicative
    stacking.
    """

    def test_returns_reduction_factor_when_event_near(self) -> None:
        """Should return a ReductionFactor with factor=0.5 when event is near."""
        from src.risk.position_sizer import ReductionFactor

        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-rf",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        rf = calendar.get_news_reduction_factor("EURUSD")

        assert rf is not None
        assert rf.source == "news"
        assert rf.factor == 0.5
        assert "15 minutes" in rf.reason
        assert "EURUSD" in rf.reason
        assert "50%" in rf.reason

    def test_returns_none_when_no_event_near(self) -> None:
        """Should return None when no high-impact event is near."""
        calendar = EconomicCalendar()
        rf = calendar.get_news_reduction_factor("EURUSD")
        assert rf is None

    def test_returns_none_for_uncorrelated_instrument(self) -> None:
        """Should return None for instruments not correlated with the event."""
        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-uncorr",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        rf = calendar.get_news_reduction_factor("AUDJPY")
        assert rf is None

    def test_reduction_factor_integrates_with_position_sizer(self) -> None:
        """ReductionFactor should be usable by the PositionSizer."""
        from decimal import Decimal
        from src.risk.position_sizer import PositionSizer

        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-sizer",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        rf = calendar.get_news_reduction_factor("EURUSD")
        assert rf is not None

        sizer = PositionSizer()
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
            reduction_factors=[rf],
        )

        # Base: (100000 * 0.01) / (50 * 1.5) = 13.33
        # After news factor (0.5): 6.66
        assert not result.rejected
        assert result.size == Decimal("6.66")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "news"
        assert result.applied_reductions[0].factor == 0.5

    def test_multiplicative_stacking_with_other_factors(self) -> None:
        """News factor should stack multiplicatively with other reduction factors."""
        from decimal import Decimal
        from src.risk.position_sizer import PositionSizer, ReductionFactor

        calendar = EconomicCalendar()
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        event = EconomicEventData(
            id="nfp-stack",
            event_name="Non-Farm Payrolls",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)
        news_rf = calendar.get_news_reduction_factor("EURUSD")
        assert news_rf is not None

        # Combine with drawdown and mistake factors
        drawdown_rf = ReductionFactor(
            source="drawdown", factor=0.25, reason="Drawdown > 10%"
        )
        mistake_rf = ReductionFactor(
            source="mistake", factor=0.7, reason="Active mistake pattern"
        )

        sizer = PositionSizer()
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
            reduction_factors=[drawdown_rf, mistake_rf, news_rf],
        )

        # Base: 13.333...
        # After drawdown (0.25) × mistake (0.7) × news (0.5) = 0.0875
        # Final: 13.333... * 0.0875 = 1.1666...
        # Quantized: 1.16
        assert not result.rejected
        assert result.size == Decimal("1.16")
        assert len(result.applied_reductions) == 3

    def test_with_explicit_current_time(self) -> None:
        """Should work correctly with explicit current_time parameter."""
        calendar = EconomicCalendar()
        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="timed-rf",
            event_name="Rate Decision",
            event_type="RATE_DECISION",
            scheduled_at=event_time,
            impact_level=EventImpact.HIGH.value,
            correlated_instruments=["USDJPY"],
        )
        calendar.add_event(event)

        # 10 minutes before event — should return ReductionFactor
        check_time = event_time - timedelta(minutes=10)
        rf = calendar.get_news_reduction_factor("USDJPY", current_time=check_time)
        assert rf is not None
        assert rf.factor == 0.5

        # 20 minutes before event — should return None
        check_time = event_time - timedelta(minutes=20)
        rf = calendar.get_news_reduction_factor("USDJPY", current_time=check_time)
        assert rf is None


# ---------------------------------------------------------------------------
# Tests: Pre-Event Risk Adjustment - Stop Widening (Task 38.2)
# ---------------------------------------------------------------------------


class TestPreEventStopWidening:
    """Tests for stop loss widening by 1.0 × ATR during pre-event adjustments.

    Validates: Requirement 23.4
    """

    @pytest.mark.asyncio
    async def test_widen_stop_callback_called_for_correlated_instruments(self) -> None:
        """Widen stop callback should be called for each correlated instrument."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="nfp-widen",
            event_name="NFP",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD", "GBPUSD"],
        )
        calendar.add_event(event)

        widen_mock = AsyncMock(return_value={
            "original_stop": Decimal("1.0800"),
            "new_stop": Decimal("1.0750"),
            "atr_used": Decimal("0.0050"),
        })
        calendar.set_position_callbacks(widen_stop=widen_mock)

        check_time = event_time - timedelta(minutes=10)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )

        assert widen_mock.call_count == 2
        widen_mock.assert_any_call("EURUSD")
        widen_mock.assert_any_call("GBPUSD")

    @pytest.mark.asyncio
    async def test_adjustment_records_stop_widening_details(self) -> None:
        """Adjustment records should capture stop widening details."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="cpi-widen",
            event_name="CPI Release",
            event_type="CPI",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        widen_mock = AsyncMock(return_value={
            "original_stop": Decimal("1.0800"),
            "new_stop": Decimal("1.0750"),
            "atr_used": Decimal("0.0050"),
        })
        calendar.set_position_callbacks(widen_stop=widen_mock)

        check_time = event_time - timedelta(minutes=10)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )

        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj.stop_widened is True
        assert adj.original_stop == Decimal("1.0800")
        assert adj.new_stop == Decimal("1.0750")
        assert adj.atr_used == Decimal("0.0050")
        assert adj.position_size_reduced is True

    @pytest.mark.asyncio
    async def test_no_widen_when_callback_returns_none(self) -> None:
        """When widen callback returns None (no position), stop_widened is False."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="no-pos-event",
            event_name="GDP",
            event_type="GDP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        widen_mock = AsyncMock(return_value=None)
        calendar.set_position_callbacks(widen_stop=widen_mock)

        check_time = event_time - timedelta(minutes=10)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )

        assert len(adjustments) == 1
        assert adjustments[0].stop_widened is False
        assert adjustments[0].position_size_reduced is True

    @pytest.mark.asyncio
    async def test_widen_callback_error_handled_gracefully(self) -> None:
        """Errors in widen callback should be handled without crashing."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="error-event",
            event_name="Rate Decision",
            event_type="RATE_DECISION",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        widen_mock = AsyncMock(side_effect=RuntimeError("Connection lost"))
        calendar.set_position_callbacks(widen_stop=widen_mock)

        check_time = event_time - timedelta(minutes=10)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )

        assert len(adjustments) == 1
        assert adjustments[0].stop_widened is False
        assert adjustments[0].position_size_reduced is True

    @pytest.mark.asyncio
    async def test_adjustment_applied_only_once_per_event(self) -> None:
        """Same event should not trigger adjustments twice."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="dedup-event",
            event_name="NFP",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        check_time = event_time - timedelta(minutes=10)

        adjustments1 = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )
        assert len(adjustments1) == 1

        adjustments2 = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )
        assert len(adjustments2) == 0

    @pytest.mark.asyncio
    async def test_event_bus_publishes_adjustment_event(self) -> None:
        """Event Bus should receive pre-event adjustment event."""
        mock_bus = AsyncMock()
        mock_bus.publish = AsyncMock(return_value=1)
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="pub-adj-event",
            event_name="NFP",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        check_time = event_time - timedelta(minutes=10)
        await calendar._check_and_apply_adjustments(current_time=check_time)

        # Verify publish was called
        assert mock_bus.publish.called
        # Find the adjustment publish call
        found = False
        for call in mock_bus.publish.call_args_list:
            channel = call[0][0]
            bus_event = call[0][1]
            if bus_event.payload.get("notification_type") == "adjustment_applied":
                found = True
                assert channel == "news.economic_event_approaching"
                assert bus_event.payload["event_id"] == "pub-adj-event"
                assert bus_event.payload["position_size_reduction"] == 0.5
                assert bus_event.payload["stop_widen_multiplier"] == 1.0
                assert "EURUSD" in bus_event.payload["correlated_instruments"]
                break
        assert found, "Pre-event adjustment event not published to Event Bus"

    @pytest.mark.asyncio
    async def test_adjustment_history_tracked(self) -> None:
        """All adjustments should be recorded in adjustment_history."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="hist-event",
            event_name="NFP",
            event_type="NFP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD", "GBPUSD", "US30"],
        )
        calendar.add_event(event)

        check_time = event_time - timedelta(minutes=10)
        await calendar._check_and_apply_adjustments(current_time=check_time)

        history = calendar.adjustment_history
        assert len(history) == 3
        instruments = {adj.instrument for adj in history}
        assert instruments == {"EURUSD", "GBPUSD", "US30"}

    @pytest.mark.asyncio
    async def test_no_adjustments_for_low_impact_events(self) -> None:
        """Low-impact events should not trigger pre-event adjustments."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="low-impact",
            event_name="Minor Report",
            event_type="earnings",
            scheduled_at=event_time,
            impact_level="LOW",
            correlated_instruments=["SPX500"],
        )
        calendar.add_event(event)

        check_time = event_time - timedelta(minutes=10)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )
        assert len(adjustments) == 0

    @pytest.mark.asyncio
    async def test_no_adjustments_outside_15_minute_window(self) -> None:
        """Events more than 15 minutes away should not trigger adjustments."""
        mock_bus = AsyncMock()
        calendar = EconomicCalendar(event_bus=mock_bus)

        event_time = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        event = EconomicEventData(
            id="far-event",
            event_name="GDP",
            event_type="GDP",
            scheduled_at=event_time,
            impact_level="HIGH",
            correlated_instruments=["EURUSD"],
        )
        calendar.add_event(event)

        check_time = event_time - timedelta(minutes=20)
        adjustments = await calendar._check_and_apply_adjustments(
            current_time=check_time
        )
        assert len(adjustments) == 0

    @pytest.mark.asyncio
    async def test_monitoring_loop_start_stop(self) -> None:
        """Monitoring loop can be started and stopped."""
        calendar = EconomicCalendar()
        await calendar.start_monitoring()
        assert calendar._running is True
        assert calendar._monitor_task is not None

        await calendar.stop_monitoring()
        assert calendar._running is False
        assert calendar._monitor_task is None
