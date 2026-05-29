"""Economic calendar for scheduled event tracking.

Manages upcoming economic events and provides risk-reduction signals
for correlated instruments around high-impact scheduled events.

Fetches economic events daily at 00:00 UTC (or on-demand) and stores
high-impact events: NFP, CPI, rate decisions, GDP, central bank speeches.

Publishes NEWS_ECONOMIC_EVENT approaching events to Event Bus 15 minutes
before their scheduled time.

Validates: Requirements 23.3, 23.4
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from src.config.constants import (
    NEWS_EVENT_SIZE_REDUCTION_FACTOR,
    NEWS_PRE_EVENT_RISK_WINDOW_MINUTES,
    NEWS_PRE_EVENT_SIGNAL_PAUSE_MINUTES,
)
from src.core.event_bus import (
    Event as BusEvent,
    EventBus,
    NEWS_ECONOMIC_EVENT,
)
from src.risk.position_sizer import ReductionFactor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class EventImpact(str, Enum):
    """Impact level for economic events."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class HighImpactEventType(str, Enum):
    """Types of high-impact economic events tracked by the calendar."""

    NFP = "NFP"
    CPI = "CPI"
    RATE_DECISION = "RATE_DECISION"
    GDP = "GDP"
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"


# Default high-impact event types to track
HIGH_IMPACT_EVENT_TYPES: set[str] = {
    HighImpactEventType.NFP.value,
    HighImpactEventType.CPI.value,
    HighImpactEventType.RATE_DECISION.value,
    HighImpactEventType.GDP.value,
    HighImpactEventType.CENTRAL_BANK_SPEECH.value,
}

# Pre-event notification window in minutes
PRE_EVENT_NOTIFICATION_MINUTES: int = 15


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class EconomicEventData:
    """Scheduled economic event with impact and correlation data.

    Attributes:
        id: Unique event identifier.
        event_name: Human-readable event name.
        event_type: Category of event (NFP, CPI, RATE_DECISION, GDP, etc.).
        scheduled_at: UTC datetime when the event is scheduled.
        impact_level: Impact level (HIGH, MEDIUM, LOW).
        currency_region: Currency or region affected (e.g., "USD", "EUR").
        correlated_instruments: List of instruments affected by this event.
        actual_value: Actual released value (populated after event).
        forecast_value: Market consensus forecast value.
        previous_value: Previous period's value.
        notified: Whether the 15-min pre-event notification has been sent.
    """

    id: str
    event_name: str
    event_type: str
    scheduled_at: datetime
    impact_level: str = EventImpact.HIGH.value
    currency_region: str = ""
    correlated_instruments: list[str] = field(default_factory=list)
    actual_value: str | None = None
    forecast_value: str | None = None
    previous_value: str | None = None
    notified: bool = False

    @property
    def scheduled_time(self) -> datetime:
        """Alias for scheduled_at."""
        return self.scheduled_at

    @property
    def expected_impact(self) -> str:
        """Alias for impact_level."""
        return self.impact_level


@dataclass
class PreEventAdjustment:
    """Record of a pre-event risk adjustment applied to a position."""

    event_id: str
    event_name: str
    instrument: str
    position_size_reduced: bool = False
    stop_widened: bool = False
    original_stop: Decimal | None = None
    new_stop: Decimal | None = None
    atr_used: Decimal | None = None
    adjusted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Event Source Protocol and Providers
# ---------------------------------------------------------------------------


class EventSourceProvider(Protocol):
    """Protocol for economic calendar data providers."""

    async def fetch_events(self, date: datetime) -> list[dict[str, Any]]:
        """Fetch economic events for a given date."""
        ...


class StaticScheduleProvider:
    """Provides events from a pre-configured static schedule."""

    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self._events: list[dict[str, Any]] = events or []

    def add_events(self, events: list[dict[str, Any]]) -> None:
        """Add events to the static schedule."""
        self._events.extend(events)

    def clear(self) -> None:
        """Clear all events from the static schedule."""
        self._events.clear()

    async def fetch_events(self, date: datetime) -> list[dict[str, Any]]:
        """Return events matching the given date."""
        target_date = date.date() if isinstance(date, datetime) else date
        return [
            event for event in self._events
            if self._event_matches_date(event, target_date)
        ]

    @staticmethod
    def _event_matches_date(event: dict[str, Any], target_date: Any) -> bool:
        """Check if an event is scheduled on the target date."""
        scheduled = event.get("scheduled_at")
        if scheduled is None:
            return False
        if isinstance(scheduled, datetime):
            return scheduled.date() == target_date
        return False


class APIEndpointProvider:
    """Fetches economic events from a configurable API endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._headers = headers or {}

    async def fetch_events(self, date: datetime) -> list[dict[str, Any]]:
        """Fetch events from the configured API endpoint."""
        try:
            import httpx

            headers = dict(self._headers)
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            params = {"date": date.strftime("%Y-%m-%d")}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self._endpoint_url,
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "events" in data:
                    return data["events"]
                return []

        except Exception as exc:
            logger.error(
                "Failed to fetch economic calendar from API",
                extra={
                    "endpoint": self._endpoint_url,
                    "date": date.isoformat(),
                    "error": str(exc),
                },
            )
            return []


# ---------------------------------------------------------------------------
# Economic Calendar
# ---------------------------------------------------------------------------


class EconomicCalendar:
    """Manages scheduled economic events and pre-event risk adjustments.

    Key behaviors:
    - Fetches events daily at 00:00 UTC from configurable source
    - Stores high-impact events: NFP, CPI, rate decisions, GDP, central bank speeches
    - Publishes NEWS_ECONOMIC_EVENT to Event Bus 15 minutes before events
    - Pause signals 5 minutes before high-impact events for correlated instruments
    - Reduce position size by 50% within 15 minutes of high-impact events
    - Track instrument-event correlations for targeted risk management
    """

    def __init__(
        self,
        event_bus: EventBus | Any | None = None,
        event_source: EventSourceProvider | None = None,
        check_interval_seconds: int = 30,
        high_impact_types: set[str] | None = None,
    ) -> None:
        self._events: list[EconomicEventData] = []
        self._last_update: datetime | None = None
        self._update_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._event_bus = event_bus
        self._event_source = event_source
        self._check_interval_seconds = check_interval_seconds
        self._high_impact_types = high_impact_types or HIGH_IMPACT_EVENT_TYPES
        self._running = False
        self._adjusted_event_ids: set[str] = set()
        self._notified_event_ids: set[str] = set()
        self._adjustment_history: list[PreEventAdjustment] = []
        self._widen_stop_callback: Any | None = None

    # Properties

    @property
    def events(self) -> list[EconomicEventData]:
        """All currently tracked economic events."""
        return list(self._events)

    @property
    def last_update(self) -> datetime | None:
        """Timestamp of the last daily update."""
        return self._last_update

    @property
    def is_running(self) -> bool:
        """Whether the calendar monitoring loop is active."""
        return self._running

    @property
    def adjustment_history(self) -> list[PreEventAdjustment]:
        """All pre-event adjustments that have been applied."""
        return list(self._adjustment_history)

    @property
    def adjusted_event_ids(self) -> set[str]:
        """IDs of events that have already had adjustments applied."""
        return set(self._adjusted_event_ids)

    # Lifecycle

    async def start(self) -> None:
        """Start the economic calendar with daily updates and event monitoring."""
        if self._running:
            return
        self._running = True
        await self.update_daily()
        self._update_task = asyncio.create_task(self._daily_update_loop())
        self._monitor_task = asyncio.create_task(self._event_monitoring_loop())
        logger.info("EconomicCalendar started", extra={"event_count": len(self._events)})

    async def stop(self) -> None:
        """Stop the calendar monitoring and update tasks."""
        self._running = False
        for task in (self._update_task, self._monitor_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._update_task = None
        self._monitor_task = None
        logger.info("EconomicCalendar stopped")

    async def start_monitoring(self) -> None:
        """Start the event monitoring loop (alias for start)."""
        await self.start()

    async def stop_monitoring(self) -> None:
        """Stop the event monitoring loop (alias for stop)."""
        await self.stop()

    def set_position_callbacks(
        self, get_positions: Any | None = None, widen_stop: Any | None = None
    ) -> None:
        """Set callbacks for position management integration."""
        self._widen_stop_callback = widen_stop

    # Daily Update

    async def update_daily(self) -> None:
        """Fetch economic calendar events for the current day (00:00 UTC)."""
        now = datetime.now(timezone.utc)
        if self._event_source is not None:
            try:
                raw_events = await self._event_source.fetch_events(now)
                new_events: list[EconomicEventData] = []
                for raw in raw_events:
                    event = self._parse_event(raw)
                    if event is not None:
                        new_events.append(event)
                self._events = new_events
                logger.info("Economic calendar updated", extra={"total_events": len(new_events)})
            except Exception as exc:
                logger.error("Failed to update economic calendar", extra={"error": str(exc)})
        self._last_update = now
        self._adjusted_event_ids.clear()
        self._notified_event_ids.clear()

    def _parse_event(self, raw: dict[str, Any]) -> EconomicEventData | None:
        """Parse a raw event dictionary into an EconomicEventData instance."""
        try:
            scheduled_at = raw.get("scheduled_at")
            if isinstance(scheduled_at, str):
                scheduled_at = datetime.fromisoformat(scheduled_at)
            if scheduled_at is None:
                return None
            if scheduled_at.tzinfo is None:
                scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
            return EconomicEventData(
                id=raw.get("id", ""),
                event_name=raw.get("event_name", ""),
                event_type=raw.get("event_type", ""),
                scheduled_at=scheduled_at,
                impact_level=raw.get("impact_level", EventImpact.HIGH.value),
                currency_region=raw.get("currency_region", ""),
                correlated_instruments=raw.get("correlated_instruments", []),
                forecast_value=raw.get("forecast_value"),
                previous_value=raw.get("previous_value"),
            )
        except Exception as exc:
            logger.warning("Failed to parse economic event", extra={"error": str(exc)})
            return None

    # Event Management

    def add_event(self, event: EconomicEventData) -> None:
        """Add an economic event to the calendar."""
        self._events.append(event)

    def clear_events(self) -> None:
        """Clear all tracked events."""
        self._events.clear()
        self._adjusted_event_ids.clear()
        self._notified_event_ids.clear()

    # Query Methods

    def get_upcoming_events(
        self, hours_ahead: int | None = None, within_minutes: int | None = None
    ) -> list[EconomicEventData]:
        """Get events scheduled within the specified time window.

        Supports: get_upcoming_events(hours_ahead=24) or
        get_upcoming_events(within_minutes=15). Defaults to 24 hours.
        """
        now = datetime.now(timezone.utc)
        if within_minutes is not None:
            cutoff = now + timedelta(minutes=within_minutes)
        elif hours_ahead is not None:
            cutoff = now + timedelta(hours=hours_ahead)
        else:
            cutoff = now + timedelta(hours=24)
        upcoming = [e for e in self._events if now <= e.scheduled_at <= cutoff]
        return sorted(upcoming, key=lambda e: e.scheduled_at)

    def get_events_within(self, minutes: int) -> list[EconomicEventData]:
        """Get events scheduled within the specified minutes window."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes)
        within = [e for e in self._events if now <= e.scheduled_at <= cutoff]
        return sorted(within, key=lambda e: e.scheduled_at)

    def is_high_impact_event_near(self, instrument: str, minutes: int = 15) -> bool:
        """Check if a high-impact event is near for a given instrument."""
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=minutes)
        for event in self._events:
            if event.impact_level != EventImpact.HIGH.value:
                continue
            if instrument not in event.correlated_instruments:
                continue
            time_to_event = event.scheduled_at - now
            if timedelta(0) <= time_to_event <= window:
                return True
        return False

    def get_correlated_instruments(self, event: EconomicEventData) -> list[str]:
        """Get instruments correlated with a specific economic event."""
        return list(event.correlated_instruments)

    def get_high_impact_events(self) -> list[EconomicEventData]:
        """Get all high-impact events currently in the calendar."""
        return [e for e in self._events if e.impact_level == EventImpact.HIGH.value]

    # Signal Pausing and Size Reduction

    def should_pause_signals(self, instrument: str, current_time: datetime | None = None) -> bool:
        """Check if signal generation should be paused for an instrument.

        Returns True if a high-impact event is within 5 minutes.
        """
        now = current_time or datetime.now(timezone.utc)
        pause_window = timedelta(minutes=NEWS_PRE_EVENT_SIGNAL_PAUSE_MINUTES)
        for event in self._events:
            if event.impact_level != "HIGH":
                continue
            if instrument not in event.correlated_instruments:
                continue
            time_to_event = event.scheduled_at - now
            if timedelta(0) <= time_to_event <= pause_window:
                return True
        return False

    def get_size_reduction_factor(self, instrument: str, current_time: datetime | None = None) -> float:
        """Get position size reduction factor for an instrument.

        Returns 0.5 if a high-impact event is within 15 minutes, else 1.0.
        """
        now = current_time or datetime.now(timezone.utc)
        risk_window = timedelta(minutes=NEWS_PRE_EVENT_RISK_WINDOW_MINUTES)
        for event in self._events:
            if event.impact_level != "HIGH":
                continue
            if instrument not in event.correlated_instruments:
                continue
            time_to_event = event.scheduled_at - now
            if timedelta(0) <= time_to_event <= risk_window:
                return NEWS_EVENT_SIZE_REDUCTION_FACTOR
        return 1.0

    def get_news_factor(self, instrument: str, current_time: datetime | None = None) -> float:
        """Get the news factor for position sizing (0.5 or 1.0)."""
        return self.get_size_reduction_factor(instrument, current_time)

    def get_news_reduction_factor(
        self, instrument: str, current_time: datetime | None = None
    ) -> ReductionFactor | None:
        """Get a ReductionFactor for the position sizer when news reduction applies.

        Per Cross-Cutting Rule 1, applied multiplicatively with other factors.
        """
        factor = self.get_news_factor(instrument, current_time)
        if factor < 1.0:
            return ReductionFactor(
                source="news",
                factor=factor,
                reason=(
                    f"High-impact economic event within "
                    f"{NEWS_PRE_EVENT_RISK_WINDOW_MINUTES} minutes for {instrument}; "
                    f"reducing position size by {int((1 - factor) * 100)}%"
                ),
            )
        return None

    # Background Tasks

    async def _daily_update_loop(self) -> None:
        """Background task that triggers daily updates at 00:00 UTC."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                await asyncio.sleep((tomorrow - now).total_seconds())
                if self._running:
                    await self.update_daily()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Daily update loop error", extra={"error": str(exc)})
                await asyncio.sleep(60)

    async def _event_monitoring_loop(self) -> None:
        """Background loop that checks for upcoming events."""
        while self._running:
            try:
                await self._check_approaching_events()
                await self._check_and_apply_adjustments()
                await asyncio.sleep(self._check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Event monitoring loop error", extra={"error": str(exc)})
                await asyncio.sleep(self._check_interval_seconds)

    # Approaching Event Notifications

    async def _check_approaching_events(self) -> None:
        """Publish NEWS_ECONOMIC_EVENT for high-impact events within 15 minutes."""
        now = datetime.now(timezone.utc)
        notification_window = timedelta(minutes=PRE_EVENT_NOTIFICATION_MINUTES)
        for event in self._events:
            if event.id in self._notified_event_ids or event.notified:
                continue
            if event.impact_level != EventImpact.HIGH.value:
                continue
            time_to_event = event.scheduled_at - now
            if timedelta(0) <= time_to_event <= notification_window:
                await self._publish_approaching_event(event, now)
                event.notified = True
                self._notified_event_ids.add(event.id)

    async def _publish_approaching_event(self, event: EconomicEventData, now: datetime) -> None:
        """Publish a NEWS_ECONOMIC_EVENT notification to the Event Bus."""
        if self._event_bus is None:
            return
        minutes_until = (event.scheduled_at - now).total_seconds() / 60.0
        bus_event = BusEvent(
            event_type=NEWS_ECONOMIC_EVENT,
            payload={
                "event_id": event.id,
                "event_name": event.event_name,
                "event_type": event.event_type,
                "scheduled_at": event.scheduled_at.isoformat(),
                "currency_region": event.currency_region,
                "impact_level": event.impact_level,
                "correlated_instruments": event.correlated_instruments,
                "minutes_until_event": round(minutes_until, 1),
                "forecast_value": event.forecast_value,
                "previous_value": event.previous_value,
                "notification_type": "approaching",
            },
        )
        try:
            await self._event_bus.publish(NEWS_ECONOMIC_EVENT, bus_event)
            logger.info("Published approaching economic event", extra={"event_id": event.id})
        except Exception as exc:
            logger.error("Failed to publish event notification", extra={"error": str(exc)})

    # Pre-Event Risk Adjustments

    async def _check_and_apply_adjustments(self, current_time: datetime | None = None) -> list[PreEventAdjustment]:
        """Apply pre-event risk adjustments for upcoming high-impact events."""
        now = current_time or datetime.now(timezone.utc)
        risk_window = timedelta(minutes=NEWS_PRE_EVENT_RISK_WINDOW_MINUTES)
        upcoming = [
            e for e in self._events
            if e.impact_level == "HIGH" and timedelta(0) <= (e.scheduled_at - now) <= risk_window
        ]
        adjustments: list[PreEventAdjustment] = []
        for event in upcoming:
            if event.id in self._adjusted_event_ids:
                continue
            correlated = self.get_correlated_instruments(event)
            if not correlated:
                continue
            event_adjustments = await self._apply_pre_event_adjustments(event, correlated, now)
            adjustments.extend(event_adjustments)
            self._adjusted_event_ids.add(event.id)
            await self._publish_pre_event_adjustment(event, event_adjustments, now)
        return adjustments

    async def _apply_pre_event_adjustments(
        self, event: EconomicEventData, correlated_instruments: list[str], now: datetime
    ) -> list[PreEventAdjustment]:
        """Apply pre-event risk adjustments for correlated instruments."""
        adjustments: list[PreEventAdjustment] = []
        for instrument in correlated_instruments:
            adjustment = PreEventAdjustment(
                event_id=event.id, event_name=event.event_name,
                instrument=instrument, position_size_reduced=True, adjusted_at=now,
            )
            if self._widen_stop_callback is not None:
                try:
                    result = await self._widen_stop_callback(instrument)
                    if result is not None:
                        adjustment.stop_widened = True
                        adjustment.original_stop = result.get("original_stop")
                        adjustment.new_stop = result.get("new_stop")
                        adjustment.atr_used = result.get("atr_used")
                except Exception as exc:
                    logger.error("Failed to widen stop", extra={"instrument": instrument, "error": str(exc)})
            adjustments.append(adjustment)
            self._adjustment_history.append(adjustment)
        return adjustments

    async def _publish_pre_event_adjustment(
        self, event: EconomicEventData, adjustments: list[PreEventAdjustment], now: datetime
    ) -> None:
        """Publish pre-event adjustment event to the Event Bus."""
        if self._event_bus is None:
            return
        bus_event = BusEvent(
            event_type=NEWS_ECONOMIC_EVENT,
            payload={
                "event_id": event.id,
                "event_name": event.event_name,
                "event_type": event.event_type,
                "scheduled_at": event.scheduled_at.isoformat(),
                "impact_level": event.impact_level,
                "currency_region": event.currency_region,
                "notification_type": "adjustment_applied",
                "position_size_reduction": NEWS_EVENT_SIZE_REDUCTION_FACTOR,
                "stop_widen_multiplier": 1.0,
                "correlated_instruments": [adj.instrument for adj in adjustments],
                "applied_at": now.isoformat(),
            },
        )
        try:
            await self._event_bus.publish(NEWS_ECONOMIC_EVENT, bus_event)
        except Exception as exc:
            logger.error("Failed to publish adjustment event", extra={"error": str(exc)})

    def is_event_adjusted(self, event_id: str) -> bool:
        """Check if an event has already had pre-event adjustments applied."""
        return event_id in self._adjusted_event_ids
