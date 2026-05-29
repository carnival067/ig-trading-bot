"""IG Lightstreamer market data streaming client.

Provides real-time price streaming via WebSocket connection to the IG
Lightstreamer service. Supports multi-instrument subscriptions (50+),
tick processing with Event Bus distribution within 50ms target,
auto-reconnect with exponential backoff, and staleness detection.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, Cross-Cutting Rule 5
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import websockets
import websockets.exceptions

from src.config.constants import (
    RECONNECT_MAX_ATTEMPTS,
    TICK_STALENESS_SECONDS,
)
from src.core.event_bus import MARKET_TICK, Event, EventBus
from src.core.exceptions import StreamDisconnectedError
from src.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECONNECT_BASE_DELAY_SECONDS: float = 1.0
"""Base delay in seconds for exponential backoff on stream reconnection."""

RECONNECT_MAX_TOTAL_SECONDS: float = 30.0
"""Maximum total time in seconds allowed for reconnection attempts."""

STALENESS_CHECK_INTERVAL_SECONDS: float = 10.0
"""Interval in seconds between staleness checks."""

SUBSCRIPTION_RETRY_MAX_ATTEMPTS: int = 3
"""Maximum retry attempts for a subscription request."""

SUBSCRIPTION_RETRY_DELAY_SECONDS: float = 2.0
"""Delay in seconds between subscription retry attempts."""

MAX_SIMULTANEOUS_INSTRUMENTS: int = 50
"""Minimum number of simultaneous instrument subscriptions supported."""


# ---------------------------------------------------------------------------
# Subscription State
# ---------------------------------------------------------------------------


class SubscriptionStatus(str, Enum):
    """Status of an instrument subscription."""

    ACTIVE = "active"
    STALE = "stale"
    SUBSCRIBING = "subscribing"
    UNSUBSCRIBED = "unsubscribed"
    ERROR = "error"


@dataclass
class SubscriptionState:
    """Tracks the state of a single instrument subscription.

    Attributes:
        epic: The IG instrument identifier.
        status: Current subscription status.
        last_tick_time: Timestamp of the last received tick.
        subscribe_time: Timestamp when the subscription was created.
        error_count: Number of consecutive errors for this subscription.
    """

    epic: str
    status: SubscriptionStatus = SubscriptionStatus.SUBSCRIBING
    last_tick_time: datetime | None = None
    subscribe_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0


# ---------------------------------------------------------------------------
# IGStream
# ---------------------------------------------------------------------------


class IGStream:
    """Lightstreamer WebSocket client for real-time IG price streaming.

    Connects to the IG Lightstreamer service and streams bid/ask/timestamp
    ticks for subscribed instruments. Publishes ticks to the Event Bus
    within a 50ms processing target.

    Features:
        - Multi-instrument subscription (50+ simultaneous instruments)
        - Tick processing and distribution to Event Bus within 50ms
        - Auto-reconnect with exponential backoff (base 1s, max 5 attempts)
        - Missed data recovery via REST API on reconnection
        - Staleness detection (60s without tick during market hours)

    Usage::

        stream = IGStream(
            stream_url="wss://push.lightstreamer.com/lightstreamer",
            cst="my-cst-token",
            security_token="my-security-token",
            event_bus=event_bus,
            ig_client=ig_client,
        )
        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        # ... ticks are published to event bus ...
        await stream.stop()
    """

    def __init__(
        self,
        stream_url: str,
        cst: str,
        security_token: str,
        event_bus: EventBus,
        ig_client: Any | None = None,
    ) -> None:
        """Initialize the IGStream client.

        Args:
            stream_url: Lightstreamer WebSocket endpoint URL.
            cst: IG CST authentication token.
            security_token: IG X-SECURITY-TOKEN.
            event_bus: Event bus instance for publishing ticks.
            ig_client: Optional IG REST client for missed data recovery.
        """
        self._stream_url = stream_url
        self._cst = cst
        self._security_token = security_token
        self._event_bus = event_bus
        self._ig_client = ig_client

        # Subscription management
        self._subscriptions: dict[str, SubscriptionState] = {}
        self._last_tick_times: dict[str, datetime] = {}

        # Connection state
        self._connected: bool = False
        self._reconnect_attempts: int = 0
        self._ws: Any | None = None

        # Background tasks
        self._listener_task: asyncio.Task[None] | None = None
        self._staleness_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Disconnect tracking for missed data recovery
        self._disconnect_time: datetime | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Lightstreamer and start the listener and staleness monitor.

        Raises:
            StreamDisconnectedError: If initial connection fails after retries.
        """
        if self._running:
            return

        self._running = True
        await self._connect()

        self._listener_task = asyncio.create_task(
            self._listen(), name="ig_stream_listener"
        )
        self._staleness_task = asyncio.create_task(
            self._staleness_monitor(), name="ig_stream_staleness"
        )

        logger.info(
            "IGStream started",
            extra={"stream_url": self._stream_url},
        )

    async def stop(self) -> None:
        """Disconnect from Lightstreamer and clean up all resources."""
        self._running = False

        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._staleness_task is not None:
            self._staleness_task.cancel()
            try:
                await self._staleness_task
            except asyncio.CancelledError:
                pass
            self._staleness_task = None

        await self._disconnect()

        self._subscriptions.clear()
        self._last_tick_times.clear()

        logger.info("IGStream stopped")

    # -------------------------------------------------------------------------
    # Subscription Management
    # -------------------------------------------------------------------------

    async def subscribe(self, epic: str) -> None:
        """Subscribe to price updates for an instrument.

        Retries up to 3 times on failure with 2-second intervals.

        Args:
            epic: The IG instrument identifier (e.g., "CS.D.EURUSD.CFD.IP").

        Raises:
            StreamDisconnectedError: If not connected and cannot subscribe.
        """
        if epic in self._subscriptions:
            existing = self._subscriptions[epic]
            if existing.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.SUBSCRIBING):
                logger.debug(
                    "Already subscribed to instrument",
                    extra={"epic": epic, "status": existing.status},
                )
                return

        self._subscriptions[epic] = SubscriptionState(
            epic=epic, status=SubscriptionStatus.SUBSCRIBING
        )

        for attempt in range(SUBSCRIPTION_RETRY_MAX_ATTEMPTS):
            try:
                await self._send_subscription(epic)
                self._subscriptions[epic].status = SubscriptionStatus.ACTIVE
                logger.info(
                    "Subscribed to instrument",
                    extra={"epic": epic, "attempt": attempt + 1},
                )
                return
            except Exception as exc:
                logger.warning(
                    "Subscription attempt failed",
                    extra={
                        "epic": epic,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )
                if attempt < SUBSCRIPTION_RETRY_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(SUBSCRIPTION_RETRY_DELAY_SECONDS)

        # All retries exhausted
        self._subscriptions[epic].status = SubscriptionStatus.ERROR
        self._subscriptions[epic].error_count += 1
        logger.error(
            "Subscription failed after all retries",
            extra={"epic": epic, "max_attempts": SUBSCRIPTION_RETRY_MAX_ATTEMPTS},
        )

    async def unsubscribe(self, epic: str) -> None:
        """Unsubscribe from price updates for an instrument.

        Args:
            epic: The IG instrument identifier to unsubscribe from.
        """
        if epic not in self._subscriptions:
            return

        try:
            await self._send_unsubscription(epic)
        except Exception as exc:
            logger.warning(
                "Unsubscription request failed",
                extra={"epic": epic, "error": str(exc)},
            )

        self._subscriptions[epic].status = SubscriptionStatus.UNSUBSCRIBED
        self._last_tick_times.pop(epic, None)
        del self._subscriptions[epic]

        logger.info("Unsubscribed from instrument", extra={"epic": epic})

    # -------------------------------------------------------------------------
    # Tick Processing
    # -------------------------------------------------------------------------

    async def _on_tick(self, epic: str, data: dict[str, Any]) -> None:
        """Process a tick and publish to Event Bus within 50ms target.

        Updates last_tick_time for staleness tracking. If the instrument was
        previously marked stale, restores it to active status.

        Args:
            epic: The instrument identifier.
            data: Tick data containing bid, ask, and timestamp fields.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc)

        # Update staleness tracking
        self._last_tick_times[epic] = now

        # Restore from stale if needed
        if epic in self._subscriptions:
            if self._subscriptions[epic].status == SubscriptionStatus.STALE:
                self._subscriptions[epic].status = SubscriptionStatus.ACTIVE
                logger.info(
                    "Instrument restored from stale",
                    extra={"epic": epic},
                )
            self._subscriptions[epic].last_tick_time = now

        # Build tick event payload
        tick_payload = {
            "epic": epic,
            "bid": data.get("bid"),
            "ask": data.get("ask"),
            "timestamp": data.get("timestamp", now.isoformat()),
            "received_at": now.isoformat(),
        }

        # Publish to Event Bus on the instrument-specific channel
        channel = MARKET_TICK.format(instrument=epic)
        event = Event(
            event_type="market.tick",
            payload=tick_payload,
        )

        try:
            await self._event_bus.publish(channel, event)
        except Exception as exc:
            logger.error(
                "Failed to publish tick to event bus",
                extra={"epic": epic, "error": str(exc)},
            )

        # Log if processing exceeded 50ms target
        elapsed_ms = (time.monotonic() - start_time) * 1000
        if elapsed_ms > 50:
            logger.warning(
                "Tick processing exceeded 50ms target",
                extra={"epic": epic, "elapsed_ms": round(elapsed_ms, 2)},
            )

    # -------------------------------------------------------------------------
    # Reconnection
    # -------------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Auto-reconnect with exponential backoff.

        Base delay is 1 second, max 5 attempts within 30 seconds total.
        On success, requests missed data from REST API for the disconnection window.

        Raises:
            StreamDisconnectedError: If all reconnection attempts are exhausted.
        """
        self._disconnect_time = datetime.now(timezone.utc)
        self._connected = False
        total_elapsed = 0.0

        for attempt in range(RECONNECT_MAX_ATTEMPTS):
            if not self._running:
                return

            delay = RECONNECT_BASE_DELAY_SECONDS * (2**attempt)
            # Cap delay so total doesn't exceed 30s
            if total_elapsed + delay > RECONNECT_MAX_TOTAL_SECONDS:
                delay = max(0, RECONNECT_MAX_TOTAL_SECONDS - total_elapsed)

            if delay > 0:
                await asyncio.sleep(delay)
                total_elapsed += delay

            self._reconnect_attempts = attempt + 1

            logger.info(
                "Stream reconnection attempt",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": RECONNECT_MAX_ATTEMPTS,
                    "total_elapsed_s": round(total_elapsed, 1),
                },
            )

            try:
                await self._connect()

                # Re-subscribe to all active instruments
                for epic, state in self._subscriptions.items():
                    if state.status in (
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.STALE,
                    ):
                        try:
                            await self._send_subscription(epic)
                            state.status = SubscriptionStatus.ACTIVE
                        except Exception:
                            logger.warning(
                                "Failed to re-subscribe after reconnect",
                                extra={"epic": epic},
                            )

                # Recover missed data via REST API
                await self._recover_missed_data()

                self._reconnect_attempts = 0
                logger.info(
                    "Stream reconnection successful",
                    extra={"attempt": attempt + 1},
                )
                return

            except Exception as exc:
                logger.warning(
                    "Stream reconnection attempt failed",
                    extra={
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )

            if total_elapsed >= RECONNECT_MAX_TOTAL_SECONDS:
                break

        # All attempts exhausted
        raise StreamDisconnectedError(
            "Stream reconnection exhausted all attempts",
            max_attempts=RECONNECT_MAX_ATTEMPTS,
            total_elapsed_s=round(total_elapsed, 1),
        )

    async def _recover_missed_data(self) -> None:
        """Request missed data from REST API for the disconnection window.

        Uses the IG REST client to fetch price data for the period between
        disconnect and reconnect for all active subscriptions.
        """
        if self._ig_client is None or self._disconnect_time is None:
            return

        reconnect_time = datetime.now(timezone.utc)
        disconnect_duration = (reconnect_time - self._disconnect_time).total_seconds()

        logger.info(
            "Recovering missed data",
            extra={
                "disconnect_duration_s": round(disconnect_duration, 1),
                "instruments": len(self._subscriptions),
            },
        )

        for epic, state in self._subscriptions.items():
            if state.status not in (SubscriptionStatus.ACTIVE, SubscriptionStatus.STALE):
                continue

            try:
                # Request recent price data to fill the gap
                # Use MINUTE resolution for short disconnections
                num_points = max(1, int(disconnect_duration / 60) + 1)
                prices = await self._ig_client.get_prices(
                    epic=epic,
                    resolution="MINUTE",
                    num_points=min(num_points, 50),
                )

                # Process each recovered price as a tick
                for price in prices:
                    snapshot = price.get("snapshotTimeUTC", "")
                    close_price = price.get("closePrice", {})
                    tick_data = {
                        "bid": close_price.get("bid"),
                        "ask": close_price.get("ask"),
                        "timestamp": snapshot,
                    }
                    await self._on_tick(epic, tick_data)

            except Exception as exc:
                logger.warning(
                    "Failed to recover missed data for instrument",
                    extra={"epic": epic, "error": str(exc)},
                )

        self._disconnect_time = None

    # -------------------------------------------------------------------------
    # Staleness Detection
    # -------------------------------------------------------------------------

    async def _staleness_monitor(self) -> None:
        """Background task: check all subscriptions for staleness every 10 seconds.

        If no tick is received for 60 seconds during market hours, marks the
        instrument as stale and notifies strategies via the Event Bus.
        """
        while self._running:
            try:
                await asyncio.sleep(STALENESS_CHECK_INTERVAL_SECONDS)

                if not self._running:
                    break

                now = datetime.now(timezone.utc)

                for epic, state in list(self._subscriptions.items()):
                    if state.status not in (
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.STALE,
                    ):
                        continue

                    # Only check staleness during market hours
                    if not self.is_market_open(epic):
                        continue

                    last_tick = self._last_tick_times.get(epic)
                    if last_tick is None:
                        # Use subscribe time as baseline
                        last_tick = state.subscribe_time

                    seconds_since_tick = (now - last_tick).total_seconds()

                    if (
                        seconds_since_tick >= TICK_STALENESS_SECONDS
                        and state.status != SubscriptionStatus.STALE
                    ):
                        state.status = SubscriptionStatus.STALE
                        logger.warning(
                            "Instrument marked stale",
                            extra={
                                "epic": epic,
                                "seconds_since_tick": round(seconds_since_tick, 1),
                            },
                        )

                        # Notify strategies to suspend signals
                        await self._notify_stale(epic)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Error in staleness monitor",
                    extra={"error": str(exc)},
                )

    async def _notify_stale(self, epic: str) -> None:
        """Publish a staleness notification to the Event Bus.

        Args:
            epic: The instrument that has become stale.
        """
        try:
            channel = MARKET_TICK.format(instrument=epic)
            event = Event(
                event_type="market.stale",
                payload={
                    "epic": epic,
                    "stale": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            await self._event_bus.publish(channel, event)
        except Exception as exc:
            logger.error(
                "Failed to publish staleness notification",
                extra={"epic": epic, "error": str(exc)},
            )

    # -------------------------------------------------------------------------
    # Market Hours (Cross-Cutting Rule 5)
    # -------------------------------------------------------------------------

    def is_market_open(self, epic: str) -> bool:
        """Check if the market is currently open for the given instrument.

        Per Cross-Cutting Rule 5, market hours are defined per instrument
        based on IG's published trading hours. This is a simplified
        implementation that defaults to True and will be refined when
        full market hours are defined in task 41.5.

        Args:
            epic: The IG instrument identifier.

        Returns:
            True if the market is considered open (default behavior).
        """
        # TODO: Implement per-instrument market hours based on IG published hours
        # (task 41.5). For now, default to True to avoid false staleness alerts.
        return True

    # -------------------------------------------------------------------------
    # Status Queries
    # -------------------------------------------------------------------------

    def get_subscription_status(self) -> dict[str, str]:
        """Get the status of all current subscriptions.

        Returns:
            Dictionary mapping epic to subscription status string.
        """
        return {
            epic: state.status.value
            for epic, state in self._subscriptions.items()
        }

    def get_stale_instruments(self) -> list[str]:
        """Get a list of instruments currently marked as stale.

        Returns:
            List of epic identifiers for stale instruments.
        """
        return [
            epic
            for epic, state in self._subscriptions.items()
            if state.status == SubscriptionStatus.STALE
        ]

    @property
    def is_connected(self) -> bool:
        """Whether the stream is currently connected."""
        return self._connected

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(
            [s for s in self._subscriptions.values() if s.status == SubscriptionStatus.ACTIVE]
        )

    # -------------------------------------------------------------------------
    # WebSocket Connection (Internal)
    # -------------------------------------------------------------------------

    async def _connect(self) -> None:
        """Establish WebSocket connection to Lightstreamer.

        Raises:
            StreamDisconnectedError: If connection cannot be established.
        """
        try:
            self._ws = await websockets.connect(
                self._stream_url,
                additional_headers={
                    "CST": self._cst,
                    "X-SECURITY-TOKEN": self._security_token,
                },
            )
            self._connected = True
            logger.info("WebSocket connection established")
        except Exception as exc:
            self._connected = False
            raise StreamDisconnectedError(
                "Failed to connect to Lightstreamer",
                url=self._stream_url,
                error=str(exc),
            ) from exc

    async def _disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    async def _listen(self) -> None:
        """Background task that reads messages from the WebSocket and dispatches ticks."""
        while self._running:
            try:
                if self._ws is None or not self._connected:
                    await asyncio.sleep(0.1)
                    continue

                message = await self._ws.recv()
                await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed:
                if not self._running:
                    break
                logger.warning("WebSocket connection closed unexpectedly")
                try:
                    await self._reconnect()
                except StreamDisconnectedError:
                    logger.error("Stream reconnection failed, stopping listener")
                    self._running = False
                    break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Error in stream listener",
                    extra={"error": str(exc)},
                )
                await asyncio.sleep(0.1)

    async def _handle_message(self, message: str | bytes) -> None:
        """Parse and route an incoming WebSocket message.

        Lightstreamer messages are typically pipe-delimited text.
        This handles the MARKET subscription update format.

        Args:
            message: Raw message from the WebSocket.
        """
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        # Lightstreamer uses a text-based protocol
        # Format: U,<subscription_id>,<item_index>,<field1>|<field2>|...
        # For market data: fields are typically BID, OFFER, UPDATE_TIME
        lines = message.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse update messages
            if line.startswith("U,"):
                await self._parse_tick_update(line)

    async def _parse_tick_update(self, line: str) -> None:
        """Parse a Lightstreamer tick update line.

        Args:
            line: A line starting with 'U,' containing tick data.
        """
        try:
            parts = line.split(",")
            if len(parts) < 4:
                return

            # Extract subscription info and fields
            # Format: U,<sub_id>,<item_pos>,<field1>|<field2>|...
            fields_str = ",".join(parts[3:])
            field_values = fields_str.split("|")

            # Map to epic from subscription tracking
            sub_id = parts[1].strip()
            epic = self._resolve_epic_from_subscription(sub_id)
            if epic is None:
                return

            # Parse tick fields (BID, OFFER/ASK, UPDATE_TIME)
            tick_data: dict[str, Any] = {}
            if len(field_values) >= 1 and field_values[0]:
                tick_data["bid"] = float(field_values[0])
            if len(field_values) >= 2 and field_values[1]:
                tick_data["ask"] = float(field_values[1])
            if len(field_values) >= 3 and field_values[2]:
                tick_data["timestamp"] = field_values[2]

            if tick_data:
                await self._on_tick(epic, tick_data)

        except (ValueError, IndexError) as exc:
            logger.debug(
                "Failed to parse tick update",
                extra={"line": line[:100], "error": str(exc)},
            )

    def _resolve_epic_from_subscription(self, sub_id: str) -> str | None:
        """Resolve an epic from a subscription ID.

        In a full implementation, this maps Lightstreamer subscription IDs
        to instrument epics. For now, uses the sub_id directly if it matches
        a known subscription, or attempts to find it in the subscriptions dict.

        Args:
            sub_id: The Lightstreamer subscription identifier.

        Returns:
            The epic string if found, None otherwise.
        """
        # Direct match
        if sub_id in self._subscriptions:
            return sub_id

        # Try numeric index mapping (subscriptions ordered by registration)
        try:
            idx = int(sub_id) - 1
            epics = list(self._subscriptions.keys())
            if 0 <= idx < len(epics):
                return epics[idx]
        except ValueError:
            pass

        return None

    async def _send_subscription(self, epic: str) -> None:
        """Send a subscription request to Lightstreamer for an instrument.

        Args:
            epic: The IG instrument identifier.

        Raises:
            StreamDisconnectedError: If not connected.
        """
        if self._ws is None or not self._connected:
            raise StreamDisconnectedError(
                "Cannot subscribe: stream not connected",
                epic=epic,
            )

        # Lightstreamer subscription message format
        # control request to subscribe to MARKET data
        sub_message = (
            f"control\n"
            f"LS_op=add\n"
            f"LS_subId={epic}\n"
            f"LS_mode=MERGE\n"
            f"LS_group=MARKET:{epic}\n"
            f"LS_schema=BID OFFER UPDATE_TIME\n"
        )

        await self._ws.send(sub_message)

    async def _send_unsubscription(self, epic: str) -> None:
        """Send an unsubscription request to Lightstreamer.

        Args:
            epic: The IG instrument identifier.

        Raises:
            StreamDisconnectedError: If not connected.
        """
        if self._ws is None or not self._connected:
            raise StreamDisconnectedError(
                "Cannot unsubscribe: stream not connected",
                epic=epic,
            )

        unsub_message = (
            f"control\n"
            f"LS_op=delete\n"
            f"LS_subId={epic}\n"
        )

        await self._ws.send(unsub_message)
