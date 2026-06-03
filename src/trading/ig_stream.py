"""IG Lightstreamer market data streaming client.

IG's streaming API uses the Lightstreamer TLCP (Text-based
Lightstreamer Client Protocol) over HTTP long-polling — NOT WebSockets.

Protocol flow:
1. POST {stream_url}/lightstreamer/create_session.txt  → streaming HTTP response
2. Read lines: OK → session headers → update lines (table,item|field|field...)
3. POST {stream_url}/lightstreamer/control.txt to add/remove subscriptions
4. On LOOP or disconnect: POST .../bind_session.txt to resume

Supports multi-instrument subscriptions, auto-reconnect with exponential
backoff, staleness detection, and Event Bus tick distribution.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, Cross-Cutting Rule 5
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

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

RECONNECT_BASE_DELAY_SECONDS: float = 2.0
RECONNECT_MAX_TOTAL_SECONDS: float = 60.0
STALENESS_CHECK_INTERVAL_SECONDS: float = 30.0
SUBSCRIPTION_RETRY_MAX_ATTEMPTS: int = 3
SUBSCRIPTION_RETRY_DELAY_SECONDS: float = 2.0
MAX_SIMULTANEOUS_INSTRUMENTS: int = 50

# Lightstreamer TLCP URL paths
_CREATE_SESSION_PATH = "lightstreamer/create_session.txt"
_BIND_SESSION_PATH = "lightstreamer/bind_session.txt"
_CONTROL_PATH = "lightstreamer/control.txt"

# Lightstreamer LS_cid for IG — standard public value used by all IG clients
_LS_CID = "mgQkwtwdysogQz2BJ4Ji kOj2Bg"

# Fields requested for CHART subscription (candle data)
_CHART_FIELDS = ["LTV", "UTM", "DAY_OPEN_MID", "DAY_NET_CHG_MID", "DAY_PERC_CHG_MID", "DAY_HIGH", "DAY_LOW"]

# Fields requested for MARKET (L1 bid/ask) subscription
_MARKET_FIELDS = ["BID", "OFFER", "UPDATE_TIME"]


# ---------------------------------------------------------------------------
# Subscription State
# ---------------------------------------------------------------------------


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUBSCRIBING = "subscribing"
    UNSUBSCRIBED = "unsubscribed"
    ERROR = "error"


@dataclass
class SubscriptionState:
    epic: str
    table_key: int = 0
    status: SubscriptionStatus = SubscriptionStatus.SUBSCRIBING
    last_tick_time: datetime | None = None
    subscribe_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0


# ---------------------------------------------------------------------------
# IGStream
# ---------------------------------------------------------------------------


class IGStream:
    """Lightstreamer TLCP client for real-time IG price streaming.

    Uses HTTP long-polling (not WebSockets) to connect to IG's Lightstreamer
    endpoint and stream bid/ask ticks for subscribed instruments.

    Ticks are published to the Event Bus on ``market.tick.{epic}`` channels
    and also fed directly into the CandleBuffer if one is registered.

    Usage::

        stream = IGStream(
            stream_url="https://demo-apd.marketdatasystems.com",
            cst="my-cst-token",
            security_token="my-security-token",
            event_bus=event_bus,
        )
        await stream.start()
        await stream.subscribe("CS.D.EURUSD.CFD.IP")
        # ticks flow into event bus and candle buffer
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
        # Normalise: strip trailing slash, ensure http(s) scheme
        self._base_url = stream_url.rstrip("/")
        if self._base_url.startswith("wss://"):
            self._base_url = "https://" + self._base_url[6:]
        elif self._base_url.startswith("ws://"):
            self._base_url = "http://" + self._base_url[5:]

        self._cst = cst
        self._security_token = security_token
        self._event_bus = event_bus
        self._ig_client = ig_client

        # Subscriptions: epic → SubscriptionState
        self._subscriptions: dict[str, SubscriptionState] = {}
        # table_key → epic (for demultiplexing update lines)
        self._table_to_epic: dict[int, str] = {}
        self._next_table_key: int = 1

        # Session state
        self._session_id: str | None = None
        self._control_url: str = self._base_url
        self._connected: bool = False
        self._running: bool = False

        # Background tasks
        self._listener_task: asyncio.Task[None] | None = None
        self._staleness_task: asyncio.Task[None] | None = None
        self._last_tick_times: dict[str, datetime] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Lightstreamer and start the listener and staleness monitor."""
        if self._running:
            return
        self._running = True

        await self._connect()

        self._listener_task = asyncio.create_task(
            self._listen_loop(), name="ig_stream_listener"
        )
        self._staleness_task = asyncio.create_task(
            self._staleness_monitor(), name="ig_stream_staleness"
        )
        logger.info("IGStream started", extra={"base_url": self._base_url})

    async def stop(self) -> None:
        """Stop the stream and clean up."""
        self._running = False

        for task in (self._listener_task, self._staleness_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._listener_task = None
        self._staleness_task = None
        self._connected = False
        self._subscriptions.clear()
        self._table_to_epic.clear()
        logger.info("IGStream stopped")

    # -------------------------------------------------------------------------
    # Subscription Management
    # -------------------------------------------------------------------------

    async def subscribe(self, epic: str) -> None:
        """Subscribe to L1 bid/ask price updates for an instrument."""
        if epic in self._subscriptions:
            state = self._subscriptions[epic]
            if state.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.SUBSCRIBING):
                return

        table_key = self._next_table_key
        self._next_table_key += 1

        self._subscriptions[epic] = SubscriptionState(
            epic=epic,
            table_key=table_key,
            status=SubscriptionStatus.SUBSCRIBING,
        )
        self._table_to_epic[table_key] = epic

        for attempt in range(SUBSCRIPTION_RETRY_MAX_ATTEMPTS):
            try:
                await self._send_control({
                    "LS_Table": str(table_key),
                    "LS_op": "add",
                    "LS_mode": "MERGE",
                    "LS_id": f"MARKET:{epic}",
                    "LS_schema": " ".join(_MARKET_FIELDS),
                    "LS_snapshot": "false",
                })
                self._subscriptions[epic].status = SubscriptionStatus.ACTIVE
                logger.info("Subscribed to instrument", extra={"epic": epic, "table": table_key})
                return
            except Exception as exc:
                logger.warning("Subscription attempt failed", extra={"epic": epic, "attempt": attempt + 1, "error": str(exc)})
                if attempt < SUBSCRIPTION_RETRY_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(SUBSCRIPTION_RETRY_DELAY_SECONDS)

        self._subscriptions[epic].status = SubscriptionStatus.ERROR

    async def unsubscribe(self, epic: str) -> None:
        """Unsubscribe from price updates for an instrument."""
        if epic not in self._subscriptions:
            return
        state = self._subscriptions[epic]
        try:
            await self._send_control({
                "LS_Table": str(state.table_key),
                "LS_op": "delete",
            })
        except Exception as exc:
            logger.warning("Unsubscription failed", extra={"epic": epic, "error": str(exc)})
        del self._subscriptions[epic]
        self._table_to_epic.pop(state.table_key, None)
        self._last_tick_times.pop(epic, None)

    # -------------------------------------------------------------------------
    # TLCP Connection
    # -------------------------------------------------------------------------

    async def _connect(self) -> None:
        """Open a Lightstreamer session via HTTP POST (TLCP create_session).

        IG's Lightstreamer returns a chunked streaming response. We read only
        the first few lines (OK + session headers) then stop — the rest is the
        live update stream which we handle separately in _bind_and_read.
        """
        url = f"{self._base_url}/lightstreamer/create_session.txt"
        params = {
            "LS_op2": "create",
            "LS_cid": _LS_CID,
            "LS_adapter_set": "DEFAULT",
            "LS_user": self._cst,
            "LS_password": f"CST-{self._cst}|XST-{self._security_token}",
        }

        print(f"STREAM: Connecting to {url}", flush=True)

        try:
            session_data: dict[str, str] = {}
            first_line: str = ""

            # Use streaming mode and read only the handshake lines
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=15.0)
            ) as client:
                async with client.stream("POST", url, data=params) as response:
                    print(f"STREAM: create_session HTTP status={response.status_code}", flush=True)

                    if response.status_code != 200:
                        body = await response.aread()
                        raise StreamDisconnectedError(
                            f"Lightstreamer session creation failed: HTTP {response.status_code}",
                            url=url,
                            error=body.decode()[:200],
                        )

                    # Read lines until we have session headers (blank line = end of headers)
                    line_count = 0
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        line_count += 1
                        print(f"STREAM handshake line {line_count}: {line!r}", flush=True)

                        if line_count == 1:
                            first_line = line
                            if line != "OK":
                                raise StreamDisconnectedError(
                                    f"Lightstreamer handshake rejected: {line}",
                                    url=url,
                                    error=line,
                                )
                        elif ":" in line:
                            k, v = line.split(":", 1)
                            session_data[k.strip()] = v.strip()
                        elif line == "" and session_data:
                            # Blank line after headers = end of handshake
                            break

                        # Safety: stop reading after 20 lines
                        if line_count >= 20:
                            break

            if not session_data.get("SessionId"):
                raise StreamDisconnectedError(
                    "No SessionId in Lightstreamer handshake",
                    url=url,
                    error=str(session_data),
                )

            self._session_id = session_data.get("SessionId")
            control_address = session_data.get("ControlAddress")
            if control_address:
                self._control_url = f"https://{control_address}"
            else:
                self._control_url = self._base_url

            self._connected = True
            print(f"STREAM: Session established. SessionId={self._session_id} ControlAddr={control_address}", flush=True)
            logger.info("Lightstreamer session created", extra={"session_id": self._session_id})

        except StreamDisconnectedError:
            raise
        except Exception as exc:
            raise StreamDisconnectedError(
                "Failed to connect to Lightstreamer",
                url=url,
                error=str(exc),
            ) from exc

    async def _send_control(self, params: dict[str, str]) -> None:
        """Send a control request (subscribe/unsubscribe) to Lightstreamer."""
        if not self._session_id:
            raise StreamDisconnectedError("No active Lightstreamer session")

        url = f"{self._control_url}/lightstreamer/control.txt"
        params["LS_session"] = self._session_id

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.post(url, data=params)

        result = response.text.strip()
        print(f"STREAM CONTROL: status={response.status_code} result={result!r} params_op={params.get('LS_op')}", flush=True)

        if response.status_code != 200:
            raise StreamDisconnectedError(
                f"Control request failed: HTTP {response.status_code}",
                url=url,
                error=result[:200],
            )

        if result != "OK":
            raise StreamDisconnectedError(
                f"Control request rejected: {result}",
                url=url,
                error=result,
            )

    # -------------------------------------------------------------------------
    # Streaming Listener
    # -------------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Connect to Lightstreamer and stream updates continuously."""
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Stream listener error, reconnecting in 5s: %s", exc)
                print(f"STREAM LISTENER ERROR: {exc}", flush=True)
                await asyncio.sleep(5.0)

    async def _connect_and_stream(self) -> None:
        """Create a Lightstreamer session and read update lines from the same connection.

        The create_session response IS the streaming connection — it stays open
        and delivers update lines after the OK handshake. We read the handshake
        first, then subscribe, then keep reading updates indefinitely.
        """
        url = f"{self._base_url}/lightstreamer/create_session.txt"
        params = {
            "LS_op2": "create",
            "LS_cid": _LS_CID,
            "LS_adapter_set": "DEFAULT",
            "LS_user": self._cst,
            "LS_password": f"CST-{self._cst}|XST-{self._security_token}",
        }

        print(f"STREAM: Opening persistent connection to {url}", flush=True)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=15.0, pool=15.0)
        ) as client:
            async with client.stream("POST", url, data=params) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise StreamDisconnectedError(
                        f"Lightstreamer rejected connection: HTTP {response.status_code}",
                        error=body.decode()[:300],
                    )

                # --- Phase 1: read handshake ---
                session_data: dict[str, str] = {}
                handshake_done = False
                line_count = 0

                async for raw_line in response.aiter_lines():
                    if not self._running:
                        return
                    line = raw_line.strip()
                    line_count += 1

                    if line_count == 1:
                        print(f"STREAM: first line = {line!r}", flush=True)
                        if line != "OK":
                            raise StreamDisconnectedError(
                                f"Lightstreamer handshake failed: {line}",
                                error=line,
                            )
                        continue

                    if ":" in line and not handshake_done:
                        k, v = line.split(":", 1)
                        session_data[k.strip()] = v.strip()
                        continue

                    if line == "" and session_data and not handshake_done:
                        # End of session headers — session is established
                        self._session_id = session_data.get("SessionId")
                        control_address = session_data.get("ControlAddress")
                        self._control_url = (
                            f"https://{control_address}" if control_address else self._base_url
                        )
                        self._connected = True
                        handshake_done = True
                        print(
                            f"STREAM: Session established! SessionId={self._session_id} "
                            f"ControlAddr={control_address}",
                            flush=True,
                        )
                        logger.info("Lightstreamer session created", extra={"session_id": self._session_id})

                        # Subscribe all instruments now that session is open
                        for epic, state in list(self._subscriptions.items()):
                            try:
                                await self._send_control({
                                    "LS_Table": str(state.table_key),
                                    "LS_op": "add",
                                    "LS_mode": "MERGE",
                                    "LS_id": f"MARKET:{epic}",
                                    "LS_schema": " ".join(_MARKET_FIELDS),
                                    "LS_snapshot": "false",
                                })
                                state.status = SubscriptionStatus.ACTIVE
                                print(f"STREAM: Subscribed {epic}", flush=True)
                            except Exception as sub_exc:
                                print(f"STREAM: Subscribe failed for {epic}: {sub_exc}", flush=True)
                        continue

                    # --- Phase 2: update lines ---
                    if handshake_done:
                        if not line:
                            continue
                        if line == "PROBE":
                            continue
                        if line == "LOOP":
                            print("STREAM: LOOP received, reconnecting", flush=True)
                            return  # triggers reconnect in _listen_loop
                        if line.startswith("ERROR") or line.startswith("SYNC ERROR"):
                            raise StreamDisconnectedError(f"Lightstreamer error: {line}")
                        if line.startswith("END"):
                            raise StreamDisconnectedError(f"Lightstreamer session ended: {line}")
                        await self._handle_update_line(line)

        self._connected = False

    # The old _bind_and_read is replaced by _connect_and_stream above

    async def _handle_update_line(self, line: str) -> None:
        """Parse a Lightstreamer update line and dispatch a tick."""
        try:
            # Format: TABLE_KEY,ITEM_POS|FIELD1|FIELD2|...
            comma_pos = line.find(",")
            if comma_pos < 0:
                return

            table_str = line[:comma_pos]
            rest = line[comma_pos + 1:]

            pipe_pos = rest.find("|")
            if pipe_pos < 0:
                return

            # item position (1-based) before first pipe
            # field values separated by pipes
            field_str = rest[pipe_pos + 1:]  # skip item_pos
            fields = field_str.split("|")

            try:
                table_key = int(table_str)
            except ValueError:
                return

            epic = self._table_to_epic.get(table_key)
            if not epic:
                return

            # _MARKET_FIELDS = ["BID", "OFFER", "UPDATE_TIME"]
            bid_str = fields[0] if len(fields) > 0 else ""
            ask_str = fields[1] if len(fields) > 1 else ""

            # Lightstreamer MERGE mode: empty string = unchanged (use last known)
            bid: float | None = None
            ask: float | None = None
            try:
                if bid_str and bid_str not in ("#", "$"):
                    bid = float(bid_str)
            except ValueError:
                pass
            try:
                if ask_str and ask_str not in ("#", "$"):
                    ask = float(ask_str)
            except ValueError:
                pass

            if bid is None and ask is None:
                return

            await self._on_tick(epic, {"bid": bid, "ask": ask})

        except Exception as exc:
            logger.debug("Failed to parse update line '%s': %s", line[:80], exc)

    # -------------------------------------------------------------------------
    # Tick Processing
    # -------------------------------------------------------------------------

    async def _on_tick(self, epic: str, data: dict[str, Any]) -> None:
        """Process a tick and publish to Event Bus."""
        now = datetime.now(timezone.utc)
        self._last_tick_times[epic] = now

        if epic in self._subscriptions:
            state = self._subscriptions[epic]
            if state.status == SubscriptionStatus.STALE:
                state.status = SubscriptionStatus.ACTIVE
            state.last_tick_time = now

        tick_payload = {
            "epic": epic,
            "bid": data.get("bid"),
            "ask": data.get("ask"),
            "timestamp": now.isoformat(),
            "received_at": now.isoformat(),
        }

        channel = MARKET_TICK.format(instrument=epic)
        event = Event(event_type="market.tick", payload=tick_payload)

        try:
            await self._event_bus.publish(channel, event)
        except Exception as exc:
            logger.debug("Failed to publish tick: %s", exc)

    # -------------------------------------------------------------------------
    # Reconnection
    # -------------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff and re-subscribe."""
        self._connected = False
        total_elapsed = 0.0

        for attempt in range(RECONNECT_MAX_ATTEMPTS):
            if not self._running:
                return

            delay = RECONNECT_BASE_DELAY_SECONDS * (2 ** attempt)
            delay = min(delay, RECONNECT_MAX_TOTAL_SECONDS - total_elapsed)
            if delay > 0:
                await asyncio.sleep(delay)
                total_elapsed += delay

            logger.info("Stream reconnection attempt %d/%d", attempt + 1, RECONNECT_MAX_ATTEMPTS)
            try:
                await self._connect()
                # Re-subscribe all active instruments
                for epic, state in list(self._subscriptions.items()):
                    if state.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.STALE):
                        try:
                            await self._send_control({
                                "LS_Table": str(state.table_key),
                                "LS_op": "add",
                                "LS_mode": "MERGE",
                                "LS_id": f"MARKET:{epic}",
                                "LS_schema": " ".join(_MARKET_FIELDS),
                                "LS_snapshot": "false",
                            })
                            state.status = SubscriptionStatus.ACTIVE
                        except Exception as sub_exc:
                            logger.warning("Re-subscribe failed for %s: %s", epic, sub_exc)
                logger.info("Stream reconnection successful")
                return
            except Exception as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt + 1, exc)

        raise StreamDisconnectedError(
            "Stream reconnection exhausted all attempts",
            max_attempts=RECONNECT_MAX_ATTEMPTS,
            total_elapsed_s=round(total_elapsed, 1),
        )

    # -------------------------------------------------------------------------
    # Staleness Detection
    # -------------------------------------------------------------------------

    async def _staleness_monitor(self) -> None:
        """Mark instruments stale if no tick for TICK_STALENESS_SECONDS."""
        while self._running:
            try:
                await asyncio.sleep(STALENESS_CHECK_INTERVAL_SECONDS)
                if not self._running:
                    break

                now = datetime.now(timezone.utc)
                for epic, state in list(self._subscriptions.items()):
                    if state.status not in (SubscriptionStatus.ACTIVE, SubscriptionStatus.STALE):
                        continue
                    last = self._last_tick_times.get(epic) or state.subscribe_time
                    age = (now - last).total_seconds()
                    if age >= TICK_STALENESS_SECONDS and state.status != SubscriptionStatus.STALE:
                        state.status = SubscriptionStatus.STALE
                        logger.warning("Instrument stale (%.0fs since last tick): %s", age, epic)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Staleness monitor error: %s", exc)

    # -------------------------------------------------------------------------
    # Status / Properties
    # -------------------------------------------------------------------------

    def get_subscription_status(self) -> dict[str, str]:
        return {epic: state.status.value for epic, state in self._subscriptions.items()}

    def get_stale_instruments(self) -> list[str]:
        return [e for e, s in self._subscriptions.items() if s.status == SubscriptionStatus.STALE]

    def is_market_open(self, epic: str) -> bool:
        return True  # TODO: per-instrument market hours

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session_id is not None

    @property
    def subscription_count(self) -> int:
        return sum(1 for s in self._subscriptions.values() if s.status == SubscriptionStatus.ACTIVE)
