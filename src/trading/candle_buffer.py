"""Rolling OHLC candle buffer fed by live streaming ticks.

Accumulates price ticks published on the Event Bus and builds 1-minute
OHLC candles per instrument. The trading loop reads from this buffer
instead of polling the /prices REST endpoint, avoiding the
``exceeded-account-historical-data-allowance`` quota error.

Usage::

    buf = CandleBuffer(candle_period_seconds=60, max_candles=200)
    buf.on_tick("CS.D.EURUSD.CFD.IP", bid=1.1234, ask=1.1236, ts=time.time())

    candles = buf.get_candles("CS.D.EURUSD.CFD.IP")
    # [{"open": ..., "high": ..., "low": ..., "close": ..., "time": ...}, ...]
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

# Default candle period in seconds (1 minute)
DEFAULT_CANDLE_PERIOD_SECONDS: int = 60

# Maximum candles to keep in memory per instrument
DEFAULT_MAX_CANDLES: int = 200

# Minimum candles required before the strategy will run
MIN_CANDLES_FOR_STRATEGY: int = 30


@dataclass
class _OpenCandle:
    """A partially-built candle that is still accumulating ticks."""

    period_start: float  # Unix timestamp of the candle's open
    open: float
    high: float
    low: float
    close: float
    tick_count: int = 0

    def update(self, price: float) -> None:
        """Incorporate a new price tick."""
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.tick_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.period_start,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_count": self.tick_count,
        }


class CandleBuffer:
    """Per-instrument rolling OHLC candle store.

    Thread-safe for asyncio (single-thread event loop). Not safe for
    multi-thread access without an explicit lock.

    Args:
        candle_period_seconds: Duration of each candle in seconds. Default 60.
        max_candles: Maximum closed candles to keep per instrument. Default 200.
    """

    def __init__(
        self,
        candle_period_seconds: int = DEFAULT_CANDLE_PERIOD_SECONDS,
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> None:
        self._period = candle_period_seconds
        self._max_candles = max_candles
        # closed candles per epic: deque of dicts
        self._closed: dict[str, deque[dict[str, Any]]] = {}
        # open (current) candle per epic
        self._open: dict[str, _OpenCandle] = {}
        # total tick counts for diagnostics
        self._tick_counts: dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def on_tick(self, epic: str, bid: float | None, ask: float | None, ts: float | None = None) -> None:
        """Feed a price tick into the buffer.

        The mid-price ((bid + ask) / 2) is used as the canonical price.
        If only one side is available, that side is used directly.

        Args:
            epic: Instrument identifier.
            bid: Bid price (may be None).
            ask: Ask price (may be None).
            ts: Unix timestamp of the tick. Defaults to ``time.time()``.
        """
        if bid is None and ask is None:
            return

        # Compute mid price
        if bid is not None and ask is not None:
            try:
                price = (float(bid) + float(ask)) / 2
            except (TypeError, ValueError):
                return
        elif bid is not None:
            try:
                price = float(bid)
            except (TypeError, ValueError):
                return
        else:
            try:
                price = float(ask)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return

        now = ts if ts is not None else time.time()
        period_start = (now // self._period) * self._period

        # Initialise closed candles deque for new epics
        if epic not in self._closed:
            self._closed[epic] = deque(maxlen=self._max_candles)
            self._tick_counts[epic] = 0

        self._tick_counts[epic] += 1

        current = self._open.get(epic)

        if current is None:
            # Start the very first candle
            self._open[epic] = _OpenCandle(
                period_start=period_start,
                open=price,
                high=price,
                low=price,
                close=price,
                tick_count=1,
            )
        elif period_start > current.period_start:
            # Close the current candle and open a new one
            self._closed[epic].append(current.to_dict())
            logger.debug(
                "Candle closed",
                extra={
                    "epic": epic,
                    "candle": current.to_dict(),
                    "total_closed": len(self._closed[epic]),
                },
            )
            self._open[epic] = _OpenCandle(
                period_start=period_start,
                open=price,
                high=price,
                low=price,
                close=price,
                tick_count=1,
            )
        else:
            # Update the existing open candle
            current.update(price)

    def get_candles(self, epic: str, include_open: bool = True) -> list[dict[str, Any]]:
        """Return closed candles for an instrument, optionally appending the open candle.

        Args:
            epic: Instrument identifier.
            include_open: If True, append the current open (incomplete) candle
                          as the last element. Useful so strategies always see
                          the most recent price even before the candle closes.

        Returns:
            List of candle dicts ordered oldest → newest.
            Each dict has keys: ``time``, ``open``, ``high``, ``low``, ``close``, ``tick_count``.
        """
        closed = list(self._closed.get(epic, []))
        if include_open and epic in self._open:
            closed = closed + [self._open[epic].to_dict()]
        return closed

    def candle_count(self, epic: str) -> int:
        """Number of closed candles available for an instrument."""
        return len(self._closed.get(epic, []))

    def is_ready(self, epic: str, min_candles: int = MIN_CANDLES_FOR_STRATEGY) -> bool:
        """Whether enough candles have accumulated to run the strategy.

        Args:
            epic: Instrument identifier.
            min_candles: Minimum closed candles required.

        Returns:
            True if candle_count(epic) >= min_candles.
        """
        return self.candle_count(epic) >= min_candles

    def tick_count(self, epic: str) -> int:
        """Total ticks received for an instrument."""
        return self._tick_counts.get(epic, 0)

    def get_status(self) -> dict[str, Any]:
        """Summary of buffer state across all instruments."""
        return {
            epic: {
                "closed_candles": len(self._closed.get(epic, [])),
                "has_open_candle": epic in self._open,
                "tick_count": self._tick_counts.get(epic, 0),
                "ready": self.is_ready(epic),
            }
            for epic in set(list(self._closed.keys()) + list(self._open.keys()))
        }
