"""Autonomous trading loop — snapshot polling strategy.

Uses /markets/{epic} REST endpoint to poll live bid/ask prices every
SNAPSHOT_INTERVAL_SECONDS. This endpoint is NOT subject to the historical
data allowance quota. Prices are fed into a CandleBuffer to build 1-minute
OHLC candles. Once 30 candles accumulate, the SMA/ATR strategy runs and
trades are placed.

No Lightstreamer / WebSocket dependency.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.config.settings import get_settings
from src.core.exceptions import IGAuthenticationError
from src.core.logging import get_logger
from src.trading.candle_buffer import CandleBuffer, MIN_CANDLES_FOR_STRATEGY

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENTS = [
    "CS.D.EURUSD.CFD.IP",   # EUR/USD — primary instrument
    # Others disabled to stay under IG demo API allowance limits.
    # Re-enable once on a higher IG API tier.
    # "CS.D.GBPUSD.CFD.IP",
    # "CS.D.USDJPY.CFD.IP",
    # "IX.D.FTSE.DAILY.IP",
    # "IX.D.DAX.DAILY.IP",
]

# How often to poll /markets snapshots (seconds)
# 1 instrument × 1 call = 1 req per cycle, well under IG demo limits
SNAPSHOT_INTERVAL_SECONDS: float = 90.0

# How often the main trading cycle runs (seconds)
LOOP_INTERVAL_SECONDS: float = 60.0

# How often to refresh account equity (seconds) — avoids /accounts rate limit
ACCOUNT_REFRESH_INTERVAL_SECONDS: float = 300.0

MAX_OPEN_POSITIONS = 5
DEMO_MAX_POSITION_PCT = 0.02


@dataclass
class TradingLoopState:
    running: bool = False
    connected: bool = False
    last_tick_time: float = 0.0
    signals_generated: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    errors: int = 0
    last_error: str = ""
    start_time: float = field(default_factory=time.time)


class AutonomousTradingLoop:
    """Snapshot-polling trading loop. No streaming required."""

    def __init__(
        self,
        instruments: list[str] | None = None,
        loop_interval: float = LOOP_INTERVAL_SECONDS,
    ) -> None:
        self._instruments = instruments or DEFAULT_INSTRUMENTS
        self._loop_interval = loop_interval
        self._state = TradingLoopState()
        self._task: asyncio.Task[None] | None = None
        self._snapshot_task: asyncio.Task[None] | None = None
        self._ig_client: Any = None
        self._candle_buffer = CandleBuffer(candle_period_seconds=60, max_candles=200)
        self._account_equity: Decimal = Decimal("20000")
        self._open_positions: list[dict[str, Any]] = []
        self._last_account_refresh: float = 0.0
        # Expose for debug endpoint compatibility
        self._ig_stream = None
        self._event_bus = None

    @property
    def state(self) -> TradingLoopState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state.running

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._state.running:
            return
        self._state = TradingLoopState()
        self._state.running = True
        self._task = asyncio.create_task(self._run_loop(), name="trading_loop")

    async def stop(self) -> None:
        self._state.running = False
        for task in (self._snapshot_task, self._task):
            if task is not None:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        self._snapshot_task = None
        self._task = None
        if self._ig_client is not None:
            try:
                await asyncio.wait_for(self._ig_client.stop(), timeout=3.0)
            except (Exception, asyncio.TimeoutError):
                pass
            self._ig_client = None
        self._state.connected = False

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    async def _run_loop(self) -> None:
        try:
            if not await self._connect_to_ig():
                self._state.running = False
                return

            self._state.connected = True

            # Start background snapshot poller
            self._snapshot_task = asyncio.create_task(
                self._snapshot_loop(), name="snapshot_poller"
            )

            # Main trading cycle
            while self._state.running:
                try:
                    await self._trading_cycle()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._state.errors += 1
                    self._state.last_error = str(exc)
                    print(f"CYCLE ERROR: {exc}", flush=True)
                    if self._state.errors > 50:
                        break
                await asyncio.sleep(self._loop_interval)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._state.last_error = str(exc)
            print(f"FATAL LOOP ERROR: {exc}", flush=True)
        finally:
            self._state.running = False

    # -------------------------------------------------------------------------
    # Snapshot Polling (replaces Lightstreamer)
    # -------------------------------------------------------------------------

    async def _snapshot_loop(self) -> None:
        """Poll /markets/{epic} every SNAPSHOT_INTERVAL_SECONDS.

        /markets does NOT count against the historical data allowance.
        Each poll feeds a bid/ask tick into the candle buffer.
        """
        print(
            f"SNAPSHOT: Starting poller for {len(self._instruments)} instruments "
            f"every {SNAPSHOT_INTERVAL_SECONDS}s",
            flush=True,
        )
        while self._state.running:
            try:
                await self._poll_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"SNAPSHOT ERROR: {exc}", flush=True)
            await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)

    async def _poll_snapshots(self) -> None:
        """Fetch current bid/ask from /markets for each instrument."""
        if self._ig_client is None or not self._ig_client.is_connected:
            return

        for epic in self._instruments:
            if not self._state.running:
                break
            try:
                details = await self._ig_client.get_market_details(epic)
                snapshot = details.get("snapshot", {})
                bid = snapshot.get("bid")
                offer = snapshot.get("offer")

                if bid is not None and offer is not None:
                    self._candle_buffer.on_tick(
                        epic=epic,
                        bid=float(bid),
                        ask=float(offer),
                    )
                    self._state.last_tick_time = time.time()
                    print(
                        f"TICK: {epic} bid={bid} ask={offer} "
                        f"candles={self._candle_buffer.candle_count(epic)} "
                        f"ticks={self._candle_buffer.tick_count(epic)}",
                        flush=True,
                    )
                else:
                    print(f"SNAPSHOT: {epic} no bid/ask in snapshot={snapshot}", flush=True)

            except Exception as exc:
                print(f"SNAPSHOT {epic}: {exc}", flush=True)

            # Spread requests: 5 instruments × 3s = 15s per cycle, well under rate limits
            await asyncio.sleep(3.0)

    # -------------------------------------------------------------------------
    # IG Connection
    # -------------------------------------------------------------------------

    async def _connect_to_ig(self) -> bool:
        from src.trading.ig_client import IGClient

        settings = get_settings()
        if not settings.ig_api_key or settings.ig_api_key in ("your_ig_api_key", ""):
            print("IG_API_KEY not configured", flush=True)
            return False
        if not settings.ig_username or settings.ig_username in ("your_ig_username", ""):
            print("IG_USERNAME not configured", flush=True)
            return False
        if not settings.ig_password or settings.ig_password in ("your_ig_password", ""):
            print("IG_PASSWORD not configured", flush=True)
            return False

        print(f"IG CONNECT: Connecting to {settings.ig_account_type}...", flush=True)
        try:
            self._ig_client = IGClient(
                api_key=settings.ig_api_key,
                username=settings.ig_username,
                password=settings.ig_password,
                account_type=settings.ig_account_type,
            )
            await self._ig_client.start()
            print("IG CONNECT: Authenticated OK", flush=True)
            return True
        except IGAuthenticationError as exc:
            print(f"IG AUTH FAILED: {exc}", flush=True)
            return False
        except Exception as exc:
            print(f"IG CONNECT ERROR: {exc}", flush=True)
            return False

    # -------------------------------------------------------------------------
    # Trading Cycle
    # -------------------------------------------------------------------------

    async def _trading_cycle(self) -> None:
        if self._ig_client is None or not self._ig_client.is_connected:
            await self._connect_to_ig()
            return

        await self._update_account_state()

        if len(self._open_positions) >= MAX_OPEN_POSITIONS:
            return

        buf_status = self._candle_buffer.get_status()
        print(f"BUFFER: {buf_status}", flush=True)

        for epic in self._instruments:
            if not self._state.running:
                break

            # Skip if already have a position in this epic
            open_epics = {
                p.get("market", {}).get("epic") or p.get("epic", "")
                for p in self._open_positions
            }
            if epic in open_epics:
                print(f"SKIP {epic}: already have open position", flush=True)
                continue

            if not self._candle_buffer.is_ready(epic, MIN_CANDLES_FOR_STRATEGY):
                cnt = self._candle_buffer.candle_count(epic)
                ticks = self._candle_buffer.tick_count(epic)
                print(
                    f"WAITING {epic}: {cnt}/{MIN_CANDLES_FOR_STRATEGY} candles "
                    f"({ticks} ticks)",
                    flush=True,
                )
                continue
            try:
                signal = await self._analyze_instrument(epic)
                if signal:
                    self._state.signals_generated += 1
                    await self._execute_signal(signal)
            except Exception as exc:
                print(f"ANALYZE ERROR {epic}: {exc}", flush=True)

    async def _update_account_state(self) -> None:
        now = time.time()
        if now - self._last_account_refresh < ACCOUNT_REFRESH_INTERVAL_SECONDS:
            # Still refresh positions every cycle (cheap call, critical for skip logic)
            try:
                self._open_positions = await self._ig_client.get_positions()
            except Exception as exc:
                print(f"POSITIONS ERROR: {exc}", flush=True)
            return
        self._last_account_refresh = now

        try:
            info = await self._ig_client.get_account_info()
            balance = info.get("balance", {}) or {}
            equity = (
                balance.get("balance")
                or balance.get("equity")
                or balance.get("available")
            )
            if equity:
                self._account_equity = Decimal(str(equity))
                print(f"ACCOUNT: equity={self._account_equity} AUD", flush=True)
        except Exception as exc:
            print(f"ACCOUNT ERROR: {exc}", flush=True)

        try:
            self._open_positions = await self._ig_client.get_positions()
        except Exception as exc:
            print(f"POSITIONS ERROR: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    async def _analyze_instrument(self, epic: str) -> dict[str, Any] | None:
        candles = self._candle_buffer.get_candles(epic, include_open=False)
        if len(candles) < MIN_CANDLES_FOR_STRATEGY:
            return None

        closes = [c["close"] for c in candles]
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]

        sma_fast = sum(closes[-10:]) / 10
        sma_slow = sum(closes[-25:]) / 25
        current  = closes[-1]

        n = min(len(highs), len(lows), len(closes))
        atr_vals = [
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(1, n)
        ]
        atr = sum(atr_vals[-14:]) / min(14, len(atr_vals)) if atr_vals else current * 0.001

        ts = abs(sma_fast - sma_slow) / atr if atr > 0 else 0
        if sma_fast > sma_slow and ts > 0.05:
            direction = "BUY"
        elif sma_fast < sma_slow and ts > 0.05:
            direction = "SELL"
        else:
            print(f"NO SIGNAL {epic}: ts={ts:.4f} fast={sma_fast:.5f} slow={sma_slow:.5f}", flush=True)
            return None

        stop  = round(atr * 1.5, 5)
        limit = round(atr * 2.0, 5)
        conf  = min(90, int(65 + ts * 20))

        print(f"SIGNAL: {direction} {epic} conf={conf} ts={ts:.4f} atr={atr:.5f}", flush=True)
        return {
            "epic": epic, "direction": direction, "current_price": current,
            "stop_distance": stop, "limit_distance": limit,
            "atr": atr, "confidence": conf,
        }

    async def _execute_signal(self, signal: dict[str, Any]) -> None:
        epic      = signal["epic"]
        direction = signal["direction"]
        stop      = signal["stop_distance"]
        limit     = signal["limit_distance"]

        # Use a fixed conservative size for demo trading.
        # IG CFD lots: 1 lot = 1 unit contract. Margin ~£250 for EUR/USD.
        size = 1.0

        print(f"PLACING: {direction} {epic} size={size} stop={stop} limit={limit}", flush=True)
        try:
            result = await self._ig_client.place_order(
                epic=epic, direction=direction, size=size,
                stop_distance=stop, limit_distance=limit,
            )
            status = result.get("dealStatus", "unknown")
            if status == "ACCEPTED":
                self._state.trades_executed += 1
                print(f"TRADE EXECUTED: {direction} {epic} ref={result.get('dealReference')}", flush=True)
            else:
                self._state.trades_rejected += 1
                print(f"TRADE REJECTED: {direction} {epic} reason={result.get('reason')} raw={result}", flush=True)
        except Exception as exc:
            self._state.trades_rejected += 1
            print(f"TRADE ERROR: {epic} {exc}", flush=True)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        uptime = time.time() - self._state.start_time if self._state.running else 0
        return {
            "running": self._state.running,
            "connected": self._state.connected,
            "streaming": "snapshot_polling",
            "uptime_seconds": round(uptime, 1),
            "account_equity": str(self._account_equity),
            "open_positions": len(self._open_positions),
            "signals_generated": self._state.signals_generated,
            "trades_executed": self._state.trades_executed,
            "trades_rejected": self._state.trades_rejected,
            "errors": self._state.errors,
            "last_error": self._state.last_error,
            "last_tick_time": self._state.last_tick_time,
            "instruments": self._instruments,
            "loop_interval_seconds": self._loop_interval,
            "snapshot_interval_seconds": SNAPSHOT_INTERVAL_SECONDS,
            "candle_buffer": self._candle_buffer.get_status(),
            "task_alive": self._task is not None and not self._task.done(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_trading_loop: AutonomousTradingLoop | None = None


def get_trading_loop() -> AutonomousTradingLoop:
    global _trading_loop
    if _trading_loop is None:
        _trading_loop = AutonomousTradingLoop()
    return _trading_loop


def _set_global_loop(loop: AutonomousTradingLoop) -> None:
    global _trading_loop
    _trading_loop = loop
