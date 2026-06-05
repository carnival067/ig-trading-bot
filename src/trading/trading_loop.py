"""Autonomous trading loop — snapshot polling strategy.

Strategy:
- Polls /markets/{epic} every 90s for live price (no historical data quota)
- Builds 1-minute OHLC candles from ticks
- SMA crossover + ATR signal generation
- Enters with stop loss (1.5×ATR) and take profit (3×ATR) → 1:2 risk/reward
- Exits when signal flips direction (trend reversal)
- IG manages stop/limit automatically — re-enters on next signal after close
- Tracks cumulative P&L across all trades
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
    # Add more once confirmed working on demo:
    # "CS.D.GBPUSD.CFD.IP",
    # "CS.D.USDJPY.CFD.IP",
]

SNAPSHOT_INTERVAL_SECONDS: float = 90.0   # /markets poll frequency
LOOP_INTERVAL_SECONDS: float = 60.0        # strategy cycle frequency
ACCOUNT_REFRESH_INTERVAL_SECONDS: float = 300.0  # /accounts refresh (5 min)

MAX_OPEN_POSITIONS = 3
TRADE_SIZE = 1.0          # fixed 1 lot per trade for demo

# ATR multipliers for stop/limit
SL_MULTIPLIER = 1.5       # stop loss  = 1.5 × ATR
TP_MULTIPLIER = 3.0       # take profit = 3.0 × ATR  → 1:2 risk/reward ratio


@dataclass
class TradingLoopState:
    running: bool = False
    connected: bool = False
    last_tick_time: float = 0.0
    signals_generated: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    trades_closed: int = 0
    total_pnl: float = 0.0
    errors: int = 0
    last_error: str = ""
    start_time: float = field(default_factory=time.time)


class AutonomousTradingLoop:
    """Snapshot-polling trading loop with full entry/exit/profit tracking."""

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
            self._snapshot_task = asyncio.create_task(
                self._snapshot_loop(), name="snapshot_poller"
            )

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
            print(f"FATAL ERROR: {exc}", flush=True)
        finally:
            self._state.running = False

    # -------------------------------------------------------------------------
    # Snapshot Polling
    # -------------------------------------------------------------------------

    async def _snapshot_loop(self) -> None:
        print(f"SNAPSHOT: polling {len(self._instruments)} instruments every {SNAPSHOT_INTERVAL_SECONDS}s", flush=True)
        while self._state.running:
            try:
                await self._poll_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"SNAPSHOT ERROR: {exc}", flush=True)
            await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)

    async def _poll_snapshots(self) -> None:
        if self._ig_client is None or not self._ig_client.is_connected:
            return
        for epic in self._instruments:
            if not self._state.running:
                break
            try:
                details = await self._ig_client.get_market_details(epic)
                snap = details.get("snapshot", {})
                bid = snap.get("bid")
                offer = snap.get("offer")
                if bid is not None and offer is not None:
                    self._candle_buffer.on_tick(epic=epic, bid=float(bid), ask=float(offer))
                    self._state.last_tick_time = time.time()
                    print(
                        f"TICK {epic}: bid={bid} ask={offer} "
                        f"candles={self._candle_buffer.candle_count(epic)}",
                        flush=True,
                    )
                else:
                    print(f"SNAPSHOT {epic}: no bid/ask — {snap}", flush=True)
            except Exception as exc:
                print(f"SNAPSHOT {epic}: {exc}", flush=True)
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
        print(f"IG CONNECT: {settings.ig_account_type}...", flush=True)
        try:
            self._ig_client = IGClient(
                api_key=settings.ig_api_key,
                username=settings.ig_username,
                password=settings.ig_password,
                account_type=settings.ig_account_type,
            )
            await self._ig_client.start()
            print("IG CONNECT: OK", flush=True)
            return True
        except IGAuthenticationError as exc:
            print(f"IG AUTH FAILED: {exc}", flush=True)
            return False
        except Exception as exc:
            print(f"IG CONNECT ERROR: {exc}", flush=True)
            return False

    # -------------------------------------------------------------------------
    # Trading Cycle — Entry + Exit + Re-entry
    # -------------------------------------------------------------------------

    async def _trading_cycle(self) -> None:
        if self._ig_client is None or not self._ig_client.is_connected:
            await self._connect_to_ig()
            return

        await self._update_account_state()

        print(
            f"CYCLE: equity={self._account_equity} positions={len(self._open_positions)} "
            f"executed={self._state.trades_executed} pnl={self._state.total_pnl:+.2f}",
            flush=True,
        )

        for epic in self._instruments:
            if not self._state.running:
                break

            if not self._candle_buffer.is_ready(epic, MIN_CANDLES_FOR_STRATEGY):
                cnt = self._candle_buffer.candle_count(epic)
                print(f"WAITING {epic}: {cnt}/{MIN_CANDLES_FOR_STRATEGY} candles", flush=True)
                continue

            try:
                signal = await self._analyze_instrument(epic)
                open_pos = self._get_open_position(epic)

                if open_pos is not None:
                    # --- Position management ---
                    pos_data = open_pos.get("position", open_pos)
                    pos_direction = pos_data.get("direction", "")
                    unrealised = pos_data.get("upl") or pos_data.get("unrealisedProfit") or 0.0

                    print(
                        f"MANAGE {epic}: direction={pos_direction} "
                        f"unrealised_pnl={unrealised}",
                        flush=True,
                    )

                    # Exit if signal flipped — trend reversal
                    if signal is not None and pos_direction and signal["direction"] != pos_direction:
                        print(f"EXIT SIGNAL FLIP: {epic} was {pos_direction} now {signal['direction']}", flush=True)
                        closed = await self._close_position(open_pos, epic)
                        if closed:
                            # Immediately re-enter in new direction
                            self._state.signals_generated += 1
                            await self._execute_signal(signal)
                    # else: IG manages SL/TP automatically — hold

                elif signal is not None and len(self._open_positions) < MAX_OPEN_POSITIONS:
                    # --- Entry ---
                    self._state.signals_generated += 1
                    await self._execute_signal(signal)

            except Exception as exc:
                print(f"CYCLE ERROR {epic}: {exc}", flush=True)

    def _get_open_position(self, epic: str) -> dict[str, Any] | None:
        for pos in self._open_positions:
            market = pos.get("market", {})
            pos_epic = market.get("epic") or pos.get("epic", "")
            if pos_epic == epic:
                return pos
        return None

    async def _close_position(self, position: dict[str, Any], epic: str) -> bool:
        """Close a position. Returns True if successful."""
        try:
            pos_data = position.get("position", position)
            deal_id = pos_data.get("dealId") or pos_data.get("deal_id", "")
            direction = pos_data.get("direction", "BUY")
            size = float(pos_data.get("size") or pos_data.get("dealSize") or TRADE_SIZE)
            close_dir = "SELL" if direction == "BUY" else "BUY"

            print(f"CLOSING {epic}: {deal_id} size={size} close_dir={close_dir}", flush=True)
            result = await self._ig_client.close_position(deal_id, close_dir, size)
            status = result.get("dealStatus", "unknown")
            pnl = result.get("profit") or 0.0
            if pnl:
                try:
                    self._state.total_pnl += float(pnl)
                except (TypeError, ValueError):
                    pass
            self._state.trades_closed += 1
            print(
                f"CLOSED {epic}: status={status} reason={result.get('reason')} "
                f"pnl={pnl} total_pnl={self._state.total_pnl:+.2f}",
                flush=True,
            )
            return status == "ACCEPTED"
        except Exception as exc:
            print(f"CLOSE ERROR {epic}: {exc}", flush=True)
            return False

    # -------------------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------------------

    async def _update_account_state(self) -> None:
        now = time.time()
        # Always refresh positions (needed for exit logic)
        try:
            self._open_positions = await self._ig_client.get_positions()
        except Exception as exc:
            print(f"POSITIONS ERROR: {exc}", flush=True)

        # Only refresh equity every 5 minutes
        if now - self._last_account_refresh < ACCOUNT_REFRESH_INTERVAL_SECONDS:
            return
        self._last_account_refresh = now

        try:
            info = await self._ig_client.get_account_info()
            balance = info.get("balance", {}) or {}
            equity = balance.get("balance") or balance.get("equity") or balance.get("available")
            if equity:
                self._account_equity = Decimal(str(equity))
                print(f"ACCOUNT: equity={self._account_equity} AUD", flush=True)
        except Exception as exc:
            print(f"ACCOUNT ERROR: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # Strategy — SMA Crossover + ATR
    # -------------------------------------------------------------------------

    async def _analyze_instrument(self, epic: str) -> dict[str, Any] | None:
        candles = self._candle_buffer.get_candles(epic, include_open=False)
        if len(candles) < MIN_CANDLES_FOR_STRATEGY:
            return None

        closes = [c["close"] for c in candles]
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]

        # Need at least 25 for slow SMA; if not enough use what we have
        slow_period = min(25, len(closes))
        fast_period = min(10, len(closes))

        sma_fast = sum(closes[-fast_period:]) / fast_period
        sma_slow = sum(closes[-slow_period:]) / slow_period
        current  = closes[-1]

        n = min(len(highs), len(lows), len(closes))
        atr_vals = [
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(1, n)
        ]
        atr = sum(atr_vals[-14:]) / min(14, len(atr_vals)) if atr_vals else current * 0.0005

        # Minimum ATR floor to ensure meaningful stop distances
        min_atr = current * 0.0003  # 3 pips for EUR/USD
        atr = max(atr, min_atr)

        ts = abs(sma_fast - sma_slow) / atr if atr > 0 else 0

        if sma_fast > sma_slow and ts > 0.05:
            direction = "BUY"
        elif sma_fast < sma_slow and ts > 0.05:
            direction = "SELL"
        else:
            print(f"NO SIGNAL {epic}: ts={ts:.4f} fast={sma_fast:.6f} slow={sma_slow:.6f}", flush=True)
            return None

        stop_dist  = round(atr * SL_MULTIPLIER, 6)   # 1.5×ATR stop loss
        limit_dist = round(atr * TP_MULTIPLIER, 6)   # 3.0×ATR take profit → 1:2 R:R
        conf = min(95, int(60 + ts * 25))

        print(
            f"SIGNAL: {direction} {epic} | conf={conf} | ts={ts:.4f} | "
            f"atr={atr:.6f} | SL={stop_dist:.6f} | TP={limit_dist:.6f} | "
            f"R:R=1:{TP_MULTIPLIER/SL_MULTIPLIER:.1f}",
            flush=True,
        )
        return {
            "epic": epic,
            "direction": direction,
            "current_price": current,
            "stop_distance": stop_dist,
            "limit_distance": limit_dist,
            "atr": atr,
            "confidence": conf,
            "rr_ratio": round(TP_MULTIPLIER / SL_MULTIPLIER, 2),
        }

    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------

    async def _execute_signal(self, signal: dict[str, Any]) -> None:
        epic      = signal["epic"]
        direction = signal["direction"]
        stop      = signal["stop_distance"]
        limit     = signal["limit_distance"]

        print(
            f"PLACING: {direction} {epic} size={TRADE_SIZE} "
            f"SL={stop:.6f} TP={limit:.6f} R:R=1:{signal.get('rr_ratio', 2)}",
            flush=True,
        )
        try:
            # Step 1: Place the market order (no SL/TP — avoids validation errors)
            result = await self._ig_client.place_order(
                epic=epic,
                direction=direction,
                size=TRADE_SIZE,
                stop_distance=None,
                limit_distance=None,
            )
            status = result.get("dealStatus", "unknown")

            if status == "ACCEPTED":
                self._state.trades_executed += 1
                deal_id   = result.get("dealId", "")
                entry_lvl = result.get("level")
                print(
                    f"✅ TRADE OPENED: {direction} {epic} | "
                    f"level={entry_lvl} | dealId={deal_id}",
                    flush=True,
                )

                # Step 2: Add SL/TP via position update (separate call, always works)
                if deal_id and entry_lvl:
                    try:
                        price = float(entry_lvl)
                        if direction == "BUY":
                            sl_level = round(price - stop,  5)
                            tp_level = round(price + limit, 5)
                        else:
                            sl_level = round(price + stop,  5)
                            tp_level = round(price - limit, 5)

                        await self._ig_client.update_position_sl_tp(
                            deal_id=deal_id,
                            stop_level=sl_level,
                            limit_level=tp_level,
                        )
                        print(
                            f"✅ SL/TP SET: {epic} SL={sl_level} TP={tp_level} "
                            f"R:R=1:{signal.get('rr_ratio', 2)}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"SL/TP UPDATE FAILED {epic}: {exc} — position open without SL/TP", flush=True)

            else:
                self._state.trades_rejected += 1
                print(
                    f"❌ TRADE REJECTED: {direction} {epic} | "
                    f"reason={result.get('reason')} | raw={result}",
                    flush=True,
                )
        except Exception as exc:
            self._state.trades_rejected += 1
            print(f"TRADE ERROR {epic}: {exc}", flush=True)

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
            "trades_closed": self._state.trades_closed,
            "total_pnl": round(self._state.total_pnl, 2),
            "errors": self._state.errors,
            "last_error": self._state.last_error,
            "last_tick_time": self._state.last_tick_time,
            "instruments": self._instruments,
            "strategy": {
                "type": "SMA_crossover_ATR",
                "sl_multiplier": SL_MULTIPLIER,
                "tp_multiplier": TP_MULTIPLIER,
                "risk_reward": f"1:{TP_MULTIPLIER / SL_MULTIPLIER:.1f}",
                "trade_size_lots": TRADE_SIZE,
            },
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
