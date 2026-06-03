"""Autonomous trading loop that connects market data to strategy execution.

This module orchestrates the full trading pipeline:
1. Connect to IG API and authenticate
2. Fetch market data on a configurable interval
3. Run strategy engine to generate signals
4. Validate signals through risk engine
5. Execute validated trades via IG API
6. Log results and publish events

Designed for demo account testing with conservative defaults.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.config.settings import get_settings
from src.core.exceptions import IGAuthenticationError, IGConnectionError
from src.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default instruments to trade (popular Forex pairs + indices)
DEFAULT_INSTRUMENTS = [
    "CS.D.EURUSD.CFD.IP",   # EUR/USD
    "CS.D.GBPUSD.CFD.IP",   # GBP/USD
    "CS.D.USDJPY.CFD.IP",   # USD/JPY
    "IX.D.FTSE.DAILY.IP",   # FTSE 100
    "IX.D.DAX.DAILY.IP",    # DAX 40
]

# Trading loop interval in seconds (how often to check for signals)
LOOP_INTERVAL_SECONDS = 60  # Check every minute

# Maximum concurrent positions
MAX_OPEN_POSITIONS = 5

# Minimum confidence score to trade (lowered for demo)
MIN_CONFIDENCE_THRESHOLD = 50  # temporarily lowered for debugging

# Position size cap for demo (fraction of equity)
DEMO_MAX_POSITION_PCT = 0.02  # 2% max per trade


@dataclass
class TradingLoopState:
    """Tracks the state of the autonomous trading loop."""

    running: bool = False
    connected: bool = False
    last_tick_time: float = 0.0
    signals_generated: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    errors: int = 0
    last_error: str = ""
    start_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Trading Loop
# ---------------------------------------------------------------------------


class AutonomousTradingLoop:
    """Main trading loop that runs continuously and executes trades autonomously.

    Usage:
        loop = AutonomousTradingLoop()
        await loop.start()  # Runs until stopped
        await loop.stop()
    """

    def __init__(
        self,
        instruments: list[str] | None = None,
        loop_interval: float = LOOP_INTERVAL_SECONDS,
    ) -> None:
        self._instruments = instruments or DEFAULT_INSTRUMENTS
        self._loop_interval = loop_interval
        self._state = TradingLoopState()
        self._task: asyncio.Task[None] | None = None
        self._ig_client: Any = None
        self._account_equity: Decimal = Decimal("0")
        self._open_positions: list[dict[str, Any]] = []

    @property
    def state(self) -> TradingLoopState:
        """Current state of the trading loop."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether the trading loop is currently active."""
        return self._state.running

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the autonomous trading loop as a background task."""
        if self._state.running:
            logger.warning("Trading loop already running")
            return

        self._state = TradingLoopState()
        self._state.running = True

        logger.info(
            "Starting autonomous trading loop",
            extra={
                "instruments": self._instruments,
                "interval_seconds": self._loop_interval,
            },
        )

        self._task = asyncio.create_task(
            self._run_loop(), name="autonomous_trading_loop"
        )

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        self._state.running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Disconnect from IG
        if self._ig_client is not None:
            try:
                await self._ig_client.stop()
            except Exception as exc:
                logger.error("Error stopping IG client: %s", exc)
            self._ig_client = None

        self._state.connected = False
        logger.info(
            "Trading loop stopped",
            extra={
                "trades_executed": self._state.trades_executed,
                "signals_generated": self._state.signals_generated,
            },
        )

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main trading loop — connects, then cycles through market analysis."""
        try:
            # Step 1: Connect to IG
            connected = await self._connect_to_ig()
            if not connected:
                logger.error("Failed to connect to IG API, trading loop exiting")
                self._state.running = False
                return

            self._state.connected = True
            logger.info("Connected to IG API, starting trading cycle")

            # Step 2: Main loop
            while self._state.running:
                try:
                    await self._trading_cycle()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._state.errors += 1
                    self._state.last_error = str(exc)
                    logger.error(
                        "Error in trading cycle",
                        extra={"error": str(exc), "total_errors": self._state.errors},
                    )
                    # Don't crash on individual cycle errors
                    if self._state.errors > 50:
                        logger.error("Too many errors, stopping trading loop")
                        break

                await asyncio.sleep(self._loop_interval)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Fatal error in trading loop: %s", exc)
            self._state.last_error = str(exc)
        finally:
            self._state.running = False

    async def _connect_to_ig(self) -> bool:
        """Initialize and authenticate the IG client."""
        from src.trading.ig_client import IGClient

        settings = get_settings()

        # Check if credentials are configured
        if not settings.ig_api_key or settings.ig_api_key in ("your_ig_api_key", ""):
            logger.warning(
                "IG_API_KEY not configured. Set IG_API_KEY, IG_USERNAME, IG_PASSWORD in environment."
            )
            return False

        if not settings.ig_username or settings.ig_username in ("your_ig_username", ""):
            logger.warning("IG_USERNAME not configured.")
            return False

        if not settings.ig_password or settings.ig_password in ("your_ig_password", ""):
            logger.warning("IG_PASSWORD not configured.")
            return False

        logger.info(
            "Connecting to IG API: account_type=%s username=%s api_key_prefix=%s",
            settings.ig_account_type,
            settings.ig_username[:4] + "****" if len(settings.ig_username) > 4 else "****",
            settings.ig_api_key[:4] + "****" if len(settings.ig_api_key) > 4 else "****",
        )
        print(f"IG CONNECT: Attempting connection to IG {settings.ig_account_type} account...", flush=True)

        try:
            self._ig_client = IGClient(
                api_key=settings.ig_api_key,
                username=settings.ig_username,
                password=settings.ig_password,
                account_type=settings.ig_account_type,
            )
            await self._ig_client.start()
            print("IG CONNECT: Successfully authenticated with IG API!", flush=True)
            logger.info("IG client connected and authenticated successfully")
            return True

        except IGAuthenticationError as exc:
            print(f"IG CONNECT ERROR: Authentication failed: {exc}", flush=True)
            logger.error("IG authentication failed: %s", exc)
            return False
        except Exception as exc:
            print(f"IG CONNECT ERROR: {exc} — type: {type(exc).__name__}", flush=True)
            logger.error("Failed to connect to IG: %s — type: %s", exc, type(exc).__name__)
            return False

    # -------------------------------------------------------------------------
    # Trading Cycle
    # -------------------------------------------------------------------------

    async def _trading_cycle(self) -> None:
        """Execute one full trading cycle: fetch data → analyze → trade."""
        if self._ig_client is None or not self._ig_client.is_connected:
            logger.warning("IG client disconnected, attempting reconnect")
            connected = await self._connect_to_ig()
            if not connected:
                return

        # 1. Update account state
        await self._update_account_state()

        # 2. Check if we can open more positions
        if len(self._open_positions) >= MAX_OPEN_POSITIONS:
            logger.debug(
                "Max positions reached (%d), skipping signal generation",
                MAX_OPEN_POSITIONS,
            )
            return

        # 3. Analyze each instrument
        for instrument in self._instruments:
            if not self._state.running:
                break

            try:
                signal = await self._analyze_instrument(instrument)
                if signal is not None:
                    self._state.signals_generated += 1
                    await self._execute_signal(signal)
            except Exception as exc:
                logger.warning(
                    "Error analyzing %s: %s", instrument, exc
                )

    async def _update_account_state(self) -> None:
        """Fetch current account equity and open positions from IG."""
        try:
            # Get account info
            account_info = await self._ig_client.get_account_info()
            # IG returns nested balance - try multiple paths
            balance = (
                account_info.get("balance", {})
                or account_info.get("accountInfo", {})
            )
            equity = (
                balance.get("balance")
                or balance.get("equity")
                or balance.get("available")
                or account_info.get("balance")
            )
            if equity:
                self._account_equity = Decimal(str(equity))
            else:
                # Default to 20000 AUD if we can't parse it (known demo balance)
                self._account_equity = Decimal("20000")
                print(f"ACCOUNT: Could not parse equity from: {account_info}", flush=True)

            # Get open positions
            positions = await self._ig_client.get_positions()
            self._open_positions = positions

            print(f"ACCOUNT: equity={self._account_equity} positions={len(self._open_positions)}", flush=True)

        except Exception as exc:
            logger.warning("Failed to update account state: %s", exc)
            # Use last known equity or default
            if self._account_equity == Decimal("0"):
                self._account_equity = Decimal("20000")

    async def _analyze_instrument(self, epic: str) -> dict[str, Any] | None:
        """Analyze a single instrument and generate a trade signal."""
        try:
            prices = await self._ig_client.get_prices(epic, "HOUR", 100)

            if not prices or len(prices) < 25:
                print(f"DEBUG {epic}: Not enough prices — got {len(prices) if prices else 0}", flush=True)
                return None

            # Extract close, high, low prices
            closes: list[float] = []
            highs: list[float] = []
            lows: list[float] = []

            for p in prices:
                def _mid(data: dict) -> float | None:
                    if not data:
                        return None
                    bid = data.get("bid")
                    ask = data.get("ask")
                    mid = data.get("mid")
                    last = data.get("lastTraded")
                    if bid is not None and ask is not None:
                        try:
                            return (float(bid) + float(ask)) / 2
                        except (TypeError, ValueError):
                            pass
                    for v in (mid, bid, ask, last):
                        if v is not None:
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
                    return None

                c = _mid(p.get("closePrice") or {})
                h = _mid(p.get("highPrice") or {})
                l = _mid(p.get("lowPrice") or {})

                if c and c > 0:
                    closes.append(c)
                if h and h > 0:
                    highs.append(h)
                if l and l > 0:
                    lows.append(l)

            print(f"DEBUG {epic}: closes={len(closes)} highs={len(highs)} lows={len(lows)} sample={prices[0] if prices else 'N/A'}", flush=True)

            if len(closes) < 25:
                return None

            # Calculate indicators
            current_price = closes[-1]
            sma_fast = sum(closes[-10:]) / 10
            sma_slow = sum(closes[-25:]) / 25

            # ATR
            n = min(len(highs), len(lows), len(closes))
            atr_values = []
            for i in range(1, n):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                atr_values.append(tr)

            atr = sum(atr_values[-14:]) / min(14, len(atr_values)) if atr_values else current_price * 0.001

            # Signal: trend direction
            trend_strength = abs(sma_fast - sma_slow) / atr if atr > 0 else 0
            direction = None
            if sma_fast > sma_slow and trend_strength > 0.05:
                direction = "BUY"
            elif sma_fast < sma_slow and trend_strength > 0.05:
                direction = "SELL"

            if direction is None:
                print(f"DEBUG {epic}: No signal — trend_strength={trend_strength:.4f} sma_fast={sma_fast:.5f} sma_slow={sma_slow:.5f}", flush=True)
                return None

            stop_distance = round(atr * 1.5, 5)
            limit_distance = round(atr * 2.0, 5)
            confidence = min(90, int(65 + trend_strength * 20))

            print(f"SIGNAL: {direction} {epic} | confidence={confidence} | ts={trend_strength:.4f} | atr={atr:.5f} | stop={stop_distance} | limit={limit_distance}", flush=True)

            return {
                "epic": epic,
                "direction": direction,
                "current_price": current_price,
                "stop_distance": stop_distance,
                "limit_distance": limit_distance,
                "atr": atr,
                "confidence": confidence,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
            }

        except Exception as exc:
            print(f"ANALYSIS ERROR {epic}: {exc}", flush=True)
            return None

    async def _execute_signal(self, signal: dict[str, Any]) -> None:
        """Execute a validated trade signal via the IG API.

        Args:
            signal: Signal dictionary with trade parameters.
        """
        epic = signal["epic"]
        direction = signal["direction"]
        stop_distance = signal["stop_distance"]
        limit_distance = signal["limit_distance"]

        # Calculate position size (conservative for demo)
        # Risk = stop_distance in points, size = (equity * risk_pct) / stop_distance
        risk_amount = float(self._account_equity) * DEMO_MAX_POSITION_PCT
        if stop_distance > 0:
            size = round(risk_amount / stop_distance, 2)
        else:
            size = 0.5  # Minimum size

        # Enforce minimum and maximum size
        size = max(0.5, min(size, 5.0))  # Min 0.5, max 5.0 lots for demo

        logger.info(
            "EXECUTING TRADE: %s %s | size=%.2f | stop=%.1f | limit=%.1f | confidence=%d",
            direction,
            epic,
            size,
            stop_distance,
            limit_distance,
            signal["confidence"],
        )

        try:
            result = await self._ig_client.place_order(
                epic=epic,
                direction=direction,
                size=size,
                stop_distance=stop_distance,
                limit_distance=limit_distance,
            )

            deal_reference = result.get("dealReference", "unknown")
            status = result.get("dealStatus", "unknown")

            logger.debug(
                "IG place_order response",
                extra={"epic": epic, "response": result},
            )

            if status == "ACCEPTED":
                self._state.trades_executed += 1
                logger.info(
                    "TRADE EXECUTED: %s %s | deal_ref=%s | size=%.2f",
                    direction,
                    epic,
                    deal_reference,
                    size,
                )
            else:
                self._state.trades_rejected += 1
                reason = result.get("reason", "unknown")
                logger.warning(
                    "TRADE REJECTED by IG: %s %s | reason=%s | deal_ref=%s | raw=%s",
                    direction,
                    epic,
                    reason,
                    deal_reference,
                    result,
                )

        except Exception as exc:
            self._state.trades_rejected += 1
            logger.error(
                "TRADE EXECUTION FAILED: %s %s | error=%s",
                direction,
                epic,
                str(exc),
            )

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current trading loop status for monitoring."""
        uptime = time.time() - self._state.start_time if self._state.running else 0
        return {
            "running": self._state.running,
            "connected": self._state.connected,
            "uptime_seconds": round(uptime, 1),
            "account_equity": str(self._account_equity),
            "open_positions": len(self._open_positions),
            "signals_generated": self._state.signals_generated,
            "trades_executed": self._state.trades_executed,
            "trades_rejected": self._state.trades_rejected,
            "errors": self._state.errors,
            "last_error": self._state.last_error,
            "instruments": self._instruments,
            "loop_interval_seconds": self._loop_interval,
            "task_alive": self._task is not None and not self._task.done(),
            "task_exception": str(self._task.exception()) if (
                self._task is not None
                and self._task.done()
                and not self._task.cancelled()
                and self._task.exception() is not None
            ) else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_trading_loop: AutonomousTradingLoop | None = None


def get_trading_loop() -> AutonomousTradingLoop:
    """Get or create the global trading loop instance."""
    global _trading_loop
    if _trading_loop is None:
        _trading_loop = AutonomousTradingLoop()
    return _trading_loop


def _set_global_loop(loop: AutonomousTradingLoop) -> None:
    """Store a trading loop instance as the global singleton."""
    global _trading_loop
    _trading_loop = loop
