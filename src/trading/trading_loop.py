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
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

import pandas as pd

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
    "CS.D.GBPUSD.CFD.IP",   # GBP/USD
    "CS.D.USDJPY.CFD.IP",   # USD/JPY
    "CS.D.AUDUSD.CFD.IP",   # AUD/USD
    "CS.D.USDCAD.CFD.IP",   # USD/CAD
]

SNAPSHOT_INTERVAL_SECONDS: float = 90.0   # /markets poll frequency
LOOP_INTERVAL_SECONDS: float = 60.0        # strategy cycle frequency
ACCOUNT_REFRESH_INTERVAL_SECONDS: float = 300.0  # /accounts refresh (5 min)

MAX_OPEN_POSITIONS = 1            # professional strategy proof phase
TRADE_SIZE = 1.0                  # fixed 1 lot per trade for demo
MAX_DAILY_TRADES = 3              # proof-phase opportunity ceiling
MIN_SECONDS_BETWEEN_TRADES_PER_INSTRUMENT = 1800  # 30-minute pair cooldown
MAX_SPREAD_POINTS = 3.0           # skip wide spreads, in pips/points after FX scaling
MIN_ATR_PCT = 0.0002              # skip dead markets
MAX_ATR_PCT = 0.0030              # skip extreme volatility
PROFESSIONAL_DAILY_MAX_LOSS_PCT = 0.01

# ATR multipliers — wider stops reduce noise-triggered SL hits
SL_MULTIPLIER = 3.0       # stop loss  = 3.0 × ATR  (wider, survives noise)
TP_MULTIPLIER = 6.0       # take profit = 6.0 × ATR  → 1:2 risk/reward ratio

# Minimum trend strength to enter — filters out weak signals
MIN_TREND_STRENGTH = 0.3  # was 0.05 — much stricter filter

# Minimum candles before trading — more data = better signal quality
MIN_CANDLES_FOR_ENTRY = 25  # need 25 closed candles (was 5)

# Only trade during London/NY session (8am-6pm UTC) — best liquidity
TRADING_HOURS_UTC = (8, 18)  # (start_hour, end_hour)


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


@dataclass
class LiquiditySweepState:
    state: str = "WAIT_BREAK"
    day: str | None = None
    last_candle_time: float | None = None
    signal_high: float | None = None
    signal_low: float | None = None
    sl_level: float | None = None
    tp_level: float | None = None


class AutonomousTradingLoop:
    """Snapshot-polling trading loop with full entry/exit/profit tracking."""

    def __init__(
        self,
        instruments: list[str] | None = None,
        loop_interval: float = LOOP_INTERVAL_SECONDS,
        risk_engine: Any | None = None,
        mistake_analyzer: Any | None = None,
        strategy_mode: str = "PROFESSIONAL",
        account_type: str = "DEMO",
        professional_live_approved: bool = False,
        professional_demo_forward_approved: bool = False,
        news_filter_mode: str = "FAIL_CLOSED",
        news_event_provider: Callable[[str, datetime], list[Any] | None] | None = None,
        news_safety_layer: Any | None = None,
    ) -> None:
        self._instruments = instruments or DEFAULT_INSTRUMENTS
        self._loop_interval = loop_interval
        self._state = TradingLoopState()
        self._task: asyncio.Task[None] | None = None
        self._snapshot_task: asyncio.Task[None] | None = None
        self._ig_client: Any = None
        self._candle_buffer = CandleBuffer(candle_period_seconds=60, max_candles=5000)
        self._account_equity: Decimal = Decimal("20000")
        self._open_positions: list[dict[str, Any]] = []
        self._last_account_refresh: float = 0.0
        self._risk_engine = risk_engine or self._build_default_risk_engine()
        self._mistake_analyzer = mistake_analyzer
        self._last_risk_decision: dict[str, Any] | None = None
        self._last_news_decision: dict[str, Any] | None = None
        self._recent_trade_events: deque[dict[str, Any]] = deque(maxlen=25)
        self._last_snapshots: dict[str, dict[str, Any]] = {}
        self._last_trade_time_by_epic: dict[str, float] = {}
        self._daily_trade_date: str = datetime.now(timezone.utc).date().isoformat()
        self._daily_trade_count: int = 0
        self._daily_realized_pnl: float = 0.0
        self._deal_records: dict[str, dict[str, str]] = {}
        self._missing_position_counts: dict[str, int] = {}
        self._strategy_mode = strategy_mode.upper()
        self._account_type = account_type.upper()
        self._professional_live_approved = professional_live_approved
        self._professional_demo_forward_approved = professional_demo_forward_approved
        self._news_filter_mode = news_filter_mode.upper()
        self._news_event_provider = news_event_provider
        self._news_safety_layer = news_safety_layer
        self._professional_positions: dict[str, dict[str, Any]] = {}
        self._liquidity_sweep_state: dict[str, LiquiditySweepState] = {}
        from src.strategy.professional import ProfessionalICTStrategy, ProfessionalStrategyConfig
        from src.strategy.strategies.legacy_sma import LegacySMAStrategy

        self._professional_strategy = ProfessionalICTStrategy(
            ProfessionalStrategyConfig(
                execution_mode=self._account_type,
                news_filter_mode=news_filter_mode,
            )
        )
        self._legacy_strategy = LegacySMAStrategy()
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
                    bid_float = float(bid)
                    offer_float = float(offer)
                    self._last_snapshots[epic] = {
                        "bid": bid_float,
                        "offer": offer_float,
                        "market_status": snap.get("marketStatus"),
                        "timestamp": time.time(),
                    }
                    self._candle_buffer.on_tick(epic=epic, bid=bid_float, ask=offer_float)
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
                    await self._manage_professional_position(open_pos, epic)

                    # Exit if signal flipped — trend reversal
                    if signal is not None and pos_direction and signal["direction"] != pos_direction:
                        print(f"EXIT SIGNAL FLIP: {epic} was {pos_direction} now {signal['direction']}", flush=True)
                        closed = await self._close_position(open_pos, epic)
                        if closed:
                            # Immediately re-enter in new direction
                            gate_reason = self._entry_gate_rejection_reason(signal)
                            if gate_reason is not None:
                                self._state.trades_rejected += 1
                                self._record_trade_event(
                                    "signal_rejected",
                                    epic=epic,
                                    direction=signal.get("direction"),
                                    reason=gate_reason,
                                )
                                continue
                            validated_signal = await self._apply_risk_controls(signal)
                            if validated_signal is not None:
                                news_checked = await self._apply_news_safety(validated_signal)
                                if news_checked is not None:
                                    self._state.signals_generated += 1
                                    await self._execute_signal(news_checked)
                    # else: IG manages SL/TP automatically — hold

                elif signal is not None and len(self._open_positions) < MAX_OPEN_POSITIONS:
                    # --- Entry ---
                    gate_reason = self._entry_gate_rejection_reason(signal)
                    if gate_reason is not None:
                        self._state.trades_rejected += 1
                        self._record_trade_event(
                            "signal_rejected",
                            epic=epic,
                            direction=signal.get("direction"),
                            confidence=signal.get("confidence"),
                            reason=gate_reason,
                        )
                        print(f"ENTRY GATE REJECTED {epic}: {gate_reason}", flush=True)
                        continue

                    validated_signal = await self._apply_risk_controls(signal)
                    if validated_signal is not None:
                        validated_signal = await self._apply_news_safety(validated_signal)
                    if validated_signal is not None:
                        self._state.signals_generated += 1
                        self._record_trade_event(
                            "signal_approved",
                            epic=epic,
                            direction=validated_signal.get("direction"),
                            confidence=validated_signal.get("confidence"),
                            reason="strategy_signal_passed_risk",
                        )
                        await self._execute_signal(validated_signal)

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
                    self._daily_realized_pnl += float(pnl)
                except (TypeError, ValueError):
                    pass
            self._state.trades_closed += 1
            await self._persist_closed_trade(deal_id, epic, result)
            self._record_trade_event(
                "trade_closed",
                epic=epic,
                deal_id=deal_id,
                status=status,
                pnl=pnl,
                reason=result.get("reason"),
            )
            print(
                f"CLOSED {epic}: status={status} reason={result.get('reason')} "
                f"pnl={pnl} total_pnl={self._state.total_pnl:+.2f}",
                flush=True,
            )
            return status == "ACCEPTED"
        except Exception as exc:
            print(f"CLOSE ERROR {epic}: {exc}", flush=True)
            return False

    async def _manage_professional_position(
        self,
        position: dict[str, Any],
        epic: str,
    ) -> None:
        """Take 50% at 1R and move the remaining broker stop to breakeven."""
        pos_data = position.get("position", position)
        deal_id = str(pos_data.get("dealId") or pos_data.get("deal_id") or "")
        management = self._professional_positions.get(deal_id)
        if not deal_id or management is None or management.get("partial_taken"):
            return
        snapshot = self._last_snapshots.get(epic, {})
        bid = snapshot.get("bid")
        offer = snapshot.get("offer")
        if bid is None or offer is None:
            return
        direction = str(pos_data.get("direction") or management["direction"])
        current = float(bid if direction == "BUY" else offer)
        tp1 = float(management["tp1_level"])
        reached = current >= tp1 if direction == "BUY" else current <= tp1
        if not reached:
            return
        total_size = float(pos_data.get("size") or pos_data.get("dealSize") or management["size"])
        close_size = round(total_size * float(management["partial_close_fraction"]), 2)
        if close_size < 0.01 or total_size - close_size < 0.01:
            self._record_trade_event(
                "partial_tp_skipped",
                epic=epic,
                deal_id=deal_id,
                reason="position_too_small_for_partial_close",
            )
            management["partial_taken"] = True
            return
        close_direction = "SELL" if direction == "BUY" else "BUY"
        result = await self._ig_client.close_position(deal_id, close_direction, close_size)
        if result.get("dealStatus") != "ACCEPTED":
            self._record_trade_event(
                "partial_tp_failed",
                epic=epic,
                deal_id=deal_id,
                reason=result.get("reason") or "broker_rejected_partial_close",
            )
            return
        entry_level = float(management["entry_level"])
        final_target = float(management["final_target_level"])
        await self._ig_client.update_position_sl_tp(
            deal_id=deal_id,
            stop_level=entry_level,
            limit_level=final_target,
        )
        management["partial_taken"] = True
        self._record_trade_event(
            "partial_tp_taken",
            epic=epic,
            deal_id=deal_id,
            close_size=close_size,
            tp1_level=tp1,
            breakeven_stop=entry_level,
            remaining_target=final_target,
        )

    # -------------------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------------------

    async def _update_account_state(self) -> None:
        now = time.time()
        # Always refresh positions (needed for exit logic)
        try:
            positions = await self._ig_client.get_positions()
            self._open_positions = positions
            await self._reconcile_broker_closed_positions(positions)
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

    async def _reconcile_broker_closed_positions(
        self,
        live_positions: list[dict[str, Any]],
    ) -> None:
        """Persist positions closed by broker-managed stops, limits, or expiry."""
        from src.db.database import get_session
        from src.db.repositories.trade_repo import TradeRepository

        live_deal_ids = {
            str(deal_id)
            for position in live_positions
            if (
                deal_id := (
                    position.get("position", position).get("dealId")
                    or position.get("position", position).get("deal_id")
                )
            )
        }

        async with get_session() as session:
            persisted_positions = await TradeRepository(session).get_open_positions()

        missing_positions = []
        persisted_deal_ids = {
            str(position.ig_deal_id)
            for position in persisted_positions
            if position.ig_deal_id
        }
        for deal_id in list(self._missing_position_counts):
            if deal_id in live_deal_ids or deal_id not in persisted_deal_ids:
                self._missing_position_counts.pop(deal_id, None)

        for position in persisted_positions:
            if not position.ig_deal_id:
                continue
            deal_id = str(position.ig_deal_id)
            if deal_id in live_deal_ids:
                self._missing_position_counts.pop(deal_id, None)
                continue
            missing_count = self._missing_position_counts.get(deal_id, 0) + 1
            self._missing_position_counts[deal_id] = missing_count
            if missing_count >= 2:
                missing_positions.append(position)

        if not missing_positions:
            return

        transactions = await self._ig_client.get_transaction_history(
            max_span_seconds=86400,
            page_size=200,
        )
        transactions_by_reference = {
            str(transaction.get("reference")): transaction
            for transaction in transactions
            if transaction.get("reference")
        }

        for position in missing_positions:
            deal_id = str(position.ig_deal_id)
            transaction = transactions_by_reference.get(deal_id)
            if transaction is None:
                print(
                    f"RECONCILE WAITING {position.instrument}: no transaction for {deal_id}",
                    flush=True,
                )
                continue

            pnl = self._parse_ig_decimal(transaction.get("profitAndLoss"))
            close_level = self._parse_ig_decimal(transaction.get("closeLevel"))
            close_result = {
                "dealStatus": "ACCEPTED",
                "profit": str(pnl),
                "closeLevel": str(close_level) if close_level is not None else None,
                "reason": "broker_managed_close",
                "transactionType": transaction.get("transactionType"),
            }
            await self._persist_closed_trade(deal_id, position.instrument, close_result)
            self._state.trades_closed += 1
            self._state.total_pnl += float(pnl)
            self._daily_realized_pnl += float(pnl)
            self._missing_position_counts.pop(deal_id, None)
            print(
                f"RECONCILED CLOSE {position.instrument}: deal_id={deal_id} pnl={pnl}",
                flush=True,
            )

    @staticmethod
    def _parse_ig_decimal(value: Any) -> Decimal:
        """Parse IG numeric strings that may contain currency text or separators."""
        if value is None:
            return Decimal("0")
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int | float):
            return Decimal(str(value))

        text = str(value).strip().replace(",", "")
        accounting_negative = text.startswith("(") and text.endswith(")")
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match is None:
            return Decimal("0")
        parsed = Decimal(match.group(0))
        signed_negative = "-" in text[: match.start()]
        return -abs(parsed) if accounting_negative or signed_negative else parsed

    # -------------------------------------------------------------------------
    # Strategy — SMA Crossover + ATR
    # -------------------------------------------------------------------------

    async def _analyze_instrument(self, epic: str) -> dict[str, Any] | None:
        if self._strategy_mode == "LEGACY_SMA":
            if self._account_type == "LIVE":
                self._record_trade_event(
                    "signal_rejected",
                    epic=epic,
                    reason="legacy_sma_is_never_authorized_for_live",
                )
                return None
            return await self._analyze_legacy_sma(epic)
        if self._strategy_mode == "LIQUIDITY_SWEEP_1M":
            return await self._analyze_liquidity_sweep_1m(epic)
        if self._strategy_mode not in {"PROFESSIONAL", "GUARDED_AUTO"}:
            self._record_trade_event(
                "signal_rejected",
                epic=epic,
                reason=f"unknown_strategy_mode:{self._strategy_mode}",
            )
            return None
        if self._account_type == "LIVE" and not self._professional_live_approved:
            self._record_trade_event(
                "signal_rejected",
                epic=epic,
                reason="professional_strategy_not_approved_for_live",
            )
            return None
        if self._account_type == "DEMO" and not self._professional_demo_forward_approved:
            self._record_trade_event(
                "signal_rejected",
                epic=epic,
                reason="professional_strategy_not_approved_for_demo_forward_test",
            )
            return None
        return await self._analyze_professional(epic)

    async def _analyze_legacy_sma(self, epic: str) -> dict[str, Any] | None:
        candles = self._candle_buffer.get_candles(epic, include_open=False)
        if len(candles) < MIN_CANDLES_FOR_ENTRY:
            print(f"WAITING {epic}: need {MIN_CANDLES_FOR_ENTRY} candles, have {len(candles)}", flush=True)
            return None

        # Market hours filter — only trade during London/NY session
        from datetime import datetime, timezone
        utc_hour = datetime.now(timezone.utc).hour
        if not (TRADING_HOURS_UTC[0] <= utc_hour < TRADING_HOURS_UTC[1]):
            print(f"MARKET HOURS: {utc_hour}:00 UTC outside trading window {TRADING_HOURS_UTC[0]}-{TRADING_HOURS_UTC[1]} UTC", flush=True)
            return None

        closes = [c["close"] for c in candles]
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]

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

        # Minimum ATR floor — ensures stop is meaningful vs spread
        min_atr = current * 0.0005  # 5 pips minimum ATR
        atr = max(atr, min_atr)

        ts = abs(sma_fast - sma_slow) / atr if atr > 0 else 0

        # Strict trend strength filter — only trade strong, clear trends
        if ts < MIN_TREND_STRENGTH:
            print(f"WEAK SIGNAL {epic}: ts={ts:.4f} < {MIN_TREND_STRENGTH} (need stronger trend)", flush=True)
            return None

        if sma_fast > sma_slow:
            direction = "BUY"
        elif sma_fast < sma_slow:
            direction = "SELL"
        else:
            return None

        stop_dist  = round(atr * SL_MULTIPLIER, 6)
        limit_dist = round(atr * TP_MULTIPLIER, 6)
        conf = min(95, int(50 + ts * 20))

        print(
            f"SIGNAL: {direction} {epic} | conf={conf} | ts={ts:.4f} | "
            f"atr={atr:.6f} | SL={stop_dist:.6f} | TP={limit_dist:.6f} | "
            f"R:R=1:{TP_MULTIPLIER/SL_MULTIPLIER:.1f} | hour={utc_hour}UTC",
            flush=True,
        )
        return {
            "epic": epic,
            "direction": direction,
            "current_price": current,
            "stop_distance": stop_dist,
            "limit_distance": limit_dist,
            "atr": atr,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "trend_strength": ts,
            "confidence": conf,
            "rr_ratio": round(TP_MULTIPLIER / SL_MULTIPLIER, 2),
            "strategy_name": "legacy_sma_atr",
            "risk_per_trade": 0.01,
        }

    async def _analyze_liquidity_sweep_1m(self, epic: str) -> dict[str, Any] | None:
        candles = self._candle_buffer.get_candles(epic, include_open=False)
        if len(candles) < MIN_CANDLES_FOR_STRATEGY:
            self._record_trade_event(
                "liquidity_sweep_decision",
                epic=epic,
                direction="SKIP",
                reason="insufficient_1m_data",
            )
            return None

        snapshot = self._last_snapshots.get(epic, {})
        bid = snapshot.get("bid")
        offer = snapshot.get("offer")
        if bid is None or offer is None:
            self._record_trade_event(
                "liquidity_sweep_decision",
                epic=epic,
                direction="SKIP",
                reason="no_recent_spread_snapshot",
            )
            return None

        indexed = [
            {
                **candle,
                "dt": datetime.fromtimestamp(float(candle["time"]), tz=timezone.utc),
            }
            for candle in candles
        ]
        latest = indexed[-1]
        latest_time = float(latest["time"])
        current_day = latest["dt"].date()
        previous_days = sorted({item["dt"].date() for item in indexed if item["dt"].date() < current_day})
        if not previous_days:
            self._record_trade_event(
                "liquidity_sweep_decision",
                epic=epic,
                direction="SKIP",
                reason="missing_previous_daily_range",
            )
            return None
        previous_day = previous_days[-1]
        previous_day_candles = [item for item in indexed if item["dt"].date() == previous_day]
        daily_high = max(float(item["high"]) for item in previous_day_candles)
        daily_low = min(float(item["low"]) for item in previous_day_candles)

        state = self._liquidity_sweep_state.setdefault(epic, LiquiditySweepState())
        if state.day != current_day.isoformat():
            state.state = "WAIT_BREAK"
            state.day = current_day.isoformat()
            state.last_candle_time = None
            state.signal_high = None
            state.signal_low = None
            state.sl_level = None
            state.tp_level = None

        pending = [
            item
            for item in indexed
            if item["dt"].date() == current_day
            and (
                state.last_candle_time is None
                or float(item["time"]) > state.last_candle_time
            )
        ]

        signal: dict[str, Any] | None = None
        for candle in pending:
            candle_time = float(candle["time"])
            is_latest_candle = candle_time == latest_time
            close = float(candle["close"])
            open_ = float(candle["open"])
            high = float(candle["high"])
            low = float(candle["low"])
            is_green = close > open_
            is_red = close < open_

            if state.state == "WAIT_BREAK":
                if close < daily_low:
                    state.state = "WAIT_BUY_SIGNAL"
                elif close > daily_high:
                    state.state = "WAIT_SELL_SIGNAL"

            if state.state == "WAIT_BUY_SIGNAL" and is_green:
                state.state = "BUY_READY"
                state.signal_high = high
                state.sl_level = low
                state.tp_level = daily_high

            if state.state == "WAIT_SELL_SIGNAL" and is_red:
                state.state = "SELL_READY"
                state.signal_low = low
                state.sl_level = high
                state.tp_level = daily_low

            if state.state == "BUY_READY":
                if state.signal_high is not None and high > state.signal_high:
                    if is_latest_candle:
                        signal = self._build_liquidity_sweep_signal(
                            epic=epic,
                            direction="BUY",
                            current_price=(float(bid) + float(offer)) / 2,
                            sl_level=state.sl_level,
                            tp_level=state.tp_level,
                            candles=candles,
                            daily_high=daily_high,
                            daily_low=daily_low,
                        )
                    state.state = "WAIT_BREAK"
                    state.signal_high = None
                    state.sl_level = None
                    state.tp_level = None
                elif is_red:
                    state.state = "WAIT_BUY_SIGNAL"

            if state.state == "SELL_READY":
                if state.signal_low is not None and low < state.signal_low:
                    if is_latest_candle:
                        signal = self._build_liquidity_sweep_signal(
                            epic=epic,
                            direction="SELL",
                            current_price=(float(bid) + float(offer)) / 2,
                            sl_level=state.sl_level,
                            tp_level=state.tp_level,
                            candles=candles,
                            daily_high=daily_high,
                            daily_low=daily_low,
                        )
                    state.state = "WAIT_BREAK"
                    state.signal_low = None
                    state.sl_level = None
                    state.tp_level = None
                elif is_green:
                    state.state = "WAIT_SELL_SIGNAL"

            state.last_candle_time = candle_time

        if signal is None:
            self._record_trade_event(
                "liquidity_sweep_decision",
                epic=epic,
                direction="SKIP",
                reason=state.state.lower(),
                previous_daily_high=daily_high,
                previous_daily_low=daily_low,
            )
            return None

        self._record_trade_event(
            "liquidity_sweep_decision",
            epic=epic,
            direction=signal["direction"],
            reason="entry_triggered",
            previous_daily_high=daily_high,
            previous_daily_low=daily_low,
        )
        return signal

    def _build_liquidity_sweep_signal(
        self,
        *,
        epic: str,
        direction: str,
        current_price: float,
        sl_level: float | None,
        tp_level: float | None,
        candles: list[dict[str, Any]],
        daily_high: float,
        daily_low: float,
    ) -> dict[str, Any] | None:
        if sl_level is None or tp_level is None or current_price <= 0:
            return None
        if direction == "BUY":
            stop_distance = current_price - float(sl_level)
            limit_distance = float(tp_level) - current_price
        else:
            stop_distance = float(sl_level) - current_price
            limit_distance = current_price - float(tp_level)
        if stop_distance <= 0 or limit_distance <= 0:
            self._record_trade_event(
                "liquidity_sweep_decision",
                epic=epic,
                direction="SKIP",
                reason="invalid_sl_tp_distance",
                current_price=current_price,
                sl_level=sl_level,
                tp_level=tp_level,
            )
            return None

        atr = self._calculate_atr(candles, fallback=current_price * 0.0005)
        return {
            "epic": epic,
            "direction": direction,
            "current_price": current_price,
            "stop_distance": round(stop_distance, 6),
            "limit_distance": round(limit_distance, 6),
            "atr": atr,
            "confidence": 65,
            "rr_ratio": round(limit_distance / stop_distance, 2),
            "risk_per_trade": 0.002,
            "strategy_name": "liquidity_sweep_1m",
            "previous_daily_high": daily_high,
            "previous_daily_low": daily_low,
            "sl_level": sl_level,
            "tp_level": tp_level,
        }

    @staticmethod
    def _calculate_atr(candles: list[dict[str, Any]], *, fallback: float) -> float:
        if len(candles) < 2:
            return fallback
        highs = [float(candle["high"]) for candle in candles]
        lows = [float(candle["low"]) for candle in candles]
        closes = [float(candle["close"]) for candle in candles]
        true_ranges = [
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
            for index in range(1, len(candles))
        ]
        if not true_ranges:
            return fallback
        sample = true_ranges[-14:]
        return max(sum(sample) / len(sample), fallback)

    async def _analyze_professional(self, epic: str) -> dict[str, Any] | None:
        candles = self._candle_buffer.get_candles(epic, include_open=False)
        one_minute = self._candles_to_frame(candles)
        if one_minute.empty:
            return None
        five_minute = self._resample_candles(one_minute, "5min")
        one_hour = self._resample_candles(one_minute, "1h")
        four_hour = self._resample_candles(one_minute, "4h")
        snapshot = self._last_snapshots.get(epic, {})
        bid = snapshot.get("bid")
        offer = snapshot.get("offer")
        if bid is None or offer is None:
            self._record_trade_event("strategy_skip", epic=epic, reason="no_recent_spread_snapshot")
            return None
        pair = self._currency_pair_for_epic(epic) or epic
        timestamp = datetime.now(timezone.utc)
        if self._news_safety_layer is not None:
            from src.strategy.professional.news_filter import NewsEvent

            calendar_events = await self._news_safety_layer.calendar_events(pair, timestamp)
            events = (
                [
                    NewsEvent(
                        timestamp=event.timestamp,
                        currencies=(event.currency,),
                        impact=event.impact,
                        title=event.title,
                    )
                    for event in calendar_events
                ]
                if calendar_events is not None
                else None
            )
        else:
            events = (
                self._news_event_provider(pair, timestamp)
                if self._news_event_provider is not None
                else None
            )
        decision = self._professional_strategy.evaluate(
            pair=pair,
            one_minute=one_minute,
            five_minute=five_minute,
            one_hour=one_hour,
            four_hour=four_hour,
            spread=abs(float(offer) - float(bid)),
            timestamp=timestamp,
            news_events=events,
        )
        self._record_trade_event(
            "professional_strategy_decision",
            epic=epic,
            direction=decision.action,
            reason=decision.reason,
            trend_bias=decision.trend_bias,
            trend_timeframe=decision.trend_timeframe,
            liquidity_sweep=decision.liquidity_sweep,
            structure_event=decision.structure_event,
            zone_type=decision.zone_type,
            spread_atr_ratio=decision.spread_atr_ratio,
            news_status=decision.news_status,
        )
        if not decision.should_trade:
            return None
        assert decision.entry_price is not None
        assert decision.stop_price is not None
        assert decision.target_price is not None
        assert decision.risk_distance is not None
        return {
            "epic": epic,
            "direction": decision.action,
            "current_price": decision.entry_price,
            "stop_distance": abs(decision.entry_price - decision.stop_price),
            "limit_distance": abs(decision.target_price - decision.entry_price),
            "tp1_distance": abs((decision.tp1_price or decision.entry_price) - decision.entry_price),
            "atr": float(decision.diagnostics["atr"]),
            "confidence": self._professional_strategy.config.confidence,
            "rr_ratio": abs(decision.target_price - decision.entry_price)
            / decision.risk_distance,
            "risk_per_trade": decision.risk_per_trade,
            "strategy_name": "professional_ict",
            "regime": decision.trend_bias.lower(),
            "partial_close_fraction": decision.partial_close_fraction,
            "trailing_enabled": decision.trailing_enabled,
            "professional_diagnostics": decision.to_log(),
        }

    @staticmethod
    def _candles_to_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        frame = pd.DataFrame(candles)
        frame.index = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame["volume"] = frame.get("tick_count", 0)
        return frame[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _resample_candles(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        return frame.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])

    def _entry_gate_rejection_reason(self, signal: dict[str, Any]) -> str | None:
        """Apply demo reliability gates before expensive risk validation/order entry."""
        self._reset_daily_trade_counter_if_needed()

        epic = str(signal.get("epic") or "")
        direction = str(signal.get("direction") or "")

        if self._daily_trade_count >= MAX_DAILY_TRADES:
            return f"Daily trade cap reached ({MAX_DAILY_TRADES})"
        if (
            self._daily_realized_pnl
            <= -float(self._account_equity) * PROFESSIONAL_DAILY_MAX_LOSS_PCT
        ):
            return "Universal daily loss cap reached (1%)"

        if self._get_open_position(epic) is not None:
            return "Instrument already has an open position"

        last_trade_time = self._last_trade_time_by_epic.get(epic)
        if last_trade_time is not None:
            remaining = MIN_SECONDS_BETWEEN_TRADES_PER_INSTRUMENT - (time.time() - last_trade_time)
            if remaining > 0:
                return f"Instrument cooldown active for {int(remaining)}s"

        spread_points = self._spread_points(epic)
        if spread_points is None:
            return "No recent spread snapshot"
        if spread_points > MAX_SPREAD_POINTS:
            return f"Spread {spread_points:.2f} above max {MAX_SPREAD_POINTS:.2f}"

        current_price = float(signal.get("current_price") or 0)
        atr = float(signal.get("atr") or 0)
        atr_pct = atr / current_price if current_price > 0 else 0.0
        if atr_pct < MIN_ATR_PCT:
            return f"ATR pct {atr_pct:.5f} below min {MIN_ATR_PCT:.5f}"
        if atr_pct > MAX_ATR_PCT:
            return f"ATR pct {atr_pct:.5f} above max {MAX_ATR_PCT:.5f}"

        if self._has_correlated_currency_exposure(epic, direction):
            return "Correlated currency exposure already open"

        return None

    async def _apply_news_safety(
        self,
        signal: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Apply a restrictive-only news overlay to an approved strategy signal."""
        if signal is None:
            return None
        if self._news_safety_layer is None:
            return signal

        from src.news.free_news_safety import NewsAction

        epic = str(signal.get("epic") or "")
        symbol = self._currency_pair_for_epic(epic) or epic
        decision = await self._news_safety_layer.evaluate(
            symbol,
            strategy_signal=True,
        )
        signal.update(decision.to_dict())
        self._last_news_decision = {"epic": epic, **decision.to_dict()}
        self._record_trade_event(
            "news_safety_decision",
            epic=epic,
            **decision.to_dict(),
        )
        if decision.news_action == NewsAction.BLOCK_TRADE:
            self._state.trades_rejected += 1
            return None
        if decision.news_action == NewsAction.REQUIRE_EXTRA_CONFIRMATION:
            if not bool(signal.get("extra_confirmation")):
                self._state.trades_rejected += 1
                signal["reason"] = "news_requires_extra_confirmation"
                return None
        if decision.news_action == NewsAction.REDUCE_SIZE:
            current_size = float(signal.get("size", TRADE_SIZE))
            signal["size"] = current_size * 0.5
            signal["news_size_reduction_factor"] = 0.5
        return signal

    def _reset_daily_trade_counter_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._daily_trade_date:
            self._daily_trade_date = today
            self._daily_trade_count = 0
            self._daily_realized_pnl = 0.0

    def _spread_points(self, epic: str) -> float | None:
        snapshot = self._last_snapshots.get(epic)
        if not snapshot:
            return None
        bid = snapshot.get("bid")
        offer = snapshot.get("offer")
        if bid is None or offer is None:
            return None
        scaling_factor = self._fx_scaling_factor(epic)
        return abs(float(offer) - float(bid)) * scaling_factor

    @staticmethod
    def _fx_scaling_factor(epic: str) -> int:
        pair = AutonomousTradingLoop._currency_pair_for_epic(epic)
        return 100 if pair and pair.endswith("JPY") else 10000

    @staticmethod
    def _currency_pair_for_epic(epic: str) -> str | None:
        match = re.search(r"([A-Z]{6})", epic)
        return match.group(1) if match else None

    @classmethod
    def _currency_exposures(cls, epic: str, direction: str) -> dict[str, int]:
        pair = cls._currency_pair_for_epic(epic)
        if pair is None:
            return {}
        base, quote = pair[:3], pair[3:]
        if direction == "BUY":
            return {base: 1, quote: -1}
        if direction == "SELL":
            return {base: -1, quote: 1}
        return {}

    def _has_correlated_currency_exposure(self, epic: str, direction: str) -> bool:
        proposed = self._currency_exposures(epic, direction)
        if not proposed:
            return False

        for pos in self._open_positions:
            pos_data = pos.get("position", pos)
            market = pos.get("market", {})
            pos_epic = market.get("epic") or pos.get("epic") or pos_data.get("epic") or ""
            pos_direction = str(pos_data.get("direction") or "")
            if pos_epic == epic:
                return True
            existing = self._currency_exposures(str(pos_epic), pos_direction)
            for currency, exposure in proposed.items():
                if exposure != 0 and existing.get(currency) == exposure:
                    return True
        return False

    # -------------------------------------------------------------------------
    # Risk Controls
    # -------------------------------------------------------------------------

    def _build_default_risk_engine(self) -> Any:
        from src.risk.drawdown_monitor import DrawdownMonitor
        from src.risk.exposure_manager import ExposureManager
        from src.risk.kill_switch import KillSwitch
        from src.risk.position_sizer import PositionSizer
        from src.risk.risk_engine import RiskEngine
        from src.risk.stop_manager import StopManager

        return RiskEngine(
            position_sizer=PositionSizer(),
            drawdown_monitor=DrawdownMonitor(initial_equity=self._account_equity),
            exposure_manager=ExposureManager(),
            kill_switch=KillSwitch(),
            stop_manager=StopManager(),
            event_bus=None,
        )

    async def _apply_risk_controls(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        """Validate a strategy signal through the central RiskEngine before execution."""
        from src.config.constants import CONFIDENCE_THRESHOLD_DEFAULT

        mistake_penalties = self._mistake_penalties_for_signal(signal)
        adjusted_confidence = int(mistake_penalties["adjusted_confidence"])
        size_reduction_factor = Decimal(str(mistake_penalties["size_reduction_factor"]))
        signal["raw_confidence"] = int(signal.get("confidence", 0))
        signal["confidence"] = adjusted_confidence
        signal["mistake_penalties"] = mistake_penalties

        if adjusted_confidence < CONFIDENCE_THRESHOLD_DEFAULT:
            self._state.trades_rejected += 1
            reason = (
                f"Confidence {adjusted_confidence} below minimum "
                f"{CONFIDENCE_THRESHOLD_DEFAULT} after learning penalties"
            )
            self._last_risk_decision = {
                "allowed": False,
                "rejection_reasons": [reason],
                "position_size": None,
                "applied_reductions": [],
                "mistake_penalties": mistake_penalties,
            }
            self._record_trade_event(
                "signal_rejected",
                epic=signal.get("epic"),
                direction=signal.get("direction"),
                confidence=adjusted_confidence,
                reason=reason,
            )
            print(f"LEARNING REJECTED {signal.get('epic')}: {reason}", flush=True)
            return None

        if self._risk_engine is None:
            signal["size"] = float(Decimal(str(TRADE_SIZE)) * size_reduction_factor)
            return signal

        try:
            from src.risk.risk_engine import TradeSignal

            epic = signal["epic"]
            direction = signal["direction"]
            entry = Decimal(str(signal["current_price"]))
            stop_distance = Decimal(str(signal["stop_distance"]))
            limit_distance = Decimal(str(signal["limit_distance"]))
            stop_loss = entry - stop_distance if direction == "BUY" else entry + stop_distance
            take_profit = entry + limit_distance if direction == "BUY" else entry - limit_distance

            # The live loop stores FX ATR in price units. The shared risk sizer
            # expects point-like units, so use IG's scaling factor when available.
            scaling_factor = Decimal("10000")
            if self._ig_client is not None:
                try:
                    scaling_factor = Decimal(str(await self._ig_client.get_scaling_factor(epic)))
                except Exception:
                    pass

            risk_signal = TradeSignal(
                instrument=epic,
                direction="LONG" if direction == "BUY" else "SHORT",
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=adjusted_confidence,
                strategy=str(signal.get("strategy_name", "legacy_sma_atr")),
                asset_class=self._asset_class_for_epic(epic),
                notional_value=self._estimate_notional_value(epic, TRADE_SIZE, entry),
                region=self._region_for_epic(epic),
                is_hft=False,
                atr=Decimal(str(signal["atr"])) * scaling_factor,
                atr_zscore=0.0,
                risk_pct=Decimal(str(signal.get("risk_per_trade", 0.01))),
            )

            result = await self._risk_engine.validate_signal(
                signal=risk_signal,
                account_equity=self._account_equity,
                current_positions=self._current_positions_for_risk(),
            )

            self._last_risk_decision = {
                "allowed": result.allowed,
                "rejection_reasons": result.rejection_reasons,
                "position_size": str(result.position_size) if result.position_size else None,
                "applied_reductions": [
                    {
                        "source": reduction.source,
                        "factor": str(reduction.factor),
                        "reason": reduction.reason,
                    }
                    for reduction in result.applied_reductions
                ],
                "mistake_penalties": mistake_penalties,
            }

            if not result.allowed or result.position_size is None:
                self._state.trades_rejected += 1
                self._record_trade_event(
                    "signal_rejected",
                    epic=epic,
                    direction=direction,
                    confidence=adjusted_confidence,
                    reason="; ".join(result.rejection_reasons) or "risk_rejected",
                )
                print(
                    f"RISK REJECTED {epic}: {result.rejection_reasons}",
                    flush=True,
                )
                return None

            risk_size = min(result.position_size, Decimal(str(TRADE_SIZE)))
            risk_size *= size_reduction_factor
            if risk_size <= Decimal("0"):
                self._state.trades_rejected += 1
                print(f"RISK REJECTED {epic}: non-positive size {risk_size}", flush=True)
                return None

            validated = dict(signal)
            validated["size"] = float(risk_size)
            validated["risk_position_size"] = str(result.position_size)
            validated["mistake_adjusted_size"] = str(risk_size)
            validated["risk_applied_reductions"] = self._last_risk_decision["applied_reductions"]
            print(
                f"RISK APPROVED {epic}: size={risk_size} raw_size={result.position_size}",
                flush=True,
            )
            return validated
        except Exception as exc:
            self._state.trades_rejected += 1
            self._state.errors += 1
            self._state.last_error = f"Risk validation failed: {exc}"
            self._record_trade_event(
                "signal_rejected",
                epic=signal.get("epic"),
                direction=signal.get("direction"),
                reason=f"risk_error: {exc}",
            )
            print(f"RISK ERROR {signal.get('epic')}: {exc}", flush=True)
            return None

    async def activate_kill_switch(self, reason: str = "manual_activation") -> bool:
        """Activate the live risk-engine kill switch."""
        kill_switch = getattr(self._risk_engine, "_kill_switch", None)
        if kill_switch is None:
            return False
        activated = await kill_switch.activate(reason=reason, trigger_source="manual")
        self._record_trade_event("kill_switch_activated", reason=reason, activated=activated)
        return activated

    async def deactivate_kill_switch(self, confirmation_token: str) -> bool:
        """Deactivate the live risk-engine kill switch when policy allows."""
        kill_switch = getattr(self._risk_engine, "_kill_switch", None)
        if kill_switch is None:
            return False
        deactivated = await kill_switch.deactivate(confirmation_token=confirmation_token)
        self._record_trade_event("kill_switch_deactivated", deactivated=deactivated)
        return deactivated

    def get_kill_switch_status(self) -> dict[str, Any]:
        kill_switch = getattr(self._risk_engine, "_kill_switch", None)
        if kill_switch is None:
            return {"active": False, "reason": "", "activation_time": None, "can_deactivate": False}
        return kill_switch.get_status()

    def _asset_class_for_epic(self, epic: str) -> str:
        if epic.startswith("CS."):
            return "forex"
        if epic.startswith("IX."):
            return "indices"
        return "forex"

    def _region_for_epic(self, epic: str) -> str | None:
        if "EUR" in epic or "GBP" in epic:
            return "europe"
        if "JPY" in epic:
            return "asia"
        return None

    def _estimate_notional_value(
        self,
        epic: str,
        size: float,
        price: Decimal,
    ) -> Decimal:
        _ = epic
        return abs(Decimal(str(size)) * price)

    def _current_positions_for_risk(self) -> list[dict[str, Any]]:
        positions: list[dict[str, Any]] = []
        for pos in self._open_positions:
            pos_data = pos.get("position", pos)
            market = pos.get("market", {})
            epic = market.get("epic") or pos.get("epic") or pos_data.get("epic") or ""
            size = Decimal(str(pos_data.get("size") or pos_data.get("dealSize") or "0"))
            level = Decimal(str(pos_data.get("level") or pos_data.get("openLevel") or "0"))
            positions.append(
                {
                    "instrument": epic,
                    "asset_class": self._asset_class_for_epic(epic),
                    "notional_value": str(abs(size * level)),
                    "region": self._region_for_epic(epic),
                }
            )
        return positions

    def _learning_indicators(self, signal: dict[str, Any]) -> dict[str, float]:
        """Extract numeric signal features that can be reused for mistake matching."""
        indicator_keys = (
            "atr",
            "sma_fast",
            "sma_slow",
            "trend_strength",
            "rr_ratio",
            "stop_distance",
            "limit_distance",
            "tp1_distance",
            "spread_atr_ratio",
        )
        indicators: dict[str, float] = {}
        for key in indicator_keys:
            value = signal.get(key)
            if value is None:
                continue
            try:
                indicators[key] = float(value)
            except (TypeError, ValueError):
                continue

        current_price = signal.get("current_price")
        atr = signal.get("atr")
        try:
            if current_price and atr:
                indicators["expected_volatility"] = abs(float(atr) / float(current_price))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        return indicators

    def _mistake_penalties_for_signal(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Calculate active mistake-pattern penalties for a prospective signal."""
        if self._mistake_analyzer is None:
            return {
                "confidence_penalty": 0,
                "adjusted_confidence": int(signal.get("confidence", 0)),
                "size_reduction_factor": 1.0,
                "applied": False,
            }

        try:
            from src.learning.mistake_analyzer import TradeSignal as LearningTradeSignal

            learning_signal = LearningTradeSignal(
                regime=str(signal.get("regime", "unknown")),
                strategy=str(signal.get("strategy_name", "legacy_sma_atr")),
                indicators=self._learning_indicators(signal),
                confidence=int(signal.get("confidence", 0)),
                is_hft=False,
            )
            confidence_penalty = (
                self._mistake_analyzer.get_confidence_penalty(learning_signal)
                if hasattr(self._mistake_analyzer, "get_confidence_penalty")
                else 0
            )
            size_reduction = (
                self._mistake_analyzer.get_size_reduction_factor(learning_signal)
                if hasattr(self._mistake_analyzer, "get_size_reduction_factor")
                else 1.0
            )
        except Exception as exc:
            print(f"MISTAKE PENALTY CHECK FAILED {signal.get('epic')}: {exc}", flush=True)
            return {
                "confidence_penalty": 0,
                "adjusted_confidence": int(signal.get("confidence", 0)),
                "size_reduction_factor": 1.0,
                "applied": False,
                "error": str(exc),
            }

        adjusted_confidence = max(
            0,
            int(signal.get("confidence", 0)) - abs(int(confidence_penalty)),
        )
        return {
            "confidence_penalty": abs(int(confidence_penalty)),
            "adjusted_confidence": adjusted_confidence,
            "size_reduction_factor": float(size_reduction),
            "applied": abs(int(confidence_penalty)) > 0 or float(size_reduction) < 1.0,
        }

    async def _record_learning_outcome(
        self,
        trade: Any,
        context: Any,
        pnl: Decimal,
        exit_price: Decimal,
        close_result: dict[str, Any],
    ) -> None:
        """Feed closed trade outcomes into mistake learning and pattern resolution."""
        if self._mistake_analyzer is None:
            return

        from src.learning.mistake_analyzer import ClosedTrade, MarketOutcome

        indicators = {
            str(key): float(value)
            for key, value in (getattr(context, "indicators_json", None) or {}).items()
            if isinstance(value, int | float | Decimal)
        }
        direction_value = getattr(trade.direction, "value", str(trade.direction))
        entry_direction = "up" if direction_value in ("LONG", "BUY") else "down"
        loss_direction = "down" if entry_direction == "up" else "up"
        entry_price = Decimal(str(trade.entry_price))
        realized_volatility = Decimal("0")
        if entry_price != 0:
            realized_volatility = abs(exit_price - entry_price) / abs(entry_price)

        closed_trade = ClosedTrade(
            trade_id=str(trade.id),
            regime=str(getattr(trade, "regime", None) or getattr(context, "regime", None) or "unknown"),
            strategy=str(getattr(trade, "strategy", "") or ""),
            indicators=indicators,
            confidence_at_entry=int(
                getattr(trade, "confidence_score", None)
                or getattr(context, "confidence", None)
                or 0
            ),
            exit_reason=str(close_result.get("reason") or close_result.get("exitReason") or "broker_close"),
            pnl=float(pnl),
            entry_conditions={
                "direction": entry_direction,
                "instrument": trade.instrument,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "size": float(trade.size),
            },
        )

        if pnl < Decimal("0"):
            outcome = MarketOutcome(
                actual_direction=loss_direction,
                volatility_realized=float(realized_volatility),
                regime_actual=closed_trade.regime,
                breakout_confirmed=True,
                timing_optimal=False,
            )
            classification = self._mistake_analyzer.classify_mistake(
                closed_trade,
                outcome,
            )
            record = self._mistake_analyzer.record_mistake(closed_trade, classification)
            await self._mistake_analyzer.mistake_db.store_record(record)
            await self._mistake_analyzer.detect_patterns()
        else:
            await self._mistake_analyzer.update_resolution_progress(closed_trade)

    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------

    async def _execute_signal(self, signal: dict[str, Any]) -> None:
        epic      = signal["epic"]
        direction = signal["direction"]
        stop      = signal["stop_distance"]
        limit     = signal["limit_distance"]
        size      = float(signal.get("size", TRADE_SIZE))

        print(
            f"PLACING: {direction} {epic} size={size} "
            f"SL={stop:.6f} TP={limit:.6f} R:R=1:{signal.get('rr_ratio', 2)}",
            flush=True,
        )
        try:
            if self._account_type == "LIVE" and not self._professional_live_approved:
                raise RuntimeError("Live execution is blocked: strategy is not approved")
            if self._account_type == "LIVE" and self._strategy_mode == "LEGACY_SMA":
                raise RuntimeError("Legacy SMA is never authorized for live execution")
            if (
                self._account_type == "DEMO"
                and self._strategy_mode in {"PROFESSIONAL", "GUARDED_AUTO"}
                and not self._professional_demo_forward_approved
            ):
                raise RuntimeError(
                    "Demo execution is blocked: strategy is not approved for forward test"
                )

            execution_permit = self._ig_client.issue_opening_order_permit()
            result = await self._ig_client.place_order(
                epic=epic,
                direction=direction,
                size=size,
                stop_distance=stop,
                limit_distance=limit,
                execution_permit=execution_permit,
            )
            status = result.get("dealStatus", "unknown")

            if status == "ACCEPTED":
                self._state.trades_executed += 1
                self._reset_daily_trade_counter_if_needed()
                self._daily_trade_count += 1
                self._last_trade_time_by_epic[epic] = time.time()
                deal_id   = result.get("dealId", "")
                entry_lvl = result.get("level")
                sl_level = result.get("stopLevel")
                tp_level = result.get("limitLevel")
                print(
                    f"✅ TRADE OPENED: {direction} {epic} | "
                    f"level={entry_lvl} | dealId={deal_id}",
                    flush=True,
                )
                self._record_trade_event(
                    "trade_opened",
                    epic=epic,
                    direction=direction,
                    deal_id=deal_id,
                    entry_level=entry_lvl,
                    confidence=signal.get("confidence"),
                    reason="ig_order_accepted",
                )

                if deal_id and entry_lvl:
                    price = float(entry_lvl)
                    sl_level = sl_level or (
                        round(price - stop, 5)
                        if direction == "BUY"
                        else round(price + stop, 5)
                    )
                    tp_level = tp_level or (
                        round(price + limit, 5)
                        if direction == "BUY"
                        else round(price - limit, 5)
                    )
                    if signal.get("strategy_name") == "professional_ict":
                        tp1_distance = float(signal.get("tp1_distance") or stop)
                        tp1_level = (
                            price + tp1_distance
                            if direction == "BUY"
                            else price - tp1_distance
                        )
                        self._professional_positions[deal_id] = {
                            "direction": direction,
                            "entry_level": price,
                            "tp1_level": tp1_level,
                            "final_target_level": tp_level,
                            "partial_close_fraction": float(
                                signal.get("partial_close_fraction", 0.5)
                            ),
                            "size": size,
                            "partial_taken": False,
                            "trailing_enabled": bool(signal.get("trailing_enabled", False)),
                        }
                else:
                    self._state.last_error = "Accepted trade missing deal_id or entry level"
                    self._record_trade_event(
                        "sltp_failed",
                        epic=epic,
                        direction=direction,
                        deal_id=deal_id,
                        reason="accepted_trade_missing_deal_id_or_entry_level",
                    )
                    await self.activate_kill_switch("Accepted trade missing deal ID or entry level")
                    return

                await self._persist_open_trade(
                    signal=signal,
                    deal_id=deal_id,
                    deal_reference=result.get("dealReference"),
                    entry_level=entry_lvl,
                    stop_level=sl_level,
                    limit_level=tp_level,
                )

            else:
                self._state.trades_rejected += 1
                self._record_trade_event(
                    "trade_rejected",
                    epic=epic,
                    direction=direction,
                    status=status,
                    reason=result.get("reason"),
                )
                print(
                    f"❌ TRADE REJECTED: {direction} {epic} | "
                    f"reason={result.get('reason')} | raw={result}",
                    flush=True,
                )
        except Exception as exc:
            self._state.trades_rejected += 1
            self._record_trade_event(
                "trade_error",
                epic=epic,
                direction=direction,
                reason=str(exc),
            )
            print(f"TRADE ERROR {epic}: {exc}", flush=True)

    def _record_trade_event(self, event: str, **details: Any) -> None:
        self._recent_trade_events.appendleft(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **{key: value for key, value in details.items() if value is not None},
            }
        )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    async def _persist_open_trade(
        self,
        signal: dict[str, Any],
        deal_id: str,
        deal_reference: str | None,
        entry_level: Any,
        stop_level: float | None,
        limit_level: float | None,
    ) -> None:
        """Persist an accepted IG trade and its open position."""
        try:
            from src.db.database import get_session
            from src.db.models import PositionStatus, TradeContext, TradeDirection, TradeStatus
            from src.db.repositories.trade_repo import TradeRepository

            entry_price = Decimal(str(entry_level or signal["current_price"]))
            size = Decimal(str(signal.get("size", TRADE_SIZE)))
            direction = (
                TradeDirection.LONG if signal["direction"] == "BUY" else TradeDirection.SHORT
            )

            async with get_session() as session:
                repo = TradeRepository(session)
                trade = await repo.create_trade(
                    {
                        "instrument": signal["epic"],
                        "direction": direction,
                        "size": size,
                        "entry_price": entry_price,
                        "ig_deal_id": deal_id or None,
                        "ig_deal_reference": deal_reference,
                        "strategy": str(signal.get("strategy_name", "legacy_sma_atr")),
                        "opened_at": datetime.now(timezone.utc),
                        "confidence_score": int(signal.get("confidence", 0)),
                        "regime": "unknown",
                        "is_hft": False,
                        "is_copied": False,
                        "status": TradeStatus.OPEN,
                    }
                )
                session.add(
                    TradeContext(
                        trade_id=trade.id,
                        indicators_json=self._learning_indicators(signal),
                        regime=str(signal.get("regime", "unknown")),
                        confidence=int(signal.get("confidence", 0)),
                        ml_predictions_json={
                            "source": str(signal.get("strategy_name", "legacy_sma_atr")),
                            "direction": signal["direction"],
                            "rr_ratio": signal.get("rr_ratio"),
                            "risk_position_size": signal.get("risk_position_size"),
                            "professional_diagnostics": signal.get(
                                "professional_diagnostics"
                            ),
                        },
                    )
                )
                position = await repo.create_position(
                    {
                        "trade_id": trade.id,
                        "instrument": signal["epic"],
                        "direction": direction,
                        "size": size,
                        "entry_price": entry_price,
                        "ig_deal_id": deal_id or None,
                        "stop_loss": Decimal(str(stop_level)) if stop_level is not None else None,
                        "take_profit": Decimal(str(limit_level)) if limit_level is not None else None,
                        "status": PositionStatus.OPEN,
                        "is_hft": False,
                    }
                )

            if deal_id:
                self._deal_records[deal_id] = {
                    "trade_id": str(trade.id),
                    "position_id": str(position.id),
                }
            print(
                f"DB PERSISTED OPEN: {signal['epic']} trade_id={trade.id} position_id={position.id}",
                flush=True,
            )
        except Exception as exc:
            self._state.last_error = f"Trade persistence failed: {exc}"
            print(f"DB PERSIST OPEN FAILED {signal.get('epic')}: {exc}", flush=True)

    async def _persist_closed_trade(
        self,
        deal_id: str,
        epic: str,
        close_result: dict[str, Any],
    ) -> None:
        """Close the matching persisted trade/position when IG confirms closure."""
        try:
            from sqlalchemy import select

            from src.db.database import get_session
            from src.db.models import TradeContext
            from src.db.repositories.trade_repo import TradeRepository

            record = self._deal_records.get(deal_id)
            pnl = Decimal(str(close_result.get("profit") or "0"))
            exit_price_raw = close_result.get("level") or close_result.get("closeLevel") or "0"
            exit_price = Decimal(str(exit_price_raw))
            closed_trade = None
            trade_context = None

            async with get_session() as session:
                repo = TradeRepository(session)
                if record is None:
                    trade = await repo.get_trade_by_ig_deal_id(deal_id) if deal_id else None
                    position = await repo.get_position_by_ig_deal_id(deal_id) if deal_id else None
                    if trade is None:
                        open_trades = await repo.get_open_trades(instrument=epic)
                        trade = open_trades[0] if open_trades else None
                    if position is None:
                        positions = await repo.get_open_positions()
                        position = next((p for p in positions if p.instrument == epic), None)
                else:
                    trade = await repo.get_trade(record["trade_id"])
                    position = await repo.get_position(record["position_id"])

                if trade is not None:
                    if exit_price == Decimal("0"):
                        exit_price = trade.entry_price
                    await repo.close_trade(trade.id, exit_price=exit_price, pnl=pnl)
                    context_result = await session.execute(
                        select(TradeContext).where(TradeContext.trade_id == trade.id)
                    )
                    trade_context = context_result.scalar_one_or_none()
                    closed_trade = trade
                if position is not None:
                    await repo.close_position(position.id)

            if closed_trade is not None:
                await self._record_learning_outcome(
                    trade=closed_trade,
                    context=trade_context,
                    pnl=pnl,
                    exit_price=exit_price,
                    close_result=close_result,
                )

            if deal_id:
                self._deal_records.pop(deal_id, None)
                self._professional_positions.pop(deal_id, None)
            print(f"DB PERSISTED CLOSE: {epic} deal_id={deal_id} pnl={pnl}", flush=True)
        except Exception as exc:
            self._state.last_error = f"Trade close persistence failed: {exc}"
            print(f"DB PERSIST CLOSE FAILED {epic}: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def _execution_approval_status(self) -> dict[str, Any]:
        if self._account_type == "LIVE":
            return {
                "enabled": self._professional_live_approved
                and self._strategy_mode != "LEGACY_SMA",
                "reason": (
                    "approved_for_live"
                    if self._professional_live_approved and self._strategy_mode != "LEGACY_SMA"
                    else "professional_strategy_not_approved_for_live"
                ),
            }
        if self._account_type == "DEMO" and self._strategy_mode in {"PROFESSIONAL", "GUARDED_AUTO"}:
            return {
                "enabled": self._professional_demo_forward_approved,
                "reason": (
                    "approved_for_demo_forward_test"
                    if self._professional_demo_forward_approved
                    else "professional_strategy_not_approved_for_demo_forward_test"
                ),
            }
        if self._account_type == "DEMO" and self._strategy_mode == "LIQUIDITY_SWEEP_1M":
            return {
                "enabled": True,
                "reason": "approved_for_demo_liquidity_sweep_1m",
            }
        return {
            "enabled": self._account_type == "DEMO",
            "reason": (
                "approved_for_demo_legacy_strategy"
                if self._account_type == "DEMO"
                else "unknown_account_type"
            ),
        }

    def get_status(self) -> dict[str, Any]:
        uptime = time.time() - self._state.start_time if self._state.running else 0
        execution_approval = self._execution_approval_status()
        return {
            "running": self._state.running,
            "connected": self._state.connected,
            "execution_enabled": execution_approval["enabled"],
            "execution_block_reason": (
                None if execution_approval["enabled"] else execution_approval["reason"]
            ),
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
            "kill_switch": self.get_kill_switch_status(),
            "recent_trade_events": list(self._recent_trade_events),
            "strategy": {
                "type": self._strategy_mode,
                "account_type": self._account_type,
                "execution_approval": execution_approval,
                "professional_live_approved": self._professional_live_approved,
                "professional_demo_forward_approved": self._professional_demo_forward_approved,
                "sl_multiplier": SL_MULTIPLIER,
                "tp_multiplier": TP_MULTIPLIER,
                "risk_reward": f"1:{TP_MULTIPLIER / SL_MULTIPLIER:.1f}",
                "max_trade_size_lots": TRADE_SIZE,
                "max_open_positions": MAX_OPEN_POSITIONS,
                "max_daily_trades": MAX_DAILY_TRADES,
                "daily_trade_count": self._daily_trade_count,
                "daily_realized_pnl": round(self._daily_realized_pnl, 2),
                "daily_max_loss_pct": PROFESSIONAL_DAILY_MAX_LOSS_PCT,
                "pair_cooldown_seconds": MIN_SECONDS_BETWEEN_TRADES_PER_INSTRUMENT,
                "max_spread_points": MAX_SPREAD_POINTS,
                "atr_pct_range": [MIN_ATR_PCT, MAX_ATR_PCT],
                "min_trend_strength": MIN_TREND_STRENGTH,
                "min_candles": MIN_CANDLES_FOR_ENTRY,
                "trading_hours_utc": f"{TRADING_HOURS_UTC[0]}:00-{TRADING_HOURS_UTC[1]}:00",
                "professional_risk_per_trade": 0.002,
                "news_filter_mode": self._news_filter_mode,
                "news_filter_fail_closed": self._news_filter_mode == "FAIL_CLOSED",
                "free_news_safety_enabled": self._news_safety_layer is not None,
                "timeframes": (
                    ["D", "1M"]
                    if self._strategy_mode == "LIQUIDITY_SWEEP_1M"
                    else ["4H", "1H", "5M", "1M"]
                ),
            },
            "last_risk_decision": self._last_risk_decision,
            "last_news_decision": self._last_news_decision,
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
