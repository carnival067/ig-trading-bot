"""Comparable simulation for the retained legacy SMA strategy."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.research.professional_backtest import (
    ProfessionalBacktestResult,
    ProfessionalTrade,
)
from src.strategy.strategies.legacy_sma import LegacySMAStrategy


class LegacySMABacktester:
    """Run legacy entries under the new conservative portfolio limits."""

    def __init__(
        self,
        *,
        initial_equity: float = 20_000,
        risk_per_trade: float = 0.002,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.3,
        commission_per_lot: float = 0.0,
        max_daily_trades: int = 3,
        max_daily_loss: float = 0.01,
        max_leverage: float = 20.0,
    ) -> None:
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.max_leverage = max_leverage
        self.strategy = LegacySMAStrategy()

    def run(
        self,
        pair: str,
        one_minute: pd.DataFrame,
        *,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ) -> ProfessionalBacktestResult:
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        spread = self.spread_pips * pip
        slippage = self.slippage_pips * pip
        start = max(25, int(len(one_minute) * start_fraction))
        end = min(len(one_minute) - 2, int(len(one_minute) * end_fraction))
        equity = self.initial_equity
        curve = [equity]
        daily_pnl: dict[str, float] = {}
        daily_trades: dict[str, int] = {}
        rejected: Counter[str] = Counter()
        trades: list[ProfessionalTrade] = []
        cursor = start

        while cursor < end:
            timestamp = one_minute.index[cursor]
            day = timestamp.date().isoformat()
            if timestamp.weekday() >= 5 or not 8 <= timestamp.hour < 18:
                rejected["outside_legacy_session"] += 1
                cursor += 1
                continue
            if daily_trades.get(day, 0) >= self.max_daily_trades:
                rejected["daily_trade_cap"] += 1
                cursor += 1
                continue
            if daily_pnl.get(day, 0.0) <= -equity * self.max_daily_loss:
                rejected["daily_loss_cap"] += 1
                cursor += 1
                continue
            signal = self.strategy.evaluate(one_minute.iloc[cursor - 199 : cursor + 1])
            if signal is None:
                rejected["no_sma_signal"] += 1
                cursor += 1
                continue

            direction = 1 if signal["direction"] == "BUY" else -1
            entry_cursor = cursor + 1
            entry_time = one_minute.index[entry_cursor]
            entry = float(one_minute["open"].iloc[entry_cursor]) + direction * (
                spread / 2 + slippage
            )
            stop_distance = float(signal["stop_distance"])
            target_distance = float(signal["limit_distance"])
            stop = entry - direction * stop_distance
            target = entry + direction * target_distance
            risk_cash = equity * self.risk_per_trade
            units = min(
                risk_cash / stop_distance,
                equity * self.max_leverage / entry,
            )
            exit_price = entry
            exit_reason = "data_end"
            exit_cursor = entry_cursor
            for future in range(entry_cursor, end):
                bar = one_minute.iloc[future]
                high, low = float(bar["high"]), float(bar["low"])
                stop_hit = low <= stop if direction == 1 else high >= stop
                target_hit = high >= target if direction == 1 else low <= target
                if stop_hit:
                    exit_price = stop - direction * (spread / 2 + slippage)
                    exit_reason = "stop"
                    exit_cursor = future
                    break
                if target_hit:
                    exit_price = target - direction * (spread / 2 + slippage)
                    exit_reason = "target"
                    exit_cursor = future
                    break
                opposite = self.strategy.evaluate(
                    one_minute.iloc[max(0, future - 199) : future + 1]
                )
                if opposite and opposite["direction"] != signal["direction"]:
                    exit_price = float(bar["close"]) - direction * (
                        spread / 2 + slippage
                    )
                    exit_reason = "signal_flip"
                    exit_cursor = future
                    break
            pnl = direction * (exit_price - entry) * units
            pnl -= self.commission_per_lot * (units / 100_000)
            equity += pnl
            curve.append(equity)
            daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
            daily_trades[day] = daily_trades.get(day, 0) + 1
            trades.append(
                ProfessionalTrade(
                    pair=pair,
                    session="LEGACY_08_18_UTC",
                    direction=signal["direction"],
                    entry_time=entry_time.isoformat(),
                    exit_time=one_minute.index[exit_cursor].isoformat(),
                    entry_price=entry,
                    stop_price=stop,
                    tp1_price=entry + direction * stop_distance,
                    target_price=target,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=pnl / risk_cash if risk_cash else 0.0,
                    partial_taken=False,
                    exit_reason=exit_reason,
                )
            )
            cursor = max(cursor + 30, exit_cursor + 1)

        pnl_values = [trade.pnl for trade in trades]
        gross_profit = sum(value for value in pnl_values if value > 0)
        gross_loss = abs(sum(value for value in pnl_values if value < 0))
        array = np.array(curve)
        peak = np.maximum.accumulate(array)
        drawdown = float(np.max((peak - array) / peak)) if len(array) else 0.0
        return ProfessionalBacktestResult(
            pair=pair,
            initial_equity=self.initial_equity,
            final_equity=equity,
            total_return=equity / self.initial_equity - 1,
            win_rate=(
                sum(value > 0 for value in pnl_values) / len(pnl_values)
                if pnl_values
                else 0.0
            ),
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss
                else (float("inf") if gross_profit else 0.0)
            ),
            max_drawdown=drawdown,
            average_r=(
                float(np.mean([trade.r_multiple for trade in trades]))
                if trades
                else 0.0
            ),
            trade_count=len(trades),
            trades_per_session={"LEGACY_08_18_UTC": len(trades)},
            rejected_reasons=dict(rejected),
            trades=trades,
        )
