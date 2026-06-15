"""Research-only diagnostics for professional strategy trade journals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.strategy.professional.fvg_detector import FairValueGapDetector
from src.strategy.professional.higher_timeframe_trend import HigherTimeframeTrend
from src.strategy.professional.liquidity_sweep import LiquiditySweepDetector
from src.strategy.professional.market_structure import MarketStructureDetector
from src.strategy.professional.order_block_detector import OrderBlockDetector


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    final_target_r: float
    partial_at_r: float | None = None
    partial_fraction: float = 0.5
    breakeven_at_r: float | None = None


def atr_series(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return ranges.ewm(alpha=1 / period, adjust=False).mean()


def classify_session(timestamp: pd.Timestamp) -> str:
    hour = timestamp.hour
    if 0 <= hour < 7:
        return "ASIAN"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 17:
        return "OVERLAP"
    if 17 <= hour < 22:
        return "NEW_YORK"
    return "ASIAN"


def enrich_trade(
    trade: pd.Series,
    five: pd.DataFrame,
    one_hour: pd.DataFrame,
    four_hour: pd.DataFrame,
    *,
    spread_pips: float,
) -> dict[str, object]:
    entry_time = pd.Timestamp(trade["entry_time"])
    signal_time = entry_time - pd.Timedelta(minutes=5)
    five_context = five.loc[:signal_time].tail(500)
    one_context = one_hour.loc[:signal_time].tail(250)
    four_context = four_hour.loc[:signal_time].tail(250)
    direction = "BULLISH" if trade["direction"] == "BUY" else "BEARISH"
    pair = str(trade["pair"])
    pip = 0.01 if pair.endswith("JPY") else 0.0001

    atr = float(atr_series(five_context).iloc[-1])
    atr_history = atr_series(five_context).tail(100)
    percentile = float((atr_history <= atr).mean()) if len(atr_history) else 0.5
    if percentile <= 0.33:
        volatility_regime = "LOW"
    elif percentile >= 0.67:
        volatility_regime = "HIGH"
    else:
        volatility_regime = "NORMAL"
    spread_atr = spread_pips * pip / atr if atr else float("inf")
    if spread_atr <= 0.10:
        spread_regime = "LOW_<=0.10"
    elif spread_atr <= 0.15:
        spread_regime = "MEDIUM_0.10-0.15"
    else:
        spread_regime = "HIGH_>0.15"

    sweep = LiquiditySweepDetector().detect(five_context, direction)
    structure = MarketStructureDetector().detect(
        five_context,
        direction,
        sweep.bar_index,
    )
    fvg = FairValueGapDetector().detect(five_context, direction)
    order_block = OrderBlockDetector().detect(five_context, direction, atr)
    fvg_valid = bool(fvg.detected and fvg.retraced)
    order_block_valid = bool(order_block.detected and order_block.retraced)
    if fvg_valid and order_block_valid:
        setup_type = "BOTH"
    elif fvg_valid:
        setup_type = "FVG"
    elif order_block_valid:
        setup_type = "ORDER_BLOCK"
    else:
        setup_type = "UNKNOWN"

    trend_4h = _ema_trend(four_context, 20, 50)
    trend_1h = _ema_trend(one_context, 20, 50)
    ema_50_200 = _ema_trend(four_context, 50, 200)
    structure_trend = _structure_trend(four_context)
    entry = float(trade["entry_price"])
    recent = four_context.tail(50)
    midpoint = (float(recent["high"].max()) + float(recent["low"].min())) / 2
    premium_discount_ok = entry <= midpoint if direction == "BULLISH" else entry >= midpoint
    support_resistance_distance = min(
        abs(entry - float(recent["low"].min())),
        abs(float(recent["high"].max()) - entry),
    )
    near_htf_sr = support_resistance_distance <= atr * 2

    exit_time = pd.Timestamp(trade["exit_time"])
    holding_minutes = (exit_time - entry_time).total_seconds() / 60
    return {
        **trade.to_dict(),
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "signal_time": signal_time.isoformat(),
        "session_diagnostic": classify_session(entry_time),
        "day_of_week": entry_time.day_name(),
        "hour_utc": entry_time.hour,
        "month": entry_time.strftime("%Y-%m"),
        "setup_type": setup_type,
        "fvg_valid": fvg_valid,
        "order_block_valid": order_block_valid,
        "confirmation_type": structure.event if structure.confirmed else "UNKNOWN",
        "liquidity_sweep_type": (
            "SELL_SIDE_SWEEP" if direction == "BULLISH" else "BUY_SIDE_SWEEP"
        ) if sweep.detected else "UNKNOWN",
        "volatility_regime": volatility_regime,
        "atr": atr,
        "atr_percentile": percentile,
        "spread_atr_ratio": spread_atr,
        "spread_atr_regime": spread_regime,
        "holding_minutes": holding_minutes,
        "trend_4h": trend_4h,
        "trend_1h": trend_1h,
        "ema_50_200_trend": ema_50_200,
        "market_structure_trend": structure_trend,
        "premium_discount_ok": premium_discount_ok,
        "near_htf_support_resistance": near_htf_sr,
    }


def replay_exit(
    trade: pd.Series,
    five: pd.DataFrame,
    policy: ExitPolicy,
    *,
    spread_pips: float,
    slippage_pips: float,
    end_time: pd.Timestamp | None = None,
) -> dict[str, object]:
    pair = str(trade["pair"])
    direction = 1 if trade["direction"] == "BUY" else -1
    entry = float(trade["entry_price"])
    risk = abs(float(trade["tp1_price"]) - entry)
    if risk <= 0:
        return {"r_multiple": 0.0, "exit_reason": "invalid_risk", "holding_minutes": 0.0}
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    exit_cost = spread_pips * pip / 2 + slippage_pips * pip
    stop = entry - direction * risk
    target = entry + direction * risk * policy.final_target_r
    partial_level = (
        entry + direction * risk * policy.partial_at_r
        if policy.partial_at_r is not None
        else None
    )
    breakeven_level = (
        entry + direction * risk * policy.breakeven_at_r
        if policy.breakeven_at_r is not None
        else None
    )
    remaining = 1.0
    realised_r = 0.0
    partial_taken = False
    breakeven_active = False
    entry_time = pd.Timestamp(trade["entry_time"])
    bars = five.loc[entry_time:end_time]
    final_time = entry_time
    exit_reason = "timeout"
    for timestamp, bar in bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        stop_hit = low <= stop if direction == 1 else high >= stop
        target_hit = high >= target if direction == 1 else low <= target
        if stop_hit:
            stop_fill = stop - direction * exit_cost
            realised_r += direction * (stop_fill - entry) / risk * remaining
            exit_reason = "partial_tp_then_breakeven" if partial_taken and breakeven_active else "stop_loss"
            final_time = timestamp
            break
        if partial_level is not None and not partial_taken:
            partial_hit = high >= partial_level if direction == 1 else low <= partial_level
            if partial_hit:
                partial_fill = partial_level - direction * exit_cost
                realised_r += (
                    direction * (partial_fill - entry) / risk * policy.partial_fraction
                )
                remaining -= policy.partial_fraction
                partial_taken = True
        if breakeven_level is not None and not breakeven_active:
            breakeven_hit = (
                high >= breakeven_level if direction == 1 else low <= breakeven_level
            )
            if breakeven_hit:
                stop = entry
                breakeven_active = True
        if target_hit:
            target_fill = target - direction * exit_cost
            realised_r += direction * (target_fill - entry) / risk * remaining
            exit_reason = "take_profit"
            final_time = timestamp
            break
    return {
        "r_multiple": realised_r,
        "exit_reason": exit_reason,
        "holding_minutes": (final_time - entry_time).total_seconds() / 60,
        "exit_time": final_time.isoformat(),
    }


def calculate_metrics(frame: pd.DataFrame, *, risk_per_trade: float = 0.002) -> dict[str, float | int]:
    if frame.empty:
        return {
            "total_trades": 0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "expectancy": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "max_consecutive_losses": 0,
        }
    ordered = frame.copy()
    ordered["entry_time"] = pd.to_datetime(ordered["entry_time"], utc=True)
    ordered = ordered.sort_values("entry_time")
    values = ordered["r_multiple"].astype(float).to_numpy()
    winners = values[values > 0]
    losers = values[values < 0]
    gross_profit = float(winners.sum())
    gross_loss = abs(float(losers.sum()))
    sleeve_columns = ["pair"] + (["window"] if "window" in ordered else [])
    ordered["_sleeve"] = ordered[sleeve_columns].astype(str).agg("|".join, axis=1)
    sleeves = sorted(ordered["_sleeve"].unique())
    sleeve_equity = {sleeve: 1.0 for sleeve in sleeves}
    portfolio_curve: list[float] = []
    for _, row in ordered.iterrows():
        sleeve = str(row["_sleeve"])
        sleeve_equity[sleeve] *= 1 + float(row["r_multiple"]) * risk_per_trade
        portfolio_curve.append(sum(sleeve_equity.values()) / len(sleeves))
    equity = np.asarray(portfolio_curve)
    peaks = np.maximum.accumulate(np.insert(equity, 0, 1.0))[1:]
    drawdown = np.max((peaks - equity) / peaks) if len(equity) else 0.0
    return {
        "total_trades": int(len(values)),
        "total_return": float(sum(sleeve_equity.values()) / len(sleeves) - 1),
        "win_rate": float((values > 0).mean()),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss
            else (float("inf") if gross_profit else 0.0)
        ),
        "max_drawdown": float(drawdown),
        "average_r": float(values.mean()),
        "median_r": float(np.median(values)),
        "expectancy": float(values.mean()),
        "average_winner": float(winners.mean()) if len(winners) else 0.0,
        "average_loser": float(losers.mean()) if len(losers) else 0.0,
        "largest_winner": float(winners.max()) if len(winners) else 0.0,
        "largest_loser": float(losers.min()) if len(losers) else 0.0,
        "max_consecutive_losses": _max_consecutive(values < 0),
    }


def experiment_status(frame: pd.DataFrame, metrics: dict[str, float | int]) -> tuple[str, str]:
    failures: list[str] = []
    if metrics["total_trades"] < 200:
        failures.append("fewer than 200 trades")
    if metrics["total_return"] <= 0:
        failures.append("non-positive return")
    if metrics["profit_factor"] < 1.25:
        failures.append("profit factor below 1.25")
    if metrics["average_r"] <= 0:
        failures.append("non-positive expectancy")
    if metrics["max_drawdown"] >= 0.15:
        failures.append("drawdown at or above 15%")
    if not frame.empty:
        pnl = frame.assign(_pnl=frame["r_multiple"].astype(float))
        positive = pnl.loc[pnl["_pnl"] > 0, "_pnl"].sum()
        if positive > 0:
            pair_share = pnl.groupby("pair")["_pnl"].sum().clip(lower=0).max() / positive
            month_share = pnl.groupby("month")["_pnl"].sum().clip(lower=0).max() / positive
            if pair_share > 0.50:
                failures.append("profit concentrated in one pair")
            if month_share > 0.50:
                failures.append("profit concentrated in one month")
        window_returns = pnl.groupby(["pair", "window"])["_pnl"].sum()
        if len(window_returns) and float((window_returns > 0).mean()) < 0.75:
            failures.append("walk-forward windows unstable")
    return ("FAIL", "; ".join(dict.fromkeys(failures))) if failures else ("PASS", "all gates passed")


def grouped_metrics(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(column, dropna=False):
        rows.append({column: str(value), **calculate_metrics(group)})
    return rows


def map_exit_reason(row: pd.Series) -> str:
    reason = str(row["exit_reason"])
    if reason == "stop":
        return "stop_loss"
    if reason == "breakeven":
        return "partial_tp_then_breakeven"
    if reason == "target":
        return "take_profit"
    if reason == "data_end":
        return "timeout"
    return reason


def apply_filter(frame: pd.DataFrame, predicate: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    return frame.loc[predicate(frame)].copy()


def _ema_trend(frame: pd.DataFrame, fast_period: int, slow_period: int) -> str:
    if len(frame) < slow_period + 3:
        return "NEUTRAL"
    close = frame["close"]
    fast = close.ewm(span=fast_period, adjust=False).mean()
    slow = close.ewm(span=slow_period, adjust=False).mean()
    slope = float(slow.iloc[-1] - slow.iloc[-3])
    if float(fast.iloc[-1]) > float(slow.iloc[-1]) and slope > 0:
        return "BULLISH"
    if float(fast.iloc[-1]) < float(slow.iloc[-1]) and slope < 0:
        return "BEARISH"
    return "NEUTRAL"


def _structure_trend(frame: pd.DataFrame, lookback: int = 20) -> str:
    if len(frame) < lookback:
        return "NEUTRAL"
    recent = frame.tail(lookback)
    half = lookback // 2
    early = recent.iloc[:half]
    late = recent.iloc[half:]
    if late["high"].max() > early["high"].max() and late["low"].min() > early["low"].min():
        return "BULLISH"
    if late["high"].max() < early["high"].max() and late["low"].min() < early["low"].min():
        return "BEARISH"
    return "NEUTRAL"


def _max_consecutive(flags: Iterable[bool]) -> int:
    maximum = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def counter_dict(values: Iterable[object]) -> dict[str, int]:
    return dict(Counter(str(value) for value in values))
