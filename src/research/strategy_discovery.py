"""Offline, cost-aware discovery of independent rule-based strategy families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

RISK_PER_TRADE = 0.002
MAX_DAILY_LOSS = 0.01
MAX_DAILY_TRADES = 3
WINDOWS = {
    "wf_1": (0.20, 0.40),
    "wf_2": (0.40, 0.60),
    "wf_3": (0.60, 0.80),
    "oos": (0.80, 1.00),
}


@dataclass(frozen=True)
class StrategySpec:
    family: str
    variant: str
    target_r: float
    stop_atr: float
    max_hold_bars: int
    signal_builder: Callable[[pd.DataFrame], pd.DataFrame]
    short_hold: bool = False


@dataclass
class DiscoveryTrade:
    family: str
    variant: str
    pair: str
    window: str
    entry_time: str
    exit_time: str
    direction: str
    session: str
    volatility_regime: str
    day_of_week: str
    month: str
    r_multiple: float
    holding_minutes: float
    exit_reason: str


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" in data:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.set_index("timestamp")
    data = data.sort_index()
    data["hour"] = data.index.hour
    data["session"] = np.select(
        [
            data["hour"].between(0, 6),
            data["hour"].between(7, 11),
            data["hour"].between(12, 16),
            data["hour"].between(17, 21),
        ],
        ["ASIAN", "LONDON", "OVERLAP", "NEW_YORK"],
        default="OFF_HOURS",
    )
    data["day"] = data.index.floor("D")
    data["volatility_bucket"] = pd.cut(
        data["volatility_regime"],
        [-np.inf, 0.8, 1.2, np.inf],
        labels=["LOW", "NORMAL", "HIGH"],
    ).astype("object").fillna("UNKNOWN")
    hourly = data["close"].resample("1h").last().dropna()
    four_hour = data["close"].resample("4h").last().dropna()
    for prefix, series in (("1h", hourly), ("4h", four_hour)):
        fast = series.ewm(span=50, adjust=False).mean()
        slow = series.ewm(span=200, adjust=False).mean()
        trend = pd.Series(0, index=series.index, dtype=int)
        trend[(fast > slow) & (slow.diff(3) > 0)] = 1
        trend[(fast < slow) & (slow.diff(3) < 0)] = -1
        data[f"trend_{prefix}"] = trend.reindex(data.index, method="ffill").fillna(0)
    date = pd.Series(data.index.floor("D"), index=data.index)
    typical = (data["high"] + data["low"] + data["close"]) / 3
    positive_volume = data["volume"].clip(lower=0)
    cumulative_volume = positive_volume.groupby(date).cumsum()
    vwap = (typical * positive_volume).groupby(date).cumsum() / cumulative_volume.replace(0, np.nan)
    data["vwap"] = vwap
    data["session_mean"] = typical.groupby([date, data["session"]]).expanding().mean().droplevel([0, 1])
    positive_volume_coverage = float((positive_volume > 0).mean())
    data["vwap_reliable"] = positive_volume_coverage >= 0.8
    data["mean_reference"] = data["vwap"].where(data["vwap_reliable"], data["session_mean"])
    return data


def strategy_specs() -> list[StrategySpec]:
    specs = [
        StrategySpec("TREND_CONTINUATION", f"EMA_PULLBACK_{target:g}R", target, 1.5, 72, _trend_signal)
        for target in (1.0, 1.5, 2.0)
    ]
    specs.extend(
        [
            StrategySpec("MEAN_REVERSION", "BOLLINGER_TO_MEAN", 1.0, 1.5, 48, _mean_reversion_signal),
            StrategySpec("BREAKOUT", "ASIAN_RANGE_CLOSE", 1.5, 1.2, 48, lambda f: _breakout_signal(f, "ASIAN", False)),
            StrategySpec("BREAKOUT", "ASIAN_RANGE_RETEST", 1.5, 1.2, 48, lambda f: _breakout_signal(f, "ASIAN", True)),
            StrategySpec("BREAKOUT", "LONDON_OPEN_CLOSE", 1.5, 1.2, 36, lambda f: _breakout_signal(f, "LONDON", False)),
            StrategySpec("BREAKOUT", "LONDON_OPEN_RETEST", 1.5, 1.2, 36, lambda f: _breakout_signal(f, "LONDON", True)),
            StrategySpec("BREAKOUT", "NEW_YORK_OPEN_CLOSE", 1.5, 1.2, 36, lambda f: _breakout_signal(f, "NEW_YORK", False)),
            StrategySpec("BREAKOUT", "NEW_YORK_OPEN_RETEST", 1.5, 1.2, 36, lambda f: _breakout_signal(f, "NEW_YORK", True)),
            StrategySpec("VOLATILITY_EXPANSION", "COMPRESSION_BREAK_1.5R", 1.5, 1.2, 48, _compression_signal),
            StrategySpec("VOLATILITY_EXPANSION", "COMPRESSION_BREAK_2R", 2.0, 1.2, 60, _compression_signal),
            StrategySpec("SUPPORT_RESISTANCE", "HTF_REJECTION", 1.5, 1.5, 72, _support_resistance_signal),
            StrategySpec("VWAP_SESSION_MEAN", "TREND_PULLBACK", 1.5, 1.2, 36, _session_mean_trend),
            StrategySpec("VWAP_SESSION_MEAN", "RANGE_REVERSION", 1.0, 1.2, 24, _session_mean_reversion),
            StrategySpec("COST_AWARE_SCALPING", "SHORT_HOLD_REJECTION", 1.0, 0.8, 6, _scalping_signal, True),
        ]
    )
    return specs


def run_spec(
    frame: pd.DataFrame,
    pair: str,
    spec: StrategySpec,
    window: str,
    *,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.3,
) -> list[DiscoveryTrade]:
    start_fraction, end_fraction = WINDOWS[window]
    signals = spec.signal_builder(frame)
    start = max(250, int(len(frame) * start_fraction))
    end = min(len(frame) - 2, int(len(frame) * end_fraction))
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    cost = (spread_pips / 2 + slippage_pips) * pip
    daily_r: dict[str, float] = {}
    daily_count: dict[str, int] = {}
    trades: list[DiscoveryTrade] = []
    cursor = start
    while cursor < end:
        direction = int(signals["signal"].iloc[cursor])
        if direction == 0:
            cursor += 1
            continue
        timestamp = frame.index[cursor]
        if timestamp.weekday() >= 5:
            cursor += 1
            continue
        day = timestamp.date().isoformat()
        if daily_count.get(day, 0) >= MAX_DAILY_TRADES or daily_r.get(day, 0) <= -5:
            cursor += 1
            continue
        atr = float(frame["atr_14"].iloc[cursor])
        if not np.isfinite(atr) or atr <= 0:
            cursor += 1
            continue
        entry_cursor = cursor + 1
        if frame.index[entry_cursor].weekday() >= 5:
            cursor += 1
            continue
        entry = float(frame["open"].iloc[entry_cursor]) + direction * cost
        stop_distance = atr * spec.stop_atr
        stop = entry - direction * stop_distance
        mean_target = float(signals["target"].iloc[cursor])
        if np.isfinite(mean_target) and direction * (mean_target - entry) > 0:
            target = mean_target
        else:
            target = entry + direction * stop_distance * spec.target_r
        target_distance = abs(target - entry)
        final_cursor = min(end, entry_cursor + spec.max_hold_bars)
        exit_price = float(frame["close"].iloc[final_cursor]) - direction * cost
        exit_reason = "timeout"
        exit_cursor = final_cursor
        for future in range(entry_cursor, final_cursor + 1):
            bar = frame.iloc[future]
            stop_hit = float(bar["low"]) <= stop if direction == 1 else float(bar["high"]) >= stop
            target_hit = float(bar["high"]) >= target if direction == 1 else float(bar["low"]) <= target
            if stop_hit:
                exit_price = stop - direction * cost
                exit_reason = "stop"
                exit_cursor = future
                break
            if target_hit:
                exit_price = target - direction * cost
                exit_reason = "target"
                exit_cursor = future
                break
        r_multiple = direction * (exit_price - entry) / stop_distance
        daily_r[day] = daily_r.get(day, 0.0) + r_multiple
        daily_count[day] = daily_count.get(day, 0) + 1
        entry_time = frame.index[entry_cursor]
        exit_time = frame.index[exit_cursor]
        trades.append(
            DiscoveryTrade(
                family=spec.family,
                variant=spec.variant,
                pair=pair,
                window=window,
                entry_time=entry_time.isoformat(),
                exit_time=exit_time.isoformat(),
                direction="BUY" if direction == 1 else "SELL",
                session=str(frame["session"].iloc[entry_cursor]),
                volatility_regime=str(frame["volatility_bucket"].iloc[entry_cursor]),
                day_of_week=entry_time.day_name(),
                month=entry_time.strftime("%Y-%m"),
                r_multiple=r_multiple,
                holding_minutes=(exit_time - entry_time).total_seconds() / 60,
                exit_reason=exit_reason,
            )
        )
        cursor = exit_cursor + 1
    return trades


def metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return _empty_metrics()
    ordered = trades.assign(entry_time=pd.to_datetime(trades["entry_time"], utc=True)).sort_values("entry_time")
    values = ordered["r_multiple"].astype(float).to_numpy()
    winners, losers = values[values > 0], values[values < 0]
    gross_profit, gross_loss = float(winners.sum()), abs(float(losers.sum()))
    sleeves = ordered["pair"].unique()
    equity = {pair: 1.0 for pair in sleeves}
    curve = []
    for _, row in ordered.iterrows():
        equity[row["pair"]] *= 1 + float(row["r_multiple"]) * RISK_PER_TRADE
        curve.append(sum(equity.values()) / len(equity))
    curve_array = np.asarray(curve)
    peaks = np.maximum.accumulate(np.insert(curve_array, 0, 1.0))[1:]
    return {
        "total_trades": len(values),
        "total_return": sum(equity.values()) / len(equity) - 1,
        "win_rate": float((values > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "max_drawdown": float(np.max((peaks - curve_array) / peaks)),
        "average_r": float(values.mean()),
        "median_r": float(np.median(values)),
        "expectancy": float(values.mean()),
        "average_holding_minutes": float(ordered["holding_minutes"].mean()),
    }


def validation_status(all_trades: pd.DataFrame, cost_metrics: dict, zero_metrics: dict) -> tuple[str, str]:
    failures = []
    if cost_metrics["total_trades"] < 200:
        failures.append("fewer than 200 trades")
    if cost_metrics["profit_factor"] < 1.25:
        failures.append("profit factor below 1.25")
    if cost_metrics["expectancy"] <= 0:
        failures.append("non-positive cost-adjusted expectancy")
    if cost_metrics["max_drawdown"] > 0.15:
        failures.append("maximum drawdown above 15%")
    if zero_metrics["expectancy"] > 0 and cost_metrics["expectancy"] <= 0:
        failures.append("zero-cost edge does not survive costs")
    window_r = all_trades.groupby(["pair", "window"])["r_multiple"].sum()
    if len(window_r) == 0 or float((window_r > 0).mean()) < 0.75:
        failures.append("walk-forward/OOS instability")
    positive = all_trades.loc[all_trades["r_multiple"] > 0, "r_multiple"].sum()
    if positive > 0:
        pair_share = all_trades.groupby("pair")["r_multiple"].sum().clip(lower=0).max() / positive
        month_share = all_trades.groupby("month")["r_multiple"].sum().clip(lower=0).max() / positive
        if pair_share > 0.50:
            failures.append("profit concentrated in one pair")
        if month_share > 0.50:
            failures.append("profit concentrated in one month")
    return ("FAIL", "; ".join(dict.fromkeys(failures))) if failures else ("CANDIDATE", "all research gates passed")


def breakdown(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    return {str(value): metrics(group) for value, group in frame.groupby(column)}


def trade_records(trades: list[DiscoveryTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(trade) for trade in trades])


def _signal_frame(frame: pd.DataFrame, long: pd.Series, short: pd.Series, target: pd.Series | float = np.nan) -> pd.DataFrame:
    signal = pd.Series(0, index=frame.index, dtype=int)
    signal[long.fillna(False)] = 1
    signal[short.fillna(False)] = -1
    target_series = target if isinstance(target, pd.Series) else pd.Series(target, index=frame.index)
    return pd.DataFrame({"signal": signal, "target": target_series}, index=frame.index)


def _trend_signal(frame: pd.DataFrame) -> pd.DataFrame:
    aligned_long = (frame["trend_4h"] == 1) & (frame["trend_1h"] == 1)
    aligned_short = (frame["trend_4h"] == -1) & (frame["trend_1h"] == -1)
    pullback = frame["low"] <= frame[["ema_20", "ema_50"]].max(axis=1)
    pullback_short = frame["high"] >= frame[["ema_20", "ema_50"]].min(axis=1)
    return _signal_frame(
        frame,
        aligned_long & pullback & (frame["close"] > frame["open"]) & (frame["close"] > frame["ema_20"]),
        aligned_short & pullback_short & (frame["close"] < frame["open"]) & (frame["close"] < frame["ema_20"]),
    )


def _mean_reversion_signal(frame: pd.DataFrame) -> pd.DataFrame:
    range_bound = (frame["trend_4h"] == 0) & (frame["volatility_regime"] < 1.1)
    long = range_bound & (frame["close"] < frame["bollinger_lower"]) & (frame["rsi_14"] < 30)
    short = range_bound & (frame["close"] > frame["bollinger_upper"]) & (frame["rsi_14"] > 70)
    middle = (frame["bollinger_upper"] + frame["bollinger_lower"]) / 2
    return _signal_frame(frame, long, short, middle)


def _breakout_signal(frame: pd.DataFrame, session: str, retest: bool) -> pd.DataFrame:
    ranges = {"ASIAN": (0, 7, 7, 12), "LONDON": (7, 8, 8, 12), "NEW_YORK": (12, 13, 13, 17)}
    start, finish, trade_start, trade_finish = ranges[session]
    range_mask = frame["hour"].between(start, finish - 1)
    high = frame["high"].where(range_mask).groupby(frame["day"]).transform("max")
    low = frame["low"].where(range_mask).groupby(frame["day"]).transform("min")
    active = frame["hour"].between(trade_start, trade_finish - 1)
    expansion = (frame["high"] - frame["low"]) > frame["atr_14"] * 1.1
    long = active & expansion & (frame["close"] > high)
    short = active & expansion & (frame["close"] < low)
    if retest:
        long &= (frame["low"] <= high) & (frame["close"] > high)
        short &= (frame["high"] >= low) & (frame["close"] < low)
    return _signal_frame(frame, long, short)


def _compression_signal(frame: pd.DataFrame) -> pd.DataFrame:
    atr_rank = frame["atr_14"].rolling(100).rank(pct=True)
    compressed = atr_rank.shift(1).rolling(12).min() < 0.2
    prior_high = frame["high"].rolling(12).max().shift(1)
    prior_low = frame["low"].rolling(12).min().shift(1)
    expansion = (frame["high"] - frame["low"]) > frame["atr_14"] * 1.3
    return _signal_frame(frame, compressed & expansion & (frame["close"] > prior_high), compressed & expansion & (frame["close"] < prior_low))


def _support_resistance_signal(frame: pd.DataFrame) -> pd.DataFrame:
    support = frame["low"].rolling(576).min().shift(1)
    resistance = frame["high"].rolling(576).max().shift(1)
    near_support = (frame["low"] - support).abs() <= frame["atr_14"] * 0.5
    near_resistance = (frame["high"] - resistance).abs() <= frame["atr_14"] * 0.5
    long = near_support & (frame["lower_wick_ratio"] > 0.5) & (frame["close"] > frame["open"]) & ((resistance - frame["close"]) > frame["atr_14"] * 2)
    short = near_resistance & (frame["upper_wick_ratio"] > 0.5) & (frame["close"] < frame["open"]) & ((frame["close"] - support) > frame["atr_14"] * 2)
    return _signal_frame(frame, long, short)


def _session_mean_trend(frame: pd.DataFrame) -> pd.DataFrame:
    reference = frame["mean_reference"]
    long = (frame["trend_1h"] == 1) & (frame["low"] <= reference) & (frame["close"] > reference) & (frame["close"] > frame["open"])
    short = (frame["trend_1h"] == -1) & (frame["high"] >= reference) & (frame["close"] < reference) & (frame["close"] < frame["open"])
    return _signal_frame(frame, long, short)


def _session_mean_reversion(frame: pd.DataFrame) -> pd.DataFrame:
    reference = frame["mean_reference"]
    range_bound = frame["trend_1h"] == 0
    long = range_bound & (frame["close"] < reference - frame["atr_14"]) & (frame["rsi_14"] < 35)
    short = range_bound & (frame["close"] > reference + frame["atr_14"]) & (frame["rsi_14"] > 65)
    return _signal_frame(frame, long, short, reference)


def _scalping_signal(frame: pd.DataFrame) -> pd.DataFrame:
    liquid = frame["session"].isin(["LONDON", "OVERLAP"])
    long = liquid & (frame["liquidity_sweep_low"] == 1) & (frame["close"] > frame["open"])
    short = liquid & (frame["liquidity_sweep_high"] == 1) & (frame["close"] < frame["open"])
    return _signal_frame(frame, long, short)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "total_trades": 0, "total_return": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "average_r": 0.0, "median_r": 0.0, "expectancy": 0.0,
        "average_holding_minutes": 0.0,
    }
