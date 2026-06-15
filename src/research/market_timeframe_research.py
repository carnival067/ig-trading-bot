"""Offline market, timeframe, and data-quality research utilities."""

from __future__ import annotations

import gzip
import io
import csv
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.features import build_features
from src.research.strategy_discovery import (
    DiscoveryTrade,
    StrategySpec,
    metrics,
    prepare_frame,
    trade_records,
)

RISK_PER_TRADE = 0.002
TIMEFRAMES = {"15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
WINDOWS = {"wf_1": (0.20, 0.40), "wf_2": (0.40, 0.60), "wf_3": (0.60, 0.80), "oos": (0.80, 1.00)}


@dataclass(frozen=True)
class MarketCost:
    spread: float
    slippage_per_side: float
    commission_per_side_bps: float = 0.0


COSTS = {
    "EURUSD": MarketCost(0.00010, 0.00003),
    "GBPUSD": MarketCost(0.00012, 0.00004),
    "AUDUSD": MarketCost(0.00012, 0.00004),
    "USDJPY": MarketCost(0.010, 0.003),
    "XAUUSD": MarketCost(0.30, 0.10),
    "BTCUSDT": MarketCost(0.0, 0.0001, 10.0),
}


def resample_market(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" in data:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.set_index("timestamp")
    data.index = pd.to_datetime(data.index, utc=True)
    columns = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    result = data[list(columns)].resample(rule, label="right", closed="right").agg(columns)
    return prepare_frame(build_features(result.dropna(subset=["open", "high", "low", "close"])))


def higher_timeframe_specs() -> list[StrategySpec]:
    return [
        StrategySpec("TREND_CONTINUATION", "EMA50_200_PULLBACK", 1.5, 1.5, 20, _trend),
        StrategySpec("EMA_VWAP_PULLBACK", "EMA20_SESSION_MEAN", 1.5, 1.3, 16, _pullback),
        StrategySpec("COMPRESSION_BREAKOUT", "ATR_COMPRESSION_CLOSE", 1.5, 1.2, 16, _compression),
        StrategySpec("SUPPORT_RESISTANCE", "HTF_REJECTION", 1.5, 1.5, 20, _support_resistance),
        StrategySpec("RANGE_MEAN_REVERSION", "BB_RSI_RANGE", 1.0, 1.4, 16, _mean_reversion),
    ]


def run_market_spec(
    frame: pd.DataFrame,
    market: str,
    timeframe: str,
    spec: StrategySpec,
    window: str,
    *,
    include_costs: bool = True,
    cost_multiplier: float = 1.0,
    slippage_multiplier: float = 1.0,
    start_fraction: float | None = None,
    end_fraction: float | None = None,
) -> list[DiscoveryTrade]:
    if start_fraction is None or end_fraction is None:
        default_start, default_end = WINDOWS[window]
        start_fraction = default_start if start_fraction is None else start_fraction
        end_fraction = default_end if end_fraction is None else end_fraction
    signals = spec.signal_builder(frame)
    signal_values = signals["signal"].to_numpy()
    target_values = signals["target"].to_numpy(dtype=float)
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    atrs = frame["atr_14"].to_numpy(dtype=float)
    sessions = frame["session"].to_numpy()
    regimes = frame["market_regime"].to_numpy()
    index = frame.index
    start = max(250, int(len(frame) * start_fraction))
    end = min(len(frame) - 2, int(len(frame) * end_fraction))
    cost = COSTS[market] if include_costs else MarketCost(0.0, 0.0, 0.0)
    daily_r: dict[str, float] = {}
    daily_count: dict[str, int] = {}
    trades: list[DiscoveryTrade] = []
    cursor = start
    while cursor < end:
        direction = int(signal_values[cursor])
        if direction == 0:
            cursor += 1
            continue
        timestamp = index[cursor]
        if market != "BTCUSDT" and timestamp.weekday() >= 5:
            cursor += 1
            continue
        atr = atrs[cursor]
        close = closes[cursor]
        total_spread = cost.spread if market != "BTCUSDT" else close * 0.0002
        total_spread *= cost_multiplier
        if not np.isfinite(atr) or atr <= 0 or total_spread / atr > 0.20:
            cursor += 1
            continue
        day = timestamp.date().isoformat()
        if daily_count.get(day, 0) >= 3 or daily_r.get(day, 0) <= -5:
            cursor += 1
            continue
        entry_cursor = cursor + 1
        raw_entry = opens[entry_cursor]
        per_side = total_spread / 2 + (
            cost.slippage_per_side * cost_multiplier * slippage_multiplier
        )
        per_side += (
            raw_entry * cost.commission_per_side_bps * cost_multiplier / 10_000
        )
        entry = raw_entry + direction * per_side
        stop_distance = atr * spec.stop_atr
        stop = entry - direction * stop_distance
        mean_target = target_values[cursor]
        target = (
            mean_target
            if np.isfinite(mean_target) and direction * (mean_target - entry) > 0
            else entry + direction * stop_distance * spec.target_r
        )
        final_cursor = min(end, entry_cursor + spec.max_hold_bars)
        exit_cursor, exit_reason = final_cursor, "timeout"
        exit_price = closes[final_cursor] - direction * per_side
        for future in range(entry_cursor, final_cursor + 1):
            stop_hit = lows[future] <= stop if direction == 1 else highs[future] >= stop
            target_hit = highs[future] >= target if direction == 1 else lows[future] <= target
            if stop_hit:
                exit_cursor, exit_reason, exit_price = future, "stop", stop - direction * per_side
                break
            if target_hit:
                exit_cursor, exit_reason, exit_price = future, "target", target - direction * per_side
                break
        r_multiple = direction * (exit_price - entry) / stop_distance
        daily_r[day] = daily_r.get(day, 0.0) + r_multiple
        daily_count[day] = daily_count.get(day, 0) + 1
        entry_time, exit_time = index[entry_cursor], index[exit_cursor]
        regime = str(regimes[entry_cursor])
        trades.append(
            DiscoveryTrade(
                family=spec.family,
                variant=f"{spec.variant}_{timeframe}",
                pair=market,
                window=window,
                entry_time=entry_time.isoformat(),
                exit_time=exit_time.isoformat(),
                direction="BUY" if direction == 1 else "SELL",
                session=str(sessions[entry_cursor]),
                volatility_regime=regime,
                day_of_week=entry_time.day_name(),
                month=entry_time.strftime("%Y-%m"),
                r_multiple=r_multiple,
                holding_minutes=(exit_time - entry_time).total_seconds() / 60,
                exit_reason=exit_reason,
            )
        )
        cursor = exit_cursor + 1
    return trades


def add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    gap = (result["ema_50"] - result["ema_200"]).abs()
    trending = (gap > result["atr_14"] * 0.75) & (result["ema_200"].diff(3).abs() > 0)
    vol_rank = result["atr_pct"].rolling(200).rank(pct=True)
    result["market_regime"] = np.select(
        [vol_rank >= 0.75, vol_rank <= 0.25, trending],
        ["HIGH_VOLATILITY", "LOW_VOLATILITY", "TRENDING"],
        default="RANGING",
    )
    result["news_event_regime"] = "UNAVAILABLE"
    return result


def validate(cost: pd.DataFrame, zero: pd.DataFrame) -> tuple[str, str]:
    cm, zm = metrics(cost), metrics(zero)
    failures = []
    if cm["total_trades"] < 200:
        failures.append("fewer than 200 trades")
    if cm["profit_factor"] < 1.25:
        failures.append("profit factor below 1.25")
    if cm["expectancy"] <= 0:
        failures.append("non-positive expectancy after costs")
    if cm["max_drawdown"] >= 0.15:
        failures.append("maximum drawdown not below 15%")
    oos = cost[cost["window"] == "oos"] if not cost.empty else cost
    if metrics(oos)["total_return"] <= 0:
        failures.append("out-of-sample return is not positive")
    window_returns = cost.groupby("window")["r_multiple"].sum() if not cost.empty else pd.Series(dtype=float)
    if len(window_returns) < 4 or float((window_returns > 0).mean()) < 0.75:
        failures.append("walk-forward results unstable")
    if zm["expectancy"] > 0 >= cm["expectancy"]:
        failures.append("zero-cost edge fails after costs")
    if not cost.empty:
        positive = cost.loc[cost["r_multiple"] > 0, "r_multiple"].sum()
        if positive > 0 and cost.groupby("month")["r_multiple"].sum().clip(lower=0).max() / positive > 0.50:
            failures.append("profit concentrated in one month")
    return ("FAIL", "; ".join(dict.fromkeys(failures))) if failures else ("CANDIDATE", "all gates passed")


def load_binance_minutes(paths: list[Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            with archive.open(member) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8"))
                bucket = current = None
                for row in reader:
                    timestamp = int(row[0])
                    next_bucket = ((timestamp // 900_000) + 1) * 900_000
                    values = [float(row[i]) for i in range(1, 6)]
                    if next_bucket != bucket:
                        if current is not None:
                            records.append(current)
                        bucket = next_bucket
                        current = [bucket, values[0], values[1], values[2], values[3], values[4]]
                    else:
                        current[2] = max(current[2], values[1])
                        current[3] = min(current[3], values[2])
                        current[4] = values[3]
                        current[5] += values[4]
                if current is not None:
                    records.append(current)
    frame = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()


def load_histdata_minutes(paths: list[Path]) -> pd.DataFrame:
    frames = []
    names = ["timestamp", "open", "high", "low", "close", "volume"]
    for path in paths:
        frame = pd.read_csv(path, sep=";", header=None, names=names, compression="infer")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%Y%m%d %H%M%S", utc=True)
        frames.append(frame.set_index("timestamp"))
    return pd.concat(frames).sort_index().astype(float)


def _signals(frame: pd.DataFrame, long: pd.Series, short: pd.Series, target=np.nan) -> pd.DataFrame:
    signal = pd.Series(0, index=frame.index, dtype=int)
    signal[long.fillna(False)] = 1
    signal[short.fillna(False)] = -1
    target = target if isinstance(target, pd.Series) else pd.Series(target, index=frame.index)
    return pd.DataFrame({"signal": signal, "target": target}, index=frame.index)


def _trend(frame: pd.DataFrame) -> pd.DataFrame:
    long = (frame["ema_50"] > frame["ema_200"]) & (frame["ema_200"].diff(3) > 0)
    short = (frame["ema_50"] < frame["ema_200"]) & (frame["ema_200"].diff(3) < 0)
    return _signals(
        frame,
        long & (frame["low"] <= frame["ema_20"]) & (frame["close"] > frame["open"]) & (frame["close"] > frame["ema_20"]),
        short & (frame["high"] >= frame["ema_20"]) & (frame["close"] < frame["open"]) & (frame["close"] < frame["ema_20"]),
    )


def _pullback(frame: pd.DataFrame) -> pd.DataFrame:
    reference = frame["mean_reference"]
    return _signals(
        frame,
        (frame["ema_20"] > frame["ema_50"]) & (frame["low"] <= reference) & (frame["close"] > reference),
        (frame["ema_20"] < frame["ema_50"]) & (frame["high"] >= reference) & (frame["close"] < reference),
    )


def _compression(frame: pd.DataFrame) -> pd.DataFrame:
    rank = frame["atr_14"].rolling(100).rank(pct=True)
    compressed = rank.shift(1).rolling(6).min() < 0.2
    high, low = frame["high"].rolling(8).max().shift(1), frame["low"].rolling(8).min().shift(1)
    expansion = (frame["high"] - frame["low"]) > frame["atr_14"] * 1.25
    return _signals(frame, compressed & expansion & (frame["close"] > high), compressed & expansion & (frame["close"] < low))


def _support_resistance(frame: pd.DataFrame) -> pd.DataFrame:
    support, resistance = frame["low"].rolling(120).min().shift(1), frame["high"].rolling(120).max().shift(1)
    return _signals(
        frame,
        ((frame["low"] - support).abs() < frame["atr_14"] * 0.5) & (frame["lower_wick_ratio"] > 0.5) & (frame["close"] > frame["open"]),
        ((frame["high"] - resistance).abs() < frame["atr_14"] * 0.5) & (frame["upper_wick_ratio"] > 0.5) & (frame["close"] < frame["open"]),
    )


def _mean_reversion(frame: pd.DataFrame) -> pd.DataFrame:
    ranging = frame["market_regime"] == "RANGING"
    middle = (frame["bollinger_upper"] + frame["bollinger_lower"]) / 2
    return _signals(
        frame,
        ranging & (frame["close"] < frame["bollinger_lower"]) & (frame["rsi_14"] < 30),
        ranging & (frame["close"] > frame["bollinger_upper"]) & (frame["rsi_14"] > 70),
        middle,
    )


def records(trades: list[DiscoveryTrade]) -> pd.DataFrame:
    return trade_records(trades)
