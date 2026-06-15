"""Causal feature engineering for historical FX research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gains / losses.replace(0, np.nan)))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Create technical and market-structure features using past/current bars only."""
    frame = candles.copy()
    close = frame["close"]
    for period in (9, 20, 50, 200):
        frame[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()
        frame[f"ema_{period}_distance"] = close / frame[f"ema_{period}"] - 1

    frame["rsi_14"] = _rsi(close)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_histogram"] = frame["macd"] - frame["macd_signal"]
    frame["atr_14"] = _atr(frame)
    frame["atr_pct"] = frame["atr_14"] / close

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    frame["bollinger_upper"] = middle + 2 * deviation
    frame["bollinger_lower"] = middle - 2 * deviation
    frame["bollinger_position"] = (close - frame["bollinger_lower"]) / (
        frame["bollinger_upper"] - frame["bollinger_lower"]
    ).replace(0, np.nan)
    frame["bollinger_width"] = (frame["bollinger_upper"] - frame["bollinger_lower"]) / middle

    frame["return_1"] = close.pct_change()
    frame["return_5"] = close.pct_change(5)
    frame["return_20"] = close.pct_change(20)
    frame["realized_volatility_20"] = frame["return_1"].rolling(20).std()
    rolling_vol = frame["realized_volatility_20"]
    frame["volatility_regime"] = rolling_vol / rolling_vol.rolling(200).median()

    body = frame["close"] - frame["open"]
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["body_ratio"] = body / candle_range
    frame["upper_wick_ratio"] = (
        frame["high"] - frame[["open", "close"]].max(axis=1)
    ) / candle_range
    frame["lower_wick_ratio"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / candle_range
    frame["doji"] = (body.abs() / candle_range < 0.1).astype(int)
    frame["bullish_engulfing"] = (
        (body > 0)
        & (body.shift(1) < 0)
        & (frame["open"] <= frame["close"].shift(1))
        & (frame["close"] >= frame["open"].shift(1))
    ).astype(int)

    prior_high = frame["high"].rolling(20).max().shift(1)
    prior_low = frame["low"].rolling(20).min().shift(1)
    frame["distance_to_resistance"] = prior_high / close - 1
    frame["distance_to_support"] = close / prior_low - 1
    frame["break_of_structure_up"] = (close > prior_high).astype(int)
    frame["break_of_structure_down"] = (close < prior_low).astype(int)
    trend = np.sign(frame["ema_20"] - frame["ema_50"])
    frame["change_of_character"] = trend.diff().ne(0).astype(int)

    frame["swing_high"] = (
        (frame["high"] > frame["high"].shift(1))
        & (frame["high"] > frame["high"].shift(2))
        & (frame["high"] >= frame["high"].shift(-1))
        & (frame["high"] >= frame["high"].shift(-2))
    ).shift(2).fillna(False).astype(int)
    frame["swing_low"] = (
        (frame["low"] < frame["low"].shift(1))
        & (frame["low"] < frame["low"].shift(2))
        & (frame["low"] <= frame["low"].shift(-1))
        & (frame["low"] <= frame["low"].shift(-2))
    ).shift(2).fillna(False).astype(int)
    frame["fair_value_gap_up"] = (frame["low"] > frame["high"].shift(2)).astype(int)
    frame["fair_value_gap_down"] = (frame["high"] < frame["low"].shift(2)).astype(int)
    frame["liquidity_sweep_high"] = (
        (frame["high"] > prior_high) & (close < prior_high)
    ).astype(int)
    frame["liquidity_sweep_low"] = ((frame["low"] < prior_low) & (close > prior_low)).astype(int)
    frame["bullish_order_block"] = (
        (body.shift(1) < 0) & (close > frame["high"].shift(1)) & (frame["return_1"] > 0)
    ).astype(int)
    frame["bearish_order_block"] = (
        (body.shift(1) > 0) & (close < frame["low"].shift(1)) & (frame["return_1"] < 0)
    ).astype(int)

    volume_mean = frame["volume"].rolling(20).mean()
    volume_std = frame["volume"].rolling(20).std()
    frame["volume_zscore"] = (frame["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    if "spread" in frame:
        frame["spread_estimate"] = frame["spread"]
        frame["spread_atr_ratio"] = frame["spread"] / frame["atr_14"]
    else:
        frame["spread_estimate"] = np.nan
        frame["spread_atr_ratio"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    "ema_9_distance",
    "ema_20_distance",
    "ema_50_distance",
    "ema_200_distance",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_pct",
    "bollinger_position",
    "bollinger_width",
    "return_1",
    "return_5",
    "return_20",
    "realized_volatility_20",
    "volatility_regime",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "doji",
    "bullish_engulfing",
    "distance_to_resistance",
    "distance_to_support",
    "break_of_structure_up",
    "break_of_structure_down",
    "change_of_character",
    "swing_high",
    "swing_low",
    "fair_value_gap_up",
    "fair_value_gap_down",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
    "bullish_order_block",
    "bearish_order_block",
    "volume_zscore",
    "spread_atr_ratio",
]
