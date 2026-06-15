"""Displacement-based order block approximation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OrderBlock:
    detected: bool
    direction: str
    lower: float | None
    upper: float | None
    created_index: int | None
    retraced: bool


class OrderBlockDetector:
    def __init__(self, search_bars: int = 20, displacement_atr: float = 1.0) -> None:
        self.search_bars = search_bars
        self.displacement_atr = displacement_atr

    def detect(self, frame: pd.DataFrame, direction: str, atr: float) -> OrderBlock:
        start = max(1, len(frame) - self.search_bars)
        for index in range(len(frame) - 2, start - 1, -1):
            candle = frame.iloc[index]
            next_candle = frame.iloc[index + 1]
            body = abs(float(next_candle["close"] - next_candle["open"]))
            if body < atr * self.displacement_atr:
                continue
            bullish = direction == "BULLISH" and float(candle["close"]) < float(candle["open"])
            bearish = direction == "BEARISH" and float(candle["close"]) > float(candle["open"])
            if not bullish and not bearish:
                continue
            lower = float(candle["low"])
            upper = float(candle["high"])
            recent = frame.iloc[index + 2 :]
            retraced = (
                not recent.empty
                and float(recent["low"].min()) <= upper
                and float(recent["high"].max()) >= lower
            )
            return OrderBlock(True, direction, lower, upper, index, retraced)
        return OrderBlock(False, direction, None, None, None, False)
