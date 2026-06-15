"""Liquidity sweep detection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LiquiditySweep:
    detected: bool
    direction: str
    level: float | None
    extreme: float | None
    bar_index: int | None
    reason: str


class LiquiditySweepDetector:
    def __init__(self, liquidity_lookback: int = 20, search_bars: int = 8) -> None:
        self.liquidity_lookback = liquidity_lookback
        self.search_bars = search_bars

    def detect(self, frame: pd.DataFrame, direction: str) -> LiquiditySweep:
        minimum = self.liquidity_lookback + self.search_bars
        if len(frame) < minimum:
            return LiquiditySweep(False, direction, None, None, None, "insufficient_data")
        start = len(frame) - self.search_bars
        for index in range(start, len(frame)):
            history = frame.iloc[index - self.liquidity_lookback : index]
            bar = frame.iloc[index]
            if direction == "BULLISH":
                level = float(history["low"].min())
                if float(bar["low"]) < level and float(bar["close"]) > level:
                    return LiquiditySweep(
                        True, direction, level, float(bar["low"]), index, "sell_side_sweep_reclaimed"
                    )
            elif direction == "BEARISH":
                level = float(history["high"].max())
                if float(bar["high"]) > level and float(bar["close"]) < level:
                    return LiquiditySweep(
                        True, direction, level, float(bar["high"]), index, "buy_side_sweep_rejected"
                    )
        return LiquiditySweep(False, direction, None, None, None, "no_recent_liquidity_sweep")
