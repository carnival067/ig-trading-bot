"""Higher-timeframe directional bias detection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrendBias:
    direction: str
    timeframe: str
    fast_ema: float
    slow_ema: float
    strength: float
    reason: str


class HigherTimeframeTrend:
    """Determine bias from 4H first, then 1H when 4H is unavailable."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def detect(self, four_hour: pd.DataFrame, one_hour: pd.DataFrame) -> TrendBias:
        for timeframe, frame in (("4H", four_hour), ("1H", one_hour)):
            if len(frame) < self.slow_period + 3:
                continue
            close = frame["close"]
            fast = close.ewm(span=self.fast_period, adjust=False).mean()
            slow = close.ewm(span=self.slow_period, adjust=False).mean()
            current_fast = float(fast.iloc[-1])
            current_slow = float(slow.iloc[-1])
            slope = float(slow.iloc[-1] - slow.iloc[-3])
            denominator = max(abs(current_slow), 1e-12)
            strength = abs(current_fast - current_slow) / denominator
            if current_fast > current_slow and slope > 0:
                direction = "BULLISH"
            elif current_fast < current_slow and slope < 0:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
            return TrendBias(
                direction=direction,
                timeframe=timeframe,
                fast_ema=current_fast,
                slow_ema=current_slow,
                strength=strength,
                reason=f"{timeframe} EMA{self.fast_period}/EMA{self.slow_period} with slope",
            )
        return TrendBias("NEUTRAL", "NONE", 0.0, 0.0, 0.0, "insufficient_higher_timeframe_data")
