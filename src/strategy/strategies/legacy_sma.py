"""Legacy SMA/ATR strategy retained for controlled fallback and comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd


class LegacySMAStrategy:
    name = "legacy_sma_atr"

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 25,
        atr_period: int = 14,
        trend_threshold: float = 0.3,
        stop_atr: float = 3.0,
        target_atr: float = 6.0,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.trend_threshold = trend_threshold
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, frame: pd.DataFrame) -> dict[str, Any] | None:
        if len(frame) < self.slow_period:
            return None
        close = frame["close"]
        fast = float(close.tail(self.fast_period).mean())
        slow = float(close.tail(self.slow_period).mean())
        previous = close.shift(1)
        ranges = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(ranges.tail(self.atr_period).mean())
        if atr <= 0:
            return None
        strength = abs(fast - slow) / atr
        if strength < self.trend_threshold or fast == slow:
            return None
        direction = "BUY" if fast > slow else "SELL"
        return {
            "direction": direction,
            "current_price": float(close.iloc[-1]),
            "stop_distance": atr * self.stop_atr,
            "limit_distance": atr * self.target_atr,
            "atr": atr,
            "sma_fast": fast,
            "sma_slow": slow,
            "trend_strength": strength,
            "confidence": min(95, int(50 + strength * 20)),
            "rr_ratio": self.target_atr / self.stop_atr,
            "strategy_name": self.name,
            "risk_per_trade": 0.01,
        }
