"""Causal BOS and CHoCH confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StructureSignal:
    confirmed: bool
    direction: str
    event: str
    broken_level: float | None
    reason: str


class MarketStructureDetector:
    def __init__(self, lookback: int = 10) -> None:
        self.lookback = lookback

    def detect(
        self,
        frame: pd.DataFrame,
        direction: str,
        after_index: int | None = None,
    ) -> StructureSignal:
        if after_index is None or len(frame) < self.lookback + 2:
            return StructureSignal(False, direction, "NONE", None, "missing_sweep_context")
        start = max(after_index + 1, self.lookback)
        for index in range(start, len(frame)):
            history = frame.iloc[index - self.lookback : index]
            close = float(frame["close"].iloc[index])
            previous_trend = float(history["close"].iloc[-1] - history["close"].iloc[0])
            if direction == "BULLISH":
                level = float(history["high"].max())
                if close > level:
                    event = "CHOCH" if previous_trend < 0 else "BOS"
                    return StructureSignal(True, direction, event, level, "bullish_structure_break")
            elif direction == "BEARISH":
                level = float(history["low"].min())
                if close < level:
                    event = "CHOCH" if previous_trend > 0 else "BOS"
                    return StructureSignal(True, direction, event, level, "bearish_structure_break")
        return StructureSignal(False, direction, "NONE", None, "no_structure_break_after_sweep")
