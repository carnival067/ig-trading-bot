"""Fair value gap detection and pullback validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FairValueGap:
    detected: bool
    direction: str
    lower: float | None
    upper: float | None
    created_index: int | None
    retraced: bool


class FairValueGapDetector:
    def __init__(self, search_bars: int = 20) -> None:
        self.search_bars = search_bars

    def detect(self, frame: pd.DataFrame, direction: str) -> FairValueGap:
        start = max(2, len(frame) - self.search_bars)
        for index in range(len(frame) - 1, start - 1, -1):
            if direction == "BULLISH" and float(frame["low"].iloc[index]) > float(
                frame["high"].iloc[index - 2]
            ):
                lower = float(frame["high"].iloc[index - 2])
                upper = float(frame["low"].iloc[index])
            elif direction == "BEARISH" and float(frame["high"].iloc[index]) < float(
                frame["low"].iloc[index - 2]
            ):
                lower = float(frame["high"].iloc[index])
                upper = float(frame["low"].iloc[index - 2])
            else:
                continue
            recent = frame.iloc[index + 1 :]
            retraced = (
                not recent.empty
                and float(recent["low"].min()) <= upper
                and float(recent["high"].max()) >= lower
            )
            return FairValueGap(True, direction, lower, upper, index, retraced)
        return FairValueGap(False, direction, None, None, None, False)
