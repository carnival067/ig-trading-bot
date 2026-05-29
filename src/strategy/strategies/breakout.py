"""Breakout strategy implementation.

Uses range detection (20-period high/low), volume confirmation, and
ATR-based false breakout filtering to generate signals on range breaks.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy using range detection and volume confirmation.

    Signal generation logic:
        - LONG: Close breaks above 20-period high with volume confirmation
        - SHORT: Close breaks below 20-period low with volume confirmation

    Filters:
        - Volume must be above average (confirmation)
        - False breakout filter: breakout distance must exceed ATR threshold
        - Best suited for RANGING regime transitioning to TRENDING
    """

    name: str = "breakout"

    def __init__(
        self,
        lookback_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        false_breakout_atr_threshold: float = 0.5,
    ) -> None:
        self.lookback_period = lookback_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.false_breakout_atr_threshold = false_breakout_atr_threshold

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate breakout signal based on range break with volume.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if breakout conditions met, None otherwise.
        """
        min_periods = max(self.lookback_period, self.atr_period) + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        atr = indicators["atr"]
        if atr <= 0:
            return None

        close = market_data["close"].iloc[-1]

        # Check for bullish breakout
        if (
            indicators["breakout_bullish"]
            and indicators["volume_confirmed"]
            and indicators["not_false_breakout_bull"]
        ):
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        # Check for bearish breakout
        elif (
            indicators["breakout_bearish"]
            and indicators["volume_confirmed"]
            and indicators["not_false_breakout_bear"]
        ):
            direction = "SHORT"
            entry_price = Decimal(str(close))
            stop_loss = entry_price + Decimal(str(atr * self.atr_sl_multiplier))
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * Decimal(str(self.risk_reward_ratio)))
        else:
            return None

        instrument = market_data.attrs.get("instrument", "UNKNOWN")

        return TradeSignal(
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_inputs=indicators,
            strategy_name=self.name,
            regime=regime.value,
        )

    def get_indicators(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate breakout indicators.

        Returns:
            Dictionary with: range_high, range_low, atr, volume_ratio,
            breakout_bullish, breakout_bearish, volume_confirmed,
            not_false_breakout_bull, not_false_breakout_bear.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]
        volume = market_data["volume"]

        # Range detection (excluding current bar)
        range_high = high.iloc[-(self.lookback_period + 1) : -1].max()
        range_low = low.iloc[-(self.lookback_period + 1) : -1].min()

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Volume confirmation
        avg_volume = volume.iloc[-(self.lookback_period + 1) : -1].mean()
        current_volume = volume.iloc[-1]
        volume_ratio = (
            float(current_volume / avg_volume) if avg_volume > 0 else 0.0
        )
        volume_confirmed = volume_ratio >= self.volume_multiplier

        # Breakout detection
        current_close = float(close.iloc[-1])
        breakout_bullish = current_close > float(range_high)
        breakout_bearish = current_close < float(range_low)

        # False breakout filter: breakout distance must exceed threshold * ATR
        breakout_distance_bull = current_close - float(range_high) if breakout_bullish else 0.0
        breakout_distance_bear = float(range_low) - current_close if breakout_bearish else 0.0

        not_false_breakout_bull = breakout_distance_bull > (
            atr * self.false_breakout_atr_threshold
        )
        not_false_breakout_bear = breakout_distance_bear > (
            atr * self.false_breakout_atr_threshold
        )

        return {
            "range_high": float(range_high),
            "range_low": float(range_low),
            "atr": float(atr),
            "volume_ratio": float(volume_ratio),
            "breakout_bullish": float(breakout_bullish),
            "breakout_bearish": float(breakout_bearish),
            "volume_confirmed": float(volume_confirmed),
            "not_false_breakout_bull": float(not_false_breakout_bull),
            "not_false_breakout_bear": float(not_false_breakout_bear),
        }

    @staticmethod
    def _calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> float:
        """Calculate Average True Range."""
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
