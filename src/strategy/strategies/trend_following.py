"""Trend Following strategy implementation.

Uses EMA crossover (10/21), ADX confirmation (>25), and trend strength
filtering to generate signals in trending markets.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class TrendFollowingStrategy(BaseStrategy):
    """Trend Following strategy using MA crossover with ADX confirmation.

    Signal generation logic:
        - LONG: EMA10 crosses above EMA21, ADX > 25, close > EMA21
        - SHORT: EMA10 crosses below EMA21, ADX > 25, close < EMA21

    Filters:
        - ADX must be above threshold (trend confirmation)
        - Trend strength: price must be on the correct side of the slow EMA
        - Best suited for TRENDING regime
    """

    name: str = "trend_following"

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 21,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate trend following signal based on EMA crossover.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if crossover conditions met, None otherwise.
        """
        if len(market_data) < self.slow_period + 2:
            return None

        indicators = self.get_indicators(market_data)

        # Check ADX confirmation
        if indicators["adx"] <= self.adx_threshold:
            return None

        # Check EMA crossover
        if not indicators["ema_crossover_bullish"] and not indicators["ema_crossover_bearish"]:
            return None

        close = market_data["close"].iloc[-1]
        atr = indicators["atr"]

        if atr <= 0:
            return None

        if indicators["ema_crossover_bullish"] and indicators["trend_strength_bullish"]:
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        elif indicators["ema_crossover_bearish"] and indicators["trend_strength_bearish"]:
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
        """Calculate trend following indicators.

        Returns:
            Dictionary with: ema_fast, ema_slow, adx, atr,
            ema_crossover_bullish, ema_crossover_bearish,
            trend_strength_bullish, trend_strength_bearish.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]

        # EMAs
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        # ADX calculation
        adx = self._calculate_adx(high, low, close, self.adx_period)

        # ATR calculation
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Crossover detection (current bar vs previous bar)
        ema_crossover_bullish = (
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
        )
        ema_crossover_bearish = (
            ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]
        )

        # Trend strength: price on correct side of slow EMA
        trend_strength_bullish = close.iloc[-1] > ema_slow.iloc[-1]
        trend_strength_bearish = close.iloc[-1] < ema_slow.iloc[-1]

        return {
            "ema_fast": float(ema_fast.iloc[-1]),
            "ema_slow": float(ema_slow.iloc[-1]),
            "adx": float(adx),
            "atr": float(atr),
            "ema_crossover_bullish": float(ema_crossover_bullish),
            "ema_crossover_bearish": float(ema_crossover_bearish),
            "trend_strength_bullish": float(trend_strength_bullish),
            "trend_strength_bearish": float(trend_strength_bearish),
        }

    @staticmethod
    def _calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> float:
        """Calculate the Average Directional Index (ADX)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

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
