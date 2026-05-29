"""Mean Reversion strategy implementation.

Uses Bollinger Band extremes (2σ), RSI divergence (<30/>70), and mean
distance to generate signals when price deviates significantly from the mean.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion strategy using Bollinger Bands and RSI.

    Signal generation logic:
        - LONG: Price touches/crosses lower BB AND RSI < 30 (oversold)
        - SHORT: Price touches/crosses upper BB AND RSI > 70 (overbought)

    Filters:
        - Mean distance must be significant (> 1.5 std devs from mean)
        - Best suited for RANGING regime
    """

    name: str = "mean_reversion"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        mean_distance_threshold: float = 1.5,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.mean_distance_threshold = mean_distance_threshold

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate mean reversion signal based on BB extremes and RSI.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if mean reversion conditions met, None otherwise.
        """
        min_periods = max(self.bb_period, self.rsi_period, self.atr_period) + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        close = market_data["close"].iloc[-1]
        atr = indicators["atr"]

        if atr <= 0:
            return None

        # Check for long signal: price at lower BB + RSI oversold
        if indicators["bb_lower_touch"] and indicators["rsi_oversold"]:
            if indicators["mean_distance_zscore"] < -self.mean_distance_threshold:
                direction = "LONG"
                entry_price = Decimal(str(close))
                stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
            else:
                return None
        # Check for short signal: price at upper BB + RSI overbought
        elif indicators["bb_upper_touch"] and indicators["rsi_overbought"]:
            if indicators["mean_distance_zscore"] > self.mean_distance_threshold:
                direction = "SHORT"
                entry_price = Decimal(str(close))
                stop_loss = entry_price + Decimal(str(atr * self.atr_sl_multiplier))
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * Decimal(str(self.risk_reward_ratio)))
            else:
                return None
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
        """Calculate mean reversion indicators.

        Returns:
            Dictionary with: bb_upper, bb_lower, bb_middle, rsi, atr,
            bb_upper_touch, bb_lower_touch, rsi_oversold, rsi_overbought,
            mean_distance_zscore.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]

        # Bollinger Bands
        bb_middle = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (self.bb_std * bb_std)
        bb_lower = bb_middle - (self.bb_std * bb_std)

        # RSI
        rsi = self._calculate_rsi(close, self.rsi_period)

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Mean distance z-score
        current_std = float(bb_std.iloc[-1]) if not np.isnan(bb_std.iloc[-1]) else 1.0
        if current_std == 0:
            current_std = 1.0
        mean_distance_zscore = (
            float(close.iloc[-1]) - float(bb_middle.iloc[-1])
        ) / current_std

        # Touch detection
        bb_lower_touch = float(close.iloc[-1]) <= float(bb_lower.iloc[-1])
        bb_upper_touch = float(close.iloc[-1]) >= float(bb_upper.iloc[-1])

        return {
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "bb_middle": float(bb_middle.iloc[-1]),
            "rsi": float(rsi),
            "atr": float(atr),
            "bb_upper_touch": float(bb_upper_touch),
            "bb_lower_touch": float(bb_lower_touch),
            "rsi_oversold": float(rsi < self.rsi_oversold),
            "rsi_overbought": float(rsi > self.rsi_overbought),
            "mean_distance_zscore": float(mean_distance_zscore),
        }

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int) -> float:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

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
