"""Momentum strategy implementation.

Uses Rate of Change (ROC 14), relative strength comparison, and
momentum divergence detection to generate signals.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using ROC and relative strength.

    Signal generation logic:
        - LONG: ROC > threshold AND relative strength rising AND no bearish divergence
        - SHORT: ROC < -threshold AND relative strength falling AND no bullish divergence

    Filters:
        - Momentum divergence detection (price vs ROC disagreement)
        - Best suited for TRENDING regime
    """

    name: str = "momentum"

    def __init__(
        self,
        roc_period: int = 14,
        roc_threshold: float = 2.0,
        rs_period: int = 14,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        divergence_lookback: int = 5,
    ) -> None:
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.rs_period = rs_period
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.divergence_lookback = divergence_lookback

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate momentum signal based on ROC and relative strength.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if momentum conditions met, None otherwise.
        """
        min_periods = max(self.roc_period, self.rs_period, self.atr_period) + self.divergence_lookback + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        atr = indicators["atr"]
        if atr <= 0:
            return None

        close = market_data["close"].iloc[-1]

        # Long signal: strong positive momentum, no bearish divergence
        if (
            indicators["roc_bullish"]
            and indicators["rs_rising"]
            and not indicators["bearish_divergence"]
        ):
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        # Short signal: strong negative momentum, no bullish divergence
        elif (
            indicators["roc_bearish"]
            and indicators["rs_falling"]
            and not indicators["bullish_divergence"]
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
        """Calculate momentum indicators.

        Returns:
            Dictionary with: roc, relative_strength, atr,
            roc_bullish, roc_bearish, rs_rising, rs_falling,
            bearish_divergence, bullish_divergence.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]

        # Rate of Change
        roc = ((close - close.shift(self.roc_period)) / close.shift(self.roc_period)) * 100

        # Relative Strength (price relative to its own moving average)
        rs_ma = close.rolling(window=self.rs_period).mean()
        relative_strength = close / rs_ma

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        current_roc = float(roc.iloc[-1]) if not np.isnan(roc.iloc[-1]) else 0.0
        current_rs = float(relative_strength.iloc[-1]) if not np.isnan(relative_strength.iloc[-1]) else 1.0
        prev_rs = float(relative_strength.iloc[-2]) if not np.isnan(relative_strength.iloc[-2]) else 1.0

        # Momentum signals
        roc_bullish = current_roc > self.roc_threshold
        roc_bearish = current_roc < -self.roc_threshold
        rs_rising = current_rs > prev_rs
        rs_falling = current_rs < prev_rs

        # Divergence detection
        bearish_divergence = self._detect_bearish_divergence(close, roc)
        bullish_divergence = self._detect_bullish_divergence(close, roc)

        return {
            "roc": current_roc,
            "relative_strength": current_rs,
            "atr": float(atr),
            "roc_bullish": float(roc_bullish),
            "roc_bearish": float(roc_bearish),
            "rs_rising": float(rs_rising),
            "rs_falling": float(rs_falling),
            "bearish_divergence": float(bearish_divergence),
            "bullish_divergence": float(bullish_divergence),
        }

    def _detect_bearish_divergence(self, close: pd.Series, roc: pd.Series) -> bool:
        """Detect bearish divergence: price making higher highs but ROC making lower highs."""
        lookback = self.divergence_lookback
        if len(close) < lookback + 1:
            return False

        recent_close = close.iloc[-lookback:]
        recent_roc = roc.iloc[-lookback:]

        # Price making higher high
        price_higher_high = float(recent_close.iloc[-1]) > float(recent_close.iloc[0])
        # ROC making lower high
        roc_lower_high = float(recent_roc.iloc[-1]) < float(recent_roc.iloc[0])

        return price_higher_high and roc_lower_high

    def _detect_bullish_divergence(self, close: pd.Series, roc: pd.Series) -> bool:
        """Detect bullish divergence: price making lower lows but ROC making higher lows."""
        lookback = self.divergence_lookback
        if len(close) < lookback + 1:
            return False

        recent_close = close.iloc[-lookback:]
        recent_roc = roc.iloc[-lookback:]

        # Price making lower low
        price_lower_low = float(recent_close.iloc[-1]) < float(recent_close.iloc[0])
        # ROC making higher low
        roc_higher_low = float(recent_roc.iloc[-1]) > float(recent_roc.iloc[0])

        return price_lower_low and roc_higher_low

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
