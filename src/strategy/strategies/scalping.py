"""Scalping strategy implementation.

Uses order flow imbalance, micro-structure patterns, and tight stops
(0.5×ATR) for short-duration trades.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class ScalpingStrategy(BaseStrategy):
    """Scalping strategy using order flow and micro-structure analysis.

    Signal generation logic:
        - LONG: Bid volume imbalance > threshold AND micro-structure bullish pattern
        - SHORT: Ask volume imbalance > threshold AND micro-structure bearish pattern

    Characteristics:
        - Very tight stops (0.5 × ATR)
        - Quick profit targets (1:1.5 risk-reward)
        - High frequency, small moves
        - Best suited for RANGING or low-volatility conditions
    """

    name: str = "scalping"

    def __init__(
        self,
        imbalance_threshold: float = 0.6,
        atr_period: int = 14,
        atr_sl_multiplier: float = 0.5,
        risk_reward_ratio: float = 1.5,
        spread_ma_period: int = 10,
        micro_lookback: int = 5,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.spread_ma_period = spread_ma_period
        self.micro_lookback = micro_lookback

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate scalping signal based on order flow imbalance.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if scalping conditions met, None otherwise.
        """
        min_periods = max(self.atr_period, self.spread_ma_period, self.micro_lookback) + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        atr = indicators["atr"]
        if atr <= 0:
            return None

        close = market_data["close"].iloc[-1]

        # Long signal: bullish order flow imbalance + micro-structure pattern
        if indicators["bullish_imbalance"] and indicators["micro_bullish"]:
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        # Short signal: bearish order flow imbalance + micro-structure pattern
        elif indicators["bearish_imbalance"] and indicators["micro_bearish"]:
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
        """Calculate scalping indicators.

        Returns:
            Dictionary with: order_flow_imbalance, atr, spread_compression,
            bullish_imbalance, bearish_imbalance, micro_bullish, micro_bearish.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]
        volume = market_data["volume"]

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Order flow imbalance approximation using volume and price direction
        # Positive = buying pressure, Negative = selling pressure
        price_change = close.diff()
        up_volume = volume.where(price_change > 0, 0.0)
        down_volume = volume.where(price_change < 0, 0.0)

        recent_up = float(up_volume.iloc[-self.micro_lookback:].sum())
        recent_down = float(down_volume.iloc[-self.micro_lookback:].sum())
        total_volume = recent_up + recent_down

        if total_volume > 0:
            order_flow_imbalance = (recent_up - recent_down) / total_volume
        else:
            order_flow_imbalance = 0.0

        # Micro-structure patterns: consecutive small moves in same direction
        recent_changes = price_change.iloc[-self.micro_lookback:]
        bullish_bars = int((recent_changes > 0).sum())
        bearish_bars = int((recent_changes < 0).sum())

        micro_bullish = bullish_bars >= (self.micro_lookback * 0.6)
        micro_bearish = bearish_bars >= (self.micro_lookback * 0.6)

        # Spread compression: current range vs average range
        current_spread = float(high.iloc[-1] - low.iloc[-1])
        avg_spread = float(
            (high - low).iloc[-self.spread_ma_period - 1 : -1].mean()
        )
        spread_compression = (
            current_spread / avg_spread if avg_spread > 0 else 1.0
        )

        # Imbalance signals
        bullish_imbalance = order_flow_imbalance > self.imbalance_threshold
        bearish_imbalance = order_flow_imbalance < -self.imbalance_threshold

        return {
            "order_flow_imbalance": float(order_flow_imbalance),
            "atr": float(atr),
            "spread_compression": float(spread_compression),
            "bullish_imbalance": float(bullish_imbalance),
            "bearish_imbalance": float(bearish_imbalance),
            "micro_bullish": float(micro_bullish),
            "micro_bearish": float(micro_bearish),
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
