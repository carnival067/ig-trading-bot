"""Volatility Trading strategy implementation.

Uses volatility expansion/contraction detection (Bollinger Band squeeze)
and straddle-like entries to profit from volatility regime changes.

Validates: Requirements 8.1
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class VolatilityStrategy(BaseStrategy):
    """Volatility Trading strategy using BB squeeze and expansion detection.

    Signal generation logic:
        - Detects Bollinger Band squeeze (low volatility compression)
        - On expansion (squeeze release), enters in the direction of the breakout
        - LONG: Squeeze release with upward expansion
        - SHORT: Squeeze release with downward expansion

    Characteristics:
        - Profits from volatility expansion after compression
        - Wider stops to accommodate volatility
        - Best suited for transition from RANGING to VOLATILE regime
    """

    name: str = "volatility"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        keltner_period: int = 20,
        keltner_atr_mult: float = 1.5,
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,
        risk_reward_ratio: float = 2.0,
        squeeze_lookback: int = 6,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.keltner_period = keltner_period
        self.keltner_atr_mult = keltner_atr_mult
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.squeeze_lookback = squeeze_lookback

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate volatility signal based on BB squeeze and expansion.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
            regime: Current market regime.

        Returns:
            TradeSignal if volatility expansion conditions met, None otherwise.
        """
        min_periods = max(self.bb_period, self.keltner_period, self.atr_period) + self.squeeze_lookback + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        atr = indicators["atr"]
        if atr <= 0:
            return None

        close = market_data["close"].iloc[-1]

        # Signal on squeeze release (was in squeeze, now expanding)
        if indicators["squeeze_release"] and indicators["expansion_bullish"]:
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        elif indicators["squeeze_release"] and indicators["expansion_bearish"]:
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
        """Calculate volatility indicators.

        Returns:
            Dictionary with: bb_width, keltner_width, atr, squeeze_active,
            squeeze_release, expansion_bullish, expansion_bearish,
            volatility_percentile.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]

        # Bollinger Bands
        bb_middle = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (self.bb_std * bb_std)
        bb_lower = bb_middle - (self.bb_std * bb_std)
        bb_width = bb_upper - bb_lower

        # Keltner Channels (for squeeze detection)
        keltner_middle = close.ewm(span=self.keltner_period, adjust=False).mean()
        keltner_atr = self._calculate_atr_series(high, low, close, self.keltner_period)
        keltner_upper = keltner_middle + (self.keltner_atr_mult * keltner_atr)
        keltner_lower = keltner_middle - (self.keltner_atr_mult * keltner_atr)

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Squeeze detection: BB inside Keltner Channel
        squeeze = (bb_lower > keltner_lower) & (bb_upper < keltner_upper)

        # Check if squeeze was active recently and just released
        recent_squeeze = squeeze.iloc[-self.squeeze_lookback - 1 : -1]
        squeeze_was_active = recent_squeeze.any()
        squeeze_now_released = not squeeze.iloc[-1]
        squeeze_release = squeeze_was_active and squeeze_now_released

        # Expansion direction based on momentum (close relative to middle band)
        momentum = close - bb_middle
        expansion_bullish = float(momentum.iloc[-1]) > 0 and squeeze_release
        expansion_bearish = float(momentum.iloc[-1]) < 0 and squeeze_release

        # Volatility percentile (current BB width vs historical)
        current_bb_width = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0.0
        historical_bb_width = bb_width.dropna()
        if len(historical_bb_width) > 0:
            volatility_percentile = float(
                (historical_bb_width < current_bb_width).sum() / len(historical_bb_width) * 100
            )
        else:
            volatility_percentile = 50.0

        return {
            "bb_width": current_bb_width,
            "keltner_width": float(keltner_upper.iloc[-1] - keltner_lower.iloc[-1])
            if not np.isnan(keltner_upper.iloc[-1])
            else 0.0,
            "atr": float(atr),
            "squeeze_active": float(squeeze.iloc[-1]),
            "squeeze_release": float(squeeze_release),
            "expansion_bullish": float(expansion_bullish),
            "expansion_bearish": float(expansion_bearish),
            "volatility_percentile": float(volatility_percentile),
        }

    @staticmethod
    def _calculate_atr_series(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate ATR as a series."""
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> float:
        """Calculate Average True Range (scalar)."""
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
