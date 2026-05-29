"""News Sentiment strategy implementation.

Uses sentiment scoring from news events, event impact classification,
and news-driven signals to generate trades based on market-moving news.

Validates: Requirements 8.1, 23.2
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.strategy.regime_detector import MarketRegime
from src.strategy.strategies.base import BaseStrategy, TradeSignal


class NewsSentimentStrategy(BaseStrategy):
    """News Sentiment strategy using NLP-based sentiment and event impact.

    Signal generation logic:
        - LONG: Strong bullish sentiment (> threshold) with confirming price action
        - SHORT: Strong bearish sentiment (< -threshold) with confirming price action

    Filters:
        - Sentiment magnitude must exceed threshold
        - Price action must confirm sentiment direction (momentum alignment)
        - Impact level must be at least MEDIUM
        - Best suited for any regime when high-impact news is present

    The strategy expects a 'sentiment_score' column in market_data or
    sentiment data passed via DataFrame attrs.
    """

    name: str = "news_sentiment"

    def __init__(
        self,
        sentiment_threshold: float = 0.5,
        impact_threshold: float = 0.6,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        momentum_period: int = 5,
        confirmation_lookback: int = 3,
    ) -> None:
        self.sentiment_threshold = sentiment_threshold
        self.impact_threshold = impact_threshold
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.momentum_period = momentum_period
        self.confirmation_lookback = confirmation_lookback

    def generate_signal(
        self, market_data: pd.DataFrame, regime: MarketRegime
    ) -> TradeSignal | None:
        """Generate news sentiment signal based on sentiment score and price action.

        Args:
            market_data: DataFrame with columns: open, high, low, close, volume.
                Optionally includes 'sentiment_score' column or sentiment data
                in attrs['sentiment_score'] and attrs['impact_score'].
            regime: Current market regime.

        Returns:
            TradeSignal if sentiment conditions met, None otherwise.
        """
        min_periods = max(self.atr_period, self.momentum_period, self.confirmation_lookback) + 2
        if len(market_data) < min_periods:
            return None

        indicators = self.get_indicators(market_data)

        atr = indicators["atr"]
        if atr <= 0:
            return None

        # No signal if sentiment is not strong enough
        if not indicators["sentiment_bullish"] and not indicators["sentiment_bearish"]:
            return None

        # No signal if impact is below threshold
        if indicators["impact_score"] < self.impact_threshold:
            return None

        close = market_data["close"].iloc[-1]

        # Long signal: bullish sentiment + confirming price momentum
        if indicators["sentiment_bullish"] and indicators["price_confirms_bullish"]:
            direction = "LONG"
            entry_price = Decimal(str(close))
            stop_loss = entry_price - Decimal(str(atr * self.atr_sl_multiplier))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * Decimal(str(self.risk_reward_ratio)))
        # Short signal: bearish sentiment + confirming price momentum
        elif indicators["sentiment_bearish"] and indicators["price_confirms_bearish"]:
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
        """Calculate news sentiment indicators.

        Extracts sentiment from market_data column or attrs, then checks
        price action confirmation.

        Returns:
            Dictionary with: sentiment_score, impact_score, atr,
            price_momentum, sentiment_bullish, sentiment_bearish,
            price_confirms_bullish, price_confirms_bearish.
        """
        close = market_data["close"]
        high = market_data["high"]
        low = market_data["low"]

        # ATR
        atr = self._calculate_atr(high, low, close, self.atr_period)

        # Extract sentiment score
        sentiment_score = self._get_sentiment_score(market_data)
        impact_score = self._get_impact_score(market_data)

        # Price momentum confirmation
        price_momentum = self._calculate_momentum(close)

        # Sentiment signals
        sentiment_bullish = sentiment_score > self.sentiment_threshold
        sentiment_bearish = sentiment_score < -self.sentiment_threshold

        # Price confirmation: recent price action aligns with sentiment
        price_confirms_bullish = price_momentum > 0
        price_confirms_bearish = price_momentum < 0

        return {
            "sentiment_score": float(sentiment_score),
            "impact_score": float(impact_score),
            "atr": float(atr),
            "price_momentum": float(price_momentum),
            "sentiment_bullish": float(sentiment_bullish),
            "sentiment_bearish": float(sentiment_bearish),
            "price_confirms_bullish": float(price_confirms_bullish),
            "price_confirms_bearish": float(price_confirms_bearish),
        }

    def _get_sentiment_score(self, market_data: pd.DataFrame) -> float:
        """Extract sentiment score from market data.

        Checks for 'sentiment_score' column first, then attrs.
        Returns 0.0 if no sentiment data available.
        """
        if "sentiment_score" in market_data.columns:
            val = market_data["sentiment_score"].iloc[-1]
            return float(val) if not np.isnan(val) else 0.0

        return float(market_data.attrs.get("sentiment_score", 0.0))

    def _get_impact_score(self, market_data: pd.DataFrame) -> float:
        """Extract impact score from market data.

        Checks for 'impact_score' column first, then attrs.
        Returns 0.0 if no impact data available.
        """
        if "impact_score" in market_data.columns:
            val = market_data["impact_score"].iloc[-1]
            return float(val) if not np.isnan(val) else 0.0

        return float(market_data.attrs.get("impact_score", 0.0))

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate short-term price momentum for confirmation.

        Uses rate of change over the confirmation lookback period.
        """
        if len(close) < self.confirmation_lookback + 1:
            return 0.0

        current = float(close.iloc[-1])
        past = float(close.iloc[-self.confirmation_lookback - 1])

        if past == 0:
            return 0.0

        return (current - past) / past

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
