"""Market regime detection and classification.

Classifies market conditions into one of four regimes based on
technical indicators: TRENDING, RANGING, VOLATILE, or CRISIS.

Priority order: CRISIS > VOLATILE > TRENDING > RANGING

Validates: Requirements 8.2
"""

from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class RegimeDetector:
    """Classifies market conditions into regimes based on technical indicators.

    Uses ADX, Bollinger Band width, ATR percentile, and optionally VIX z-score
    and price decline to determine the current market regime.

    Thresholds:
        CRISIS: ATR percentile > 95 OR (VIX z-score > 3 AND price decline > 3% in 24h)
        VOLATILE: ATR percentile > 75
        TRENDING: ADX > 25 AND ATR percentile < 75
        RANGING: ADX <= 25 AND BB width < 50th percentile (0.5)

    Priority: CRISIS > VOLATILE > TRENDING > RANGING (default fallback)
    """

    # Thresholds
    ADX_TRENDING_THRESHOLD: float = 25.0
    ATR_VOLATILE_PERCENTILE: float = 75.0
    ATR_CRISIS_PERCENTILE: float = 95.0
    BB_WIDTH_RANGING_PERCENTILE: float = 0.5
    VIX_ZSCORE_CRISIS_THRESHOLD: float = 3.0
    PRICE_DECLINE_CRISIS_THRESHOLD: float = 3.0  # percent

    def classify(
        self,
        adx: float,
        bb_width: float,
        atr_percentile: float,
        vix_zscore: float | None = None,
        price_decline_24h: float | None = None,
    ) -> MarketRegime:
        """Classify the current market regime based on indicator values.

        Args:
            adx: Average Directional Index value (0-100).
            bb_width: Bollinger Band width as a percentile (0.0-1.0).
            atr_percentile: ATR percentile rank (0-100).
            vix_zscore: Optional VIX z-score for crisis detection.
            price_decline_24h: Optional 24-hour price decline percentage (positive = decline).

        Returns:
            MarketRegime classification.
        """
        # Priority 1: CRISIS detection
        if self._is_crisis(atr_percentile, vix_zscore, price_decline_24h):
            return MarketRegime.CRISIS

        # Priority 2: VOLATILE detection
        if self._is_volatile(atr_percentile):
            return MarketRegime.VOLATILE

        # Priority 3: TRENDING detection
        if self._is_trending(adx, atr_percentile):
            return MarketRegime.TRENDING

        # Priority 4: RANGING detection (also serves as default fallback)
        return MarketRegime.RANGING

    def _is_crisis(
        self,
        atr_percentile: float,
        vix_zscore: float | None,
        price_decline_24h: float | None,
    ) -> bool:
        """Check if market is in crisis regime.

        Crisis conditions:
            - ATR percentile > 95, OR
            - VIX z-score > 3 AND price decline > 3% in 24h
        """
        if atr_percentile > self.ATR_CRISIS_PERCENTILE:
            return True

        if vix_zscore is not None and price_decline_24h is not None:
            if (
                vix_zscore > self.VIX_ZSCORE_CRISIS_THRESHOLD
                and price_decline_24h > self.PRICE_DECLINE_CRISIS_THRESHOLD
            ):
                return True

        return False

    def _is_volatile(self, atr_percentile: float) -> bool:
        """Check if market is in volatile regime.

        Volatile condition: ATR percentile > 75
        """
        return atr_percentile > self.ATR_VOLATILE_PERCENTILE

    def _is_trending(self, adx: float, atr_percentile: float) -> bool:
        """Check if market is in trending regime.

        Trending conditions: ADX > 25 AND ATR percentile < 75
        """
        return adx > self.ADX_TRENDING_THRESHOLD and atr_percentile < self.ATR_VOLATILE_PERCENTILE
