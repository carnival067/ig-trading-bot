"""Unit tests for the Market Regime Detector.

Tests the RegimeDetector class and MarketRegime enum for correct
classification of market conditions based on technical indicators.

Validates: Requirements 8.2
"""

import pytest

from src.strategy.regime_detector import MarketRegime, RegimeDetector


@pytest.fixture
def detector() -> RegimeDetector:
    """Create a RegimeDetector instance for testing."""
    return RegimeDetector()


class TestMarketRegimeEnum:
    """Tests for MarketRegime enum values."""

    def test_regime_values(self):
        """All four regimes should have correct string values."""
        assert MarketRegime.TRENDING.value == "trending"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.CRISIS.value == "crisis"

    def test_regime_count(self):
        """There should be exactly 4 market regimes."""
        assert len(MarketRegime) == 4


class TestCrisisDetection:
    """Tests for CRISIS regime classification."""

    def test_crisis_high_atr_percentile(self, detector: RegimeDetector):
        """ATR percentile > 95 should classify as CRISIS."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=96.0)
        assert result == MarketRegime.CRISIS

    def test_crisis_atr_at_boundary(self, detector: RegimeDetector):
        """ATR percentile exactly at 95 should NOT be CRISIS (> not >=)."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=95.0)
        assert result != MarketRegime.CRISIS

    def test_crisis_vix_and_price_decline(self, detector: RegimeDetector):
        """VIX z-score > 3 AND price decline > 3% should classify as CRISIS."""
        result = detector.classify(
            adx=15.0,
            bb_width=0.3,
            atr_percentile=50.0,
            vix_zscore=3.5,
            price_decline_24h=4.0,
        )
        assert result == MarketRegime.CRISIS

    def test_crisis_vix_only_not_enough(self, detector: RegimeDetector):
        """VIX z-score > 3 alone (without price decline) should NOT be CRISIS."""
        result = detector.classify(
            adx=15.0,
            bb_width=0.3,
            atr_percentile=50.0,
            vix_zscore=3.5,
            price_decline_24h=None,
        )
        assert result != MarketRegime.CRISIS

    def test_crisis_price_decline_only_not_enough(self, detector: RegimeDetector):
        """Price decline > 3% alone (without VIX) should NOT be CRISIS."""
        result = detector.classify(
            adx=15.0,
            bb_width=0.3,
            atr_percentile=50.0,
            vix_zscore=None,
            price_decline_24h=5.0,
        )
        assert result != MarketRegime.CRISIS

    def test_crisis_vix_below_threshold(self, detector: RegimeDetector):
        """VIX z-score <= 3 with price decline should NOT trigger CRISIS."""
        result = detector.classify(
            adx=15.0,
            bb_width=0.3,
            atr_percentile=50.0,
            vix_zscore=2.9,
            price_decline_24h=5.0,
        )
        assert result != MarketRegime.CRISIS

    def test_crisis_price_decline_below_threshold(self, detector: RegimeDetector):
        """Price decline <= 3% with high VIX should NOT trigger CRISIS."""
        result = detector.classify(
            adx=15.0,
            bb_width=0.3,
            atr_percentile=50.0,
            vix_zscore=4.0,
            price_decline_24h=2.5,
        )
        assert result != MarketRegime.CRISIS


class TestVolatileDetection:
    """Tests for VOLATILE regime classification."""

    def test_volatile_high_atr(self, detector: RegimeDetector):
        """ATR percentile > 75 (but <= 95) should classify as VOLATILE."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=80.0)
        assert result == MarketRegime.VOLATILE

    def test_volatile_at_boundary(self, detector: RegimeDetector):
        """ATR percentile exactly at 75 should NOT be VOLATILE (> not >=)."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=75.0)
        assert result != MarketRegime.VOLATILE

    def test_volatile_just_above_boundary(self, detector: RegimeDetector):
        """ATR percentile just above 75 should be VOLATILE."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=75.1)
        assert result == MarketRegime.VOLATILE

    def test_volatile_overrides_trending(self, detector: RegimeDetector):
        """VOLATILE should take priority over TRENDING when ATR > 75."""
        # ADX > 25 would normally be TRENDING, but ATR > 75 makes it VOLATILE
        result = detector.classify(adx=40.0, bb_width=0.5, atr_percentile=80.0)
        assert result == MarketRegime.VOLATILE


class TestTrendingDetection:
    """Tests for TRENDING regime classification."""

    def test_trending_high_adx(self, detector: RegimeDetector):
        """ADX > 25 with ATR < 75 should classify as TRENDING."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=50.0)
        assert result == MarketRegime.TRENDING

    def test_trending_at_boundary(self, detector: RegimeDetector):
        """ADX exactly at 25 should NOT be TRENDING (> not >=)."""
        result = detector.classify(adx=25.0, bb_width=0.3, atr_percentile=50.0)
        assert result != MarketRegime.TRENDING

    def test_trending_just_above_boundary(self, detector: RegimeDetector):
        """ADX just above 25 should be TRENDING."""
        result = detector.classify(adx=25.1, bb_width=0.5, atr_percentile=50.0)
        assert result == MarketRegime.TRENDING

    def test_trending_requires_low_atr(self, detector: RegimeDetector):
        """ADX > 25 but ATR > 75 should NOT be TRENDING (VOLATILE takes priority)."""
        result = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=80.0)
        assert result != MarketRegime.TRENDING


class TestRangingDetection:
    """Tests for RANGING regime classification."""

    def test_ranging_low_adx_low_bb(self, detector: RegimeDetector):
        """ADX <= 25 and BB width < 0.5 should classify as RANGING."""
        result = detector.classify(adx=20.0, bb_width=0.3, atr_percentile=50.0)
        assert result == MarketRegime.RANGING

    def test_ranging_is_default_fallback(self, detector: RegimeDetector):
        """When no other regime matches, RANGING is the fallback."""
        # ADX <= 25, ATR < 75, no crisis conditions
        result = detector.classify(adx=20.0, bb_width=0.6, atr_percentile=50.0)
        assert result == MarketRegime.RANGING

    def test_ranging_adx_at_boundary(self, detector: RegimeDetector):
        """ADX exactly at 25 should fall to RANGING."""
        result = detector.classify(adx=25.0, bb_width=0.3, atr_percentile=50.0)
        assert result == MarketRegime.RANGING


class TestPriorityOrder:
    """Tests for regime classification priority order."""

    def test_crisis_highest_priority(self, detector: RegimeDetector):
        """CRISIS should override all other conditions."""
        # All conditions met: high ADX (trending), high ATR (volatile + crisis)
        result = detector.classify(adx=40.0, bb_width=0.8, atr_percentile=96.0)
        assert result == MarketRegime.CRISIS

    def test_volatile_over_trending(self, detector: RegimeDetector):
        """VOLATILE should override TRENDING."""
        result = detector.classify(adx=40.0, bb_width=0.5, atr_percentile=80.0)
        assert result == MarketRegime.VOLATILE

    def test_trending_over_ranging(self, detector: RegimeDetector):
        """TRENDING should override RANGING when ADX > 25."""
        result = detector.classify(adx=30.0, bb_width=0.3, atr_percentile=50.0)
        assert result == MarketRegime.TRENDING

    def test_classification_is_deterministic(self, detector: RegimeDetector):
        """Same inputs should always produce the same regime."""
        inputs = {"adx": 30.0, "bb_width": 0.5, "atr_percentile": 60.0}
        result1 = detector.classify(**inputs)
        result2 = detector.classify(**inputs)
        result3 = detector.classify(**inputs)
        assert result1 == result2 == result3

    def test_all_regimes_reachable(self, detector: RegimeDetector):
        """All four regimes should be reachable with appropriate inputs."""
        crisis = detector.classify(adx=10.0, bb_width=0.5, atr_percentile=96.0)
        volatile = detector.classify(adx=10.0, bb_width=0.5, atr_percentile=80.0)
        trending = detector.classify(adx=30.0, bb_width=0.5, atr_percentile=50.0)
        ranging = detector.classify(adx=15.0, bb_width=0.3, atr_percentile=50.0)

        assert crisis == MarketRegime.CRISIS
        assert volatile == MarketRegime.VOLATILE
        assert trending == MarketRegime.TRENDING
        assert ranging == MarketRegime.RANGING

    def test_mutually_exclusive(self, detector: RegimeDetector):
        """Each classification should return exactly one regime."""
        test_cases = [
            {"adx": 10.0, "bb_width": 0.3, "atr_percentile": 96.0},
            {"adx": 40.0, "bb_width": 0.8, "atr_percentile": 80.0},
            {"adx": 30.0, "bb_width": 0.5, "atr_percentile": 50.0},
            {"adx": 15.0, "bb_width": 0.3, "atr_percentile": 40.0},
        ]
        for inputs in test_cases:
            result = detector.classify(**inputs)
            assert isinstance(result, MarketRegime)
            # Only one regime should match
            regimes = [MarketRegime.CRISIS, MarketRegime.VOLATILE, MarketRegime.TRENDING, MarketRegime.RANGING]
            assert result in regimes


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_values(self, detector: RegimeDetector):
        """Zero values for all inputs should return RANGING."""
        result = detector.classify(adx=0.0, bb_width=0.0, atr_percentile=0.0)
        assert result == MarketRegime.RANGING

    def test_max_values(self, detector: RegimeDetector):
        """Maximum values should return CRISIS."""
        result = detector.classify(adx=100.0, bb_width=1.0, atr_percentile=100.0)
        assert result == MarketRegime.CRISIS

    def test_none_optional_params(self, detector: RegimeDetector):
        """None values for optional params should not cause errors."""
        result = detector.classify(
            adx=30.0,
            bb_width=0.5,
            atr_percentile=50.0,
            vix_zscore=None,
            price_decline_24h=None,
        )
        assert result == MarketRegime.TRENDING

    def test_negative_adx_treated_as_low(self, detector: RegimeDetector):
        """Negative ADX (unusual) should not trigger trending."""
        result = detector.classify(adx=-5.0, bb_width=0.3, atr_percentile=50.0)
        assert result == MarketRegime.RANGING
