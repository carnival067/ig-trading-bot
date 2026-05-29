"""Integration tests for the end-to-end position size multiplication pipeline.

Tests the full multiplicative pipeline:
    base_size × volatility_factor × drawdown_factor × mistake_factor × news_factor
    → reject if < min_lot_size

Validates: Cross-Cutting Rule 1
"""

from decimal import Decimal

import pytest

from src.risk.position_sizer import PositionSizer, ReductionFactor, PositionSizeResult


@pytest.fixture
def sizer() -> PositionSizer:
    """Create a fresh PositionSizer instance."""
    return PositionSizer()


# Standard test parameters for a predictable base size
EQUITY = Decimal("100000")
RISK_PCT = Decimal("0.01")
ATR = Decimal("50")
ATR_MULTIPLIER = Decimal("1.5")
# Base size = (100000 * 0.01) / (50 * 1.5) = 1000 / 75 = 13.333...
# Quantized to 13.33


def _volatility_factor() -> ReductionFactor:
    """Volatility factor: 0.5 when ATR z-score > 2.0 (applied internally)."""
    # This is applied internally by the sizer when z-score > 2.0
    # We don't pass it as an external factor
    pass


def _drawdown_factor() -> ReductionFactor:
    """Drawdown factor: 0.25 when drawdown > 10%."""
    return ReductionFactor(
        source="drawdown",
        factor=0.25,
        reason="Drawdown exceeds 10%, reducing position size by 75%",
    )


def _mistake_factor_active() -> ReductionFactor:
    """Mistake factor: 0.7 for active pattern."""
    return ReductionFactor(
        source="mistake",
        factor=0.7,
        reason="Active mistake pattern detected, reducing by 30%",
    )


def _mistake_factor_reactivated() -> ReductionFactor:
    """Mistake factor: 0.5 for reactivated pattern."""
    return ReductionFactor(
        source="mistake",
        factor=0.5,
        reason="Reactivated mistake pattern, reducing by 50%",
    )


def _news_factor() -> ReductionFactor:
    """News factor: 0.5 when high-impact event within 15 min."""
    return ReductionFactor(
        source="news",
        factor=0.5,
        reason="High-impact news event within 15 minutes",
    )


class TestIndividualFactors:
    """Test each reduction factor applied individually."""

    def test_volatility_factor_alone(self, sizer: PositionSizer) -> None:
        """Volatility factor (0.5) applied when z-score > 2.0."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,  # triggers 0.5 reduction
        )
        # Base: 13.333... × 0.5 = 6.666... → 6.66
        assert not result.rejected
        assert result.size == Decimal("6.66")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "volatility"
        assert result.applied_reductions[0].factor == 0.5

    def test_drawdown_factor_alone(self, sizer: PositionSizer) -> None:
        """Drawdown factor (0.25) applied when drawdown > 10%."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_drawdown_factor()],
        )
        # Base: 13.333... × 0.25 = 3.333... → 3.33
        assert not result.rejected
        assert result.size == Decimal("3.33")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "drawdown"

    def test_mistake_factor_active_alone(self, sizer: PositionSizer) -> None:
        """Mistake factor (0.7) for active pattern."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_mistake_factor_active()],
        )
        # Base: 13.333... × 0.7 = 9.333... → 9.33
        assert not result.rejected
        assert result.size == Decimal("9.33")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "mistake"
        assert result.applied_reductions[0].factor == 0.7

    def test_mistake_factor_reactivated_alone(self, sizer: PositionSizer) -> None:
        """Mistake factor (0.5) for reactivated pattern."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_mistake_factor_reactivated()],
        )
        # Base: 13.333... × 0.5 = 6.666... → 6.66
        assert not result.rejected
        assert result.size == Decimal("6.66")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "mistake"
        assert result.applied_reductions[0].factor == 0.5

    def test_news_factor_alone(self, sizer: PositionSizer) -> None:
        """News factor (0.5) when high-impact event within 15 min."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_news_factor()],
        )
        # Base: 13.333... × 0.5 = 6.666... → 6.66
        assert not result.rejected
        assert result.size == Decimal("6.66")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "news"
        assert result.applied_reductions[0].factor == 0.5


class TestPairwiseCombinations:
    """Test pairs of reduction factors applied together."""

    def test_volatility_and_drawdown(self, sizer: PositionSizer) -> None:
        """Volatility (0.5) × Drawdown (0.25) = 0.125."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=[_drawdown_factor()],
        )
        # Base: 13.333... × 0.5 × 0.25 = 1.666... → 1.66
        assert not result.rejected
        assert result.size == Decimal("1.66")
        assert len(result.applied_reductions) == 2

    def test_drawdown_and_mistake(self, sizer: PositionSizer) -> None:
        """Drawdown (0.25) × Mistake active (0.7) = 0.175."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_drawdown_factor(), _mistake_factor_active()],
        )
        # Base: 13.333... × 0.25 × 0.7 = 2.333... → 2.33
        assert not result.rejected
        assert result.size == Decimal("2.33")
        assert len(result.applied_reductions) == 2

    def test_mistake_and_news(self, sizer: PositionSizer) -> None:
        """Mistake active (0.7) × News (0.5) = 0.35."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_mistake_factor_active(), _news_factor()],
        )
        # Base: 13.333... × 0.7 × 0.5 = 4.666... → 4.66
        assert not result.rejected
        assert result.size == Decimal("4.66")
        assert len(result.applied_reductions) == 2

    def test_volatility_and_news(self, sizer: PositionSizer) -> None:
        """Volatility (0.5) × News (0.5) = 0.25."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=[_news_factor()],
        )
        # Base: 13.333... × 0.5 × 0.5 = 3.333... → 3.33
        assert not result.rejected
        assert result.size == Decimal("3.33")
        assert len(result.applied_reductions) == 2


class TestFullPipeline:
    """Test the complete end-to-end multiplicative pipeline with all 4 factors."""

    def test_all_four_factors_active_pattern(self, sizer: PositionSizer) -> None:
        """Full pipeline: volatility × drawdown × mistake(active) × news.

        Combined factor: 0.5 × 0.25 × 0.7 × 0.5 = 0.04375
        """
        factors = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,  # triggers volatility factor
            reduction_factors=factors,
        )
        # Base: 13.333... × 0.5 × 0.25 × 0.7 × 0.5 = 0.58333...
        # Quantized: 0.58
        assert not result.rejected
        assert result.size == Decimal("0.58")
        assert len(result.applied_reductions) == 4
        # Verify all sources are present
        sources = [r.source for r in result.applied_reductions]
        assert "volatility" in sources
        assert "drawdown" in sources
        assert "mistake" in sources
        assert "news" in sources

    def test_all_four_factors_reactivated_pattern(self, sizer: PositionSizer) -> None:
        """Full pipeline: volatility × drawdown × mistake(reactivated) × news.

        Combined factor: 0.5 × 0.25 × 0.5 × 0.5 = 0.03125
        """
        factors = [_drawdown_factor(), _mistake_factor_reactivated(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 13.333... × 0.5 × 0.25 × 0.5 × 0.5 = 0.41666...
        # Quantized: 0.41
        assert not result.rejected
        assert result.size == Decimal("0.41")
        assert len(result.applied_reductions) == 4

    def test_multiplicative_order_independence(self, sizer: PositionSizer) -> None:
        """Multiplication is commutative — order of factors doesn't matter."""
        factors_order_1 = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        factors_order_2 = [_news_factor(), _drawdown_factor(), _mistake_factor_active()]
        factors_order_3 = [_mistake_factor_active(), _news_factor(), _drawdown_factor()]

        result_1 = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors_order_1,
        )
        result_2 = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors_order_2,
        )
        result_3 = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors_order_3,
        )
        assert result_1.size == result_2.size == result_3.size

    def test_pipeline_produces_correct_numeric_value(self, sizer: PositionSizer) -> None:
        """Verify exact numeric calculation for the full pipeline.

        equity=100000, risk_pct=0.01, ATR=50, multiplier=1.5
        base_size = (100000 * 0.01) / (50 * 1.5) = 13.3333...

        Factors: 0.5 (vol) × 0.25 (dd) × 0.7 (mistake) × 0.5 (news) = 0.04375
        Final: 13.3333... × 0.04375 = 0.583333...
        Quantized (ROUND_DOWN): 0.58
        """
        factors = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        expected = Decimal("0.58")
        assert result.size == expected


class TestMinLotSizeRejection:
    """Test rejection when combined factors produce size below min_lot_size."""

    def test_reject_when_all_factors_reduce_below_min_lot(
        self, sizer: PositionSizer
    ) -> None:
        """Small equity + all factors → size below 0.01 → rejected."""
        factors = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=Decimal("5000"),
            risk_pct=RISK_PCT,
            atr=Decimal("200"),
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: (5000 * 0.01) / (200 * 1.5) = 50 / 300 = 0.1666...
        # All factors: 0.5 × 0.25 × 0.7 × 0.5 = 0.04375
        # Final: 0.1666... × 0.04375 = 0.00729...
        # Quantized: 0.00 (ROUND_DOWN)
        # 0.00 < 0.01 → rejected
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "below minimum lot size"

    def test_reject_with_reactivated_pattern_below_min_lot(
        self, sizer: PositionSizer
    ) -> None:
        """Reactivated pattern (0.5) makes the reduction even more aggressive."""
        factors = [_drawdown_factor(), _mistake_factor_reactivated(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=Decimal("5000"),
            risk_pct=RISK_PCT,
            atr=Decimal("200"),
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 0.1666...
        # All factors: 0.5 × 0.25 × 0.5 × 0.5 = 0.03125
        # Final: 0.1666... × 0.03125 = 0.00520...
        # Quantized: 0.00 → rejected
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "below minimum lot size"

    def test_reject_with_custom_min_lot_size(self, sizer: PositionSizer) -> None:
        """Custom min_lot_size of 1.0 causes rejection with moderate factors."""
        factors = [_drawdown_factor(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
            min_lot_size=Decimal("1.0"),
        )
        # Base: 13.333... × 0.5 × 0.25 × 0.5 = 0.833...
        # Quantized: 0.83
        # 0.83 < 1.0 → rejected
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "below minimum lot size"

    def test_accept_just_above_min_lot_size(self, sizer: PositionSizer) -> None:
        """Size exactly at min_lot_size boundary is accepted."""
        # We need final size to be exactly 0.01
        # Use equity=1500, risk=0.01, atr=100, mult=1.5
        # Base: (1500 * 0.01) / (100 * 1.5) = 15 / 150 = 0.1
        # With drawdown factor (0.25): 0.1 × 0.25 = 0.025
        # Quantized: 0.02 → >= 0.01, accepted
        result = sizer.calculate_size(
            account_equity=Decimal("1500"),
            risk_pct=RISK_PCT,
            atr=Decimal("100"),
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=[_drawdown_factor()],
            min_lot_size=Decimal("0.01"),
        )
        assert not result.rejected
        assert result.size == Decimal("0.02")

    def test_borderline_rejection_at_min_lot(self, sizer: PositionSizer) -> None:
        """Size that quantizes to exactly 0.00 is rejected."""
        # equity=1000, risk=0.01, atr=500, mult=1.5
        # Base: (1000 * 0.01) / (500 * 1.5) = 10 / 750 = 0.01333...
        # With all 4 factors (vol + dd + mistake + news):
        # 0.01333... × 0.5 × 0.25 × 0.7 × 0.5 = 0.01333... × 0.04375 = 0.000583...
        # Quantized: 0.00 → rejected
        factors = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=Decimal("1000"),
            risk_pct=RISK_PCT,
            atr=Decimal("500"),
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        assert result.rejected
        assert result.rejection_reason == "below minimum lot size"


class TestTripleCombinations:
    """Test three-factor combinations to verify stacking correctness."""

    def test_volatility_drawdown_mistake(self, sizer: PositionSizer) -> None:
        """Volatility (0.5) × Drawdown (0.25) × Mistake active (0.7) = 0.0875."""
        factors = [_drawdown_factor(), _mistake_factor_active()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 13.333... × 0.5 × 0.25 × 0.7 = 1.1666...
        # Quantized: 1.16
        assert not result.rejected
        assert result.size == Decimal("1.16")
        assert len(result.applied_reductions) == 3

    def test_volatility_drawdown_news(self, sizer: PositionSizer) -> None:
        """Volatility (0.5) × Drawdown (0.25) × News (0.5) = 0.0625."""
        factors = [_drawdown_factor(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 13.333... × 0.5 × 0.25 × 0.5 = 0.8333...
        # Quantized: 0.83
        assert not result.rejected
        assert result.size == Decimal("0.83")
        assert len(result.applied_reductions) == 3

    def test_drawdown_mistake_news(self, sizer: PositionSizer) -> None:
        """Drawdown (0.25) × Mistake active (0.7) × News (0.5) = 0.0875."""
        factors = [_drawdown_factor(), _mistake_factor_active(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,  # no volatility factor
            reduction_factors=factors,
        )
        # Base: 13.333... × 0.25 × 0.7 × 0.5 = 1.1666...
        # Quantized: 1.16
        assert not result.rejected
        assert result.size == Decimal("1.16")
        assert len(result.applied_reductions) == 3


class TestPipelineWithDifferentEquityLevels:
    """Test the pipeline with varying equity to verify scaling behavior."""

    def test_large_equity_all_factors_still_accepted(self, sizer: PositionSizer) -> None:
        """With large equity, even aggressive reductions produce tradeable size."""
        factors = [_drawdown_factor(), _mistake_factor_reactivated(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=Decimal("1000000"),
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: (1000000 * 0.01) / (50 * 1.5) = 133.333...
        # All factors: 0.5 × 0.25 × 0.5 × 0.5 = 0.03125
        # Final: 133.333... × 0.03125 = 4.1666...
        # Quantized: 4.16
        assert not result.rejected
        assert result.size == Decimal("4.16")

    def test_small_equity_all_factors_rejected(self, sizer: PositionSizer) -> None:
        """With small equity, all factors combined push below min lot."""
        factors = [_drawdown_factor(), _mistake_factor_reactivated(), _news_factor()]
        result = sizer.calculate_size(
            account_equity=Decimal("2000"),
            risk_pct=RISK_PCT,
            atr=Decimal("100"),
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: (2000 * 0.01) / (100 * 1.5) = 20 / 150 = 0.1333...
        # All factors: 0.5 × 0.25 × 0.5 × 0.5 = 0.03125
        # Final: 0.1333... × 0.03125 = 0.004166...
        # Quantized: 0.00 → rejected
        assert result.rejected
        assert result.rejection_reason == "below minimum lot size"

    def test_no_factors_baseline_comparison(self, sizer: PositionSizer) -> None:
        """Verify that without any factors, the base size is as expected."""
        result = sizer.calculate_size(
            account_equity=EQUITY,
            risk_pct=RISK_PCT,
            atr=ATR,
            atr_multiplier=ATR_MULTIPLIER,
            current_volatility_zscore=0.0,
            reduction_factors=None,
        )
        assert not result.rejected
        assert result.size == Decimal("13.33")
        assert result.applied_reductions == []
