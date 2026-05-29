"""Unit tests for the ATR-based position sizer.

Tests cover:
- Task 3.1: Base formula calculation and ATR zero/negative rejection
- Task 3.2: Volatility-based size reduction (50% when z-score > 2.0)
- Task 3.3: Hard cap enforcement (position size <= 5% of equity)
- Task 3.4: Multiplicative reduction factor application with min lot floor
"""

from decimal import Decimal

import pytest

from src.risk.position_sizer import PositionSizer, ReductionFactor, PositionSizeResult


@pytest.fixture
def sizer() -> PositionSizer:
    return PositionSizer()


class TestBaseFormula:
    """Task 3.1: ATR-based position sizing formula."""

    def test_basic_calculation(self, sizer: PositionSizer) -> None:
        """size = (equity * risk_pct) / (atr * atr_multiplier)"""
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        # Expected: (100000 * 0.01) / (50 * 1.5) = 1000 / 75 = 13.33
        assert not result.rejected
        assert result.size == Decimal("13.33")
        assert result.rejection_reason is None

    def test_different_risk_pct(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("50000"),
            risk_pct=Decimal("0.02"),
            atr=Decimal("100"),
            atr_multiplier=Decimal("2.0"),
            current_volatility_zscore=0.0,
        )
        # Expected: (50000 * 0.02) / (100 * 2.0) = 1000 / 200 = 5.00
        assert not result.rejected
        assert result.size == Decimal("5.00")

    def test_reject_when_atr_is_zero(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("0"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "insufficient volatility data"

    def test_reject_when_atr_is_negative(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("-5"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "insufficient volatility data"

    def test_reject_when_equity_is_zero(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("0"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "account equity must be positive"

    def test_reject_when_equity_is_negative(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("-1000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "account equity must be positive"

    def test_reject_when_risk_pct_is_zero(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert "risk percentage" in result.rejection_reason

    def test_reject_when_risk_pct_exceeds_max(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.06"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert "risk percentage" in result.rejection_reason

    def test_accept_risk_pct_at_max_boundary(self, sizer: PositionSizer) -> None:
        """risk_pct == 0.05 (5%) is the maximum allowed and should be accepted."""
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.05"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        # (100000 * 0.05) / (50 * 1.5) = 5000 / 75 = 66.66
        # But 66.66 > 5000 (5% of 100000)? No: 5% of 100000 = 5000, 66.66 < 5000
        assert not result.rejected
        assert result.size == Decimal("66.66")

    def test_reject_when_risk_pct_is_negative(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("-0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.size is None
        assert "risk percentage" in result.rejection_reason

    def test_result_has_no_reductions_when_none_applied(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
        )
        assert result.applied_reductions == []


class TestVolatilityReduction:
    """Task 3.2: 50% reduction when ATR z-score > 2.0."""

    def test_no_reduction_below_threshold(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=1.9,
        )
        assert not result.rejected
        # No volatility reduction applied
        assert len(result.applied_reductions) == 0
        assert result.size == Decimal("13.33")

    def test_no_reduction_at_threshold(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.0,
        )
        # z-score == 2.0 does NOT trigger (must be > 2.0)
        assert not result.rejected
        assert len(result.applied_reductions) == 0
        assert result.size == Decimal("13.33")

    def test_reduction_above_threshold(self, sizer: PositionSizer) -> None:
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.5,
        )
        assert not result.rejected
        # Base: 13.33, after 50% reduction: 6.66
        assert result.size == Decimal("6.66")
        assert len(result.applied_reductions) == 1
        assert result.applied_reductions[0].source == "volatility"
        assert result.applied_reductions[0].factor == 0.5


class TestHardCap:
    """Task 3.3: Position size must not exceed 5% of equity."""

    def test_reject_when_size_exceeds_cap(self, sizer: PositionSizer) -> None:
        # With very low ATR, the position size will be huge
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.05"),
            atr=Decimal("0.01"),
            atr_multiplier=Decimal("1.0"),
            current_volatility_zscore=0.0,
        )
        # Expected: (100000 * 0.05) / (0.01 * 1.0) = 5000 / 0.01 = 500000
        # Max allowed: 100000 * 0.05 = 5000
        # 500000 > 5000 → rejected
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "position size limit exceeded"

    def test_accept_at_cap_boundary(self, sizer: PositionSizer) -> None:
        # size = (equity * risk_pct) / (atr * multiplier)
        # We want size == 5000 (5% of 100000)
        # 5000 = (100000 * risk_pct) / (atr * multiplier)
        # Let risk_pct = 0.01, atr = 0.2, multiplier = 1.0
        # size = (100000 * 0.01) / (0.2 * 1.0) = 1000 / 0.2 = 5000
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("0.2"),
            atr_multiplier=Decimal("1.0"),
            current_volatility_zscore=0.0,
        )
        # 5000 == 5000 → not exceeded (<=), should be accepted
        assert not result.rejected
        assert result.size == Decimal("5000.00")

    def test_reject_just_above_cap(self, sizer: PositionSizer) -> None:
        # size = (100000 * 0.01) / (0.19 * 1.0) = 1000 / 0.19 ≈ 5263.15
        # Max: 5000 → rejected
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("0.19"),
            atr_multiplier=Decimal("1.0"),
            current_volatility_zscore=0.0,
        )
        assert result.rejected
        assert result.rejection_reason == "position size limit exceeded"


class TestMultiplicativeReductions:
    """Task 3.4: Multiplicative reduction factor application."""

    def test_single_reduction_factor(self, sizer: PositionSizer) -> None:
        drawdown_factor = ReductionFactor(
            source="drawdown",
            factor=0.25,
            reason="Drawdown > 10%, reducing by 75%",
        )
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
            reduction_factors=[drawdown_factor],
        )
        # Base: 13.33, after 0.25 factor: 3.33
        assert not result.rejected
        assert result.size == Decimal("3.33")
        assert len(result.applied_reductions) == 1

    def test_multiple_reduction_factors_multiplicative(self, sizer: PositionSizer) -> None:
        factors = [
            ReductionFactor(source="drawdown", factor=0.25, reason="Drawdown reduction"),
            ReductionFactor(source="mistake", factor=0.7, reason="Mistake pattern"),
            ReductionFactor(source="news", factor=0.5, reason="High-impact news"),
        ]
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
            reduction_factors=factors,
        )
        # Base: 13.33
        # After drawdown (0.25): 3.3325 → 3.33
        # After mistake (0.7): 2.33275 → 2.33 (but applied to unquantized)
        # After news (0.5): 1.166375 → 1.16
        # Actually all applied to unquantized base then quantized at end:
        # 13.333... * 0.25 * 0.7 * 0.5 = 13.333... * 0.0875 = 1.1666...
        # Quantized: 1.16
        assert not result.rejected
        assert result.size == Decimal("1.16")
        assert len(result.applied_reductions) == 3

    def test_volatility_plus_external_factors(self, sizer: PositionSizer) -> None:
        """Volatility reduction stacks with external factors."""
        factors = [
            ReductionFactor(source="drawdown", factor=0.25, reason="Drawdown"),
        ]
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 13.333...
        # Volatility (0.5): 6.666...
        # Drawdown (0.25): 1.666...
        # Quantized: 1.66
        assert not result.rejected
        assert result.size == Decimal("1.66")
        assert len(result.applied_reductions) == 2
        assert result.applied_reductions[0].source == "volatility"
        assert result.applied_reductions[1].source == "drawdown"

    def test_reject_below_min_lot_size(self, sizer: PositionSizer) -> None:
        """If final size < min_lot_size after reductions, reject."""
        factors = [
            ReductionFactor(source="drawdown", factor=0.25, reason="Drawdown"),
            ReductionFactor(source="mistake", factor=0.5, reason="Mistake"),
            ReductionFactor(source="news", factor=0.5, reason="News"),
        ]
        result = sizer.calculate_size(
            account_equity=Decimal("10000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("100"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.5,  # triggers volatility reduction too
            reduction_factors=factors,
        )
        # Base: (10000 * 0.01) / (100 * 1.5) = 100 / 150 = 0.666...
        # Volatility (0.5): 0.333...
        # Drawdown (0.25): 0.0833...
        # Mistake (0.5): 0.0416...
        # News (0.5): 0.0208...
        # Quantized: 0.02
        # min_lot_size default is 0.01, so 0.02 >= 0.01 → accepted
        # Let's use a higher min_lot_size
        result = sizer.calculate_size(
            account_equity=Decimal("10000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("100"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.5,
            reduction_factors=factors,
            min_lot_size=Decimal("0.05"),
        )
        # 0.02 < 0.05 → rejected
        assert result.rejected
        assert result.size is None
        assert result.rejection_reason == "below minimum lot size"

    def test_accept_at_min_lot_size_boundary(self, sizer: PositionSizer) -> None:
        """Size exactly at min_lot_size should be accepted."""
        # We need size to quantize to exactly 0.01
        # size = (equity * risk_pct) / (atr * multiplier) * factor
        # 0.01 = (1000 * 0.01) / (atr * 1.0) * 1.0
        # 0.01 = 10 / atr → atr = 1000
        result = sizer.calculate_size(
            account_equity=Decimal("1000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("1000"),
            atr_multiplier=Decimal("1.0"),
            current_volatility_zscore=0.0,
            min_lot_size=Decimal("0.01"),
        )
        # size = (1000 * 0.01) / (1000 * 1.0) = 10 / 1000 = 0.01
        assert not result.rejected
        assert result.size == Decimal("0.01")

    def test_no_reduction_factors_passed(self, sizer: PositionSizer) -> None:
        """None reduction_factors should work fine."""
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=0.0,
            reduction_factors=None,
        )
        assert not result.rejected
        assert result.size == Decimal("13.33")
        assert result.applied_reductions == []

    def test_full_stacking_scenario(self, sizer: PositionSizer) -> None:
        """Full scenario: volatility × drawdown × mistake × news."""
        factors = [
            ReductionFactor(source="drawdown", factor=0.25, reason="Drawdown > 10%"),
            ReductionFactor(source="mistake", factor=0.7, reason="Active mistake pattern"),
            ReductionFactor(source="news", factor=0.5, reason="High-impact event"),
        ]
        result = sizer.calculate_size(
            account_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            atr=Decimal("50"),
            atr_multiplier=Decimal("1.5"),
            current_volatility_zscore=2.5,
            reduction_factors=factors,
        )
        # Base: 13.333...
        # All factors: 0.5 * 0.25 * 0.7 * 0.5 = 0.04375
        # Final: 13.333... * 0.04375 = 0.58333...
        # Quantized: 0.58
        assert not result.rejected
        assert result.size == Decimal("0.58")
        assert len(result.applied_reductions) == 4
