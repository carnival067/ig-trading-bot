"""Unit tests for HFT Override Manager.

Tests the HFTOverrideManager class which coordinates the interaction between
HFT signals and overtrading guard, mistake pattern penalties, and HFT-specific
risk controls per Cross-Cutting Rule 2.

Covers:
- HFT signals bypass overtrading guard
- HFT signals retain mistake pattern penalties
- Non-HFT signals go through overtrading guard normally
- HFT signals still subject to HFT risk controls
"""

from decimal import Decimal

import pytest

from src.strategy.hft_override import (
    HFTOverrideManager,
    HFTRiskCheckResult,
    Penalty,
)


@pytest.fixture
def manager() -> HFTOverrideManager:
    """Create an HFTOverrideManager instance for testing."""
    return HFTOverrideManager()


# =============================================================================
# Test: HFT signals bypass overtrading guard
# =============================================================================


class TestOvertradingGuardBypass:
    """Tests that HFT signals bypass the overtrading guard."""

    def test_hft_signal_bypasses_overtrading_guard(self, manager: HFTOverrideManager) -> None:
        """HFT signals should NOT have overtrading guard applied."""
        result = manager.should_apply_overtrading_guard(is_hft=True)
        assert result is False

    def test_non_hft_signal_goes_through_overtrading_guard(self, manager: HFTOverrideManager) -> None:
        """Non-HFT signals should have overtrading guard applied."""
        result = manager.should_apply_overtrading_guard(is_hft=False)
        assert result is True

    def test_hft_bypass_is_consistent(self, manager: HFTOverrideManager) -> None:
        """Multiple calls with is_hft=True should always return False."""
        for _ in range(10):
            assert manager.should_apply_overtrading_guard(is_hft=True) is False

    def test_non_hft_guard_is_consistent(self, manager: HFTOverrideManager) -> None:
        """Multiple calls with is_hft=False should always return True."""
        for _ in range(10):
            assert manager.should_apply_overtrading_guard(is_hft=False) is True


# =============================================================================
# Test: HFT signals retain mistake pattern penalties
# =============================================================================


class TestMistakePatternPenalties:
    """Tests that HFT signals still receive mistake pattern penalties."""

    def test_hft_with_active_mistake_pattern_gets_confidence_penalty(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signals matching an active mistake pattern get -20 confidence penalty."""
        penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=True, is_reactivated=False
        )
        confidence_penalties = [p for p in penalties if p.penalty_type == "confidence"]
        assert len(confidence_penalties) == 1
        assert confidence_penalties[0].value == -20
        assert confidence_penalties[0].source == "mistake_pattern"

    def test_hft_with_active_mistake_pattern_gets_size_reduction(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signals matching an active mistake pattern get 0.7 size reduction."""
        penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=True, is_reactivated=False
        )
        size_penalties = [p for p in penalties if p.penalty_type == "size_reduction"]
        assert len(size_penalties) == 1
        assert size_penalties[0].value == 0.7
        assert size_penalties[0].source == "mistake_pattern"

    def test_hft_with_reactivated_pattern_gets_stronger_confidence_penalty(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signals matching a reactivated pattern get -30 confidence penalty."""
        penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=True, is_reactivated=True
        )
        confidence_penalties = [p for p in penalties if p.penalty_type == "confidence"]
        assert len(confidence_penalties) == 1
        assert confidence_penalties[0].value == -30
        assert confidence_penalties[0].source == "mistake_pattern_reactivated"

    def test_hft_with_reactivated_pattern_gets_stronger_size_reduction(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signals matching a reactivated pattern get 0.5 size reduction."""
        penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=True, is_reactivated=True
        )
        size_penalties = [p for p in penalties if p.penalty_type == "size_reduction"]
        assert len(size_penalties) == 1
        assert size_penalties[0].value == 0.5
        assert size_penalties[0].source == "mistake_pattern_reactivated"

    def test_hft_without_mistake_pattern_gets_no_penalties(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signals without a mistake pattern match get no penalties."""
        penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=False
        )
        assert penalties == []

    def test_non_hft_with_mistake_pattern_also_gets_penalties(
        self, manager: HFTOverrideManager
    ) -> None:
        """Non-HFT signals also get mistake pattern penalties (universal rule)."""
        penalties = manager.get_applicable_penalties(
            is_hft=False, has_mistake_pattern=True, is_reactivated=False
        )
        assert len(penalties) == 2  # confidence + size_reduction
        confidence_penalties = [p for p in penalties if p.penalty_type == "confidence"]
        size_penalties = [p for p in penalties if p.penalty_type == "size_reduction"]
        assert confidence_penalties[0].value == -20
        assert size_penalties[0].value == 0.7

    def test_penalties_same_for_hft_and_non_hft(
        self, manager: HFTOverrideManager
    ) -> None:
        """Mistake pattern penalties are identical for HFT and non-HFT signals."""
        hft_penalties = manager.get_applicable_penalties(
            is_hft=True, has_mistake_pattern=True, is_reactivated=False
        )
        non_hft_penalties = manager.get_applicable_penalties(
            is_hft=False, has_mistake_pattern=True, is_reactivated=False
        )
        # Same number and values
        assert len(hft_penalties) == len(non_hft_penalties)
        for hp, nhp in zip(hft_penalties, non_hft_penalties):
            assert hp.penalty_type == nhp.penalty_type
            assert hp.value == nhp.value


# =============================================================================
# Test: Non-HFT signals go through overtrading guard normally
# =============================================================================


class TestNonHFTOvertradingGuard:
    """Tests that non-HFT signals are subject to overtrading guard."""

    def test_non_hft_requires_overtrading_guard(self, manager: HFTOverrideManager) -> None:
        """Non-HFT signals must pass through the overtrading guard."""
        assert manager.should_apply_overtrading_guard(is_hft=False) is True

    def test_non_hft_with_no_pattern_gets_no_penalties(
        self, manager: HFTOverrideManager
    ) -> None:
        """Non-HFT signals without mistake patterns get no penalties."""
        penalties = manager.get_applicable_penalties(
            is_hft=False, has_mistake_pattern=False
        )
        assert penalties == []

    def test_non_hft_with_reactivated_pattern(
        self, manager: HFTOverrideManager
    ) -> None:
        """Non-HFT signals with reactivated patterns get the stronger penalties."""
        penalties = manager.get_applicable_penalties(
            is_hft=False, has_mistake_pattern=True, is_reactivated=True
        )
        confidence_penalties = [p for p in penalties if p.penalty_type == "confidence"]
        size_penalties = [p for p in penalties if p.penalty_type == "size_reduction"]
        assert confidence_penalties[0].value == -30
        assert size_penalties[0].value == 0.5


# =============================================================================
# Test: HFT signals still subject to HFT risk controls
# =============================================================================


class TestHFTRiskControls:
    """Tests that HFT signals are still subject to HFT-specific risk controls."""

    def test_trade_within_size_limit_allowed(self, manager: HFTOverrideManager) -> None:
        """Trade within 0.5% of equity should be allowed."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.allowed is True
        assert result.reason is None

    def test_trade_exceeding_size_limit_rejected(self, manager: HFTOverrideManager) -> None:
        """Trade exceeding 0.5% of equity should be rejected."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("600"),  # 0.6% of 100000
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.allowed is False
        assert "0.5%" in result.reason

    def test_trade_at_exact_size_limit_allowed(self, manager: HFTOverrideManager) -> None:
        """Trade at exactly 0.5% of equity should be allowed."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("500"),  # exactly 0.5% of 100000
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.allowed is True

    def test_exposure_within_limit_allowed(self, manager: HFTOverrideManager) -> None:
        """Total HFT exposure within 15% should be allowed."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("500"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("14000"),  # 14% + 0.5% = 14.5% < 15%
        )
        assert result.allowed is True

    def test_exposure_exceeding_limit_rejected(self, manager: HFTOverrideManager) -> None:
        """Total HFT exposure exceeding 15% should be rejected."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("500"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("14600"),  # 14.6% + 0.5% = 15.1% > 15%
        )
        assert result.allowed is False
        assert "15%" in result.reason

    def test_exposure_at_exact_limit_allowed(self, manager: HFTOverrideManager) -> None:
        """Total HFT exposure at exactly 15% should be allowed."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("500"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("14500"),  # 14.5% + 0.5% = 15% exactly
        )
        assert result.allowed is True

    def test_zero_equity_rejected(self, manager: HFTOverrideManager) -> None:
        """Zero account equity should reject the trade."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("100"),
            account_equity=Decimal("0"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.allowed is False
        assert "positive" in result.reason

    def test_negative_equity_rejected(self, manager: HFTOverrideManager) -> None:
        """Negative account equity should reject the trade."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("100"),
            account_equity=Decimal("-1000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.allowed is False

    def test_risk_check_returns_limits(self, manager: HFTOverrideManager) -> None:
        """Risk check result should include the calculated limits."""
        result = manager.check_hft_risk_controls(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
        )
        assert result.max_trade_size == Decimal("500")  # 0.5% of 100000
        assert result.max_exposure == Decimal("15000")  # 15% of 100000


# =============================================================================
# Test: Full HFT signal evaluation pipeline
# =============================================================================


class TestEvaluateHFTSignal:
    """Tests the full evaluate_hft_signal pipeline."""

    def test_clean_hft_signal_passes(self, manager: HFTOverrideManager) -> None:
        """HFT signal with no mistakes and within limits should pass."""
        bypassed, penalties, risk_result, confidence = manager.evaluate_hft_signal(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
            has_mistake_pattern=False,
            base_confidence=85,
        )
        assert bypassed is True
        assert penalties == []
        assert risk_result.allowed is True
        assert confidence == 85

    def test_hft_signal_with_mistake_pattern_reduces_confidence(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signal with active mistake pattern has confidence reduced by 20."""
        bypassed, penalties, risk_result, confidence = manager.evaluate_hft_signal(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
            has_mistake_pattern=True,
            is_reactivated=False,
            base_confidence=85,
        )
        assert bypassed is True
        assert len(penalties) == 2  # confidence + size_reduction
        assert risk_result.allowed is True
        assert confidence == 65  # 85 - 20

    def test_hft_signal_with_reactivated_pattern_reduces_confidence(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signal with reactivated pattern has confidence reduced by 30."""
        bypassed, penalties, risk_result, confidence = manager.evaluate_hft_signal(
            trade_size=Decimal("400"),
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
            has_mistake_pattern=True,
            is_reactivated=True,
            base_confidence=85,
        )
        assert bypassed is True
        assert confidence == 55  # 85 - 30

    def test_hft_signal_exceeding_risk_limits_fails(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signal exceeding risk limits should fail even without mistakes."""
        bypassed, penalties, risk_result, confidence = manager.evaluate_hft_signal(
            trade_size=Decimal("600"),  # > 0.5% of 100000
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
            has_mistake_pattern=False,
            base_confidence=90,
        )
        assert bypassed is True  # overtrading still bypassed
        assert risk_result.allowed is False
        assert confidence == 90  # no penalty applied

    def test_hft_signal_with_both_mistakes_and_risk_breach(
        self, manager: HFTOverrideManager
    ) -> None:
        """HFT signal with mistakes AND risk breach: both are reported."""
        bypassed, penalties, risk_result, confidence = manager.evaluate_hft_signal(
            trade_size=Decimal("600"),  # exceeds 0.5%
            account_equity=Decimal("100000"),
            current_hft_exposure=Decimal("0"),
            has_mistake_pattern=True,
            is_reactivated=True,
            base_confidence=90,
        )
        assert bypassed is True
        assert len(penalties) == 2  # confidence + size_reduction
        assert risk_result.allowed is False
        assert confidence == 60  # 90 - 30
