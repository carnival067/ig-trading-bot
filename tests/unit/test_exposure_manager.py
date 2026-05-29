"""Unit tests for the ExposureManager.

Tests per-asset-class exposure limits, total exposure limits,
geopolitical risk integration, and position validation logic.

Validates: Requirements 5.4, 5.5, 23.16
"""

from decimal import Decimal

import pytest

from src.risk.exposure_manager import (
    AssetClass,
    ExposureCheckResult,
    ExposureManager,
    GEOPOLITICAL_RISK_THRESHOLD,
    Position,
)


@pytest.fixture
def manager() -> ExposureManager:
    """Create an ExposureManager with default limits (30% per class, 70% total)."""
    return ExposureManager()


@pytest.fixture
def equity() -> Decimal:
    """Standard account equity for tests."""
    return Decimal("100000")


# =============================================================================
# Task 5.1: Per-asset-class exposure tracking and geopolitical risk integration
# =============================================================================


class TestGetClassExposure:
    """Tests for get_class_exposure calculation."""

    def test_no_positions_returns_zero(self, manager: ExposureManager, equity: Decimal) -> None:
        result = manager.get_class_exposure(AssetClass.FOREX, [], equity)
        assert result == Decimal("0")

    def test_single_position_in_class(self, manager: ExposureManager, equity: Decimal) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("20000")),
        ]
        result = manager.get_class_exposure(AssetClass.FOREX, positions, equity)
        assert result == Decimal("0.2")  # 20000 / 100000

    def test_multiple_positions_in_same_class(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("10000")),
            Position("GBP/USD", AssetClass.FOREX, Decimal("15000")),
        ]
        result = manager.get_class_exposure(AssetClass.FOREX, positions, equity)
        assert result == Decimal("0.25")  # 25000 / 100000

    def test_ignores_other_asset_classes(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("20000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("30000")),
        ]
        result = manager.get_class_exposure(AssetClass.FOREX, positions, equity)
        assert result == Decimal("0.2")  # Only FOREX counted

    def test_uses_absolute_notional_value(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("-15000")),  # Short position
        ]
        result = manager.get_class_exposure(AssetClass.FOREX, positions, equity)
        assert result == Decimal("0.15")

    def test_zero_equity_returns_zero(self, manager: ExposureManager) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("10000")),
        ]
        result = manager.get_class_exposure(AssetClass.FOREX, positions, Decimal("0"))
        assert result == Decimal("0")


class TestGeopoliticalRiskIntegration:
    """Tests for geopolitical risk halving the per-class limit."""

    def test_no_geo_risk_uses_full_limit(self, manager: ExposureManager, equity: Decimal) -> None:
        # 29% exposure should be allowed without geo risk
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("29000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True

    def test_geo_risk_above_threshold_halves_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # 20% exposure should be rejected when geo risk > 70 (limit becomes 15%)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("20000"), region="europe")
        geo_scores = {"europe": 75.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is False
        assert "geopolitical risk" in result.rejection_reason.lower()

    def test_geo_risk_at_threshold_uses_full_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Exactly 70 should NOT trigger halving (must be > 70)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region="europe")
        geo_scores = {"europe": 70.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is True

    def test_geo_risk_below_threshold_uses_full_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region="europe")
        geo_scores = {"europe": 50.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is True

    def test_geo_risk_halved_limit_allows_within_15pct(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # 14% should be allowed even with halved limit
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("14000"), region="europe")
        geo_scores = {"europe": 80.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is True

    def test_geo_risk_only_affects_matching_region(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # High geo risk in "asia" should not affect "europe" positions
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region="europe")
        geo_scores = {"asia": 90.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is True

    def test_no_region_on_position_uses_full_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Position without region should use full limit regardless of geo scores
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region=None)
        geo_scores = {"europe": 90.0}
        result = manager.check_exposure(new_pos, [], equity, geo_scores)
        assert result.allowed is True


# =============================================================================
# Task 5.2: Total exposure limit enforcement (max 70%)
# =============================================================================


class TestGetTotalExposure:
    """Tests for get_total_exposure calculation."""

    def test_no_positions_returns_zero(self, manager: ExposureManager, equity: Decimal) -> None:
        result = manager.get_total_exposure([], equity)
        assert result == Decimal("0")

    def test_single_position(self, manager: ExposureManager, equity: Decimal) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("30000")),
        ]
        result = manager.get_total_exposure(positions, equity)
        assert result == Decimal("0.3")

    def test_multiple_positions_across_classes(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("20000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("25000")),
            Position("GOLD", AssetClass.COMMODITIES, Decimal("15000")),
        ]
        result = manager.get_total_exposure(positions, equity)
        assert result == Decimal("0.6")  # 60000 / 100000

    def test_uses_absolute_values(self, manager: ExposureManager, equity: Decimal) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("-20000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("30000")),
        ]
        result = manager.get_total_exposure(positions, equity)
        assert result == Decimal("0.5")  # 50000 / 100000

    def test_zero_equity_returns_zero(self, manager: ExposureManager) -> None:
        positions = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("10000")),
        ]
        result = manager.get_total_exposure(positions, Decimal("0"))
        assert result == Decimal("0")


class TestTotalExposureLimit:
    """Tests for total exposure limit enforcement at 70%."""

    def test_within_total_limit_allowed(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Existing 40% + new 25% = 65% < 70%
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("20000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("20000")),
        ]
        new_pos = Position("GOLD", AssetClass.COMMODITIES, Decimal("25000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is True

    def test_exceeding_total_limit_rejected(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Existing 60% + new 15% = 75% > 70%
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("30000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("30000")),
        ]
        new_pos = Position("GOLD", AssetClass.COMMODITIES, Decimal("15000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is False
        assert "total exposure limit" in result.rejection_reason.lower()

    def test_exactly_at_total_limit_rejected(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Existing 50% + new 20.01% > 70% (just over)
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("25000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("25000")),
        ]
        new_pos = Position("GOLD", AssetClass.COMMODITIES, Decimal("20001"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is False

    def test_exactly_at_70pct_allowed(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Existing 50% + new 20% = 70% exactly (not exceeding)
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("25000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("25000")),
        ]
        new_pos = Position("GOLD", AssetClass.COMMODITIES, Decimal("20000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is True


# =============================================================================
# Task 5.3: Position validation that rejects trades breaching either limit
# =============================================================================


class TestCheckExposure:
    """Tests for the main check_exposure validation method."""

    def test_empty_portfolio_allows_small_position(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("10000"))
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True
        assert result.rejection_reason is None
        assert result.current_class_exposure == Decimal("0")
        assert result.current_total_exposure == Decimal("0")

    def test_per_class_limit_breach_rejected(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Already at 25% FOREX, adding 10% would exceed 30%
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("25000")),
        ]
        new_pos = Position("GBP/USD", AssetClass.FOREX, Decimal("10000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is False
        assert "per-asset-class" in result.rejection_reason.lower()
        assert result.current_class_exposure == Decimal("0.25")

    def test_total_limit_breach_rejected(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # Spread across classes but total > 70%
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("25000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("25000")),
            Position("GOLD", AssetClass.COMMODITIES, Decimal("15000")),
        ]
        new_pos = Position("BTC", AssetClass.CRYPTO, Decimal("10000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is False
        assert "total exposure limit" in result.rejection_reason.lower()

    def test_both_limits_satisfied_allowed(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("20000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("20000")),
        ]
        new_pos = Position("GOLD", AssetClass.COMMODITIES, Decimal("15000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is True
        assert result.current_total_exposure == Decimal("0.4")

    def test_zero_equity_rejected(self, manager: ExposureManager) -> None:
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("1000"))
        result = manager.check_exposure(new_pos, [], Decimal("0"))
        assert result.allowed is False
        assert "equity" in result.rejection_reason.lower()

    def test_negative_equity_rejected(self, manager: ExposureManager) -> None:
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("1000"))
        result = manager.check_exposure(new_pos, [], Decimal("-5000"))
        assert result.allowed is False

    def test_per_class_checked_before_total(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # If both limits would be breached, per-class is checked first
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("29000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("29000")),
            Position("GOLD", AssetClass.COMMODITIES, Decimal("10000")),
        ]
        # Adding 5000 FOREX: class would be 34% (>30%) AND total would be 73% (>70%)
        new_pos = Position("GBP/USD", AssetClass.FOREX, Decimal("5000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is False
        assert "per-asset-class" in result.rejection_reason.lower()

    def test_result_contains_current_exposures(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("15000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("10000")),
        ]
        new_pos = Position("GBP/USD", AssetClass.FOREX, Decimal("5000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.current_class_exposure == Decimal("0.15")
        assert result.current_total_exposure == Decimal("0.25")

    def test_custom_limits(self, equity: Decimal) -> None:
        # Custom manager with tighter limits
        manager = ExposureManager(
            max_per_class=Decimal("0.20"),
            max_total=Decimal("0.50"),
        )
        # 15% should be allowed (< 20% per class, < 50% total)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("15000"))
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True

        # 25% should be rejected (> 20% per class)
        new_pos_large = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"))
        result = manager.check_exposure(new_pos_large, [], equity)
        assert result.allowed is False


class TestEdgeCases:
    """Edge case tests for exposure manager."""

    def test_zero_notional_position_allowed(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("0"))
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True

    def test_all_asset_classes_tracked(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        # One position per class, each at 10%
        existing = [
            Position("EUR/USD", AssetClass.FOREX, Decimal("10000")),
            Position("FTSE100", AssetClass.INDICES, Decimal("10000")),
            Position("GOLD", AssetClass.COMMODITIES, Decimal("10000")),
            Position("BTC", AssetClass.CRYPTO, Decimal("10000")),
            Position("AAPL", AssetClass.STOCKS, Decimal("10000")),
        ]
        # Total is 50%, adding 15% FOREX = 65% total, 25% FOREX class
        new_pos = Position("GBP/USD", AssetClass.FOREX, Decimal("15000"))
        result = manager.check_exposure(new_pos, existing, equity)
        assert result.allowed is True

    def test_geopolitical_risk_threshold_constant(self) -> None:
        """Verify the threshold constant is 70."""
        assert GEOPOLITICAL_RISK_THRESHOLD == 70.0


# =============================================================================
# Task 39.2: update_geo_risk method and internal score integration
# =============================================================================


class TestUpdateGeoRisk:
    """Tests for the update_geo_risk method and internal geo risk score management."""

    def test_update_geo_risk_stores_score(self, manager: ExposureManager) -> None:
        manager.update_geo_risk("europe", 75)
        assert manager.geo_risk_scores == {"europe": 75.0}

    def test_update_geo_risk_multiple_regions(self, manager: ExposureManager) -> None:
        manager.update_geo_risk("europe", 75)
        manager.update_geo_risk("asia", 40)
        assert manager.geo_risk_scores == {"europe": 75.0, "asia": 40.0}

    def test_update_geo_risk_overwrites_existing(self, manager: ExposureManager) -> None:
        manager.update_geo_risk("europe", 50)
        manager.update_geo_risk("europe", 80)
        assert manager.geo_risk_scores["europe"] == 80.0

    def test_update_geo_risk_clamps_to_max_100(self, manager: ExposureManager) -> None:
        manager.update_geo_risk("europe", 150)
        assert manager.geo_risk_scores["europe"] == 100.0

    def test_update_geo_risk_clamps_to_min_0(self, manager: ExposureManager) -> None:
        manager.update_geo_risk("europe", -10)
        assert manager.geo_risk_scores["europe"] == 0.0

    def test_internal_scores_used_when_no_explicit_scores(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        """check_exposure uses internal geo risk scores when none are passed."""
        manager.update_geo_risk("europe", 80)
        # 20% exposure should be rejected when internal geo risk > 70 (limit becomes 15%)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("20000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is False
        assert "geopolitical risk" in result.rejection_reason.lower()

    def test_internal_scores_allow_within_halved_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        """14% exposure allowed with halved limit from internal scores."""
        manager.update_geo_risk("europe", 80)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("14000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True

    def test_explicit_scores_override_internal(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        """Explicit geo scores take precedence over internal scores."""
        # Internal score is high
        manager.update_geo_risk("europe", 80)
        # But explicit scores say low risk
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity, geopolitical_risk_scores={"europe": 30.0})
        assert result.allowed is True

    def test_internal_score_at_threshold_uses_full_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        """Score exactly at 70 should NOT trigger halving (must be > 70)."""
        manager.update_geo_risk("europe", 70)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("25000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is True

    def test_internal_score_just_above_threshold_halves_limit(
        self, manager: ExposureManager, equity: Decimal
    ) -> None:
        """Score of 71 should trigger halving."""
        manager.update_geo_risk("europe", 71)
        new_pos = Position("EUR/USD", AssetClass.FOREX, Decimal("20000"), region="europe")
        result = manager.check_exposure(new_pos, [], equity)
        assert result.allowed is False
        assert "geopolitical risk" in result.rejection_reason.lower()

    def test_geo_risk_scores_property_returns_copy(self, manager: ExposureManager) -> None:
        """The geo_risk_scores property returns a copy, not the internal dict."""
        manager.update_geo_risk("europe", 75)
        scores = manager.geo_risk_scores
        scores["asia"] = 90.0
        # Internal state should not be affected
        assert "asia" not in manager.geo_risk_scores
