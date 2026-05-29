"""Unit tests for the allocation manager module.

Tests cover:
- Task 21.2: Proportional allocation (capped at 10% equity per trader, min 1%, max 10 traders)
- Allocation validation and bounds checking
"""

from decimal import Decimal

import pytest

from src.copy_trading.allocation_manager import AllocationManager, AllocationResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager() -> AllocationManager:
    return AllocationManager()


# =============================================================================
# Single Allocation Calculation
# =============================================================================


class TestCalculateAllocation:
    """Task 21.2: Proportional allocation with per-trader caps."""

    def test_proportional_allocation_basic(self, manager: AllocationManager) -> None:
        """Allocation is proportional to trader_score / total_scores."""
        # 30/100 = 30% of equity, but capped at 10%
        allocation = manager.calculate_allocation(
            trader_score=30.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 30% of 100k = 30k, capped at 10% = 10k
        assert allocation == Decimal("10000.00")

    def test_allocation_below_cap(self, manager: AllocationManager) -> None:
        """Allocation below 10% is not capped."""
        allocation = manager.calculate_allocation(
            trader_score=5.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 5% of 100k = 5000, within bounds [1%, 10%]
        assert allocation == Decimal("5000.00")

    def test_allocation_capped_at_10_percent(self, manager: AllocationManager) -> None:
        """Allocation exceeding 10% is capped."""
        allocation = manager.calculate_allocation(
            trader_score=80.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 80% would be 80k, capped at 10% = 10k
        assert allocation <= Decimal("10000.00")

    def test_allocation_floored_at_1_percent(self, manager: AllocationManager) -> None:
        """Allocation below 1% is floored."""
        allocation = manager.calculate_allocation(
            trader_score=0.5,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        # 0.5% would be 500, floored at 1% = 1000
        assert allocation >= Decimal("1000.00")

    def test_zero_score_returns_zero(self, manager: AllocationManager) -> None:
        """Zero trader score returns zero allocation."""
        allocation = manager.calculate_allocation(
            trader_score=0.0,
            total_scores=100.0,
            equity=Decimal("100000"),
        )
        assert allocation == Decimal("0")

    def test_zero_total_scores_returns_zero(self, manager: AllocationManager) -> None:
        """Zero total scores returns zero allocation."""
        allocation = manager.calculate_allocation(
            trader_score=50.0,
            total_scores=0.0,
            equity=Decimal("100000"),
        )
        assert allocation == Decimal("0")

    def test_zero_equity_returns_zero(self, manager: AllocationManager) -> None:
        """Zero equity returns zero allocation."""
        allocation = manager.calculate_allocation(
            trader_score=50.0,
            total_scores=100.0,
            equity=Decimal("0"),
        )
        assert allocation == Decimal("0")

    def test_negative_equity_returns_zero(self, manager: AllocationManager) -> None:
        """Negative equity returns zero allocation."""
        allocation = manager.calculate_allocation(
            trader_score=50.0,
            total_scores=100.0,
            equity=Decimal("-1000"),
        )
        assert allocation == Decimal("0")

    def test_allocation_quantized_to_cents(self, manager: AllocationManager) -> None:
        """Allocation is quantized to 2 decimal places."""
        allocation = manager.calculate_allocation(
            trader_score=7.0,
            total_scores=100.0,
            equity=Decimal("99999"),
        )
        # Should be rounded to 2 decimal places
        assert allocation == allocation.quantize(Decimal("0.01"))


# =============================================================================
# Batch Allocation Calculation
# =============================================================================


class TestCalculateAllocations:
    """Task 21.2: Batch allocation with max 10 traders."""

    def test_max_10_traders_enforced(self, manager: AllocationManager) -> None:
        """Should limit to maximum 10 traders."""
        scores = {f"T{i:02d}": float(50 + i) for i in range(15)}
        results = manager.calculate_allocations(scores, Decimal("100000"))
        assert len(results) <= 10

    def test_selects_top_traders_by_score(self, manager: AllocationManager) -> None:
        """Should select the top 10 traders by score."""
        scores = {f"T{i:02d}": float(i * 5) for i in range(15)}
        results = manager.calculate_allocations(scores, Decimal("100000"))
        result_ids = {r.trader_id for r in results}
        # Top 10 are T05 through T14 (scores 25 through 70)
        for i in range(5, 15):
            assert f"T{i:02d}" in result_ids

    def test_empty_scores_returns_empty(self, manager: AllocationManager) -> None:
        """Empty trader scores returns empty list."""
        results = manager.calculate_allocations({}, Decimal("100000"))
        assert results == []

    def test_zero_equity_returns_empty(self, manager: AllocationManager) -> None:
        """Zero equity returns empty allocations."""
        results = manager.calculate_allocations({"T1": 50.0}, Decimal("0"))
        assert results == []

    def test_results_sorted_by_allocation_descending(self, manager: AllocationManager) -> None:
        """Results are sorted by allocation amount descending."""
        scores = {"T1": 80.0, "T2": 60.0, "T3": 40.0}
        results = manager.calculate_allocations(scores, Decimal("100000"))
        allocations = [r.allocation for r in results]
        assert allocations == sorted(allocations, reverse=True)

    def test_single_trader_allocation(self, manager: AllocationManager) -> None:
        """Single trader gets full proportional allocation (capped at 10%)."""
        results = manager.calculate_allocations({"T1": 80.0}, Decimal("100000"))
        assert len(results) == 1
        # Single trader: 80/80 = 100% → capped at 10% = 10000
        assert results[0].allocation == Decimal("10000.00")
        assert results[0].trader_id == "T1"

    def test_allocation_result_has_correct_fields(self, manager: AllocationManager) -> None:
        """AllocationResult contains all expected fields."""
        results = manager.calculate_allocations({"T1": 50.0, "T2": 50.0}, Decimal("100000"))
        for result in results:
            assert isinstance(result, AllocationResult)
            assert isinstance(result.trader_id, str)
            assert isinstance(result.allocation, Decimal)
            assert isinstance(result.allocation_pct, Decimal)
            assert isinstance(result.capped, bool)


# =============================================================================
# Allocation Validation
# =============================================================================


class TestValidateAllocation:
    """Allocation validation within bounds."""

    def test_valid_allocation_within_bounds(self, manager: AllocationManager) -> None:
        """Allocation between 1% and 10% is valid."""
        assert manager.validate_allocation(Decimal("5000"), Decimal("100000")) is True
        assert manager.validate_allocation(Decimal("1000"), Decimal("100000")) is True
        assert manager.validate_allocation(Decimal("10000"), Decimal("100000")) is True

    def test_allocation_below_minimum_invalid(self, manager: AllocationManager) -> None:
        """Allocation below 1% is invalid."""
        assert manager.validate_allocation(Decimal("500"), Decimal("100000")) is False

    def test_allocation_above_maximum_invalid(self, manager: AllocationManager) -> None:
        """Allocation above 10% is invalid."""
        assert manager.validate_allocation(Decimal("15000"), Decimal("100000")) is False

    def test_zero_equity_invalid(self, manager: AllocationManager) -> None:
        """Zero equity makes any allocation invalid."""
        assert manager.validate_allocation(Decimal("5000"), Decimal("0")) is False

    def test_negative_equity_invalid(self, manager: AllocationManager) -> None:
        """Negative equity makes any allocation invalid."""
        assert manager.validate_allocation(Decimal("5000"), Decimal("-100000")) is False


# =============================================================================
# Properties
# =============================================================================


class TestAllocationManagerProperties:
    """Test configurable properties."""

    def test_default_max_traders(self, manager: AllocationManager) -> None:
        """Default max traders is 10."""
        assert manager.max_traders == 10

    def test_default_max_per_trader_pct(self, manager: AllocationManager) -> None:
        """Default max per trader is 10%."""
        assert manager.max_per_trader_pct == Decimal("0.10")

    def test_default_min_per_trader_pct(self, manager: AllocationManager) -> None:
        """Default min per trader is 1%."""
        assert manager.min_per_trader_pct == Decimal("0.01")

    def test_custom_configuration(self) -> None:
        """Custom configuration is respected."""
        custom = AllocationManager(
            max_per_trader_pct=0.15,
            min_per_trader_pct=0.02,
            max_traders=5,
        )
        assert custom.max_traders == 5
        assert custom.max_per_trader_pct == Decimal("0.15")
        assert custom.min_per_trader_pct == Decimal("0.02")
