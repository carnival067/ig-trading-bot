"""Unit tests for Monte Carlo simulation.

Tests cover:
- Task 23.3: Trade-order shuffling simulation (1000 iterations)
- Task 23.4: Percentile calculations (95th percentile worst drawdown,
             probability distribution, confidence intervals)
"""

import numpy as np
import pytest

from src.backtesting.monte_carlo import MonteCarloResult, MonteCarloSimulator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simulator() -> MonteCarloSimulator:
    """Create a simulator with fixed seed for reproducibility."""
    return MonteCarloSimulator(iterations=1000, seed=42)


@pytest.fixture
def profitable_returns() -> list[float]:
    """Returns with net positive outcome."""
    # 70% wins of +100, 30% losses of -80
    wins = [100.0] * 70
    losses = [-80.0] * 30
    return wins + losses


@pytest.fixture
def losing_returns() -> list[float]:
    """Returns with net negative outcome."""
    # 30% wins of +50, 70% losses of -100
    wins = [50.0] * 30
    losses = [-100.0] * 70
    return wins + losses


@pytest.fixture
def breakeven_returns() -> list[float]:
    """Returns that roughly break even."""
    # 50% wins of +100, 50% losses of -100
    wins = [100.0] * 50
    losses = [-100.0] * 50
    return wins + losses


# =============================================================================
# Task 23.3: Trade-Order Shuffling
# =============================================================================


class TestTradeOrderShuffling:
    """Task 23.3: Monte Carlo with trade-order shuffling."""

    def test_default_iterations(self) -> None:
        """Default is 1000 iterations."""
        sim = MonteCarloSimulator()
        assert sim.iterations == 1000

    def test_custom_iterations(self) -> None:
        """Custom iteration count is respected."""
        sim = MonteCarloSimulator(iterations=500)
        assert sim.iterations == 500

    def test_invalid_iterations_raises(self) -> None:
        """Zero or negative iterations raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            MonteCarloSimulator(iterations=0)
        with pytest.raises(ValueError, match="at least 1"):
            MonteCarloSimulator(iterations=-10)

    def test_empty_returns_raises(self, simulator: MonteCarloSimulator) -> None:
        """Empty trade returns raises ValueError."""
        with pytest.raises(ValueError, match="not be empty"):
            simulator.simulate([])

    def test_result_iteration_count(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Result reports correct iteration count."""
        result = simulator.simulate(profitable_returns)
        assert result.iteration_count == 1000

    def test_shuffling_preserves_trade_set(
        self, profitable_returns: list[float]
    ) -> None:
        """Each iteration uses the same trades, just reordered.

        The total return should be the same regardless of order
        (sum is commutative), so median_return should equal the
        actual total return divided by initial equity.
        """
        sim = MonteCarloSimulator(iterations=100, seed=42)
        initial_equity = 100000.0
        result = sim.simulate(profitable_returns, initial_equity=initial_equity)

        # Total PnL is always the same regardless of order
        expected_total_pnl = sum(profitable_returns)
        expected_return = expected_total_pnl / initial_equity

        # Median return should equal expected (since sum is order-independent)
        assert abs(result.median_return - expected_return) < 0.001

    def test_single_trade(self, simulator: MonteCarloSimulator) -> None:
        """Single trade produces deterministic result."""
        result = simulator.simulate([500.0], initial_equity=10000.0)
        # Only one trade, shuffling doesn't change anything
        assert abs(result.median_return - 0.05) < 0.001  # 500/10000
        assert result.prob_profit == 1.0

    def test_reproducibility_with_seed(self, profitable_returns: list[float]) -> None:
        """Same seed produces same results."""
        sim1 = MonteCarloSimulator(iterations=100, seed=123)
        sim2 = MonteCarloSimulator(iterations=100, seed=123)

        result1 = sim1.simulate(profitable_returns)
        result2 = sim2.simulate(profitable_returns)

        assert result1.median_return == result2.median_return
        assert result1.p95_drawdown == result2.p95_drawdown


# =============================================================================
# Task 23.4: Percentile Calculations
# =============================================================================


class TestPercentileCalculations:
    """Task 23.4: 95th percentile drawdown, probability distribution, CIs."""

    def test_p95_drawdown_greater_than_p5(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """95th percentile drawdown >= 5th percentile drawdown."""
        result = simulator.simulate(profitable_returns)
        assert result.p95_drawdown >= result.p5_drawdown

    def test_p95_drawdown_greater_than_median(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """95th percentile drawdown >= median max drawdown."""
        result = simulator.simulate(profitable_returns)
        assert result.p95_drawdown >= result.median_max_drawdown

    def test_drawdowns_non_negative(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """All drawdown percentiles are non-negative."""
        result = simulator.simulate(profitable_returns)
        assert result.p5_drawdown >= 0
        assert result.p95_drawdown >= 0
        assert result.median_max_drawdown >= 0

    def test_prob_profit_bounded(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Probability of profit is between 0 and 1."""
        result = simulator.simulate(profitable_returns)
        assert 0.0 <= result.prob_profit <= 1.0

    def test_prob_profit_high_for_profitable(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Profitable trade set has high probability of profit.

        Since sum of returns is always the same (positive), prob_profit = 1.0.
        """
        result = simulator.simulate(profitable_returns)
        # Sum is always positive regardless of order
        assert result.prob_profit == 1.0

    def test_prob_profit_low_for_losing(
        self, simulator: MonteCarloSimulator, losing_returns: list[float]
    ) -> None:
        """Losing trade set has low probability of profit.

        Since sum of returns is always negative, prob_profit = 0.0.
        """
        result = simulator.simulate(losing_returns)
        assert result.prob_profit == 0.0

    def test_prob_ruin_for_losing_set(
        self, simulator: MonteCarloSimulator, losing_returns: list[float]
    ) -> None:
        """Losing trade set has high probability of ruin (ending in loss)."""
        result = simulator.simulate(losing_returns)
        # Sum is always negative regardless of order → prob_ruin = 1.0
        assert result.prob_ruin == 1.0

    def test_prob_ruin_for_profitable_set(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Profitable trade set has zero probability of ruin."""
        result = simulator.simulate(profitable_returns)
        # Sum is always positive regardless of order → prob_ruin = 0.0
        assert result.prob_ruin == 0.0

    def test_prob_ruin_plus_profit_lte_one(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """prob_ruin + prob_profit <= 1.0 (some iterations may break even)."""
        result = simulator.simulate(profitable_returns)
        assert result.prob_ruin + result.prob_profit <= 1.0 + 1e-9

    def test_confidence_intervals_present(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Result includes 90%, 95%, and 99% confidence intervals."""
        result = simulator.simulate(profitable_returns)
        assert "90%" in result.confidence_intervals
        assert "95%" in result.confidence_intervals
        assert "99%" in result.confidence_intervals

    def test_confidence_intervals_ordered(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Wider confidence intervals contain narrower ones."""
        result = simulator.simulate(profitable_returns)
        ci_90 = result.confidence_intervals["90%"]
        ci_95 = result.confidence_intervals["95%"]
        ci_99 = result.confidence_intervals["99%"]

        # 99% CI should be wider than 95% which is wider than 90%
        assert ci_99[0] <= ci_95[0] <= ci_90[0]
        assert ci_90[1] <= ci_95[1] <= ci_99[1]

    def test_confidence_interval_structure(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Each CI is a tuple of (lower, upper) with lower <= upper."""
        result = simulator.simulate(profitable_returns)
        for key, (lower, upper) in result.confidence_intervals.items():
            assert lower <= upper, f"CI {key}: lower ({lower}) > upper ({upper})"

    def test_median_return_within_95_ci(
        self, simulator: MonteCarloSimulator, profitable_returns: list[float]
    ) -> None:
        """Median return falls within the 95% confidence interval."""
        result = simulator.simulate(profitable_returns)
        ci_95 = result.confidence_intervals["95%"]
        assert ci_95[0] <= result.median_return <= ci_95[1]

    def test_drawdown_with_all_positive_returns(
        self, simulator: MonteCarloSimulator
    ) -> None:
        """All positive returns still have some drawdown path (from ordering)."""
        # With all positive returns, no drawdown is possible
        returns = [100.0] * 20
        result = simulator.simulate(returns)
        # No drawdown possible when all returns are positive
        assert result.p95_drawdown == 0.0
        assert result.median_max_drawdown == 0.0

    def test_drawdown_increases_with_volatility(self) -> None:
        """Higher variance in returns leads to larger drawdowns."""
        sim = MonteCarloSimulator(iterations=500, seed=42)

        # Low volatility
        low_vol = [10.0, -5.0] * 50
        result_low = sim.simulate(low_vol)

        # High volatility (same mean)
        high_vol = [100.0, -95.0] * 50
        sim2 = MonteCarloSimulator(iterations=500, seed=42)
        result_high = sim2.simulate(high_vol)

        assert result_high.p95_drawdown > result_low.p95_drawdown
