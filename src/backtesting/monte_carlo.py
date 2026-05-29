"""Monte Carlo simulation for trade return distribution analysis.

Implements trade-order shuffling to estimate the distribution of possible
outcomes from a given set of trade returns.

Validates: Requirements 13.4
"""

from dataclasses import dataclass, field

import numpy as np

from src.config.constants import MONTE_CARLO_ITERATIONS


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation.

    Attributes:
        median_return: Median total return across all iterations.
        p5_drawdown: 5th percentile worst drawdown (optimistic bound).
        p95_drawdown: 95th percentile worst drawdown (pessimistic bound).
        prob_ruin: Probability of ruin (fraction of iterations ending in loss).
        prob_profit: Probability of ending with a profit (fraction of iterations).
        confidence_intervals: Dict with confidence interval bounds for returns.
        iteration_count: Number of iterations performed.
        median_max_drawdown: Median maximum drawdown across iterations.
    """

    median_return: float
    p5_drawdown: float
    p95_drawdown: float
    prob_ruin: float
    prob_profit: float
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    iteration_count: int = 0
    median_max_drawdown: float = 0.0


class MonteCarloSimulator:
    """Monte Carlo simulator using trade-order shuffling.

    Shuffles the order of trades (random permutation) to generate
    different equity curve paths, estimating the distribution of
    possible outcomes from the same set of trades.

    Attributes:
        iterations: Number of simulation iterations (default 1000).
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        iterations: int = MONTE_CARLO_ITERATIONS,
        seed: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("Iterations must be at least 1")
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        trade_returns: list[float],
        initial_equity: float = 100000.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation by shuffling trade order.

        Each iteration creates a random permutation of the trade returns
        and calculates the resulting equity curve, total return, and
        maximum drawdown.

        Args:
            trade_returns: List of trade returns (absolute PnL values).
            initial_equity: Starting equity for simulation (default 100000).

        Returns:
            MonteCarloResult with distribution statistics.

        Raises:
            ValueError: If trade_returns is empty.
        """
        if not trade_returns:
            raise ValueError("trade_returns must not be empty")

        returns_arr = np.array(trade_returns, dtype=np.float64)
        n_trades = len(returns_arr)

        total_returns = np.zeros(self.iterations)
        max_drawdowns = np.zeros(self.iterations)

        for i in range(self.iterations):
            # Shuffle trade order (random permutation)
            shuffled = self.rng.permutation(returns_arr)

            # Build equity curve
            equity_curve = np.empty(n_trades + 1)
            equity_curve[0] = initial_equity
            equity_curve[1:] = initial_equity + np.cumsum(shuffled)

            # Calculate total return
            final_equity = equity_curve[-1]
            total_returns[i] = (final_equity - initial_equity) / initial_equity

            # Calculate max drawdown for this path
            peak = np.maximum.accumulate(equity_curve)
            with np.errstate(divide="ignore", invalid="ignore"):
                drawdowns = np.where(peak > 0, (peak - equity_curve) / peak, 0.0)
            max_drawdowns[i] = float(np.max(drawdowns))

        # Calculate statistics
        median_return = float(np.median(total_returns))
        p5_drawdown = float(np.percentile(max_drawdowns, 5))
        p95_drawdown = float(np.percentile(max_drawdowns, 95))
        median_max_drawdown = float(np.median(max_drawdowns))
        prob_profit = float(np.mean(total_returns > 0))
        prob_ruin = float(np.mean(total_returns < 0))

        # Confidence intervals for returns
        confidence_intervals = {
            "90%": (
                float(np.percentile(total_returns, 5)),
                float(np.percentile(total_returns, 95)),
            ),
            "95%": (
                float(np.percentile(total_returns, 2.5)),
                float(np.percentile(total_returns, 97.5)),
            ),
            "99%": (
                float(np.percentile(total_returns, 0.5)),
                float(np.percentile(total_returns, 99.5)),
            ),
        }

        return MonteCarloResult(
            median_return=median_return,
            p5_drawdown=p5_drawdown,
            p95_drawdown=p95_drawdown,
            prob_ruin=prob_ruin,
            prob_profit=prob_profit,
            confidence_intervals=confidence_intervals,
            iteration_count=self.iterations,
            median_max_drawdown=median_max_drawdown,
        )
