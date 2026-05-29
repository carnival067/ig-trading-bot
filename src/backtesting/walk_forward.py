"""Walk-forward optimization for strategy validation.

Implements chronological data splitting and walk-forward optimization
to validate strategy performance on out-of-sample data.

Validates: Requirements 13.2
"""

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.metrics import BacktestResult
from src.config.constants import (
    BACKTEST_WALK_FORWARD_IS_RATIO,
    BACKTEST_WALK_FORWARD_OOS_RATIO,
)
from src.strategy.strategies.base import BaseStrategy


@dataclass
class OptimizationResult:
    """Result of walk-forward optimization.

    Attributes:
        in_sample_result: Backtest result on in-sample data.
        out_of_sample_result: Backtest result on out-of-sample data.
        is_data_size: Number of rows in the in-sample dataset.
        oos_data_size: Number of rows in the out-of-sample dataset.
        is_sharpe: In-sample Sharpe ratio.
        oos_sharpe: Out-of-sample Sharpe ratio.
        passed_gating: Whether OOS Sharpe meets deployment threshold.
    """

    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult
    is_data_size: int
    oos_data_size: int
    is_sharpe: float
    oos_sharpe: float
    passed_gating: bool


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy validation.

    Splits historical data chronologically into in-sample (IS) and
    out-of-sample (OOS) periods, optimizes on IS, and validates on OOS.
    OOS always comes after IS to prevent look-ahead bias.

    Attributes:
        is_ratio: Fraction of data used for in-sample (default 0.70).
        oos_ratio: Fraction of data used for out-of-sample (default 0.30).
    """

    def __init__(
        self,
        is_ratio: float = BACKTEST_WALK_FORWARD_IS_RATIO,
        oos_ratio: float = BACKTEST_WALK_FORWARD_OOS_RATIO,
    ) -> None:
        if abs((is_ratio + oos_ratio) - 1.0) > 1e-9:
            raise ValueError(
                f"IS ratio ({is_ratio}) + OOS ratio ({oos_ratio}) must equal 1.0"
            )
        if is_ratio <= 0 or oos_ratio <= 0:
            raise ValueError("Both IS and OOS ratios must be positive")

        self.is_ratio = is_ratio
        self.oos_ratio = oos_ratio

    def split_data(
        self,
        data: pd.DataFrame,
        is_ratio: float | None = None,
        oos_ratio: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into in-sample and out-of-sample.

        The split preserves chronological order: OOS data always comes
        after IS data. No overlap between the two sets.

        Args:
            data: DataFrame with chronologically ordered data.
            is_ratio: Override in-sample ratio (default uses instance value).
            oos_ratio: Override out-of-sample ratio (default uses instance value).

        Returns:
            Tuple of (in_sample_data, out_of_sample_data).

        Raises:
            ValueError: If data is empty or too small to split.
        """
        if data.empty:
            raise ValueError("Cannot split empty dataset")

        ratio = is_ratio if is_ratio is not None else self.is_ratio
        split_idx = int(len(data) * ratio)

        # Ensure both splits have at least 1 row
        if split_idx == 0:
            split_idx = 1
        if split_idx >= len(data):
            split_idx = len(data) - 1

        in_sample = data.iloc[:split_idx].copy()
        out_of_sample = data.iloc[split_idx:].copy()

        return in_sample, out_of_sample

    def optimize(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_equity: Decimal = Decimal("100000"),
        backtest_engine: BacktestEngine | None = None,
    ) -> OptimizationResult:
        """Run walk-forward optimization: optimize on IS, validate on OOS.

        Args:
            strategy: Strategy to optimize and validate.
            data: Full historical dataset (chronologically ordered).
            initial_equity: Starting equity for backtest simulation.
            backtest_engine: Optional BacktestEngine instance (creates default if None).

        Returns:
            OptimizationResult with IS and OOS performance metrics.
        """
        engine = backtest_engine or BacktestEngine()

        # Split data chronologically
        is_data, oos_data = self.split_data(data)

        # Run backtest on in-sample data
        is_result = engine.run(strategy, is_data, initial_equity)

        # Run backtest on out-of-sample data
        oos_result = engine.run(strategy, oos_data, initial_equity)

        # Check Sharpe gating
        passed_gating = oos_result.sharpe_ratio >= 1.0

        return OptimizationResult(
            in_sample_result=is_result,
            out_of_sample_result=oos_result,
            is_data_size=len(is_data),
            oos_data_size=len(oos_data),
            is_sharpe=is_result.sharpe_ratio,
            oos_sharpe=oos_result.sharpe_ratio,
            passed_gating=passed_gating,
        )
