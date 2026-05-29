"""Backtesting engine module.

Provides backtest simulation, walk-forward optimization, and Monte Carlo
analysis for strategy validation before live deployment.
"""

from src.backtesting.backtest_engine import (
    BacktestEngine,
    InsufficientDataError,
    SharpeGatingError,
)
from src.backtesting.metrics import BacktestResult, Trade, calculate_metrics
from src.backtesting.monte_carlo import MonteCarloResult, MonteCarloSimulator
from src.backtesting.walk_forward import OptimizationResult, WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "InsufficientDataError",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "OptimizationResult",
    "SharpeGatingError",
    "Trade",
    "WalkForwardOptimizer",
    "calculate_metrics",
]
