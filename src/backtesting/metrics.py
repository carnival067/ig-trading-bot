"""Backtesting performance metrics and calculations.

Provides BacktestResult dataclass and individual metric functions for
evaluating strategy performance during backtesting.

Validates: Requirements 13.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

from src.config.constants import BACKTEST_MIN_TRADES


@dataclass
class Trade:
    """Represents a completed trade for backtesting analysis.

    Attributes:
        entry_time: Timestamp when the trade was opened.
        exit_time: Timestamp when the trade was closed.
        entry_price: Price at which the trade was entered.
        exit_price: Price at which the trade was exited.
        direction: Trade direction, "LONG" or "SHORT".
        size: Position size in lots.
        pnl: Realized profit/loss for this trade.
        costs: Total costs (spread + slippage + commission) applied.
    """

    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    direction: str
    size: Decimal
    pnl: Decimal
    costs: Decimal = Decimal("0")


@dataclass
class BacktestResult:
    """Complete backtest performance summary.

    Attributes:
        total_return: Total percentage return over the backtest period.
        sharpe_ratio: Annualized Sharpe ratio (risk-free rate assumed 0).
        max_drawdown: Maximum peak-to-trough drawdown as a fraction.
        win_rate: Fraction of trades that were profitable.
        profit_factor: Gross profit / gross loss ratio.
        avg_trade_duration: Average duration of trades.
        trade_count: Total number of trades executed.
        equity_curve: List of equity values over time.
    """

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: timedelta
    trade_count: int
    equity_curve: list[float] = field(default_factory=list)


def sharpe_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio assuming risk-free rate of 0.

    Args:
        returns: List of periodic returns (e.g., daily returns).
        periods_per_year: Number of trading periods per year (252 for daily).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if insufficient data or zero std.
    """
    if len(returns) < 2:
        return 0.0

    arr = np.array(returns, dtype=np.float64)
    mean_return = float(np.mean(arr))
    std_return = float(np.std(arr, ddof=1))

    # Use relative tolerance to handle floating point imprecision
    # (e.g., identical values may produce tiny non-zero std)
    if std_return < 1e-12:
        return 0.0

    return (mean_return / std_return) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum peak-to-trough drawdown.

    Args:
        equity_curve: Sequential equity values.

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 = 15% drawdown).
        Returns 0.0 if equity curve has fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0

    arr = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(arr)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peak > 0, (peak - arr) / peak, 0.0)

    return float(np.max(drawdowns))


def profit_factor(trades: list[Trade]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of completed trades.

    Returns:
        Profit factor ratio. Returns float('inf') if no losing trades,
        0.0 if no winning trades.
    """
    gross_profit = sum(float(t.pnl) for t in trades if t.pnl > 0)
    gross_loss = abs(sum(float(t.pnl) for t in trades if t.pnl < 0))

    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def win_rate(trades: list[Trade]) -> float:
    """Calculate win rate (fraction of profitable trades).

    Args:
        trades: List of completed trades.

    Returns:
        Win rate as a fraction [0.0, 1.0]. Returns 0.0 if no trades.
    """
    if not trades:
        return 0.0

    winners = sum(1 for t in trades if t.pnl > 0)
    return winners / len(trades)


def calculate_metrics(trades: list[Trade], initial_equity: Decimal) -> BacktestResult:
    """Calculate comprehensive backtest metrics from a list of trades.

    Args:
        trades: List of completed trades in chronological order.
        initial_equity: Starting account equity.

    Returns:
        BacktestResult with all performance metrics calculated.
    """
    if not trades:
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_duration=timedelta(0),
            trade_count=0,
            equity_curve=[float(initial_equity)],
        )

    # Build equity curve
    equity = float(initial_equity)
    equity_curve = [equity]
    trade_returns: list[float] = []

    for trade in trades:
        pnl = float(trade.pnl)
        trade_returns.append(pnl / equity if equity > 0 else 0.0)
        equity += pnl
        equity_curve.append(equity)

    # Calculate total return
    total_return = (equity - float(initial_equity)) / float(initial_equity)

    # Calculate average trade duration
    durations = [t.exit_time - t.entry_time for t in trades]
    avg_duration = sum(durations, timedelta(0)) / len(durations)

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio(trade_returns),
        max_drawdown=max_drawdown(equity_curve),
        win_rate=win_rate(trades),
        profit_factor=profit_factor(trades),
        avg_trade_duration=avg_duration,
        trade_count=len(trades),
        equity_curve=equity_curve,
    )
