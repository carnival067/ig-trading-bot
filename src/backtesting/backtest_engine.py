"""Backtest simulation engine with realistic cost modeling.

Provides BacktestEngine for running strategy backtests with spread,
slippage, and commission costs applied to each trade.

Validates: Requirements 13.1, 13.5, 13.6
"""

from datetime import timedelta
from decimal import Decimal

import pandas as pd

from src.backtesting.metrics import BacktestResult, Trade, calculate_metrics
from src.config.constants import (
    BACKTEST_DEFAULT_SLIPPAGE_PIPS,
    BACKTEST_MIN_DAYS,
    BACKTEST_MIN_TRADES,
    STRATEGY_ENABLE_SHARPE_THRESHOLD,
)
from src.strategy.strategies.base import BaseStrategy


class InsufficientDataError(Exception):
    """Raised when historical data does not meet minimum requirements."""

    pass


class SharpeGatingError(Exception):
    """Raised when OOS Sharpe ratio is below the deployment threshold."""

    def __init__(self, sharpe: float, threshold: float) -> None:
        self.sharpe = sharpe
        self.threshold = threshold
        super().__init__(
            f"OOS Sharpe ratio {sharpe:.4f} is below deployment threshold {threshold:.2f}"
        )


class BacktestEngine:
    """Realistic backtest simulation engine.

    Simulates strategy execution on historical data with realistic costs
    including spread, slippage, and commission per lot.

    Attributes:
        spread_pips: Spread cost in pips applied to each trade.
        slippage_pips: Simulated slippage in pips (default 0.5).
        commission_per_lot: Commission charged per lot traded.
    """

    def __init__(
        self,
        spread_pips: float = 1.0,
        slippage_pips: float = BACKTEST_DEFAULT_SLIPPAGE_PIPS,
        commission_per_lot: float = 7.0,
    ) -> None:
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot

    def validate_data(self, historical_data: pd.DataFrame) -> None:
        """Validate that historical data meets minimum requirements.

        Args:
            historical_data: DataFrame with datetime index and OHLCV columns.

        Raises:
            InsufficientDataError: If data has < 30 days or < 100 rows.
        """
        if historical_data.empty:
            raise InsufficientDataError("Historical data is empty")

        # Check minimum rows (proxy for minimum trades potential)
        if len(historical_data) < BACKTEST_MIN_TRADES:
            raise InsufficientDataError(
                f"Insufficient data: {len(historical_data)} rows, "
                f"minimum {BACKTEST_MIN_TRADES} required"
            )

        # Check minimum calendar days
        if hasattr(historical_data.index, "min") and hasattr(historical_data.index, "max"):
            try:
                date_range = historical_data.index.max() - historical_data.index.min()
                if hasattr(date_range, "days"):
                    days = date_range.days
                else:
                    days = date_range.total_seconds() / 86400
                if days < BACKTEST_MIN_DAYS:
                    raise InsufficientDataError(
                        f"Insufficient data span: {days:.0f} days, "
                        f"minimum {BACKTEST_MIN_DAYS} days required"
                    )
            except (TypeError, AttributeError):
                # If index is not datetime-like, check using a 'timestamp' column
                if "timestamp" in historical_data.columns:
                    date_range = (
                        historical_data["timestamp"].max()
                        - historical_data["timestamp"].min()
                    )
                    days = date_range.days if hasattr(date_range, "days") else 0
                    if days < BACKTEST_MIN_DAYS:
                        raise InsufficientDataError(
                            f"Insufficient data span: {days} days, "
                            f"minimum {BACKTEST_MIN_DAYS} days required"
                        )

    def _calculate_trade_costs(self, size: Decimal, pip_value: Decimal) -> Decimal:
        """Calculate total costs for a trade (spread + slippage + commission).

        Args:
            size: Position size in lots.
            pip_value: Value of one pip for the instrument.

        Returns:
            Total cost as a positive Decimal.
        """
        spread_cost = Decimal(str(self.spread_pips)) * pip_value * size
        slippage_cost = Decimal(str(self.slippage_pips)) * pip_value * size
        commission = Decimal(str(self.commission_per_lot)) * size
        return spread_cost + slippage_cost + commission

    def run(
        self,
        strategy: BaseStrategy,
        historical_data: pd.DataFrame,
        initial_equity: Decimal,
        pip_value: Decimal = Decimal("10"),
    ) -> BacktestResult:
        """Run a backtest simulation with realistic costs.

        Iterates through historical data, generates signals using the strategy,
        and simulates trade execution with spread, slippage, and commission costs.

        Args:
            strategy: Strategy instance to backtest.
            historical_data: DataFrame with datetime index and OHLCV columns
                (open, high, low, close, volume). Must have bid/ask columns
                for spread calculation, or spread_pips is used.
            initial_equity: Starting account equity.
            pip_value: Value of one pip for the instrument (default 10).

        Returns:
            BacktestResult with all performance metrics.

        Raises:
            InsufficientDataError: If data doesn't meet minimum requirements.
        """
        self.validate_data(historical_data)

        trades: list[Trade] = []
        equity = initial_equity
        position_open = False
        entry_price = Decimal("0")
        entry_time = None
        direction = ""
        size = Decimal("0")

        # Use a rolling window for signal generation
        min_window = min(50, len(historical_data) // 2)

        for i in range(min_window, len(historical_data)):
            window = historical_data.iloc[max(0, i - 200) : i + 1]
            current_bar = historical_data.iloc[i]

            if not position_open:
                # Try to generate a signal
                from src.strategy.regime_detector import MarketRegime

                signal = strategy.generate_signal(window, MarketRegime.TRENDING)

                if signal is not None:
                    position_open = True
                    direction = signal.direction
                    entry_price = Decimal(str(current_bar["close"]))
                    entry_time = (
                        historical_data.index[i]
                        if hasattr(historical_data.index[i], "timestamp")
                        else pd.Timestamp.now()
                    )
                    # Size based on equity fraction
                    size = Decimal(str(max(0.01, float(equity) * 0.01 / 100)))
            else:
                # Simple exit logic: exit after N bars or on stop/target
                current_price = Decimal(str(current_bar["close"]))
                bars_held = i - (
                    historical_data.index.get_loc(entry_time)
                    if entry_time in historical_data.index
                    else i - 10
                )

                # Exit after 10 bars or if price moved significantly
                price_change = abs(float(current_price - entry_price)) / float(entry_price)
                should_exit = bars_held >= 10 or price_change > 0.02

                if should_exit:
                    exit_price = current_price
                    exit_time = (
                        historical_data.index[i]
                        if hasattr(historical_data.index[i], "timestamp")
                        else pd.Timestamp.now()
                    )

                    # Calculate raw PnL
                    if direction == "LONG":
                        raw_pnl = (exit_price - entry_price) * size * pip_value
                    else:
                        raw_pnl = (entry_price - exit_price) * size * pip_value

                    # Apply costs
                    costs = self._calculate_trade_costs(size, pip_value)
                    net_pnl = raw_pnl - costs

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        size=size,
                        pnl=net_pnl,
                        costs=costs,
                    )
                    trades.append(trade)
                    equity += net_pnl
                    position_open = False

        return calculate_metrics(trades, initial_equity)

    def run_from_trades(
        self,
        trades: list[Trade],
        initial_equity: Decimal,
    ) -> BacktestResult:
        """Calculate backtest metrics from a pre-computed list of trades.

        This is useful when trades have already been generated externally
        (e.g., from a more sophisticated simulation or imported data).

        Args:
            trades: List of completed trades with costs already applied.
            initial_equity: Starting account equity.

        Returns:
            BacktestResult with all performance metrics.
        """
        return calculate_metrics(trades, initial_equity)

    @staticmethod
    def check_sharpe_gating(
        oos_sharpe: float,
        threshold: float = STRATEGY_ENABLE_SHARPE_THRESHOLD,
    ) -> bool:
        """Check if OOS Sharpe ratio meets deployment threshold.

        Args:
            oos_sharpe: Out-of-sample Sharpe ratio.
            threshold: Minimum Sharpe ratio for live deployment (default 1.0).

        Returns:
            True if Sharpe meets threshold, False otherwise.

        Raises:
            SharpeGatingError: If Sharpe is below threshold.
        """
        if oos_sharpe < threshold:
            raise SharpeGatingError(sharpe=oos_sharpe, threshold=threshold)
        return True
