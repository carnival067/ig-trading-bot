"""Unit tests for the backtesting engine.

Tests cover:
- Task 22.1: Realistic simulation with spread, slippage, and commission
- Task 22.2: Performance metric calculations
- Task 22.3: Sharpe ratio gating (OOS Sharpe < 1.0 blocks deployment)
- Task 22.4: Minimum data validation (< 30 days or < 100 trades rejected)
- Task 23.1: Walk-forward data splitting (70/30, chronological)
- Task 23.2: Walk-forward optimization loop
"""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import (
    BacktestEngine,
    InsufficientDataError,
    SharpeGatingError,
)
from src.backtesting.metrics import (
    BacktestResult,
    Trade,
    calculate_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from src.backtesting.walk_forward import OptimizationResult, WalkForwardOptimizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine(spread_pips=1.0, slippage_pips=0.5, commission_per_lot=7.0)


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Generate a sample list of trades for testing."""
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    trades = []
    for i in range(10):
        entry_time = base_time + timedelta(hours=i * 4)
        exit_time = entry_time + timedelta(hours=2)
        # Alternate wins and losses
        if i % 3 == 0:
            pnl = Decimal("-50")
        else:
            pnl = Decimal("100")
        trades.append(
            Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=Decimal("1.1000"),
                exit_price=Decimal("1.1010") if pnl > 0 else Decimal("1.0990"),
                direction="LONG",
                size=Decimal("1.0"),
                pnl=pnl,
                costs=Decimal("17"),  # spread + slippage + commission
            )
        )
    return trades


@pytest.fixture
def valid_historical_data() -> pd.DataFrame:
    """Create valid historical data with > 30 days and > 100 rows."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="4h")
    np.random.seed(42)
    close_prices = 1.1000 + np.cumsum(np.random.randn(200) * 0.001)
    return pd.DataFrame(
        {
            "open": close_prices - 0.0005,
            "high": close_prices + 0.001,
            "low": close_prices - 0.001,
            "close": close_prices,
            "volume": np.random.randint(100, 10000, 200),
        },
        index=dates,
    )


# =============================================================================
# Task 22.1: Realistic Simulation Costs
# =============================================================================


class TestRealisticSimulation:
    """Task 22.1: Backtest engine applies spread, slippage, and commission."""

    def test_engine_default_slippage(self) -> None:
        """Default slippage is 0.5 pips."""
        engine = BacktestEngine()
        assert engine.slippage_pips == 0.5

    def test_engine_custom_costs(self) -> None:
        """Custom cost parameters are stored correctly."""
        engine = BacktestEngine(spread_pips=2.0, slippage_pips=1.0, commission_per_lot=10.0)
        assert engine.spread_pips == 2.0
        assert engine.slippage_pips == 1.0
        assert engine.commission_per_lot == 10.0

    def test_trade_costs_calculation(self, engine: BacktestEngine) -> None:
        """Costs = spread_cost + slippage_cost + commission."""
        size = Decimal("1.0")
        pip_value = Decimal("10")
        costs = engine._calculate_trade_costs(size, pip_value)
        # spread: 1.0 * 10 * 1.0 = 10
        # slippage: 0.5 * 10 * 1.0 = 5
        # commission: 7.0 * 1.0 = 7
        # total: 22
        assert costs == Decimal("22.0")

    def test_trade_costs_scale_with_size(self, engine: BacktestEngine) -> None:
        """Costs scale linearly with position size."""
        pip_value = Decimal("10")
        costs_1lot = engine._calculate_trade_costs(Decimal("1.0"), pip_value)
        costs_2lots = engine._calculate_trade_costs(Decimal("2.0"), pip_value)
        assert costs_2lots == costs_1lot * 2

    def test_costs_always_reduce_pnl(self, sample_trades: list[Trade]) -> None:
        """All trades have positive costs that reduce net PnL."""
        for trade in sample_trades:
            assert trade.costs > 0


# =============================================================================
# Task 22.2: Performance Metrics
# =============================================================================


class TestPerformanceMetrics:
    """Task 22.2: Metric calculations."""

    def test_sharpe_ratio_positive_returns(self) -> None:
        """Positive varying returns yield positive Sharpe."""
        returns = [0.01, 0.02, 0.005, 0.015, 0.012] * 10
        result = sharpe_ratio(returns)
        assert result > 0

    def test_sharpe_ratio_zero_std(self) -> None:
        """Identical returns (effectively zero std) yield 0 Sharpe."""
        returns = [0.01] * 10
        # All same → std ≈ 0 (floating point) → Sharpe = 0
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_sharpe_ratio_insufficient_data(self) -> None:
        """Less than 2 returns yields 0 Sharpe."""
        assert sharpe_ratio([]) == 0.0
        assert sharpe_ratio([0.01]) == 0.0

    def test_sharpe_ratio_negative_returns(self) -> None:
        """Negative returns yield negative Sharpe."""
        returns = [-0.01, -0.02, -0.005, -0.015, -0.01]
        result = sharpe_ratio(returns)
        assert result < 0

    def test_max_drawdown_simple(self) -> None:
        """Simple drawdown calculation."""
        equity = [100, 110, 105, 95, 100, 90]
        dd = max_drawdown(equity)
        # Peak at 110, trough at 90 → (110-90)/110 = 0.1818...
        assert abs(dd - (20 / 110)) < 0.001

    def test_max_drawdown_no_drawdown(self) -> None:
        """Monotonically increasing equity has 0 drawdown."""
        equity = [100, 101, 102, 103, 104, 105]
        assert max_drawdown(equity) == 0.0

    def test_max_drawdown_empty(self) -> None:
        """Empty or single-point equity returns 0."""
        assert max_drawdown([]) == 0.0
        assert max_drawdown([100]) == 0.0

    def test_profit_factor_mixed(self, sample_trades: list[Trade]) -> None:
        """Profit factor with mixed wins and losses."""
        pf = profit_factor(sample_trades)
        # 6 wins * 100 = 600, 4 losses * 50 = 200
        # Actually: i%3==0 → losses at i=0,3,6,9 → 4 losses
        # Wins at i=1,2,4,5,7,8 → 6 wins
        # PF = 600 / 200 = 3.0
        assert abs(pf - 3.0) < 0.01

    def test_profit_factor_no_losses(self) -> None:
        """All winning trades → infinite profit factor."""
        trades = [
            Trade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 1, 1),
                entry_price=Decimal("100"),
                exit_price=Decimal("101"),
                direction="LONG",
                size=Decimal("1"),
                pnl=Decimal("10"),
            )
        ]
        assert profit_factor(trades) == float("inf")

    def test_profit_factor_no_wins(self) -> None:
        """All losing trades → 0 profit factor."""
        trades = [
            Trade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 1, 1),
                entry_price=Decimal("100"),
                exit_price=Decimal("99"),
                direction="LONG",
                size=Decimal("1"),
                pnl=Decimal("-10"),
            )
        ]
        assert profit_factor(trades) == 0.0

    def test_win_rate_calculation(self, sample_trades: list[Trade]) -> None:
        """Win rate = winners / total."""
        wr = win_rate(sample_trades)
        # 6 wins out of 10
        assert abs(wr - 0.6) < 0.01

    def test_win_rate_empty(self) -> None:
        """Empty trades → 0 win rate."""
        assert win_rate([]) == 0.0

    def test_calculate_metrics_full(self, sample_trades: list[Trade]) -> None:
        """Full metrics calculation from trades."""
        result = calculate_metrics(sample_trades, Decimal("10000"))
        assert isinstance(result, BacktestResult)
        assert result.trade_count == 10
        assert result.win_rate == 0.6
        assert result.total_return > 0  # Net positive trades
        assert len(result.equity_curve) == 11  # initial + 10 trades

    def test_calculate_metrics_empty_trades(self) -> None:
        """Empty trades produce zero metrics."""
        result = calculate_metrics([], Decimal("10000"))
        assert result.trade_count == 0
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0


# =============================================================================
# Task 22.3: Sharpe Ratio Gating
# =============================================================================


class TestSharpeGating:
    """Task 22.3: OOS Sharpe < 1.0 blocks live deployment."""

    def test_sharpe_below_threshold_raises(self) -> None:
        """Sharpe < 1.0 raises SharpeGatingError."""
        with pytest.raises(SharpeGatingError) as exc_info:
            BacktestEngine.check_sharpe_gating(0.8)
        assert exc_info.value.sharpe == 0.8
        assert exc_info.value.threshold == 1.0

    def test_sharpe_at_threshold_passes(self) -> None:
        """Sharpe == 1.0 passes gating."""
        assert BacktestEngine.check_sharpe_gating(1.0) is True

    def test_sharpe_above_threshold_passes(self) -> None:
        """Sharpe > 1.0 passes gating."""
        assert BacktestEngine.check_sharpe_gating(1.5) is True

    def test_sharpe_negative_raises(self) -> None:
        """Negative Sharpe raises SharpeGatingError."""
        with pytest.raises(SharpeGatingError):
            BacktestEngine.check_sharpe_gating(-0.5)

    def test_custom_threshold(self) -> None:
        """Custom threshold is respected."""
        # 0.8 passes with threshold 0.5
        assert BacktestEngine.check_sharpe_gating(0.8, threshold=0.5) is True
        # 0.4 fails with threshold 0.5
        with pytest.raises(SharpeGatingError):
            BacktestEngine.check_sharpe_gating(0.4, threshold=0.5)


# =============================================================================
# Task 22.4: Minimum Data Validation
# =============================================================================


class TestMinimumDataValidation:
    """Task 22.4: Reject if < 30 days or < 100 trades."""

    def test_reject_empty_data(self, engine: BacktestEngine) -> None:
        """Empty DataFrame is rejected."""
        empty_df = pd.DataFrame()
        with pytest.raises(InsufficientDataError, match="empty"):
            engine.validate_data(empty_df)

    def test_reject_insufficient_rows(self, engine: BacktestEngine) -> None:
        """Fewer than 100 rows is rejected."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
        df = pd.DataFrame(
            {"close": np.random.randn(50)},
            index=dates,
        )
        with pytest.raises(InsufficientDataError, match="rows"):
            engine.validate_data(df)

    def test_reject_insufficient_days(self, engine: BacktestEngine) -> None:
        """Data spanning < 30 days is rejected."""
        # 150 rows but only 10 days
        dates = pd.date_range(start="2024-01-01", periods=150, freq="1h")
        df = pd.DataFrame(
            {"close": np.random.randn(150)},
            index=dates,
        )
        # 150 hours = ~6.25 days < 30 days
        with pytest.raises(InsufficientDataError, match="days"):
            engine.validate_data(df)

    def test_accept_valid_data(
        self, engine: BacktestEngine, valid_historical_data: pd.DataFrame
    ) -> None:
        """Valid data (>100 rows, >30 days) passes validation."""
        # Should not raise
        engine.validate_data(valid_historical_data)

    def test_accept_exactly_100_rows_30_days(self, engine: BacktestEngine) -> None:
        """Exactly 100 rows spanning 30 days passes."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="8h")
        # 100 * 8h = 800h = 33.3 days > 30
        df = pd.DataFrame(
            {"close": np.random.randn(100)},
            index=dates,
        )
        engine.validate_data(df)


# =============================================================================
# Task 23.1: Walk-Forward Data Splitting
# =============================================================================


class TestWalkForwardSplitting:
    """Task 23.1: 70/30 chronological split."""

    def test_default_split_ratio(self) -> None:
        """Default split is 70% IS, 30% OOS."""
        optimizer = WalkForwardOptimizer()
        assert optimizer.is_ratio == 0.70
        assert optimizer.oos_ratio == 0.30

    def test_split_preserves_chronological_order(self) -> None:
        """OOS data always comes after IS data."""
        optimizer = WalkForwardOptimizer()
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        df = pd.DataFrame({"close": range(100)}, index=dates)

        is_data, oos_data = optimizer.split_data(df)

        # Last IS timestamp must be before first OOS timestamp
        assert is_data.index[-1] < oos_data.index[0]

    def test_split_no_overlap(self) -> None:
        """IS and OOS data do not overlap."""
        optimizer = WalkForwardOptimizer()
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        df = pd.DataFrame({"close": range(100)}, index=dates)

        is_data, oos_data = optimizer.split_data(df)

        # No common indices
        common = is_data.index.intersection(oos_data.index)
        assert len(common) == 0

    def test_split_sizes_approximate_ratio(self) -> None:
        """IS contains ~70% and OOS contains ~30% of data."""
        optimizer = WalkForwardOptimizer()
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
        df = pd.DataFrame({"close": range(1000)}, index=dates)

        is_data, oos_data = optimizer.split_data(df)

        assert len(is_data) == 700
        assert len(oos_data) == 300

    def test_split_covers_all_data(self) -> None:
        """IS + OOS = total data (no data lost)."""
        optimizer = WalkForwardOptimizer()
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        df = pd.DataFrame({"close": range(100)}, index=dates)

        is_data, oos_data = optimizer.split_data(df)

        assert len(is_data) + len(oos_data) == len(df)

    def test_split_empty_raises(self) -> None:
        """Empty DataFrame raises ValueError."""
        optimizer = WalkForwardOptimizer()
        with pytest.raises(ValueError, match="empty"):
            optimizer.split_data(pd.DataFrame())

    def test_invalid_ratios_raise(self) -> None:
        """Ratios not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError):
            WalkForwardOptimizer(is_ratio=0.5, oos_ratio=0.3)

    def test_custom_split_ratio(self) -> None:
        """Custom split ratios work correctly."""
        optimizer = WalkForwardOptimizer(is_ratio=0.80, oos_ratio=0.20)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        df = pd.DataFrame({"close": range(100)}, index=dates)

        is_data, oos_data = optimizer.split_data(df)

        assert len(is_data) == 80
        assert len(oos_data) == 20
