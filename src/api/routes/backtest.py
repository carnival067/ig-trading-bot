"""Backtesting routes: execution, results, and comparison.

Provides endpoints for running backtests, retrieving results,
and comparing multiple backtest runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BacktestRequest(BaseModel):
    """Backtest execution request."""

    strategy_id: str = Field(..., description="Strategy to backtest")
    instrument: str = Field(..., description="Instrument to test on")
    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: str = Field(..., description="End date (ISO format)")
    initial_capital: float = Field(default=100000.0, gt=0)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    walk_forward: bool = Field(default=False, description="Enable walk-forward analysis")
    monte_carlo_runs: int = Field(default=0, ge=0, le=10000, description="Monte Carlo simulations")


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: str
    total_pnl: str
    max_drawdown: str
    sharpe_ratio: str
    profit_factor: str
    avg_trade_duration: str
    max_consecutive_losses: int
    recovery_factor: str


class BacktestResponse(BaseModel):
    """Backtest execution response."""

    backtest_id: str
    strategy_id: str
    instrument: str
    start_date: str
    end_date: str
    status: str  # RUNNING, COMPLETED, FAILED
    metrics: BacktestMetrics | None = None
    created_at: str
    completed_at: str | None = None


class BacktestComparisonResponse(BaseModel):
    """Comparison of multiple backtest results."""

    backtest_ids: list[str]
    metrics_comparison: list[dict[str, str]]
    best_sharpe: str
    best_profit_factor: str
    recommendation: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/run", response_model=BacktestResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Start a new backtest execution.

    Returns immediately with a backtest ID. Results can be polled via GET.
    """
    backtest_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Backtest started",
        extra={
            "backtest_id": backtest_id,
            "strategy_id": request.strategy_id,
            "instrument": request.instrument,
        },
    )

    return BacktestResponse(
        backtest_id=backtest_id,
        strategy_id=request.strategy_id,
        instrument=request.instrument,
        start_date=request.start_date,
        end_date=request.end_date,
        status="RUNNING",
        created_at=now,
    )


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_result(backtest_id: str) -> BacktestResponse:
    """Get results for a specific backtest run."""
    now = datetime.now(timezone.utc).isoformat()

    return BacktestResponse(
        backtest_id=backtest_id,
        strategy_id="unknown",
        instrument="unknown",
        start_date="",
        end_date="",
        status="COMPLETED",
        metrics=BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate="0.00",
            total_pnl="0.00",
            max_drawdown="0.00",
            sharpe_ratio="0.00",
            profit_factor="0.00",
            avg_trade_duration="0:00:00",
            max_consecutive_losses=0,
            recovery_factor="0.00",
        ),
        created_at=now,
        completed_at=now,
    )


@router.get("/", response_model=list[BacktestResponse])
async def list_backtests(
    strategy_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[BacktestResponse]:
    """List recent backtest runs with optional strategy filter."""
    return []


@router.post("/compare", response_model=BacktestComparisonResponse)
async def compare_backtests(backtest_ids: list[str]) -> BacktestComparisonResponse:
    """Compare metrics across multiple backtest runs."""
    if len(backtest_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 backtest IDs required for comparison",
        )

    return BacktestComparisonResponse(
        backtest_ids=backtest_ids,
        metrics_comparison=[],
        best_sharpe="N/A",
        best_profit_factor="N/A",
        recommendation="Insufficient data for recommendation",
    )
