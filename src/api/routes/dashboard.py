"""Dashboard routes: aggregated dashboard data endpoints.

Provides endpoints for the trading dashboard with aggregated views
of portfolio, performance, and system status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PortfolioSummary(BaseModel):
    """Portfolio overview for dashboard."""

    account_equity: str
    daily_pnl: str
    daily_pnl_pct: str
    weekly_pnl: str
    monthly_pnl: str
    total_pnl: str
    open_positions: int
    pending_orders: int


class SystemStatusSummary(BaseModel):
    """System health status for dashboard."""

    trading_active: bool
    kill_switch_active: bool
    news_engine_running: bool
    news_degraded_mode: bool
    hft_enabled: bool
    hft_circuit_breaker_active: bool
    active_strategies: int
    total_strategies: int
    risk_level: str


class RecentTradeItem(BaseModel):
    """Recent trade for dashboard display."""

    trade_id: str
    instrument: str
    direction: str
    size: str
    pnl: str
    closed_at: str


class AlertItem(BaseModel):
    """Alert/notification for dashboard."""

    alert_id: str
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    category: str
    timestamp: str
    acknowledged: bool


class DashboardResponse(BaseModel):
    """Complete dashboard data response."""

    portfolio: PortfolioSummary
    system_status: SystemStatusSummary
    recent_trades: list[RecentTradeItem]
    active_alerts: list[AlertItem]
    timestamp: str


class PerformanceChartData(BaseModel):
    """Performance chart data points."""

    timestamps: list[str]
    equity_curve: list[float]
    daily_pnl: list[float]
    drawdown: list[float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/", response_model=DashboardResponse)
async def get_dashboard(request: Request) -> DashboardResponse:
    """Get aggregated dashboard data.

    Returns portfolio summary, system status, recent trades, and alerts
    in a single response for efficient dashboard rendering.
    """
    now = datetime.now(timezone.utc).isoformat()
    news_engine = getattr(request.app.state, "news_engine", None)
    hft_pipeline = getattr(request.app.state, "hft_pipeline", None)

    return DashboardResponse(
        portfolio=PortfolioSummary(
            account_equity="100000.00",
            daily_pnl="0.00",
            daily_pnl_pct="0.00",
            weekly_pnl="0.00",
            monthly_pnl="0.00",
            total_pnl="0.00",
            open_positions=0,
            pending_orders=0,
        ),
        system_status=SystemStatusSummary(
            trading_active=True,
            kill_switch_active=False,
            news_engine_running=news_engine.is_running if news_engine else False,
            news_degraded_mode=news_engine.degraded_mode if news_engine else False,
            hft_enabled=hft_pipeline is not None,
            hft_circuit_breaker_active=False,
            active_strategies=0,
            total_strategies=0,
            risk_level="LOW",
        ),
        recent_trades=[],
        active_alerts=[],
        timestamp=now,
    )


@router.get("/performance", response_model=PerformanceChartData)
async def get_performance_chart(
    period_days: int = 30,
) -> PerformanceChartData:
    """Get performance chart data for the specified period."""
    return PerformanceChartData(
        timestamps=[],
        equity_curve=[],
        daily_pnl=[],
        drawdown=[],
    )
