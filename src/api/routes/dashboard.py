"""Dashboard routes: aggregated dashboard data endpoints.

Provides endpoints for the trading dashboard with aggregated views
of portfolio, performance, and system status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import func, select

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


class PnLMetrics(BaseModel):
    daily: float
    weekly: float
    monthly: float
    all_time: float


class DashboardPosition(BaseModel):
    id: str
    instrument: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    strategy: str
    opened_at: str


class DashboardMetrics(BaseModel):
    pnl: PnLMetrics
    win_rate: float
    drawdown: float
    open_positions: list[DashboardPosition]
    ai_confidence: float
    market_regime: str


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


@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(request: Request) -> DashboardMetrics:
    """Get frontend dashboard metrics backed by persisted trades and positions."""
    _ = request
    try:
        return await _load_dashboard_metrics()
    except Exception as exc:
        logger.warning("Unable to load dashboard metrics from database: %s", exc)
        return _empty_dashboard_metrics()


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


async def _load_dashboard_metrics() -> DashboardMetrics:
    from src.db.database import get_session
    from src.db.models import PositionStatus, Trade
    from src.db.repositories.trade_repo import TradeRepository

    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    async with get_session() as session:
        repo = TradeRepository(session)
        open_positions = await repo.get_open_positions()

        daily_pnl = await repo.get_daily_pnl(now.date())
        weekly_pnl = await _sum_pnl_since(session, week_start)
        monthly_pnl = await _sum_pnl_since(session, month_start)
        all_time_pnl = await _sum_pnl_since(session, datetime.min.replace(tzinfo=timezone.utc))

        closed_trades = await repo.get_closed_trades_since(datetime.min.replace(tzinfo=timezone.utc))
        wins = sum(1 for trade in closed_trades if trade.pnl is not None and trade.pnl > Decimal("0"))
        win_rate = wins / len(closed_trades) if closed_trades else 0.0

        confidence_result = await session.execute(
            select(func.avg(Trade.confidence_score))
        )
        avg_confidence = confidence_result.scalar_one_or_none()

    return DashboardMetrics(
        pnl=PnLMetrics(
            daily=float(daily_pnl),
            weekly=float(weekly_pnl),
            monthly=float(monthly_pnl),
            all_time=float(all_time_pnl),
        ),
        win_rate=win_rate,
        drawdown=0.0,
        open_positions=[
            _dashboard_position_from_model(position)
            for position in open_positions
            if position.status == PositionStatus.OPEN
        ],
        ai_confidence=(float(avg_confidence) / 100.0) if avg_confidence is not None else 0.0,
        market_regime="unknown",
    )


async def _sum_pnl_since(session: Any, since: datetime) -> Decimal:
    from src.db.models import Trade, TradeStatus

    result = await session.execute(
        select(func.coalesce(func.sum(Trade.pnl), Decimal("0"))).where(
            Trade.status == TradeStatus.CLOSED,
            Trade.closed_at >= since,
            Trade.pnl.is_not(None),
        )
    )
    value = result.scalar_one()
    return Decimal(str(value)) if value is not None else Decimal("0")


def _dashboard_position_from_model(position: Any) -> DashboardPosition:
    direction = getattr(position.direction, "value", str(position.direction))
    return DashboardPosition(
        id=str(position.id),
        instrument=position.instrument,
        direction="long" if direction == "LONG" else "short",
        size=float(position.size),
        entry_price=float(position.entry_price),
        current_price=float(position.entry_price),
        unrealized_pnl=0.0,
        strategy="autonomous_sma_atr",
        opened_at=position.created_at.isoformat() if position.created_at else "",
    )


def _empty_dashboard_metrics() -> DashboardMetrics:
    return DashboardMetrics(
        pnl=PnLMetrics(daily=0.0, weekly=0.0, monthly=0.0, all_time=0.0),
        win_rate=0.0,
        drawdown=0.0,
        open_positions=[],
        ai_confidence=0.0,
        market_regime="unknown",
    )
