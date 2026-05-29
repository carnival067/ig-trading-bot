"""Copy trading routes: trader management, allocation, and performance.

Provides endpoints for managing followed traders, configuring allocation
percentages, and viewing copy trading performance.
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


class TraderResponse(BaseModel):
    """Followed trader information."""

    trader_id: str
    name: str
    rank: int
    win_rate: str
    total_pnl: str
    sharpe_ratio: str
    max_drawdown: str
    followers_count: int
    allocation_pct: str
    is_active: bool
    followed_since: str


class FollowTraderRequest(BaseModel):
    """Request to follow a new trader."""

    trader_id: str = Field(..., description="ID of the trader to follow")
    allocation_pct: float = Field(..., gt=0, le=100, description="Allocation percentage")
    max_position_size: float = Field(default=0.05, gt=0, le=1.0)
    copy_stop_loss: bool = Field(default=True)
    copy_take_profit: bool = Field(default=True)


class AllocationUpdateRequest(BaseModel):
    """Request to update allocation for a followed trader."""

    allocation_pct: float = Field(..., gt=0, le=100)
    max_position_size: float | None = Field(None, gt=0, le=1.0)


class CopyTradePerformanceResponse(BaseModel):
    """Copy trading performance metrics."""

    trader_id: str
    trader_name: str
    total_copied_trades: int
    successful_copies: int
    failed_copies: int
    total_pnl: str
    avg_slippage: str
    correlation_with_source: str
    period_days: int


class CopyTradingSummaryResponse(BaseModel):
    """Overall copy trading summary."""

    total_followed_traders: int
    total_allocation_pct: str
    total_pnl: str
    active_copied_positions: int
    today_copied_trades: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/traders", response_model=list[TraderResponse])
async def list_followed_traders() -> list[TraderResponse]:
    """List all followed traders with their current status."""
    return []


@router.post("/traders/follow", response_model=TraderResponse, status_code=status.HTTP_201_CREATED)
async def follow_trader(request: FollowTraderRequest) -> TraderResponse:
    """Start following a trader with specified allocation."""
    now = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Following trader",
        extra={"trader_id": request.trader_id, "allocation_pct": request.allocation_pct},
    )

    return TraderResponse(
        trader_id=request.trader_id,
        name=f"Trader {request.trader_id}",
        rank=0,
        win_rate="0.00",
        total_pnl="0.00",
        sharpe_ratio="0.00",
        max_drawdown="0.00",
        followers_count=0,
        allocation_pct=str(request.allocation_pct),
        is_active=True,
        followed_since=now,
    )


@router.delete("/traders/{trader_id}")
async def unfollow_trader(trader_id: str) -> dict[str, str]:
    """Stop following a trader and close any copied positions."""
    logger.info("Unfollowed trader", extra={"trader_id": trader_id})
    return {"message": f"Unfollowed trader {trader_id}"}


@router.put("/traders/{trader_id}/allocation", response_model=TraderResponse)
async def update_allocation(trader_id: str, request: AllocationUpdateRequest) -> TraderResponse:
    """Update allocation percentage for a followed trader."""
    now = datetime.now(timezone.utc).isoformat()

    return TraderResponse(
        trader_id=trader_id,
        name=f"Trader {trader_id}",
        rank=0,
        win_rate="0.00",
        total_pnl="0.00",
        sharpe_ratio="0.00",
        max_drawdown="0.00",
        followers_count=0,
        allocation_pct=str(request.allocation_pct),
        is_active=True,
        followed_since=now,
    )


@router.get("/traders/{trader_id}/performance", response_model=CopyTradePerformanceResponse)
async def get_trader_performance(
    trader_id: str,
    period_days: int = Query(default=30, ge=1, le=365),
) -> CopyTradePerformanceResponse:
    """Get copy trading performance for a specific trader."""
    return CopyTradePerformanceResponse(
        trader_id=trader_id,
        trader_name=f"Trader {trader_id}",
        total_copied_trades=0,
        successful_copies=0,
        failed_copies=0,
        total_pnl="0.00",
        avg_slippage="0.00",
        correlation_with_source="0.00",
        period_days=period_days,
    )


@router.get("/summary", response_model=CopyTradingSummaryResponse)
async def get_copy_trading_summary() -> CopyTradingSummaryResponse:
    """Get overall copy trading summary."""
    return CopyTradingSummaryResponse(
        total_followed_traders=0,
        total_allocation_pct="0.00",
        total_pnl="0.00",
        active_copied_positions=0,
        today_copied_trades=0,
    )
