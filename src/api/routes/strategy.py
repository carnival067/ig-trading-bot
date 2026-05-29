"""Strategy routes: enable/disable, performance, configuration, and mistake patterns.

Provides endpoints for managing trading strategies, viewing their performance,
updating configuration, and monitoring mistake patterns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StrategyStatusResponse(BaseModel):
    """Strategy status information."""

    strategy_id: str
    name: str
    enabled: bool
    regime: str
    confidence_threshold: int
    active_positions: int
    daily_pnl: str
    win_rate: str
    last_signal_at: str | None = None


class StrategyToggleRequest(BaseModel):
    """Request to enable/disable a strategy."""

    enabled: bool
    reason: str = ""


class StrategyPerformanceResponse(BaseModel):
    """Strategy performance metrics."""

    strategy_id: str
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: str
    total_pnl: str
    avg_pnl_per_trade: str
    max_drawdown: str
    sharpe_ratio: str
    profit_factor: str
    period_days: int


class StrategyConfigResponse(BaseModel):
    """Strategy configuration."""

    strategy_id: str
    name: str
    parameters: dict[str, str | int | float | bool]
    risk_per_trade_pct: str
    max_positions: int
    instruments: list[str]
    regime_filter: list[str]


class StrategyConfigUpdateRequest(BaseModel):
    """Request to update strategy configuration."""

    parameters: dict[str, str | int | float | bool] | None = None
    risk_per_trade_pct: float | None = Field(None, gt=0, le=0.05)
    max_positions: int | None = Field(None, ge=1, le=50)
    instruments: list[str] | None = None


class MistakePatternResponse(BaseModel):
    """Mistake pattern information."""

    pattern_id: str
    classification: str
    occurrence_count: int
    confidence_penalty: int
    size_reduction_pct: str
    status: str  # ACTIVE, RESOLVING, RESOLVED
    first_detected_at: str
    resolution_progress: int  # consecutive profitable trades


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[StrategyStatusResponse])
async def list_strategies() -> list[StrategyStatusResponse]:
    """List all configured strategies with their current status."""
    return []


@router.get("/{strategy_id}", response_model=StrategyStatusResponse)
async def get_strategy(strategy_id: str) -> StrategyStatusResponse:
    """Get detailed status for a specific strategy."""
    now = datetime.now(timezone.utc).isoformat()

    return StrategyStatusResponse(
        strategy_id=strategy_id,
        name=strategy_id,
        enabled=True,
        regime="trending",
        confidence_threshold=60,
        active_positions=0,
        daily_pnl="0.00",
        win_rate="0.00",
        last_signal_at=now,
    )


@router.post("/{strategy_id}/toggle", response_model=StrategyStatusResponse)
async def toggle_strategy(strategy_id: str, request: StrategyToggleRequest) -> StrategyStatusResponse:
    """Enable or disable a strategy."""
    action = "enabled" if request.enabled else "disabled"
    logger.info("Strategy %s %s", strategy_id, action, extra={"reason": request.reason})

    now = datetime.now(timezone.utc).isoformat()
    return StrategyStatusResponse(
        strategy_id=strategy_id,
        name=strategy_id,
        enabled=request.enabled,
        regime="trending",
        confidence_threshold=60,
        active_positions=0,
        daily_pnl="0.00",
        win_rate="0.00",
        last_signal_at=now,
    )


@router.get("/{strategy_id}/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    strategy_id: str,
    period_days: int = Query(default=30, ge=1, le=365),
) -> StrategyPerformanceResponse:
    """Get performance metrics for a strategy over a given period."""
    return StrategyPerformanceResponse(
        strategy_id=strategy_id,
        name=strategy_id,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate="0.00",
        total_pnl="0.00",
        avg_pnl_per_trade="0.00",
        max_drawdown="0.00",
        sharpe_ratio="0.00",
        profit_factor="0.00",
        period_days=period_days,
    )


@router.get("/{strategy_id}/config", response_model=StrategyConfigResponse)
async def get_strategy_config(strategy_id: str) -> StrategyConfigResponse:
    """Get strategy configuration parameters."""
    return StrategyConfigResponse(
        strategy_id=strategy_id,
        name=strategy_id,
        parameters={},
        risk_per_trade_pct="1.00",
        max_positions=5,
        instruments=[],
        regime_filter=["trending", "ranging", "volatile"],
    )


@router.put("/{strategy_id}/config", response_model=StrategyConfigResponse)
async def update_strategy_config(
    strategy_id: str, request: StrategyConfigUpdateRequest
) -> StrategyConfigResponse:
    """Update strategy configuration parameters."""
    logger.info("Strategy config updated", extra={"strategy_id": strategy_id})

    return StrategyConfigResponse(
        strategy_id=strategy_id,
        name=strategy_id,
        parameters=request.parameters or {},
        risk_per_trade_pct=str(request.risk_per_trade_pct or 1.00),
        max_positions=request.max_positions or 5,
        instruments=request.instruments or [],
        regime_filter=["trending", "ranging", "volatile"],
    )


@router.get("/mistakes/patterns", response_model=list[MistakePatternResponse])
async def get_mistake_patterns() -> list[MistakePatternResponse]:
    """Get all detected mistake patterns and their resolution status."""
    return []
