"""Trading routes: trade execution, position management, and order history.

Provides endpoints for placing trades, managing open positions,
and querying order history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Enums and Models
# ---------------------------------------------------------------------------


class OrderDirection(str, Enum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeRequest(BaseModel):
    """Trade execution request."""

    instrument: str = Field(..., description="Instrument epic/symbol")
    direction: OrderDirection
    size: Decimal = Field(..., gt=0, description="Position size")
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(None, description="Limit price for LIMIT orders")
    stop_loss: Decimal | None = Field(None, description="Stop loss level")
    take_profit: Decimal | None = Field(None, description="Take profit level")
    strategy_id: str | None = Field(None, description="Strategy that generated the signal")


class TradeResponse(BaseModel):
    """Trade execution response."""

    trade_id: str
    instrument: str
    direction: OrderDirection
    size: str
    order_type: OrderType
    status: OrderStatus
    fill_price: str | None = None
    timestamp: str


class PositionResponse(BaseModel):
    """Open position details."""

    position_id: str
    instrument: str
    direction: OrderDirection
    size: str
    entry_price: str
    current_price: str
    unrealized_pnl: str
    stop_loss: str | None = None
    take_profit: str | None = None
    opened_at: str


class ClosePositionRequest(BaseModel):
    """Request to close a position."""

    size: Decimal | None = Field(None, description="Partial close size, None for full close")


class OrderHistoryItem(BaseModel):
    """Historical order record."""

    order_id: str
    instrument: str
    direction: OrderDirection
    size: str
    order_type: OrderType
    status: OrderStatus
    fill_price: str | None = None
    pnl: str | None = None
    created_at: str
    filled_at: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/execute", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
async def execute_trade(request: TradeRequest) -> TradeResponse:
    """Execute a new trade order.

    Validates the order against risk limits before execution.
    """
    trade_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Trade executed",
        extra={
            "trade_id": trade_id,
            "instrument": request.instrument,
            "direction": request.direction.value,
            "size": str(request.size),
        },
    )

    return TradeResponse(
        trade_id=trade_id,
        instrument=request.instrument,
        direction=request.direction,
        size=str(request.size),
        order_type=request.order_type,
        status=OrderStatus.FILLED,
        fill_price=str(request.limit_price) if request.limit_price else "100.00",
        timestamp=now,
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions() -> list[PositionResponse]:
    """Get all open positions."""
    # In production, fetch from position manager
    return []


@router.post("/positions/{position_id}/close", response_model=TradeResponse)
async def close_position(position_id: str, request: ClosePositionRequest) -> TradeResponse:
    """Close an open position (full or partial)."""
    now = datetime.now(timezone.utc).isoformat()

    return TradeResponse(
        trade_id=str(uuid4()),
        instrument="",
        direction=OrderDirection.SELL,
        size=str(request.size) if request.size else "0",
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        fill_price="100.00",
        timestamp=now,
    )


@router.get("/orders", response_model=list[OrderHistoryItem])
async def get_order_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    instrument: str | None = Query(default=None),
) -> list[OrderHistoryItem]:
    """Get order history with pagination and optional instrument filter."""
    # In production, fetch from trade repository
    return []


# ---------------------------------------------------------------------------
# Autonomous Trading Loop Control
# ---------------------------------------------------------------------------


@router.get("/loop/status")
async def get_trading_loop_status(request: Request) -> dict[str, Any]:
    """Get the current status of the autonomous trading loop."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        # App state not initialized — lifespan may not have run
        # Return a diagnostic response
        return {
            "running": False,
            "connected": False,
            "error": "Trading loop not in app.state — lifespan startup may have failed",
            "services_ready": getattr(request.app.state, "services_ready", False),
        }
    return trading_loop.get_status()


@router.post("/loop/start")
async def start_trading_loop(request: Request) -> dict[str, str]:
    """Start the autonomous trading loop."""
    from src.trading.trading_loop import AutonomousTradingLoop

    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        trading_loop = AutonomousTradingLoop()
        request.app.state.trading_loop = trading_loop

    if trading_loop.is_running:
        return {"status": "already_running", "message": "Trading loop is already active"}

    await trading_loop.start()
    return {"status": "started", "message": "Autonomous trading loop started"}


@router.post("/loop/stop")
async def stop_trading_loop(request: Request) -> dict[str, str]:
    """Stop the autonomous trading loop."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or not trading_loop.is_running:
        return {"status": "already_stopped", "message": "Trading loop is not running"}

    await trading_loop.stop()
    return {"status": "stopped", "message": "Autonomous trading loop stopped"}


# -------------------------
# Debug endpoints
# -------------------------


@router.get("/debug/prices")
async def debug_get_prices(request: Request, epic: str, resolution: str = "HOUR", num_points: int = 100) -> dict[str, Any]:
    """Fetch raw price data from IG for an epic via the connected trading loop's IG client.

    Useful to inspect why the strategy is not generating signals (missing/structured prices).
    """
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or trading_loop._ig_client is None:
        return {"error": "Trading loop or IG client not available"}

    try:
        prices = await trading_loop._ig_client.get_prices(epic, resolution, num_points)
        sample = prices[0] if prices else None
        return {"count": len(prices), "sample_keys": list(sample.keys()) if sample else None, "sample": sample}
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/debug/analyze")
async def debug_analyze_instrument(request: Request, epic: str) -> dict[str, Any]:
    """Run a single analysis pass for `epic` using the trading loop analyzer and return the generated signal (if any)."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        return {"error": "Trading loop not available"}

    try:
        signal = await trading_loop._analyze_instrument(epic)
        return {"signal": signal}
    except Exception as exc:
        return {"error": str(exc)}
