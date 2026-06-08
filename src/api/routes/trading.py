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


def _require_debug_trading_enabled() -> None:
    from src.config.settings import get_settings

    if not get_settings().enable_debug_trading_endpoints:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Debug trading endpoint is disabled",
        )


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
async def execute_trade(payload: TradeRequest, request: Request) -> TradeResponse:
    """Execute a new trade order.

    Executes through the connected IG client. This endpoint refuses to return
    synthetic fills when the live trading client is unavailable.
    """
    if payload.order_type != OrderType.MARKET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Only MARKET orders are currently supported by the live execution path.",
        )

    trading_loop = getattr(request.app.state, "trading_loop", None)
    ig_client = getattr(trading_loop, "_ig_client", None) if trading_loop is not None else None
    if ig_client is None or not getattr(ig_client, "is_connected", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IG trading client is not connected; order was not submitted.",
        )

    now = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Manual trade requested",
        extra={
            "instrument": payload.instrument,
            "direction": payload.direction.value,
            "size": str(payload.size),
        },
    )

    try:
        result = await ig_client.place_order(
            epic=payload.instrument,
            direction=payload.direction.value,
            size=float(payload.size),
            stop_distance=None,
            limit_distance=None,
        )
    except Exception as exc:
        logger.exception("Manual trade submission failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"IG order submission failed: {str(exc)[:200]}",
        ) from exc

    deal_status = result.get("dealStatus", "unknown")
    if deal_status != "ACCEPTED":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "IG rejected the order.",
                "deal_status": deal_status,
                "reason": result.get("reason"),
            },
        )

    deal_id = result.get("dealId") or str(uuid4())
    fill_price = result.get("level")

    if payload.stop_loss is not None or payload.take_profit is not None:
        try:
            await ig_client.update_position_sl_tp(
                deal_id=deal_id,
                stop_level=float(payload.stop_loss) if payload.stop_loss is not None else None,
                limit_level=float(payload.take_profit) if payload.take_profit is not None else None,
            )
        except Exception as exc:
            logger.exception("Manual trade SL/TP update failed")
            close_direction = (
                OrderDirection.SELL
                if payload.direction == OrderDirection.BUY
                else OrderDirection.BUY
            )
            try:
                await ig_client.close_position(
                    deal_id,
                    close_direction.value,
                    float(payload.size),
                )
            except Exception as close_exc:
                logger.exception("Manual trade emergency close failed after SL/TP failure")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        "Order was accepted by IG, stop-loss/take-profit update failed, "
                        f"and emergency close failed: {str(close_exc)[:200]}"
                    ),
                ) from close_exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Order was accepted by IG, stop-loss/take-profit update failed, "
                    "and the position was closed: "
                    f"{str(exc)[:200]}"
                ),
            ) from exc

    return TradeResponse(
        trade_id=deal_id,
        instrument=payload.instrument,
        direction=payload.direction,
        size=str(payload.size),
        order_type=payload.order_type,
        status=OrderStatus.FILLED,
        fill_price=str(fill_price) if fill_price is not None else None,
        timestamp=now,
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions() -> list[PositionResponse]:
    """Get all open positions."""
    from src.db.database import get_session
    from src.db.repositories.trade_repo import TradeRepository

    try:
        async with get_session() as session:
            repo = TradeRepository(session)
            positions = await repo.get_open_positions()
    except Exception as exc:
        logger.warning("Unable to fetch positions from database: %s", exc)
        return []

    return [
        PositionResponse(
            position_id=str(position.id),
            instrument=position.instrument,
            direction=_api_direction(position.direction),
            size=str(position.size),
            entry_price=str(position.entry_price),
            current_price=str(position.entry_price),
            unrealized_pnl="0",
            stop_loss=str(position.stop_loss) if position.stop_loss is not None else None,
            take_profit=str(position.take_profit) if position.take_profit is not None else None,
            opened_at=position.created_at.isoformat() if position.created_at else "",
        )
        for position in positions
    ]


@router.post("/positions/{position_id}/close", response_model=TradeResponse)
async def close_position(
    position_id: str,
    payload: ClosePositionRequest,
    request: Request,
) -> TradeResponse:
    """Close an open position through IG and update persisted trade state."""
    from src.db.database import get_session
    from src.db.repositories.trade_repo import TradeRepository

    trading_loop = getattr(request.app.state, "trading_loop", None) if hasattr(request, "app") else None
    ig_client = getattr(trading_loop, "_ig_client", None) if trading_loop is not None else None
    if ig_client is None or not getattr(ig_client, "is_connected", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IG trading client is not connected; position was not closed.",
        )

    now = datetime.now(timezone.utc).isoformat()

    try:
        from sqlalchemy import select

        from src.db.models import TradeContext

        async with get_session() as session:
            repo = TradeRepository(session)
            position = await repo.get_position(position_id)
            if position is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Position not found.",
                )

            trade = await repo.get_trade(position.trade_id)
            if not position.ig_deal_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Position has no IG deal ID; cannot close safely through broker.",
                )

            close_size = position.size
            if payload.size is not None:
                requested_size = Decimal(str(payload.size))
                if requested_size != position.size:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Partial position closes are not yet supported by the persistence path.",
                    )
                close_size = requested_size

            close_direction = (
                OrderDirection.SELL
                if _api_direction(position.direction) == OrderDirection.BUY
                else OrderDirection.BUY
            )

            result = await ig_client.close_position(
                position.ig_deal_id,
                close_direction.value,
                float(close_size),
            )

            deal_status = result.get("dealStatus", "unknown")
            if deal_status != "ACCEPTED":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "message": "IG rejected the close request.",
                        "deal_status": deal_status,
                        "reason": result.get("reason"),
                    },
                )

            pnl = Decimal(str(result.get("profit") or "0"))
            exit_price_raw = result.get("level") or result.get("closeLevel") or position.entry_price
            exit_price = Decimal(str(exit_price_raw))
            if trade is not None:
                await repo.close_trade(trade.id, exit_price=exit_price, pnl=pnl)
                context_result = await session.execute(
                    select(TradeContext).where(TradeContext.trade_id == trade.id)
                )
                trade_context = context_result.scalar_one_or_none()
            else:
                trade_context = None
            await repo.close_position(position.id)

            instrument = position.instrument
            response_size = str(close_size)
            response_trade_id = str(trade.id if trade is not None else position.id)
            fill_price = str(exit_price)

        if trade is not None and hasattr(trading_loop, "_record_learning_outcome"):
            await trading_loop._record_learning_outcome(
                trade=trade,
                context=trade_context,
                pnl=pnl,
                exit_price=exit_price,
                close_result=result,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Position close failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Position close failed: {str(exc)[:200]}",
        ) from exc

    return TradeResponse(
        trade_id=response_trade_id,
        instrument=instrument,
        direction=close_direction,
        size=response_size,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        fill_price=fill_price,
        timestamp=now,
    )


@router.get("/orders", response_model=list[OrderHistoryItem])
async def get_order_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    instrument: str | None = Query(default=None),
) -> list[OrderHistoryItem]:
    """Get order history with pagination and optional instrument filter."""
    from src.db.database import get_session
    from src.db.repositories.trade_repo import TradeRepository

    try:
        async with get_session() as session:
            repo = TradeRepository(session)
            if instrument:
                trades = await repo.get_trades_by_instrument(instrument, limit=limit + offset)
            else:
                trades = await repo.get_recent_trades(limit=limit + offset)
    except Exception as exc:
        logger.warning("Unable to fetch order history from database: %s", exc)
        return []

    return [
        OrderHistoryItem(
            order_id=str(trade.id),
            instrument=trade.instrument,
            direction=_api_direction(trade.direction),
            size=str(trade.size),
            order_type=OrderType.MARKET,
            status=_api_order_status(trade.status),
            fill_price=str(trade.entry_price),
            pnl=str(trade.pnl) if trade.pnl is not None else None,
            created_at=trade.opened_at.isoformat(),
            filled_at=trade.opened_at.isoformat(),
        )
        for trade in trades[offset : offset + limit]
    ]


def _api_direction(direction: Any) -> OrderDirection:
    value = getattr(direction, "value", str(direction))
    return OrderDirection.BUY if value in ("LONG", "BUY") else OrderDirection.SELL


def _api_order_status(trade_status: Any) -> OrderStatus:
    value = getattr(trade_status, "value", str(trade_status))
    if value == "OPEN":
        return OrderStatus.FILLED
    if value == "CLOSED":
        return OrderStatus.FILLED
    if value == "CANCELLED":
        return OrderStatus.CANCELLED
    return OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Autonomous Trading Loop Control
# ---------------------------------------------------------------------------


@router.get("/loop/status")
async def get_trading_loop_status(request: Request) -> dict[str, Any]:
    """Get the current status of the autonomous trading loop."""
    # Try app.state first, fall back to global singleton
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        from src.trading.trading_loop import _trading_loop
        trading_loop = _trading_loop
    if trading_loop is None:
        return {
            "running": False,
            "connected": False,
            "error": "Trading loop not initialized — check Render logs for startup errors",
            "services_ready": getattr(request.app.state, "services_ready", False),
        }
    return trading_loop.get_status()


@router.post("/loop/start")
async def start_trading_loop(request: Request) -> dict[str, str]:
    """Start the autonomous trading loop."""
    from src.trading.trading_loop import AutonomousTradingLoop

    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        trading_loop = AutonomousTradingLoop(
            mistake_analyzer=getattr(request.app.state, "mistake_analyzer", None)
        )
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


@router.get("/debug/stream-status")
async def debug_stream_status(request: Request) -> dict[str, Any]:
    """Show streaming connection status and last error."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        return {"error": "Trading loop not available"}

    ig_stream = trading_loop._ig_stream
    event_bus = trading_loop._event_bus

    return {
        "streaming": ig_stream is not None and getattr(ig_stream, "is_connected", False),
        "stream_object_exists": ig_stream is not None,
        "event_bus_running": event_bus is not None and getattr(event_bus, "is_running", False),
        "session_id": getattr(ig_stream, "_session_id", None) if ig_stream else None,
        "subscriptions": getattr(ig_stream, "get_subscription_status", lambda: {})() if ig_stream else {},
        "candle_buffer": trading_loop._candle_buffer.get_status(),
        "ig_client_connected": trading_loop._ig_client.is_connected if trading_loop._ig_client else False,
        "ig_client_cst_present": bool(getattr(trading_loop._ig_client, "_cst", None)) if trading_loop._ig_client else False,
    }


@router.post("/debug/close-all")
async def debug_close_all_positions(request: Request) -> dict[str, Any]:
    """Close all open positions on the demo account."""
    _require_debug_trading_enabled()
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or trading_loop._ig_client is None:
        return {"error": "Trading loop or IG client not available"}

    import traceback
    try:
        positions = await trading_loop._ig_client.get_positions()
        results = []
        for pos in positions:
            pos_data = pos.get("position", pos)
            deal_id = pos_data.get("dealId") or pos_data.get("deal_id")
            direction = pos_data.get("direction", "BUY")
            size = float(pos_data.get("size") or pos_data.get("dealSize") or 1.0)
            close_dir = "SELL" if direction == "BUY" else "BUY"
            result = await trading_loop._ig_client.close_position(deal_id, close_dir, size)
            results.append({"deal_id": deal_id, "status": result.get("dealStatus"), "reason": result.get("reason")})
        return {"closed": len(results), "results": results}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


@router.post("/debug/test-order")
async def debug_test_order(request: Request) -> dict[str, Any]:
    """Place a minimal test order for EURUSD and return the full IG response including rejection reason."""
    _require_debug_trading_enabled()
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or trading_loop._ig_client is None:
        return {"error": "Trading loop or IG client not available"}

    import traceback
    try:
        result = await trading_loop._ig_client.place_order(
            epic="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=1.0,
            stop_distance=0.0010,    # 10 points after ×10000
            limit_distance=0.0020,   # 20 points after ×10000
        )
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


@router.post("/debug/restart-stream")
async def debug_restart_stream(request: Request) -> dict[str, Any]:
    """Return an explicit message for the retired streaming debug action."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        return {"error": "Trading loop not available"}

    return {
        "success": False,
        "streaming": "snapshot_polling",
        "message": "Streaming restart is not supported because this bot currently polls IG snapshots.",
        "snapshot_interval_seconds": trading_loop.get_status().get("snapshot_interval_seconds"),
        "candle_buffer": trading_loop._candle_buffer.get_status(),
    }


@router.get("/debug/market-details")
async def debug_market_details(request: Request, epic: str) -> dict[str, Any]:
    """Fetch market details for an epic: scaling, min deal size, and currency."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or trading_loop._ig_client is None:
        return {"error": "Trading loop or IG client not available"}

    try:
        details = await trading_loop._ig_client.get_market_details(epic)
        instrument = details.get("instrument", {})
        dealing = details.get("dealingRules", {})
        return {
            "epic": epic,
            "currency": instrument.get("currencies", [{}])[0].get("code") if instrument.get("currencies") else None,
            "scaling_factor": instrument.get("scalingFactor"),
            "min_deal_size": dealing.get("minDealSize", {}).get("value"),
            "max_deal_size": dealing.get("maxDealSize", {}).get("value"),
            "min_stop_distance": dealing.get("minNormalStopOrLimitDistance", {}).get("value"),
            "market_status": details.get("snapshot", {}).get("marketStatus"),
            "instrument_type": instrument.get("type"),
            "raw_instrument_keys": list(instrument.keys()),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/debug/account-currency")
async def debug_account_currency(request: Request) -> dict[str, Any]:
    """Fetch the IG account currency to verify it matches what's sent in orders."""
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None or trading_loop._ig_client is None:
        return {"error": "Trading loop or IG client not available"}

    try:
        currency = await trading_loop._ig_client.get_account_currency()
        account_info = await trading_loop._ig_client.get_account_info()
        return {"currency": currency, "raw_account": account_info}
    except Exception as exc:
        return {"error": str(exc)}
