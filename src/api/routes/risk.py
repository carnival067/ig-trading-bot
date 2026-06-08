"""Risk management routes: risk status, kill switch, exposure, and HFT risk.

Provides endpoints for monitoring and controlling risk parameters,
kill switch activation/deactivation, and HFT-specific risk status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class RiskStatusResponse(BaseModel):
    """Overall risk status."""

    daily_pnl: str
    daily_pnl_pct: str
    max_daily_loss_pct: str
    current_drawdown_pct: str
    kill_switch_threshold_pct: str
    kill_switch_active: bool
    position_count: int
    total_exposure: str
    total_exposure_pct: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str


class KillSwitchRequest(BaseModel):
    """Kill switch control request."""

    activate: bool = Field(..., description="True to activate, False to deactivate")
    reason: str = Field(default="", description="Reason for activation/deactivation")


class KillSwitchResponse(BaseModel):
    """Kill switch status response."""

    active: bool
    activated_at: str | None = None
    reason: str
    message: str


class ExposureResponse(BaseModel):
    """Portfolio exposure breakdown."""

    total_exposure: str
    total_exposure_pct: str
    by_instrument: dict[str, str]
    by_direction: dict[str, str]
    by_strategy: dict[str, str]
    max_allowed_pct: str


class HFTRiskStatusResponse(BaseModel):
    """HFT-specific risk status."""

    enabled: bool
    circuit_breaker_active: bool
    circuit_breaker_activations: int
    global_order_rate: int
    max_order_rate: int
    current_exposure_pct: str
    max_exposure_pct: str
    rolling_pnl: str
    throttle_active: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status", response_model=RiskStatusResponse)
async def get_risk_status(request: Request) -> RiskStatusResponse:
    """Get current overall risk status."""
    now = datetime.now(timezone.utc).isoformat()
    trading_loop = getattr(request.app.state, "trading_loop", None)
    loop_status = trading_loop.get_status() if trading_loop is not None else {}
    kill_switch = loop_status.get("kill_switch", {})

    return RiskStatusResponse(
        daily_pnl="0.00",
        daily_pnl_pct="0.00",
        max_daily_loss_pct="3.00",
        current_drawdown_pct="0.00",
        kill_switch_threshold_pct="15.00",
        kill_switch_active=bool(kill_switch.get("active", False)),
        position_count=int(loop_status.get("open_positions", 0)),
        total_exposure="0.00",
        total_exposure_pct="0.00",
        risk_level="LOW",
        timestamp=now,
    )


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def control_kill_switch(payload: KillSwitchRequest, request: Request) -> KillSwitchResponse:
    """Activate or deactivate the kill switch.

    When activated, all trading is halted and open positions may be closed.
    """
    now = datetime.now(timezone.utc).isoformat()
    trading_loop = getattr(request.app.state, "trading_loop", None)
    if trading_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading loop is not available",
        )

    action = "activated" if payload.activate else "deactivated"
    logger.warning(
        "Kill switch %s", action,
        extra={"reason": payload.reason},
    )
    if payload.activate:
        changed = await trading_loop.activate_kill_switch(
            payload.reason or "manual_dashboard_activation"
        )
    else:
        changed = await trading_loop.deactivate_kill_switch(
            payload.reason or "manual_dashboard_deactivation"
        )

    kill_status = trading_loop.get_kill_switch_status()

    return KillSwitchResponse(
        active=bool(kill_status.get("active", payload.activate)),
        activated_at=kill_status.get("activation_time") or (now if payload.activate else None),
        reason=str(kill_status.get("reason") or payload.reason),
        message=(
            f"Kill switch {action} successfully"
            if changed
            else f"Kill switch {action} request received but state did not change"
        ),
    )


@router.post("/kill-switch/activate", response_model=KillSwitchResponse)
async def activate_kill_switch(request: Request) -> KillSwitchResponse:
    """Dashboard-compatible endpoint to activate the live kill switch."""
    return await control_kill_switch(
        KillSwitchRequest(activate=True, reason="manual_dashboard_activation"),
        request,
    )


@router.post("/kill-switch/deactivate", response_model=KillSwitchResponse)
async def deactivate_kill_switch(request: Request) -> KillSwitchResponse:
    """Dashboard-compatible endpoint to deactivate the live kill switch."""
    return await control_kill_switch(
        KillSwitchRequest(activate=False, reason="manual_dashboard_deactivation"),
        request,
    )


@router.get("/exposure", response_model=ExposureResponse)
async def get_exposure() -> ExposureResponse:
    """Get current portfolio exposure breakdown."""
    return ExposureResponse(
        total_exposure="0.00",
        total_exposure_pct="0.00",
        by_instrument={},
        by_direction={"long": "0.00", "short": "0.00"},
        by_strategy={},
        max_allowed_pct="5.00",
    )


@router.get("/hft", response_model=HFTRiskStatusResponse)
async def get_hft_risk_status(request: Request) -> HFTRiskStatusResponse:
    """Get HFT-specific risk status including circuit breaker state."""
    now = datetime.now(timezone.utc).isoformat()
    hft_pipeline = getattr(request.app.state, "hft_pipeline", None)

    if hft_pipeline is None:
        return HFTRiskStatusResponse(
            enabled=False,
            circuit_breaker_active=False,
            circuit_breaker_activations=0,
            global_order_rate=0,
            max_order_rate=100,
            current_exposure_pct="0.00",
            max_exposure_pct="15.00",
            rolling_pnl="0.00",
            throttle_active=False,
            timestamp=now,
        )

    return HFTRiskStatusResponse(
        enabled=True,
        circuit_breaker_active=False,
        circuit_breaker_activations=0,
        global_order_rate=0,
        max_order_rate=100,
        current_exposure_pct="0.00",
        max_exposure_pct="15.00",
        rolling_pnl="0.00",
        throttle_active=False,
        timestamp=now,
    )
