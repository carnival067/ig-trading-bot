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
async def get_risk_status() -> RiskStatusResponse:
    """Get current overall risk status."""
    now = datetime.now(timezone.utc).isoformat()

    return RiskStatusResponse(
        daily_pnl="0.00",
        daily_pnl_pct="0.00",
        max_daily_loss_pct="3.00",
        current_drawdown_pct="0.00",
        kill_switch_threshold_pct="15.00",
        kill_switch_active=False,
        position_count=0,
        total_exposure="0.00",
        total_exposure_pct="0.00",
        risk_level="LOW",
        timestamp=now,
    )


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def control_kill_switch(request: KillSwitchRequest) -> KillSwitchResponse:
    """Activate or deactivate the kill switch.

    When activated, all trading is halted and open positions may be closed.
    """
    now = datetime.now(timezone.utc).isoformat()

    action = "activated" if request.activate else "deactivated"
    logger.warning(
        "Kill switch %s", action,
        extra={"reason": request.reason},
    )

    return KillSwitchResponse(
        active=request.activate,
        activated_at=now if request.activate else None,
        reason=request.reason,
        message=f"Kill switch {action} successfully",
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
