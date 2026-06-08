"""Unit tests for live risk-control routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.routes.risk import (
    KillSwitchRequest,
    activate_kill_switch,
    control_kill_switch,
    get_risk_status,
)


def _request(loop: SimpleNamespace | None) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(trading_loop=loop)))


@pytest.mark.asyncio
async def test_kill_switch_route_activates_live_loop() -> None:
    loop = SimpleNamespace(
        activate_kill_switch=AsyncMock(return_value=True),
        deactivate_kill_switch=AsyncMock(return_value=False),
        get_kill_switch_status=lambda: {
            "active": True,
            "reason": "manual",
            "activation_time": "2026-06-08T12:00:00+00:00",
        },
    )

    response = await control_kill_switch(
        KillSwitchRequest(activate=True, reason="manual"),
        _request(loop),
    )

    loop.activate_kill_switch.assert_awaited_once_with("manual")
    assert response.active is True
    assert response.reason == "manual"


@pytest.mark.asyncio
async def test_dashboard_activate_endpoint_uses_live_loop() -> None:
    loop = SimpleNamespace(
        activate_kill_switch=AsyncMock(return_value=True),
        deactivate_kill_switch=AsyncMock(return_value=False),
        get_kill_switch_status=lambda: {
            "active": True,
            "reason": "manual_dashboard_activation",
            "activation_time": "2026-06-08T12:00:00+00:00",
        },
    )

    response = await activate_kill_switch(_request(loop))

    loop.activate_kill_switch.assert_awaited_once()
    assert response.active is True


@pytest.mark.asyncio
async def test_risk_status_reflects_live_loop_kill_switch() -> None:
    loop = SimpleNamespace(
        get_status=lambda: {
            "open_positions": 2,
            "kill_switch": {"active": True},
        }
    )

    response = await get_risk_status(_request(loop))

    assert response.kill_switch_active is True
    assert response.position_count == 2
