"""Unit tests for trading API route behavior."""

from __future__ import annotations

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes.trading import (
    ClosePositionRequest,
    OrderDirection,
    OrderType,
    TradeRequest,
    close_position,
    debug_close_all_positions,
    debug_test_order,
    execute_trade,
)
from src.db.models import TradeDirection


class _SessionContext:
    async def __aenter__(self) -> object:
        return _FakeSession()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _ScalarResult:
    def scalar_one_or_none(self) -> None:
        return None


class _FakeSession:
    async def execute(self, statement) -> _ScalarResult:
        return _ScalarResult()


class _FakeTradeRepository:
    def __init__(self, position: SimpleNamespace | None, trade: SimpleNamespace | None) -> None:
        self.position = position
        self.trade = trade
        self.closed_trade: tuple[uuid.UUID, Decimal, Decimal] | None = None
        self.closed_position: uuid.UUID | None = None

    async def get_position(self, position_id: str) -> SimpleNamespace | None:
        if self.position is None or str(self.position.id) != position_id:
            return None
        return self.position

    async def get_trade(self, trade_id: uuid.UUID) -> SimpleNamespace | None:
        if self.trade is None or self.trade.id != trade_id:
            return None
        return self.trade

    async def close_trade(
        self,
        trade_id: uuid.UUID,
        exit_price: Decimal,
        pnl: Decimal,
    ) -> SimpleNamespace | None:
        self.closed_trade = (trade_id, exit_price, pnl)
        return self.trade

    async def close_position(self, position_id: uuid.UUID) -> SimpleNamespace | None:
        self.closed_position = position_id
        return self.position


def _app_request(ig_client: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                trading_loop=SimpleNamespace(_ig_client=ig_client),
            ),
        ),
    )


def _connected_ig_client(**overrides) -> SimpleNamespace:
    values = {
        "is_connected": True,
        "place_order": AsyncMock(
            return_value={"dealStatus": "ACCEPTED", "dealId": "D1", "level": "1.1000"}
        ),
        "update_position_sl_tp": AsyncMock(return_value={"dealStatus": "ACCEPTED"}),
        "close_position": AsyncMock(return_value={"dealStatus": "ACCEPTED"}),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _position_and_trade(direction: TradeDirection = TradeDirection.LONG) -> tuple[SimpleNamespace, SimpleNamespace]:
    trade = SimpleNamespace(id=uuid.uuid4())
    position = SimpleNamespace(
        id=uuid.uuid4(),
        trade_id=trade.id,
        instrument="CS.D.EURUSD.CFD.IP",
        direction=direction,
        size=Decimal("1.0"),
        entry_price=Decimal("1.1000"),
        ig_deal_id="DIAAAABBBCCC123",
    )
    return position, trade


def _patch_repo(monkeypatch: pytest.MonkeyPatch, repo: _FakeTradeRepository) -> None:
    monkeypatch.setattr("src.db.database.get_session", lambda: _SessionContext())
    monkeypatch.setattr("src.db.repositories.trade_repo.TradeRepository", lambda session: repo)


@pytest.mark.asyncio
async def test_close_position_closes_broker_position_and_persists_state(monkeypatch: pytest.MonkeyPatch) -> None:
    position, trade = _position_and_trade()
    repo = _FakeTradeRepository(position, trade)
    _patch_repo(monkeypatch, repo)

    ig_client = SimpleNamespace(
        is_connected=True,
        close_position=AsyncMock(
            return_value={
                "dealStatus": "ACCEPTED",
                "profit": "12.50",
                "level": "1.1010",
            }
        ),
    )

    response = await close_position(
        str(position.id),
        ClosePositionRequest(),
        _app_request(ig_client),
    )

    ig_client.close_position.assert_awaited_once_with("DIAAAABBBCCC123", "SELL", 1.0)
    assert repo.closed_trade == (trade.id, Decimal("1.1010"), Decimal("12.50"))
    assert repo.closed_position == position.id
    assert response.trade_id == str(trade.id)
    assert response.instrument == position.instrument
    assert response.direction == OrderDirection.SELL
    assert response.size == "1.0"
    assert response.fill_price == "1.1010"


@pytest.mark.asyncio
async def test_close_position_requires_connected_ig_client() -> None:
    ig_client = SimpleNamespace(is_connected=False, close_position=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await close_position(
            str(uuid.uuid4()),
            ClosePositionRequest(),
            _app_request(ig_client),
        )

    assert exc_info.value.status_code == 503
    ig_client.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_close_position_rejects_partial_close_before_broker_call(monkeypatch: pytest.MonkeyPatch) -> None:
    position, trade = _position_and_trade()
    repo = _FakeTradeRepository(position, trade)
    _patch_repo(monkeypatch, repo)
    ig_client = SimpleNamespace(is_connected=True, close_position=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await close_position(
            str(position.id),
            ClosePositionRequest(size=Decimal("0.5")),
            _app_request(ig_client),
        )

    assert exc_info.value.status_code == 501
    ig_client.close_position.assert_not_called()
    assert repo.closed_trade is None
    assert repo.closed_position is None


@pytest.mark.asyncio
async def test_close_position_requires_broker_deal_id(monkeypatch: pytest.MonkeyPatch) -> None:
    position, trade = _position_and_trade()
    position.ig_deal_id = None
    repo = _FakeTradeRepository(position, trade)
    _patch_repo(monkeypatch, repo)
    ig_client = SimpleNamespace(is_connected=True, close_position=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await close_position(
            str(position.id),
            ClosePositionRequest(),
            _app_request(ig_client),
        )

    assert exc_info.value.status_code == 409
    ig_client.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_execute_trade_is_blocked_by_universal_execution_gate() -> None:
    ig_client = _connected_ig_client()
    ig_client.update_position_sl_tp.side_effect = RuntimeError("stop update failed")

    payload = TradeRequest(
        instrument="CS.D.EURUSD.CFD.IP",
        direction=OrderDirection.BUY,
        size=Decimal("1.0"),
        order_type=OrderType.MARKET,
        stop_loss=Decimal("1.0990"),
        take_profit=Decimal("1.1020"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await execute_trade(payload, _app_request(ig_client))

    assert exc_info.value.status_code == 403
    ig_client.place_order.assert_not_awaited()
    ig_client.close_position.assert_not_awaited()


@pytest.mark.asyncio
async def test_debug_trading_endpoints_disabled_by_default() -> None:
    ig_client = _connected_ig_client()

    with pytest.raises(HTTPException) as close_exc:
        await debug_close_all_positions(_app_request(ig_client))
    with pytest.raises(HTTPException) as order_exc:
        await debug_test_order(_app_request(ig_client))

    assert close_exc.value.status_code == 404
    assert order_exc.value.status_code == 404
    ig_client.close_position.assert_not_called()
    ig_client.place_order.assert_not_called()
