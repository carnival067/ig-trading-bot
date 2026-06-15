from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.news.free_news_safety import (
    CalendarEvent,
    FreeNewsSafetyLayer,
    NewsAction,
    NewsHeadline,
    NewsRiskDecision,
)
from src.trading.trading_loop import AutonomousTradingLoop


NOW = datetime(2026, 6, 14, 2, 0, tzinfo=timezone.utc)


class StaticFreeProvider:
    provides_economic_calendar = True

    def __init__(self, events=None, headlines=None) -> None:
        self.events = events or []
        self.headlines = headlines or []

    async def fetch_calendar(self, start, end):
        return list(self.events)

    async def fetch_headlines(self, keywords, since):
        return list(self.headlines)


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
async def test_high_impact_us_cpi_blocks_correlated_symbols(symbol: str) -> None:
    layer = FreeNewsSafetyLayer([StaticFreeProvider(events=[
        CalendarEvent("US CPI", NOW + timedelta(minutes=10), "HIGH", "USD")
    ])])

    decision = await layer.evaluate(symbol, strategy_signal=True, now=NOW)

    assert decision.news_action == NewsAction.BLOCK_TRADE
    assert decision.news_risk_score == 100
    assert "US CPI" in decision.matched_calendar_events[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["XAUUSD", "BTCUSD"])
async def test_trump_tariff_headline_reduces_or_blocks_gold_and_bitcoin(symbol: str) -> None:
    layer = FreeNewsSafetyLayer([StaticFreeProvider(headlines=[
        NewsHeadline(
            "Trump announces sweeping new tariff measures",
            NOW - timedelta(minutes=5),
            "Marketaux",
            sentiment=-0.7,
        )
    ])])

    decision = await layer.evaluate(symbol, strategy_signal=True, now=NOW)

    assert decision.news_action in {NewsAction.REDUCE_SIZE, NewsAction.BLOCK_TRADE}
    assert decision.matched_news_headlines


@pytest.mark.asyncio
async def test_low_impact_news_does_not_block_trade() -> None:
    layer = FreeNewsSafetyLayer([StaticFreeProvider(headlines=[
        NewsHeadline("Gold market weekly recap", NOW - timedelta(minutes=5), "FMP", sentiment=0.1)
    ])])

    decision = await layer.evaluate("XAUUSD", strategy_signal=True, now=NOW)

    assert decision.news_action == NewsAction.ALLOW_NORMAL
    assert decision.news_risk_score < 35


@pytest.mark.asyncio
async def test_news_cannot_create_trade_without_strategy_signal() -> None:
    layer = FreeNewsSafetyLayer([StaticFreeProvider()])

    decision = await layer.evaluate("BTCUSD", strategy_signal=False, now=NOW)

    assert decision.news_action == NewsAction.BLOCK_TRADE
    assert decision.reason == "no_strategy_signal"


@pytest.mark.asyncio
async def test_missing_real_calendar_fails_closed() -> None:
    provider = StaticFreeProvider()
    provider.provides_economic_calendar = False
    layer = FreeNewsSafetyLayer([provider])

    decision = await layer.evaluate("EURUSD", strategy_signal=True, now=NOW)

    assert decision.news_action == NewsAction.BLOCK_TRADE
    assert decision.reason == "economic_calendar_unavailable_fail_closed"


@pytest.mark.asyncio
async def test_news_cannot_override_existing_safety_rejection() -> None:
    layer = SimpleNamespace(
        evaluate=AsyncMock(
            return_value=NewsRiskDecision(0, NewsAction.ALLOW_NORMAL, reason="clear")
        )
    )
    loop = AutonomousTradingLoop(risk_engine=None, news_safety_layer=layer)

    result = await loop._apply_news_safety(None)

    assert result is None
    layer.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_extra_confirmation_is_required_when_news_requests_it() -> None:
    layer = SimpleNamespace(
        evaluate=AsyncMock(
            return_value=NewsRiskDecision(
                45,
                NewsAction.REQUIRE_EXTRA_CONFIRMATION,
                reason="moderate_news_risk",
            )
        )
    )
    loop = AutonomousTradingLoop(risk_engine=None, news_safety_layer=layer)

    rejected = await loop._apply_news_safety({
        "epic": "CS.D.EURUSD.CFD.IP",
        "direction": "BUY",
        "size": 0.4,
    })
    accepted = await loop._apply_news_safety({
        "epic": "CS.D.EURUSD.CFD.IP",
        "direction": "BUY",
        "size": 0.4,
        "extra_confirmation": True,
    })

    assert rejected is None
    assert accepted is not None


@pytest.mark.asyncio
async def test_news_reduction_only_decreases_risk_approved_size() -> None:
    layer = SimpleNamespace(
        evaluate=AsyncMock(
            return_value=NewsRiskDecision(
                65,
                NewsAction.REDUCE_SIZE,
                matched_news_headlines=("Trump tariff escalation",),
                reason="elevated_news_risk",
            )
        )
    )
    loop = AutonomousTradingLoop(risk_engine=None, news_safety_layer=layer)
    signal = {"epic": "CS.D.XAUUSD.CFD.IP", "direction": "BUY", "size": 0.4}

    result = await loop._apply_news_safety(signal)

    assert result is not None
    assert result["size"] == pytest.approx(0.2)
    assert result["news_action"] == "REDUCE_SIZE"


@pytest.mark.asyncio
async def test_protected_btc_and_gbpusd_demo_positions_are_untouched() -> None:
    layer = SimpleNamespace(
        evaluate=AsyncMock(
            return_value=NewsRiskDecision(100, NewsAction.BLOCK_TRADE, reason="event")
        )
    )
    loop = AutonomousTradingLoop(
        risk_engine=None,
        account_type="DEMO",
        news_safety_layer=layer,
    )
    positions = [
        {
            "market": {"epic": "CS.D.BTCUSD.CFD.IP"},
            "position": {"dealId": "BTC1", "stopLevel": 60000, "limitLevel": 70000},
        },
        {
            "market": {"epic": "CS.D.GBPUSD.CFD.IP"},
            "position": {"dealId": "GBP1", "stopLevel": 1.25, "limitLevel": 1.30},
        },
    ]
    loop._open_positions = positions
    loop._ig_client = SimpleNamespace(
        close_position=AsyncMock(),
        update_position_sl_tp=AsyncMock(),
    )

    result = await loop._apply_news_safety({
        "epic": "CS.D.BTCUSD.CFD.IP",
        "direction": "BUY",
        "size": 0.1,
    })

    assert result is None
    assert loop._open_positions == positions
    loop._ig_client.close_position.assert_not_awaited()
    loop._ig_client.update_position_sl_tp.assert_not_awaited()
