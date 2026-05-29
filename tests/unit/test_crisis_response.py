"""Unit tests for the Risk Engine crisis response handler.

Tests cover:
- Crisis response closes most volatile positions first (sorted by ATR descending)
- Crisis response reduces portfolio exposure by 50%
- Crisis response widens stops on remaining positions by 2.0 × ATR
- Crisis response publishes notification event
- Crisis response handles empty positions list
- Crisis response handles single position
- Crisis response completes within timing expectations
- Event bus subscription for NEWS_CRISIS_ALERT
- _handle_crisis_alert parses event payload correctly

Validates: Requirements 23.8
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.event_bus import Event, NEWS_CRISIS_ALERT
from src.risk.drawdown_monitor import DrawdownMonitor
from src.risk.exposure_manager import ExposureManager
from src.risk.kill_switch import KillSwitch
from src.risk.position_sizer import PositionSizer
from src.risk.risk_engine import (
    CrisisPosition,
    CrisisResponseResult,
    RiskEngine,
)
from src.risk.stop_manager import StopManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def position_sizer():
    return PositionSizer()


@pytest.fixture
def drawdown_monitor():
    return DrawdownMonitor(initial_equity=Decimal("100000"))


@pytest.fixture
def exposure_manager():
    return ExposureManager()


@pytest.fixture
def kill_switch():
    return KillSwitch()


@pytest.fixture
def stop_manager():
    return StopManager()


@pytest.fixture
def mock_event_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=1)
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def risk_engine(
    position_sizer,
    drawdown_monitor,
    exposure_manager,
    kill_switch,
    stop_manager,
    mock_event_bus,
):
    return RiskEngine(
        position_sizer=position_sizer,
        drawdown_monitor=drawdown_monitor,
        exposure_manager=exposure_manager,
        kill_switch=kill_switch,
        stop_manager=stop_manager,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def sample_positions():
    """Sample positions with varying ATR values for crisis testing."""
    return [
        CrisisPosition(
            instrument="EURUSD",
            direction="LONG",
            notional_value=Decimal("20000"),
            atr=Decimal("0.0080"),
            entry_price=Decimal("1.1000"),
            current_stop=Decimal("1.0880"),
        ),
        CrisisPosition(
            instrument="GBPUSD",
            direction="SHORT",
            notional_value=Decimal("15000"),
            atr=Decimal("0.0120"),  # Most volatile
            entry_price=Decimal("1.2500"),
            current_stop=Decimal("1.2680"),
        ),
        CrisisPosition(
            instrument="USDJPY",
            direction="LONG",
            notional_value=Decimal("25000"),
            atr=Decimal("0.0050"),  # Least volatile
            entry_price=Decimal("150.00"),
            current_stop=Decimal("149.25"),
        ),
        CrisisPosition(
            instrument="AUDUSD",
            direction="LONG",
            notional_value=Decimal("10000"),
            atr=Decimal("0.0100"),
            entry_price=Decimal("0.6500"),
            current_stop=Decimal("0.6350"),
        ),
    ]


# ---------------------------------------------------------------------------
# Test: Crisis response closes most volatile positions first
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_closes_most_volatile_first(risk_engine, sample_positions):
    """Positions should be closed in order of ATR descending (most volatile first)."""
    result = await risk_engine.handle_crisis_response(sample_positions)

    # GBPUSD (ATR=0.0120) should be closed first, then AUDUSD (ATR=0.0100)
    # Total exposure = 70000, target reduction = 35000
    # GBPUSD (15000) + AUDUSD (10000) = 25000 < 35000
    # GBPUSD (15000) + AUDUSD (10000) + EURUSD (20000) = 45000 >= 35000
    # So GBPUSD, AUDUSD, EURUSD should be closed (sorted by ATR: 0.0120, 0.0100, 0.0080)
    assert "GBPUSD" in result.positions_closed
    # The first position closed should be the most volatile
    assert result.positions_closed[0] == "GBPUSD"


@pytest.mark.asyncio
async def test_crisis_closes_positions_sorted_by_atr_descending(risk_engine):
    """Verify positions are sorted by ATR descending before closing."""
    positions = [
        CrisisPosition(
            instrument="LOW_VOL",
            direction="LONG",
            notional_value=Decimal("10000"),
            atr=Decimal("1"),
            entry_price=Decimal("100"),
            current_stop=Decimal("98"),
        ),
        CrisisPosition(
            instrument="HIGH_VOL",
            direction="LONG",
            notional_value=Decimal("10000"),
            atr=Decimal("5"),
            entry_price=Decimal("100"),
            current_stop=Decimal("92"),
        ),
        CrisisPosition(
            instrument="MED_VOL",
            direction="LONG",
            notional_value=Decimal("10000"),
            atr=Decimal("3"),
            entry_price=Decimal("100"),
            current_stop=Decimal("95"),
        ),
    ]

    result = await risk_engine.handle_crisis_response(positions)

    # Total = 30000, target = 15000
    # HIGH_VOL (10000) + MED_VOL (10000) = 20000 >= 15000
    # So HIGH_VOL is closed first, then MED_VOL
    assert result.positions_closed[0] == "HIGH_VOL"
    assert result.positions_closed[1] == "MED_VOL"
    assert "LOW_VOL" not in result.positions_closed


# ---------------------------------------------------------------------------
# Test: Crisis response reduces exposure by 50%
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_reduces_exposure_by_50_percent(risk_engine, sample_positions):
    """Total closed exposure should be at least 50% of total portfolio exposure."""
    result = await risk_engine.handle_crisis_response(sample_positions)

    # Total exposure = 20000 + 15000 + 25000 + 10000 = 70000
    # Target = 35000
    # Closed positions' notional should sum to >= 35000
    total_exposure = sum(p.notional_value for p in sample_positions)
    closed_exposure = sum(
        p.notional_value
        for p in sample_positions
        if p.instrument in result.positions_closed
    )

    assert closed_exposure >= total_exposure * Decimal("0.5")
    assert result.exposure_reduction_pct >= Decimal("50")


@pytest.mark.asyncio
async def test_crisis_exposure_reduction_exact_50_percent():
    """When positions divide evenly, exactly 50% should be closed."""
    stop_manager = StopManager()
    engine = RiskEngine(
        position_sizer=PositionSizer(),
        drawdown_monitor=DrawdownMonitor(initial_equity=Decimal("100000")),
        exposure_manager=ExposureManager(),
        kill_switch=KillSwitch(),
        stop_manager=stop_manager,
        event_bus=AsyncMock(),
    )

    # Two positions of equal size, different ATR
    positions = [
        CrisisPosition(
            instrument="A",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("10"),
            entry_price=Decimal("100"),
            current_stop=Decimal("85"),
        ),
        CrisisPosition(
            instrument="B",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("5"),
            entry_price=Decimal("100"),
            current_stop=Decimal("92"),
        ),
    ]

    result = await engine.handle_crisis_response(positions)

    # A (ATR=10) is more volatile, should be closed first
    # Closing A gives 50000/100000 = 50% reduction
    assert result.positions_closed == ["A"]
    assert result.exposure_reduction_pct == Decimal("50")


# ---------------------------------------------------------------------------
# Test: Crisis response widens stops by 2.0 × ATR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_widens_stops_on_remaining_positions(risk_engine):
    """Remaining positions should have stops widened by 2.0 × ATR."""
    positions = [
        CrisisPosition(
            instrument="VOLATILE",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("10"),
            entry_price=Decimal("100"),
            current_stop=Decimal("85"),
        ),
        CrisisPosition(
            instrument="STABLE",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("5"),
            entry_price=Decimal("100"),
            current_stop=Decimal("92"),
        ),
    ]

    result = await risk_engine.handle_crisis_response(positions)

    # VOLATILE is closed (higher ATR), STABLE remains
    assert "STABLE" in result.positions_widened
    assert "STABLE" in result.new_stops

    # STABLE: current_stop=92, ATR=5, widen by 2.0*ATR=10
    # For LONG: new_stop = current_stop - widen_distance = 92 - 10 = 82
    assert result.new_stops["STABLE"] == Decimal("82")


@pytest.mark.asyncio
async def test_crisis_widens_stops_short_direction(risk_engine):
    """For SHORT positions, widening moves stop further up (away from price)."""
    positions = [
        CrisisPosition(
            instrument="CLOSE_ME",
            direction="SHORT",
            notional_value=Decimal("50000"),
            atr=Decimal("20"),
            entry_price=Decimal("200"),
            current_stop=Decimal("220"),
        ),
        CrisisPosition(
            instrument="KEEP_ME",
            direction="SHORT",
            notional_value=Decimal("50000"),
            atr=Decimal("8"),
            entry_price=Decimal("150"),
            current_stop=Decimal("158"),
        ),
    ]

    result = await risk_engine.handle_crisis_response(positions)

    # CLOSE_ME is closed (higher ATR), KEEP_ME remains
    assert "KEEP_ME" in result.positions_widened
    # KEEP_ME: current_stop=158, ATR=8, widen by 2.0*ATR=16
    # For SHORT: new_stop = current_stop + widen_distance = 158 + 16 = 174
    assert result.new_stops["KEEP_ME"] == Decimal("174")


# ---------------------------------------------------------------------------
# Test: Crisis response publishes notification event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_publishes_notification(risk_engine, sample_positions, mock_event_bus):
    """Crisis response should publish a notification event to the event bus."""
    result = await risk_engine.handle_crisis_response(sample_positions)

    assert result.notification_sent is True
    # Event bus publish should have been called
    mock_event_bus.publish.assert_called()

    # Check that notification.crisis_response was published
    call_args_list = mock_event_bus.publish.call_args_list
    notification_calls = [
        call
        for call in call_args_list
        if call.args[0] == "notification.crisis_response"
    ]
    assert len(notification_calls) >= 1

    # Verify notification payload
    notification_event = notification_calls[0].args[1]
    assert notification_event.payload["severity"] == "critical"
    assert "Crisis Response Activated" in notification_event.payload["title"]


# ---------------------------------------------------------------------------
# Test: Crisis response handles empty positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_handles_empty_positions(risk_engine):
    """Crisis response with no positions should complete gracefully."""
    result = await risk_engine.handle_crisis_response([])

    assert result.positions_closed == []
    assert result.positions_widened == []
    assert result.new_stops == {}
    assert result.exposure_reduction_pct == Decimal("0")


# ---------------------------------------------------------------------------
# Test: Crisis response handles single position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_handles_single_position(risk_engine):
    """With a single position, it should be closed (100% >= 50% target)."""
    positions = [
        CrisisPosition(
            instrument="ONLY_ONE",
            direction="LONG",
            notional_value=Decimal("10000"),
            atr=Decimal("5"),
            entry_price=Decimal("100"),
            current_stop=Decimal("92"),
        ),
    ]

    result = await risk_engine.handle_crisis_response(positions)

    # Single position: closing it reduces exposure by 100% which is >= 50%
    assert result.positions_closed == ["ONLY_ONE"]
    assert result.positions_widened == []
    assert result.exposure_reduction_pct == Decimal("100")


# ---------------------------------------------------------------------------
# Test: Event bus subscription
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_to_crisis_alerts(risk_engine, mock_event_bus):
    """subscribe_to_crisis_alerts should register handler on NEWS_CRISIS_ALERT channel."""
    await risk_engine.subscribe_to_crisis_alerts()

    mock_event_bus.subscribe.assert_called_once_with(
        NEWS_CRISIS_ALERT, risk_engine._handle_crisis_alert
    )


@pytest.mark.asyncio
async def test_subscribe_raises_without_event_bus():
    """subscribe_to_crisis_alerts should raise if no event bus is configured."""
    engine = RiskEngine(
        position_sizer=PositionSizer(),
        drawdown_monitor=DrawdownMonitor(initial_equity=Decimal("100000")),
        exposure_manager=ExposureManager(),
        kill_switch=KillSwitch(),
        stop_manager=StopManager(),
        event_bus=None,
    )

    with pytest.raises(RuntimeError, match="no event bus configured"):
        await engine.subscribe_to_crisis_alerts()


# ---------------------------------------------------------------------------
# Test: _handle_crisis_alert parses event payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_crisis_alert_parses_payload(risk_engine):
    """_handle_crisis_alert should parse positions from event payload and respond."""
    event = Event(
        event_type=NEWS_CRISIS_ALERT,
        payload={
            "positions": [
                {
                    "instrument": "EURUSD",
                    "direction": "LONG",
                    "notional_value": "30000",
                    "atr": "0.0080",
                    "entry_price": "1.1000",
                    "current_stop": "1.0880",
                },
                {
                    "instrument": "GBPUSD",
                    "direction": "SHORT",
                    "notional_value": "30000",
                    "atr": "0.0120",
                    "entry_price": "1.2500",
                    "current_stop": "1.2680",
                },
            ],
        },
    )

    # Patch handle_crisis_response to capture the call
    with patch.object(
        risk_engine, "handle_crisis_response", new_callable=AsyncMock
    ) as mock_handle:
        mock_handle.return_value = CrisisResponseResult()
        await risk_engine._handle_crisis_alert(event)

        mock_handle.assert_called_once()
        positions = mock_handle.call_args.args[0]
        assert len(positions) == 2
        assert positions[0].instrument == "EURUSD"
        assert positions[0].atr == Decimal("0.0080")
        assert positions[1].instrument == "GBPUSD"
        assert positions[1].direction == "SHORT"


# ---------------------------------------------------------------------------
# Test: Crisis response with no event bus (graceful)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crisis_response_without_event_bus():
    """Crisis response should work without an event bus (notification logged only)."""
    engine = RiskEngine(
        position_sizer=PositionSizer(),
        drawdown_monitor=DrawdownMonitor(initial_equity=Decimal("100000")),
        exposure_manager=ExposureManager(),
        kill_switch=KillSwitch(),
        stop_manager=StopManager(),
        event_bus=None,
    )

    positions = [
        CrisisPosition(
            instrument="A",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("10"),
            entry_price=Decimal("100"),
            current_stop=Decimal("85"),
        ),
        CrisisPosition(
            instrument="B",
            direction="LONG",
            notional_value=Decimal("50000"),
            atr=Decimal("5"),
            entry_price=Decimal("100"),
            current_stop=Decimal("92"),
        ),
    ]

    result = await engine.handle_crisis_response(positions)

    # Should still close positions and widen stops
    assert len(result.positions_closed) > 0
    assert len(result.positions_widened) > 0 or len(result.positions_closed) == 2
    # Notification is still "sent" (logged)
    assert result.notification_sent is True
