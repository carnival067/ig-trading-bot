"""Unit tests for the Risk Engine orchestrator.

Tests cover:
- Signal rejected when kill switch active
- Signal rejected when daily loss limit hit
- Signal rejected when exposure limit breached
- Signal rejected when RR < 1.5
- Signal allowed with all checks passing
- Reduction factors applied from drawdown monitor
- Kill switch triggered by drawdown monitor
- Event publishing on rejection and kill switch activation
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.risk.drawdown_monitor import (
    DrawdownCheckResult,
    DrawdownMonitor,
    ReductionFactor,
    TradeDecision,
)
from src.risk.exposure_manager import ExposureCheckResult, ExposureManager
from src.risk.kill_switch import KillSwitch
from src.risk.position_sizer import PositionSizer
from src.risk.risk_engine import RiskEngine, TradeSignal, ValidationResult
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
    return bus


@pytest.fixture
def risk_engine(position_sizer, drawdown_monitor, exposure_manager, kill_switch, stop_manager, mock_event_bus):
    return RiskEngine(
        position_sizer=position_sizer,
        drawdown_monitor=drawdown_monitor,
        exposure_manager=exposure_manager,
        kill_switch=kill_switch,
        stop_manager=stop_manager,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def valid_signal():
    """A valid trade signal that should pass all checks under normal conditions.

    ATR is set so that position size = (100000 * 0.01) / (50 * 1.5) = 13.33
    which is well below the 5% cap of 5000.
    """
    return TradeSignal(
        instrument="FTSE100",
        direction="LONG",
        entry_price=Decimal("7500"),
        stop_loss=Decimal("7425"),  # 75 points risk
        take_profit=Decimal("7650"),  # 150 points reward → RR = 2.0
        confidence=80,
        strategy="trend_following",
        asset_class="indices",
        notional_value=Decimal("10000"),
        region="europe",
        is_hft=False,
        atr=Decimal("50"),
        atr_zscore=1.0,
    )


# ---------------------------------------------------------------------------
# Test: Signal rejected when kill switch active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_rejected_when_kill_switch_active(risk_engine, valid_signal, kill_switch, mock_event_bus):
    """When the kill switch is active, all signals must be rejected."""
    # Activate the kill switch
    await kill_switch.activate(reason="Test activation", trigger_source="test")

    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is False
    assert len(result.rejection_reasons) == 1
    assert "Kill switch is active" in result.rejection_reasons[0]
    assert result.position_size is None
    # Event should be published
    mock_event_bus.publish.assert_called()


# ---------------------------------------------------------------------------
# Test: Signal rejected when daily loss limit hit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_rejected_when_daily_loss_limit_hit(risk_engine, valid_signal, drawdown_monitor, mock_event_bus):
    """When daily loss limit is breached, signals must be rejected."""
    # Simulate daily loss exceeding 3% of equity (3000 on 100000)
    drawdown_monitor.update_on_trade_close(Decimal("-3100"))

    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is False
    assert len(result.rejection_reasons) == 1
    assert "Daily" in result.rejection_reasons[0] or "loss" in result.rejection_reasons[0].lower()
    assert result.position_size is None
    mock_event_bus.publish.assert_called()


# ---------------------------------------------------------------------------
# Test: Signal rejected when exposure limit breached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_rejected_when_exposure_limit_breached(risk_engine, valid_signal, mock_event_bus):
    """When adding a position would breach exposure limits, signal is rejected."""
    # Create existing positions that use up most of the indices exposure (30% limit)
    current_positions = [
        {
            "instrument": "DAX",
            "asset_class": "indices",
            "notional_value": "28000",
            "region": "europe",
        },
    ]

    # Signal with notional value that would push over 30% limit
    valid_signal.notional_value = Decimal("5000")

    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=current_positions,
    )

    assert result.allowed is False
    assert len(result.rejection_reasons) == 1
    assert "exposure" in result.rejection_reasons[0].lower() or "Exposure" in result.rejection_reasons[0]
    assert result.position_size is None
    mock_event_bus.publish.assert_called()


# ---------------------------------------------------------------------------
# Test: Signal rejected when RR < 1.5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_rejected_when_rr_below_minimum(risk_engine, mock_event_bus):
    """When risk-reward ratio is below 1.5, signal must be rejected."""
    bad_rr_signal = TradeSignal(
        instrument="FTSE100",
        direction="LONG",
        entry_price=Decimal("7500"),
        stop_loss=Decimal("7400"),  # 100 points risk
        take_profit=Decimal("7550"),  # 50 points reward → RR = 0.5
        confidence=80,
        strategy="trend_following",
        asset_class="indices",
        notional_value=Decimal("10000"),
        region="europe",
        is_hft=False,
        atr=Decimal("50"),
        atr_zscore=1.0,
    )

    result = await risk_engine.validate_signal(
        signal=bad_rr_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is False
    assert len(result.rejection_reasons) == 1
    assert "Risk-reward" in result.rejection_reasons[0] or "1.5" in result.rejection_reasons[0]
    assert result.position_size is None
    mock_event_bus.publish.assert_called()


# ---------------------------------------------------------------------------
# Test: Signal allowed with all checks passing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_allowed_with_all_checks_passing(risk_engine, valid_signal):
    """When all risk checks pass, signal should be allowed with position size and stops."""
    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is True
    assert result.rejection_reasons == []
    assert result.position_size is not None
    assert result.position_size > Decimal("0")
    assert result.stop_loss is not None
    assert result.take_profit_levels is not None
    assert len(result.take_profit_levels) > 0
    assert result.trigger_kill_switch is False


# ---------------------------------------------------------------------------
# Test: Reduction factors applied from drawdown monitor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reduction_factors_applied_from_drawdown(risk_engine, valid_signal, drawdown_monitor):
    """When drawdown exceeds 10%, a 75% reduction factor should be applied."""
    # Set peak equity high, then check with lower equity to trigger 10%+ drawdown
    drawdown_monitor.peak_equity = Decimal("120000")

    # Current equity = 105000 → drawdown = (120000 - 105000) / 120000 = 12.5%
    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("105000"),
        current_positions=[],
    )

    assert result.allowed is True
    # Should have a drawdown reduction factor applied
    drawdown_reductions = [r for r in result.applied_reductions if r.source == "drawdown"]
    assert len(drawdown_reductions) == 1
    assert drawdown_reductions[0].factor == Decimal("0.25")


# ---------------------------------------------------------------------------
# Test: Kill switch triggered by drawdown monitor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_triggered_by_drawdown(risk_engine, valid_signal, drawdown_monitor, kill_switch, mock_event_bus):
    """When drawdown exceeds 15%, the kill switch should be activated."""
    # Set peak equity high, then check with much lower equity to trigger 15%+ drawdown
    drawdown_monitor.peak_equity = Decimal("120000")

    # Current equity = 100000 → drawdown = (120000 - 100000) / 120000 = 16.7%
    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is False
    assert result.trigger_kill_switch is True
    assert kill_switch.is_active is True
    # Kill switch activation event should be published
    mock_event_bus.publish.assert_called()
    # Check that the event type was kill_switch.activated
    call_args_list = mock_event_bus.publish.call_args_list
    event_types = [call.args[0] for call in call_args_list]
    assert "kill_switch.activated" in event_types


# ---------------------------------------------------------------------------
# Test: Event publishing on rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_published_on_rejection(risk_engine, mock_event_bus):
    """Risk events should be published to the event bus when signals are rejected."""
    bad_rr_signal = TradeSignal(
        instrument="DAX",
        direction="SHORT",
        entry_price=Decimal("15000"),
        stop_loss=Decimal("15200"),  # 200 points risk
        take_profit=Decimal("14950"),  # 50 points reward → RR = 0.25
        confidence=75,
        strategy="mean_reversion",
        asset_class="indices",
        notional_value=Decimal("5000"),
        atr=Decimal("80"),
        atr_zscore=0.5,
    )

    await risk_engine.validate_signal(
        signal=bad_rr_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    # Verify event bus was called
    mock_event_bus.publish.assert_called()
    call_args = mock_event_bus.publish.call_args
    channel = call_args.args[0]
    assert channel == "risk.signal_rejected"


# ---------------------------------------------------------------------------
# Test: Event publishing on kill switch activation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_published_on_kill_switch_activation(risk_engine, valid_signal, drawdown_monitor, mock_event_bus):
    """Kill switch activation events should be published to the event bus."""
    # Trigger kill switch via drawdown
    drawdown_monitor.peak_equity = Decimal("120000")

    await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    # Find the kill_switch.activated event
    call_args_list = mock_event_bus.publish.call_args_list
    kill_switch_calls = [
        call for call in call_args_list if call.args[0] == "kill_switch.activated"
    ]
    assert len(kill_switch_calls) == 1
    event = kill_switch_calls[0].args[1]
    assert event.payload["trigger_source"] == "drawdown"


# ---------------------------------------------------------------------------
# Test: No event bus configured (graceful handling)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_event_bus_configured(position_sizer, drawdown_monitor, exposure_manager, kill_switch, stop_manager, valid_signal):
    """Risk engine should work without an event bus (events are just logged)."""
    engine = RiskEngine(
        position_sizer=position_sizer,
        drawdown_monitor=drawdown_monitor,
        exposure_manager=exposure_manager,
        kill_switch=kill_switch,
        stop_manager=stop_manager,
        event_bus=None,
    )

    result = await engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    # Should still work without errors
    assert result.allowed is True
    assert result.position_size is not None


# ---------------------------------------------------------------------------
# Test: Volatility reduction applied when ATR z-score > 2.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_volatility_reduction_applied(risk_engine, valid_signal):
    """When ATR z-score > 2.0, a 50% volatility reduction should be applied."""
    valid_signal.atr_zscore = 2.5  # Above threshold

    result = await risk_engine.validate_signal(
        signal=valid_signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is True
    volatility_reductions = [r for r in result.applied_reductions if r.source == "volatility"]
    assert len(volatility_reductions) == 1
    assert volatility_reductions[0].factor == Decimal("0.5")


# ---------------------------------------------------------------------------
# Test: Signal rejected by position size cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_rejected_by_position_size_cap(risk_engine):
    """When calculated position size exceeds 5% of equity, signal is rejected."""
    # Use a very small ATR so the formula produces a huge position size:
    # size = (100000 * 0.01) / (0.001 * 1.5) = 666666.67 which exceeds 5% cap (5000)
    signal = TradeSignal(
        instrument="EURUSD",
        direction="LONG",
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0985"),  # 0.0015 risk
        take_profit=Decimal("1.1030"),  # 0.003 reward → RR = 2.0
        confidence=80,
        strategy="trend_following",
        asset_class="forex",
        notional_value=Decimal("5000"),
        region="europe",
        is_hft=False,
        atr=Decimal("0.001"),
        atr_zscore=0.5,
    )

    result = await risk_engine.validate_signal(
        signal=signal,
        account_equity=Decimal("100000"),
        current_positions=[],
    )

    assert result.allowed is False
    assert any("position size limit" in r.lower() for r in result.rejection_reasons)


# ---------------------------------------------------------------------------
# Test: Integration of all components together
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_all_components(position_sizer, exposure_manager, kill_switch, stop_manager, mock_event_bus):
    """Full integration test: drawdown reduction + volatility reduction applied together."""
    # Create a drawdown monitor with peak equity higher than current
    drawdown_monitor = DrawdownMonitor(initial_equity=Decimal("110000"))
    drawdown_monitor.peak_equity = Decimal("120000")

    engine = RiskEngine(
        position_sizer=position_sizer,
        drawdown_monitor=drawdown_monitor,
        exposure_manager=exposure_manager,
        kill_switch=kill_switch,
        stop_manager=stop_manager,
        event_bus=mock_event_bus,
    )

    # Current equity = 105000 → drawdown = (120000 - 105000) / 120000 = 12.5% (triggers reduction)
    # ATR z-score = 2.5 → triggers volatility reduction
    signal = TradeSignal(
        instrument="FTSE100",
        direction="LONG",
        entry_price=Decimal("7500"),
        stop_loss=Decimal("7425"),  # 75 points risk
        take_profit=Decimal("7650"),  # 150 points reward → RR = 2.0
        confidence=80,
        strategy="trend_following",
        asset_class="indices",
        notional_value=Decimal("10000"),
        region="europe",
        is_hft=False,
        atr=Decimal("50"),
        atr_zscore=2.5,
    )

    result = await engine.validate_signal(
        signal=signal,
        account_equity=Decimal("105000"),
        current_positions=[],
    )

    assert result.allowed is True
    assert result.position_size is not None
    assert result.position_size > Decimal("0")
    assert result.stop_loss is not None
    assert len(result.take_profit_levels) > 0

    # Both drawdown and volatility reductions should be applied
    sources = [r.source for r in result.applied_reductions]
    assert "drawdown" in sources
    assert "volatility" in sources

    # Position size should be smaller than without reductions
    # Without reductions: (105000 * 0.01) / (50 * 1.5) = 14.0
    # With 0.25 * 0.5 = 0.125 factor: 14.0 * 0.125 = 1.75
    assert result.position_size <= Decimal("14.0")
