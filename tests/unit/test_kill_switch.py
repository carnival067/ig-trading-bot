"""Unit tests for the KillSwitch module.

Tests cover VIX-based activation, portfolio loss activation, close-all-positions
logic with retries, signal rejection while active, minimum active duration,
manual deactivation with confirmation, and single-activation-event processing.

Validates: Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 6.7, Cross-Cutting Rule 3
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.risk.kill_switch import (
    CloseAllResult,
    KillSwitch,
    KillSwitchActivationEvent,
    KillSwitchState,
    PositionCloseResult,
)


# =============================================================================
# Task 6.1: VIX-based activation trigger
# =============================================================================


class TestVixActivation:
    """Tests for VIX-based kill switch activation."""

    @pytest.mark.asyncio
    async def test_activates_when_vix_exceeds_3_sigma(self) -> None:
        ks = KillSwitch()
        # mean=20, std=5 → threshold = 20 + 3*5 = 35
        result = await ks.evaluate_vix(vix_value=36.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert result is True
        assert ks.is_active is True
        assert "VIX" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_does_not_activate_below_3_sigma(self) -> None:
        ks = KillSwitch()
        # mean=20, std=5 → threshold = 35, vix=34 is below
        result = await ks.evaluate_vix(vix_value=34.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert result is False
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_does_not_activate_at_exact_threshold(self) -> None:
        ks = KillSwitch()
        # mean=20, std=5 → threshold = 35, vix=35 is NOT > threshold
        result = await ks.evaluate_vix(vix_value=35.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert result is False
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_activation_event_records_vix_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_vix(vix_value=40.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert ks.activation_event is not None
        assert ks.activation_event.trigger_source == "vix"


# =============================================================================
# Task 6.1: Portfolio loss activation trigger
# =============================================================================


class TestPortfolioLossActivation:
    """Tests for 24-hour portfolio loss kill switch activation."""

    @pytest.mark.asyncio
    async def test_activates_when_loss_exceeds_20_percent(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_portfolio_loss(loss_pct_24h=0.25)
        assert result is True
        assert ks.is_active is True
        assert "25.0%" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_does_not_activate_at_20_percent(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_portfolio_loss(loss_pct_24h=0.20)
        assert result is False
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_does_not_activate_below_20_percent(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_portfolio_loss(loss_pct_24h=0.15)
        assert result is False
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_activation_event_records_portfolio_loss_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_portfolio_loss(loss_pct_24h=0.21)
        assert ks.activation_event is not None
        assert ks.activation_event.trigger_source == "portfolio_loss"


# =============================================================================
# Task 6.2: Close all positions with timeout and retries
# =============================================================================


class TestCloseAllPositions:
    """Tests for close-all-positions logic with timeout and retries."""

    @pytest.mark.asyncio
    async def test_closes_all_positions_successfully(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}, {"id": "pos2"}, {"id": "pos3"}]
        close_fn = AsyncMock(return_value=True)

        result = await ks.close_all_positions(positions, close_fn)

        assert result.total_positions == 3
        assert result.closed_successfully == 3
        assert result.failed_positions == []

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}]
        # Fail twice, succeed on third attempt
        close_fn = AsyncMock(side_effect=[False, False, True])

        result = await ks.close_all_positions(positions, close_fn)

        assert result.total_positions == 1
        assert result.closed_successfully == 1
        assert result.failed_positions == []
        assert close_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self) -> None:
        ks = KillSwitch(close_timeout_seconds=1, retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}]

        call_count = 0

        async def slow_then_fast(pos: dict) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                await asyncio.sleep(5)  # Will timeout
            return True

        result = await ks.close_all_positions(positions, slow_then_fast)

        assert result.total_positions == 1
        assert result.closed_successfully == 1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_flags_failed_positions_after_all_retries(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}]
        close_fn = AsyncMock(return_value=False)

        result = await ks.close_all_positions(positions, close_fn)

        assert result.total_positions == 1
        assert result.closed_successfully == 0
        assert len(result.failed_positions) == 1
        assert result.failed_positions[0].position_id == "pos1"
        assert result.failed_positions[0].success is False
        assert result.failed_positions[0].attempts == 4  # 1 initial + 3 retries

    @pytest.mark.asyncio
    async def test_handles_empty_positions_list(self) -> None:
        ks = KillSwitch()
        close_fn = AsyncMock(return_value=True)

        result = await ks.close_all_positions([], close_fn)

        assert result.total_positions == 0
        assert result.closed_successfully == 0
        assert result.failed_positions == []
        close_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_exception_in_close_fn(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}]
        close_fn = AsyncMock(side_effect=RuntimeError("broker error"))

        result = await ks.close_all_positions(positions, close_fn)

        assert result.total_positions == 1
        assert result.closed_successfully == 0
        assert len(result.failed_positions) == 1
        assert "broker error" in result.failed_positions[0].error

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)
        positions = [{"id": "pos1"}, {"id": "pos2"}]

        call_count = {"pos1": 0, "pos2": 0}

        async def mixed_close(pos: dict) -> bool:
            pid = pos["id"]
            call_count[pid] += 1
            if pid == "pos1":
                return True
            return False  # pos2 always fails

        result = await ks.close_all_positions(positions, mixed_close)

        assert result.total_positions == 2
        assert result.closed_successfully == 1
        assert len(result.failed_positions) == 1
        assert result.failed_positions[0].position_id == "pos2"

    @pytest.mark.asyncio
    async def test_position_with_id_attribute(self) -> None:
        ks = KillSwitch(retry_interval_seconds=0.01)

        class Position:
            def __init__(self, id: str):
                self.id = id

        positions = [Position("attr_pos")]
        close_fn = AsyncMock(return_value=True)

        result = await ks.close_all_positions(positions, close_fn)

        assert result.closed_successfully == 1


# =============================================================================
# Task 6.3: Signal rejection while active
# =============================================================================


class TestSignalRejection:
    """Tests for signal rejection while kill switch is active."""

    @pytest.mark.asyncio
    async def test_signals_allowed_when_inactive(self) -> None:
        ks = KillSwitch()
        assert ks.is_signal_allowed() is True
        assert ks.is_signal_allowed(source="hft") is True
        assert ks.is_signal_allowed(source="copy_trading") is True
        assert ks.is_signal_allowed(source="manual") is True
        assert ks.is_signal_allowed(source="strategy") is True

    @pytest.mark.asyncio
    async def test_all_signals_blocked_when_active(self) -> None:
        ks = KillSwitch()
        await ks.activate("test activation", trigger_source="test")

        assert ks.is_signal_allowed() is False
        assert ks.is_signal_allowed(source="hft") is False
        assert ks.is_signal_allowed(source="copy_trading") is False
        assert ks.is_signal_allowed(source="manual") is False
        assert ks.is_signal_allowed(source="strategy") is False

    @pytest.mark.asyncio
    async def test_signals_allowed_after_deactivation(self) -> None:
        ks = KillSwitch(min_active_minutes=0)  # Allow immediate deactivation for test
        await ks.activate("test", trigger_source="test")
        assert ks.is_signal_allowed() is False

        # Patch time to simulate 5 minutes passing
        await ks.deactivate("CONFIRM_DEACTIVATION")
        assert ks.is_signal_allowed() is True


# =============================================================================
# Task 6.4: Minimum active duration and manual deactivation
# =============================================================================


class TestDeactivation:
    """Tests for 5-minute minimum active duration and manual deactivation."""

    @pytest.mark.asyncio
    async def test_cannot_deactivate_before_5_minutes(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")

        # Immediately try to deactivate — should fail
        result = await ks.deactivate("CONFIRM")
        assert result is False
        assert ks.is_active is True

    @pytest.mark.asyncio
    async def test_can_deactivate_after_5_minutes(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")

        # Simulate time passing by patching activation_time
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=6)

        result = await ks.deactivate("CONFIRM_DEACTIVATION")
        assert result is True
        assert ks.is_active is False
        assert ks.state == KillSwitchState.INACTIVE

    @pytest.mark.asyncio
    async def test_deactivation_requires_confirmation_token(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=6)

        with pytest.raises(ValueError, match="Confirmation token is required"):
            await ks.deactivate("")

        with pytest.raises(ValueError, match="Confirmation token is required"):
            await ks.deactivate("   ")

    @pytest.mark.asyncio
    async def test_deactivation_when_not_active_returns_false(self) -> None:
        ks = KillSwitch()
        result = await ks.deactivate("CONFIRM")
        assert result is False

    @pytest.mark.asyncio
    async def test_can_deactivate_property_false_when_inactive(self) -> None:
        ks = KillSwitch()
        assert ks.can_deactivate is False

    @pytest.mark.asyncio
    async def test_can_deactivate_property_false_before_5_minutes(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")
        assert ks.can_deactivate is False

    @pytest.mark.asyncio
    async def test_can_deactivate_property_true_after_5_minutes(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert ks.can_deactivate is True

    @pytest.mark.asyncio
    async def test_custom_min_active_duration(self) -> None:
        ks = KillSwitch(min_active_minutes=1)
        await ks.activate("test", trigger_source="test")

        # After 30 seconds — still too early
        ks._activation_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        assert ks.can_deactivate is False

        # After 61 seconds — can deactivate
        ks._activation_time = datetime.now(timezone.utc) - timedelta(seconds=61)
        assert ks.can_deactivate is True


# =============================================================================
# Task 6.5: Single-activation-event processing
# =============================================================================


class TestSingleActivationEvent:
    """Tests for single-activation-event processing with asyncio.Lock."""

    @pytest.mark.asyncio
    async def test_only_first_activation_succeeds(self) -> None:
        ks = KillSwitch()

        result1 = await ks.activate("VIX spike", trigger_source="vix")
        result2 = await ks.activate("Portfolio loss", trigger_source="portfolio_loss")

        assert result1 is True
        assert result2 is False
        assert ks.activation_reason == "VIX spike"
        assert ks.activation_event.trigger_source == "vix"

    @pytest.mark.asyncio
    async def test_concurrent_activations_only_one_succeeds(self) -> None:
        ks = KillSwitch()

        # Simulate concurrent activation attempts
        results = await asyncio.gather(
            ks.activate("trigger_1", trigger_source="vix"),
            ks.activate("trigger_2", trigger_source="portfolio_loss"),
            ks.activate("trigger_3", trigger_source="drawdown"),
        )

        # Exactly one should succeed
        assert sum(results) == 1
        assert ks.is_active is True

    @pytest.mark.asyncio
    async def test_can_reactivate_after_deactivation(self) -> None:
        ks = KillSwitch(min_active_minutes=0)
        await ks.activate("first", trigger_source="vix")
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        await ks.deactivate("CONFIRM")

        assert ks.is_active is False

        result = await ks.activate("second", trigger_source="portfolio_loss")
        assert result is True
        assert ks.is_active is True
        assert ks.activation_reason == "second"


# =============================================================================
# State and Properties
# =============================================================================


class TestKillSwitchState:
    """Tests for kill switch state management."""

    def test_initial_state_is_inactive(self) -> None:
        ks = KillSwitch()
        assert ks.state == KillSwitchState.INACTIVE
        assert ks.is_active is False
        assert ks.activation_time is None
        assert ks.activation_reason == ""
        assert ks.activation_event is None

    @pytest.mark.asyncio
    async def test_state_transitions_to_active(self) -> None:
        ks = KillSwitch()
        await ks.activate("test reason", trigger_source="test")
        assert ks.state == KillSwitchState.ACTIVE
        assert ks.is_active is True
        assert ks.activation_time is not None
        assert ks.activation_reason == "test reason"

    @pytest.mark.asyncio
    async def test_state_transitions_back_to_inactive(self) -> None:
        ks = KillSwitch(min_active_minutes=0)
        await ks.activate("test", trigger_source="test")
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        await ks.deactivate("CONFIRM")
        assert ks.state == KillSwitchState.INACTIVE
        assert ks.is_active is False


class TestKillSwitchStateEnum:
    """Tests for the KillSwitchState enum."""

    def test_all_values_exist(self) -> None:
        assert KillSwitchState.INACTIVE.value == "inactive"
        assert KillSwitchState.ACTIVE.value == "active"
        assert KillSwitchState.COOLDOWN.value == "cooldown"


class TestPositionCloseResult:
    """Tests for the PositionCloseResult dataclass."""

    def test_creation_success(self) -> None:
        result = PositionCloseResult(position_id="pos1", success=True, attempts=1)
        assert result.position_id == "pos1"
        assert result.success is True
        assert result.attempts == 1
        assert result.error is None

    def test_creation_failure(self) -> None:
        result = PositionCloseResult(
            position_id="pos2", success=False, attempts=4, error="timeout"
        )
        assert result.position_id == "pos2"
        assert result.success is False
        assert result.attempts == 4
        assert result.error == "timeout"


class TestCloseAllResultDataclass:
    """Tests for the CloseAllResult dataclass."""

    def test_creation(self) -> None:
        result = CloseAllResult(
            total_positions=3,
            closed_successfully=2,
            failed_positions=[
                PositionCloseResult(position_id="pos3", success=False, attempts=4)
            ],
        )
        assert result.total_positions == 3
        assert result.closed_successfully == 2
        assert len(result.failed_positions) == 1


class TestKillSwitchActivationEventDataclass:
    """Tests for the KillSwitchActivationEvent dataclass."""

    def test_creation(self) -> None:
        now = datetime.now(timezone.utc)
        event = KillSwitchActivationEvent(
            reason="VIX spike", timestamp=now, trigger_source="vix"
        )
        assert event.reason == "VIX spike"
        assert event.timestamp == now
        assert event.trigger_source == "vix"


# =============================================================================
# get_status() method
# =============================================================================


class TestGetStatus:
    """Tests for the get_status() method."""

    def test_status_when_inactive(self) -> None:
        ks = KillSwitch()
        status = ks.get_status()
        assert status["active"] is False
        assert status["reason"] == ""
        assert status["activation_time"] is None
        assert status["duration"] is None
        assert status["can_deactivate"] is False

    @pytest.mark.asyncio
    async def test_status_when_active(self) -> None:
        ks = KillSwitch()
        await ks.activate("VIX spike", trigger_source="vix")
        status = ks.get_status()
        assert status["active"] is True
        assert status["reason"] == "VIX spike"
        assert status["activation_time"] is not None
        assert status["duration"] is not None
        assert status["duration"] >= 0
        assert status["can_deactivate"] is False

    @pytest.mark.asyncio
    async def test_status_can_deactivate_after_5_minutes(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="test")
        ks._activation_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        status = ks.get_status()
        assert status["active"] is True
        assert status["can_deactivate"] is True
        assert status["duration"] >= 360  # at least 6 minutes in seconds
