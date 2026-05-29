"""Unit tests for Kill Switch trigger unification.

Tests that all trigger sources (drawdown 15%, VIX 3σ, portfolio loss 20%/24h,
news crisis persistence 30min) route to a single unified activation handler.

Validates: Cross-Cutting Rule 3
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.risk.kill_switch import (
    KillSwitch,
    KillSwitchState,
)


# =============================================================================
# Each trigger source activates the kill switch
# =============================================================================


class TestDrawdownTrigger:
    """Tests for drawdown trigger (15% from peak) routing to unified handler."""

    @pytest.mark.asyncio
    async def test_drawdown_above_15_percent_activates(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_drawdown(drawdown_pct=0.16)
        assert result is True
        assert ks.is_active is True
        assert ks.activation_event.trigger_source == "drawdown"
        assert "16.0%" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_drawdown_at_15_percent_does_not_activate(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_drawdown(drawdown_pct=0.15)
        assert result is False
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_drawdown_below_15_percent_does_not_activate(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_drawdown(drawdown_pct=0.10)
        assert result is False
        assert ks.is_active is False


class TestVixTrigger:
    """Tests for VIX trigger (VIX > mean + 3σ) routing to unified handler."""

    @pytest.mark.asyncio
    async def test_vix_above_3_sigma_activates(self) -> None:
        ks = KillSwitch()
        # mean=20, std=5 → threshold = 35
        result = await ks.evaluate_vix(vix_value=36.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert result is True
        assert ks.is_active is True
        assert ks.activation_event.trigger_source == "vix"
        assert "VIX" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_vix_at_threshold_does_not_activate(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_vix(vix_value=35.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert result is False
        assert ks.is_active is False


class TestPortfolioLossTrigger:
    """Tests for portfolio loss trigger (20% in 24h) routing to unified handler."""

    @pytest.mark.asyncio
    async def test_portfolio_loss_above_20_percent_activates(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_portfolio_loss(loss_pct_24h=0.25)
        assert result is True
        assert ks.is_active is True
        assert ks.activation_event.trigger_source == "portfolio_loss"
        assert "25.0%" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_portfolio_loss_at_20_percent_does_not_activate(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_portfolio_loss(loss_pct_24h=0.20)
        assert result is False
        assert ks.is_active is False


class TestCrisisPersistenceTrigger:
    """Tests for news crisis persistence trigger (30 min) routing to unified handler."""

    @pytest.mark.asyncio
    async def test_crisis_persistence_at_30_min_activates(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_crisis_persistence(
            crisis_region="europe", persistence_minutes=30.0
        )
        assert result is True
        assert ks.is_active is True
        assert ks.activation_event.trigger_source == "crisis"
        assert "europe" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_crisis_persistence_above_30_min_activates(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_crisis_persistence(
            crisis_region="asia", persistence_minutes=45.0
        )
        assert result is True
        assert ks.is_active is True
        assert "asia" in ks.activation_reason

    @pytest.mark.asyncio
    async def test_crisis_persistence_below_30_min_does_not_activate(self) -> None:
        ks = KillSwitch()
        result = await ks.evaluate_crisis_persistence(
            crisis_region="europe", persistence_minutes=29.0
        )
        assert result is False
        assert ks.is_active is False


# =============================================================================
# Multiple simultaneous triggers only activate once (Cross-Cutting Rule 3)
# =============================================================================


class TestSingleActivationFromMultipleTriggers:
    """Tests that multiple simultaneous triggers only activate once."""

    @pytest.mark.asyncio
    async def test_second_trigger_does_not_reactivate(self) -> None:
        ks = KillSwitch()
        # First trigger activates
        result1 = await ks.evaluate_vix(vix_value=40.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        # Second trigger is ignored
        result2 = await ks.evaluate_portfolio_loss(loss_pct_24h=0.25)

        assert result1 is True
        assert result2 is False
        assert ks.activation_event.trigger_source == "vix"

    @pytest.mark.asyncio
    async def test_all_four_triggers_only_one_activates(self) -> None:
        ks = KillSwitch()

        # Fire all four triggers sequentially
        r1 = await ks.evaluate_drawdown(drawdown_pct=0.20)
        r2 = await ks.evaluate_vix(vix_value=50.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        r3 = await ks.evaluate_portfolio_loss(loss_pct_24h=0.30)
        r4 = await ks.evaluate_crisis_persistence(crisis_region="global", persistence_minutes=60.0)

        # Only the first one should have activated
        assert r1 is True
        assert r2 is False
        assert r3 is False
        assert r4 is False
        assert ks.activation_event.trigger_source == "drawdown"

    @pytest.mark.asyncio
    async def test_concurrent_triggers_only_one_activates(self) -> None:
        ks = KillSwitch()

        # Fire triggers concurrently
        results = await asyncio.gather(
            ks.activate("drawdown trigger", trigger_source="drawdown"),
            ks.activate("vix trigger", trigger_source="vix"),
            ks.activate("portfolio loss trigger", trigger_source="portfolio_loss"),
            ks.activate("crisis trigger", trigger_source="crisis"),
        )

        # Exactly one should succeed
        assert sum(results) == 1
        assert ks.is_active is True

    @pytest.mark.asyncio
    async def test_direct_activate_prevents_duplicate(self) -> None:
        ks = KillSwitch()
        r1 = await ks.activate("first reason", trigger_source="drawdown")
        r2 = await ks.activate("second reason", trigger_source="vix")

        assert r1 is True
        assert r2 is False
        assert ks.activation_reason == "first reason"


# =============================================================================
# Trigger source is logged and tracked
# =============================================================================


class TestTriggerSourceAudit:
    """Tests that trigger sources are logged and tracked for audit."""

    @pytest.mark.asyncio
    async def test_single_trigger_recorded_in_audit(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_vix(vix_value=40.0, vix_30d_mean=20.0, vix_30d_std=5.0)

        sources = ks.trigger_sources
        assert len(sources) == 1
        assert sources[0]["trigger_source"] == "vix"
        assert sources[0]["activated"] is True
        assert "VIX" in sources[0]["reason"]
        assert isinstance(sources[0]["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_multiple_triggers_all_recorded(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_drawdown(drawdown_pct=0.20)
        await ks.evaluate_vix(vix_value=40.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        await ks.evaluate_portfolio_loss(loss_pct_24h=0.25)

        sources = ks.trigger_sources
        assert len(sources) == 3

        # First trigger activated, others did not
        assert sources[0]["trigger_source"] == "drawdown"
        assert sources[0]["activated"] is True
        assert sources[1]["trigger_source"] == "vix"
        assert sources[1]["activated"] is False
        assert sources[2]["trigger_source"] == "portfolio_loss"
        assert sources[2]["activated"] is False

    @pytest.mark.asyncio
    async def test_trigger_sources_returns_copy(self) -> None:
        ks = KillSwitch()
        await ks.activate("test", trigger_source="vix")

        sources = ks.trigger_sources
        sources.clear()  # Modifying the returned list

        # Internal state should be unaffected
        assert len(ks.trigger_sources) == 1

    @pytest.mark.asyncio
    async def test_trigger_below_threshold_not_recorded(self) -> None:
        ks = KillSwitch()
        # These don't meet thresholds, so no trigger is recorded
        await ks.evaluate_drawdown(drawdown_pct=0.10)
        await ks.evaluate_vix(vix_value=30.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        await ks.evaluate_portfolio_loss(loss_pct_24h=0.15)
        await ks.evaluate_crisis_persistence(crisis_region="europe", persistence_minutes=20.0)

        assert len(ks.trigger_sources) == 0

    @pytest.mark.asyncio
    async def test_event_published_on_activation(self) -> None:
        publisher = AsyncMock()
        ks = KillSwitch(event_publisher=publisher)

        await ks.activate("VIX spike", trigger_source="vix")

        publisher.assert_called_once()
        call_args = publisher.call_args
        assert call_args[0][0] == "kill_switch.activated"
        assert call_args[0][1]["trigger_source"] == "vix"
        assert call_args[0][1]["reason"] == "VIX spike"

    @pytest.mark.asyncio
    async def test_event_not_published_on_duplicate_activation(self) -> None:
        publisher = AsyncMock()
        ks = KillSwitch(event_publisher=publisher)

        await ks.activate("first", trigger_source="vix")
        await ks.activate("second", trigger_source="drawdown")

        # Only one event published (for the first activation)
        assert publisher.call_count == 1

    @pytest.mark.asyncio
    async def test_event_publisher_failure_does_not_prevent_activation(self) -> None:
        publisher = AsyncMock(side_effect=RuntimeError("publish failed"))
        ks = KillSwitch(event_publisher=publisher)

        result = await ks.activate("VIX spike", trigger_source="vix")

        assert result is True
        assert ks.is_active is True


# =============================================================================
# All sources blocked when active (HFT, copy trading, manual, strategies)
# =============================================================================


class TestAllSourcesBlockedWhenActive:
    """Tests that all signal sources are blocked when kill switch is active."""

    @pytest.mark.asyncio
    async def test_hft_signals_blocked_after_drawdown_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_drawdown(drawdown_pct=0.20)
        assert ks.is_signal_allowed(source="hft") is False

    @pytest.mark.asyncio
    async def test_copy_trading_signals_blocked_after_vix_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_vix(vix_value=40.0, vix_30d_mean=20.0, vix_30d_std=5.0)
        assert ks.is_signal_allowed(source="copy_trading") is False

    @pytest.mark.asyncio
    async def test_manual_signals_blocked_after_portfolio_loss_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_portfolio_loss(loss_pct_24h=0.25)
        assert ks.is_signal_allowed(source="manual") is False

    @pytest.mark.asyncio
    async def test_strategy_signals_blocked_after_crisis_trigger(self) -> None:
        ks = KillSwitch()
        await ks.evaluate_crisis_persistence(crisis_region="global", persistence_minutes=35.0)
        assert ks.is_signal_allowed(source="strategy") is False

    @pytest.mark.asyncio
    async def test_all_sources_blocked_regardless_of_trigger(self) -> None:
        """All signal sources are blocked no matter which trigger activated."""
        ks = KillSwitch()
        await ks.activate("any reason", trigger_source="drawdown")

        sources_to_check = ["hft", "copy_trading", "manual", "strategy", ""]
        for source in sources_to_check:
            assert ks.is_signal_allowed(source=source) is False, (
                f"Signal from '{source}' should be blocked when kill switch is active"
            )

    @pytest.mark.asyncio
    async def test_all_sources_allowed_when_inactive(self) -> None:
        ks = KillSwitch()
        sources_to_check = ["hft", "copy_trading", "manual", "strategy", ""]
        for source in sources_to_check:
            assert ks.is_signal_allowed(source=source) is True
