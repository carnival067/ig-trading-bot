"""Unit tests for notification formatters and notification service retry logic.

Tests cover:
- Trade notification formatting (Req 17.2)
- Kill switch notification formatting and delivery to ALL channels (Req 17.3)
- Retry logic: 3 retries, 30-second intervals, fallback (Req 17.4)
- HFT circuit breaker notification formatting (Req 22.9)
- Crisis alert notification formatting (Req 23.7)
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.notifications.formatters import (
    CrisisAlertData,
    HFTCircuitBreakerData,
    HFTModeDisabledData,
    KillSwitchNotificationData,
    NotificationType,
    Priority,
    TradeDirection,
    TradeNotificationData,
    format_crisis_alert_notification,
    format_hft_circuit_breaker_notification,
    format_hft_mode_disabled_notification,
    format_kill_switch_notification,
    format_trade_notification,
)
from src.notifications.notification_service import (
    ChannelConfig,
    DeliveryResult,
    NotificationService,
    RetryConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockChannel:
    """Mock notification channel for testing."""

    def __init__(self, name: str, should_succeed: bool = True) -> None:
        self._name = name
        self._should_succeed = should_succeed
        self.sent_messages: list[str] = []
        self.send_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        if self._should_succeed:
            self.sent_messages.append(message)
            return True
        return False


class FailThenSucceedChannel:
    """Channel that fails N times then succeeds."""

    def __init__(self, name: str, fail_count: int = 2) -> None:
        self._name = name
        self._fail_count = fail_count
        self.send_count = 0
        self.sent_messages: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        if self.send_count <= self._fail_count:
            return False
        self.sent_messages.append(message)
        return True


class ExceptionChannel:
    """Channel that raises exceptions."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.send_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        raise ConnectionError("Network unreachable")


class SlowChannel:
    """Channel that takes a long time to respond."""

    def __init__(self, name: str, delay: float = 10.0) -> None:
        self._name = name
        self._delay = delay
        self.send_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        await asyncio.sleep(self._delay)
        return True


# ---------------------------------------------------------------------------
# Trade Notification Formatting Tests (Req 17.2)
# ---------------------------------------------------------------------------


class TestTradeNotificationFormatting:
    """Tests for trade notification formatting."""

    def test_format_trade_opened(self) -> None:
        """Trade opened notification includes instrument, direction, size, entry, strategy."""
        data = TradeNotificationData(
            instrument="EUR/USD",
            direction=TradeDirection.LONG,
            size=Decimal("1.5"),
            entry_price=Decimal("1.0850"),
            strategy="Trend Following",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        result = format_trade_notification(data)

        assert "TRADE OPENED" in result
        assert "EUR/USD" in result
        assert "LONG" in result
        assert "1.5" in result
        assert "1.0850" in result
        assert "Trend Following" in result
        assert "2024-01-15 10:30:00 UTC" in result

    def test_format_trade_closed_profit(self) -> None:
        """Trade closed with profit shows PnL with positive sign."""
        data = TradeNotificationData(
            instrument="GBP/USD",
            direction=TradeDirection.SHORT,
            size=Decimal("2.0"),
            entry_price=Decimal("1.2700"),
            exit_price=Decimal("1.2650"),
            pnl=Decimal("100.00"),
            strategy="Mean Reversion",
            timestamp=datetime(2024, 1, 15, 14, 0, 0),
        )
        result = format_trade_notification(data)

        assert "TRADE CLOSED" in result
        assert "GBP/USD" in result
        assert "SHORT" in result
        assert "2.0" in result
        assert "1.2700" in result
        assert "1.2650" in result
        assert "+100.00" in result
        assert "Mean Reversion" in result
        assert "✅" in result

    def test_format_trade_closed_loss(self) -> None:
        """Trade closed with loss shows negative PnL."""
        data = TradeNotificationData(
            instrument="USD/JPY",
            direction=TradeDirection.LONG,
            size=Decimal("0.5"),
            entry_price=Decimal("150.00"),
            exit_price=Decimal("149.50"),
            pnl=Decimal("-50.00"),
            strategy="Breakout",
            timestamp=datetime(2024, 1, 15, 16, 0, 0),
        )
        result = format_trade_notification(data)

        assert "TRADE CLOSED" in result
        assert "-50.00" in result
        assert "❌" in result

    def test_format_trade_opened_uses_current_time_if_none(self) -> None:
        """If no timestamp provided, uses current UTC time."""
        data = TradeNotificationData(
            instrument="XAU/USD",
            direction=TradeDirection.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("2000.00"),
            strategy="Momentum",
        )
        result = format_trade_notification(data)
        assert "UTC" in result


# ---------------------------------------------------------------------------
# Kill Switch Notification Formatting Tests (Req 17.3)
# ---------------------------------------------------------------------------


class TestKillSwitchNotificationFormatting:
    """Tests for kill switch notification formatting."""

    def test_format_kill_switch_basic(self) -> None:
        """Kill switch notification includes reason and position count."""
        data = KillSwitchNotificationData(
            activation_reason="Drawdown exceeded 15% from peak equity",
            positions_being_closed=[
                {"instrument": "EUR/USD", "direction": "LONG", "size": Decimal("1.0")},
                {"instrument": "GBP/USD", "direction": "SHORT", "size": Decimal("0.5")},
            ],
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )
        result = format_kill_switch_notification(data)

        assert "KILL SWITCH ACTIVATED" in result
        assert "Drawdown exceeded 15% from peak equity" in result
        assert "Positions Being Closed: 2" in result
        assert "EUR/USD" in result
        assert "GBP/USD" in result
        assert "ALL TRADING HALTED" in result

    def test_format_kill_switch_many_positions(self) -> None:
        """Kill switch with >10 positions truncates the list."""
        positions = [
            {"instrument": f"INST_{i}", "direction": "LONG", "size": Decimal("1.0")}
            for i in range(15)
        ]
        data = KillSwitchNotificationData(
            activation_reason="VIX exceeded 3 standard deviations",
            positions_being_closed=positions,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )
        result = format_kill_switch_notification(data)

        assert "Positions Being Closed: 15" in result
        assert "and 5 more" in result

    def test_format_kill_switch_empty_positions(self) -> None:
        """Kill switch with no positions still formats correctly."""
        data = KillSwitchNotificationData(
            activation_reason="Portfolio loss exceeded 20% in 24h",
            positions_being_closed=[],
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )
        result = format_kill_switch_notification(data)

        assert "Positions Being Closed: 0" in result
        assert "KILL SWITCH ACTIVATED" in result


# ---------------------------------------------------------------------------
# HFT Circuit Breaker Notification Tests (Req 22.9)
# ---------------------------------------------------------------------------


class TestHFTCircuitBreakerFormatting:
    """Tests for HFT circuit breaker notification formatting."""

    def test_format_hft_circuit_breaker(self) -> None:
        """HFT circuit breaker notification includes PnL loss and breaker count."""
        data = HFTCircuitBreakerData(
            pnl_loss=Decimal("-500.00"),
            account_equity=Decimal("100000.00"),
            breaker_count=1,
            window_duration_seconds=60,
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
        )
        result = format_hft_circuit_breaker_notification(data)

        assert "HFT CIRCUIT BREAKER ACTIVATED" in result
        assert "-500.00" in result
        assert "100000.00" in result
        assert "1/3" in result
        assert "60s" in result

    def test_format_hft_circuit_breaker_third_activation(self) -> None:
        """Third activation shows 3/3 count."""
        data = HFTCircuitBreakerData(
            pnl_loss=Decimal("-750.00"),
            account_equity=Decimal("100000.00"),
            breaker_count=3,
            timestamp=datetime(2024, 1, 15, 15, 0, 0),
        )
        result = format_hft_circuit_breaker_notification(data)
        assert "3/3" in result

    def test_format_hft_mode_disabled(self) -> None:
        """HFT mode disabled notification includes escalation details."""
        data = HFTModeDisabledData(
            breaker_count=3,
            time_window_minutes=60,
            timestamp=datetime(2024, 1, 15, 15, 0, 0),
        )
        result = format_hft_mode_disabled_notification(data)

        assert "HFT MODE DISABLED" in result
        assert "3" in result
        assert "Manual re-enablement" in result


# ---------------------------------------------------------------------------
# Crisis Alert Notification Tests (Req 23.7)
# ---------------------------------------------------------------------------


class TestCrisisAlertFormatting:
    """Tests for crisis alert notification formatting."""

    def test_format_crisis_alert(self) -> None:
        """Crisis alert includes region, sentiment, articles, and instruments."""
        data = CrisisAlertData(
            region="Middle East",
            sentiment_avg=-0.85,
            article_count=5,
            affected_instruments=["XAU/USD", "OIL/USD", "USD/SAR"],
            action_taken="Reducing portfolio exposure by 50%",
            timestamp=datetime(2024, 1, 15, 8, 0, 0),
        )
        result = format_crisis_alert_notification(data)

        assert "NEWS CRISIS ALERT" in result
        assert "Middle East" in result
        assert "-0.85" in result
        assert "5" in result
        assert "XAU/USD" in result
        assert "Reducing portfolio exposure by 50%" in result

    def test_format_crisis_alert_many_instruments(self) -> None:
        """Crisis alert with >5 instruments truncates the list."""
        data = CrisisAlertData(
            region="Europe",
            sentiment_avg=-0.75,
            article_count=3,
            affected_instruments=[f"INST_{i}" for i in range(8)],
            action_taken="Widening stops by 2.0 × ATR",
        )
        result = format_crisis_alert_notification(data)

        assert "+3 more" in result


# ---------------------------------------------------------------------------
# Retry Logic Tests (Req 17.4)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for notification delivery retry logic."""

    @pytest.mark.asyncio
    async def test_successful_delivery_no_retry(self) -> None:
        """Successful delivery on first attempt doesn't retry."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
            ),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is True
        assert result.channel_name == "telegram"
        assert result.attempts == 1
        assert channel.send_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_succeed(self) -> None:
        """Retries on failure and succeeds on subsequent attempt."""
        channel = FailThenSucceedChannel("telegram", fail_count=2)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
            ),
            retry_config=RetryConfig(max_retries=3, interval_seconds=0.01),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is True
        assert result.attempts == 3
        assert channel.send_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_permanent_failure(self) -> None:
        """After 3 failed retries, reports permanent failure."""
        channel = MockChannel("telegram", should_succeed=False)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
            ),
            retry_config=RetryConfig(max_retries=3, interval_seconds=0.01),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is False
        assert result.attempts == 3
        assert channel.send_count == 3

    @pytest.mark.asyncio
    async def test_fallback_to_next_channel_on_permanent_failure(self) -> None:
        """Falls back to next channel when first channel permanently fails."""
        failing_channel = MockChannel("telegram", should_succeed=False)
        backup_channel = MockChannel("discord", should_succeed=True)

        service = NotificationService(
            channels=[failing_channel, backup_channel],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.RISK_ALERT: ["telegram", "discord"]
                }
            ),
            retry_config=RetryConfig(max_retries=3, interval_seconds=0.01),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is True
        assert result.channel_name == "discord"
        # Telegram tried 3 times, Discord tried 1 time
        assert failing_channel.send_count == 3
        assert backup_channel.send_count == 1

    @pytest.mark.asyncio
    async def test_fallback_all_channels_fail(self) -> None:
        """When all channels fail after retries, returns failure with fallback flag."""
        ch1 = MockChannel("telegram", should_succeed=False)
        ch2 = MockChannel("discord", should_succeed=False)
        ch3 = MockChannel("email", should_succeed=False)

        service = NotificationService(
            channels=[ch1, ch2, ch3],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.RISK_ALERT: ["telegram", "discord", "email"]
                }
            ),
            retry_config=RetryConfig(max_retries=3, interval_seconds=0.01),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is False
        assert result.fallback_used is True
        # Each channel tried 3 times
        assert ch1.send_count == 3
        assert ch2.send_count == 3
        assert ch3.send_count == 3

    @pytest.mark.asyncio
    async def test_retry_handles_exceptions(self) -> None:
        """Retry logic handles channel exceptions gracefully."""
        exception_channel = ExceptionChannel("telegram")
        backup_channel = MockChannel("discord", should_succeed=True)

        service = NotificationService(
            channels=[exception_channel, backup_channel],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.RISK_ALERT: ["telegram", "discord"]
                }
            ),
            retry_config=RetryConfig(max_retries=3, interval_seconds=0.01),
        )

        result = await service.send(
            "Test message",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is True
        assert result.channel_name == "discord"
        assert exception_channel.send_count == 3


# ---------------------------------------------------------------------------
# Kill Switch Notification to ALL Channels Tests (Req 17.3)
# ---------------------------------------------------------------------------


class TestKillSwitchNotificationDelivery:
    """Tests for kill switch notification delivery to all channels."""

    @pytest.mark.asyncio
    async def test_kill_switch_sends_to_all_channels(self) -> None:
        """Kill switch notification is sent to ALL configured channels."""
        ch1 = MockChannel("telegram", should_succeed=True)
        ch2 = MockChannel("discord", should_succeed=True)
        ch3 = MockChannel("email", should_succeed=True)

        service = NotificationService(channels=[ch1, ch2, ch3])

        data = KillSwitchNotificationData(
            activation_reason="Drawdown exceeded 15%",
            positions_being_closed=[
                {"instrument": "EUR/USD", "direction": "LONG", "size": Decimal("1.0")},
            ],
        )

        results = await service.send_kill_switch_notification(data)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert ch1.send_count == 1
        assert ch2.send_count == 1
        assert ch3.send_count == 1

    @pytest.mark.asyncio
    async def test_kill_switch_partial_failure(self) -> None:
        """Kill switch still delivers to working channels if one fails."""
        ch1 = MockChannel("telegram", should_succeed=True)
        ch2 = MockChannel("discord", should_succeed=False)
        ch3 = MockChannel("email", should_succeed=True)

        service = NotificationService(channels=[ch1, ch2, ch3])

        data = KillSwitchNotificationData(
            activation_reason="VIX spike",
            positions_being_closed=[],
        )

        results = await service.send_kill_switch_notification(data)

        assert len(results) == 3
        successful = [r for r in results if r.success]
        assert len(successful) == 2

    @pytest.mark.asyncio
    async def test_kill_switch_timeout_within_5_seconds(self) -> None:
        """Kill switch notification respects 5-second timeout."""
        ch1 = MockChannel("telegram", should_succeed=True)
        ch2 = SlowChannel("discord", delay=10.0)  # Will timeout

        service = NotificationService(channels=[ch1, ch2])

        data = KillSwitchNotificationData(
            activation_reason="Test timeout",
            positions_being_closed=[],
        )

        start = asyncio.get_event_loop().time()
        results = await service.send_kill_switch_notification(data)
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete within ~5 seconds (with some tolerance)
        assert elapsed < 6.0


# ---------------------------------------------------------------------------
# Trade Notification Delivery Tests (Req 17.2)
# ---------------------------------------------------------------------------


class TestTradeNotificationDelivery:
    """Tests for trade notification delivery within 10 seconds."""

    @pytest.mark.asyncio
    async def test_trade_notification_delivery(self) -> None:
        """Trade notification is delivered successfully."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.TRADE_OPENED: ["telegram"]}
            ),
        )

        data = TradeNotificationData(
            instrument="EUR/USD",
            direction=TradeDirection.LONG,
            size=Decimal("1.0"),
            entry_price=Decimal("1.0850"),
            strategy="Trend Following",
        )

        result = await service.send_trade_notification(data)

        assert result.success is True
        assert "EUR/USD" in channel.sent_messages[0]
        assert "LONG" in channel.sent_messages[0]

    @pytest.mark.asyncio
    async def test_trade_notification_fallback_on_failure(self) -> None:
        """Trade notification falls back to next channel on failure."""
        ch1 = MockChannel("telegram", should_succeed=False)
        ch2 = MockChannel("discord", should_succeed=True)

        service = NotificationService(
            channels=[ch1, ch2],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.TRADE_OPENED: ["telegram", "discord"]
                }
            ),
        )

        data = TradeNotificationData(
            instrument="GBP/USD",
            direction=TradeDirection.SHORT,
            size=Decimal("0.5"),
            entry_price=Decimal("1.2700"),
            strategy="Scalping",
        )

        result = await service.send_trade_notification(data)

        assert result.success is True
        assert result.channel_name == "discord"


# ---------------------------------------------------------------------------
# HFT Circuit Breaker Notification Delivery Tests (Req 22.9)
# ---------------------------------------------------------------------------


class TestHFTNotificationDelivery:
    """Tests for HFT circuit breaker notification delivery."""

    @pytest.mark.asyncio
    async def test_hft_circuit_breaker_notification(self) -> None:
        """HFT circuit breaker notification is delivered."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.HFT_CIRCUIT_BREAKER: ["telegram"]
                }
            ),
        )

        data = HFTCircuitBreakerData(
            pnl_loss=Decimal("-500.00"),
            account_equity=Decimal("100000.00"),
            breaker_count=1,
        )

        result = await service.send_hft_circuit_breaker_notification(data)

        assert result.success is True
        assert "HFT CIRCUIT BREAKER" in channel.sent_messages[0]

    @pytest.mark.asyncio
    async def test_hft_mode_disabled_notification(self) -> None:
        """HFT mode disabled notification is delivered."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={
                    NotificationType.HFT_MODE_DISABLED: ["telegram"]
                }
            ),
        )

        data = HFTModeDisabledData(breaker_count=3)

        result = await service.send_hft_mode_disabled_notification(data)

        assert result.success is True
        assert "HFT MODE DISABLED" in channel.sent_messages[0]


# ---------------------------------------------------------------------------
# Crisis Alert Notification Delivery Tests (Req 23.7)
# ---------------------------------------------------------------------------


class TestCrisisAlertDelivery:
    """Tests for crisis alert notification delivery."""

    @pytest.mark.asyncio
    async def test_crisis_alert_notification(self) -> None:
        """Crisis alert notification is delivered."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.CRISIS_ALERT: ["telegram"]}
            ),
        )

        data = CrisisAlertData(
            region="Middle East",
            sentiment_avg=-0.85,
            article_count=5,
            affected_instruments=["XAU/USD", "OIL/USD"],
            action_taken="Reducing exposure by 50%",
        )

        result = await service.send_crisis_alert_notification(data)

        assert result.success is True
        assert "NEWS CRISIS ALERT" in channel.sent_messages[0]
        assert "Middle East" in channel.sent_messages[0]


# ---------------------------------------------------------------------------
# send_alert Protocol Compatibility Tests
# ---------------------------------------------------------------------------


class TestSendAlertProtocol:
    """Tests for send_alert method (NotificationServiceProtocol compatibility)."""

    @pytest.mark.asyncio
    async def test_send_alert_info_level(self) -> None:
        """send_alert with info level routes to RISK_ALERT type."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
            ),
        )

        await service.send_alert("Test info alert", level="info")

        assert channel.send_count == 1
        assert "Test info alert" in channel.sent_messages[0]

    @pytest.mark.asyncio
    async def test_send_alert_critical_level(self) -> None:
        """send_alert with critical level routes to SYSTEM_ERROR type."""
        channel = MockChannel("telegram", should_succeed=True)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.SYSTEM_ERROR: ["telegram"]}
            ),
        )

        await service.send_alert("Critical failure", level="critical")

        assert channel.send_count == 1


# ---------------------------------------------------------------------------
# Edge Cases and Configuration Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and configuration validation."""

    @pytest.mark.asyncio
    async def test_no_channels_configured_discards_notification(self) -> None:
        """Notification is discarded when no channels are configured for its type."""
        service = NotificationService(
            channels=[],
            channel_config=ChannelConfig(type_to_channels={}),
        )

        result = await service.send(
            "Test",
            notification_type=NotificationType.TRADE_OPENED,
        )

        assert result.success is False
        assert "No channels configured" in (result.error or "")

    @pytest.mark.asyncio
    async def test_kill_switch_with_no_channels(self) -> None:
        """Kill switch notification with no channels returns empty list."""
        service = NotificationService(channels=[])

        data = KillSwitchNotificationData(
            activation_reason="Test",
            positions_being_closed=[],
        )

        results = await service.send_kill_switch_notification(data)
        assert results == []

    @pytest.mark.asyncio
    async def test_retry_config_respected(self) -> None:
        """Custom retry config is respected."""
        channel = MockChannel("telegram", should_succeed=False)
        service = NotificationService(
            channels=[channel],
            channel_config=ChannelConfig(
                type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
            ),
            retry_config=RetryConfig(max_retries=2, interval_seconds=0.01),
        )

        result = await service.send(
            "Test",
            notification_type=NotificationType.RISK_ALERT,
        )

        assert result.success is False
        # Only 2 retries (custom config)
        assert channel.send_count == 2
        assert result.attempts == 2
