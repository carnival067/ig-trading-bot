"""Unit tests for the NotificationService.

Tests channel routing, priority handling, per-notification-type configuration,
retry logic, fallback delivery, and delivery status tracking.

Requirements: 17.1
"""

from __future__ import annotations

import pytest

from src.notifications.notification_service import (
    ChannelConfig,
    ChannelDeliveryStatus,
    DeliveryStatus,
    NotificationChannel,
    NotificationRecord,
    NotificationService,
    NotificationType,
    Priority,
    RetryConfig,
)


class MockChannel:
    """A mock notification channel for testing."""

    def __init__(self, channel_name: str, should_fail: bool = False) -> None:
        self._name = channel_name
        self.should_fail = should_fail
        self.sent_messages: list[str] = []
        self.send_count = 0
        self.fail_count = 0  # Number of times to fail before succeeding
        self._fail_counter = 0

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        if self.should_fail:
            return False
        if self._fail_counter < self.fail_count:
            self._fail_counter += 1
            return False
        self.sent_messages.append(message)
        return True


class ExceptionChannel:
    """A channel that raises exceptions."""

    def __init__(self, channel_name: str) -> None:
        self._name = channel_name
        self.send_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def send(self, message: str) -> bool:
        self.send_count += 1
        raise ConnectionError(f"Failed to connect to {self._name}")


class TestNotificationServiceRegistration:
    """Tests for channel registration."""

    def test_register_single_channel(self):
        """Can register a single channel."""
        service = NotificationService()
        channel = MockChannel("telegram")
        service.register_channel(channel)

        assert "telegram" in service._channels
        assert service._channel_order == ["telegram"]

    def test_register_multiple_channels(self):
        """Channels are ordered by registration order."""
        service = NotificationService()
        service.register_channel(MockChannel("telegram"))
        service.register_channel(MockChannel("discord"))
        service.register_channel(MockChannel("email"))

        assert service._channel_order == ["telegram", "discord", "email"]

    def test_register_duplicate_channel_does_not_duplicate_order(self):
        """Re-registering a channel updates it but doesn't duplicate in order."""
        service = NotificationService()
        service.register_channel(MockChannel("telegram"))
        service.register_channel(MockChannel("telegram"))

        assert service._channel_order == ["telegram"]

    def test_register_via_constructor(self):
        """Channels can be registered via constructor."""
        channels = [MockChannel("telegram"), MockChannel("discord")]
        service = NotificationService(channels=channels)

        assert service._channel_order == ["telegram", "discord"]


class TestNotificationTypeConfiguration:
    """Tests for per-notification-type channel configuration."""

    def test_configure_type_to_specific_channels(self):
        """Can configure a notification type to specific channels."""
        service = NotificationService()
        service.configure_notification_type(
            NotificationType.KILL_SWITCH, ["telegram", "discord", "email"]
        )

        assert service._channel_config.type_to_channels[NotificationType.KILL_SWITCH] == [
            "telegram",
            "discord",
            "email",
        ]

    def test_configure_type_to_single_channel(self):
        """Can configure a notification type to a single channel."""
        service = NotificationService()
        service.configure_notification_type(
            NotificationType.TRADE_ALERT, ["telegram"]
        )

        assert service._channel_config.type_to_channels[NotificationType.TRADE_ALERT] == [
            "telegram"
        ]

    def test_configure_via_channel_config(self):
        """Can configure via ChannelConfig in constructor."""
        config = ChannelConfig(
            type_to_channels={
                NotificationType.KILL_SWITCH: ["telegram", "discord", "email"],
                NotificationType.TRADE_ALERT: ["telegram"],
            }
        )
        service = NotificationService(
            channels=[MockChannel("telegram"), MockChannel("discord"), MockChannel("email")],
            channel_config=config,
        )

        channels = service._get_channels_for_notification(
            NotificationType.KILL_SWITCH, Priority.HIGH
        )
        assert channels == ["telegram", "discord", "email"]


class TestPriorityRouting:
    """Tests for priority-based channel routing."""

    def _make_service(self) -> NotificationService:
        """Create a service with three registered channels and zero retry interval."""
        service = NotificationService(
            channels=[MockChannel("telegram"), MockChannel("discord"), MockChannel("email")],
            retry_config=RetryConfig(max_retries=3, interval_seconds=0),
        )
        return service

    @pytest.mark.asyncio
    async def test_critical_routes_to_all_channels(self):
        """CRITICAL priority sends to all registered channels."""
        service = self._make_service()

        record = await service.send(
            NotificationType.KILL_SWITCH,
            {"message": "Emergency!"},
            Priority.CRITICAL,
        )

        assert len(record.channel_statuses) == 3
        assert all(
            cs.status == DeliveryStatus.DELIVERED
            for cs in record.channel_statuses.values()
        )

    @pytest.mark.asyncio
    async def test_high_routes_to_primary_and_secondary(self):
        """HIGH priority sends to first two channels (primary + secondary)."""
        service = self._make_service()

        record = await service.send(
            NotificationType.CRISIS_ALERT,
            {"message": "Crisis detected"},
            Priority.HIGH,
        )

        # Should target telegram (primary) and discord (secondary)
        assert "telegram" in record.channel_statuses
        assert "discord" in record.channel_statuses
        assert "email" not in record.channel_statuses

    @pytest.mark.asyncio
    async def test_normal_routes_to_primary_only(self):
        """NORMAL priority sends to primary channel only."""
        service = self._make_service()

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Trade opened"},
            Priority.NORMAL,
        )

        assert "telegram" in record.channel_statuses
        assert len(record.channel_statuses) == 1

    @pytest.mark.asyncio
    async def test_critical_overrides_type_config(self):
        """CRITICAL priority sends to ALL channels even if type is configured for fewer."""
        config = ChannelConfig(
            type_to_channels={NotificationType.TRADE_ALERT: ["telegram"]}
        )
        service = NotificationService(
            channels=[MockChannel("telegram"), MockChannel("discord"), MockChannel("email")],
            channel_config=config,
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Important trade"},
            Priority.CRITICAL,
        )

        # CRITICAL overrides type config - sends to all
        assert len(record.channel_statuses) == 3


class TestPerTypeChannelRouting:
    """Tests for per-notification-type channel configuration routing."""

    @pytest.mark.asyncio
    async def test_type_config_routes_to_configured_channels(self):
        """Notification type config determines which channels receive the message."""
        telegram = MockChannel("telegram")
        discord = MockChannel("discord")
        email = MockChannel("email")

        config = ChannelConfig(
            type_to_channels={
                NotificationType.KILL_SWITCH: ["telegram", "discord", "email"],
                NotificationType.TRADE_ALERT: ["telegram"],
            }
        )
        service = NotificationService(
            channels=[telegram, discord, email],
            channel_config=config,
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        # Send trade alert at HIGH priority - should still only go to telegram
        # because type config limits to telegram
        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Trade opened"},
            Priority.HIGH,
        )

        assert "telegram" in record.channel_statuses
        assert len(record.channel_statuses) == 1
        assert telegram.send_count == 1
        assert discord.send_count == 0

    @pytest.mark.asyncio
    async def test_type_config_normal_uses_first_configured(self):
        """NORMAL priority with type config uses only the first configured channel."""
        telegram = MockChannel("telegram")
        discord = MockChannel("discord")

        config = ChannelConfig(
            type_to_channels={
                NotificationType.CRISIS_ALERT: ["discord", "telegram"],
            }
        )
        service = NotificationService(
            channels=[telegram, discord],
            channel_config=config,
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.CRISIS_ALERT,
            {"message": "Crisis"},
            Priority.NORMAL,
        )

        # NORMAL with type config → first configured channel only
        assert "discord" in record.channel_statuses
        assert len(record.channel_statuses) == 1

    @pytest.mark.asyncio
    async def test_unregistered_channels_in_config_are_filtered(self):
        """Channels in type config that aren't registered are filtered out."""
        telegram = MockChannel("telegram")

        config = ChannelConfig(
            type_to_channels={
                NotificationType.TRADE_ALERT: ["telegram", "slack", "sms"],
            }
        )
        service = NotificationService(
            channels=[telegram],
            channel_config=config,
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Trade"},
            Priority.HIGH,
        )

        # Only telegram is registered, so only telegram gets the message
        assert "telegram" in record.channel_statuses
        assert len(record.channel_statuses) == 1


class TestRetryLogic:
    """Tests for retry and fallback delivery."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Service retries delivery up to max_retries times."""
        failing_channel = MockChannel("telegram", should_fail=True)
        service = NotificationService(
            channels=[failing_channel],
            retry_config=RetryConfig(max_retries=3, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        assert failing_channel.send_count == 3
        assert record.channel_statuses["telegram"].status == DeliveryStatus.FAILED
        assert record.channel_statuses["telegram"].attempts == 3

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        """Service succeeds if channel works after initial failures."""
        channel = MockChannel("telegram")
        channel.fail_count = 2  # Fail twice, then succeed
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=3, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        assert channel.send_count == 3
        assert record.channel_statuses["telegram"].status == DeliveryStatus.DELIVERED
        assert record.channel_statuses["telegram"].attempts == 3

    @pytest.mark.asyncio
    async def test_fallback_to_next_channel_on_failure(self):
        """After exhausting retries, falls back to next available channel."""
        failing = MockChannel("telegram", should_fail=True)
        backup = MockChannel("discord")
        service = NotificationService(
            channels=[failing, backup],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        # Primary failed, fallback to discord
        assert record.channel_statuses["telegram"].status == DeliveryStatus.FAILED
        assert "discord" in record.channel_statuses
        assert record.channel_statuses["discord"].status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_handles_exception_in_channel(self):
        """Service handles exceptions from channel send gracefully."""
        exception_channel = ExceptionChannel("telegram")
        backup = MockChannel("discord")
        service = NotificationService(
            channels=[exception_channel, backup],
            retry_config=RetryConfig(max_retries=2, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        assert record.channel_statuses["telegram"].status == DeliveryStatus.FAILED
        assert record.channel_statuses["telegram"].error is not None
        assert "discord" in record.channel_statuses
        assert record.channel_statuses["discord"].status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_no_fallback_when_all_channels_targeted(self):
        """Fallback doesn't target a channel that's already in the target list."""
        failing_telegram = MockChannel("telegram", should_fail=True)
        failing_discord = MockChannel("discord", should_fail=True)
        service = NotificationService(
            channels=[failing_telegram, failing_discord],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        # HIGH targets both telegram and discord
        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.HIGH,
        )

        # Both targeted channels failed, no other fallback available
        assert record.channel_statuses["telegram"].status == DeliveryStatus.FAILED
        assert record.channel_statuses["discord"].status == DeliveryStatus.FAILED
        assert record.all_failed


class TestDeliveryStatusTracking:
    """Tests for delivery status tracking."""

    @pytest.mark.asyncio
    async def test_tracks_delivery_record(self):
        """Delivery records are stored and retrievable."""
        service = NotificationService(
            channels=[MockChannel("telegram")],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        retrieved = service.get_delivery_status(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.notification_type == NotificationType.TRADE_ALERT
        assert retrieved.priority == Priority.NORMAL
        assert retrieved.is_fully_delivered

    @pytest.mark.asyncio
    async def test_is_fully_delivered_true_when_at_least_one_succeeds(self):
        """is_fully_delivered is True if at least one channel delivered."""
        service = NotificationService(
            channels=[MockChannel("telegram", should_fail=True), MockChannel("discord")],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.CRISIS_ALERT,
            {"message": "Crisis"},
            Priority.CRITICAL,
        )

        assert record.is_fully_delivered

    @pytest.mark.asyncio
    async def test_all_failed_when_no_delivery_succeeds(self):
        """all_failed is True when all channels fail."""
        service = NotificationService(
            channels=[MockChannel("telegram", should_fail=True)],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        # Telegram failed, fallback also not available (only one channel)
        assert record.all_failed

    @pytest.mark.asyncio
    async def test_get_delivery_status_returns_none_for_unknown(self):
        """Returns None for unknown notification ID."""
        service = NotificationService()
        assert service.get_delivery_status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_record_has_created_at_timestamp(self):
        """Notification record has a creation timestamp."""
        from datetime import timezone

        service = NotificationService(
            channels=[MockChannel("telegram")],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        assert record.created_at is not None
        assert record.created_at.tzinfo == timezone.utc


class TestNoChannelsConfigured:
    """Tests for behavior when no channels are configured."""

    @pytest.mark.asyncio
    async def test_no_channels_registered_discards_notification(self):
        """When no channels are registered, notification is discarded."""
        service = NotificationService(
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        record = await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Test"},
            Priority.NORMAL,
        )

        assert len(record.channel_statuses) == 0
        assert record.all_failed

    def test_startup_warnings_for_unconfigured_types(self):
        """Startup check warns about notification types with no channels."""
        service = NotificationService()
        # No channels registered at all
        warnings = service.check_startup_warnings()

        # Should have warnings for all notification types
        assert len(warnings) == len(NotificationType)

    def test_startup_warnings_none_when_channels_registered(self):
        """No warnings when channels are registered (default routing works)."""
        service = NotificationService(channels=[MockChannel("telegram")])

        warnings = service.check_startup_warnings()
        assert len(warnings) == 0

    def test_startup_warnings_for_unregistered_configured_channels(self):
        """Warns when type config references channels that aren't registered."""
        config = ChannelConfig(
            type_to_channels={NotificationType.TRADE_ALERT: ["slack"]}
        )
        service = NotificationService(
            channels=[MockChannel("telegram")],
            channel_config=config,
        )

        warnings = service.check_startup_warnings()
        # Should warn about TRADE_ALERT having unregistered channels
        assert any("trade_alert" in w for w in warnings)


class TestMessageFormatting:
    """Tests for notification message formatting."""

    @pytest.mark.asyncio
    async def test_trade_alert_formatting(self):
        """Trade alerts are formatted with instrument, direction, size, price, strategy."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.TRADE_ALERT,
            {
                "instrument": "EUR/USD",
                "direction": "BUY",
                "size": "1.0",
                "price": "1.0850",
                "strategy": "TrendFollowing",
                "pnl": "+150.00",
            },
            Priority.NORMAL,
        )

        msg = channel.sent_messages[0]
        assert "EUR/USD" in msg
        assert "BUY" in msg
        assert "1.0" in msg
        assert "1.0850" in msg
        assert "TrendFollowing" in msg
        assert "+150.00" in msg

    @pytest.mark.asyncio
    async def test_kill_switch_formatting(self):
        """Kill switch alerts include reason and positions count."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.KILL_SWITCH,
            {"reason": "VIX spike", "positions_count": 5},
            Priority.CRITICAL,
        )

        msg = channel.sent_messages[0]
        assert "KILL SWITCH" in msg
        assert "VIX spike" in msg
        assert "5" in msg

    @pytest.mark.asyncio
    async def test_crisis_alert_formatting(self):
        """Crisis alerts include region, sentiment, and affected instruments."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.CRISIS_ALERT,
            {
                "region": "Europe",
                "sentiment_avg": -0.85,
                "affected_instruments": ["EUR/USD", "DAX"],
            },
            Priority.HIGH,
        )

        msg = channel.sent_messages[0]
        assert "CRISIS" in msg
        assert "Europe" in msg
        assert "EUR/USD" in msg

    @pytest.mark.asyncio
    async def test_hft_circuit_breaker_formatting(self):
        """HFT circuit breaker alerts include PnL and activation count."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.HFT_CIRCUIT_BREAKER,
            {"pnl": "-500.00", "breaker_count": 2},
            Priority.HIGH,
        )

        msg = channel.sent_messages[0]
        assert "Circuit Breaker" in msg
        assert "-500.00" in msg
        assert "#2" in msg

    @pytest.mark.asyncio
    async def test_strategy_disabled_formatting(self):
        """Strategy disabled alerts include strategy name and reason."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.STRATEGY_DISABLED,
            {"strategy": "MeanReversion", "reason": "Sharpe < 0.5"},
            Priority.NORMAL,
        )

        msg = channel.sent_messages[0]
        assert "Strategy Disabled" in msg
        assert "MeanReversion" in msg
        assert "Sharpe < 0.5" in msg

    @pytest.mark.asyncio
    async def test_custom_message_field_used_directly(self):
        """If payload has a 'message' field, it's used directly."""
        channel = MockChannel("telegram")
        service = NotificationService(
            channels=[channel],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        await service.send(
            NotificationType.TRADE_ALERT,
            {"message": "Custom formatted message"},
            Priority.NORMAL,
        )

        assert channel.sent_messages[0] == "Custom formatted message"


class TestNotificationEnums:
    """Tests for enum values."""

    def test_notification_type_values(self):
        """NotificationType has all required values."""
        assert NotificationType.TRADE_ALERT.value == "trade_alert"
        assert NotificationType.KILL_SWITCH.value == "kill_switch"
        assert NotificationType.CRISIS_ALERT.value == "crisis_alert"
        assert NotificationType.HFT_CIRCUIT_BREAKER.value == "hft_circuit_breaker"
        assert NotificationType.STRATEGY_DISABLED.value == "strategy_disabled"

    def test_priority_values(self):
        """Priority has all required values."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.NORMAL.value == "normal"

    def test_delivery_status_values(self):
        """DeliveryStatus has all required values."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.FAILED.value == "failed"
        assert DeliveryStatus.RETRYING.value == "retrying"


class TestSendToAllChannels:
    """Tests for the send_to_all_channels convenience method."""

    @pytest.mark.asyncio
    async def test_sends_to_all_registered_channels(self):
        """send_to_all_channels delivers to every registered channel."""
        telegram = MockChannel("telegram")
        discord = MockChannel("discord")
        service = NotificationService(
            channels=[telegram, discord],
            retry_config=RetryConfig(max_retries=1, interval_seconds=0),
        )

        results = await service.send_to_all_channels("Emergency message")

        assert len(results) == 2
        assert all(r.success for r in results)
        assert telegram.sent_messages == ["Emergency message"]
        assert discord.sent_messages == ["Emergency message"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_channels(self):
        """send_to_all_channels returns empty list when no channels registered."""
        service = NotificationService()
        results = await service.send_to_all_channels("Test")
        assert results == []
