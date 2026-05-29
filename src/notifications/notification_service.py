"""Notification service orchestrator with multi-channel routing and retry logic.

Implements:
- Channel routing with per-notification-type configuration
- Priority-based routing: CRITICAL (all channels), HIGH (primary + secondary),
  NORMAL (primary only) (Req 17.1)
- Retry logic: 3 retries, 30-second intervals, fallback to next channel (Req 17.4)
- Trade notifications within 10 seconds (Req 17.2)
- Kill switch notification to ALL channels within 5 seconds (Req 17.3)
- HFT circuit breaker and crisis alert notifications (Req 22.9, 23.7)
- Delivery status tracking per notification
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from src.notifications.formatters import (
    CrisisAlertData,
    HFTCircuitBreakerData,
    HFTModeDisabledData,
    KillSwitchNotificationData,
    NotificationType,
    Priority,
    TradeNotificationData,
    format_crisis_alert_notification,
    format_hft_circuit_breaker_notification,
    format_hft_mode_disabled_notification,
    format_kill_switch_notification,
    format_trade_notification,
)

logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Status of a notification delivery attempt."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class NotificationChannel(Protocol):
    """Protocol for notification delivery channels."""

    @property
    def name(self) -> str: ...

    async def send(self, message: str) -> bool: ...


@dataclass
class RetryConfig:
    """Configuration for notification retry behavior."""

    max_retries: int = 3
    interval_seconds: float = 30.0


@dataclass
class ChannelDeliveryStatus:
    """Tracks delivery status for a single channel."""

    channel_name: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    last_attempt_at: datetime | None = None
    delivered_at: datetime | None = None
    error: str | None = None


@dataclass
class NotificationRecord:
    """Tracks the full delivery status of a notification across all channels."""

    id: str
    notification_type: NotificationType
    priority: Priority
    payload: dict[str, Any]
    created_at: datetime
    channel_statuses: dict[str, ChannelDeliveryStatus] = field(default_factory=dict)

    @property
    def is_fully_delivered(self) -> bool:
        """Check if notification was delivered to at least one channel."""
        return any(
            cs.status == DeliveryStatus.DELIVERED for cs in self.channel_statuses.values()
        )

    @property
    def all_failed(self) -> bool:
        """Check if all delivery attempts have failed."""
        if not self.channel_statuses:
            return True
        return all(
            cs.status == DeliveryStatus.FAILED for cs in self.channel_statuses.values()
        )


@dataclass
class DeliveryResult:
    """Simple result of a notification delivery attempt."""

    success: bool
    channel_name: str
    attempts: int = 1
    error: str | None = None
    fallback_used: bool = False


@dataclass
class ChannelConfig:
    """Configuration mapping notification types to channels."""

    # Maps notification type to list of channel names (in priority order)
    type_to_channels: dict[NotificationType, list[str]] = field(default_factory=dict)


class NotificationService:
    """Multi-channel notification service with priority routing, retry, and fallback.

    Routes notifications to configured channels based on notification type and priority.
    Implements retry logic with 30-second intervals and fallback to next channel
    on permanent failure.

    Priority routing:
    - CRITICAL: All registered channels
    - HIGH: Primary + secondary channels (first two in registration order)
    - NORMAL: Primary channel only (first registered)

    Per-notification-type configuration overrides default priority routing
    (except CRITICAL which always sends to all channels).
    """

    def __init__(
        self,
        channels: list[NotificationChannel] | None = None,
        channel_config: ChannelConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._channels: dict[str, NotificationChannel] = {}
        self._channel_order: list[str] = []
        self._channel_config = channel_config or ChannelConfig()
        self._retry_config = retry_config or RetryConfig()
        self._delivery_records: dict[str, NotificationRecord] = {}

        if channels:
            for ch in channels:
                self.register_channel(ch)

        self._validate_config()

    def register_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel handler.

        Channels are ordered by registration order: first registered is primary,
        second is secondary, etc.

        Args:
            channel: The channel handler to register.
        """
        self._channels[channel.name] = channel
        if channel.name not in self._channel_order:
            self._channel_order.append(channel.name)

    def configure_notification_type(
        self, notification_type: NotificationType, channels: list[str]
    ) -> None:
        """Configure which channels a notification type should be routed to.

        Args:
            notification_type: The type of notification to configure.
            channels: List of channel names to route this type to.
        """
        self._channel_config.type_to_channels[notification_type] = channels

    def _validate_config(self) -> None:
        """Validate channel configuration at startup.

        Logs warnings for notification types with no configured channels (Req 17.5).
        """
        for ntype in NotificationType:
            configured = self._channel_config.type_to_channels.get(ntype)
            if configured:
                available = [ch for ch in configured if ch in self._channels]
                if not available and self._channels:
                    logger.warning(
                        "Configured channels for '%s' are not registered. "
                        "Notifications of this type will be discarded.",
                        ntype.value,
                    )
            elif not self._channel_order:
                logger.warning(
                    "No delivery channel configured for notification type: %s. "
                    "Notifications of this type will be discarded.",
                    ntype.value,
                )

    def check_startup_warnings(self) -> list[str]:
        """Check for missing channel configurations at startup.

        Returns a list of warning messages for notification types
        that have no configured channels.

        Returns:
            List of warning messages.
        """
        warnings = []
        for ntype in NotificationType:
            configured = self._channel_config.type_to_channels.get(ntype)
            if configured:
                available = [ch for ch in configured if ch in self._channels]
                if not available:
                    msg = (
                        f"Configured channels for '{ntype.value}' are not registered. "
                        f"Notifications of this type will be discarded."
                    )
                    warnings.append(msg)
                    logger.warning(msg)
            elif not self._channel_order:
                msg = (
                    f"No delivery channel configured for notification type "
                    f"'{ntype.value}'. Notifications of this type will be discarded."
                )
                warnings.append(msg)
                logger.warning(msg)
        return warnings

    def _get_channels_for_notification(
        self, notification_type: NotificationType, priority: Priority
    ) -> list[str]:
        """Determine which channels to use based on type config and priority.

        Priority routing:
        - CRITICAL: All registered channels (overrides type config)
        - HIGH: Primary + secondary channels (or all configured for type)
        - NORMAL/LOW: Primary channel only (or first configured for type)

        If a per-type configuration exists, it takes precedence for NORMAL and HIGH.
        CRITICAL always sends to all channels regardless of type config.
        """
        if priority == Priority.CRITICAL:
            return list(self._channel_order)

        # Check per-type configuration
        configured = self._channel_config.type_to_channels.get(notification_type)
        if configured:
            available = [ch for ch in configured if ch in self._channels]
            if priority == Priority.HIGH:
                return available
            else:  # NORMAL or LOW
                return available[:1] if available else []

        # Default priority-based routing when no type config exists
        if priority == Priority.HIGH:
            return self._channel_order[:2]
        else:  # NORMAL or LOW
            return self._channel_order[:1]

    async def send(
        self,
        message_or_type: NotificationType | str,
        payload_or_type: dict[str, Any] | str | NotificationType = "",
        priority: Priority = Priority.NORMAL,
        *,
        notification_type: NotificationType | None = None,
    ) -> NotificationRecord | DeliveryResult:
        """Send a notification routed to appropriate channels.

        Supports two call signatures:
        - New: send(NotificationType.X, {"key": "value"}, Priority.HIGH)
        - Old: send("message", notification_type=NotificationType.X, priority=...)

        Args:
            message_or_type: Either a NotificationType enum or a message string.
            payload_or_type: Either a payload dict/string, or a NotificationType.
            priority: Priority level determining channel routing.
            notification_type: Keyword-only arg for old-style calls.

        Returns:
            NotificationRecord (new style) or DeliveryResult (old style).
        """
        # Old-style keyword call: send("message", notification_type=..., priority=...)
        if notification_type is not None:
            actual_type = notification_type
            actual_payload: dict[str, Any] = {"message": str(message_or_type)}
            # Old-style calls use fallback delivery (try all configured channels)
            record = await self._send_internal(
                actual_type, actual_payload, Priority.HIGH
            )
            return self._record_to_delivery_result(record)

        # Old-style positional call: send("message_str", NotificationType.X, priority)
        if isinstance(message_or_type, str) and isinstance(
            payload_or_type, NotificationType
        ):
            actual_type = payload_or_type
            actual_payload = {"message": message_or_type}
            # Old-style calls use fallback delivery (try all configured channels)
            record = await self._send_internal(actual_type, actual_payload, Priority.HIGH)
            return self._record_to_delivery_result(record)

        # New-style call: send(NotificationType.X, {"key": "value"}, priority)
        if isinstance(message_or_type, NotificationType):
            actual_type = message_or_type
            if isinstance(payload_or_type, str):
                actual_payload = {"message": payload_or_type} if payload_or_type else {}
            elif isinstance(payload_or_type, dict):
                actual_payload = payload_or_type
            else:
                actual_payload = {"message": str(payload_or_type)}
            return await self._send_internal(actual_type, actual_payload, priority)

        # Fallback: treat first arg as a string notification type value
        try:
            actual_type = NotificationType(str(message_or_type))
        except ValueError:
            actual_type = NotificationType.SYSTEM_ERROR

        if isinstance(payload_or_type, dict):
            actual_payload = payload_or_type
        elif isinstance(payload_or_type, str):
            actual_payload = {"message": payload_or_type} if payload_or_type else {}
        else:
            actual_payload = {"message": str(payload_or_type)}

        return await self._send_internal(actual_type, actual_payload, priority)

    def _record_to_delivery_result(self, record: NotificationRecord) -> DeliveryResult:
        """Convert a NotificationRecord to a DeliveryResult for backward compatibility."""
        if record.is_fully_delivered:
            for cs in record.channel_statuses.values():
                if cs.status == DeliveryStatus.DELIVERED:
                    return DeliveryResult(
                        success=True,
                        channel_name=cs.channel_name,
                        attempts=cs.attempts,
                    )
        if record.channel_statuses:
            last_status = list(record.channel_statuses.values())[-1]
            return DeliveryResult(
                success=False,
                channel_name=last_status.channel_name,
                attempts=last_status.attempts,
                error=last_status.error or "All retries exhausted",
                fallback_used=len(record.channel_statuses) > 1,
            )
        return DeliveryResult(
            success=False,
            channel_name="none",
            error="No channels configured for this notification type",
        )

    async def _send_internal(
        self,
        notification_type: NotificationType,
        payload: dict[str, Any],
        priority: Priority,
    ) -> NotificationRecord:
        """Internal send implementation with full tracking.

        Args:
            notification_type: The type of notification.
            payload: The notification data.
            priority: Priority level.

        Returns:
            NotificationRecord with delivery status.
        """
        record = NotificationRecord(
            id=str(uuid.uuid4()),
            notification_type=notification_type,
            priority=priority,
            payload=payload,
            created_at=datetime.now(timezone.utc),
        )

        target_channels = self._get_channels_for_notification(notification_type, priority)

        if not target_channels:
            logger.warning(
                "No delivery channel configured for notification type %s. "
                "Discarding notification %s.",
                notification_type.value,
                record.id,
            )
            self._delivery_records[record.id] = record
            return record

        message = self._format_message(notification_type, payload)

        # Attempt delivery to each target channel
        for channel_name in target_channels:
            channel_status = ChannelDeliveryStatus(channel_name=channel_name)
            record.channel_statuses[channel_name] = channel_status

            delivered = await self._attempt_delivery_with_retry(
                channel_name, message, channel_status
            )

            if delivered:
                continue

            # If all retries exhausted, try fallback to next available channel
            fallback_channel = self._get_fallback_channel(
                channel_name, target_channels, record
            )
            if fallback_channel:
                fallback_status = ChannelDeliveryStatus(channel_name=fallback_channel)
                record.channel_statuses[fallback_channel] = fallback_status
                await self._attempt_delivery_with_retry(
                    fallback_channel, message, fallback_status
                )

        self._delivery_records[record.id] = record
        return record

    async def _attempt_delivery_with_retry(
        self,
        channel_name: str,
        message: str,
        status: ChannelDeliveryStatus,
    ) -> bool:
        """Attempt to deliver a message with retry logic.

        Retries up to max_retries times with configured interval (Req 17.4).

        Args:
            channel_name: Name of the channel to deliver to.
            message: The formatted message.
            status: The delivery status tracker to update.

        Returns:
            True if delivery succeeded, False if all retries exhausted.
        """
        channel = self._channels.get(channel_name)
        if not channel:
            status.status = DeliveryStatus.FAILED
            status.error = f"Channel '{channel_name}' not registered"
            return False

        for attempt in range(1, self._retry_config.max_retries + 1):
            status.attempts = attempt
            status.last_attempt_at = datetime.now(timezone.utc)
            status.status = DeliveryStatus.RETRYING if attempt > 1 else DeliveryStatus.PENDING

            try:
                success = await channel.send(message)
                if success:
                    status.status = DeliveryStatus.DELIVERED
                    status.delivered_at = datetime.now(timezone.utc)
                    return True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                status.error = str(e)
                logger.warning(
                    "Delivery attempt %d/%d failed for channel '%s': %s",
                    attempt,
                    self._retry_config.max_retries,
                    channel_name,
                    e,
                )

            # Wait before retry (skip wait on last attempt)
            if attempt < self._retry_config.max_retries:
                await asyncio.sleep(self._retry_config.interval_seconds)

        status.status = DeliveryStatus.FAILED
        logger.error(
            "Permanent delivery failure for channel '%s' after %d attempts.",
            channel_name,
            self._retry_config.max_retries,
        )
        return False

    def _get_fallback_channel(
        self,
        failed_channel: str,
        already_targeted: list[str],
        record: NotificationRecord,
    ) -> str | None:
        """Find the next available channel not already targeted.

        Args:
            failed_channel: The channel that failed.
            already_targeted: Channels already in the target list.
            record: The notification record (to check already-attempted channels).

        Returns:
            Name of fallback channel, or None if no fallback available.
        """
        for ch_name in self._channel_order:
            if (
                ch_name != failed_channel
                and ch_name not in already_targeted
                and ch_name not in record.channel_statuses
            ):
                return ch_name
        return None

    def _format_message(
        self, notification_type: NotificationType, payload: dict[str, Any]
    ) -> str:
        """Format a notification payload into a human-readable message.

        Args:
            notification_type: The type of notification.
            payload: The notification data.

        Returns:
            Formatted message string.
        """
        message = payload.get("message")
        if message:
            return str(message)

        # Type-specific formatting
        if notification_type in (
            NotificationType.TRADE_ALERT,
            NotificationType.TRADE_OPENED,
            NotificationType.TRADE_CLOSED,
        ):
            return self._format_trade_alert(payload)
        elif notification_type == NotificationType.KILL_SWITCH:
            return self._format_kill_switch(payload)
        elif notification_type == NotificationType.CRISIS_ALERT:
            return self._format_crisis_alert(payload)
        elif notification_type == NotificationType.HFT_CIRCUIT_BREAKER:
            return self._format_hft_circuit_breaker(payload)
        elif notification_type in (
            NotificationType.STRATEGY_DISABLED,
            NotificationType.STRATEGY_CHANGE,
        ):
            return self._format_strategy_disabled(payload)

        return str(payload)

    def _format_trade_alert(self, payload: dict[str, Any]) -> str:
        """Format a trade alert notification."""
        instrument = payload.get("instrument", "Unknown")
        direction = payload.get("direction", "Unknown")
        size = payload.get("size", "Unknown")
        price = payload.get("price", payload.get("entry_price", "Unknown"))
        strategy = payload.get("strategy", "Unknown")
        pnl = payload.get("pnl")

        msg = f"Trade Alert: {direction} {size} {instrument} @ {price} [{strategy}]"
        if pnl is not None:
            msg += f" | PnL: {pnl}"
        return msg

    def _format_kill_switch(self, payload: dict[str, Any]) -> str:
        """Format a kill switch notification."""
        reason = payload.get("reason", payload.get("activation_reason", "Unknown reason"))
        positions_count = payload.get("positions_count", 0)
        return (
            f"⚠️ KILL SWITCH ACTIVATED: {reason} | "
            f"Closing {positions_count} positions"
        )

    def _format_crisis_alert(self, payload: dict[str, Any]) -> str:
        """Format a crisis alert notification."""
        region = payload.get("region", "Unknown")
        sentiment = payload.get("sentiment_avg", "N/A")
        instruments = payload.get("affected_instruments", [])
        return (
            f"🚨 CRISIS ALERT [{region}]: Sentiment {sentiment} | "
            f"Affected: {', '.join(instruments) if instruments else 'N/A'}"
        )

    def _format_hft_circuit_breaker(self, payload: dict[str, Any]) -> str:
        """Format an HFT circuit breaker notification."""
        pnl = payload.get("pnl", payload.get("pnl_loss", "N/A"))
        breaker_count = payload.get("breaker_count", 0)
        return (
            f"⚡ HFT Circuit Breaker Activated: PnL {pnl} | "
            f"Activation #{breaker_count}"
        )

    def _format_strategy_disabled(self, payload: dict[str, Any]) -> str:
        """Format a strategy disabled notification."""
        strategy = payload.get("strategy", "Unknown")
        reason = payload.get("reason", "Performance below threshold")
        return f"Strategy Disabled: {strategy} | Reason: {reason}"

    def get_delivery_status(self, notification_id: str) -> NotificationRecord | None:
        """Get the delivery status of a notification by ID.

        Args:
            notification_id: The notification record ID.

        Returns:
            The NotificationRecord if found, None otherwise.
        """
        return self._delivery_records.get(notification_id)

    # --- Convenience methods for specific notification types ---

    async def send_to_all_channels(
        self,
        message: str,
        timeout_seconds: float = 5.0,
    ) -> list[DeliveryResult]:
        """Send a notification to ALL channels simultaneously.

        Used for critical notifications like kill switch activation.
        Must complete within the specified timeout (default 5 seconds per Req 17.3).

        Args:
            message: The formatted notification message.
            timeout_seconds: Maximum time to wait for all deliveries.

        Returns:
            List of DeliveryResults, one per channel.
        """
        if not self._channels:
            return []

        async def _deliver_single(channel: NotificationChannel) -> DeliveryResult:
            try:
                success = await channel.send(message)
                return DeliveryResult(
                    success=success,
                    channel_name=channel.name,
                    attempts=1,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return DeliveryResult(
                    success=False,
                    channel_name=channel.name,
                    attempts=1,
                    error=str(e),
                )

        tasks = [_deliver_single(channel) for channel in self._channels.values()]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )
            delivery_results: list[DeliveryResult] = []
            for i, result in enumerate(results):
                if isinstance(result, DeliveryResult):
                    delivery_results.append(result)
                elif isinstance(result, Exception):
                    channel_name = list(self._channels.keys())[i]
                    delivery_results.append(
                        DeliveryResult(
                            success=False,
                            channel_name=channel_name,
                            error=str(result),
                        )
                    )
            return delivery_results
        except asyncio.TimeoutError:
            logger.error(
                "Notification timed out after %.1fs", timeout_seconds
            )
            return [
                DeliveryResult(success=False, channel_name=name, error="Timeout")
                for name in self._channels
            ]

    async def send_alert(self, message: str, level: str = "info") -> None:
        """Send a generic alert (compatible with NotificationServiceProtocol).

        Args:
            message: Alert message text.
            level: Alert level (info, warning, error, critical).
        """
        ntype = (
            NotificationType.SYSTEM_ERROR
            if level in ("error", "critical")
            else NotificationType.RISK_ALERT
        )
        priority = Priority.CRITICAL if level == "critical" else Priority.NORMAL
        await self.send(ntype, {"message": message}, priority)

    async def send_trade_notification(
        self, trade_data: TradeNotificationData
    ) -> DeliveryResult:
        """Format and send a trade notification within 10 seconds (Req 17.2).

        Args:
            trade_data: Trade notification data.

        Returns:
            DeliveryResult indicating delivery outcome.
        """
        start_time = time.monotonic()
        message = format_trade_notification(trade_data)

        ntype = (
            NotificationType.TRADE_CLOSED
            if trade_data.exit_price is not None
            else NotificationType.TRADE_OPENED
        )

        # Get all configured channels for this type (use HIGH to get all configured)
        channel_names = self._get_channels_for_notification(ntype, Priority.HIGH)
        if not channel_names:
            # Fallback to all channels
            channel_names = self._channel_order

        if not channel_names:
            return DeliveryResult(
                success=False, channel_name="none", error="No channels configured"
            )

        # Try each channel within the 10-second window
        for channel_name in channel_names:
            elapsed = time.monotonic() - start_time
            if elapsed >= 10.0:
                logger.error("Trade notification exceeded 10-second deadline")
                break

            channel = self._channels.get(channel_name)
            if channel is None:
                continue

            try:
                remaining = 10.0 - elapsed
                success = await asyncio.wait_for(
                    channel.send(message), timeout=remaining
                )
                if success:
                    return DeliveryResult(
                        success=True, channel_name=channel_name, attempts=1
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Trade notification timed out on channel=%s", channel_name
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "Trade notification error on channel=%s: %s", channel_name, e
                )

        return DeliveryResult(
            success=False,
            channel_name=channel_names[-1] if channel_names else "none",
            error="Trade notification delivery failed within 10s deadline",
        )

    async def send_kill_switch_notification(
        self, data: KillSwitchNotificationData
    ) -> list[DeliveryResult]:
        """Send kill switch notification to ALL channels within 5 seconds (Req 17.3).

        Args:
            data: Kill switch notification data.

        Returns:
            List of DeliveryResults, one per channel.
        """
        message = format_kill_switch_notification(data)
        return await self.send_to_all_channels(message, timeout_seconds=5.0)

    async def send_hft_circuit_breaker_notification(
        self, data: HFTCircuitBreakerData
    ) -> DeliveryResult:
        """Send HFT circuit breaker activation notification (Req 22.9).

        Args:
            data: HFT circuit breaker data.

        Returns:
            DeliveryResult indicating delivery outcome.
        """
        message = format_hft_circuit_breaker_notification(data)
        result = await self.send(
            message, notification_type=NotificationType.HFT_CIRCUIT_BREAKER, priority=Priority.HIGH
        )
        if isinstance(result, DeliveryResult):
            return result
        return self._record_to_delivery_result(result)

    async def send_hft_mode_disabled_notification(
        self, data: HFTModeDisabledData
    ) -> DeliveryResult:
        """Send HFT mode disabled (escalation) notification (Req 22.10).

        Args:
            data: HFT mode disabled data.

        Returns:
            DeliveryResult indicating delivery outcome.
        """
        message = format_hft_mode_disabled_notification(data)
        result = await self.send(
            message, notification_type=NotificationType.HFT_MODE_DISABLED, priority=Priority.CRITICAL
        )
        if isinstance(result, DeliveryResult):
            return result
        return self._record_to_delivery_result(result)

    async def send_crisis_alert_notification(
        self, data: CrisisAlertData
    ) -> DeliveryResult:
        """Send news crisis alert notification (Req 23.7).

        Args:
            data: Crisis alert data.

        Returns:
            DeliveryResult indicating delivery outcome.
        """
        message = format_crisis_alert_notification(data)
        result = await self.send(
            message, notification_type=NotificationType.CRISIS_ALERT, priority=Priority.HIGH
        )
        if isinstance(result, DeliveryResult):
            return result
        return self._record_to_delivery_result(result)
