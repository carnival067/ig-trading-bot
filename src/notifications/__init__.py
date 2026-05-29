"""Notification service module.

Provides multi-channel notification delivery via Telegram, Discord, and Email
with retry logic, fallback, and specialized formatters for trading events.
"""

from src.notifications.formatters import (
    CrisisAlertData,
    HFTCircuitBreakerData,
    HFTModeDisabledData,
    KillSwitchNotificationData,
    NotificationType,
    Priority,
    TradeDirection,
    TradeNotificationData,
)
from src.notifications.notification_service import (
    ChannelConfig,
    DeliveryResult,
    NotificationChannel,
    NotificationService,
    RetryConfig,
)
from src.notifications.telegram import TelegramChannel

__all__ = [
    "ChannelConfig",
    "CrisisAlertData",
    "DeliveryResult",
    "HFTCircuitBreakerData",
    "HFTModeDisabledData",
    "KillSwitchNotificationData",
    "NotificationChannel",
    "NotificationService",
    "NotificationType",
    "Priority",
    "RetryConfig",
    "TelegramChannel",
    "TradeDirection",
    "TradeNotificationData",
]
