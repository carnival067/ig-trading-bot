"""Async SMTP email notification channel.

Implements the NotificationChannel interface for delivering trading
notifications via email using aiosmtplib.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiosmtplib

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Priority level for notifications."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Notification:
    """A notification to be delivered."""

    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    metadata: dict[str, Any] = field(default_factory=dict)


class NotificationChannel(ABC):
    """Abstract base class for notification delivery channels."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send a notification. Returns True on success, False on failure."""
        ...


@dataclass
class EmailConfig:
    """Configuration for the SMTP email channel."""

    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    to_addresses: list[str]
    use_tls: bool = True


class EmailChannel(NotificationChannel):
    """Async SMTP email notification channel.

    Sends HTML-formatted emails via aiosmtplib with TLS/SSL support.
    Handles connection errors gracefully by logging and returning False.
    """

    def __init__(self, config: EmailConfig) -> None:
        self.config = config

    async def send(self, notification: Notification) -> bool:
        """Send a notification as an HTML email.

        Args:
            notification: The notification to deliver.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        subject = self._format_subject(notification)
        html_body = self._format_html(notification)
        message = self._build_message(subject, html_body)

        try:
            await self._send_smtp(message)
            logger.info(
                "Email notification sent successfully",
                extra={
                    "subject": subject,
                    "recipients": self.config.to_addresses,
                    "level": notification.level.value,
                },
            )
            return True
        except aiosmtplib.SMTPConnectError as e:
            logger.error(
                "Failed to connect to SMTP server: %s",
                str(e),
                extra={
                    "smtp_host": self.config.smtp_host,
                    "smtp_port": self.config.smtp_port,
                },
            )
            return False
        except aiosmtplib.SMTPAuthenticationError as e:
            logger.error(
                "SMTP authentication failed: %s",
                str(e),
                extra={"username": self.config.username},
            )
            return False
        except aiosmtplib.SMTPResponseException as e:
            logger.error(
                "SMTP server returned error: %s (code: %d)",
                e.message,
                e.code,
                extra={"smtp_code": e.code},
            )
            return False
        except (OSError, TimeoutError) as e:
            logger.error(
                "Network error sending email: %s",
                str(e),
                extra={
                    "smtp_host": self.config.smtp_host,
                    "smtp_port": self.config.smtp_port,
                },
            )
            return False

    async def _send_smtp(self, message: str) -> None:
        """Send the raw email message via SMTP."""
        await aiosmtplib.send(
            message,
            hostname=self.config.smtp_host,
            port=self.config.smtp_port,
            username=self.config.username,
            password=self.config.password,
            use_tls=self.config.use_tls,
        )

    def _format_subject(self, notification: Notification) -> str:
        """Format the email subject line based on notification level."""
        prefix_map = {
            NotificationLevel.CRITICAL: "[CRITICAL]",
            NotificationLevel.WARNING: "[WARNING]",
            NotificationLevel.INFO: "[INFO]",
        }
        prefix = prefix_map.get(notification.level, "[INFO]")
        return f"{prefix} Trading System - {notification.title}"

    def _format_html(self, notification: Notification) -> str:
        """Format the notification as an HTML email body."""
        level_color = {
            NotificationLevel.CRITICAL: "#dc3545",
            NotificationLevel.WARNING: "#ffc107",
            NotificationLevel.INFO: "#17a2b8",
        }
        color = level_color.get(notification.level, "#17a2b8")

        metadata_rows = ""
        if notification.metadata:
            rows = "".join(
                f"<tr><td style='padding:4px 8px;font-weight:bold;'>{k}</td>"
                f"<td style='padding:4px 8px;'>{v}</td></tr>"
                for k, v in notification.metadata.items()
            )
            metadata_rows = (
                "<table style='border-collapse:collapse;width:100%;margin-top:12px;'>"
                f"{rows}</table>"
            )

        return (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'></head>"
            "<body style='font-family:Arial,sans-serif;margin:0;padding:20px;"
            "background:#f5f5f5;'>"
            "<div style='max-width:600px;margin:0 auto;background:#fff;"
            "border-radius:8px;overflow:hidden;"
            "box-shadow:0 2px 4px rgba(0,0,0,0.1);'>"
            f"<div style='background:{color};padding:16px 24px;'>"
            f"<h2 style='color:#fff;margin:0;'>{notification.title}</h2></div>"
            "<div style='padding:24px;'>"
            f"<p style='color:#333;line-height:1.6;margin:0 0 16px 0;'>"
            f"{notification.message}</p>"
            f"{metadata_rows}</div>"
            "<div style='background:#f8f9fa;padding:12px 24px;text-align:center;"
            "color:#6c757d;font-size:12px;'>"
            "Institutional AI Trading System</div></div></body></html>"
        )

    def _build_message(self, subject: str, html_body: str) -> str:
        """Build a raw MIME email message string."""
        boundary = "----=_Part_Trading_System_Boundary"
        recipients = ", ".join(self.config.to_addresses)

        return (
            f"From: {self.config.from_address}\r\n"
            f"To: {recipients}\r\n"
            f"Subject: {subject}\r\n"
            f"MIME-Version: 1.0\r\n"
            f'Content-Type: multipart/alternative; boundary="{boundary}"\r\n'
            f"\r\n"
            f"--{boundary}\r\n"
            f"Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Transfer-Encoding: 7bit\r\n"
            f"\r\n"
            f"{html_body}\r\n"
            f"--{boundary}--\r\n"
        )
