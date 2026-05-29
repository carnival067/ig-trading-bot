"""Unit tests for the EmailChannel notification channel."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.notifications.email import (
    EmailChannel,
    EmailConfig,
    Notification,
    NotificationChannel,
    NotificationLevel,
)


@pytest.fixture
def email_config() -> EmailConfig:
    """Create a test email configuration."""
    return EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        username="trader@example.com",
        password="secret123",
        from_address="alerts@trading-system.com",
        to_addresses=["user1@example.com", "user2@example.com"],
        use_tls=True,
    )


@pytest.fixture
def email_channel(email_config: EmailConfig) -> EmailChannel:
    """Create an EmailChannel instance for testing."""
    return EmailChannel(config=email_config)


@pytest.fixture
def sample_notification() -> Notification:
    """Create a sample notification."""
    return Notification(
        title="Trade Executed",
        message="Bought 1.5 lots of EUR/USD at 1.0850",
        level=NotificationLevel.INFO,
        metadata={"instrument": "EUR/USD", "direction": "LONG", "size": "1.5"},
    )


class TestEmailChannelInterface:
    """Tests that EmailChannel implements the NotificationChannel interface."""

    def test_implements_notification_channel(self, email_channel: EmailChannel) -> None:
        """EmailChannel should be an instance of NotificationChannel."""
        assert isinstance(email_channel, NotificationChannel)

    def test_has_send_method(self, email_channel: EmailChannel) -> None:
        """EmailChannel should have an async send method."""
        assert hasattr(email_channel, "send")
        assert callable(email_channel.send)


class TestEmailChannelSend:
    """Tests for the send method."""

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_send_success(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return True when email is sent successfully."""
        result = await email_channel.send(sample_notification)

        assert result is True
        mock_send.assert_called_once()

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_send_passes_smtp_config(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should pass correct SMTP configuration to aiosmtplib."""
        await email_channel.send(sample_notification)

        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["hostname"] == "smtp.example.com"
        assert call_kwargs["port"] == 587
        assert call_kwargs["username"] == "trader@example.com"
        assert call_kwargs["password"] == "secret123"
        assert call_kwargs["use_tls"] is True

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_send_with_tls_disabled(
        self,
        mock_send: AsyncMock,
        sample_notification: Notification,
    ) -> None:
        """Should support non-TLS connections."""
        config = EmailConfig(
            smtp_host="localhost",
            smtp_port=25,
            username="user",
            password="pass",
            from_address="from@test.com",
            to_addresses=["to@test.com"],
            use_tls=False,
        )
        channel = EmailChannel(config=config)

        await channel.send(sample_notification)

        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["use_tls"] is False


class TestEmailChannelErrorHandling:
    """Tests for graceful error handling."""

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_connection_error_returns_false(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return False on SMTP connection error."""
        import aiosmtplib

        mock_send.side_effect = aiosmtplib.SMTPConnectError("Connection refused")

        result = await email_channel.send(sample_notification)

        assert result is False

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_authentication_error_returns_false(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return False on SMTP authentication failure."""
        import aiosmtplib

        mock_send.side_effect = aiosmtplib.SMTPAuthenticationError(535, "Auth failed")

        result = await email_channel.send(sample_notification)

        assert result is False

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_smtp_response_error_returns_false(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return False on SMTP response error."""
        import aiosmtplib

        mock_send.side_effect = aiosmtplib.SMTPResponseException(550, "Mailbox not found")

        result = await email_channel.send(sample_notification)

        assert result is False

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_network_timeout_returns_false(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return False on network timeout."""
        mock_send.side_effect = TimeoutError("Connection timed out")

        result = await email_channel.send(sample_notification)

        assert result is False

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_os_error_returns_false(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Should return False on OS-level network error."""
        mock_send.side_effect = OSError("Network unreachable")

        result = await email_channel.send(sample_notification)

        assert result is False


class TestEmailFormatting:
    """Tests for email subject and body formatting."""

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_info_subject_prefix(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """INFO notifications should have [INFO] prefix in subject."""
        notification = Notification(
            title="Trade Opened", message="Test", level=NotificationLevel.INFO
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "[INFO] Trading System - Trade Opened" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_warning_subject_prefix(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """WARNING notifications should have [WARNING] prefix in subject."""
        notification = Notification(
            title="Risk Alert", message="Test", level=NotificationLevel.WARNING
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "[WARNING] Trading System - Risk Alert" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_critical_subject_prefix(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """CRITICAL notifications should have [CRITICAL] prefix in subject."""
        notification = Notification(
            title="Kill Switch Activated",
            message="Emergency halt",
            level=NotificationLevel.CRITICAL,
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "[CRITICAL] Trading System - Kill Switch Activated" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_html_body_contains_message(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """Email body should contain the notification message."""
        notification = Notification(
            title="Test", message="Position closed with +150 pips profit"
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "Position closed with +150 pips profit" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_html_body_contains_metadata(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """Email body should include metadata as a table."""
        notification = Notification(
            title="Trade",
            message="Executed",
            metadata={"instrument": "GBP/USD", "pnl": "+$500"},
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "instrument" in raw_message
        assert "GBP/USD" in raw_message
        assert "pnl" in raw_message
        assert "+$500" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_message_includes_from_address(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Email should include the configured from address."""
        await email_channel.send(sample_notification)

        raw_message = mock_send.call_args[0][0]
        assert "From: alerts@trading-system.com" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_message_includes_to_addresses(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Email should include all configured recipients."""
        await email_channel.send(sample_notification)

        raw_message = mock_send.call_args[0][0]
        assert "user1@example.com" in raw_message
        assert "user2@example.com" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_message_is_html_mime_type(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
        sample_notification: Notification,
    ) -> None:
        """Email should be sent as HTML MIME type."""
        await email_channel.send(sample_notification)

        raw_message = mock_send.call_args[0][0]
        assert "Content-Type: text/html" in raw_message

    @patch("src.notifications.email.aiosmtplib.send", new_callable=AsyncMock)
    async def test_critical_notification_uses_red_color(
        self,
        mock_send: AsyncMock,
        email_channel: EmailChannel,
    ) -> None:
        """CRITICAL notifications should use red color in HTML header."""
        notification = Notification(
            title="Kill Switch", message="Activated", level=NotificationLevel.CRITICAL
        )
        await email_channel.send(notification)

        raw_message = mock_send.call_args[0][0]
        assert "#dc3545" in raw_message


class TestEmailConfig:
    """Tests for EmailConfig dataclass."""

    def test_default_tls_enabled(self) -> None:
        """TLS should be enabled by default."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_address="from@test.com",
            to_addresses=["to@test.com"],
        )
        assert config.use_tls is True

    def test_multiple_recipients(self) -> None:
        """Should support multiple recipient addresses."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_address="from@test.com",
            to_addresses=["a@test.com", "b@test.com", "c@test.com"],
        )
        assert len(config.to_addresses) == 3
