"""Unit tests for the Telegram notification channel.

Tests cover message sending, error handling, message truncation,
and protocol compliance with mocked HTTP calls.

Validates: Requirements 17.1
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.notifications.telegram import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TelegramChannel,
)


@pytest.fixture
def channel() -> TelegramChannel:
    """Create a TelegramChannel instance for testing."""
    return TelegramChannel(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        chat_id="-1001234567890",
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data or {"ok": True, "result": {"message_id": 1}},
    )


# =============================================================================
# Protocol compliance
# =============================================================================


class TestProtocolCompliance:
    """Verify TelegramChannel implements NotificationChannel protocol."""

    def test_has_name_property(self, channel: TelegramChannel) -> None:
        assert channel.name == "telegram"

    def test_has_send_method(self, channel: TelegramChannel) -> None:
        assert hasattr(channel, "send")
        assert callable(channel.send)


# =============================================================================
# Successful message sending
# =============================================================================


class TestSendSuccess:
    """Tests for successful message delivery."""

    @pytest.mark.asyncio
    async def test_send_returns_true_on_success(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await channel.send("Hello, World!")

        assert result is True

    @pytest.mark.asyncio
    async def test_send_calls_correct_url(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("Test message")

        expected_url = (
            "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/sendMessage"
        )
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == expected_url

    @pytest.mark.asyncio
    async def test_send_payload_contains_chat_id_and_message(
        self, channel: TelegramChannel
    ) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("Test message")

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["chat_id"] == "-1001234567890"
        assert payload["text"] == "Test message"
        assert payload["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_send_with_disable_notification(self) -> None:
        channel = TelegramChannel(
            bot_token="token",
            chat_id="123",
            disable_notification=True,
        )
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("Test")

        payload = mock_post.call_args[1]["json"]
        assert payload["disable_notification"] is True

    @pytest.mark.asyncio
    async def test_send_default_notification_enabled(
        self, channel: TelegramChannel
    ) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("Test")

        payload = mock_post.call_args[1]["json"]
        assert payload["disable_notification"] is False


# =============================================================================
# Error handling
# =============================================================================


class TestSendErrors:
    """Tests for error handling during message delivery."""

    @pytest.mark.asyncio
    async def test_api_returns_not_ok(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(
            200, {"ok": False, "description": "Bad Request: chat not found"}
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await channel.send("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_http_500_error(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(500, {"ok": False, "description": "Internal Server Error"})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await channel.send("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_http_429_rate_limited(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(429, {"ok": False, "description": "Too Many Requests"})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await channel.send("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_error(self, channel: TelegramChannel) -> None:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Connection timed out")
            result = await channel.send("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(self, channel: TelegramChannel) -> None:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            result = await channel.send("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_unexpected_exception(self, channel: TelegramChannel) -> None:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = RuntimeError("Something unexpected")
            result = await channel.send("Test")

        assert result is False


# =============================================================================
# Message truncation
# =============================================================================


class TestMessageTruncation:
    """Tests for message length limit enforcement (4096 chars max)."""

    def test_short_message_not_truncated(self, channel: TelegramChannel) -> None:
        message = "Short message"
        result = channel._truncate_message(message)
        assert result == message

    def test_exact_limit_not_truncated(self, channel: TelegramChannel) -> None:
        message = "x" * TELEGRAM_MAX_MESSAGE_LENGTH
        result = channel._truncate_message(message)
        assert result == message
        assert len(result) == TELEGRAM_MAX_MESSAGE_LENGTH

    def test_over_limit_is_truncated(self, channel: TelegramChannel) -> None:
        message = "x" * (TELEGRAM_MAX_MESSAGE_LENGTH + 100)
        result = channel._truncate_message(message)
        assert len(result) <= TELEGRAM_MAX_MESSAGE_LENGTH
        assert "truncated" in result

    def test_truncated_message_within_limit(self, channel: TelegramChannel) -> None:
        message = "A" * 10000
        result = channel._truncate_message(message)
        assert len(result) <= TELEGRAM_MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_long_message_truncated_before_sending(
        self, channel: TelegramChannel
    ) -> None:
        long_message = "x" * 5000
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send(long_message)

        payload = mock_post.call_args[1]["json"]
        assert len(payload["text"]) <= TELEGRAM_MAX_MESSAGE_LENGTH


# =============================================================================
# Integration with NotificationService
# =============================================================================


class TestNotificationServiceIntegration:
    """Tests verifying TelegramChannel works with NotificationService."""

    @pytest.mark.asyncio
    async def test_channel_works_with_notification_service(self) -> None:
        """Verify TelegramChannel satisfies the NotificationChannel protocol."""
        from src.notifications.notification_service import NotificationService

        channel = TelegramChannel(bot_token="test-token", chat_id="123")

        # Should not raise - channel satisfies the protocol
        service = NotificationService(channels=[channel])
        assert "telegram" in service._channels

    @pytest.mark.asyncio
    async def test_send_via_notification_service(self) -> None:
        """Verify messages can be sent through the NotificationService."""
        from src.notifications.formatters import NotificationType
        from src.notifications.notification_service import (
            ChannelConfig,
            NotificationService,
        )

        channel = TelegramChannel(bot_token="test-token", chat_id="123")
        config = ChannelConfig(
            type_to_channels={NotificationType.RISK_ALERT: ["telegram"]}
        )
        service = NotificationService(channels=[channel], channel_config=config)

        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await service.send(
                "Test alert",
                notification_type=NotificationType.RISK_ALERT,
            )

        assert result.success is True
        assert result.channel_name == "telegram"


# =============================================================================
# Client lifecycle
# =============================================================================


class TestClientLifecycle:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_close_client(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("Test")

        # Client should be created
        assert channel._client is not None

        await channel.close()
        assert channel._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, channel: TelegramChannel) -> None:
        # Should not raise
        await channel.close()
        assert channel._client is None

    @pytest.mark.asyncio
    async def test_client_reused_across_sends(self, channel: TelegramChannel) -> None:
        mock_response = _mock_response(200, {"ok": True, "result": {"message_id": 1}})

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await channel.send("First message")
            client_after_first = channel._client
            await channel.send("Second message")
            client_after_second = channel._client

        assert client_after_first is client_after_second
