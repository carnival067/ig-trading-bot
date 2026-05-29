"""Unit tests for the Discord webhook notification channel.

Tests cover message sending, embed formatting, rate limit handling,
error recovery, retry logic, and message truncation.

Validates: Requirements 17.1
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.notifications.discord import (
    DISCORD_MAX_CONTENT_LENGTH,
    DISCORD_MAX_EMBED_DESCRIPTION_LENGTH,
    DISCORD_MAX_EMBED_TITLE_LENGTH,
    DiscordChannel,
    DiscordEmbed,
    EmbedColor,
    EmbedField,
)


@pytest.fixture
def webhook_url() -> str:
    """Test webhook URL."""
    return "https://discord.com/api/webhooks/123456/abcdef"


@pytest.fixture
def channel(webhook_url: str) -> DiscordChannel:
    """Create a DiscordChannel instance for testing."""
    return DiscordChannel(
        webhook_url=webhook_url,
        username="Test Bot",
        avatar_url="https://example.com/avatar.png",
        max_retries=3,
        retry_delay=0.01,  # Fast retries for tests
        timeout=5.0,
    )


def _mock_response(
    status_code: int = 204,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    """Create a mock httpx.Response."""
    request = httpx.Request("POST", "https://discord.com/api/webhooks/123456/abcdef")
    return httpx.Response(
        status_code=status_code,
        json=json_data or {},
        headers=headers or {},
        request=request,
    )


class TestDiscordChannelInit:
    """Tests for DiscordChannel initialization."""

    def test_init_with_valid_url(self, webhook_url: str) -> None:
        channel = DiscordChannel(webhook_url=webhook_url)
        assert channel._webhook_url == webhook_url
        assert channel._username == "Trading Bot"
        assert channel._avatar_url is None
        assert channel._max_retries == 3

    def test_init_with_custom_params(self, webhook_url: str) -> None:
        channel = DiscordChannel(
            webhook_url=webhook_url,
            username="Custom Bot",
            avatar_url="https://example.com/img.png",
            max_retries=5,
            retry_delay=2.0,
            timeout=15.0,
        )
        assert channel._username == "Custom Bot"
        assert channel._avatar_url == "https://example.com/img.png"
        assert channel._max_retries == 5
        assert channel._retry_delay == 2.0
        assert channel._timeout == 15.0

    def test_init_with_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="webhook_url must not be empty"):
            DiscordChannel(webhook_url="")


class TestDiscordChannelSend:
    """Tests for sending messages via Discord webhook."""

    @pytest.mark.asyncio
    async def test_send_simple_message_success(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        result = await channel.send("Hello, Discord!")

        assert result is True
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["content"] == "Hello, Discord!"
        assert payload["username"] == "Test Bot"
        assert payload["avatar_url"] == "https://example.com/avatar.png"

    @pytest.mark.asyncio
    async def test_send_with_200_status(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(200)
        mock_client.is_closed = False
        channel._client = mock_client

        result = await channel.send("Test message")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_with_title_creates_embed(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        result = await channel.send(
            "Trade executed successfully",
            title="Trade Alert",
            level="success",
        )

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"] == "Trade Alert"
        assert embed["description"] == "Trade executed successfully"
        assert embed["color"] == EmbedColor.SUCCESS

    @pytest.mark.asyncio
    async def test_send_with_fields(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        fields = [
            EmbedField(name="Instrument", value="EUR/USD", inline=True),
            EmbedField(name="Direction", value="LONG", inline=True),
            EmbedField(name="PnL", value="+$150.00", inline=False),
        ]

        result = await channel.send(
            "Position closed",
            title="Trade Closed",
            fields=fields,
            level="info",
        )

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        embed = payload["embeds"][0]
        assert len(embed["fields"]) == 3
        assert embed["fields"][0]["name"] == "Instrument"
        assert embed["fields"][0]["value"] == "EUR/USD"
        assert embed["fields"][0]["inline"] is True
        assert embed["fields"][2]["inline"] is False

    @pytest.mark.asyncio
    async def test_send_with_embed_object(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        embed = DiscordEmbed(
            title="Kill Switch Activated",
            description="Emergency halt triggered",
            color=EmbedColor.CRITICAL,
            fields=[EmbedField(name="Reason", value="Drawdown > 15%")],
            footer="Trading System",
        )

        result = await channel.send("ignored content", embed=embed)

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["embeds"][0]["title"] == "Kill Switch Activated"
        assert payload["embeds"][0]["color"] == EmbedColor.CRITICAL

    @pytest.mark.asyncio
    async def test_send_embed_directly(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        embed = DiscordEmbed(
            title="Test Embed",
            description="Direct embed send",
            color=EmbedColor.WARNING,
        )

        result = await channel.send_embed(embed)
        assert result is True


class TestDiscordChannelRateLimiting:
    """Tests for rate limit handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_json_body(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        rate_limit_response = _mock_response(
            429, json_data={"retry_after": 0.01, "message": "Rate limited"}
        )
        success_response = _mock_response(204)

        mock_client.post.side_effect = [rate_limit_response, success_response]
        channel._client = mock_client

        result = await channel.send("Rate limited message")

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_header(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        rate_limit_response = _mock_response(
            429, headers={"Retry-After": "0.01"}
        )
        success_response = _mock_response(204)

        mock_client.post.side_effect = [rate_limit_response, success_response]
        channel._client = mock_client

        result = await channel.send("Rate limited message")

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhausts_retries(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        rate_limit_response = _mock_response(
            429, json_data={"retry_after": 0.001}
        )
        mock_client.post.return_value = rate_limit_response
        channel._client = mock_client

        result = await channel.send("Always rate limited")

        assert result is False
        assert mock_client.post.call_count == 3  # max_retries = 3


class TestDiscordChannelErrorHandling:
    """Tests for error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_server_error_retries_with_backoff(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        error_response = _mock_response(500)
        success_response = _mock_response(204)

        mock_client.post.side_effect = [error_response, success_response]
        channel._client = mock_client

        result = await channel.send("Server error test")

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_server_error_exhausts_retries(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        error_response = _mock_response(503)
        mock_client.post.return_value = error_response
        channel._client = mock_client

        result = await channel.send("Persistent server error")

        assert result is False
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_client_error_no_retry(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        error_response = _mock_response(400, json_data={"message": "Bad request"})
        mock_client.post.return_value = error_response
        channel._client = mock_client

        result = await channel.send("Bad request test")

        assert result is False
        # Client errors (4xx except 429) should not retry
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_forbidden_error_no_retry(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        error_response = _mock_response(403)
        mock_client.post.return_value = error_response
        channel._client = mock_client

        result = await channel.send("Forbidden test")

        assert result is False
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_retries(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        mock_client.post.side_effect = [
            httpx.TimeoutException("Connection timed out"),
            _mock_response(204),
        ]
        channel._client = mock_client

        result = await channel.send("Timeout test")

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_connection_error_retries(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        mock_client.post.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            _mock_response(204),
        ]
        channel._client = mock_client

        result = await channel.send("Connection error test")

        assert result is True
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_on_timeout(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False

        mock_client.post.side_effect = httpx.TimeoutException("Always times out")
        channel._client = mock_client

        result = await channel.send("Always timeout")

        assert result is False
        assert mock_client.post.call_count == 3


class TestDiscordChannelTruncation:
    """Tests for message length limit enforcement."""

    @pytest.mark.asyncio
    async def test_content_truncated_at_2000_chars(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        long_message = "A" * 2500

        await channel.send(long_message)

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        content = payload["content"]
        assert len(content) <= DISCORD_MAX_CONTENT_LENGTH
        assert content.endswith("...")

    @pytest.mark.asyncio
    async def test_embed_description_truncated(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(204)
        mock_client.is_closed = False
        channel._client = mock_client

        long_description = "B" * 5000

        await channel.send(long_description, title="Long Description")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        description = payload["embeds"][0]["description"]
        assert len(description) <= DISCORD_MAX_EMBED_DESCRIPTION_LENGTH
        assert description.endswith("...")

    @pytest.mark.asyncio
    async def test_embed_title_truncated(self, channel: DiscordChannel) -> None:
        long_title = "C" * 300
        embed = DiscordEmbed(title=long_title, description="Test")
        result = embed.to_dict()
        assert len(result["title"]) <= DISCORD_MAX_EMBED_TITLE_LENGTH

    def test_short_message_not_truncated(self, channel: DiscordChannel) -> None:
        result = channel._truncate("Short message", DISCORD_MAX_CONTENT_LENGTH)
        assert result == "Short message"

    def test_exact_length_not_truncated(self, channel: DiscordChannel) -> None:
        exact = "A" * DISCORD_MAX_CONTENT_LENGTH
        result = channel._truncate(exact, DISCORD_MAX_CONTENT_LENGTH)
        assert result == exact
        assert len(result) == DISCORD_MAX_CONTENT_LENGTH


class TestDiscordEmbed:
    """Tests for DiscordEmbed formatting."""

    def test_embed_to_dict_basic(self) -> None:
        embed = DiscordEmbed(
            title="Test Title",
            description="Test Description",
            color=EmbedColor.SUCCESS,
        )
        result = embed.to_dict()

        assert result["title"] == "Test Title"
        assert result["description"] == "Test Description"
        assert result["color"] == EmbedColor.SUCCESS

    def test_embed_to_dict_with_fields(self) -> None:
        embed = DiscordEmbed(
            title="Trade Alert",
            fields=[
                EmbedField(name="Symbol", value="EURUSD", inline=True),
                EmbedField(name="Action", value="BUY", inline=True),
            ],
        )
        result = embed.to_dict()

        assert len(result["fields"]) == 2
        assert result["fields"][0]["name"] == "Symbol"
        assert result["fields"][1]["value"] == "BUY"

    def test_embed_to_dict_with_timestamp(self) -> None:
        embed = DiscordEmbed(
            title="Alert",
            timestamp="2024-01-15T10:30:00+00:00",
        )
        result = embed.to_dict()
        assert result["timestamp"] == "2024-01-15T10:30:00+00:00"

    def test_embed_to_dict_with_footer(self) -> None:
        embed = DiscordEmbed(
            title="Alert",
            footer="Trading System v1.0",
        )
        result = embed.to_dict()
        assert result["footer"]["text"] == "Trading System v1.0"

    def test_embed_to_dict_no_optional_fields(self) -> None:
        embed = DiscordEmbed(title="Minimal")
        result = embed.to_dict()

        assert result["title"] == "Minimal"
        assert "description" not in result or result.get("description") == ""
        assert "fields" not in result
        assert "timestamp" not in result
        assert "footer" not in result

    def test_embed_field_value_truncated(self) -> None:
        long_value = "X" * 1500
        embed = DiscordEmbed(
            title="Test",
            fields=[EmbedField(name="Field", value=long_value)],
        )
        result = embed.to_dict()
        assert len(result["fields"][0]["value"]) <= 1024

    def test_embed_max_25_fields(self) -> None:
        fields = [EmbedField(name=f"Field {i}", value=f"Value {i}") for i in range(30)]
        embed = DiscordEmbed(title="Many Fields", fields=fields)
        result = embed.to_dict()
        assert len(result["fields"]) == 25


class TestDiscordChannelLevelColors:
    """Tests for notification level to color mapping."""

    def test_info_level(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("info") == EmbedColor.INFO

    def test_success_level(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("success") == EmbedColor.SUCCESS

    def test_warning_level(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("warning") == EmbedColor.WARNING

    def test_error_level(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("error") == EmbedColor.ERROR

    def test_critical_level(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("critical") == EmbedColor.CRITICAL

    def test_unknown_level_defaults_to_info(self, channel: DiscordChannel) -> None:
        assert channel._level_to_color("unknown") == EmbedColor.INFO


class TestDiscordChannelClientManagement:
    """Tests for HTTP client lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_client(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        channel._client = mock_client

        await channel.close()

        mock_client.aclose.assert_called_once()
        assert channel._client is None

    @pytest.mark.asyncio
    async def test_close_already_closed_client(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = True
        channel._client = mock_client

        await channel.close()
        # Should not call aclose on already closed client
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_no_client(self, channel: DiscordChannel) -> None:
        channel._client = None
        # Should not raise
        await channel.close()

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, channel: DiscordChannel) -> None:
        channel._client = None
        client = await channel._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        # Cleanup
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, channel: DiscordChannel) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        channel._client = mock_client

        client = await channel._get_client()
        assert client is mock_client
