"""Discord webhook notification channel.

Implements the NotificationChannel interface for sending notifications
via Discord webhooks with rich embed formatting.

Validates: Requirements 17.1
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

# Discord API limits
DISCORD_MAX_CONTENT_LENGTH = 2000
DISCORD_MAX_EMBED_TITLE_LENGTH = 256
DISCORD_MAX_EMBED_DESCRIPTION_LENGTH = 4096
DISCORD_MAX_EMBED_FIELDS = 25
DISCORD_MAX_FIELD_NAME_LENGTH = 256
DISCORD_MAX_FIELD_VALUE_LENGTH = 1024

# Rate limit retry defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0


class NotificationChannel(Protocol):
    """Protocol for notification delivery channels."""

    async def send(self, message: str, **kwargs: Any) -> bool:
        """Send a notification message. Returns True on success."""
        ...


class EmbedColor(int, Enum):
    """Discord embed colors for different notification types."""

    INFO = 0x3498DB  # Blue
    SUCCESS = 0x2ECC71  # Green
    WARNING = 0xF39C12  # Orange
    ERROR = 0xE74C3C  # Red
    CRITICAL = 0x9B59B6  # Purple


@dataclass
class EmbedField:
    """A field within a Discord embed."""

    name: str
    value: str
    inline: bool = True


@dataclass
class DiscordEmbed:
    """Discord embed structure for rich notifications."""

    title: str
    description: str = ""
    color: int = EmbedColor.INFO
    fields: list[EmbedField] = field(default_factory=list)
    timestamp: str | None = None
    footer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert embed to Discord API payload format."""
        embed: dict[str, Any] = {
            "title": self.title[:DISCORD_MAX_EMBED_TITLE_LENGTH],
        }

        if self.description:
            embed["description"] = self.description[:DISCORD_MAX_EMBED_DESCRIPTION_LENGTH]

        embed["color"] = self.color

        if self.fields:
            embed["fields"] = [
                {
                    "name": f.name[:DISCORD_MAX_FIELD_NAME_LENGTH],
                    "value": f.value[:DISCORD_MAX_FIELD_VALUE_LENGTH],
                    "inline": f.inline,
                }
                for f in self.fields[:DISCORD_MAX_EMBED_FIELDS]
            ]

        if self.timestamp:
            embed["timestamp"] = self.timestamp

        if self.footer:
            embed["footer"] = {"text": self.footer}

        return embed


class DiscordChannel:
    """Discord webhook notification channel.

    Sends messages via Discord webhook URL using httpx async client.
    Supports rich embed formatting and handles API errors and rate limits.
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "Trading Bot",
        avatar_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the Discord channel.

        Args:
            webhook_url: Discord webhook URL for sending messages.
            username: Display name for the bot in Discord.
            avatar_url: Optional avatar URL for the bot.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries in seconds.
            timeout: HTTP request timeout in seconds.
        """
        if not webhook_url:
            raise ValueError("webhook_url must not be empty")

        self._webhook_url = webhook_url
        self._username = username
        self._avatar_url = avatar_url
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send(self, message: str, **kwargs: Any) -> bool:
        """Send a notification message via Discord webhook.

        Args:
            message: The message content to send.
            **kwargs: Additional options:
                - level: Notification level (info, success, warning, error, critical)
                - title: Optional embed title
                - fields: Optional list of EmbedField objects
                - embed: Optional pre-built DiscordEmbed object

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        level = kwargs.get("level", "info")
        title = kwargs.get("title")
        fields = kwargs.get("fields")
        embed = kwargs.get("embed")

        if embed is not None:
            payload = self._build_embed_payload(embed)
        elif title or fields:
            color = self._level_to_color(level)
            discord_embed = DiscordEmbed(
                title=title or "Notification",
                description=self._truncate(message, DISCORD_MAX_EMBED_DESCRIPTION_LENGTH),
                color=color,
                fields=fields or [],
                timestamp=datetime.now(timezone.utc).isoformat(),
                footer="Trading System",
            )
            payload = self._build_embed_payload(discord_embed)
        else:
            payload = self._build_content_payload(message)

        return await self._send_with_retry(payload)

    async def send_embed(self, embed: DiscordEmbed) -> bool:
        """Send a rich embed notification.

        Args:
            embed: The DiscordEmbed to send.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        payload = self._build_embed_payload(embed)
        return await self._send_with_retry(payload)

    def _build_content_payload(self, message: str) -> dict[str, Any]:
        """Build a simple content payload."""
        payload: dict[str, Any] = {
            "content": self._truncate(message, DISCORD_MAX_CONTENT_LENGTH),
            "username": self._username,
        }
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url
        return payload

    def _build_embed_payload(self, embed: DiscordEmbed) -> dict[str, Any]:
        """Build an embed payload."""
        payload: dict[str, Any] = {
            "embeds": [embed.to_dict()],
            "username": self._username,
        }
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url
        return payload

    async def _send_with_retry(self, payload: dict[str, Any]) -> bool:
        """Send payload with retry logic and rate limit handling.

        Args:
            payload: The JSON payload to send to the webhook.

        Returns:
            True if sent successfully, False if all retries exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                client = await self._get_client()
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                )

                if response.status_code in (200, 204):
                    return True

                if response.status_code == 429:
                    # Rate limited - respect retry_after header
                    retry_after = self._get_retry_after(response)
                    logger.warning(
                        "Discord rate limited, waiting %.2f seconds",
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    # Server error - retry with backoff
                    logger.warning(
                        "Discord server error %d on attempt %d/%d",
                        response.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    last_error = httpx.HTTPStatusError(
                        f"Discord returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    await asyncio.sleep(self._retry_delay * (2**attempt))
                    continue

                # Client error (4xx except 429) - don't retry
                logger.error(
                    "Discord client error %d: %s",
                    response.status_code,
                    response.text,
                )
                return False

            except httpx.TimeoutException as e:
                logger.warning(
                    "Discord request timeout on attempt %d/%d: %s",
                    attempt + 1,
                    self._max_retries,
                    str(e),
                )
                last_error = e
                await asyncio.sleep(self._retry_delay * (2**attempt))

            except httpx.HTTPError as e:
                logger.warning(
                    "Discord HTTP error on attempt %d/%d: %s",
                    attempt + 1,
                    self._max_retries,
                    str(e),
                )
                last_error = e
                await asyncio.sleep(self._retry_delay * (2**attempt))

        logger.error(
            "Discord notification failed after %d attempts. Last error: %s",
            self._max_retries,
            str(last_error),
        )
        return False

    @staticmethod
    def _get_retry_after(response: httpx.Response) -> float:
        """Extract retry_after value from rate limit response.

        Discord returns retry_after in the JSON body (in seconds)
        or as a header.
        """
        try:
            data = response.json()
            return float(data.get("retry_after", 1.0))
        except Exception:
            # Fallback to header or default
            header_val = response.headers.get("Retry-After", "1.0")
            try:
                return float(header_val)
            except (ValueError, TypeError):
                return 1.0

    @staticmethod
    def _level_to_color(level: str) -> int:
        """Map notification level to embed color."""
        color_map = {
            "info": EmbedColor.INFO,
            "success": EmbedColor.SUCCESS,
            "warning": EmbedColor.WARNING,
            "error": EmbedColor.ERROR,
            "critical": EmbedColor.CRITICAL,
        }
        return color_map.get(level, EmbedColor.INFO)

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate text to max_length, appending '...' if truncated."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."
