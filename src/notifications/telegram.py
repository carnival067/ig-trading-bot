"""Telegram notification channel implementation.

Sends messages via the Telegram Bot API using httpx async client.
Supports trade alerts, kill switch alerts, crisis alerts, and HFT circuit breaker alerts.

Implements the NotificationChannel protocol defined in notification_service.py:
- name property (str)
- async send(message: str) -> bool

Validates: Requirements 17.1
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Telegram message length limit
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramChannel:
    """Telegram Bot API notification channel.

    Sends messages via the Telegram Bot API using httpx async client.
    Messages are formatted with Markdown for readability.
    Implements the NotificationChannel protocol.
    """

    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        timeout: float = 10.0,
        disable_notification: bool = False,
    ) -> None:
        """Initialize the Telegram channel.

        Args:
            bot_token: Telegram Bot API token.
            chat_id: Target chat ID for messages.
            timeout: HTTP request timeout in seconds.
            disable_notification: If True, send messages silently.
        """
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._disable_notification = disable_notification
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Channel name identifier for routing."""
        return "telegram"

    @property
    def _api_url(self) -> str:
        """Construct the Telegram sendMessage API URL."""
        return f"{self.BASE_URL}/bot{self._bot_token}/sendMessage"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _truncate_message(self, message: str) -> str:
        """Truncate message to Telegram's maximum length.

        If the message exceeds 4096 characters, it is truncated with
        an ellipsis indicator.

        Args:
            message: The message to potentially truncate.

        Returns:
            The message, truncated if necessary.
        """
        if len(message) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            return message

        truncation_suffix = "\n\n... (message truncated)"
        max_content_length = TELEGRAM_MAX_MESSAGE_LENGTH - len(truncation_suffix)
        return message[:max_content_length] + truncation_suffix

    async def send(self, message: str) -> bool:
        """Send a message via Telegram Bot API.

        Messages are sent with Markdown parse mode. If the message exceeds
        4096 characters, it is truncated.

        Implements the NotificationChannel protocol.

        Args:
            message: The formatted message to send.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        truncated_message = self._truncate_message(message)

        payload = {
            "chat_id": self._chat_id,
            "text": truncated_message,
            "parse_mode": "Markdown",
            "disable_notification": self._disable_notification,
        }

        try:
            client = await self._get_client()
            response = await client.post(self._api_url, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("ok"):
                    logger.info(
                        "Telegram message sent successfully",
                        extra={"chat_id": self._chat_id},
                    )
                    return True
                else:
                    error_desc = response_data.get("description", "Unknown API error")
                    logger.warning(
                        "Telegram API returned error: %s",
                        error_desc,
                        extra={"chat_id": self._chat_id},
                    )
                    return False
            else:
                logger.warning(
                    "Telegram API request failed with status %d",
                    response.status_code,
                    extra={"chat_id": self._chat_id},
                )
                return False

        except httpx.TimeoutException:
            logger.error(
                "Telegram request timed out",
                extra={"chat_id": self._chat_id, "timeout": self._timeout},
            )
            return False

        except httpx.HTTPError as e:
            logger.error(
                "Telegram HTTP error: %s",
                e,
                extra={"chat_id": self._chat_id},
            )
            return False

        except Exception as e:
            logger.error(
                "Unexpected error sending Telegram message: %s",
                e,
                extra={"chat_id": self._chat_id},
            )
            return False
