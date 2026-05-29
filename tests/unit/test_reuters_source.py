"""Unit tests for the Reuters news source adapter.

Tests API key authentication, connection lifecycle, article parsing,
reconnection logic, and health check behavior.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.config.constants import (
    NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS,
    RECONNECT_MAX_ATTEMPTS,
    SOURCE_CREDIBILITY_TIER1,
)
from src.news.sources.base import RawArticle
from src.news.sources.reuters import ReutersSource


class TestReutersSourceInit:
    """Tests for ReutersSource initialization."""

    def test_default_initialization(self) -> None:
        """Source initializes with correct defaults."""
        source = ReutersSource(api_key="test-key")
        assert source.name == "Reuters"
        assert source.tier == SOURCE_CREDIBILITY_TIER1
        assert source.api_key == "test-key"
        assert source.is_connected is False
        assert source.last_message_time is None
        assert source.reconnect_count == 0

    def test_custom_urls(self) -> None:
        """Custom API and health URLs are stored."""
        source = ReutersSource(
            api_key="key",
            api_url="wss://custom.reuters.com/stream",
            health_url="https://custom.reuters.com/health",
        )
        assert source._api_url == "wss://custom.reuters.com/stream"
        assert source._health_url == "https://custom.reuters.com/health"

    def test_default_urls(self) -> None:
        """Default URLs are used when not specified."""
        source = ReutersSource(api_key="key")
        assert source._api_url == ReutersSource.DEFAULT_WS_URL
        assert source._health_url == ReutersSource.DEFAULT_HEALTH_URL

    def test_tier_is_tier1(self) -> None:
        """Reuters is a tier-1 source with credibility weight 1.0."""
        source = ReutersSource()
        assert source.tier == 1.0
        assert source.tier == SOURCE_CREDIBILITY_TIER1

    def test_custom_reconnect_params(self) -> None:
        """Custom reconnection parameters are stored."""
        source = ReutersSource(
            api_key="key",
            max_reconnect_attempts=10,
            reconnect_base_delay=5.0,
        )
        assert source._max_reconnect_attempts == 10
        assert source._reconnect_base_delay == 5.0


class TestReutersSourceAuth:
    """Tests for API key authentication."""

    def test_build_auth_headers_with_key(self) -> None:
        """Auth headers include Bearer token when API key is set."""
        source = ReutersSource(api_key="my-secret-key")
        headers = source._build_auth_headers()
        assert headers["Authorization"] == "Bearer my-secret-key"
        assert "X-Reuters-Source" in headers
        assert headers["Accept"] == "application/json"

    def test_build_auth_headers_without_key_raises(self) -> None:
        """Building auth headers without API key raises ValueError."""
        source = ReutersSource()
        with pytest.raises(ValueError, match="API key is required"):
            source._build_auth_headers()


class TestReutersSourceConnection:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_without_api_key_warns(self) -> None:
        """Connecting without API key logs warning but succeeds."""
        source = ReutersSource()
        await source.connect()
        assert source.is_connected is True
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self) -> None:
        """Connecting with API key succeeds."""
        source = ReutersSource(api_key="test-key")
        await source.connect()
        assert source.is_connected is True
        assert source._shutting_down is False
        assert source._listener_task is not None
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self) -> None:
        """Disconnect cancels tasks and resets state."""
        source = ReutersSource(api_key="test-key")
        await source.connect()
        assert source._listener_task is not None

        await source.disconnect()
        assert source.is_connected is False
        assert source._listener_task is None
        assert source._ws_connection is None
        assert source._shutting_down is True
        assert source.reconnect_count == 0

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self) -> None:
        """Connection failure raises ConnectionError."""
        source = ReutersSource(api_key="test-key")

        with patch.object(
            source, "_establish_connection", side_effect=OSError("Network unreachable")
        ):
            with pytest.raises(ConnectionError, match="Unable to establish"):
                await source.connect()
            assert source.is_connected is False


class TestReutersSourceHealthCheck:
    """Tests for health check behavior."""

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(self) -> None:
        """Health check returns False when not connected."""
        source = ReutersSource(api_key="test-key")
        assert await source.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_when_connected_no_messages(self) -> None:
        """Health check returns True when connected but no messages yet."""
        source = ReutersSource(api_key="test-key")
        await source.connect()
        assert await source.health_check() is True
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_recent_message(self) -> None:
        """Health check returns True when last message is recent."""
        source = ReutersSource(api_key="test-key")
        await source.connect()
        source._last_message_time = datetime.now(timezone.utc)
        assert await source.health_check() is True
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_stale_message(self) -> None:
        """Health check returns False when last message is stale."""
        source = ReutersSource(api_key="test-key")
        await source.connect()
        # Set last message time to beyond the staleness threshold
        source._last_message_time = datetime.now(timezone.utc) - timedelta(
            seconds=NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS + 10
        )
        assert await source.health_check() is False
        await source.disconnect()


class TestReutersSourceArticleParsing:
    """Tests for article parsing and normalization."""

    def setup_method(self) -> None:
        self.source = ReutersSource(api_key="test-key")

    def test_parse_valid_article(self) -> None:
        """Valid article message is parsed correctly."""
        message = {
            "type": "article",
            "headline": "Fed raises rates by 25bps",
            "body": "The Federal Reserve raised interest rates today.",
            "published_at": "2024-01-15T10:30:00Z",
            "category": "monetary_policy",
            "metadata": {"story_id": "abc123", "urgency": "high"},
        }
        article = self.source._parse_message(message)
        assert article is not None
        assert article.headline == "Fed raises rates by 25bps"
        assert article.body == "The Federal Reserve raised interest rates today."
        assert article.source_name == "Reuters"
        assert article.source_tier == SOURCE_CREDIBILITY_TIER1
        assert article.category == "monetary_policy"
        assert article.metadata == {"story_id": "abc123", "urgency": "high"}
        assert article.published_at == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_parse_breaking_news(self) -> None:
        """Breaking news type is also parsed."""
        message = {
            "type": "breaking",
            "headline": "Major earthquake hits region",
            "body": "A 7.2 magnitude earthquake struck...",
            "category": "natural_disaster",
        }
        article = self.source._parse_message(message)
        assert article is not None
        assert article.headline == "Major earthquake hits region"

    def test_parse_news_type(self) -> None:
        """News type messages are parsed."""
        message = {
            "type": "news",
            "headline": "Quarterly earnings beat expectations",
            "body": "Company X reported...",
        }
        article = self.source._parse_message(message)
        assert article is not None

    def test_skip_heartbeat_message(self) -> None:
        """Heartbeat messages are skipped."""
        message = {"type": "heartbeat", "timestamp": "2024-01-15T10:30:00Z"}
        article = self.source._parse_message(message)
        assert article is None

    def test_skip_subscription_ack(self) -> None:
        """Subscription acknowledgment messages are skipped."""
        message = {"type": "subscription_ack", "channel": "markets"}
        article = self.source._parse_message(message)
        assert article is None

    def test_skip_empty_headline(self) -> None:
        """Messages with empty headlines are skipped."""
        message = {"type": "article", "headline": "", "body": "Some body"}
        article = self.source._parse_message(message)
        assert article is None

    def test_parse_with_timestamp_field(self) -> None:
        """Fallback to 'timestamp' field for published_at."""
        message = {
            "type": "article",
            "headline": "Test",
            "body": "Body",
            "timestamp": "2024-06-01T14:00:00+00:00",
        }
        article = self.source._parse_message(message)
        assert article is not None
        assert article.published_at == datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)

    def test_parse_with_invalid_timestamp(self) -> None:
        """Invalid timestamp falls back to current time."""
        message = {
            "type": "article",
            "headline": "Test",
            "body": "Body",
            "published_at": "not-a-date",
        }
        article = self.source._parse_message(message)
        assert article is not None
        # Should use current time as fallback
        assert (datetime.now(timezone.utc) - article.published_at).total_seconds() < 5

    def test_parse_with_string_metadata(self) -> None:
        """String metadata is parsed as JSON."""
        message = {
            "type": "article",
            "headline": "Test",
            "body": "Body",
            "metadata": '{"key": "value"}',
        }
        article = self.source._parse_message(message)
        assert article is not None
        assert article.metadata == {"key": "value"}

    def test_parse_with_invalid_string_metadata(self) -> None:
        """Invalid JSON metadata string results in empty dict."""
        message = {
            "type": "article",
            "headline": "Test",
            "body": "Body",
            "metadata": "not-json",
        }
        article = self.source._parse_message(message)
        assert article is not None
        assert article.metadata == {}


class TestReutersSourceCallbacks:
    """Tests for article callback notification."""

    @pytest.mark.asyncio
    async def test_process_message_notifies_callbacks(self) -> None:
        """_process_message creates article and notifies callbacks."""
        source = ReutersSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)
        await source._process_message("Breaking news", "Full story here", "markets")

        assert len(received) == 1
        assert received[0].headline == "Breaking news"
        assert received[0].body == "Full story here"
        assert received[0].source_name == "Reuters"
        assert received[0].source_tier == SOURCE_CREDIBILITY_TIER1
        assert received[0].category == "markets"

    @pytest.mark.asyncio
    async def test_process_message_updates_last_message_time(self) -> None:
        """_process_message updates the last_message_time."""
        source = ReutersSource(api_key="test-key")
        assert source.last_message_time is None

        await source._process_message("Test", "Body")
        assert source.last_message_time is not None
        elapsed = (datetime.now(timezone.utc) - source.last_message_time).total_seconds()
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self) -> None:
        """All registered callbacks are invoked."""
        source = ReutersSource(api_key="test-key")
        results: list[str] = []

        async def cb1(article: RawArticle) -> None:
            results.append("cb1")

        async def cb2(article: RawArticle) -> None:
            results.append("cb2")

        source.on_article_received(cb1)
        source.on_article_received(cb2)
        await source._process_message("Test", "Body")

        assert "cb1" in results
        assert "cb2" in results


class TestReutersSourceReconnection:
    """Tests for reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_with_backoff_succeeds(self) -> None:
        """Reconnection succeeds after initial failure."""
        source = ReutersSource(
            api_key="test-key",
            max_reconnect_attempts=3,
            reconnect_base_delay=0.01,  # Fast for testing
        )

        attempt_count = 0

        async def mock_establish():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise OSError("Connection refused")
            source._ws_connection = True

        with patch.object(source, "_establish_connection", side_effect=mock_establish):
            await source._reconnect_with_backoff()

        assert source.is_connected is True
        assert source.reconnect_count == 0  # Reset on success
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_reconnect_exhausts_attempts(self) -> None:
        """Reconnection gives up after max attempts."""
        source = ReutersSource(
            api_key="test-key",
            max_reconnect_attempts=3,
            reconnect_base_delay=0.01,
        )

        with patch.object(
            source,
            "_establish_connection",
            side_effect=OSError("Connection refused"),
        ):
            await source._reconnect_with_backoff()

        assert source.is_connected is False

    @pytest.mark.asyncio
    async def test_reconnect_stops_on_shutdown(self) -> None:
        """Reconnection stops if shutdown is requested."""
        source = ReutersSource(
            api_key="test-key",
            max_reconnect_attempts=5,
            reconnect_base_delay=0.01,
        )
        source._shutting_down = True

        with patch.object(
            source,
            "_establish_connection",
            side_effect=OSError("Connection refused"),
        ) as mock_conn:
            await source._reconnect_with_backoff()
            mock_conn.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_disconnect_starts_reconnection(self) -> None:
        """Disconnect handler starts reconnection task."""
        source = ReutersSource(
            api_key="test-key",
            max_reconnect_attempts=1,
            reconnect_base_delay=0.01,
        )
        source._shutting_down = False

        with patch.object(source, "_reconnect_with_backoff", new_callable=AsyncMock):
            await source._handle_disconnect()
            assert source._reconnect_task is not None

        # Clean up
        if source._reconnect_task:
            source._reconnect_task.cancel()
            try:
                await source._reconnect_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_handle_disconnect_skipped_on_shutdown(self) -> None:
        """Disconnect handler does nothing during shutdown."""
        source = ReutersSource(api_key="test-key")
        source._shutting_down = True

        await source._handle_disconnect()
        assert source._reconnect_task is None
