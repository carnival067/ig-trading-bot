"""Unit tests for the Bloomberg B-PIPE news source adapter.

Tests cover:
- Connection lifecycle (connect, disconnect, reconnection)
- Authentication behavior
- Topic subscription
- B-PIPE message parsing and normalization
- Health check logic
- Article callback notification
- Connection state management
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.config.constants import SOURCE_CREDIBILITY_TIER1
from src.news.sources.base import RawArticle
from src.news.sources.bloomberg import (
    BloombergSource,
    BPipeConnectionState,
    BPipeMessage,
)


class TestBloombergSourceInit:
    """Tests for BloombergSource initialization."""

    def test_default_initialization(self) -> None:
        """Source initializes with correct defaults."""
        source = BloombergSource()
        assert source.name == "Bloomberg"
        assert source.tier == SOURCE_CREDIBILITY_TIER1
        assert source.state == BPipeConnectionState.DISCONNECTED
        assert source.is_connected is False
        assert source.is_authenticated is False
        assert source.message_count == 0
        assert source.last_message_at is None

    def test_custom_api_key(self) -> None:
        """Source accepts custom API key."""
        source = BloombergSource(api_key="test-key-123")
        assert source._api_key == "test-key-123"

    def test_custom_host_and_port(self) -> None:
        """Source accepts custom host and port."""
        source = BloombergSource(host="custom.host.com", port=9999)
        assert source._host == "custom.host.com"
        assert source._port == 9999

    def test_custom_api_url(self) -> None:
        """Source accepts custom API URL."""
        source = BloombergSource(api_url="wss://custom.endpoint/stream")
        assert source._api_url == "wss://custom.endpoint/stream"


class TestBloombergSourceConnect:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_sets_state(self) -> None:
        """Connect transitions to STREAMING state."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            assert source.is_connected is True
            assert source.is_authenticated is True
            assert source.state == BPipeConnectionState.STREAMING
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_connect_subscribes_default_topics(self) -> None:
        """Connect subscribes to default news topics."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            assert len(source._bpipe_topics) > 0
            assert "TOP_NEWS" in source._bpipe_topics
            assert "MARKET_MOVING" in source._bpipe_topics
            assert "ECONOMICS" in source._bpipe_topics
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        """Calling connect when already connected is a no-op."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            # Second connect should not raise
            await source.connect()
            assert source.is_connected is True
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_resets_state(self) -> None:
        """Disconnect resets all connection state."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        await source.disconnect()

        assert source.is_connected is False
        assert source.is_authenticated is False
        assert source.state == BPipeConnectionState.DISCONNECTED
        assert len(source._bpipe_topics) == 0

    @pytest.mark.asyncio
    async def test_disconnect_without_connect(self) -> None:
        """Disconnect on unconnected source does not raise."""
        source = BloombergSource(api_key="test-key")
        await source.disconnect()
        assert source.state == BPipeConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_without_api_key_simulation_mode(self) -> None:
        """Connect without API key operates in simulation mode."""
        source = BloombergSource()
        await source.connect()
        try:
            assert source.is_connected is True
            assert source.is_authenticated is True
        finally:
            await source.disconnect()


class TestBloombergSourceSubscribe:
    """Tests for topic subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_adds_topics(self) -> None:
        """Subscribe adds topics to the subscription set."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            await source.subscribe(["CUSTOM_TOPIC_1", "CUSTOM_TOPIC_2"])
            assert "CUSTOM_TOPIC_1" in source._bpipe_topics
            assert "CUSTOM_TOPIC_2" in source._bpipe_topics
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_is_additive(self) -> None:
        """Multiple subscribe calls are additive."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            initial_count = len(source._bpipe_topics)
            await source.subscribe(["NEW_TOPIC"])
            assert len(source._bpipe_topics) == initial_count + 1
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_deduplicates(self) -> None:
        """Subscribing to same topic twice doesn't duplicate."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            count_before = len(source._bpipe_topics)
            await source.subscribe(["TOP_NEWS"])  # Already subscribed by default
            assert len(source._bpipe_topics) == count_before
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_when_disconnected_raises(self) -> None:
        """Subscribe raises ConnectionError when not connected."""
        source = BloombergSource(api_key="test-key")
        with pytest.raises(ConnectionError):
            await source.subscribe(["TOPIC"])

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_topic(self) -> None:
        """Unsubscribe removes a topic from subscriptions."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            assert "TOP_NEWS" in source._bpipe_topics
            await source.unsubscribe_topic("TOP_NEWS")
            assert "TOP_NEWS" not in source._bpipe_topics
        finally:
            await source.disconnect()


class TestBloombergSourceHealthCheck:
    """Tests for health check logic."""

    @pytest.mark.asyncio
    async def test_health_check_when_connected(self) -> None:
        """Health check returns True when connected and authenticated."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            assert await source.health_check() is True
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(self) -> None:
        """Health check returns False when disconnected."""
        source = BloombergSource(api_key="test-key")
        assert await source.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_stale_heartbeat(self) -> None:
        """Health check returns False when heartbeat is stale."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            # Simulate stale heartbeat
            source._last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=60)
            assert await source.health_check() is False
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_no_heartbeat_yet(self) -> None:
        """Health check returns True when no heartbeat has been sent yet."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            # _last_heartbeat is None initially (from base class it may be set)
            # but our own tracking hasn't set it yet
            source._last_heartbeat = None
            assert await source.health_check() is True
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_error_state(self) -> None:
        """Health check returns False in error state."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            source._state = BPipeConnectionState.ERROR
            assert await source.health_check() is False
        finally:
            source._state = BPipeConnectionState.STREAMING
            await source.disconnect()


class TestBloombergSourceMessageParsing:
    """Tests for B-PIPE message parsing and normalization."""

    def test_parse_bpipe_message_full(self) -> None:
        """Full B-PIPE message is parsed correctly."""
        source = BloombergSource(api_key="test-key")
        raw_data = {
            "STORY_ID": "BBG-12345",
            "HEADLINE": "Fed Raises Rates by 25bps",
            "BODY": "The Federal Reserve raised interest rates by 25 basis points.",
            "STORY_DT": "2024-01-15T14:30:00+00:00",
            "NEWS_CATEGORY": "CENTRAL_BANKS",
            "REGION_CD": ["US", "NA"],
            "TICKER_CD": ["EURUSD", "US10Y"],
            "URGENCY": 2,
            "EXTRA_FIELD": "extra_value",
        }

        message = source._parse_bpipe_message(raw_data)

        assert message.story_id == "BBG-12345"
        assert message.headline == "Fed Raises Rates by 25bps"
        assert message.body == "The Federal Reserve raised interest rates by 25 basis points."
        assert message.timestamp == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        assert message.category_code == "CENTRAL_BANKS"
        assert message.region_codes == ["US", "NA"]
        assert message.ticker_codes == ["EURUSD", "US10Y"]
        assert message.urgency == 2
        assert message.metadata == {"EXTRA_FIELD": "extra_value"}

    def test_parse_bpipe_message_minimal(self) -> None:
        """Minimal B-PIPE message with missing fields uses defaults."""
        source = BloombergSource(api_key="test-key")
        raw_data = {
            "HEADLINE": "Breaking news",
        }

        message = source._parse_bpipe_message(raw_data)

        assert message.story_id == ""
        assert message.headline == "Breaking news"
        assert message.body == ""
        assert message.category_code is None
        assert message.region_codes == []
        assert message.ticker_codes == []
        assert message.urgency == 0
        assert message.timestamp.tzinfo == timezone.utc

    def test_parse_bpipe_message_invalid_timestamp(self) -> None:
        """Invalid timestamp falls back to current time."""
        source = BloombergSource(api_key="test-key")
        raw_data = {
            "HEADLINE": "Test",
            "STORY_DT": "not-a-date",
        }

        message = source._parse_bpipe_message(raw_data)
        # Should use current time, not crash
        assert message.timestamp.tzinfo == timezone.utc
        assert (datetime.now(timezone.utc) - message.timestamp).total_seconds() < 2

    def test_parse_bpipe_message_naive_timestamp(self) -> None:
        """Naive timestamp gets UTC timezone attached."""
        source = BloombergSource(api_key="test-key")
        raw_data = {
            "HEADLINE": "Test",
            "STORY_DT": "2024-06-01T10:00:00",
        }

        message = source._parse_bpipe_message(raw_data)
        assert message.timestamp.tzinfo == timezone.utc
        assert message.timestamp.year == 2024

    def test_normalize_article(self) -> None:
        """BPipeMessage is normalized to RawArticle correctly."""
        source = BloombergSource(api_key="test-key")
        message = BPipeMessage(
            story_id="BBG-99999",
            headline="Oil Prices Surge",
            body="Crude oil prices surged 5% on supply concerns.",
            timestamp=datetime(2024, 3, 10, 8, 0, tzinfo=timezone.utc),
            category_code="COMMODITIES",
            region_codes=["ME"],
            ticker_codes=["CL1"],
            urgency=1,
        )

        article = source._normalize_article(message)

        assert article.headline == "Oil Prices Surge"
        assert article.body == "Crude oil prices surged 5% on supply concerns."
        assert article.source_name == "Bloomberg"
        assert article.source_tier == SOURCE_CREDIBILITY_TIER1
        assert article.published_at == datetime(2024, 3, 10, 8, 0, tzinfo=timezone.utc)
        assert article.category == "COMMODITIES"
        assert article.metadata["story_id"] == "BBG-99999"
        assert article.metadata["urgency"] == 1
        assert article.metadata["region_codes"] == ["ME"]
        assert article.metadata["ticker_codes"] == ["CL1"]


class TestBloombergSourceProcessing:
    """Tests for article processing and callback notification."""

    @pytest.mark.asyncio
    async def test_process_message_simple(self) -> None:
        """Simple message processing creates RawArticle and notifies callbacks."""
        source = BloombergSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)
        await source.connect()
        try:
            await source._process_message("Test Headline", "Test Body", "ECONOMICS")

            assert len(received) == 1
            assert received[0].headline == "Test Headline"
            assert received[0].body == "Test Body"
            assert received[0].source_name == "Bloomberg"
            assert received[0].source_tier == SOURCE_CREDIBILITY_TIER1
            assert received[0].category == "ECONOMICS"
            assert source.message_count == 1
            assert source.last_message_at is not None
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_message_with_raw_data(self) -> None:
        """Message processing with raw B-PIPE data uses full parsing."""
        source = BloombergSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)
        await source.connect()
        try:
            raw_data = {
                "STORY_ID": "BBG-TEST-001",
                "HEADLINE": "ECB Rate Decision",
                "BODY": "ECB holds rates steady.",
                "STORY_DT": "2024-06-15T12:00:00+00:00",
                "NEWS_CATEGORY": "CENTRAL_BANKS",
                "REGION_CD": ["EU"],
                "TICKER_CD": ["EURUSD"],
                "URGENCY": 1,
            }
            await source._process_message("", "", raw_data=raw_data)

            assert len(received) == 1
            assert received[0].headline == "ECB Rate Decision"
            assert received[0].metadata["story_id"] == "BBG-TEST-001"
            assert received[0].metadata["urgency"] == 1
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_raw_bpipe_data(self) -> None:
        """process_raw_bpipe_data processes and notifies callbacks."""
        source = BloombergSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)
        await source.connect()
        try:
            raw_data = {
                "STORY_ID": "BBG-LIVE-001",
                "HEADLINE": "Market Flash",
                "BODY": "S&P 500 hits all-time high.",
                "NEWS_CATEGORY": "EQUITIES",
                "URGENCY": 2,
            }
            await source.process_raw_bpipe_data(raw_data)

            assert len(received) == 1
            assert received[0].headline == "Market Flash"
            assert source.message_count == 1
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_raw_bpipe_data_empty_skipped(self) -> None:
        """Empty B-PIPE messages are skipped."""
        source = BloombergSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)
        await source.connect()
        try:
            raw_data = {"STORY_ID": "BBG-EMPTY", "HEADLINE": "", "BODY": ""}
            await source.process_raw_bpipe_data(raw_data)

            assert len(received) == 0
            assert source.message_count == 0
        finally:
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_raw_bpipe_data_when_disconnected(self) -> None:
        """B-PIPE data received while disconnected is ignored."""
        source = BloombergSource(api_key="test-key")
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)

        raw_data = {"STORY_ID": "BBG-001", "HEADLINE": "Test", "BODY": "Body"}
        await source.process_raw_bpipe_data(raw_data)

        assert len(received) == 0


class TestBloombergSourceReconnection:
    """Tests for reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnection_state_transition(self) -> None:
        """Initiating reconnection transitions to RECONNECTING state."""
        source = BloombergSource(api_key="test-key")
        await source.connect()
        try:
            # Manually trigger reconnection
            await source._initiate_reconnection()
            # Give the reconnect task a moment to start
            await asyncio.sleep(0.1)
            assert source._state in (
                BPipeConnectionState.RECONNECTING,
                BPipeConnectionState.CONNECTED,
                BPipeConnectionState.STREAMING,
            )
        finally:
            # Clean up - cancel reconnect task if still running
            if source._reconnect_task and not source._reconnect_task.done():
                source._reconnect_task.cancel()
                try:
                    await source._reconnect_task
                except asyncio.CancelledError:
                    pass
            await source.disconnect()

    @pytest.mark.asyncio
    async def test_reconnection_restores_connection(self) -> None:
        """Successful reconnection restores connected state."""
        source = BloombergSource(api_key="test-key")
        await source.connect()

        # Simulate connection loss and reconnection
        source._connected = False
        source._state = BPipeConnectionState.RECONNECTING

        # Cancel existing tasks
        if source._listener_task:
            source._listener_task.cancel()
            try:
                await source._listener_task
            except asyncio.CancelledError:
                pass
        if source._heartbeat_task:
            source._heartbeat_task.cancel()
            try:
                await source._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Run reconnect loop directly
        await source._reconnect_loop()

        assert source.is_connected is True
        assert source.is_authenticated is True
        assert source._state == BPipeConnectionState.STREAMING
        await source.disconnect()

    def test_reconnect_max_attempts_constant(self) -> None:
        """MAX_RECONNECT_ATTEMPTS is set to 5."""
        assert BloombergSource.MAX_RECONNECT_ATTEMPTS == 5

    def test_reconnect_exponential_backoff_config(self) -> None:
        """Reconnection uses exponential backoff with correct bounds."""
        assert BloombergSource.RECONNECT_BASE_DELAY_SECONDS == 1.0
        assert BloombergSource.RECONNECT_MAX_DELAY_SECONDS == 30.0
