"""Unit tests for the NewsEngine orchestrator functionality.

Tests multi-source ingestion, 30-second max ingestion delay enforcement,
source health monitoring (60s interval, failover after 5min unavailability),
article deduplication, and event bus integration.

Validates: Requirements 23.1, 23.17
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.constants import (
    NEWS_MAX_INGESTION_DELAY_SECONDS,
    NEWS_MIN_SOURCES,
    NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS,
    NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS,
    SOURCE_CREDIBILITY_SOCIAL,
    SOURCE_CREDIBILITY_TIER1,
    SOURCE_CREDIBILITY_TIER2,
)
from src.core.event_bus import Event, EventBus, NEWS_ARTICLE_RECEIVED
from src.news.news_engine import NewsEngine, SourceHealthStatus
from src.news.sources.base import NewsSource, RawArticle


# =============================================================================
# Test Helpers
# =============================================================================


class MockNewsSource(NewsSource):
    """Mock news source for testing that can be controlled."""

    def __init__(
        self,
        name: str = "MockSource",
        tier: float = SOURCE_CREDIBILITY_TIER1,
        connect_raises: bool = False,
        health_check_result: bool = True,
    ) -> None:
        super().__init__(name=name, tier=tier)
        self._connect_raises = connect_raises
        self._health_check_result = health_check_result

    async def connect(self) -> None:
        if self._connect_raises:
            raise ConnectionError("Mock connection failure")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe(self, topics: list[str]) -> None:
        if not self._connected:
            raise ConnectionError("Not connected")
        self._subscribed_topics.extend(topics)

    async def health_check(self) -> bool:
        return self._health_check_result

    def set_health(self, healthy: bool) -> None:
        """Control health check result for testing."""
        self._health_check_result = healthy


def make_article(
    source: str = "Reuters",
    headline: str = "Test headline",
    body: str = "Test body content",
    published_at: datetime | None = None,
    received_at: datetime | None = None,
    category: str | None = None,
    source_tier: float = SOURCE_CREDIBILITY_TIER1,
) -> dict[str, Any]:
    """Create a test article dict."""
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid.uuid4()),
        "source": source,
        "source_tier": source_tier,
        "headline": headline,
        "body": body,
        "published_at": published_at or now,
        "received_at": received_at or now,
        "category": category,
    }


# =============================================================================
# Multi-Source Ingestion Tests
# =============================================================================


class TestMultiSourceIngestion:
    """Tests for multi-source ingestion (min 3 sources)."""

    @pytest.mark.asyncio
    async def test_engine_requires_min_3_sources_config(self) -> None:
        """Engine is configured with minimum 3 sources by default."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        assert engine._min_sources == NEWS_MIN_SOURCES
        assert NEWS_MIN_SOURCES == 3

    @pytest.mark.asyncio
    async def test_engine_starts_with_3_sources(self) -> None:
        """Engine starts successfully with 3 healthy sources."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        try:
            assert engine.is_running is True
            assert len(engine.healthy_sources) == 3
            assert "Reuters" in engine.healthy_sources
            assert "Bloomberg" in engine.healthy_sources
            assert "SocialMedia" in engine.healthy_sources
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_handles_source_connect_failure(self) -> None:
        """Engine continues with remaining sources if one fails to connect."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1, connect_raises=True),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        try:
            assert engine.is_running is True
            assert len(engine.healthy_sources) == 2
            assert "Reuters" in engine.healthy_sources
            assert "Bloomberg" not in engine.healthy_sources
            assert "SocialMedia" in engine.healthy_sources
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_all_sources_down_detection(self) -> None:
        """Engine detects when all sources are unavailable."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1, connect_raises=True),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1, connect_raises=True),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL, connect_raises=True),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        try:
            assert engine.is_all_sources_down() is True
            assert len(engine.healthy_sources) == 0
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_registers_callbacks_on_sources(self) -> None:
        """Engine registers article callbacks on all sources during start."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        try:
            for source in sources:
                assert len(source._callbacks) == 1
        finally:
            await engine.stop()


# =============================================================================
# Ingestion Delay Tests (30-second max)
# =============================================================================


class TestIngestionDelay:
    """Tests for 30-second max ingestion delay enforcement."""

    @pytest.mark.asyncio
    async def test_article_within_delay_limit_processed(self) -> None:
        """Articles within 30-second delay are processed normally."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        now = datetime.now(timezone.utc)
        article = make_article(
            published_at=now - timedelta(seconds=10),
            received_at=now,
        )

        await engine.on_article_received(article)

        assert article.get("ingestion_delay_seconds") == pytest.approx(10.0, abs=1.0)
        assert article.get("delayed") is None or article.get("delayed") is False
        assert engine.articles_processed == 1

    @pytest.mark.asyncio
    async def test_article_exceeding_delay_flagged(self) -> None:
        """Articles exceeding 30-second delay are flagged but still processed."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        now = datetime.now(timezone.utc)
        article = make_article(
            published_at=now - timedelta(seconds=45),
            received_at=now,
        )

        await engine.on_article_received(article)

        assert article["ingestion_delay_seconds"] == pytest.approx(45.0, abs=1.0)
        assert article["delayed"] is True
        assert engine.articles_delayed == 1
        assert engine.articles_processed == 1

    @pytest.mark.asyncio
    async def test_article_at_exact_delay_limit_not_flagged(self) -> None:
        """Articles at exactly 30 seconds are not flagged as delayed."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        now = datetime.now(timezone.utc)
        article = make_article(
            published_at=now - timedelta(seconds=30),
            received_at=now,
        )

        await engine.on_article_received(article)

        assert article["ingestion_delay_seconds"] == pytest.approx(30.0, abs=1.0)
        assert article.get("delayed") is None or article.get("delayed") is False
        assert engine.articles_delayed == 0

    @pytest.mark.asyncio
    async def test_article_without_published_at_skips_delay_check(self) -> None:
        """Articles without published_at skip the delay check."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        article = make_article()
        article["published_at"] = None

        await engine.on_article_received(article)

        assert "ingestion_delay_seconds" not in article
        assert engine.articles_delayed == 0
        assert engine.articles_processed == 1

    @pytest.mark.asyncio
    async def test_max_ingestion_delay_constant_is_30(self) -> None:
        """The max ingestion delay constant is 30 seconds."""
        assert NEWS_MAX_INGESTION_DELAY_SECONDS == 30


# =============================================================================
# Source Health Monitoring Tests
# =============================================================================


class TestSourceHealthMonitoring:
    """Tests for source health monitoring (60s interval, 5min failover)."""

    def test_health_check_interval_is_60_seconds(self) -> None:
        """Health check interval constant is 60 seconds."""
        assert NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS == 60

    def test_unavailable_threshold_is_300_seconds(self) -> None:
        """Source unavailable threshold is 300 seconds (5 minutes)."""
        assert NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS == 300

    def test_source_health_status_initial_state(self) -> None:
        """SourceHealthStatus starts as unhealthy."""
        status = SourceHealthStatus("TestSource")
        assert status.is_healthy is False
        assert status.last_successful_check is None
        assert status.failed_since is None
        assert status.failover_logged is False

    def test_source_health_status_mark_healthy(self) -> None:
        """Marking a source healthy updates timestamps."""
        status = SourceHealthStatus("TestSource")
        now = datetime.now(timezone.utc)
        status.mark_healthy(now)

        assert status.is_healthy is True
        assert status.last_successful_check == now
        assert status.failed_since is None
        assert status.failover_logged is False

    def test_source_health_status_mark_unhealthy(self) -> None:
        """Marking a source unhealthy records failure start time."""
        status = SourceHealthStatus("TestSource")
        now = datetime.now(timezone.utc)
        status.mark_unhealthy(now)

        assert status.is_healthy is False
        assert status.failed_since == now

    def test_source_health_status_failover_not_required_initially(self) -> None:
        """Failover not required when source just became unhealthy."""
        status = SourceHealthStatus("TestSource")
        now = datetime.now(timezone.utc)
        status.mark_unhealthy(now)

        assert status.is_failover_required(now) is False

    def test_source_health_status_failover_required_after_5min(self) -> None:
        """Failover required after source unavailable for 5+ minutes."""
        status = SourceHealthStatus("TestSource")
        failed_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        status.mark_unhealthy(failed_time)

        assert status.is_failover_required() is True

    def test_source_health_status_failover_not_required_at_4min(self) -> None:
        """Failover not required at 4 minutes of unavailability."""
        status = SourceHealthStatus("TestSource")
        failed_time = datetime.now(timezone.utc) - timedelta(minutes=4)
        status.mark_unhealthy(failed_time)

        assert status.is_failover_required() is False

    def test_source_health_status_recovery_resets_failover(self) -> None:
        """Recovering a source resets the failover state."""
        status = SourceHealthStatus("TestSource")
        failed_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        status.mark_unhealthy(failed_time)
        status.failover_logged = True

        # Source recovers
        status.mark_healthy()

        assert status.is_healthy is True
        assert status.failed_since is None
        assert status.failover_logged is False

    @pytest.mark.asyncio
    async def test_health_check_marks_source_healthy(self) -> None:
        """Health check marks source as healthy when check passes."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            # All sources healthy after start
            for source in sources:
                status = engine.get_source_health(source.name)
                assert status is not None
                assert status.is_healthy is True
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_health_check_detects_unhealthy_source(self) -> None:
        """Health check detects when a source becomes unhealthy."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            # Make Reuters unhealthy
            sources[0].set_health(False)

            # Perform health checks manually
            await engine._perform_health_checks()

            status = engine.get_source_health("Reuters")
            assert status is not None
            assert status.is_healthy is False
            assert "Reuters" not in engine.healthy_sources
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_failover_logged_after_5min_unavailability(self) -> None:
        """Failover is logged when source unavailable for 5+ minutes."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            # Make Reuters unhealthy
            sources[0].set_health(False)

            # Simulate that it's been unhealthy for 6 minutes
            status = engine.get_source_health("Reuters")
            assert status is not None
            status.mark_unhealthy(
                datetime.now(timezone.utc) - timedelta(minutes=6)
            )

            # Perform health check
            await engine._perform_health_checks()

            assert status.failover_logged is True
            assert "Reuters" not in engine.healthy_sources
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_failover_not_logged_twice(self) -> None:
        """Failover is only logged once per failure episode."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            sources[0].set_health(False)
            status = engine.get_source_health("Reuters")
            assert status is not None
            status.mark_unhealthy(
                datetime.now(timezone.utc) - timedelta(minutes=6)
            )

            # First check logs failover
            await engine._perform_health_checks()
            assert status.failover_logged is True

            # Second check doesn't re-log
            await engine._perform_health_checks()
            assert status.failover_logged is True  # Still True, not re-logged
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_source_recovery_after_failover(self) -> None:
        """Source can recover after being marked as failed."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            # Make Reuters unhealthy and trigger failover
            sources[0].set_health(False)
            status = engine.get_source_health("Reuters")
            assert status is not None
            status.mark_unhealthy(
                datetime.now(timezone.utc) - timedelta(minutes=6)
            )
            await engine._perform_health_checks()
            assert status.failover_logged is True

            # Reuters recovers
            sources[0].set_health(True)
            await engine._perform_health_checks()

            assert status.is_healthy is True
            assert status.failover_logged is False
            assert "Reuters" in engine.healthy_sources
        finally:
            await engine.stop()


# =============================================================================
# Article Deduplication Tests
# =============================================================================


class TestArticleDeduplication:
    """Tests for article deduplication (same story from multiple sources)."""

    @pytest.mark.asyncio
    async def test_duplicate_article_filtered(self) -> None:
        """Same article body from different sources is deduplicated."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        # First article
        article1 = make_article(
            source="Reuters",
            headline="Breaking: Fed raises rates",
            body="The Federal Reserve raised interest rates by 25 basis points.",
        )
        await engine.on_article_received(article1)
        assert engine.articles_processed == 1

        # Same body from different source
        article2 = make_article(
            source="Bloomberg",
            headline="Breaking: Fed raises rates",
            body="The Federal Reserve raised interest rates by 25 basis points.",
        )
        await engine.on_article_received(article2)

        # Second article should be deduplicated
        assert engine.articles_processed == 1
        assert engine.articles_deduplicated == 1

    @pytest.mark.asyncio
    async def test_different_articles_not_deduplicated(self) -> None:
        """Different article bodies are not deduplicated."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        article1 = make_article(
            source="Reuters",
            body="The Federal Reserve raised interest rates.",
        )
        article2 = make_article(
            source="Bloomberg",
            body="European Central Bank holds rates steady.",
        )

        await engine.on_article_received(article1)
        await engine.on_article_received(article2)

        assert engine.articles_processed == 2
        assert engine.articles_deduplicated == 0

    @pytest.mark.asyncio
    async def test_dedup_cache_bounded(self) -> None:
        """Deduplication cache doesn't grow unbounded."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        # Process many unique articles
        for i in range(100):
            article = make_article(body=f"Unique article body number {i}")
            await engine.on_article_received(article)

        assert engine.articles_processed == 100
        assert len(engine._seen_articles) <= engine._DEDUP_CACHE_MAX_SIZE


# =============================================================================
# Event Bus Integration Tests
# =============================================================================


class TestEventBusIntegration:
    """Tests for event bus publishing of news events."""

    @pytest.mark.asyncio
    async def test_article_published_to_event_bus(self) -> None:
        """Processed articles are published to the event bus."""
        mock_bus = AsyncMock(spec=EventBus)
        mock_bus.publish = AsyncMock(return_value=1)

        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = make_article(
            headline="Market rally continues",
            body="Stocks surge on positive earnings.",
            category="earnings",
        )
        await engine.on_article_received(article)

        # Verify event was published
        mock_bus.publish.assert_called()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == NEWS_ARTICLE_RECEIVED

    @pytest.mark.asyncio
    async def test_no_publish_without_event_bus(self) -> None:
        """No error when event bus is not configured."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources, event_bus=None)

        article = make_article()
        # Should not raise
        await engine.on_article_received(article)
        assert engine.articles_processed == 1

    @pytest.mark.asyncio
    async def test_duplicate_not_published_to_event_bus(self) -> None:
        """Deduplicated articles are not published to the event bus."""
        mock_bus = AsyncMock(spec=EventBus)
        mock_bus.publish = AsyncMock(return_value=1)

        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article1 = make_article(body="Same content here")
        article2 = make_article(body="Same content here", source="Bloomberg")

        await engine.on_article_received(article1)
        await engine.on_article_received(article2)

        # Only one publish call (for the first article)
        assert mock_bus.publish.call_count == 1


# =============================================================================
# Engine Lifecycle Tests
# =============================================================================


class TestEngineLifecycle:
    """Tests for engine start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        """Calling start() twice doesn't create duplicate tasks."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        task1 = engine._health_check_task
        await engine.start()  # Second call should be no-op
        task2 = engine._health_check_task

        assert task1 is task2
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self) -> None:
        """Stop disconnects all sources and cancels tasks."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        assert engine.is_running is True

        await engine.stop()
        assert engine.is_running is False
        assert engine._health_check_task is None
        assert len(engine.healthy_sources) == 0

    @pytest.mark.asyncio
    async def test_raw_article_callback_integration(self) -> None:
        """RawArticle from source is properly converted and processed."""
        sources = [
            MockNewsSource("Reuters", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("Bloomberg", SOURCE_CREDIBILITY_TIER1),
            MockNewsSource("SocialMedia", SOURCE_CREDIBILITY_SOCIAL),
        ]
        engine = NewsEngine(sources=sources)
        await engine.start()

        try:
            # Simulate a source delivering an article via callback
            raw = RawArticle(
                headline="Test breaking news",
                body="Important financial news content.",
                source_name="Reuters",
                source_tier=SOURCE_CREDIBILITY_TIER1,
                published_at=datetime.now(timezone.utc),
                category="monetary_policy",
            )
            await engine._on_raw_article_received(raw)

            assert engine.articles_processed == 1
        finally:
            await engine.stop()
