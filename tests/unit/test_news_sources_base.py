"""Unit tests for the NewsSource abstract base class and related dataclasses.

Tests the abstract interface, dataclass models, and base class behavior
defined in src/news/sources/base.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.news.sources.base import (
    ArticleCallback,
    ImpactLevel,
    NewsArticle,
    NewsSource,
    RawArticle,
    SourceHealth,
    SourceTier,
    _tier_from_weight,
)


# --- Concrete implementation for testing ---


class MockNewsSource(NewsSource):
    """Concrete implementation of NewsSource for testing."""

    def __init__(self, name: str = "MockSource", tier: float = 1.0) -> None:
        super().__init__(name=name, tier=tier)
        self._connect_called = False
        self._disconnect_called = False
        self._subscribed: list[str] = []

    async def connect(self) -> None:
        self._connect_called = True
        self._connected = True

    async def disconnect(self) -> None:
        self._disconnect_called = True
        self._connected = False

    async def subscribe(self, topics: list[str]) -> None:
        if not self._connected:
            raise ConnectionError("Not connected")
        self._subscribed.extend(topics)
        self._subscribed_topics.extend(topics)

    async def health_check(self) -> bool:
        return self._connected


# --- Tests for SourceTier enum ---


class TestSourceTier:
    """Tests for the SourceTier enum."""

    def test_tier_values(self) -> None:
        assert SourceTier.TIER_1 == "tier-1"
        assert SourceTier.TIER_2 == "tier-2"
        assert SourceTier.SOCIAL == "social"

    def test_tier_from_weight_tier1(self) -> None:
        assert _tier_from_weight(1.0) == SourceTier.TIER_1

    def test_tier_from_weight_tier2(self) -> None:
        assert _tier_from_weight(0.7) == SourceTier.TIER_2

    def test_tier_from_weight_social(self) -> None:
        assert _tier_from_weight(0.4) == SourceTier.SOCIAL

    def test_tier_from_weight_below_social(self) -> None:
        assert _tier_from_weight(0.2) == SourceTier.SOCIAL


# --- Tests for ImpactLevel enum ---


class TestImpactLevel:
    """Tests for the ImpactLevel enum."""

    def test_impact_values(self) -> None:
        assert ImpactLevel.HIGH == "HIGH"
        assert ImpactLevel.MEDIUM == "MEDIUM"
        assert ImpactLevel.LOW == "LOW"


# --- Tests for NewsArticle dataclass ---


class TestNewsArticle:
    """Tests for the NewsArticle dataclass."""

    def test_create_minimal(self) -> None:
        now = datetime.now(timezone.utc)
        article = NewsArticle(
            id="art-001",
            title="Fed raises rates",
            content="The Federal Reserve raised rates by 25bps.",
            source="Reuters",
            published_at=now,
        )
        assert article.id == "art-001"
        assert article.title == "Fed raises rates"
        assert article.content == "The Federal Reserve raised rates by 25bps."
        assert article.source == "Reuters"
        assert article.published_at == now
        assert article.instruments == []
        assert article.sentiment_score == 0.0
        assert article.impact_level == ImpactLevel.LOW
        assert article.region is None
        assert article.category is None

    def test_create_full(self) -> None:
        now = datetime.now(timezone.utc)
        article = NewsArticle(
            id="art-002",
            title="Oil prices surge",
            content="Crude oil surged 5% on supply concerns.",
            source="Bloomberg",
            published_at=now,
            instruments=["OIL_CRUDE", "BRENT"],
            sentiment_score=-0.8,
            impact_level=ImpactLevel.HIGH,
            region="Middle East",
            category="commodity_supply",
        )
        assert article.instruments == ["OIL_CRUDE", "BRENT"]
        assert article.sentiment_score == -0.8
        assert article.impact_level == ImpactLevel.HIGH
        assert article.region == "Middle East"
        assert article.category == "commodity_supply"


# --- Tests for SourceHealth dataclass ---


class TestSourceHealth:
    """Tests for the SourceHealth dataclass."""

    def test_create_minimal(self) -> None:
        health = SourceHealth(source_name="Reuters", is_connected=True)
        assert health.source_name == "Reuters"
        assert health.is_connected is True
        assert health.last_heartbeat is None
        assert health.articles_received_count == 0

    def test_create_full(self) -> None:
        now = datetime.now(timezone.utc)
        health = SourceHealth(
            source_name="Bloomberg",
            is_connected=True,
            last_heartbeat=now,
            articles_received_count=42,
        )
        assert health.source_name == "Bloomberg"
        assert health.is_connected is True
        assert health.last_heartbeat == now
        assert health.articles_received_count == 42


# --- Tests for RawArticle dataclass ---


class TestRawArticle:
    """Tests for the RawArticle dataclass."""

    def test_create_with_defaults(self) -> None:
        article = RawArticle(
            headline="Breaking news",
            body="Something happened.",
            source_name="Reuters",
            source_tier=1.0,
        )
        assert article.headline == "Breaking news"
        assert article.body == "Something happened."
        assert article.source_name == "Reuters"
        assert article.source_tier == 1.0
        assert article.published_at is None
        assert article.received_at is not None
        assert article.category is None
        assert article.metadata == {}


# --- Tests for NewsSource abstract base class ---


class TestNewsSource:
    """Tests for the NewsSource abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            NewsSource(name="Test", tier=1.0)  # type: ignore[abstract]

    def test_source_name_property(self) -> None:
        source = MockNewsSource(name="TestSource", tier=1.0)
        assert source.source_name == "TestSource"

    def test_source_tier_property_tier1(self) -> None:
        source = MockNewsSource(name="Reuters", tier=1.0)
        assert source.source_tier == SourceTier.TIER_1

    def test_source_tier_property_tier2(self) -> None:
        source = MockNewsSource(name="FinNews", tier=0.7)
        assert source.source_tier == SourceTier.TIER_2

    def test_source_tier_property_social(self) -> None:
        source = MockNewsSource(name="Twitter", tier=0.4)
        assert source.source_tier == SourceTier.SOCIAL

    def test_name_property_backward_compat(self) -> None:
        source = MockNewsSource(name="Reuters", tier=1.0)
        assert source.name == "Reuters"

    def test_tier_property_backward_compat(self) -> None:
        source = MockNewsSource(name="Reuters", tier=1.0)
        assert source.tier == 1.0

    def test_is_connected_initially_false(self) -> None:
        source = MockNewsSource()
        assert source.is_connected is False

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        source = MockNewsSource()
        await source.connect()
        assert source.is_connected is True
        assert source._connect_called is True

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        source = MockNewsSource()
        await source.connect()
        await source.disconnect()
        assert source.is_connected is False
        assert source._disconnect_called is True

    @pytest.mark.asyncio
    async def test_subscribe(self) -> None:
        source = MockNewsSource()
        await source.connect()
        await source.subscribe(["forex", "commodities"])
        assert source.subscribed_topics == ["forex", "commodities"]

    @pytest.mark.asyncio
    async def test_subscribe_raises_when_not_connected(self) -> None:
        source = MockNewsSource()
        with pytest.raises(ConnectionError):
            await source.subscribe(["forex"])

    @pytest.mark.asyncio
    async def test_subscribe_additive(self) -> None:
        source = MockNewsSource()
        await source.connect()
        await source.subscribe(["forex"])
        await source.subscribe(["commodities"])
        assert source.subscribed_topics == ["forex", "commodities"]

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        source = MockNewsSource()
        assert await source.health_check() is False
        await source.connect()
        assert await source.health_check() is True

    @pytest.mark.asyncio
    async def test_on_article_received_callback(self) -> None:
        source = MockNewsSource()
        received: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received.append(article)

        source.on_article_received(callback)

        article = RawArticle(
            headline="Test",
            body="Test body",
            source_name="MockSource",
            source_tier=1.0,
        )
        await source._notify_callbacks(article)

        assert len(received) == 1
        assert received[0].headline == "Test"

    @pytest.mark.asyncio
    async def test_notify_callbacks_updates_health(self) -> None:
        source = MockNewsSource()
        health_before = source.get_health()
        assert health_before.articles_received_count == 0
        assert health_before.last_heartbeat is None

        article = RawArticle(
            headline="Test",
            body="Body",
            source_name="MockSource",
            source_tier=1.0,
        )
        await source._notify_callbacks(article)

        health_after = source.get_health()
        assert health_after.articles_received_count == 1
        assert health_after.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_affect_others(self) -> None:
        source = MockNewsSource()
        results: list[str] = []

        async def failing_callback(article: RawArticle) -> None:
            raise RuntimeError("Callback failed")

        async def working_callback(article: RawArticle) -> None:
            results.append("ok")

        source.on_article_received(failing_callback)
        source.on_article_received(working_callback)

        article = RawArticle(
            headline="Test",
            body="Body",
            source_name="MockSource",
            source_tier=1.0,
        )
        await source._notify_callbacks(article)

        assert results == ["ok"]

    def test_get_health(self) -> None:
        source = MockNewsSource(name="TestSource", tier=0.7)
        health = source.get_health()
        assert isinstance(health, SourceHealth)
        assert health.source_name == "TestSource"
        assert health.is_connected is False
        assert health.last_heartbeat is None
        assert health.articles_received_count == 0

    @pytest.mark.asyncio
    async def test_get_health_after_activity(self) -> None:
        source = MockNewsSource(name="TestSource", tier=0.7)
        await source.connect()

        article = RawArticle(
            headline="News",
            body="Content",
            source_name="TestSource",
            source_tier=0.7,
        )
        await source._notify_callbacks(article)
        await source._notify_callbacks(article)

        health = source.get_health()
        assert health.is_connected is True
        assert health.articles_received_count == 2
        assert health.last_heartbeat is not None
