"""Base class for news source adapters.

Defines the abstract interface that all news source implementations must follow.
Each source has a credibility tier weight used in impact classification.

Validates: Requirements 23.1
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine


from src.config.constants import (
    SOURCE_CREDIBILITY_SOCIAL,
    SOURCE_CREDIBILITY_TIER1,
    SOURCE_CREDIBILITY_TIER2,
)


class SourceTier(str, Enum):
    """News source credibility tiers."""

    TIER_1 = "tier-1"
    TIER_2 = "tier-2"
    SOCIAL = "social"


class ImpactLevel(str, Enum):
    """Impact level classification for news articles."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class NewsArticle:
    """Processed news article with sentiment and impact analysis.

    Represents a fully processed article after passing through the
    news engine pipeline (ingestion → sentiment → impact → correlation).
    """

    id: str
    title: str
    content: str
    source: str
    published_at: datetime
    instruments: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    impact_level: ImpactLevel = ImpactLevel.LOW
    region: str | None = None
    category: str | None = None


@dataclass
class SourceHealth:
    """Health status of a news source connection.

    Used by the news engine to monitor source availability and
    trigger failover when sources become unavailable.
    """

    source_name: str
    is_connected: bool
    last_heartbeat: datetime | None = None
    articles_received_count: int = 0


@dataclass
class RawArticle:
    """Raw article data received from a news source before processing."""

    headline: str
    body: str
    source_name: str
    source_tier: float
    published_at: datetime | None = None
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for article callback
ArticleCallback = Callable[[RawArticle], Coroutine[Any, Any, None]]


def _tier_from_weight(weight: float) -> SourceTier:
    """Map a credibility weight to a SourceTier enum value."""
    if weight >= SOURCE_CREDIBILITY_TIER1:
        return SourceTier.TIER_1
    elif weight >= SOURCE_CREDIBILITY_TIER2:
        return SourceTier.TIER_2
    return SourceTier.SOCIAL


class NewsSource(ABC):
    """Abstract base class for news source adapters.

    Each news source has a name and credibility tier weight:
    - Tier 1 (1.0): Reuters, Bloomberg — highest credibility
    - Tier 2 (0.7): Major financial news outlets
    - Social (0.4): Social media sources — lowest credibility

    Subclasses must implement:
    - connect(): Establish connection to the news source
    - disconnect(): Gracefully disconnect
    - subscribe(topics): Subscribe to specific news topics/categories
    - health_check(): Check if the source connection is healthy

    Attributes:
        name: Human-readable name of the news source.
        tier: Credibility weight for impact classification.
    """

    def __init__(self, name: str, tier: float) -> None:
        self._name = name
        self._tier = tier
        self._source_tier_enum = _tier_from_weight(tier)
        self._callbacks: list[ArticleCallback] = []
        self._connected = False
        self._subscribed_topics: list[str] = []
        self._last_heartbeat: datetime | None = None
        self._articles_received_count: int = 0

    @property
    def source_name(self) -> str:
        """Human-readable name of the news source."""
        return self._name

    @property
    def source_tier(self) -> SourceTier:
        """Credibility tier classification (tier-1, tier-2, social)."""
        return self._source_tier_enum

    @property
    def name(self) -> str:
        """Human-readable name of the news source (alias for source_name)."""
        return self._name

    @property
    def tier(self) -> float:
        """Credibility weight for impact classification."""
        return self._tier

    @property
    def is_connected(self) -> bool:
        """Whether the source is currently connected."""
        return self._connected

    @property
    def subscribed_topics(self) -> list[str]:
        """Currently subscribed topics."""
        return list(self._subscribed_topics)

    def get_health(self) -> SourceHealth:
        """Get current health status of this source.

        Returns:
            SourceHealth dataclass with connection and activity info.
        """
        return SourceHealth(
            source_name=self._name,
            is_connected=self._connected,
            last_heartbeat=self._last_heartbeat,
            articles_received_count=self._articles_received_count,
        )

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the news source.

        Raises:
            ConnectionError: If the source cannot be reached.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the news source."""
        ...

    @abstractmethod
    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to specific news topics or categories.

        Subscribes the source to receive articles matching the given
        topics. Topics are additive — calling subscribe multiple times
        adds to the existing subscriptions.

        Args:
            topics: List of topic strings to subscribe to
                    (e.g., ["forex", "commodities", "central-banks"]).

        Raises:
            ConnectionError: If not connected to the source.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the source connection is healthy and responsive.

        Returns:
            True if the source is healthy, False otherwise.
        """
        ...

    def on_article_received(self, callback: ArticleCallback) -> None:
        """Register a callback to be invoked when an article is received.

        Args:
            callback: Async callable that receives a RawArticle.
        """
        self._callbacks.append(callback)

    async def _notify_callbacks(self, article: RawArticle) -> None:
        """Invoke all registered callbacks with the received article.

        Also updates internal health tracking (heartbeat and article count).
        """
        self._last_heartbeat = datetime.now(timezone.utc)
        self._articles_received_count += 1

        for callback in self._callbacks:
            try:
                await callback(article)
            except Exception:
                # Individual callback failures should not affect other callbacks
                pass
