"""Main news engine orchestrating ingestion, analysis, and event publishing.

Coordinates all news module components: source ingestion, sentiment analysis,
impact classification, crisis detection, and event bus publishing.

Implements all-sources-down degradation logic per Requirement 23.18:
When ALL news sources are unavailable, the confidence threshold for all
trading signals is raised to 80 (from default 60) until at least one
source is restored.

Validates: Requirements 23.1, 23.17, 23.18
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config.constants import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    CONFIDENCE_THRESHOLD_NEWS_DOWN,
    NEWS_MAX_INGESTION_DELAY_SECONDS,
    NEWS_MIN_SOURCES,
    NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS,
    NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS,
    NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS,
)
from src.core.event_bus import (
    Event,
    EventBus,
    NEWS_ALL_SOURCES_DOWN,
    NEWS_ARTICLE_RECEIVED,
    NEWS_CRISIS_ALERT,
    NEWS_HIGH_IMPACT,
    NEWS_SOURCES_RESTORED,
)
from src.news.correlation_mapper import CorrelationMapper
from src.news.crisis_detector import CrisisDetector
from src.news.sentiment_analyzer import SentimentAnalyzer
from src.news.sources.base import NewsSource, RawArticle


logger = logging.getLogger(__name__)


class SourceHealthStatus:
    """Tracks health status for a single news source.

    Attributes:
        name: Source name identifier.
        last_successful_check: Timestamp of last successful health check.
        is_healthy: Whether the source is currently considered healthy.
        failed_since: Timestamp when the source first became unhealthy, or None.
        failover_logged: Whether the 5-minute failover has been logged.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_successful_check: datetime | None = None
        self.is_healthy: bool = False
        self.failed_since: datetime | None = None
        self.failover_logged: bool = False

    def mark_healthy(self, now: datetime | None = None) -> None:
        """Mark the source as healthy."""
        now = now or datetime.now(timezone.utc)
        self.last_successful_check = now
        self.is_healthy = True
        self.failed_since = None
        self.failover_logged = False

    def mark_unhealthy(self, now: datetime | None = None) -> None:
        """Mark the source as unhealthy, recording when failure started."""
        now = now or datetime.now(timezone.utc)
        self.is_healthy = False
        if self.failed_since is None:
            self.failed_since = now

    def is_failover_required(self, now: datetime | None = None) -> bool:
        """Check if the source has been unavailable for 5+ minutes.

        Returns:
            True if the source has been unavailable for longer than
            NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS (300s / 5 min).
        """
        if self.failed_since is None:
            return False
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.failed_since).total_seconds()
        return elapsed >= NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS


class NewsEngine:
    """Main news engine coordinating ingestion, analysis, and publishing.

    Pipeline for each article (must complete within 5 seconds):
    1. Ingest from source
    2. Check ingestion delay (max 30 seconds from publication)
    3. Deduplicate (same story from multiple sources)
    4. Sentiment analysis
    5. Impact classification
    6. Correlation mapping
    7. Crisis detection
    8. Publish to event bus

    Health monitoring:
    - Checks source health every 60 seconds
    - Marks source as failed after 5 minutes unavailability
    - Logs failover warning and notifies when source fails
    - Tracks all-sources-down condition

    Degraded mode (Requirement 23.18):
    - When ALL sources are unavailable, enters degraded mode
    - Publishes NEWS_ALL_SOURCES_DOWN event to Event Bus
    - Raises confidence threshold to 80 for all signals
    - When at least one source is restored, exits degraded mode
    - Publishes NEWS_SOURCES_RESTORED event
    - Logs all state transitions (normal → degraded → normal)

    Args:
        sources: List of NewsSource instances to ingest from.
        event_bus: Optional EventBus for publishing events.
        min_sources: Minimum healthy sources required (default: 3).
    """

    # Maximum number of article hashes to keep for deduplication
    _DEDUP_CACHE_MAX_SIZE: int = 10000

    def __init__(
        self,
        sources: list[NewsSource],
        event_bus: EventBus | None = None,
        min_sources: int = NEWS_MIN_SOURCES,
    ) -> None:
        self._sources = sources
        self._event_bus = event_bus
        self._min_sources = min_sources
        self._running = False
        self._health_check_task: asyncio.Task[None] | None = None

        # Source health tracking
        self._source_health: dict[str, SourceHealthStatus] = {
            source.name: SourceHealthStatus(source.name) for source in sources
        }

        # Degraded mode tracking (Requirement 23.18)
        self._degraded_mode: bool = False
        self._degraded_mode_entered_at: datetime | None = None

        # Article deduplication: ordered dict of body_hash -> first received timestamp
        self._seen_articles: OrderedDict[str, datetime] = OrderedDict()

        # Sub-components
        self._sentiment_analyzer = SentimentAnalyzer()
        self._crisis_detector = CrisisDetector(event_bus=event_bus)
        self._correlation_mapper = CorrelationMapper()

        # Metrics
        self._articles_processed: int = 0
        self._articles_deduplicated: int = 0
        self._articles_delayed: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def healthy_sources(self) -> set[str]:
        """Names of currently healthy sources."""
        return {
            name for name, status in self._source_health.items()
            if status.is_healthy
        }

    @property
    def is_running(self) -> bool:
        """Whether the engine is currently active."""
        return self._running

    @property
    def articles_processed(self) -> int:
        """Total number of articles successfully processed."""
        return self._articles_processed

    @property
    def articles_deduplicated(self) -> int:
        """Total number of duplicate articles filtered out."""
        return self._articles_deduplicated

    @property
    def articles_delayed(self) -> int:
        """Total number of articles that exceeded the max ingestion delay."""
        return self._articles_delayed

    def is_all_sources_down(self) -> bool:
        """Check if all news sources are unavailable.

        Returns:
            True if no sources are healthy, False otherwise.
        """
        return len(self.healthy_sources) == 0

    @property
    def degraded_mode(self) -> bool:
        """Whether the engine is in degraded mode (all sources down).

        When in degraded mode, the confidence threshold for all trading
        signals should be raised to 80 (from default 60) per Requirement 23.18.
        """
        return self._degraded_mode

    def get_confidence_threshold(self) -> int:
        """Get the current confidence threshold based on source availability.

        Returns:
            CONFIDENCE_THRESHOLD_NEWS_DOWN (80) if all sources are down
            (degraded mode), otherwise CONFIDENCE_THRESHOLD_DEFAULT (60).
        """
        if self._degraded_mode:
            return CONFIDENCE_THRESHOLD_NEWS_DOWN
        return CONFIDENCE_THRESHOLD_DEFAULT

    def get_source_health(self, source_name: str) -> SourceHealthStatus | None:
        """Get health status for a specific source.

        Args:
            source_name: Name of the source to query.

        Returns:
            SourceHealthStatus or None if source not found.
        """
        return self._source_health.get(source_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start ingesting from all sources and begin health monitoring.

        Connects to all configured sources and registers article callbacks.
        Starts the periodic health check loop (every 60 seconds).
        """
        if self._running:
            return

        self._running = True

        # Connect to all sources
        for source in self._sources:
            try:
                await source.connect()
                source.on_article_received(self._on_raw_article_received)
                self._source_health[source.name].mark_healthy()
                logger.info(
                    "News source connected",
                    extra={"source": source.name, "tier": source.tier},
                )
            except Exception as exc:
                self._source_health[source.name].mark_unhealthy()
                logger.warning(
                    "News source failed to connect",
                    extra={"source": source.name, "error": str(exc)},
                )

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Evaluate initial degraded mode state (Requirement 23.18)
        # If no sources connected successfully, enter degraded mode immediately
        now = datetime.now(timezone.utc)
        await self._evaluate_degraded_mode(now)

        logger.info(
            "NewsEngine started",
            extra={
                "total_sources": len(self._sources),
                "healthy_sources": len(self.healthy_sources),
                "min_sources_required": self._min_sources,
                "degraded_mode": self._degraded_mode,
            },
        )

    async def stop(self) -> None:
        """Stop all source ingestion and health monitoring."""
        self._running = False

        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Disconnect all sources
        for source in self._sources:
            try:
                await source.disconnect()
            except Exception:
                pass

        # Reset health statuses
        for status in self._source_health.values():
            status.is_healthy = False

        logger.info("NewsEngine stopped")

    # ------------------------------------------------------------------
    # Article Ingestion
    # ------------------------------------------------------------------

    async def _on_raw_article_received(self, raw_article: RawArticle) -> None:
        """Internal callback for raw articles from sources.

        Converts RawArticle to dict format and passes to the processing pipeline.
        """
        article_data = {
            "id": str(uuid.uuid4()),
            "source": raw_article.source_name,
            "source_tier": raw_article.source_tier,
            "headline": raw_article.headline,
            "body": raw_article.body,
            "published_at": raw_article.published_at,
            "received_at": raw_article.received_at,
            "category": raw_article.category,
        }
        await self.on_article_received(article_data)

    async def on_article_received(self, article: dict[str, Any]) -> None:
        """Process an article through the full pipeline.

        Pipeline stages (must complete within 5 seconds total):
        1. Check ingestion delay (max 30 seconds from publication)
        2. Deduplicate (same story from multiple sources)
        3. Sentiment analysis
        4. Impact classification
        5. Correlation mapping
        6. Crisis detection
        7. Publish to event bus

        Args:
            article: Dict with article data including headline, body,
                     source, source_tier, published_at, received_at, category.
        """
        try:
            now = datetime.now(timezone.utc)

            # Stage 1: Check ingestion delay
            published_at = article.get("published_at")
            received_at = article.get("received_at", now)

            if published_at is not None:
                delay = (received_at - published_at).total_seconds()
                article["ingestion_delay_seconds"] = delay

                if delay > NEWS_MAX_INGESTION_DELAY_SECONDS:
                    self._articles_delayed += 1
                    logger.warning(
                        "Article exceeded max ingestion delay",
                        extra={
                            "source": article.get("source"),
                            "headline": article.get("headline", "")[:80],
                            "delay_seconds": delay,
                            "max_allowed": NEWS_MAX_INGESTION_DELAY_SECONDS,
                        },
                    )
                    # Still process the article but flag it as delayed
                    article["delayed"] = True

            # Stage 2: Deduplication
            body = article.get("body", article.get("headline", ""))
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            article["body_hash"] = body_hash

            if self._is_duplicate(body_hash):
                self._articles_deduplicated += 1
                logger.debug(
                    "Duplicate article filtered",
                    extra={
                        "source": article.get("source"),
                        "headline": article.get("headline", "")[:80],
                    },
                )
                return

            self._record_article_hash(body_hash, now)

            # Stage 3: Sentiment analysis
            text = f"{article.get('headline', '')} {body}"
            sentiment_score = self._sentiment_analyzer.analyze(text)
            article["sentiment_score"] = sentiment_score

            # Stage 4: Impact classification
            source_tier = article.get("source_tier", 0.4)
            corroboration = self._count_corroboration(article)
            impact_level = self._sentiment_analyzer.classify_impact(
                sentiment_score, source_tier, corroboration
            )
            article["impact_level"] = impact_level

            # Stage 5: Correlation mapping
            category = article.get("category")
            if category:
                affected = self._correlation_mapper.get_affected_instruments(category)
                article["correlated_instruments"] = affected

            # Stage 5.5: High-impact news notification (Requirement 23.11)
            # Notify Strategy_Engine within 5 seconds of article receipt
            if impact_level == "HIGH":
                await self._publish_high_impact_notification(article)

            # Stage 6: Crisis detection
            crisis = self._crisis_detector.evaluate(article)
            if crisis is not None:
                await self._publish_crisis_alert(crisis)

            # Stage 6.5: Dispatch any pending Kill Switch events from persistence check
            await self._crisis_detector.dispatch_pending_events()

            # Stage 7: Publish article event
            await self._publish_article_event(article)

            self._articles_processed += 1

        except Exception as exc:
            logger.error(
                "Article processing pipeline failed",
                extra={
                    "source": article.get("source"),
                    "headline": article.get("headline", "")[:80],
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(self, body_hash: str) -> bool:
        """Check if an article with this hash has already been processed.

        Args:
            body_hash: SHA-256 hash of the article body.

        Returns:
            True if this article has been seen before.
        """
        return body_hash in self._seen_articles

    def _record_article_hash(self, body_hash: str, received_at: datetime) -> None:
        """Record an article hash for deduplication tracking.

        Maintains a bounded cache to prevent unbounded memory growth.

        Args:
            body_hash: SHA-256 hash of the article body.
            received_at: Timestamp when the article was received.
        """
        self._seen_articles[body_hash] = received_at

        # Evict oldest entries if cache exceeds max size
        while len(self._seen_articles) > self._DEDUP_CACHE_MAX_SIZE:
            self._seen_articles.popitem(last=False)

    # ------------------------------------------------------------------
    # Corroboration
    # ------------------------------------------------------------------

    def _count_corroboration(self, article: dict[str, Any]) -> int:
        """Count corroborating articles within 5-minute window.

        Checks how many other articles with similar headlines/topics
        have been received from different sources recently.

        Returns:
            Number of corroborating articles from different sources.
        """
        recent = self._crisis_detector._recent_articles
        source = article.get("source", "")
        count = sum(1 for a in recent if a.source != source)
        return count

    # ------------------------------------------------------------------
    # Health Monitoring
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Periodic health check for all news sources.

        Runs every 60 seconds (NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS).
        If a source is unavailable for 5+ minutes during market hours,
        logs a warning and marks it as failed (failover).
        """
        while self._running:
            try:
                await asyncio.sleep(NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Health check loop error",
                    extra={"error": str(exc)},
                )
                await asyncio.sleep(1)

    async def _perform_health_checks(self) -> None:
        """Execute health checks on all sources and handle failover.

        For each source:
        - If healthy: update last successful check timestamp
        - If unhealthy and unavailable > 5 min: log failover, mark failed

        After checking all sources, evaluates degraded mode transitions:
        - If all sources down and not already degraded: enter degraded mode
        - If at least one source restored and currently degraded: exit degraded mode
        """
        now = datetime.now(timezone.utc)

        for source in self._sources:
            status = self._source_health[source.name]
            try:
                healthy = await source.health_check()
                if healthy:
                    status.mark_healthy(now)
                else:
                    self._handle_unhealthy_source(source.name, status, now)
            except Exception:
                self._handle_unhealthy_source(source.name, status, now)

        # Evaluate degraded mode state transitions (Requirement 23.18)
        await self._evaluate_degraded_mode(now)

    def _handle_unhealthy_source(
        self, source_name: str, status: SourceHealthStatus, now: datetime
    ) -> None:
        """Handle an unhealthy source, checking for failover condition.

        If the source has been unavailable for 5+ minutes and failover
        hasn't been logged yet, log the failover warning.

        Args:
            source_name: Name of the unhealthy source.
            status: Current health status for the source.
            now: Current timestamp.
        """
        status.mark_unhealthy(now)

        if status.is_failover_required(now) and not status.failover_logged:
            status.failover_logged = True
            logger.warning(
                "News source failover: source unavailable for 5+ minutes",
                extra={
                    "source": source_name,
                    "failed_since": (
                        status.failed_since.isoformat() if status.failed_since else None
                    ),
                    "action": "switching to remaining active sources",
                },
            )

    async def _evaluate_degraded_mode(self, now: datetime) -> None:
        """Evaluate and handle degraded mode state transitions.

        Transitions:
        - normal → degraded: When ALL sources become unavailable
        - degraded → normal: When at least one source is restored

        On each transition:
        - Publishes appropriate event to Event Bus
        - Logs the state transition

        Args:
            now: Current timestamp for recording transition time.
        """
        all_down = self.is_all_sources_down()

        if all_down and not self._degraded_mode:
            # Transition: normal → degraded
            await self._enter_degraded_mode(now)
        elif not all_down and self._degraded_mode:
            # Transition: degraded → normal
            await self._exit_degraded_mode(now)

    async def _enter_degraded_mode(self, now: datetime) -> None:
        """Enter degraded mode when all sources are down.

        Sets the degraded mode flag, records the transition time,
        publishes NEWS_ALL_SOURCES_DOWN event, and logs the transition.

        Args:
            now: Timestamp when degraded mode was entered.
        """
        self._degraded_mode = True
        self._degraded_mode_entered_at = now

        logger.critical(
            "NEWS ENGINE DEGRADED: All sources down - confidence threshold "
            "raised to %d for all signals",
            CONFIDENCE_THRESHOLD_NEWS_DOWN,
            extra={
                "transition": "normal → degraded",
                "source_count": len(self._sources),
                "confidence_threshold": CONFIDENCE_THRESHOLD_NEWS_DOWN,
                "entered_at": now.isoformat(),
            },
        )

        # Publish all-sources-down event to Event Bus
        await self._publish_all_sources_down_event(now)

    async def _exit_degraded_mode(self, now: datetime) -> None:
        """Exit degraded mode when at least one source is restored.

        Clears the degraded mode flag, publishes NEWS_SOURCES_RESTORED
        event, and logs the transition including duration.

        Args:
            now: Timestamp when degraded mode was exited.
        """
        duration_seconds = None
        if self._degraded_mode_entered_at is not None:
            duration_seconds = (now - self._degraded_mode_entered_at).total_seconds()

        self._degraded_mode = False

        restored_sources = list(self.healthy_sources)

        logger.info(
            "NEWS ENGINE RECOVERED: At least one source restored - confidence "
            "threshold restored to %d",
            CONFIDENCE_THRESHOLD_DEFAULT,
            extra={
                "transition": "degraded → normal",
                "confidence_threshold": CONFIDENCE_THRESHOLD_DEFAULT,
                "restored_sources": restored_sources,
                "degraded_duration_seconds": duration_seconds,
                "recovered_at": now.isoformat(),
            },
        )

        self._degraded_mode_entered_at = None

        # Publish sources-restored event to Event Bus
        await self._publish_sources_restored_event(now, restored_sources, duration_seconds)

    async def _publish_all_sources_down_event(self, now: datetime) -> None:
        """Publish NEWS_ALL_SOURCES_DOWN event to the Event Bus.

        Args:
            now: Timestamp when all sources went down.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=NEWS_ALL_SOURCES_DOWN,
            payload={
                "timestamp": now.isoformat(),
                "source_count": len(self._sources),
                "confidence_threshold": CONFIDENCE_THRESHOLD_NEWS_DOWN,
                "message": "All news sources unavailable - confidence threshold elevated",
            },
        )
        try:
            await self._event_bus.publish(NEWS_ALL_SOURCES_DOWN, event)
        except Exception as exc:
            logger.error(
                "Failed to publish all-sources-down event",
                extra={"error": str(exc)},
            )

    async def _publish_sources_restored_event(
        self,
        now: datetime,
        restored_sources: list[str],
        degraded_duration_seconds: float | None,
    ) -> None:
        """Publish NEWS_SOURCES_RESTORED event to the Event Bus.

        Args:
            now: Timestamp when sources were restored.
            restored_sources: List of source names that are now healthy.
            degraded_duration_seconds: How long the system was in degraded mode.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=NEWS_SOURCES_RESTORED,
            payload={
                "timestamp": now.isoformat(),
                "restored_sources": restored_sources,
                "confidence_threshold": CONFIDENCE_THRESHOLD_DEFAULT,
                "degraded_duration_seconds": degraded_duration_seconds,
                "message": "News source restored - confidence threshold returned to normal",
            },
        )
        try:
            await self._event_bus.publish(NEWS_SOURCES_RESTORED, event)
        except Exception as exc:
            logger.error(
                "Failed to publish sources-restored event",
                extra={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Event Publishing
    # ------------------------------------------------------------------

    async def _publish_article_event(self, article: dict[str, Any]) -> None:
        """Publish processed article to the event bus.

        Args:
            article: Enriched article dict with sentiment, impact, etc.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=NEWS_ARTICLE_RECEIVED,
            payload={
                "id": article.get("id"),
                "source": article.get("source"),
                "headline": article.get("headline"),
                "sentiment_score": article.get("sentiment_score"),
                "impact_level": article.get("impact_level"),
                "category": article.get("category"),
                "correlated_instruments": article.get("correlated_instruments", []),
                "ingestion_delay_seconds": article.get("ingestion_delay_seconds"),
                "delayed": article.get("delayed", False),
            },
        )
        try:
            await self._event_bus.publish(NEWS_ARTICLE_RECEIVED, event)
        except Exception as exc:
            logger.error(
                "Failed to publish article event",
                extra={"article_id": article.get("id"), "error": str(exc)},
            )

    async def _publish_high_impact_notification(self, article: dict[str, Any]) -> None:
        """Publish high-impact news notification for the Strategy Engine.

        When an article is classified as HIGH impact, notifies the Strategy
        Engine with affected instruments, sentiment score, and impact
        classification so it can apply confidence penalties (-25 points).

        This notification must be published within 5 seconds of article receipt
        per Requirement 23.11.

        Args:
            article: Enriched article dict with sentiment, impact, and
                     correlated instruments already populated.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=NEWS_HIGH_IMPACT,
            payload={
                "article_id": article.get("id"),
                "affected_instruments": article.get("correlated_instruments", []),
                "sentiment_score": article.get("sentiment_score"),
                "impact_level": article.get("impact_level", "HIGH"),
                "source": article.get("source"),
                "headline": article.get("headline"),
                "category": article.get("category"),
            },
        )
        try:
            await self._event_bus.publish(NEWS_HIGH_IMPACT, event)
            logger.info(
                "High-impact news notification published to Strategy_Engine",
                extra={
                    "article_id": article.get("id"),
                    "affected_instruments": article.get("correlated_instruments", []),
                    "sentiment_score": article.get("sentiment_score"),
                    "impact_level": "HIGH",
                },
            )
        except Exception as exc:
            logger.error(
                "Failed to publish high-impact news notification",
                extra={"article_id": article.get("id"), "error": str(exc)},
            )

    async def _publish_crisis_alert(self, crisis: Any) -> None:
        """Publish crisis alert to the event bus.

        Args:
            crisis: CrisisAlertData instance from the crisis detector.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=NEWS_CRISIS_ALERT,
            payload={
                "crisis_id": crisis.id,
                "region": crisis.region,
                "sentiment_avg": crisis.sentiment_avg,
                "article_count": len(crisis.trigger_articles),
                "started_at": crisis.started_at.isoformat(),
            },
        )
        try:
            await self._event_bus.publish(NEWS_CRISIS_ALERT, event)
        except Exception as exc:
            logger.error(
                "Failed to publish crisis alert",
                extra={"crisis_id": crisis.id, "error": str(exc)},
            )
