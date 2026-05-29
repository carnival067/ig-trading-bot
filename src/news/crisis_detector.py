"""Crisis detection engine for the news module.

Monitors incoming high-impact negative articles and triggers crisis alerts
when thresholds are exceeded within a sliding time window, grouped by
region or asset class.

Crisis detection rule (Requirement 23.7):
  3+ HIGH-impact articles with sentiment < -0.7 within a 10-minute window
  referencing the same geopolitical region or asset class triggers a crisis alert.

Persistent crises (no recovery above -0.3 within 30 minutes) escalate to
Kill Switch activation (Requirement 23.9, Cross-Cutting Rule 3).
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from src.config.constants import (
    NEWS_CRISIS_ARTICLE_THRESHOLD,
    NEWS_CRISIS_PERSISTENCE_MINUTES,
    NEWS_CRISIS_RECOVERY_THRESHOLD,
    NEWS_CRISIS_SENTIMENT_THRESHOLD,
    NEWS_CRISIS_TIME_WINDOW_MINUTES,
)
from src.core.event_bus import KILL_SWITCH_ACTIVATED, Event

logger = logging.getLogger(__name__)


class EventPublisher(Protocol):
    """Protocol for event publishing (allows decoupled testing)."""

    async def publish(self, channel: str, event: Event) -> int: ...


@dataclass
class CrisisArticleRecord:
    """Record of a high-impact negative article for crisis tracking."""

    article_id: str
    source: str
    headline: str
    sentiment_score: float
    received_at: datetime
    category: str | None = None
    region: str | None = None
    asset_class: str | None = None


@dataclass
class CrisisAlertData:
    """Active crisis alert with tracking information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    region: str = "global"
    sentiment_avg: float = 0.0
    trigger_articles: list[CrisisArticleRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    escalated_to_kill_switch: bool = False
    active: bool = True


@dataclass
class CrisisSentimentRecord:
    """Tracks a sentiment reading received after a crisis started."""

    sentiment_score: float
    region: str
    received_at: datetime


class CrisisDetector:
    """Detects crisis conditions from accumulated negative news.

    Crisis detection rules (Requirement 23.7):
    - 3+ HIGH-impact articles with sentiment < -0.7 within 10 minutes
      referencing the same region or asset class triggers a crisis alert.
    - If no recovery above -0.3 within 30 minutes, escalate to Kill Switch
      (Requirement 23.9).

    The detector maintains a sliding window of recent HIGH-impact articles
    and groups them by region and asset class to detect localized crises.

    Attributes:
        CRISIS_THRESHOLD: Number of articles required (3).
        SENTIMENT_THRESHOLD: Sentiment score threshold (-0.7).
        TIME_WINDOW_MINUTES: Accumulation window (10 min).
        PERSISTENCE_MINUTES: Persistence before escalation (30 min).
    """

    CRISIS_THRESHOLD: int = NEWS_CRISIS_ARTICLE_THRESHOLD
    SENTIMENT_THRESHOLD: float = NEWS_CRISIS_SENTIMENT_THRESHOLD
    TIME_WINDOW_MINUTES: int = NEWS_CRISIS_TIME_WINDOW_MINUTES
    PERSISTENCE_MINUTES: int = NEWS_CRISIS_PERSISTENCE_MINUTES

    def __init__(self, event_bus: EventPublisher | None = None) -> None:
        self._recent_articles: deque[CrisisArticleRecord] = deque()
        self._active_crises: dict[str, CrisisAlertData] = {}
        self._event_bus: EventPublisher | None = event_bus
        # Track which grouping keys have active crises to avoid duplicates
        self._active_crisis_keys: dict[str, str] = {}  # grouping_key -> crisis_id
        # Per-crisis sentiment readings since crisis started
        self._crisis_sentiments: dict[str, list[CrisisSentimentRecord]] = {}

    @property
    def active_crises(self) -> list[CrisisAlertData]:
        """List of currently active crisis alerts."""
        return [c for c in self._active_crises.values() if c.active]

    def evaluate(self, article: dict[str, Any]) -> CrisisAlertData | None:
        """Evaluate whether an article contributes to a crisis condition.

        Checks if the article has HIGH impact and negative sentiment below
        the threshold. Groups articles by region and asset class, and triggers
        a crisis alert when 3+ qualifying articles appear within the time
        window for the same grouping key.

        Also performs persistence checking for active crises on each article.

        Args:
            article: Dict with keys: id, source, headline, sentiment_score,
                     received_at, impact_level, region, asset_class, category.

        Returns:
            CrisisAlertData if a new crisis is triggered, None otherwise.
        """
        sentiment = article.get("sentiment_score", 0.0)
        impact_level = article.get("impact_level", "LOW")
        region = article.get("region")
        asset_class = article.get("asset_class")
        received_at = article.get("received_at", datetime.now(timezone.utc))

        # Check persistence and recovery for active crises on every article
        self._check_active_crises_on_article(sentiment, region, received_at)

        # Only track HIGH-impact articles with sentiment below threshold
        if impact_level != "HIGH" or sentiment > self.SENTIMENT_THRESHOLD:
            return None

        record = CrisisArticleRecord(
            article_id=article.get("id", str(uuid.uuid4())),
            source=article.get("source", "unknown"),
            headline=article.get("headline", ""),
            sentiment_score=sentiment,
            received_at=received_at,
            category=article.get("category"),
            region=region,
            asset_class=asset_class,
        )

        self._recent_articles.append(record)
        self._prune_old_articles()

        # Check crisis conditions grouped by region and asset class
        return self._check_crisis_conditions(record)

    def check_persistence(
        self,
        crisis_id: str,
        current_sentiments: list[float] | None = None,
        current_time: datetime | None = None,
    ) -> bool:
        """Check if a crisis has persisted without recovery.

        If no sentiment reading has recovered above -0.3 within 30 minutes
        of crisis start, this returns True indicating Kill Switch should
        be triggered (Requirement 23.9).

        Args:
            crisis_id: ID of the active crisis to check.
            current_sentiments: Recent sentiment scores since crisis started.
                If None, uses internally tracked sentiments.
            current_time: Time to use for elapsed calculation. Defaults to now.

        Returns:
            True if crisis persists (no recovery) and Kill Switch should trigger.
            False if recovery detected or crisis not found.
        """
        crisis = self._active_crises.get(crisis_id)
        if crisis is None or not crisis.active:
            return False

        now = current_time or datetime.now(timezone.utc)
        elapsed = now - crisis.started_at

        # Check if persistence window has elapsed
        if elapsed < timedelta(minutes=self.PERSISTENCE_MINUTES):
            return False

        # Use provided sentiments or internally tracked ones
        if current_sentiments is None:
            tracked = self._crisis_sentiments.get(crisis_id, [])
            current_sentiments = [s.sentiment_score for s in tracked]

        # Check if any sentiment has recovered above threshold
        for sentiment in current_sentiments:
            if sentiment > NEWS_CRISIS_RECOVERY_THRESHOLD:
                # Recovery detected — resolve crisis
                self.resolve_crisis(crisis_id)
                return False

        # No recovery within persistence window — escalate
        crisis.escalated_to_kill_switch = True
        self._emit_kill_switch_activation(crisis)
        return True

    def resolve_crisis(self, crisis_id: str) -> None:
        """Mark a crisis as resolved.

        Args:
            crisis_id: ID of the crisis to resolve.
        """
        crisis = self._active_crises.get(crisis_id)
        if crisis is not None:
            crisis.active = False
            crisis.resolved_at = datetime.now(timezone.utc)
            # Remove from active keys tracking
            keys_to_remove = [
                key
                for key, cid in self._active_crisis_keys.items()
                if cid == crisis_id
            ]
            for key in keys_to_remove:
                del self._active_crisis_keys[key]
            # Clean up tracked sentiments
            self._crisis_sentiments.pop(crisis_id, None)

    async def dispatch_pending_events(self) -> None:
        """Dispatch any pending Kill Switch activation events.

        Should be called by the NewsEngine after each article evaluation
        to ensure async event publishing occurs.
        """
        event = getattr(self, "_pending_kill_switch_event", None)
        if event is not None and self._event_bus is not None:
            try:
                await self._event_bus.publish(KILL_SWITCH_ACTIVATED, event)
                logger.info(
                    "Kill Switch activation event published",
                    extra={"event_type": event.event_type},
                )
            except Exception as exc:
                logger.error(
                    "Failed to publish Kill Switch activation event",
                    extra={"error": str(exc)},
                )
            finally:
                self._pending_kill_switch_event = None

    def _check_crisis_conditions(
        self, new_record: CrisisArticleRecord
    ) -> CrisisAlertData | None:
        """Check if crisis threshold is met for any region/asset class group.

        Groups articles in the current window by region and asset class,
        then checks if any group containing the new article has reached
        the crisis threshold. Only triggers for groups that don't already
        have an active crisis.

        Args:
            new_record: The newly added article record.

        Returns:
            CrisisAlertData if a new crisis is detected, None otherwise.
        """
        window_articles = self._get_articles_in_window()

        # Build grouping keys for the new article to check relevant groups
        candidate_keys = self._get_grouping_keys(new_record)

        for grouping_key in candidate_keys:
            # Skip if there's already an active crisis for this key
            if grouping_key in self._active_crisis_keys:
                continue

            # Get articles matching this grouping key
            matching_articles = self._get_articles_for_key(
                window_articles, grouping_key
            )

            if len(matching_articles) >= self.CRISIS_THRESHOLD:
                # Trigger new crisis
                avg_sentiment = sum(
                    a.sentiment_score for a in matching_articles
                ) / len(matching_articles)

                # Determine the region label for the crisis
                region_label = self._resolve_region_label(grouping_key)

                crisis = CrisisAlertData(
                    region=region_label,
                    trigger_articles=list(matching_articles),
                    sentiment_avg=avg_sentiment,
                )
                self._active_crises[crisis.id] = crisis
                self._active_crisis_keys[grouping_key] = crisis.id
                self._crisis_sentiments[crisis.id] = []
                return crisis

        return None

    def _check_active_crises_on_article(
        self,
        sentiment: float,
        region: str | None,
        received_at: datetime,
    ) -> None:
        """Check all active crises for recovery or persistence on each article.

        For each active crisis matching the article's region:
        - If sentiment > -0.3: recovery detected, resolve the crisis.
        - If 30 minutes have elapsed without recovery: escalate to Kill Switch.

        Args:
            sentiment: Sentiment score of the incoming article.
            region: Region of the incoming article.
            received_at: Timestamp of the incoming article.
        """
        crises_to_check = [
            (cid, crisis)
            for cid, crisis in self._active_crises.items()
            if crisis.active and not crisis.escalated_to_kill_switch
        ]

        for crisis_id, crisis in crises_to_check:
            # Only consider articles relevant to the crisis region
            article_region = region or "global"
            crisis_region = crisis.region or "global"
            if crisis_region != article_region and crisis_region != "global" and article_region != "global":
                continue

            # Record this sentiment reading for the crisis
            record = CrisisSentimentRecord(
                sentiment_score=sentiment,
                region=article_region,
                received_at=received_at,
            )
            if crisis_id in self._crisis_sentiments:
                self._crisis_sentiments[crisis_id].append(record)

            # Check for recovery: sentiment above -0.3 means crisis resolved
            if sentiment > NEWS_CRISIS_RECOVERY_THRESHOLD:
                self.resolve_crisis(crisis_id)
                logger.info(
                    "Crisis resolved via sentiment recovery",
                    extra={
                        "crisis_id": crisis_id,
                        "region": region,
                        "recovery_sentiment": sentiment,
                    },
                )
                continue

            # Check for persistence: 30 minutes elapsed without recovery
            elapsed = received_at - crisis.started_at
            if elapsed >= timedelta(minutes=self.PERSISTENCE_MINUTES):
                sentiments = self._crisis_sentiments.get(crisis_id, [])
                has_recovery = any(
                    s.sentiment_score > NEWS_CRISIS_RECOVERY_THRESHOLD
                    for s in sentiments
                )
                if not has_recovery:
                    crisis.escalated_to_kill_switch = True
                    self._emit_kill_switch_activation(crisis)
                    logger.warning(
                        "Crisis persistent — Kill Switch activation requested",
                        extra={
                            "crisis_id": crisis_id,
                            "region": crisis.region,
                            "elapsed_minutes": elapsed.total_seconds() / 60,
                        },
                    )

    def _get_grouping_keys(self, record: CrisisArticleRecord) -> list[str]:
        """Get all grouping keys an article belongs to.

        An article can belong to both a region group and an asset class group.

        Args:
            record: The article record.

        Returns:
            List of grouping key strings (e.g., "region:europe", "asset_class:forex").
        """
        keys: list[str] = []
        if record.region:
            keys.append(f"region:{record.region.lower()}")
        if record.asset_class:
            keys.append(f"asset_class:{record.asset_class.lower()}")
        # If neither region nor asset_class is specified, use a global fallback
        if not keys:
            keys.append("global")
        return keys

    def _get_articles_for_key(
        self,
        articles: list[CrisisArticleRecord],
        grouping_key: str,
    ) -> list[CrisisArticleRecord]:
        """Filter articles that match a given grouping key.

        Args:
            articles: List of articles in the current window.
            grouping_key: The key to filter by (e.g., "region:europe").

        Returns:
            List of matching articles.
        """
        matching: list[CrisisArticleRecord] = []
        for article in articles:
            article_keys = self._get_grouping_keys(article)
            if grouping_key in article_keys:
                matching.append(article)
        return matching

    def _resolve_region_label(self, grouping_key: str) -> str:
        """Convert a grouping key to a human-readable region label.

        Args:
            grouping_key: The internal grouping key.

        Returns:
            A region/asset class label string.
        """
        if grouping_key == "global":
            return "global"
        # Format: "region:europe" -> "europe", "asset_class:forex" -> "forex"
        parts = grouping_key.split(":", 1)
        if len(parts) == 2:
            return parts[1]
        return grouping_key

    def _emit_kill_switch_activation(self, crisis: CrisisAlertData) -> None:
        """Emit a Kill Switch activation request via the Event Bus.

        Args:
            crisis: The persistent crisis that triggered the escalation.
        """
        if self._event_bus is None:
            return

        event = Event(
            event_type=KILL_SWITCH_ACTIVATED,
            payload={
                "trigger_source": "crisis_persistence",
                "crisis_id": crisis.id,
                "region": crisis.region,
                "sentiment_avg": crisis.sentiment_avg,
                "started_at": crisis.started_at.isoformat(),
                "reason": (
                    f"News crisis persistent for {self.PERSISTENCE_MINUTES} minutes "
                    f"in region '{crisis.region}' with no sentiment recovery above "
                    f"{NEWS_CRISIS_RECOVERY_THRESHOLD}"
                ),
            },
        )
        self._pending_kill_switch_event = event

    def _prune_old_articles(self) -> None:
        """Remove articles older than the time window from tracking."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.TIME_WINDOW_MINUTES
        )
        while self._recent_articles and self._recent_articles[0].received_at < cutoff:
            self._recent_articles.popleft()

    def _get_articles_in_window(self) -> list[CrisisArticleRecord]:
        """Get all articles within the current time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.TIME_WINDOW_MINUTES
        )
        return [a for a in self._recent_articles if a.received_at >= cutoff]
