"""Unit tests for crisis persistence check (Task 37.4).

Tests the CrisisDetector's ability to:
- Monitor incoming articles for sentiment recovery after a crisis is detected
- Resolve a crisis when sentiment > -0.3 for the same region is received
- Escalate to Kill Switch when no recovery within 30 minutes
- Emit KILL_SWITCH activation event via the Event Bus
- Track per-region/asset-class sentiment readings

Validates: Requirements 23.9, Cross-Cutting Rule 3
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.config.constants import (
    NEWS_CRISIS_ARTICLE_THRESHOLD,
    NEWS_CRISIS_PERSISTENCE_MINUTES,
    NEWS_CRISIS_RECOVERY_THRESHOLD,
    NEWS_CRISIS_SENTIMENT_THRESHOLD,
)
from src.core.event_bus import KILL_SWITCH_ACTIVATED, Event
from src.news.crisis_detector import CrisisAlertData, CrisisDetector


# =============================================================================
# Helpers
# =============================================================================


def _make_crisis_article(
    sentiment: float = -0.8,
    region: str = "global",
    offset_seconds: int = 0,
    source: str | None = None,
) -> dict:
    """Create a high-impact negative article for crisis triggering."""
    return {
        "id": str(uuid.uuid4()),
        "source": source or f"source_{uuid.uuid4().hex[:4]}",
        "headline": "Crisis headline",
        "sentiment_score": sentiment,
        "impact_level": "HIGH",
        "received_at": datetime.now(timezone.utc) - timedelta(seconds=offset_seconds),
        "category": "geopolitical_conflict",
        "region": region,
    }


def _trigger_crisis(detector: CrisisDetector, region: str = "global") -> CrisisAlertData:
    """Trigger a crisis by feeding enough high-impact articles."""
    result = None
    for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
        result = detector.evaluate(_make_crisis_article(region=region))
    assert result is not None
    return result


# =============================================================================
# Tests: Recovery Detection
# =============================================================================


class TestCrisisPersistenceRecovery:
    """Tests for crisis recovery via sentiment improvement."""

    def setup_method(self) -> None:
        self.detector = CrisisDetector()

    def test_recovery_resolves_crisis_on_article_evaluation(self) -> None:
        """An article with sentiment > -0.3 for the same region resolves the crisis."""
        crisis = _trigger_crisis(self.detector, region="europe")

        # Feed a recovery article for the same region
        recovery_article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Markets stabilize",
            "sentiment_score": -0.1,  # Above -0.3 threshold
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "europe",
        }
        self.detector.evaluate(recovery_article)

        # Crisis should be resolved
        assert crisis.active is False
        assert crisis.resolved_at is not None

    def test_recovery_at_exact_threshold_does_not_resolve(self) -> None:
        """Sentiment exactly at -0.3 does NOT resolve the crisis (must be > -0.3)."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Feed article with sentiment exactly at threshold
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Slight improvement",
            "sentiment_score": NEWS_CRISIS_RECOVERY_THRESHOLD,  # -0.3 exactly
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        # Crisis should still be active (threshold is strictly >)
        assert crisis.active is True

    def test_recovery_above_threshold_resolves(self) -> None:
        """Sentiment just above -0.3 resolves the crisis."""
        crisis = _trigger_crisis(self.detector, region="global")

        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Recovery signs",
            "sentiment_score": NEWS_CRISIS_RECOVERY_THRESHOLD + 0.01,
            "impact_level": "LOW",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        assert crisis.active is False

    def test_different_region_does_not_resolve_crisis(self) -> None:
        """Recovery in a different region does not resolve the crisis."""
        crisis = _trigger_crisis(self.detector, region="asia")

        # Feed recovery article for a different region
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Europe recovers",
            "sentiment_score": 0.5,  # Very positive
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "europe",
        }
        self.detector.evaluate(article)

        # Crisis in Asia should still be active
        assert crisis.active is True

    def test_global_region_matches_any_article(self) -> None:
        """A crisis with region 'global' is resolved by any region's recovery."""
        crisis = _trigger_crisis(self.detector, region="global")

        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Recovery",
            "sentiment_score": 0.0,  # Above -0.3
            "impact_level": "LOW",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "europe",
        }
        self.detector.evaluate(article)

        assert crisis.active is False


# =============================================================================
# Tests: Persistence Escalation to Kill Switch
# =============================================================================


class TestCrisisPersistenceEscalation:
    """Tests for Kill Switch escalation when crisis persists."""

    def setup_method(self) -> None:
        self.detector = CrisisDetector()

    def test_persistence_escalates_after_30_minutes(self) -> None:
        """Crisis without recovery after 30 minutes escalates to Kill Switch."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Backdate crisis start to simulate 30+ minutes elapsed
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        # Feed another negative article (no recovery)
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Continued crisis",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        assert crisis.escalated_to_kill_switch is True

    def test_no_escalation_before_30_minutes(self) -> None:
        """Crisis does not escalate before 30 minutes even without recovery."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Crisis just started — feed negative articles
        for _ in range(5):
            article = {
                "id": str(uuid.uuid4()),
                "source": "reuters",
                "headline": "Bad news",
                "sentiment_score": -0.9,
                "impact_level": "MEDIUM",
                "received_at": datetime.now(timezone.utc),
                "category": "markets",
                "region": "global",
            }
            self.detector.evaluate(article)

        assert crisis.escalated_to_kill_switch is False
        assert crisis.active is True

    def test_recovery_prevents_escalation(self) -> None:
        """Recovery before 30 minutes prevents Kill Switch escalation."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Backdate to 25 minutes (before persistence window)
        crisis.started_at = datetime.now(timezone.utc) - timedelta(minutes=25)

        # Feed recovery article
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Markets recover",
            "sentiment_score": 0.0,
            "impact_level": "LOW",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        # Crisis resolved, no escalation
        assert crisis.active is False
        assert crisis.escalated_to_kill_switch is False

    def test_escalation_only_happens_once(self) -> None:
        """Once escalated, subsequent articles don't re-escalate."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Backdate crisis
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 5
        )

        # First article triggers escalation
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Still bad",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)
        assert crisis.escalated_to_kill_switch is True

        # Second article — already escalated, should not re-process
        article2 = {
            "id": str(uuid.uuid4()),
            "source": "bloomberg",
            "headline": "More bad news",
            "sentiment_score": -0.9,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        # Should not raise or cause issues
        self.detector.evaluate(article2)
        assert crisis.escalated_to_kill_switch is True

    def test_check_persistence_method_with_no_recovery(self) -> None:
        """Direct check_persistence call escalates when no recovery."""
        crisis = _trigger_crisis(self.detector, region="global")

        after_window = crisis.started_at + timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        result = self.detector.check_persistence(
            crisis.id,
            current_sentiments=[-0.8, -0.9, -0.7],
            current_time=after_window,
        )
        assert result is True
        assert crisis.escalated_to_kill_switch is True

    def test_check_persistence_method_with_recovery(self) -> None:
        """Direct check_persistence resolves crisis when recovery found."""
        crisis = _trigger_crisis(self.detector, region="global")

        after_window = crisis.started_at + timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        result = self.detector.check_persistence(
            crisis.id,
            current_sentiments=[-0.8, -0.1, -0.7],  # -0.1 is recovery
            current_time=after_window,
        )
        assert result is False
        assert crisis.active is False

    def test_check_persistence_uses_internal_sentiments_when_none(self) -> None:
        """check_persistence uses internally tracked sentiments when none provided."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Feed some negative articles to build internal tracking
        for _ in range(3):
            article = {
                "id": str(uuid.uuid4()),
                "source": "reuters",
                "headline": "Bad",
                "sentiment_score": -0.8,
                "impact_level": "LOW",
                "received_at": datetime.now(timezone.utc),
                "category": "markets",
                "region": "global",
            }
            self.detector.evaluate(article)

        # Backdate crisis
        after_window = crisis.started_at + timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        result = self.detector.check_persistence(
            crisis.id,
            current_sentiments=None,
            current_time=after_window,
        )
        assert result is True


# =============================================================================
# Tests: Event Bus Integration
# =============================================================================


class TestCrisisPersistenceEventBus:
    """Tests for Kill Switch event emission via Event Bus."""

    def setup_method(self) -> None:
        self.mock_event_bus = AsyncMock()
        self.mock_event_bus.publish = AsyncMock(return_value=1)
        self.detector = CrisisDetector(event_bus=self.mock_event_bus)

    def test_kill_switch_event_emitted_on_persistence(self) -> None:
        """Kill Switch activation event is emitted when crisis persists."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Backdate crisis
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        # Feed negative article to trigger persistence check
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Continued crisis",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        # Verify pending event was created
        assert hasattr(self.detector, "_pending_kill_switch_event")
        assert self.detector._pending_kill_switch_event is not None

    @pytest.mark.asyncio
    async def test_dispatch_pending_events_publishes(self) -> None:
        """dispatch_pending_events publishes the Kill Switch event."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Backdate crisis
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        # Trigger persistence
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Continued crisis",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        # Dispatch
        await self.detector.dispatch_pending_events()

        # Verify event was published
        self.mock_event_bus.publish.assert_called_once()
        call_args = self.mock_event_bus.publish.call_args
        assert call_args[0][0] == KILL_SWITCH_ACTIVATED
        event = call_args[0][1]
        assert event.event_type == KILL_SWITCH_ACTIVATED
        assert event.payload["trigger_source"] == "crisis_persistence"
        assert event.payload["crisis_id"] == crisis.id

    @pytest.mark.asyncio
    async def test_dispatch_clears_pending_event(self) -> None:
        """After dispatch, pending event is cleared."""
        crisis = _trigger_crisis(self.detector, region="global")
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Bad",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)
        await self.detector.dispatch_pending_events()

        # Pending event should be cleared
        assert self.detector._pending_kill_switch_event is None

    @pytest.mark.asyncio
    async def test_no_event_emitted_without_event_bus(self) -> None:
        """No event is emitted when event_bus is None."""
        detector = CrisisDetector(event_bus=None)
        crisis = _trigger_crisis(detector, region="global")
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Bad",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        detector.evaluate(article)
        await detector.dispatch_pending_events()

        # Should not raise, just no-op

    @pytest.mark.asyncio
    async def test_event_payload_contains_reason(self) -> None:
        """Kill Switch event payload includes a human-readable reason."""
        crisis = _trigger_crisis(self.detector, region="europe")
        crisis.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Europe crisis continues",
            "sentiment_score": -0.8,
            "impact_level": "MEDIUM",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "europe",
        }
        self.detector.evaluate(article)
        await self.detector.dispatch_pending_events()

        call_args = self.mock_event_bus.publish.call_args
        event = call_args[0][1]
        assert "europe" in event.payload["reason"]
        assert str(NEWS_CRISIS_PERSISTENCE_MINUTES) in event.payload["reason"]
        assert event.payload["region"] == "europe"


# =============================================================================
# Tests: Sentiment Tracking
# =============================================================================


class TestCrisisSentimentTracking:
    """Tests for per-crisis sentiment tracking."""

    def setup_method(self) -> None:
        self.detector = CrisisDetector()

    def test_sentiments_tracked_after_crisis_starts(self) -> None:
        """Sentiment readings are tracked for active crises."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Feed articles after crisis
        for i in range(3):
            article = {
                "id": str(uuid.uuid4()),
                "source": "reuters",
                "headline": f"Update {i}",
                "sentiment_score": -0.5 - (i * 0.1),
                "impact_level": "LOW",
                "received_at": datetime.now(timezone.utc),
                "category": "markets",
                "region": "global",
            }
            self.detector.evaluate(article)

        # Check internal tracking
        sentiments = self.detector._crisis_sentiments.get(crisis.id, [])
        assert len(sentiments) == 3

    def test_sentiments_cleaned_up_on_resolve(self) -> None:
        """Sentiment tracking is cleaned up when crisis is resolved."""
        crisis = _trigger_crisis(self.detector, region="global")

        # Feed some articles
        article = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Update",
            "sentiment_score": -0.5,
            "impact_level": "LOW",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "global",
        }
        self.detector.evaluate(article)

        # Resolve
        self.detector.resolve_crisis(crisis.id)

        # Sentiments should be cleaned up
        assert crisis.id not in self.detector._crisis_sentiments

    def test_multiple_crises_tracked_independently(self) -> None:
        """Different region crises track sentiments independently."""
        # Trigger crisis in region A
        crisis_a = _trigger_crisis(self.detector, region="asia")

        # Now trigger crisis in region B (need to clear the window first)
        # Since the detector checks for active crises per region, we can
        # trigger another for a different region
        detector2 = CrisisDetector()
        crisis_b = _trigger_crisis(detector2, region="europe")

        # Feed articles for region A
        article_a = {
            "id": str(uuid.uuid4()),
            "source": "reuters",
            "headline": "Asia update",
            "sentiment_score": -0.6,
            "impact_level": "LOW",
            "received_at": datetime.now(timezone.utc),
            "category": "markets",
            "region": "asia",
        }
        self.detector.evaluate(article_a)

        sentiments_a = self.detector._crisis_sentiments.get(crisis_a.id, [])
        assert len(sentiments_a) == 1
        assert sentiments_a[0].region == "asia"
