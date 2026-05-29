"""Unit tests for the CrisisDetector class.

Tests crisis event detection logic including:
- 3+ HIGH-impact articles with sentiment < -0.7 within 10-minute window
- Grouping by region and asset class
- Duplicate crisis prevention for same region/asset class
- Sliding window cleanup of expired articles
- Crisis persistence and resolution
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.news.crisis_detector import CrisisAlertData, CrisisDetector


def _make_article(
    sentiment: float = -0.8,
    impact_level: str = "HIGH",
    region: str | None = "europe",
    asset_class: str | None = None,
    received_at: datetime | None = None,
    article_id: str | None = None,
    source: str = "reuters",
    headline: str = "Market crash",
) -> dict:
    """Helper to create a test article dict."""
    if received_at is None:
        received_at = datetime.now(timezone.utc)
    return {
        "id": article_id or f"art-{id(received_at)}",
        "source": source,
        "headline": headline,
        "sentiment_score": sentiment,
        "impact_level": impact_level,
        "received_at": received_at,
        "region": region,
        "asset_class": asset_class,
        "category": "geopolitical_conflict",
    }


class TestCrisisDetectorEvaluate:
    """Tests for the evaluate method."""

    def test_ignores_non_high_impact_articles(self):
        """Articles with impact != HIGH should not trigger crisis."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(5):
            result = detector.evaluate(
                _make_article(
                    sentiment=-0.9,
                    impact_level="MEDIUM",
                    region="europe",
                    received_at=now + timedelta(seconds=i),
                )
            )
            assert result is None

    def test_ignores_articles_above_sentiment_threshold(self):
        """Articles with sentiment > -0.7 should not trigger crisis."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(5):
            result = detector.evaluate(
                _make_article(
                    sentiment=-0.5,
                    impact_level="HIGH",
                    region="europe",
                    received_at=now + timedelta(seconds=i),
                )
            )
            assert result is None

    def test_triggers_crisis_at_threshold(self):
        """3 HIGH-impact articles with sentiment < -0.7 in same region triggers crisis."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # First two articles - no crisis yet
        result1 = detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="europe",
                received_at=now,
                article_id="art-1",
            )
        )
        assert result1 is None

        result2 = detector.evaluate(
            _make_article(
                sentiment=-0.9,
                region="europe",
                received_at=now + timedelta(seconds=30),
                article_id="art-2",
            )
        )
        assert result2 is None

        # Third article triggers crisis
        result3 = detector.evaluate(
            _make_article(
                sentiment=-0.75,
                region="europe",
                received_at=now + timedelta(seconds=60),
                article_id="art-3",
            )
        )
        assert result3 is not None
        assert isinstance(result3, CrisisAlertData)
        assert result3.region == "europe"
        assert result3.active is True
        assert len(result3.trigger_articles) == 3
        assert result3.sentiment_avg == pytest.approx((-0.8 + -0.9 + -0.75) / 3)

    def test_triggers_crisis_by_asset_class(self):
        """3 HIGH-impact articles referencing same asset class triggers crisis."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(2):
            detector.evaluate(
                _make_article(
                    sentiment=-0.85,
                    region=None,
                    asset_class="forex",
                    received_at=now + timedelta(seconds=i * 30),
                    article_id=f"art-{i}",
                )
            )

        result = detector.evaluate(
            _make_article(
                sentiment=-0.9,
                region=None,
                asset_class="forex",
                received_at=now + timedelta(seconds=90),
                article_id="art-2",
            )
        )
        assert result is not None
        assert result.region == "forex"

    def test_different_regions_do_not_combine(self):
        """Articles from different regions should not combine to trigger crisis."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # 2 articles for europe
        for i in range(2):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"eu-{i}",
                )
            )

        # 1 article for asia - should not trigger crisis for either region
        result = detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="asia",
                received_at=now + timedelta(seconds=30),
                article_id="asia-0",
            )
        )
        assert result is None

    def test_no_duplicate_crisis_for_same_region(self):
        """Once a crisis is active for a region, additional articles don't trigger new one."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        # Additional article should not trigger another crisis
        result = detector.evaluate(
            _make_article(
                sentiment=-0.9,
                region="europe",
                received_at=now + timedelta(seconds=60),
                article_id="art-extra",
            )
        )
        assert result is None

    def test_articles_outside_window_do_not_count(self):
        """Articles older than 10 minutes should be pruned and not count."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Two articles 11 minutes ago (outside window)
        old_time = now - timedelta(minutes=11)
        for i in range(2):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=old_time + timedelta(seconds=i),
                    article_id=f"old-{i}",
                )
            )

        # One recent article - should not trigger (only 1 in window)
        result = detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="europe",
                received_at=now,
                article_id="new-0",
            )
        )
        assert result is None

    def test_crisis_alert_has_required_fields(self):
        """CrisisAlertData should have id, region, sentiment_avg, trigger_articles, started_at."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(3):
            result = detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="middle_east",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        assert result is not None
        assert result.id is not None and len(result.id) > 0
        assert result.region == "middle_east"
        assert result.sentiment_avg < 0
        assert len(result.trigger_articles) >= 3
        assert result.started_at is not None

    def test_global_fallback_when_no_region_or_asset_class(self):
        """Articles without region or asset_class use global grouping."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(3):
            result = detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region=None,
                    asset_class=None,
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        assert result is not None
        assert result.region == "global"

    def test_article_with_both_region_and_asset_class(self):
        """An article with both region and asset_class can trigger crisis via either group."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # 2 articles with region=europe, asset_class=equities
        for i in range(2):
            detector.evaluate(
                _make_article(
                    sentiment=-0.85,
                    region="europe",
                    asset_class="equities",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"both-{i}",
                )
            )

        # 1 article with only region=europe (no asset_class)
        # This should trigger crisis for "europe" since 3 articles match region:europe
        result = detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="europe",
                asset_class=None,
                received_at=now + timedelta(seconds=30),
                article_id="region-only",
            )
        )
        assert result is not None
        assert result.region == "europe"


class TestCrisisDetectorPersistence:
    """Tests for crisis persistence checking."""

    def test_persistence_returns_false_before_window(self):
        """Should not escalate before 30 minutes have passed."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger a crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        crisis = detector.active_crises[0]
        # Check persistence immediately (before 30 min)
        result = detector.check_persistence(crisis.id, [-0.8, -0.9])
        assert result is False

    def test_persistence_escalates_after_window_no_recovery(self):
        """Should escalate to kill switch after 30 min with no recovery."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger a crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        crisis = detector.active_crises[0]
        # Simulate time passing beyond persistence window
        crisis.started_at = now - timedelta(minutes=31)

        result = detector.check_persistence(crisis.id, [-0.8, -0.9, -0.5])
        assert result is True
        assert crisis.escalated_to_kill_switch is True

    def test_persistence_resolves_on_recovery(self):
        """Should resolve crisis if sentiment recovers above -0.3."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger a crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        crisis = detector.active_crises[0]
        crisis.started_at = now - timedelta(minutes=31)

        # Recovery detected (sentiment > -0.3)
        result = detector.check_persistence(crisis.id, [-0.8, -0.2])
        assert result is False
        assert crisis.active is False

    def test_persistence_returns_false_for_unknown_crisis(self):
        """Should return False for non-existent crisis ID."""
        detector = CrisisDetector()
        result = detector.check_persistence("nonexistent-id", [-0.8])
        assert result is False


class TestCrisisDetectorResolution:
    """Tests for crisis resolution."""

    def test_resolve_marks_crisis_inactive(self):
        """Resolving a crisis should mark it inactive."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        crisis = detector.active_crises[0]
        detector.resolve_crisis(crisis.id)

        assert crisis.active is False
        assert crisis.resolved_at is not None
        assert len(detector.active_crises) == 0

    def test_new_crisis_can_trigger_after_resolution(self):
        """After resolving a crisis, a new one can trigger for the same region."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger first crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"first-{i}",
                )
            )

        crisis = detector.active_crises[0]
        detector.resolve_crisis(crisis.id)

        # After resolution, old articles are still in the window.
        # A new article for the same region should immediately re-trigger
        # since there are already 3+ articles in the window for that region.
        result = detector.evaluate(
            _make_article(
                sentiment=-0.85,
                region="europe",
                received_at=now + timedelta(seconds=60),
                article_id="second-0",
            )
        )
        assert result is not None
        assert result.region == "europe"
        # The trigger articles include both old and new articles in the window
        assert len(result.trigger_articles) >= 3


class TestCrisisDetectorSlidingWindow:
    """Tests for sliding window cleanup."""

    def test_old_articles_are_pruned(self):
        """Articles older than the time window should be removed."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Add article 11 minutes ago
        detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="europe",
                received_at=now - timedelta(minutes=11),
                article_id="old",
            )
        )

        # Add a new article to trigger pruning
        detector.evaluate(
            _make_article(
                sentiment=-0.8,
                region="europe",
                received_at=now,
                article_id="new",
            )
        )

        # Only the new article should remain
        articles = detector._get_articles_in_window()
        assert len(articles) == 1
        assert articles[0].article_id == "new"

    def test_active_crises_property(self):
        """active_crises should only return active (non-resolved) crises."""
        detector = CrisisDetector()
        now = datetime.now(timezone.utc)

        # Trigger a crisis
        for i in range(3):
            detector.evaluate(
                _make_article(
                    sentiment=-0.8,
                    region="europe",
                    received_at=now + timedelta(seconds=i * 10),
                    article_id=f"art-{i}",
                )
            )

        assert len(detector.active_crises) == 1

        # Resolve it
        detector.resolve_crisis(detector.active_crises[0].id)
        assert len(detector.active_crises) == 0
