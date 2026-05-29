"""Unit tests for the GeopoliticalRiskScorer.

Tests cover:
- Per-region risk scoring in range [0, 100]
- Risk factor contributions (armed_conflict, sanctions, political_instability, natural_disaster)
- Score decay over time
- High-risk threshold detection (>= 70)
- Event bus publishing on threshold crossing
- All supported regions initialization
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.event_bus import Event, RISK_ALERT
from src.news.geopolitical_risk import (
    DEFAULT_DECAY_RATE,
    HIGH_RISK_THRESHOLD,
    RISK_FACTOR_WEIGHTS,
    SUPPORTED_REGIONS,
    GeopoliticalRiskScorer,
    RegionRiskState,
    RiskArticle,
    RiskFactor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> GeopoliticalRiskScorer:
    """Create a scorer with no event bus."""
    return GeopoliticalRiskScorer()


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def scorer_with_bus(mock_event_bus: AsyncMock) -> GeopoliticalRiskScorer:
    """Create a scorer with a mock event bus."""
    return GeopoliticalRiskScorer(event_bus=mock_event_bus)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_all_supported_regions_initialized(self, scorer: GeopoliticalRiskScorer) -> None:
        scores = scorer.get_all_scores()
        for region in SUPPORTED_REGIONS:
            assert region in scores
            assert scores[region] == 0

    def test_default_decay_rate(self, scorer: GeopoliticalRiskScorer) -> None:
        assert scorer.decay_rate == DEFAULT_DECAY_RATE

    def test_custom_decay_rate(self) -> None:
        scorer = GeopoliticalRiskScorer(decay_rate=5.0)
        assert scorer.decay_rate == 5.0

    def test_negative_decay_rate_clamped(self) -> None:
        scorer = GeopoliticalRiskScorer(decay_rate=-1.0)
        # Setter clamps to 0
        scorer.decay_rate = -1.0
        assert scorer.decay_rate == 0.0

    def test_initial_scores_are_zero(self, scorer: GeopoliticalRiskScorer) -> None:
        for region in SUPPORTED_REGIONS:
            assert scorer.get_risk_score(region) == 0

    def test_unknown_region_returns_zero(self, scorer: GeopoliticalRiskScorer) -> None:
        assert scorer.get_risk_score("Unknown") == 0


# ---------------------------------------------------------------------------
# Risk Factor Contribution Tests
# ---------------------------------------------------------------------------


class TestRiskFactorContributions:
    def test_armed_conflict_adds_30_points(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a1",
            region="US",
            risk_factor=RiskFactor.ARMED_CONFLICT,
            severity=1.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("US") == 30

    def test_sanctions_adds_20_points(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a2",
            region="Europe",
            risk_factor=RiskFactor.SANCTIONS,
            severity=1.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("Europe") == 20

    def test_political_instability_adds_15_points(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a3",
            region="Asia",
            risk_factor=RiskFactor.POLITICAL_INSTABILITY,
            severity=1.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("Asia") == 15

    def test_natural_disaster_adds_25_points(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a4",
            region="Oceania",
            risk_factor=RiskFactor.NATURAL_DISASTER,
            severity=1.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("Oceania") == 25

    def test_severity_scales_contribution(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a5",
            region="US",
            risk_factor=RiskFactor.ARMED_CONFLICT,
            severity=0.5,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("US") == 15  # 30 * 0.5

    def test_multiple_factors_accumulate(self, scorer: GeopoliticalRiskScorer) -> None:
        articles = [
            RiskArticle(
                article_id="a1",
                region="Middle East",
                risk_factor=RiskFactor.ARMED_CONFLICT,
                severity=1.0,
            ),
            RiskArticle(
                article_id="a2",
                region="Middle East",
                risk_factor=RiskFactor.SANCTIONS,
                severity=1.0,
            ),
        ]
        scorer.process_articles(articles)
        # 30 + 20 = 50
        assert scorer.get_risk_score("Middle East") == 50

    def test_score_capped_at_100(self, scorer: GeopoliticalRiskScorer) -> None:
        # Add enough articles to exceed 100
        for i in range(5):
            scorer.process_article(
                RiskArticle(
                    article_id=f"a{i}",
                    region="Africa",
                    risk_factor=RiskFactor.ARMED_CONFLICT,
                    severity=1.0,
                )
            )
        assert scorer.get_risk_score("Africa") == 100

    def test_severity_clamped_to_0_1(self, scorer: GeopoliticalRiskScorer) -> None:
        # Severity > 1.0 should be clamped to 1.0
        article = RiskArticle(
            article_id="a1",
            region="US",
            risk_factor=RiskFactor.ARMED_CONFLICT,
            severity=2.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("US") == 30  # clamped to 1.0 * 30

    def test_negative_severity_clamped_to_zero(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a1",
            region="US",
            risk_factor=RiskFactor.ARMED_CONFLICT,
            severity=-0.5,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("US") == 0

    def test_factor_contributions_breakdown(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.process_article(
            RiskArticle(
                article_id="a1",
                region="US",
                risk_factor=RiskFactor.ARMED_CONFLICT,
                severity=1.0,
            )
        )
        scorer.process_article(
            RiskArticle(
                article_id="a2",
                region="US",
                risk_factor=RiskFactor.SANCTIONS,
                severity=0.5,
            )
        )
        contributions = scorer.get_factor_contributions("US")
        assert contributions["armed_conflict"] == 30.0
        assert contributions["sanctions"] == 10.0
        assert contributions["political_instability"] == 0.0
        assert contributions["natural_disaster"] == 0.0


# ---------------------------------------------------------------------------
# High-Risk Threshold Tests
# ---------------------------------------------------------------------------


class TestHighRiskThreshold:
    def test_is_high_risk_at_threshold(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 70)
        assert scorer.is_high_risk("US") is True

    def test_is_not_high_risk_below_threshold(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 69)
        assert scorer.is_high_risk("US") is False

    def test_is_high_risk_above_threshold(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 85)
        assert scorer.is_high_risk("US") is True

    def test_unknown_region_not_high_risk(self, scorer: GeopoliticalRiskScorer) -> None:
        assert scorer.is_high_risk("Unknown") is False

    def test_get_high_risk_regions(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 80)
        scorer.set_score("Europe", 50)
        scorer.set_score("Asia", 75)
        high_risk = scorer.get_high_risk_regions()
        assert "US" in high_risk
        assert "Asia" in high_risk
        assert "Europe" not in high_risk

    def test_no_high_risk_regions_initially(self, scorer: GeopoliticalRiskScorer) -> None:
        assert scorer.get_high_risk_regions() == []


# ---------------------------------------------------------------------------
# Score Decay Tests
# ---------------------------------------------------------------------------


class TestScoreDecay:
    def test_decay_reduces_score(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 50)
        scorer.apply_decay()
        assert scorer.get_risk_score("US") < 50

    def test_decay_does_not_go_below_zero(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 1)
        # Apply decay multiple times
        for _ in range(100):
            scorer.apply_decay()
        assert scorer.get_risk_score("US") == 0

    def test_zero_score_not_affected_by_decay(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.apply_decay()
        assert scorer.get_risk_score("US") == 0

    def test_custom_decay_rate(self) -> None:
        scorer = GeopoliticalRiskScorer(decay_rate=10.0)
        scorer.set_score("US", 50)
        scorer.apply_decay()
        # With decay_rate=10, score should decrease by 10
        assert scorer.get_risk_score("US") == 40

    def test_zero_decay_rate_no_change(self) -> None:
        scorer = GeopoliticalRiskScorer(decay_rate=0.0)
        scorer.set_score("US", 50)
        scorer.apply_decay()
        assert scorer.get_risk_score("US") == 50

    def test_decay_affects_all_regions(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 50)
        scorer.set_score("Europe", 30)
        scorer.apply_decay()
        assert scorer.get_risk_score("US") < 50
        assert scorer.get_risk_score("Europe") < 30


# ---------------------------------------------------------------------------
# Event Bus Integration Tests
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_event_published_on_threshold_crossing_up(
        self, scorer_with_bus: GeopoliticalRiskScorer, mock_event_bus: AsyncMock
    ) -> None:
        scorer_with_bus.set_score("US", 75)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == RISK_ALERT
        event: Event = call_args[0][1]
        assert event.payload["alert_type"] == "geopolitical_risk_elevated"
        assert event.payload["region"] == "US"
        assert event.payload["score"] == 75
        assert event.payload["direction"] == "above"

    @pytest.mark.asyncio
    async def test_event_published_on_threshold_crossing_down(
        self, scorer_with_bus: GeopoliticalRiskScorer, mock_event_bus: AsyncMock
    ) -> None:
        # First cross above
        scorer_with_bus.set_score("US", 80)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.reset_mock()

        # Then cross below
        scorer_with_bus.set_score("US", 60)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        event: Event = call_args[0][1]
        assert event.payload["alert_type"] == "geopolitical_risk_recovered"
        assert event.payload["direction"] == "below"

    @pytest.mark.asyncio
    async def test_no_event_when_staying_below_threshold(
        self, scorer_with_bus: GeopoliticalRiskScorer, mock_event_bus: AsyncMock
    ) -> None:
        scorer_with_bus.set_score("US", 30)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_event_when_staying_above_threshold(
        self, scorer_with_bus: GeopoliticalRiskScorer, mock_event_bus: AsyncMock
    ) -> None:
        scorer_with_bus.set_score("US", 75)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.reset_mock()

        # Change score but stay above threshold
        scorer_with_bus.set_score("US", 80)
        await scorer_with_bus.dispatch_pending_events()
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_event_without_event_bus(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 80)
        # Should not raise even without event bus
        await scorer.dispatch_pending_events()

    @pytest.mark.asyncio
    async def test_event_bus_error_handled_gracefully(
        self, mock_event_bus: AsyncMock
    ) -> None:
        mock_event_bus.publish.side_effect = Exception("Connection failed")
        scorer = GeopoliticalRiskScorer(event_bus=mock_event_bus)
        scorer.set_score("US", 80)
        # Should not raise
        await scorer.dispatch_pending_events()


# ---------------------------------------------------------------------------
# Legacy Interface Tests
# ---------------------------------------------------------------------------


class TestLegacyInterface:
    def test_update_scores_with_indicators(self, scorer: GeopoliticalRiskScorer) -> None:
        indicators = {
            "US": [
                {"type": "armed_conflict", "severity": 1.0},
                {"type": "sanctions", "severity": 0.5},
            ]
        }
        scorer.update_scores(indicators)
        # 30 + 10 = 40
        assert scorer.get_risk_score("US") == 40

    def test_update_scores_unknown_type_ignored(self, scorer: GeopoliticalRiskScorer) -> None:
        indicators = {
            "US": [
                {"type": "unknown_factor", "severity": 1.0},
            ]
        }
        scorer.update_scores(indicators)
        assert scorer.get_risk_score("US") == 0

    def test_set_score_clamps_to_range(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 150)
        assert scorer.get_risk_score("US") == 100

        scorer.set_score("US", -10)
        assert scorer.get_risk_score("US") == 0

    def test_reset_region(self, scorer: GeopoliticalRiskScorer) -> None:
        scorer.set_score("US", 80)
        scorer.reset_region("US")
        assert scorer.get_risk_score("US") == 0


# ---------------------------------------------------------------------------
# Auto-Registration Tests
# ---------------------------------------------------------------------------


class TestAutoRegistration:
    def test_unknown_region_auto_registered(self, scorer: GeopoliticalRiskScorer) -> None:
        article = RiskArticle(
            article_id="a1",
            region="Antarctica",
            risk_factor=RiskFactor.NATURAL_DISASTER,
            severity=1.0,
        )
        scorer.process_article(article)
        assert scorer.get_risk_score("Antarctica") == 25

    def test_get_factor_contributions_unknown_region(
        self, scorer: GeopoliticalRiskScorer
    ) -> None:
        assert scorer.get_factor_contributions("Unknown") == {}


# ---------------------------------------------------------------------------
# Lifecycle Tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, scorer: GeopoliticalRiskScorer) -> None:
        await scorer.start()
        assert scorer.is_running is True
        await scorer.stop()
        assert scorer.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, scorer: GeopoliticalRiskScorer) -> None:
        await scorer.start()
        await scorer.start()  # Should not raise
        assert scorer.is_running is True
        await scorer.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, scorer: GeopoliticalRiskScorer) -> None:
        await scorer.stop()  # Should not raise
        assert scorer.is_running is False
