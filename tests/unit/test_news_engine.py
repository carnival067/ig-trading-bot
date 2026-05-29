"""Unit tests for the News Engine module.

Covers sentiment analysis bounds, impact classification, crisis detection,
economic calendar logic, geopolitical risk scoring, and correlation mapping.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.config.constants import (
    GEOPOLITICAL_RISK_HIGH_THRESHOLD,
    NEWS_CRISIS_ARTICLE_THRESHOLD,
    NEWS_CRISIS_PERSISTENCE_MINUTES,
    NEWS_CRISIS_RECOVERY_THRESHOLD,
    NEWS_CRISIS_SENTIMENT_THRESHOLD,
    NEWS_CRISIS_TIME_WINDOW_MINUTES,
    NEWS_EVENT_SIZE_REDUCTION_FACTOR,
    NEWS_PRE_EVENT_RISK_WINDOW_MINUTES,
    NEWS_PRE_EVENT_SIGNAL_PAUSE_MINUTES,
    SOURCE_CREDIBILITY_SOCIAL,
    SOURCE_CREDIBILITY_TIER1,
    SOURCE_CREDIBILITY_TIER2,
)
from src.news.correlation_mapper import CorrelationMapper
from src.news.crisis_detector import CrisisDetector
from src.news.economic_calendar import EconomicCalendar, EconomicEventData
from src.news.geopolitical_risk import GeopoliticalRiskScorer
from src.news.news_engine import NewsEngine
from src.news.sentiment_analyzer import SentimentAnalyzer
from src.news.sources.base import NewsSource, RawArticle
from src.news.sources.bloomberg import BloombergSource
from src.news.sources.reuters import ReutersSource
from src.news.sources.social_media import SocialMediaSource


# =============================================================================
# Sentiment Analyzer Tests
# =============================================================================


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    def setup_method(self) -> None:
        self.analyzer = SentimentAnalyzer()

    def test_analyze_returns_score_in_valid_range(self) -> None:
        """Sentiment score must be in [-1.0, +1.0]."""
        score = self.analyzer.analyze("Market crash leads to massive sell-off")
        assert -1.0 <= score <= 1.0

    def test_analyze_negative_text(self) -> None:
        """Negative text should produce negative sentiment."""
        score = self.analyzer.analyze("Market crash and crisis lead to collapse")
        assert score < 0.0

    def test_analyze_positive_text(self) -> None:
        """Positive text should produce positive sentiment."""
        score = self.analyzer.analyze("Strong earnings surge and bullish rally")
        assert score > 0.0

    def test_analyze_neutral_text(self) -> None:
        """Neutral text should produce score near zero."""
        score = self.analyzer.analyze("The meeting was held on Tuesday")
        assert score == 0.0

    def test_analyze_empty_text(self) -> None:
        """Empty text returns 0.0."""
        assert self.analyzer.analyze("") == 0.0
        assert self.analyzer.analyze("   ") == 0.0

    def test_analyze_clamps_to_bounds(self) -> None:
        """Score is always clamped to [-1.0, +1.0]."""
        # Even with many negative keywords, should not exceed -1.0
        text = "crash crisis collapse bankruptcy default recession war sanctions"
        score = self.analyzer.analyze(text)
        assert score >= -1.0

    def test_classify_impact_high_tier1_strong_sentiment(self) -> None:
        """Tier-1 source with |sentiment| > 0.7 → HIGH impact."""
        result = self.analyzer.classify_impact(
            sentiment_score=-0.8,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == "HIGH"

    def test_classify_impact_high_corroboration(self) -> None:
        """Any source with corroboration >= 2 and |sentiment| > 0.5 → HIGH."""
        result = self.analyzer.classify_impact(
            sentiment_score=0.6,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=2,
        )
        assert result == "HIGH"

    def test_classify_impact_medium_tier2(self) -> None:
        """Tier-2 source with |sentiment| > 0.5 → MEDIUM."""
        result = self.analyzer.classify_impact(
            sentiment_score=-0.6,
            source_tier=SOURCE_CREDIBILITY_TIER2,
            corroboration=0,
        )
        assert result == "MEDIUM"

    def test_classify_impact_medium_tier1_moderate(self) -> None:
        """Tier-1 source with moderate sentiment (0.3 < |sentiment| <= 0.7) → MEDIUM."""
        result = self.analyzer.classify_impact(
            sentiment_score=-0.5,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == "MEDIUM"

    def test_classify_impact_low_default(self) -> None:
        """Low credibility, low sentiment, no corroboration → LOW."""
        result = self.analyzer.classify_impact(
            sentiment_score=-0.1,
            source_tier=SOURCE_CREDIBILITY_SOCIAL,
            corroboration=0,
        )
        assert result == "LOW"

    def test_classify_impact_positive_high(self) -> None:
        """Positive sentiment with tier-1 source can also be HIGH."""
        result = self.analyzer.classify_impact(
            sentiment_score=0.9,
            source_tier=SOURCE_CREDIBILITY_TIER1,
            corroboration=0,
        )
        assert result == "HIGH"


# =============================================================================
# Crisis Detector Tests
# =============================================================================


class TestCrisisDetector:
    """Tests for CrisisDetector."""

    def setup_method(self) -> None:
        self.detector = CrisisDetector()

    def _make_crisis_article(
        self, sentiment: float = -0.8, offset_seconds: int = 0
    ) -> dict:
        """Helper to create a high-impact negative article."""
        return {
            "id": str(uuid.uuid4()),
            "source": f"source_{uuid.uuid4().hex[:4]}",
            "headline": "Crisis headline",
            "sentiment_score": sentiment,
            "impact_level": "HIGH",
            "received_at": datetime.now(timezone.utc) - timedelta(seconds=offset_seconds),
            "category": "geopolitical_conflict",
        }

    def test_no_crisis_below_threshold(self) -> None:
        """Fewer than 3 articles should not trigger crisis."""
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD - 1):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is None

    def test_crisis_triggered_at_threshold(self) -> None:
        """Exactly 3 high-impact negative articles triggers crisis."""
        result = None
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is not None
        assert result.active is True

    def test_no_crisis_for_low_impact(self) -> None:
        """LOW impact articles don't contribute to crisis."""
        for _ in range(5):
            article = self._make_crisis_article()
            article["impact_level"] = "LOW"
            result = self.detector.evaluate(article)
        assert result is None

    def test_no_crisis_for_mild_sentiment(self) -> None:
        """Articles with sentiment above threshold don't contribute."""
        for _ in range(5):
            result = self.detector.evaluate(
                self._make_crisis_article(sentiment=-0.3)
            )
        assert result is None

    def test_crisis_sentiment_average(self) -> None:
        """Crisis alert contains average sentiment of trigger articles."""
        sentiments = [-0.8, -0.9, -0.75]
        result = None
        for s in sentiments:
            result = self.detector.evaluate(self._make_crisis_article(sentiment=s))
        assert result is not None
        expected_avg = sum(sentiments) / len(sentiments)
        assert abs(result.sentiment_avg - expected_avg) < 0.01

    def test_check_persistence_before_window(self) -> None:
        """Persistence check returns False before 30-minute window."""
        # Trigger crisis
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is not None

        # Check immediately — should not escalate
        persists = self.detector.check_persistence(result.id, [-0.8, -0.9])
        assert persists is False

    def test_check_persistence_with_recovery(self) -> None:
        """Recovery above -0.3 resolves the crisis."""
        # Trigger crisis with backdated start
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is not None

        # Backdate the crisis start to simulate 30+ minutes elapsed
        result.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        # Recovery sentiment above threshold
        persists = self.detector.check_persistence(result.id, [-0.2])
        assert persists is False
        # Crisis should be resolved
        assert result.active is False

    def test_check_persistence_escalates(self) -> None:
        """No recovery after 30 minutes triggers Kill Switch escalation."""
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is not None

        # Backdate crisis start
        result.started_at = datetime.now(timezone.utc) - timedelta(
            minutes=NEWS_CRISIS_PERSISTENCE_MINUTES + 1
        )

        # No recovery — all sentiments still very negative
        persists = self.detector.check_persistence(result.id, [-0.8, -0.9, -0.7])
        assert persists is True
        assert result.escalated_to_kill_switch is True

    def test_resolve_crisis(self) -> None:
        """Resolving a crisis marks it inactive."""
        for _ in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            result = self.detector.evaluate(self._make_crisis_article())
        assert result is not None

        self.detector.resolve_crisis(result.id)
        assert result.active is False
        assert result.resolved_at is not None


# =============================================================================
# Economic Calendar Tests
# =============================================================================


class TestEconomicCalendar:
    """Tests for EconomicCalendar."""

    def setup_method(self) -> None:
        self.calendar = EconomicCalendar()

    def _make_event(
        self,
        minutes_ahead: int = 10,
        impact: str = "HIGH",
        instruments: list[str] | None = None,
    ) -> EconomicEventData:
        """Helper to create an economic event."""
        return EconomicEventData(
            id=str(uuid.uuid4()),
            event_name="NFP Release",
            event_type="employment",
            scheduled_at=datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead),
            impact_level=impact,
            correlated_instruments=instruments or ["EURUSD", "GBPUSD", "US30"],
        )

    def test_get_upcoming_events_within_window(self) -> None:
        """Events within the window are returned."""
        event = self._make_event(minutes_ahead=10)
        self.calendar.add_event(event)

        upcoming = self.calendar.get_upcoming_events(within_minutes=15)
        assert len(upcoming) == 1
        assert upcoming[0].id == event.id

    def test_get_upcoming_events_outside_window(self) -> None:
        """Events outside the window are not returned."""
        event = self._make_event(minutes_ahead=30)
        self.calendar.add_event(event)

        upcoming = self.calendar.get_upcoming_events(within_minutes=15)
        assert len(upcoming) == 0

    def test_get_upcoming_events_past_events_excluded(self) -> None:
        """Past events are not returned."""
        event = self._make_event(minutes_ahead=-5)
        self.calendar.add_event(event)

        upcoming = self.calendar.get_upcoming_events(within_minutes=15)
        assert len(upcoming) == 0

    def test_get_correlated_instruments(self) -> None:
        """Returns instruments correlated with the event."""
        event = self._make_event(instruments=["EURUSD", "GBPUSD"])
        instruments = self.calendar.get_correlated_instruments(event)
        assert instruments == ["EURUSD", "GBPUSD"]

    def test_should_pause_signals_within_5min(self) -> None:
        """Signals paused when high-impact event within 5 minutes."""
        event = self._make_event(minutes_ahead=3, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.should_pause_signals("EURUSD") is True

    def test_should_not_pause_signals_outside_5min(self) -> None:
        """Signals not paused when event is more than 5 minutes away."""
        event = self._make_event(minutes_ahead=10, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.should_pause_signals("EURUSD") is False

    def test_should_not_pause_uncorrelated_instrument(self) -> None:
        """Signals not paused for uncorrelated instruments."""
        event = self._make_event(minutes_ahead=3, instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.should_pause_signals("USDJPY") is False

    def test_should_not_pause_for_low_impact(self) -> None:
        """Low-impact events don't pause signals."""
        event = self._make_event(minutes_ahead=3, impact="LOW", instruments=["EURUSD"])
        self.calendar.add_event(event)

        assert self.calendar.should_pause_signals("EURUSD") is False

    def test_size_reduction_within_15min(self) -> None:
        """Position size reduced by 50% within 15 minutes of high-impact event."""
        event = self._make_event(minutes_ahead=10, instruments=["EURUSD"])
        self.calendar.add_event(event)

        factor = self.calendar.get_size_reduction_factor("EURUSD")
        assert factor == NEWS_EVENT_SIZE_REDUCTION_FACTOR

    def test_size_reduction_outside_15min(self) -> None:
        """No size reduction when event is more than 15 minutes away."""
        event = self._make_event(minutes_ahead=20, instruments=["EURUSD"])
        self.calendar.add_event(event)

        factor = self.calendar.get_size_reduction_factor("EURUSD")
        assert factor == 1.0

    def test_size_reduction_uncorrelated(self) -> None:
        """No size reduction for uncorrelated instruments."""
        event = self._make_event(minutes_ahead=10, instruments=["EURUSD"])
        self.calendar.add_event(event)

        factor = self.calendar.get_size_reduction_factor("USDJPY")
        assert factor == 1.0

    def test_clear_events(self) -> None:
        """Clearing events removes all tracked events."""
        self.calendar.add_event(self._make_event())
        self.calendar.add_event(self._make_event())
        self.calendar.clear_events()
        assert len(self.calendar.events) == 0


# =============================================================================
# Geopolitical Risk Scorer Tests
# =============================================================================


class TestGeopoliticalRiskScorer:
    """Tests for GeopoliticalRiskScorer."""

    def setup_method(self) -> None:
        self.scorer = GeopoliticalRiskScorer()

    def test_get_score_unknown_region(self) -> None:
        """Unknown regions return 0."""
        assert self.scorer.get_risk_score("unknown_region") == 0

    def test_set_and_get_score(self) -> None:
        """Setting a score and retrieving it works correctly."""
        self.scorer.set_score("middle_east", 85.0)
        assert self.scorer.get_risk_score("middle_east") == 85

    def test_set_score_clamped_to_range(self) -> None:
        """Scores are clamped to [0, 100]."""
        self.scorer.set_score("region_a", 150.0)
        assert self.scorer.get_risk_score("region_a") == 100

        self.scorer.set_score("region_b", -10.0)
        assert self.scorer.get_risk_score("region_b") == 0

    def test_get_high_risk_regions(self) -> None:
        """Returns regions with scores >= threshold."""
        self.scorer.set_score("middle_east", 85.0)
        self.scorer.set_score("europe", 40.0)
        self.scorer.set_score("asia_pacific", 72.0)

        high_risk = self.scorer.get_high_risk_regions()
        assert "middle_east" in high_risk
        assert "asia_pacific" in high_risk
        assert "europe" not in high_risk

    def test_get_high_risk_regions_custom_threshold(self) -> None:
        """High risk regions use the default threshold (70)."""
        self.scorer.set_score("region_a", 50.0)
        self.scorer.set_score("region_b", 30.0)

        # Default threshold is 70, so neither should be high risk
        high_risk = self.scorer.get_high_risk_regions()
        assert "region_a" not in high_risk
        assert "region_b" not in high_risk

    def test_update_scores_from_indicators(self) -> None:
        """Updating scores from indicators changes region scores."""
        indicators = {
            "middle_east": [
                {"type": "armed_conflict", "severity": 0.9},
                {"type": "sanctions", "severity": 0.7},
            ],
        }
        self.scorer.update_scores(indicators)
        score = self.scorer.get_risk_score("middle_east")
        assert score > 0

    def test_reset_region(self) -> None:
        """Resetting a region removes its score."""
        self.scorer.set_score("region_a", 80.0)
        self.scorer.reset_region("region_a")
        assert self.scorer.get_risk_score("region_a") == 0


# =============================================================================
# Correlation Mapper Tests
# =============================================================================


class TestCorrelationMapper:
    """Tests for CorrelationMapper."""

    def setup_method(self) -> None:
        self.mapper = CorrelationMapper()

    def test_categories_defined(self) -> None:
        """All expected categories are defined."""
        expected = [
            "monetary_policy",
            "geopolitical_conflict",
            "natural_disaster",
            "earnings",
            "commodity_supply",
        ]
        assert self.mapper.categories == expected

    def test_get_affected_instruments_monetary_policy(self) -> None:
        """Monetary policy category returns relevant instruments."""
        instruments = self.mapper.get_affected_instruments("monetary_policy")
        assert len(instruments) > 0
        assert "EURUSD" in instruments

    def test_get_affected_instruments_unknown_category(self) -> None:
        """Unknown category returns empty list."""
        instruments = self.mapper.get_affected_instruments("unknown_category")
        assert instruments == []

    def test_set_correlations(self) -> None:
        """Setting correlations updates the mapping."""
        self.mapper.set_correlations("earnings", ["AAPL", "MSFT", "GOOGL"])
        instruments = self.mapper.get_affected_instruments("earnings")
        assert instruments == ["AAPL", "MSFT", "GOOGL"]

    @pytest.mark.asyncio
    async def test_update_weekly(self) -> None:
        """Weekly update adjusts correlations based on historical data."""
        reactions = [
            {"category": "earnings", "instrument": "TSLA", "reaction_magnitude": 5.0},
            {"category": "earnings", "instrument": "AAPL", "reaction_magnitude": 3.0},
            {"category": "earnings", "instrument": "MSFT", "reaction_magnitude": 2.0},
        ]
        await self.mapper.update_weekly(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "TSLA" in instruments
        assert instruments[0] == "TSLA"  # Highest magnitude first


class TestCorrelationMapperWeeklyUpdate:
    """Tests for CorrelationMapper weekly update from historical price data."""

    def setup_method(self) -> None:
        self.mapper = CorrelationMapper()

    def test_update_from_historical_data_adds_instrument_above_threshold(self) -> None:
        """Instrument reacting >60% of events is added to mapping."""
        # NEW_INST reacts in 7 out of 10 events (70% > 60% threshold)
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 7 else 0.01  # 7 reactions, 3 non-reactions
            reactions.append({
                "category": "earnings",
                "instrument": "NEW_INST",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        result = self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "NEW_INST" in instruments
        assert "NEW_INST" in result["added"].get("earnings", [])

    def test_update_from_historical_data_does_not_add_below_threshold(self) -> None:
        """Instrument reacting <=60% of events is NOT added."""
        # NEW_INST reacts in 5 out of 10 events (50% <= 60% threshold)
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 5 else 0.01
            reactions.append({
                "category": "earnings",
                "instrument": "BELOW_THRESH",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "BELOW_THRESH" not in instruments

    def test_update_from_historical_data_removes_non_reacting_instrument(self) -> None:
        """Non-default instrument reacting <30% is removed from mapping."""
        # First add a non-default instrument
        self.mapper._correlations["earnings"].append("TEMP_INST")
        assert "TEMP_INST" in self.mapper.get_affected_instruments("earnings")

        # TEMP_INST reacts in 2 out of 10 events (20% < 30% threshold)
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 2 else 0.01
            reactions.append({
                "category": "earnings",
                "instrument": "TEMP_INST",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        result = self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "TEMP_INST" not in instruments
        assert "TEMP_INST" in result["removed"].get("earnings", [])

    def test_update_from_historical_data_preserves_default_instruments(self) -> None:
        """Default/core instruments are never removed even with low reaction rate."""
        # SPX500 is a default for "earnings" - give it 0% reaction rate
        reactions = []
        for _ in range(10):
            reactions.append({
                "category": "earnings",
                "instrument": "SPX500",
                "price_change_pct": 0.01,  # Below MIN_REACTION_PCT
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "SPX500" in instruments  # Still present (core mapping)

    def test_update_from_historical_data_empty_reactions(self) -> None:
        """Empty reaction list does not change mappings."""
        original = self.mapper.get_affected_instruments("earnings")
        result = self.mapper.update_from_historical_data([])
        after = self.mapper.get_affected_instruments("earnings")
        assert set(original) == set(after)
        assert result["added"] == {}
        assert result["removed"] == {}

    def test_update_from_historical_data_ignores_unknown_category(self) -> None:
        """Reactions with unknown categories are ignored."""
        reactions = [
            {
                "category": "unknown_category",
                "instrument": "SOME_INST",
                "price_change_pct": 5.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        result = self.mapper.update_from_historical_data(reactions)
        assert result["added"] == {}
        assert result["removed"] == {}

    def test_update_from_historical_data_sets_last_update_timestamp(self) -> None:
        """Update sets the last_weekly_update timestamp."""
        assert self.mapper.last_weekly_update is None
        self.mapper.update_from_historical_data([])
        assert self.mapper.last_weekly_update is not None

    def test_should_run_weekly_update_initially_true(self) -> None:
        """Should run weekly update when no update has been performed."""
        assert self.mapper.should_run_weekly_update() is True

    def test_should_run_weekly_update_false_after_recent_update(self) -> None:
        """Should not run weekly update if last update was recent."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc)
        assert self.mapper.should_run_weekly_update() is False

    def test_should_run_weekly_update_true_after_7_days(self) -> None:
        """Should run weekly update if 7+ days have passed."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc) - timedelta(days=8)
        assert self.mapper.should_run_weekly_update() is True

    def test_get_default_instruments(self) -> None:
        """get_default_instruments returns the core mapping."""
        defaults = self.mapper.get_default_instruments("monetary_policy")
        assert "EURUSD" in defaults
        assert "GOLD" in defaults

    def test_get_default_instruments_unknown_category(self) -> None:
        """get_default_instruments returns empty for unknown category."""
        assert self.mapper.get_default_instruments("unknown") == []

    def test_update_result_contains_timestamp(self) -> None:
        """Update result includes an ISO timestamp."""
        result = self.mapper.update_from_historical_data([])
        assert "timestamp" in result
        # Should be parseable as ISO format
        datetime.fromisoformat(result["timestamp"])


# =============================================================================
# News Source Tests
# =============================================================================


class TestNewsSources:
    """Tests for news source adapters."""

    @pytest.mark.asyncio
    async def test_reuters_source_connect(self) -> None:
        """Reuters source connects successfully."""
        source = ReutersSource()
        await source.connect()
        assert source.is_connected is True
        assert source.tier == SOURCE_CREDIBILITY_TIER1
        await source.disconnect()
        assert source.is_connected is False

    @pytest.mark.asyncio
    async def test_bloomberg_source_connect(self) -> None:
        """Bloomberg source connects successfully."""
        source = BloombergSource()
        await source.connect()
        assert source.is_connected is True
        assert source.tier == SOURCE_CREDIBILITY_TIER1
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_social_media_source_connect(self) -> None:
        """Social media source connects with correct tier."""
        source = SocialMediaSource(platform="twitter")
        await source.connect()
        assert source.is_connected is True
        assert source.tier == SOURCE_CREDIBILITY_SOCIAL
        assert source.platform == "twitter"
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_source_health_check(self) -> None:
        """Health check returns True when connected."""
        source = ReutersSource()
        await source.connect()
        assert await source.health_check() is True
        await source.disconnect()
        assert await source.health_check() is False

    @pytest.mark.asyncio
    async def test_source_callback_registration(self) -> None:
        """Article callbacks are invoked when articles arrive."""
        source = ReutersSource()
        received_articles: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received_articles.append(article)

        source.on_article_received(callback)
        await source.connect()

        # Simulate article reception
        await source._process_message("Test headline", "Test body", "earnings")

        assert len(received_articles) == 1
        assert received_articles[0].headline == "Test headline"
        assert received_articles[0].source_name == "Reuters"
        assert received_articles[0].source_tier == SOURCE_CREDIBILITY_TIER1


# =============================================================================
# News Engine Integration Tests
# =============================================================================


class TestNewsEngine:
    """Tests for the main NewsEngine."""

    @pytest.mark.asyncio
    async def test_engine_start_stop(self) -> None:
        """Engine starts and stops cleanly."""
        sources = [ReutersSource(), BloombergSource(), SocialMediaSource()]
        engine = NewsEngine(sources=sources)

        await engine.start()
        assert engine.is_running is True
        assert len(engine.healthy_sources) == 3

        await engine.stop()
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_engine_all_sources_down(self) -> None:
        """Engine detects when all sources are down."""
        sources = [ReutersSource(), BloombergSource(), SocialMediaSource()]
        engine = NewsEngine(sources=sources)

        # Before start, no sources are healthy
        assert engine.is_all_sources_down() is True

        await engine.start()
        assert engine.is_all_sources_down() is False
        await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_article_processing(self) -> None:
        """Engine processes articles through the full pipeline."""
        sources = [ReutersSource(), BloombergSource(), SocialMediaSource()]
        engine = NewsEngine(sources=sources)

        article = {
            "id": str(uuid.uuid4()),
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Market crash leads to massive sell-off",
            "body": "Global markets experienced a severe crash today.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

        await engine.on_article_received(article)

        # Article should have been enriched with sentiment and impact
        assert "sentiment_score" in article
        assert "impact_level" in article
        assert "body_hash" in article
        assert article["sentiment_score"] < 0.0


# =============================================================================
# All-Sources-Down Degraded Mode Tests (Requirement 23.18)
# =============================================================================


class MockNewsSource(NewsSource):
    """Mock news source for testing degraded mode without external dependencies."""

    def __init__(
        self, name: str = "MockSource", tier: float = SOURCE_CREDIBILITY_TIER1, healthy: bool = True
    ) -> None:
        super().__init__(name=name, tier=tier)
        self._mock_healthy = healthy

    async def connect(self) -> None:
        """Simulate connection."""
        if self._mock_healthy:
            self._connected = True
        else:
            raise ConnectionError(f"Mock source {self._name} unavailable")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False

    async def subscribe(self, topics: list[str]) -> None:
        """Simulate subscription."""
        self._subscribed_topics.extend(topics)

    async def health_check(self) -> bool:
        """Return mock health status."""
        return self._mock_healthy

    def set_healthy(self, healthy: bool) -> None:
        """Set mock health status for testing."""
        self._mock_healthy = healthy
        if healthy:
            self._connected = True
        else:
            self._connected = False


class MockEventBus:
    """Mock event bus that records published events for testing."""

    def __init__(self) -> None:
        self.published_events: list[tuple[str, Any]] = []

    async def publish(self, channel: str, event: Any) -> int:
        """Record the published event."""
        self.published_events.append((channel, event))
        return 1

    def get_events_by_channel(self, channel: str) -> list[Any]:
        """Get all events published to a specific channel."""
        return [event for ch, event in self.published_events if ch == channel]


class TestDegradedMode:
    """Tests for all-sources-down degraded mode handling (Requirement 23.18)."""

    @pytest.mark.asyncio
    async def test_degraded_mode_initial_state(self) -> None:
        """Engine is not in degraded mode before start."""
        sources = [MockNewsSource("Source1"), MockNewsSource("Source2")]
        engine = NewsEngine(sources=sources)

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

    @pytest.mark.asyncio
    async def test_degraded_mode_when_all_sources_fail_on_start(self) -> None:
        """Engine enters degraded mode when all sources fail to connect."""
        sources = [
            MockNewsSource("Source1", healthy=False),
            MockNewsSource("Source2", healthy=False),
            MockNewsSource("Source3", healthy=False),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()

        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == 80
        assert engine.is_all_sources_down() is True

        await engine.stop()

    @pytest.mark.asyncio
    async def test_not_degraded_when_sources_healthy(self) -> None:
        """Engine is not in degraded mode when sources are healthy."""
        sources = [
            MockNewsSource("Source1", healthy=True),
            MockNewsSource("Source2", healthy=True),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60
        assert engine.is_all_sources_down() is False

        await engine.stop()

    @pytest.mark.asyncio
    async def test_not_degraded_when_at_least_one_source_healthy(self) -> None:
        """Engine is not degraded if at least one source is healthy."""
        sources = [
            MockNewsSource("Source1", healthy=True),
            MockNewsSource("Source2", healthy=False),
            MockNewsSource("Source3", healthy=False),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

        await engine.stop()

    @pytest.mark.asyncio
    async def test_confidence_threshold_elevated_in_degraded_mode(self) -> None:
        """Confidence threshold is 80 when in degraded mode."""
        from src.config.constants import CONFIDENCE_THRESHOLD_NEWS_DOWN

        sources = [MockNewsSource("Source1", healthy=False)]
        engine = NewsEngine(sources=sources)

        await engine.start()

        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == CONFIDENCE_THRESHOLD_NEWS_DOWN
        assert engine.get_confidence_threshold() == 80

        await engine.stop()

    @pytest.mark.asyncio
    async def test_confidence_threshold_restored_when_source_recovers(self) -> None:
        """Confidence threshold returns to 60 when a source is restored."""
        from src.config.constants import CONFIDENCE_THRESHOLD_DEFAULT

        sources = [
            MockNewsSource("Source1", healthy=False),
            MockNewsSource("Source2", healthy=False),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == 80

        # Simulate source recovery
        sources[0].set_healthy(True)
        await engine._perform_health_checks()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == CONFIDENCE_THRESHOLD_DEFAULT
        assert engine.get_confidence_threshold() == 60

        await engine.stop()

    @pytest.mark.asyncio
    async def test_degraded_mode_transition_normal_to_degraded(self) -> None:
        """Engine transitions from normal to degraded when all sources go down."""
        sources = [
            MockNewsSource("Source1", healthy=True),
            MockNewsSource("Source2", healthy=True),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        assert engine.degraded_mode is False

        # All sources go down
        sources[0].set_healthy(False)
        sources[1].set_healthy(False)
        await engine._perform_health_checks()

        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == 80

        await engine.stop()

    @pytest.mark.asyncio
    async def test_degraded_mode_transition_degraded_to_normal(self) -> None:
        """Engine transitions from degraded to normal when a source is restored."""
        sources = [
            MockNewsSource("Source1", healthy=False),
            MockNewsSource("Source2", healthy=False),
        ]
        engine = NewsEngine(sources=sources)

        await engine.start()
        assert engine.degraded_mode is True

        # One source comes back
        sources[1].set_healthy(True)
        await engine._perform_health_checks()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

        await engine.stop()

    @pytest.mark.asyncio
    async def test_publishes_all_sources_down_event(self) -> None:
        """Engine publishes NEWS_ALL_SOURCES_DOWN event when entering degraded mode."""
        from src.core.event_bus import NEWS_ALL_SOURCES_DOWN

        sources = [
            MockNewsSource("Source1", healthy=True),
            MockNewsSource("Source2", healthy=True),
        ]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        assert engine.degraded_mode is False

        # All sources go down
        sources[0].set_healthy(False)
        sources[1].set_healthy(False)
        await engine._perform_health_checks()

        # Check event was published
        down_events = mock_bus.get_events_by_channel(NEWS_ALL_SOURCES_DOWN)
        assert len(down_events) == 1
        assert down_events[0].event_type == NEWS_ALL_SOURCES_DOWN
        assert down_events[0].payload["confidence_threshold"] == 80

        await engine.stop()

    @pytest.mark.asyncio
    async def test_publishes_sources_restored_event(self) -> None:
        """Engine publishes NEWS_SOURCES_RESTORED event when exiting degraded mode."""
        from src.core.event_bus import NEWS_SOURCES_RESTORED

        sources = [
            MockNewsSource("Source1", healthy=False),
            MockNewsSource("Source2", healthy=False),
        ]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        assert engine.degraded_mode is True

        # One source comes back
        sources[0].set_healthy(True)
        await engine._perform_health_checks()

        # Check event was published
        restored_events = mock_bus.get_events_by_channel(NEWS_SOURCES_RESTORED)
        assert len(restored_events) == 1
        assert restored_events[0].event_type == NEWS_SOURCES_RESTORED
        assert restored_events[0].payload["confidence_threshold"] == 60
        assert "Source1" in restored_events[0].payload["restored_sources"]

        await engine.stop()

    @pytest.mark.asyncio
    async def test_no_duplicate_degraded_events(self) -> None:
        """Engine does not publish duplicate events when already in degraded mode."""
        from src.core.event_bus import NEWS_ALL_SOURCES_DOWN

        sources = [MockNewsSource("Source1", healthy=False)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        # First transition to degraded
        assert engine.degraded_mode is True

        # Run health checks again — should not publish another event
        await engine._perform_health_checks()
        await engine._perform_health_checks()

        down_events = mock_bus.get_events_by_channel(NEWS_ALL_SOURCES_DOWN)
        assert len(down_events) == 1  # Only one event from initial start

        await engine.stop()

    @pytest.mark.asyncio
    async def test_no_duplicate_restored_events(self) -> None:
        """Engine does not publish duplicate restored events when already normal."""
        from src.core.event_bus import NEWS_SOURCES_RESTORED

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        assert engine.degraded_mode is False

        # Run health checks — should not publish restored event
        await engine._perform_health_checks()
        await engine._perform_health_checks()

        restored_events = mock_bus.get_events_by_channel(NEWS_SOURCES_RESTORED)
        assert len(restored_events) == 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_degraded_mode_full_cycle(self) -> None:
        """Full cycle: normal → degraded → normal with correct events."""
        from src.core.event_bus import NEWS_ALL_SOURCES_DOWN, NEWS_SOURCES_RESTORED

        sources = [
            MockNewsSource("Source1", healthy=True),
            MockNewsSource("Source2", healthy=True),
        ]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

        # All sources go down
        sources[0].set_healthy(False)
        sources[1].set_healthy(False)
        await engine._perform_health_checks()

        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == 80

        # One source recovers
        sources[0].set_healthy(True)
        await engine._perform_health_checks()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

        # Verify events
        down_events = mock_bus.get_events_by_channel(NEWS_ALL_SOURCES_DOWN)
        restored_events = mock_bus.get_events_by_channel(NEWS_SOURCES_RESTORED)
        assert len(down_events) == 1
        assert len(restored_events) == 1

        await engine.stop()

    @pytest.mark.asyncio
    async def test_degraded_mode_without_event_bus(self) -> None:
        """Degraded mode works correctly even without an event bus."""
        sources = [MockNewsSource("Source1", healthy=False)]
        engine = NewsEngine(sources=sources, event_bus=None)

        await engine.start()

        assert engine.degraded_mode is True
        assert engine.get_confidence_threshold() == 80

        # Source recovers
        sources[0].set_healthy(True)
        await engine._perform_health_checks()

        assert engine.degraded_mode is False
        assert engine.get_confidence_threshold() == 60

        await engine.stop()

    @pytest.mark.asyncio
    async def test_restored_event_includes_duration(self) -> None:
        """Restored event includes how long the system was in degraded mode."""
        from src.core.event_bus import NEWS_SOURCES_RESTORED

        sources = [MockNewsSource("Source1", healthy=False)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        await engine.start()
        assert engine.degraded_mode is True

        # Source recovers
        sources[0].set_healthy(True)
        await engine._perform_health_checks()

        restored_events = mock_bus.get_events_by_channel(NEWS_SOURCES_RESTORED)
        assert len(restored_events) == 1
        assert restored_events[0].payload["degraded_duration_seconds"] is not None
        assert restored_events[0].payload["degraded_duration_seconds"] >= 0

        await engine.stop()


# =============================================================================
# High-Impact News Notification Tests (Requirement 23.11)
# =============================================================================


class TestHighImpactNotification:
    """Tests for high-impact news notification to Strategy_Engine.

    Validates Requirement 23.11: When a news event is classified as High impact,
    the News_Engine SHALL identify all correlated instruments and notify the
    Strategy_Engine with the affected instrument list, sentiment score, and
    impact classification within 5 seconds.
    """

    @pytest.mark.asyncio
    async def test_high_impact_article_publishes_notification(self) -> None:
        """HIGH impact article triggers a dedicated notification event."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-001",
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Major market crash leads to global crisis collapse",
            "body": "Markets experienced a severe crash and crisis today.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

        await engine.on_article_received(article)

        # Verify high-impact notification was published
        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 1

        event = high_impact_events[0]
        assert event.event_type == NEWS_HIGH_IMPACT
        assert event.payload["impact_level"] == "HIGH"
        assert event.payload["article_id"] == "test-article-001"

    @pytest.mark.asyncio
    async def test_high_impact_notification_includes_affected_instruments(self) -> None:
        """Notification payload includes correlated instruments from mapping."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-002",
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Severe geopolitical crisis and war escalation collapse",
            "body": "Armed conflict escalation leads to market crash.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

        await engine.on_article_received(article)

        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 1

        payload = high_impact_events[0].payload
        affected = payload["affected_instruments"]
        # geopolitical_conflict maps to GOLD, XAUUSD, OIL, USDJPY, USDCHF, VIX, US30
        assert len(affected) > 0
        assert "GOLD" in affected

    @pytest.mark.asyncio
    async def test_high_impact_notification_includes_sentiment_score(self) -> None:
        """Notification payload includes the sentiment score."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-003",
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Market crash and crisis lead to massive collapse",
            "body": "Global markets crash in severe crisis.",
            "received_at": datetime.now(timezone.utc),
            "category": "monetary_policy",
        }

        await engine.on_article_received(article)

        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 1

        payload = high_impact_events[0].payload
        assert "sentiment_score" in payload
        assert isinstance(payload["sentiment_score"], float)
        assert -1.0 <= payload["sentiment_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_low_impact_article_does_not_publish_notification(self) -> None:
        """LOW impact articles do NOT trigger high-impact notification."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-004",
            "source": "SocialMedia",
            "source_tier": SOURCE_CREDIBILITY_SOCIAL,
            "headline": "Minor market update today",
            "body": "Nothing significant happened.",
            "received_at": datetime.now(timezone.utc),
            "category": "earnings",
        }

        await engine.on_article_received(article)

        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 0

    @pytest.mark.asyncio
    async def test_medium_impact_article_does_not_publish_notification(self) -> None:
        """MEDIUM impact articles do NOT trigger high-impact notification."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        # Tier-2 source with moderate sentiment → MEDIUM impact
        article = {
            "id": "test-article-005",
            "source": "Tier2Source",
            "source_tier": SOURCE_CREDIBILITY_TIER2,
            "headline": "Market decline raises concerns",
            "body": "Moderate decline in markets today.",
            "received_at": datetime.now(timezone.utc),
            "category": "monetary_policy",
        }

        await engine.on_article_received(article)

        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 0

    @pytest.mark.asyncio
    async def test_high_impact_notification_without_event_bus(self) -> None:
        """No error when event bus is None and article is HIGH impact."""
        sources = [MockNewsSource("Source1", healthy=True)]
        engine = NewsEngine(sources=sources, event_bus=None)

        article = {
            "id": "test-article-006",
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Major crash and crisis collapse in markets",
            "body": "Severe market crash today.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

        # Should not raise any exception
        await engine.on_article_received(article)
        assert engine.articles_processed == 1

    @pytest.mark.asyncio
    async def test_high_impact_notification_payload_structure(self) -> None:
        """Notification payload has all required fields per Requirement 23.11."""
        from src.core.event_bus import NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-007",
            "source": "Bloomberg",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Massive crash and crisis lead to collapse",
            "body": "Global financial crisis escalates.",
            "received_at": datetime.now(timezone.utc),
            "category": "commodity_supply",
        }

        await engine.on_article_received(article)

        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        assert len(high_impact_events) == 1

        payload = high_impact_events[0].payload
        # Required fields per Requirement 23.11
        assert "affected_instruments" in payload
        assert "sentiment_score" in payload
        assert "impact_level" in payload
        assert payload["impact_level"] == "HIGH"
        assert "article_id" in payload
        assert "source" in payload
        assert "headline" in payload
        assert "category" in payload

    @pytest.mark.asyncio
    async def test_high_impact_also_publishes_general_article_event(self) -> None:
        """HIGH impact articles still publish the general article event too."""
        from src.core.event_bus import NEWS_ARTICLE_RECEIVED, NEWS_HIGH_IMPACT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        article = {
            "id": "test-article-008",
            "source": "Reuters",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Major crash and crisis collapse globally",
            "body": "Markets crash in severe crisis.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

        await engine.on_article_received(article)

        # Both events should be published
        high_impact_events = mock_bus.get_events_by_channel(NEWS_HIGH_IMPACT)
        article_events = mock_bus.get_events_by_channel(NEWS_ARTICLE_RECEIVED)
        assert len(high_impact_events) == 1
        assert len(article_events) == 1


# =============================================================================
# Crisis Alert Emission Tests (Requirement 23.7)
# =============================================================================


class TestCrisisAlertEmission:
    """Tests for crisis alert emission to Risk_Engine via Event Bus.

    Validates Requirement 23.7: When the News_Engine detects a crisis event
    (3+ High-impact articles with sentiment < -0.7 within 10-minute window),
    it SHALL emit a crisis alert to the Risk_Engine within 10 seconds of detection.
    """

    def _make_crisis_article(self, index: int = 0) -> dict:
        """Create a high-impact negative article that contributes to crisis detection."""
        return {
            "id": f"crisis-article-{index}",
            "source": f"source_{index}",
            "source_tier": SOURCE_CREDIBILITY_TIER1,
            "headline": "Major crash and crisis collapse in global markets",
            "body": f"Severe market crash and crisis collapse event {index}.",
            "received_at": datetime.now(timezone.utc),
            "category": "geopolitical_conflict",
        }

    @pytest.mark.asyncio
    async def test_crisis_alert_published_when_threshold_met(self) -> None:
        """NEWS_CRISIS_ALERT event is published when 3+ crisis articles are detected."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        # Feed enough high-impact negative articles to trigger crisis
        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        # Verify crisis alert was published
        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        assert len(crisis_events) == 1

    @pytest.mark.asyncio
    async def test_crisis_alert_payload_contains_required_fields(self) -> None:
        """Crisis alert payload includes crisis_id, region, sentiment_avg, article_count, started_at."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        assert len(crisis_events) == 1

        payload = crisis_events[0].payload
        assert "crisis_id" in payload
        assert "region" in payload
        assert "sentiment_avg" in payload
        assert "article_count" in payload
        assert "started_at" in payload

    @pytest.mark.asyncio
    async def test_crisis_alert_event_type_is_correct(self) -> None:
        """Published event has event_type set to NEWS_CRISIS_ALERT."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        assert len(crisis_events) == 1
        assert crisis_events[0].event_type == NEWS_CRISIS_ALERT

    @pytest.mark.asyncio
    async def test_crisis_alert_article_count_matches_threshold(self) -> None:
        """article_count in payload matches the number of trigger articles."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        payload = crisis_events[0].payload
        assert payload["article_count"] == NEWS_CRISIS_ARTICLE_THRESHOLD

    @pytest.mark.asyncio
    async def test_crisis_alert_sentiment_avg_is_negative(self) -> None:
        """sentiment_avg in payload reflects the average of trigger articles."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        payload = crisis_events[0].payload
        # All articles have very negative sentiment, so average should be < -0.7
        assert payload["sentiment_avg"] < NEWS_CRISIS_SENTIMENT_THRESHOLD

    @pytest.mark.asyncio
    async def test_crisis_alert_started_at_is_iso_format(self) -> None:
        """started_at in payload is a valid ISO format timestamp."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        payload = crisis_events[0].payload
        # Should be parseable as ISO datetime
        started_at = datetime.fromisoformat(payload["started_at"])
        assert started_at is not None

    @pytest.mark.asyncio
    async def test_no_crisis_alert_below_threshold(self) -> None:
        """No NEWS_CRISIS_ALERT is published when fewer than 3 articles are received."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        # Feed fewer articles than the threshold
        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD - 1):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        assert len(crisis_events) == 0

    @pytest.mark.asyncio
    async def test_no_crisis_alert_without_event_bus(self) -> None:
        """No error when event bus is None and crisis is detected."""
        sources = [MockNewsSource("Source1", healthy=True)]
        engine = NewsEngine(sources=sources, event_bus=None)

        # Should not raise any exception
        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        # Engine still processes articles successfully
        assert engine.articles_processed == NEWS_CRISIS_ARTICLE_THRESHOLD

    @pytest.mark.asyncio
    async def test_crisis_alert_published_on_news_crisis_alert_channel(self) -> None:
        """Crisis alert is published specifically to the NEWS_CRISIS_ALERT channel."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        # Verify the channel used is NEWS_CRISIS_ALERT = "news.crisis_alert"
        crisis_channels = [ch for ch, _ in mock_bus.published_events if ch == NEWS_CRISIS_ALERT]
        assert len(crisis_channels) >= 1

    @pytest.mark.asyncio
    async def test_crisis_alert_has_valid_crisis_id(self) -> None:
        """crisis_id in payload is a non-empty string (UUID)."""
        from src.core.event_bus import NEWS_CRISIS_ALERT

        sources = [MockNewsSource("Source1", healthy=True)]
        mock_bus = MockEventBus()
        engine = NewsEngine(sources=sources, event_bus=mock_bus)

        for i in range(NEWS_CRISIS_ARTICLE_THRESHOLD):
            article = self._make_crisis_article(index=i)
            await engine.on_article_received(article)

        crisis_events = mock_bus.get_events_by_channel(NEWS_CRISIS_ALERT)
        payload = crisis_events[0].payload
        assert isinstance(payload["crisis_id"], str)
        assert len(payload["crisis_id"]) > 0
        # Should be a valid UUID
        uuid.UUID(payload["crisis_id"])
