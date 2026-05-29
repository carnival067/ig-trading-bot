"""Unit tests for NewsRepository async CRUD operations.

Uses an in-memory SQLite database with aiosqlite for fast, isolated testing.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.database import Base
from src.db.models import CrisisAlert, EconomicEvent, GeopoliticalRiskScore, NewsArticle
from src.db.repositories.news_repo import NewsRepository


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def repo(async_session: AsyncSession) -> NewsRepository:
    """Create a NewsRepository instance with the test session."""
    return NewsRepository(async_session)


def _make_article_data(
    *,
    impact_level: str = "HIGH",
    category: str = "monetary_policy",
    sentiment_score: float = -0.8,
    received_at: datetime | None = None,
) -> dict:
    """Helper to create article data dictionaries."""
    return {
        "source": "reuters",
        "headline": "Test headline",
        "body_hash": uuid.uuid4().hex[:64],
        "sentiment_score": sentiment_score,
        "impact_level": impact_level,
        "category": category,
        "correlated_instruments_json": ["EUR/USD", "GBP/USD"],
        "received_at": received_at or datetime.now(timezone.utc),
        "published_at": datetime.now(timezone.utc),
    }


# ─── News Articles Tests ──────────────────────────────────────────────────────


class TestStoreArticle:
    async def test_store_article_returns_news_article(self, repo: NewsRepository) -> None:
        data = _make_article_data()
        result = await repo.store_article(data)

        assert isinstance(result, NewsArticle)
        assert result.source == "reuters"
        assert result.headline == "Test headline"
        assert result.impact_level == "HIGH"
        assert result.id is not None

    async def test_store_article_persists_all_fields(self, repo: NewsRepository) -> None:
        data = _make_article_data(sentiment_score=-0.5, category="geopolitical_conflict")
        result = await repo.store_article(data)

        assert result.sentiment_score == -0.5
        assert result.category == "geopolitical_conflict"


class TestGetRecentArticles:
    async def test_returns_empty_list_when_no_articles(self, repo: NewsRepository) -> None:
        result = await repo.get_recent_articles()
        assert result == []

    async def test_returns_articles_ordered_by_received_at_desc(
        self, repo: NewsRepository
    ) -> None:
        now = datetime.now(timezone.utc)
        for i in range(3):
            data = _make_article_data(received_at=now - timedelta(minutes=i))
            await repo.store_article(data)

        result = await repo.get_recent_articles()
        assert len(result) == 3
        # Most recent first
        assert result[0].received_at >= result[1].received_at
        assert result[1].received_at >= result[2].received_at

    async def test_respects_limit_parameter(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        for i in range(5):
            data = _make_article_data(received_at=now - timedelta(minutes=i))
            await repo.store_article(data)

        result = await repo.get_recent_articles(limit=3)
        assert len(result) == 3


class TestGetArticlesByImpact:
    async def test_filters_by_impact_level(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(_make_article_data(impact_level="HIGH", received_at=now))
        await repo.store_article(_make_article_data(impact_level="LOW", received_at=now))

        result = await repo.get_articles_by_impact("HIGH", now - timedelta(minutes=1))
        assert len(result) == 1
        assert result[0].impact_level == "HIGH"

    async def test_filters_by_time_window(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(
            _make_article_data(impact_level="HIGH", received_at=now - timedelta(hours=2))
        )
        await repo.store_article(
            _make_article_data(impact_level="HIGH", received_at=now)
        )

        result = await repo.get_articles_by_impact("HIGH", now - timedelta(hours=1))
        assert len(result) == 1


class TestArticleExists:
    async def test_returns_true_for_existing_hash(self, repo: NewsRepository) -> None:
        data = _make_article_data()
        body_hash = data["body_hash"]
        await repo.store_article(data)

        result = await repo.article_exists(body_hash)
        assert result is True

    async def test_returns_false_for_nonexistent_hash(self, repo: NewsRepository) -> None:
        result = await repo.article_exists("nonexistent_hash_value")
        assert result is False

    async def test_detects_duplicate_articles(self, repo: NewsRepository) -> None:
        data = _make_article_data()
        body_hash = data["body_hash"]
        await repo.store_article(data)

        # Before storing a second article, check if it already exists
        assert await repo.article_exists(body_hash) is True
        # A different hash should not exist
        assert await repo.article_exists("different_hash") is False


class TestGetArticlesByCategory:
    async def test_filters_by_category(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(
            _make_article_data(category="monetary_policy", received_at=now)
        )
        await repo.store_article(
            _make_article_data(category="natural_disaster", received_at=now)
        )

        result = await repo.get_articles_by_category("monetary_policy", now - timedelta(minutes=1))
        assert len(result) == 1
        assert result[0].category == "monetary_policy"

    async def test_filters_by_time_window(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(
            _make_article_data(category="monetary_policy", received_at=now - timedelta(hours=2))
        )
        await repo.store_article(
            _make_article_data(category="monetary_policy", received_at=now)
        )

        result = await repo.get_articles_by_category("monetary_policy", now - timedelta(hours=1))
        assert len(result) == 1

    async def test_backward_compat_alias(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(
            _make_article_data(category="monetary_policy", received_at=now)
        )

        result = await repo.get_articles_by_region("monetary_policy", now - timedelta(minutes=1))
        assert len(result) == 1
        assert result[0].category == "monetary_policy"


class TestGetHighImpactArticlesInWindow:
    async def test_returns_high_impact_articles_within_window(
        self, repo: NewsRepository
    ) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.8, received_at=now - timedelta(minutes=5))
        )
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.9, received_at=now - timedelta(minutes=15))
        )
        await repo.store_article(
            _make_article_data(impact_level="LOW", sentiment_score=-0.8, received_at=now - timedelta(minutes=3))
        )

        result = await repo.get_high_impact_articles_in_window(minutes=10)
        assert len(result) == 1
        assert result[0].impact_level == "HIGH"

    async def test_filters_by_sentiment_threshold(
        self, repo: NewsRepository
    ) -> None:
        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=10)
        # Below threshold (should be included)
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.8, received_at=now - timedelta(minutes=2))
        )
        # Above threshold (should be excluded)
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.3, received_at=now - timedelta(minutes=3))
        )

        result = await repo.get_high_impact_articles_in_window(since=since, sentiment_threshold=-0.7)
        assert len(result) == 1
        assert result[0].sentiment_score == -0.8

    async def test_accepts_since_parameter(
        self, repo: NewsRepository
    ) -> None:
        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=5)
        # Within window
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.9, received_at=now - timedelta(minutes=3))
        )
        # Outside window
        await repo.store_article(
            _make_article_data(impact_level="HIGH", sentiment_score=-0.9, received_at=now - timedelta(minutes=10))
        )

        result = await repo.get_high_impact_articles_in_window(since=since)
        assert len(result) == 1


# ─── Crisis Alerts Tests ──────────────────────────────────────────────────────


class TestCreateCrisisAlert:
    async def test_creates_crisis_alert(self, repo: NewsRepository) -> None:
        data = {
            "region": "Middle East",
            "trigger_articles_json": ["article-1", "article-2"],
            "sentiment_avg": -0.85,
            "started_at": datetime.now(timezone.utc),
        }
        result = await repo.create_crisis_alert(data)

        assert isinstance(result, CrisisAlert)
        assert result.region == "Middle East"
        assert result.sentiment_avg == -0.85
        assert result.resolved_at is None
        assert result.escalated_to_kill_switch is False


class TestGetActiveCrises:
    async def test_returns_only_unresolved_crises(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        # Active crisis
        await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.9,
            "started_at": now,
        })
        # Resolved crisis
        resolved = await repo.create_crisis_alert({
            "region": "Asia Pacific",
            "trigger_articles_json": [],
            "sentiment_avg": -0.75,
            "started_at": now - timedelta(hours=2),
        })
        await repo.resolve_crisis(str(resolved.id))

        result = await repo.get_active_crises()
        assert len(result) == 1
        assert result[0].region == "Eastern Europe"


class TestResolveCrisis:
    async def test_marks_crisis_as_resolved(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Middle East",
            "trigger_articles_json": [],
            "sentiment_avg": -0.8,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.resolve_crisis(str(alert.id))
        assert result is not None
        assert result.resolved_at is not None
        assert result.escalated_to_kill_switch is False

    async def test_resolve_with_escalation(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.95,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.resolve_crisis(str(alert.id), escalated=True)
        assert result is not None
        assert result.resolved_at is not None
        assert result.escalated_to_kill_switch is True

    async def test_accepts_uuid_directly(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Middle East",
            "trigger_articles_json": [],
            "sentiment_avg": -0.8,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.resolve_crisis(alert.id)
        assert result is not None
        assert result.resolved_at is not None

    async def test_returns_none_for_nonexistent_alert(self, repo: NewsRepository) -> None:
        result = await repo.resolve_crisis(str(uuid.uuid4()))
        assert result is None


class TestEscalateCrisis:
    async def test_marks_crisis_as_escalated(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.9,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.escalate_crisis(str(alert.id))
        assert result is not None
        assert result.escalated_to_kill_switch is True

    async def test_accepts_uuid_directly(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.9,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.escalate_crisis(alert.id)
        assert result is not None
        assert result.escalated_to_kill_switch is True

    async def test_returns_none_for_nonexistent_alert(self, repo: NewsRepository) -> None:
        result = await repo.escalate_crisis(str(uuid.uuid4()))
        assert result is None

    async def test_backward_compat_alias(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.9,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.update_crisis_escalation(alert.id)
        assert result is not None
        assert result.escalated_to_kill_switch is True


# ─── Economic Events Tests ────────────────────────────────────────────────────


class TestStoreEvents:
    async def test_bulk_inserts_events(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        events_data = [
            {
                "event_name": "Non-Farm Payrolls",
                "event_type": "employment",
                "scheduled_at": now + timedelta(hours=1),
                "impact_level": "HIGH",
                "correlated_instruments_json": ["USD/JPY"],
            },
            {
                "event_name": "CPI Release",
                "event_type": "inflation",
                "scheduled_at": now + timedelta(hours=2),
                "impact_level": "HIGH",
                "correlated_instruments_json": ["EUR/USD"],
            },
        ]

        result = await repo.store_events(events_data)
        assert len(result) == 2
        assert all(isinstance(e, EconomicEvent) for e in result)
        assert result[0].event_name == "Non-Farm Payrolls"
        assert result[1].event_name == "CPI Release"

    async def test_store_empty_list(self, repo: NewsRepository) -> None:
        result = await repo.store_events([])
        assert result == []


class TestGetUpcomingEvents:
    async def test_returns_events_within_window(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_events([
            {
                "event_name": "Upcoming Event",
                "event_type": "rates",
                "scheduled_at": now + timedelta(minutes=10),
                "impact_level": "HIGH",
                "correlated_instruments_json": [],
            },
            {
                "event_name": "Far Future Event",
                "event_type": "rates",
                "scheduled_at": now + timedelta(hours=5),
                "impact_level": "MEDIUM",
                "correlated_instruments_json": [],
            },
        ])

        result = await repo.get_upcoming_events(within_minutes=15)
        assert len(result) == 1
        assert result[0].event_name == "Upcoming Event"

    async def test_excludes_past_events(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        await repo.store_events([
            {
                "event_name": "Past Event",
                "event_type": "gdp",
                "scheduled_at": now - timedelta(minutes=30),
                "impact_level": "HIGH",
                "correlated_instruments_json": [],
            },
        ])

        result = await repo.get_upcoming_events(within_minutes=15)
        assert len(result) == 0


class TestGetEventsForDate:
    async def test_returns_events_for_specific_date(self, repo: NewsRepository) -> None:
        target_date = datetime(2024, 6, 15, tzinfo=timezone.utc)
        await repo.store_events([
            {
                "event_name": "Target Day Event",
                "event_type": "rates",
                "scheduled_at": target_date.replace(hour=14, minute=30),
                "impact_level": "HIGH",
                "correlated_instruments_json": [],
            },
            {
                "event_name": "Other Day Event",
                "event_type": "employment",
                "scheduled_at": datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
                "impact_level": "MEDIUM",
                "correlated_instruments_json": [],
            },
        ])

        result = await repo.get_events_for_date(target_date)
        assert len(result) == 1
        assert result[0].event_name == "Target Day Event"


class TestUpdateEventActual:
    async def test_updates_actual_value(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        events = await repo.store_events([
            {
                "event_name": "Non-Farm Payrolls",
                "event_type": "employment",
                "scheduled_at": now + timedelta(hours=1),
                "impact_level": "HIGH",
                "correlated_instruments_json": ["USD/JPY"],
                "forecast_value": "180.0",
            },
        ])

        result = await repo.update_event_actual(str(events[0].id), actual_value="225.0")
        assert result is not None
        assert result.actual_value == "225.0"
        assert result.event_name == "Non-Farm Payrolls"

    async def test_returns_none_for_nonexistent_event(self, repo: NewsRepository) -> None:
        result = await repo.update_event_actual(str(uuid.uuid4()), actual_value="100.0")
        assert result is None


# ─── Geopolitical Risk Tests ──────────────────────────────────────────────────


class TestUpsertScore:
    async def test_creates_new_risk_score(self, repo: NewsRepository) -> None:
        result = await repo.upsert_score(
            region="Middle East",
            score=75.0,
            indicators={"armed_conflict": True, "sanctions": False},
        )

        assert isinstance(result, GeopoliticalRiskScore)
        assert result.region == "Middle East"
        assert result.score == 75.0
        assert result.indicators_json == {"armed_conflict": True, "sanctions": False}

    async def test_updates_existing_risk_score(self, repo: NewsRepository) -> None:
        await repo.upsert_score(
            region="Eastern Europe", score=60.0, indicators={"conflict": True}
        )
        result = await repo.upsert_score(
            region="Eastern Europe", score=80.0, indicators={"conflict": True, "escalation": True}
        )

        assert result.score == 80.0
        assert result.indicators_json == {"conflict": True, "escalation": True}

    async def test_backward_compat_alias(self, repo: NewsRepository) -> None:
        result = await repo.update_risk_score(
            region="Asia Pacific", score=55.0, indicators={"tension": True}
        )
        assert isinstance(result, GeopoliticalRiskScore)
        assert result.region == "Asia Pacific"
        assert result.score == 55.0


class TestGetAllScores:
    async def test_returns_latest_score_per_region(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Middle East", 75.0, {})
        await repo.upsert_score("Eastern Europe", 60.0, {})

        result = await repo.get_all_scores()
        assert len(result) == 2
        regions = {r.region for r in result}
        assert regions == {"Middle East", "Eastern Europe"}

    async def test_returns_empty_when_no_scores(self, repo: NewsRepository) -> None:
        result = await repo.get_all_scores()
        assert result == []

    async def test_backward_compat_alias(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Middle East", 75.0, {})

        result = await repo.get_all_risk_scores()
        assert len(result) == 1


class TestGetHighRiskRegions:
    async def test_returns_regions_above_threshold(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Middle East", 80.0, {})
        await repo.upsert_score("Eastern Europe", 50.0, {})
        await repo.upsert_score("Asia Pacific", 72.0, {})

        result = await repo.get_high_risk_regions(threshold=70.0)
        assert len(result) == 2
        regions = {r.region for r in result}
        assert regions == {"Middle East", "Asia Pacific"}

    async def test_default_threshold_is_70(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Safe Region", 30.0, {})
        await repo.upsert_score("Risky Region", 71.0, {})

        result = await repo.get_high_risk_regions()
        assert len(result) == 1
        assert result[0].region == "Risky Region"


class TestGetScoreForRegion:
    async def test_returns_score_for_existing_region(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Middle East", 75.0, {"armed_conflict": True})

        result = await repo.get_score_for_region("Middle East")
        assert result is not None
        assert result.region == "Middle East"
        assert result.score == 75.0
        assert result.indicators_json == {"armed_conflict": True}

    async def test_returns_none_for_nonexistent_region(self, repo: NewsRepository) -> None:
        result = await repo.get_score_for_region("Nonexistent Region")
        assert result is None

    async def test_returns_latest_score_when_multiple_exist(self, repo: NewsRepository) -> None:
        await repo.upsert_score("Eastern Europe", 60.0, {"v1": True})
        await repo.upsert_score("Eastern Europe", 80.0, {"v2": True})

        result = await repo.get_score_for_region("Eastern Europe")
        assert result is not None
        assert result.score == 80.0
        assert result.indicators_json == {"v2": True}


# ─── Spec-Compliant Alias Tests ───────────────────────────────────────────────


class TestUpsertRiskScore:
    """Tests for the upsert_risk_score spec-compliant alias."""

    async def test_creates_new_risk_score(self, repo: NewsRepository) -> None:
        result = await repo.upsert_risk_score(
            region="South America",
            score=45.0,
            indicators={"political_instability": True},
        )

        assert isinstance(result, GeopoliticalRiskScore)
        assert result.region == "South America"
        assert result.score == 45.0

    async def test_updates_existing_risk_score(self, repo: NewsRepository) -> None:
        await repo.upsert_risk_score("Africa", 50.0, {"drought": True})
        result = await repo.upsert_risk_score("Africa", 65.0, {"drought": True, "conflict": True})

        assert result.score == 65.0
        assert result.indicators_json == {"drought": True, "conflict": True}


class TestGetRiskScore:
    """Tests for the get_risk_score spec-compliant alias."""

    async def test_returns_score_for_existing_region(self, repo: NewsRepository) -> None:
        await repo.upsert_risk_score("North Africa", 55.0, {"sanctions": True})

        result = await repo.get_risk_score("North Africa")
        assert result is not None
        assert result.region == "North Africa"
        assert result.score == 55.0

    async def test_returns_none_for_nonexistent_region(self, repo: NewsRepository) -> None:
        result = await repo.get_risk_score("Unknown Region")
        assert result is None


class TestGetRiskScoreForRegion:
    """Tests for the get_risk_score_for_region spec interface alias."""

    async def test_returns_score_for_existing_region(self, repo: NewsRepository) -> None:
        await repo.upsert_risk_score("Central Asia", 62.0, {"instability": True})

        result = await repo.get_risk_score_for_region("Central Asia")
        assert result is not None
        assert result.region == "Central Asia"
        assert result.score == 62.0

    async def test_returns_none_for_nonexistent_region(self, repo: NewsRepository) -> None:
        result = await repo.get_risk_score_for_region("Nowhere")
        assert result is None


class TestResolveCrisisWithTimestamp:
    """Tests for resolve_crisis with explicit resolved_at parameter."""

    async def test_uses_provided_resolved_at_timestamp(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Middle East",
            "trigger_articles_json": [],
            "sentiment_avg": -0.8,
            "started_at": datetime.now(timezone.utc),
        })

        custom_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = await repo.resolve_crisis(alert.id, resolved_at=custom_time)
        assert result is not None
        assert result.resolved_at == custom_time

    async def test_defaults_to_utc_now_when_no_resolved_at(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Eastern Europe",
            "trigger_articles_json": [],
            "sentiment_avg": -0.9,
            "started_at": datetime.now(timezone.utc),
        })

        before = datetime.now(timezone.utc)
        result = await repo.resolve_crisis(alert.id)
        after = datetime.now(timezone.utc)

        assert result is not None
        assert result.resolved_at is not None
        assert before <= result.resolved_at <= after

    async def test_sets_active_to_false(self, repo: NewsRepository) -> None:
        alert = await repo.create_crisis_alert({
            "region": "Asia Pacific",
            "trigger_articles_json": [],
            "sentiment_avg": -0.75,
            "started_at": datetime.now(timezone.utc),
        })

        result = await repo.resolve_crisis(alert.id)
        assert result is not None
        assert result.active is False


class TestUpdateEventActualWithFloat:
    """Tests for update_event_actual accepting float values."""

    async def test_accepts_float_value(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        events = await repo.store_events([
            {
                "event_name": "GDP Release",
                "event_type": "gdp",
                "scheduled_at": now + timedelta(hours=1),
                "impact_level": "HIGH",
                "correlated_instruments_json": ["GBP/USD"],
                "forecast_value": "2.5",
            },
        ])

        result = await repo.update_event_actual(events[0].id, actual_value=2.8)
        assert result is not None
        assert result.actual_value == "2.8"

    async def test_accepts_string_value(self, repo: NewsRepository) -> None:
        now = datetime.now(timezone.utc)
        events = await repo.store_events([
            {
                "event_name": "CPI Release",
                "event_type": "inflation",
                "scheduled_at": now + timedelta(hours=2),
                "impact_level": "HIGH",
                "correlated_instruments_json": ["EUR/USD"],
                "forecast_value": "3.2",
            },
        ])

        result = await repo.update_event_actual(events[0].id, actual_value="3.5")
        assert result is not None
        assert result.actual_value == "3.5"
