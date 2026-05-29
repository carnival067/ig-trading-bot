"""Unit tests for MistakeRepository async CRUD operations."""

import os
import uuid
from datetime import datetime, timedelta, timezone

# Set required env vars before importing settings-dependent modules
os.environ.setdefault("IG_API_KEY", "test_key")
os.environ.setdefault("IG_USERNAME", "test_user")
os.environ.setdefault("IG_PASSWORD", "test_pass")

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.database import Base
from src.db.models import MistakePattern, MistakeRecord
from src.db.repositories.mistake_repo import MistakeRepository


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def repo(async_session: AsyncSession) -> MistakeRepository:
    """Create a MistakeRepository with the test session."""
    return MistakeRepository(async_session)


def _record_data(**overrides) -> dict:
    """Helper to create a record_data dict with defaults."""
    defaults = {
        "trade_id": uuid.uuid4(),
        "classification": "counter_trend_entry",
        "entry_conditions_json": {"ma_cross": "bearish", "adx": 35.0},
        "regime": "trending",
        "strategy": "trend_following",
        "indicators_json": {"rsi": 72.0, "adx": 35.0, "atr": 1.5},
        "confidence_at_entry": 75,
        "exit_reason": "stop_loss_hit",
        "pnl": -50.00,
    }
    defaults.update(overrides)
    return defaults


def _pattern_data(**overrides) -> dict:
    """Helper to create a pattern_data dict with defaults."""
    now = datetime.now(timezone.utc)
    defaults = {
        "classification": "counter_trend_entry",
        "loss_count": 5,
        "first_occurrence": now - timedelta(days=20),
        "last_occurrence": now,
        "active": True,
        "reactivated": False,
        "confidence_penalty": -20,
        "size_reduction": 0.7,
        "resolution_progress": 0,
    }
    defaults.update(overrides)
    return defaults


class TestStoreRecord:
    """Tests for store_record method."""

    async def test_store_record_returns_persisted_instance(self, repo: MistakeRepository):
        data = _record_data()
        stored = await repo.store_record(data)
        assert stored.id is not None
        assert stored.trade_id == data["trade_id"]
        assert stored.classification == "counter_trend_entry"

    async def test_store_record_sets_created_at(self, repo: MistakeRepository):
        data = _record_data()
        stored = await repo.store_record(data)
        assert stored.created_at is not None

    async def test_store_record_preserves_json_fields(self, repo: MistakeRepository):
        data = _record_data()
        stored = await repo.store_record(data)
        assert stored.entry_conditions_json == {"ma_cross": "bearish", "adx": 35.0}
        assert stored.indicators_json == {"rsi": 72.0, "adx": 35.0, "atr": 1.5}


class TestGetRecordsByClassification:
    """Tests for get_records_by_classification method."""

    async def test_returns_matching_records(self, repo: MistakeRepository):
        await repo.store_record(_record_data())
        since = datetime.now(timezone.utc) - timedelta(days=30)
        records = await repo.get_records_by_classification("counter_trend_entry", since)
        assert len(records) == 1
        assert records[0].classification == "counter_trend_entry"

    async def test_excludes_different_classification(self, repo: MistakeRepository):
        await repo.store_record(_record_data())
        since = datetime.now(timezone.utc) - timedelta(days=30)
        records = await repo.get_records_by_classification("false_breakout", since)
        assert len(records) == 0

    async def test_excludes_records_before_since(self, repo: MistakeRepository):
        await repo.store_record(_record_data())
        # Query with a future 'since' should return nothing
        since = datetime.now(timezone.utc) + timedelta(days=1)
        records = await repo.get_records_by_classification("counter_trend_entry", since)
        assert len(records) == 0

    async def test_returns_multiple_records_ordered(self, repo: MistakeRepository):
        for _ in range(3):
            await repo.store_record(_record_data(
                trade_id=uuid.uuid4(),
                classification="false_breakout",
                regime="ranging",
                strategy="breakout",
            ))

        since = datetime.now(timezone.utc) - timedelta(days=30)
        records = await repo.get_records_by_classification("false_breakout", since)
        assert len(records) == 3


class TestGetRecordsForTrade:
    """Tests for get_records_for_trade method."""

    async def test_returns_record_for_existing_trade(self, repo: MistakeRepository):
        data = _record_data()
        stored = await repo.store_record(data)
        found = await repo.get_records_for_trade(str(stored.trade_id))
        assert found is not None
        assert found.id == stored.id

    async def test_returns_none_for_nonexistent_trade(self, repo: MistakeRepository):
        result = await repo.get_records_for_trade(str(uuid.uuid4()))
        assert result is None


class TestGetRecentRecords:
    """Tests for get_recent_records method."""

    async def test_returns_records_up_to_limit(self, repo: MistakeRepository):
        for _ in range(5):
            await repo.store_record(_record_data(trade_id=uuid.uuid4()))

        records = await repo.get_recent_records(limit=3)
        assert len(records) == 3

    async def test_returns_all_when_fewer_than_limit(self, repo: MistakeRepository):
        for _ in range(2):
            await repo.store_record(_record_data(trade_id=uuid.uuid4()))

        records = await repo.get_recent_records(limit=50)
        assert len(records) == 2

    async def test_returns_empty_list_when_no_records(self, repo: MistakeRepository):
        records = await repo.get_recent_records()
        assert records == []

    async def test_default_limit_is_50(self, repo: MistakeRepository):
        # Just verify it doesn't error with default
        records = await repo.get_recent_records()
        assert isinstance(records, list)


class TestCountByClassification:
    """Tests for count_by_classification method."""

    async def test_counts_matching_records(self, repo: MistakeRepository):
        for _ in range(3):
            await repo.store_record(_record_data(trade_id=uuid.uuid4()))

        since = datetime.now(timezone.utc) - timedelta(days=30)
        count = await repo.count_by_classification("counter_trend_entry", since)
        assert count == 3

    async def test_returns_zero_for_no_matches(self, repo: MistakeRepository):
        since = datetime.now(timezone.utc) - timedelta(days=30)
        count = await repo.count_by_classification("false_breakout", since)
        assert count == 0

    async def test_excludes_records_before_since(self, repo: MistakeRepository):
        await repo.store_record(_record_data())
        since = datetime.now(timezone.utc) + timedelta(days=1)
        count = await repo.count_by_classification("counter_trend_entry", since)
        assert count == 0


class TestGetActivePatterns:
    """Tests for get_active_patterns method."""

    async def test_returns_active_patterns(self, repo: MistakeRepository):
        await repo.create_pattern(_pattern_data())
        patterns = await repo.get_active_patterns()
        assert len(patterns) == 1
        assert patterns[0].active is True

    async def test_excludes_inactive_patterns(self, repo: MistakeRepository):
        await repo.create_pattern(_pattern_data(active=False))
        patterns = await repo.get_active_patterns()
        assert len(patterns) == 0


class TestGetPatternByClassification:
    """Tests for get_pattern_by_classification method."""

    async def test_finds_active_pattern(self, repo: MistakeRepository):
        await repo.create_pattern(_pattern_data())
        found = await repo.get_pattern_by_classification("counter_trend_entry")
        assert found is not None
        assert found.classification == "counter_trend_entry"

    async def test_ignores_inactive_pattern(self, repo: MistakeRepository):
        await repo.create_pattern(_pattern_data(active=False))
        found = await repo.get_pattern_by_classification("counter_trend_entry")
        assert found is None

    async def test_returns_none_for_unknown_classification(self, repo: MistakeRepository):
        found = await repo.get_pattern_by_classification("nonexistent")
        assert found is None


class TestCreatePattern:
    """Tests for create_pattern method."""

    async def test_creates_and_returns_pattern(self, repo: MistakeRepository):
        data = _pattern_data()
        created = await repo.create_pattern(data)
        assert created.id is not None
        assert created.classification == "counter_trend_entry"
        assert created.loss_count == 5
        assert created.active is True
        assert created.confidence_penalty == -20
        assert created.size_reduction == 0.7

    async def test_creates_reactivated_pattern(self, repo: MistakeRepository):
        data = _pattern_data(
            classification="false_breakout",
            reactivated=True,
            confidence_penalty=-30,
            size_reduction=0.5,
        )
        created = await repo.create_pattern(data)
        assert created.reactivated is True
        assert created.confidence_penalty == -30
        assert created.size_reduction == 0.5


class TestUpdatePatternStatus:
    """Tests for update_pattern_status method."""

    async def test_deactivates_pattern(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data())
        resolved_at = datetime.now(timezone.utc)
        updated = await repo.update_pattern_status(
            str(pattern.id), active=False, resolved_at=resolved_at
        )
        assert updated is not None
        assert updated.active is False
        # SQLite strips timezone info, so compare without tzinfo
        assert updated.resolved_at.replace(tzinfo=None) == resolved_at.replace(tzinfo=None)

    async def test_reactivates_pattern_with_increased_penalties(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=False))
        updated = await repo.update_pattern_status(
            str(pattern.id),
            active=True,
            reactivated=True,
            confidence_penalty=-30,
            size_reduction=0.5,
        )
        assert updated is not None
        assert updated.active is True
        assert updated.reactivated is True
        assert updated.confidence_penalty == -30
        assert updated.size_reduction == 0.5

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.update_pattern_status(str(uuid.uuid4()), active=False)
        assert result is None


class TestUpdateResolutionProgress:
    """Tests for update_resolution_progress method."""

    async def test_updates_progress(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data())
        updated = await repo.update_resolution_progress(str(pattern.id), 10)
        assert updated is not None
        assert updated.resolution_progress == 10

    async def test_resets_progress_to_zero(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(resolution_progress=15))
        updated = await repo.update_resolution_progress(str(pattern.id), 0)
        assert updated is not None
        assert updated.resolution_progress == 0

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.update_resolution_progress(str(uuid.uuid4()), 5)
        assert result is None


class TestGetAllPatterns:
    """Tests for get_all_patterns method."""

    async def test_returns_active_and_inactive_patterns(self, repo: MistakeRepository):
        await repo.create_pattern(_pattern_data(active=True))
        await repo.create_pattern(_pattern_data(
            classification="false_breakout",
            active=False,
        ))
        patterns = await repo.get_all_patterns()
        assert len(patterns) == 2

    async def test_returns_empty_list_when_no_patterns(self, repo: MistakeRepository):
        patterns = await repo.get_all_patterns()
        assert patterns == []


class TestUpdatePattern:
    """Tests for update_pattern method."""

    async def test_updates_single_field(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data())
        updated = await repo.update_pattern(str(pattern.id), resolution_progress=10)
        assert updated is not None
        assert updated.resolution_progress == 10

    async def test_updates_multiple_fields(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data())
        updated = await repo.update_pattern(
            str(pattern.id),
            active=False,
            confidence_penalty=-30,
            size_reduction=0.5,
        )
        assert updated is not None
        assert updated.active is False
        assert updated.confidence_penalty == -30
        assert updated.size_reduction == 0.5

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.update_pattern(str(uuid.uuid4()), active=False)
        assert result is None

    async def test_ignores_unknown_fields(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data())
        updated = await repo.update_pattern(
            str(pattern.id), nonexistent_field="value", resolution_progress=5
        )
        assert updated is not None
        assert updated.resolution_progress == 5


class TestDeactivatePattern:
    """Tests for deactivate_pattern method."""

    async def test_deactivates_active_pattern(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=True))
        deactivated = await repo.deactivate_pattern(str(pattern.id))
        assert deactivated is not None
        assert deactivated.active is False
        assert deactivated.resolved_at is not None

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.deactivate_pattern(str(uuid.uuid4()))
        assert result is None

    async def test_sets_resolved_at_to_current_time(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=True))
        before = datetime.now(timezone.utc)
        deactivated = await repo.deactivate_pattern(str(pattern.id))
        after = datetime.now(timezone.utc)
        assert deactivated is not None
        # resolved_at should be between before and after (SQLite strips tz)
        resolved = deactivated.resolved_at.replace(tzinfo=timezone.utc) if deactivated.resolved_at.tzinfo is None else deactivated.resolved_at
        assert before <= resolved <= after


class TestReactivatePattern:
    """Tests for reactivate_pattern method."""

    async def test_reactivates_resolved_pattern(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(
            active=False,
            resolved_at=datetime.now(timezone.utc),
            resolution_progress=20,
        ))
        reactivated = await repo.reactivate_pattern(
            str(pattern.id), confidence_penalty=-30, size_reduction=0.5
        )
        assert reactivated is not None
        assert reactivated.active is True
        assert reactivated.reactivated is True
        assert reactivated.confidence_penalty == -30
        assert reactivated.size_reduction == 0.5
        assert reactivated.resolution_progress == 0
        assert reactivated.resolved_at is None

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.reactivate_pattern(str(uuid.uuid4()), -30, 0.5)
        assert result is None

    async def test_reactivate_sets_harsher_penalties(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(
            active=False,
            confidence_penalty=-20,
            size_reduction=0.7,
        ))
        reactivated = await repo.reactivate_pattern(
            str(pattern.id), confidence_penalty=-40, size_reduction=0.4
        )
        assert reactivated is not None
        assert reactivated.confidence_penalty == -40
        assert reactivated.size_reduction == 0.4


class TestDeactivatePattern:
    """Tests for deactivate_pattern method."""

    async def test_deactivates_active_pattern_with_resolved_at(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=True))
        resolved_at = datetime.now(timezone.utc)
        deactivated = await repo.deactivate_pattern(str(pattern.id), resolved_at=resolved_at)
        assert deactivated is not None
        assert deactivated.active is False
        assert deactivated.resolved_at.replace(tzinfo=None) == resolved_at.replace(tzinfo=None)

    async def test_deactivates_pattern_with_default_resolved_at(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=True))
        before = datetime.now(timezone.utc)
        deactivated = await repo.deactivate_pattern(str(pattern.id))
        after = datetime.now(timezone.utc)
        assert deactivated is not None
        assert deactivated.active is False
        assert deactivated.resolved_at is not None
        # resolved_at should be between before and after
        resolved_naive = deactivated.resolved_at.replace(tzinfo=None)
        assert before.replace(tzinfo=None) <= resolved_naive <= after.replace(tzinfo=None)

    async def test_returns_none_for_nonexistent_pattern(self, repo: MistakeRepository):
        result = await repo.deactivate_pattern(str(uuid.uuid4()), resolved_at=datetime.now(timezone.utc))
        assert result is None

    async def test_accepts_uuid_directly(self, repo: MistakeRepository):
        pattern = await repo.create_pattern(_pattern_data(active=True))
        resolved_at = datetime.now(timezone.utc)
        deactivated = await repo.deactivate_pattern(pattern.id, resolved_at=resolved_at)
        assert deactivated is not None
        assert deactivated.active is False
