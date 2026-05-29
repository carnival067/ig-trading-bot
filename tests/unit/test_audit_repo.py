"""Unit tests for the AuditRepository."""

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Set required env vars before importing src modules
os.environ.setdefault("IG_API_KEY", "test_key")
os.environ.setdefault("IG_USERNAME", "test_user")
os.environ.setdefault("IG_PASSWORD", "test_pass")

from src.db.database import Base
from src.db.models import AuditLog
from src.db.repositories.audit_repo import (
    ACCOUNT_LOCKOUT,
    AUDITABLE_ACTIONS,
    CONFIG_CHANGE,
    HFT_MODE_CHANGE,
    HFT_MODE_DISABLE,
    HFT_MODE_ENABLE,
    KILL_SWITCH_ACTIVATE,
    KILL_SWITCH_DEACTIVATE,
    LOGIN,
    LOGIN_FAILED,
    LOGOUT,
    MANUAL_ORDER,
    STRATEGY_DISABLE,
    STRATEGY_ENABLE,
    USER_CREATE,
    USER_DELETE,
    USER_LOGIN,
    USER_LOGIN_FAILED,
    USER_MANAGEMENT,
    AuditRepository,
)


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
def repo(async_session: AsyncSession) -> AuditRepository:
    """Create an AuditRepository instance with the test session."""
    return AuditRepository(async_session)


class TestLogAction:
    """Tests for AuditRepository.log_action."""

    async def test_creates_audit_entry(self, repo: AuditRepository) -> None:
        """log_action creates an entry with correct fields."""
        entry = await repo.log_action(
            user="admin",
            action=LOGIN,
            details={"method": "password"},
            ip_address="192.168.1.1",
        )

        assert entry.id is not None
        assert entry.user == "admin"
        assert entry.action == LOGIN
        assert entry.details == {"method": "password"}
        assert entry.ip_address == "192.168.1.1"
        assert entry.timestamp is not None

    async def test_creates_entry_with_defaults(self, repo: AuditRepository) -> None:
        """log_action works with only required fields (details=None, ip='')."""
        entry = await repo.log_action(user="system", action=KILL_SWITCH_ACTIVATE)

        assert entry.user == "system"
        assert entry.action == KILL_SWITCH_ACTIVATE
        assert entry.details is None
        assert entry.ip_address is None

    async def test_creates_entry_with_explicit_none_details(
        self, repo: AuditRepository
    ) -> None:
        """log_action accepts None for details parameter."""
        entry = await repo.log_action(user="admin", action=CONFIG_CHANGE, details=None)
        assert entry.details is None

    async def test_empty_ip_address_stored_as_none(
        self, repo: AuditRepository
    ) -> None:
        """log_action stores empty string ip_address as None."""
        entry = await repo.log_action(
            user="admin",
            action=CONFIG_CHANGE,
            details={"key": "value"},
            ip_address="",
        )
        assert entry.ip_address is None

    async def test_timestamp_is_utc(self, repo: AuditRepository) -> None:
        """log_action sets timestamp close to current UTC time."""
        before = datetime.now(timezone.utc)
        entry = await repo.log_action(
            user="admin", action=CONFIG_CHANGE, details={"setting": "risk_pct"}
        )
        after = datetime.now(timezone.utc)
        assert before <= entry.timestamp <= after


class TestGetLogs:
    """Tests for AuditRepository.get_logs."""

    async def test_returns_empty_when_no_logs(self, repo: AuditRepository) -> None:
        """get_logs returns empty list when no entries exist."""
        logs = await repo.get_logs()
        assert logs == []

    async def test_returns_all_logs_ordered_by_timestamp_desc(
        self, repo: AuditRepository
    ) -> None:
        """get_logs returns entries in reverse chronological order."""
        await repo.log_action(user="user1", action=LOGIN, details={})
        await repo.log_action(user="user2", action=LOGOUT, details={})
        await repo.log_action(user="user3", action=CONFIG_CHANGE, details={})

        logs = await repo.get_logs()
        assert len(logs) == 3
        assert logs[0].user == "user3"
        assert logs[1].user == "user2"
        assert logs[2].user == "user1"

    async def test_respects_limit(self, repo: AuditRepository) -> None:
        """get_logs respects the limit parameter."""
        for i in range(5):
            await repo.log_action(user=f"user{i}", action=LOGIN, details={})
        logs = await repo.get_logs(limit=3)
        assert len(logs) == 3

    async def test_respects_offset(self, repo: AuditRepository) -> None:
        """get_logs respects the offset parameter for pagination."""
        for i in range(5):
            await repo.log_action(user=f"user{i}", action=LOGIN, details={})

        # Get all logs to know the order
        all_logs = await repo.get_logs(limit=5)
        # Get with offset=2
        offset_logs = await repo.get_logs(limit=5, offset=2)
        assert len(offset_logs) == 3
        assert offset_logs[0].user == all_logs[2].user

    async def test_filters_by_user(self, repo: AuditRepository) -> None:
        """get_logs filters by user when specified."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="trader", action=LOGIN, details={})
        await repo.log_action(user="admin", action=CONFIG_CHANGE, details={})

        logs = await repo.get_logs(user="admin")
        assert len(logs) == 2
        assert all(log.user == "admin" for log in logs)

    async def test_filters_by_action(self, repo: AuditRepository) -> None:
        """get_logs filters by action when specified."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="admin", action=KILL_SWITCH_ACTIVATE, details={})
        await repo.log_action(user="trader", action=LOGIN, details={})

        logs = await repo.get_logs(action=LOGIN)
        assert len(logs) == 2
        assert all(log.action == LOGIN for log in logs)

    async def test_filters_by_since(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs filters by since timestamp when specified."""
        now = datetime.now(timezone.utc)

        old_entry = AuditLog(
            timestamp=now - timedelta(days=10), user="admin", action=LOGIN
        )
        recent_entry = AuditLog(
            timestamp=now - timedelta(days=1), user="admin", action=LOGOUT
        )
        async_session.add_all([old_entry, recent_entry])
        await async_session.flush()

        since = now - timedelta(days=5)
        logs = await repo.get_logs(since=since)
        assert len(logs) == 1
        assert logs[0].action == LOGOUT

    async def test_combines_multiple_filters(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs combines user, action, and since filters."""
        now = datetime.now(timezone.utc)

        entries = [
            AuditLog(
                timestamp=now - timedelta(days=10), user="admin", action=LOGIN
            ),
            AuditLog(
                timestamp=now - timedelta(hours=1), user="admin", action=LOGIN
            ),
            AuditLog(
                timestamp=now - timedelta(hours=2), user="admin", action=CONFIG_CHANGE
            ),
            AuditLog(
                timestamp=now - timedelta(hours=3), user="trader", action=LOGIN
            ),
        ]
        async_session.add_all(entries)
        await async_session.flush()

        since = now - timedelta(days=5)
        logs = await repo.get_logs(since=since, user="admin", action=LOGIN)
        assert len(logs) == 1
        assert logs[0].user == "admin"
        assert logs[0].action == LOGIN

    async def test_returns_empty_for_unknown_user(
        self, repo: AuditRepository
    ) -> None:
        """get_logs returns empty list for non-existent user."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        logs = await repo.get_logs(user="unknown")
        assert logs == []

    async def test_returns_empty_for_unknown_action(
        self, repo: AuditRepository
    ) -> None:
        """get_logs returns empty list for non-existent action."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        logs = await repo.get_logs(action="NONEXISTENT")
        assert logs == []


class TestGetLogsByUser:
    """Tests for AuditRepository.get_logs_by_user."""

    async def test_returns_logs_for_user(self, repo: AuditRepository) -> None:
        """get_logs_by_user returns only logs for the specified user."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="trader", action=LOGIN, details={})
        await repo.log_action(user="admin", action=CONFIG_CHANGE, details={})

        logs = await repo.get_logs_by_user("admin")
        assert len(logs) == 2
        assert all(log.user == "admin" for log in logs)

    async def test_returns_empty_for_unknown_user(self, repo: AuditRepository) -> None:
        """get_logs_by_user returns empty list for non-existent user."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        logs = await repo.get_logs_by_user("unknown")
        assert logs == []

    async def test_respects_limit(self, repo: AuditRepository) -> None:
        """get_logs_by_user respects the limit parameter."""
        for _ in range(5):
            await repo.log_action(user="admin", action=LOGIN, details={})
        logs = await repo.get_logs_by_user("admin", limit=3)
        assert len(logs) == 3

    async def test_ordered_by_timestamp_desc(self, repo: AuditRepository) -> None:
        """get_logs_by_user returns results in reverse chronological order."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="admin", action=CONFIG_CHANGE, details={})
        await repo.log_action(user="admin", action=STRATEGY_ENABLE, details={})

        logs = await repo.get_logs_by_user("admin")
        assert logs[0].action == STRATEGY_ENABLE
        assert logs[2].action == LOGIN


class TestGetLogsByAction:
    """Tests for AuditRepository.get_logs_by_action."""

    async def test_returns_logs_for_action(self, repo: AuditRepository) -> None:
        """get_logs_by_action returns only logs for the specified action."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="trader", action=LOGIN, details={})
        await repo.log_action(user="admin", action=CONFIG_CHANGE, details={})

        logs = await repo.get_logs_by_action(LOGIN)
        assert len(logs) == 2
        assert all(log.action == LOGIN for log in logs)

    async def test_returns_empty_for_unknown_action(self, repo: AuditRepository) -> None:
        """get_logs_by_action returns empty list for non-existent action."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        logs = await repo.get_logs_by_action("nonexistent_action")
        assert logs == []

    async def test_respects_limit(self, repo: AuditRepository) -> None:
        """get_logs_by_action respects the limit parameter."""
        for _ in range(5):
            await repo.log_action(user="admin", action=KILL_SWITCH_ACTIVATE, details={})
        logs = await repo.get_logs_by_action(KILL_SWITCH_ACTIVATE, limit=3)
        assert len(logs) == 3

    async def test_ordered_by_timestamp_desc(self, repo: AuditRepository) -> None:
        """get_logs_by_action returns results in reverse chronological order."""
        await repo.log_action(user="user1", action=LOGIN, details={})
        await repo.log_action(user="user2", action=LOGIN, details={})
        await repo.log_action(user="user3", action=LOGIN, details={})

        logs = await repo.get_logs_by_action(LOGIN)
        assert logs[0].user == "user3"
        assert logs[2].user == "user1"


class TestGetLogsInRange:
    """Tests for AuditRepository.get_logs_in_range."""

    async def test_returns_logs_within_range(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs_in_range returns only logs within the specified time range."""
        now = datetime.now(timezone.utc)

        entries = [
            AuditLog(timestamp=now - timedelta(days=10), user="admin", action=LOGIN),
            AuditLog(timestamp=now - timedelta(days=5), user="admin", action=CONFIG_CHANGE),
            AuditLog(timestamp=now - timedelta(days=1), user="admin", action=LOGOUT),
        ]
        async_session.add_all(entries)
        await async_session.flush()

        start = now - timedelta(days=7)
        end = now - timedelta(days=2)
        logs = await repo.get_logs_in_range(start, end)
        assert len(logs) == 1
        assert logs[0].action == CONFIG_CHANGE

    async def test_returns_empty_when_no_logs_in_range(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs_in_range returns empty list when no logs exist in range."""
        now = datetime.now(timezone.utc)

        entry = AuditLog(timestamp=now - timedelta(days=10), user="admin", action=LOGIN)
        async_session.add(entry)
        await async_session.flush()

        start = now - timedelta(days=5)
        end = now
        logs = await repo.get_logs_in_range(start, end)
        assert logs == []

    async def test_inclusive_boundaries(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs_in_range includes entries at exact start and end boundaries."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=5)
        end = now - timedelta(days=1)

        entries = [
            AuditLog(timestamp=start, user="admin", action=LOGIN),
            AuditLog(timestamp=end, user="admin", action=LOGOUT),
        ]
        async_session.add_all(entries)
        await async_session.flush()

        logs = await repo.get_logs_in_range(start, end)
        assert len(logs) == 2

    async def test_ordered_by_timestamp_desc(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_logs_in_range returns results in reverse chronological order."""
        now = datetime.now(timezone.utc)

        entries = [
            AuditLog(timestamp=now - timedelta(days=4), user="user1", action=LOGIN),
            AuditLog(timestamp=now - timedelta(days=3), user="user2", action=LOGIN),
            AuditLog(timestamp=now - timedelta(days=2), user="user3", action=LOGIN),
        ]
        async_session.add_all(entries)
        await async_session.flush()

        start = now - timedelta(days=5)
        end = now
        logs = await repo.get_logs_in_range(start, end)
        assert logs[0].user == "user3"
        assert logs[2].user == "user1"


class TestGetAuthAttempts:
    """Tests for AuditRepository.get_auth_attempts."""

    async def test_returns_auth_attempts(self, repo: AuditRepository) -> None:
        """get_auth_attempts returns login, login_failed, account_lockout."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="admin", action=LOGIN_FAILED, details={})
        await repo.log_action(user="admin", action=CONFIG_CHANGE, details={})

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        attempts = await repo.get_auth_attempts("admin", since)

        assert len(attempts) == 2
        assert all(
            a.action in (LOGIN, LOGIN_FAILED, ACCOUNT_LOCKOUT) for a in attempts
        )

    async def test_filters_by_user(self, repo: AuditRepository) -> None:
        """get_auth_attempts only returns attempts for the specified user."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        await repo.log_action(user="trader", action=LOGIN_FAILED, details={})

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        attempts = await repo.get_auth_attempts("admin", since)

        assert len(attempts) == 1
        assert attempts[0].user == "admin"

    async def test_filters_by_time(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """get_auth_attempts only returns attempts after the since timestamp."""
        now = datetime.now(timezone.utc)

        old_attempt = AuditLog(
            timestamp=now - timedelta(hours=2), user="admin", action=LOGIN_FAILED
        )
        recent_attempt = AuditLog(
            timestamp=now - timedelta(minutes=5), user="admin", action=LOGIN_FAILED
        )
        async_session.add_all([old_attempt, recent_attempt])
        await async_session.flush()

        since = now - timedelta(hours=1)
        attempts = await repo.get_auth_attempts("admin", since)
        assert len(attempts) == 1

    async def test_includes_account_lockout(self, repo: AuditRepository) -> None:
        """get_auth_attempts includes account_lockout action."""
        await repo.log_action(user="admin", action=ACCOUNT_LOCKOUT, details={})

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        attempts = await repo.get_auth_attempts("admin", since)

        assert len(attempts) == 1
        assert attempts[0].action == ACCOUNT_LOCKOUT


class TestCleanupOldLogs:
    """Tests for AuditRepository.cleanup_old_logs."""

    async def test_deletes_old_logs(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """cleanup_old_logs removes entries older than retention period."""
        now = datetime.now(timezone.utc)

        old_entry = AuditLog(
            timestamp=now - timedelta(days=100), user="admin", action=LOGIN
        )
        recent_entry = AuditLog(
            timestamp=now - timedelta(days=10), user="admin", action=LOGIN
        )
        async_session.add_all([old_entry, recent_entry])
        await async_session.flush()

        deleted = await repo.cleanup_old_logs(retention_days=90)
        assert deleted == 1
        logs = await repo.get_logs()
        assert len(logs) == 1

    async def test_returns_zero_when_nothing_to_delete(
        self, repo: AuditRepository
    ) -> None:
        """cleanup_old_logs returns 0 when no old entries exist."""
        await repo.log_action(user="admin", action=LOGIN, details={})
        deleted = await repo.cleanup_old_logs(retention_days=90)
        assert deleted == 0

    async def test_custom_retention_period(
        self, repo: AuditRepository, async_session: AsyncSession
    ) -> None:
        """cleanup_old_logs respects custom retention_days parameter."""
        now = datetime.now(timezone.utc)

        entry = AuditLog(
            timestamp=now - timedelta(days=40), user="admin", action=LOGIN
        )
        async_session.add(entry)
        await async_session.flush()

        # 90 days retention - should not delete
        deleted_90 = await repo.cleanup_old_logs(retention_days=90)
        assert deleted_90 == 0

        # 30 days retention - should delete
        deleted_30 = await repo.cleanup_old_logs(retention_days=30)
        assert deleted_30 == 1


class TestAuditActionConstants:
    """Tests for the audit action constants."""

    def test_all_constants_are_strings(self) -> None:
        """All action constants are non-empty strings."""
        constants = [
            LOGIN, LOGIN_FAILED, LOGOUT, ACCOUNT_LOCKOUT,
            KILL_SWITCH_ACTIVATE, KILL_SWITCH_DEACTIVATE,
            STRATEGY_ENABLE, STRATEGY_DISABLE,
            HFT_MODE_ENABLE, HFT_MODE_DISABLE, HFT_MODE_CHANGE,
            CONFIG_CHANGE, MANUAL_ORDER,
            USER_CREATE, USER_DELETE,
            USER_LOGIN, USER_LOGIN_FAILED, USER_MANAGEMENT,
        ]
        for const in constants:
            assert isinstance(const, str)
            assert len(const) > 0

    def test_auditable_actions_contains_all_required_actions(self) -> None:
        """AUDITABLE_ACTIONS includes all required actions from the spec."""
        expected = [
            "login", "login_failed", "logout",
            "kill_switch_activate", "kill_switch_deactivate",
            "strategy_enable", "strategy_disable",
            "config_change",
            "hft_mode_enable", "hft_mode_disable", "hft_mode_change",
            "manual_order",
            "user_create", "user_delete",
            "account_lockout",
            "user_login", "user_login_failed", "user_management",
        ]
        for action in expected:
            assert action in AUDITABLE_ACTIONS

    def test_task_specified_actions_present(self) -> None:
        """AUDITABLE_ACTIONS includes all actions specified in task 2.5."""
        task_actions = [
            "kill_switch_activate",
            "kill_switch_deactivate",
            "strategy_enable",
            "strategy_disable",
            "config_change",
            "hft_mode_change",
            "user_login",
            "user_login_failed",
            "manual_order",
            "user_management",
        ]
        for action in task_actions:
            assert action in AUDITABLE_ACTIONS
