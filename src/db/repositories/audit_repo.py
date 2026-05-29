"""Audit logging repository for tracking admin and system actions."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AuditLog

# Audit action constants matching the task specification
LOGIN = "login"
LOGIN_FAILED = "login_failed"
LOGOUT = "logout"
KILL_SWITCH_ACTIVATE = "kill_switch_activate"
KILL_SWITCH_DEACTIVATE = "kill_switch_deactivate"
STRATEGY_ENABLE = "strategy_enable"
STRATEGY_DISABLE = "strategy_disable"
CONFIG_CHANGE = "config_change"
HFT_MODE_ENABLE = "hft_mode_enable"
HFT_MODE_DISABLE = "hft_mode_disable"
HFT_MODE_CHANGE = "hft_mode_change"
MANUAL_ORDER = "manual_order"
USER_CREATE = "user_create"
USER_DELETE = "user_delete"
ACCOUNT_LOCKOUT = "account_lockout"
USER_LOGIN = "user_login"
USER_LOGIN_FAILED = "user_login_failed"
USER_MANAGEMENT = "user_management"

# Legacy constants for backward compatibility
AUTH_LOGIN = LOGIN
AUTH_LOGOUT = LOGOUT
AUTH_FAILED = LOGIN_FAILED
AUTH_LOCKOUT = ACCOUNT_LOCKOUT
KILL_SWITCH_ACTIVATED = KILL_SWITCH_ACTIVATE
KILL_SWITCH_DEACTIVATED = KILL_SWITCH_DEACTIVATE
STRATEGY_ENABLED = STRATEGY_ENABLE
STRATEGY_DISABLED = STRATEGY_DISABLE
STRATEGY_SUSPENDED = "strategy_suspended"
HFT_MODE_ENABLED = HFT_MODE_ENABLE
HFT_MODE_DISABLED = HFT_MODE_DISABLE
CONFIG_CHANGED = CONFIG_CHANGE
MANUAL_ORDER_PLACED = MANUAL_ORDER
MANUAL_POSITION_CLOSED = "manual_position_closed"
USER_CREATED = USER_CREATE
USER_UPDATED = "user_updated"

# All supported auditable actions
AUDITABLE_ACTIONS = [
    LOGIN,
    LOGIN_FAILED,
    LOGOUT,
    KILL_SWITCH_ACTIVATE,
    KILL_SWITCH_DEACTIVATE,
    STRATEGY_ENABLE,
    STRATEGY_DISABLE,
    CONFIG_CHANGE,
    HFT_MODE_ENABLE,
    HFT_MODE_DISABLE,
    HFT_MODE_CHANGE,
    MANUAL_ORDER,
    USER_CREATE,
    USER_DELETE,
    ACCOUNT_LOCKOUT,
    USER_LOGIN,
    USER_LOGIN_FAILED,
    USER_MANAGEMENT,
]


class AuditRepository:
    """Repository for audit log CRUD operations.

    Provides methods to create, query, and manage audit log entries
    for tracking authentication attempts and administrative actions.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log_action(
        self,
        user: str,
        action: str,
        details: dict | None = None,
        ip_address: str = "",
    ) -> AuditLog:
        """Create an audit log entry with the current UTC timestamp.

        Args:
            user: The user performing the action.
            action: The action type (e.g., "login", "kill_switch_activate").
            details: Optional JSON-serializable details about the action.
            ip_address: Optional IP address of the request origin.

        Returns:
            The created AuditLog record.
        """
        entry = AuditLog(
            timestamp=datetime.now(timezone.utc),
            user=user,
            action=action,
            details=details if details else None,
            ip_address=ip_address if ip_address else None,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
        user: str | None = None,
        action: str | None = None,
    ) -> list[AuditLog]:
        """Retrieve audit logs with optional filtering and pagination.

        Args:
            limit: Maximum number of records to return (default 100).
            offset: Number of records to skip for pagination (default 0).
            since: Only return logs after this timestamp (timezone-aware).
            user: Filter by user.
            action: Filter by action type.

        Returns:
            List of AuditLog entries ordered by most recent first.
        """
        stmt = select(AuditLog)

        if since is not None:
            stmt = stmt.where(AuditLog.timestamp >= since)
        if user is not None:
            stmt = stmt.where(AuditLog.user == user)
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)

        stmt = stmt.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_logs_by_user(
        self, user: str, limit: int = 100
    ) -> list[AuditLog]:
        """Retrieve audit logs filtered by user.

        Args:
            user: The user to filter by.
            limit: Maximum number of records to return (default 100).

        Returns:
            List of AuditLog entries for the specified user, most recent first.
        """
        stmt = (
            select(AuditLog)
            .where(AuditLog.user == user)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_logs_by_action(
        self, action: str, limit: int = 100
    ) -> list[AuditLog]:
        """Retrieve audit logs filtered by action type.

        Args:
            action: The action type to filter by.
            limit: Maximum number of records to return (default 100).

        Returns:
            List of AuditLog entries for the specified action, most recent first.
        """
        stmt = (
            select(AuditLog)
            .where(AuditLog.action == action)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_logs_in_range(
        self, start: datetime, end: datetime
    ) -> list[AuditLog]:
        """Retrieve audit logs within a time range.

        Args:
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).

        Returns:
            List of AuditLog entries within the range, most recent first.
        """
        stmt = (
            select(AuditLog)
            .where(AuditLog.timestamp >= start, AuditLog.timestamp <= end)
            .order_by(AuditLog.timestamp.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_auth_attempts(self, user: str, since: datetime) -> list[AuditLog]:
        """Get authentication attempts for a user since a given time.

        Used for account lockout checking (Req 19.5: lock after 5 failed attempts).
        Returns login, login_failed, and account_lockout entries.

        Args:
            user: The user to check authentication attempts for.
            since: Only return attempts after this timestamp (timezone-aware).

        Returns:
            List of AuditLog entries representing auth attempts.
        """
        auth_actions = [LOGIN, LOGIN_FAILED, ACCOUNT_LOCKOUT]
        stmt = (
            select(AuditLog)
            .where(
                AuditLog.user == user,
                AuditLog.action.in_(auth_actions),
                AuditLog.timestamp >= since,
            )
            .order_by(AuditLog.timestamp.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def cleanup_old_logs(self, retention_days: int = 90) -> int:
        """Delete audit logs older than the retention period.

        Per Req 19.4, audit records are retained for a minimum of 90 days.

        Args:
            retention_days: Number of days to retain logs (default 90).

        Returns:
            The number of deleted records.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        stmt = delete(AuditLog).where(AuditLog.timestamp < cutoff)
        result = await self.session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]
