"""Audit trail logging for all admin actions.

Provides comprehensive audit logging for security-sensitive operations including:
- Kill switch activations/deactivations
- Strategy changes (enable/disable/parameter updates)
- Configuration changes
- HFT mode changes
- User management (login, logout, password changes, lockouts)

Audit logs are retained for 90 days with automatic cleanup.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Retention period for audit logs
AUDIT_RETENTION_DAYS = 90


class AuditAction(str, Enum):
    """Enumeration of auditable admin actions."""

    # Kill switch
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"

    # Strategy changes
    STRATEGY_ENABLED = "strategy_enabled"
    STRATEGY_DISABLED = "strategy_disabled"
    STRATEGY_PARAMS_UPDATED = "strategy_params_updated"

    # Configuration changes
    CONFIG_UPDATED = "config_updated"
    RISK_PARAMS_UPDATED = "risk_params_updated"

    # HFT mode changes
    HFT_ENABLED = "hft_enabled"
    HFT_DISABLED = "hft_disabled"
    HFT_PARAMS_UPDATED = "hft_params_updated"

    # User management
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    USER_LOCKED = "user_locked"
    USER_UNLOCKED = "user_unlocked"
    PASSWORD_CHANGED = "password_changed"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"

    # Trade overrides
    TRADE_MANUALLY_CLOSED = "trade_manually_closed"
    TRADE_OVERRIDE = "trade_override"


class AuditEntry(BaseModel):
    """A single audit log entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: AuditAction
    actor: str = Field(description="Username or system identifier performing the action")
    target: str = Field(default="", description="Target of the action (e.g., strategy name, user)")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    ip_address: str = Field(default="", description="IP address of the actor")
    success: bool = Field(default=True, description="Whether the action succeeded")


class AuditLogger:
    """Audit trail logger with 90-day retention.

    Stores audit entries in memory with periodic flush to persistent storage.
    In production, this would write to a dedicated audit database table.

    Usage:
        audit = AuditLogger()
        audit.log(
            action=AuditAction.KILL_SWITCH_ACTIVATED,
            actor="admin_user",
            details={"reason": "Max drawdown exceeded"}
        )
    """

    def __init__(self, retention_days: int = AUDIT_RETENTION_DAYS) -> None:
        """Initialize the audit logger.

        Args:
            retention_days: Number of days to retain audit entries (default: 90).
        """
        self._entries: list[AuditEntry] = []
        self._retention_days = retention_days
        self._logger = logging.getLogger(f"{__name__}.AuditLogger")

    @property
    def entries(self) -> list[AuditEntry]:
        """Return all current audit entries."""
        return self._entries.copy()

    def log(
        self,
        action: AuditAction,
        actor: str,
        target: str = "",
        details: dict[str, Any] | None = None,
        ip_address: str = "",
        success: bool = True,
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            action: The type of action being audited.
            actor: Username or system identifier performing the action.
            target: Target of the action (e.g., strategy name, username).
            details: Additional context about the action.
            ip_address: IP address of the actor.
            success: Whether the action succeeded.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            action=action,
            actor=actor,
            target=target,
            details=details or {},
            ip_address=ip_address,
            success=success,
        )

        self._entries.append(entry)

        # Log to structured logger for persistence
        self._logger.info(
            "AUDIT: %s by %s on %s (success=%s)",
            action.value,
            actor,
            target or "N/A",
            success,
            extra={
                "audit_id": entry.id,
                "audit_action": action.value,
                "audit_actor": actor,
                "audit_target": target,
                "audit_details": details or {},
                "audit_ip": ip_address,
                "audit_success": success,
            },
        )

        return entry

    def get_entries(
        self,
        action: AuditAction | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters.

        Args:
            action: Filter by action type.
            actor: Filter by actor username.
            since: Filter entries after this timestamp.
            until: Filter entries before this timestamp.
            limit: Maximum number of entries to return.

        Returns:
            List of matching audit entries, most recent first.
        """
        results = self._entries.copy()

        if action is not None:
            results = [e for e in results if e.action == action]

        if actor is not None:
            results = [e for e in results if e.actor == actor]

        if since is not None:
            results = [e for e in results if e.timestamp >= since]

        if until is not None:
            results = [e for e in results if e.timestamp <= until]

        # Sort by timestamp descending (most recent first)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[:limit]

    def cleanup_expired(self) -> int:
        """Remove audit entries older than the retention period.

        Returns:
            Number of entries removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed = original_count - len(self._entries)

        if removed > 0:
            self._logger.info(
                "Audit cleanup: removed %d entries older than %d days",
                removed,
                self._retention_days,
            )

        return removed

    def get_entry_count(self) -> int:
        """Return the total number of audit entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all audit entries. Used for testing only."""
        self._entries.clear()


# Module-level singleton for application-wide audit logging
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger singleton.

    Returns:
        The application-wide AuditLogger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
