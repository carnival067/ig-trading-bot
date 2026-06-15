"""Liquid FX trading-session filter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SessionDecision:
    allowed: bool
    session: str
    reason: str


class SessionFilter:
    def evaluate(self, timestamp: datetime | None = None) -> SessionDecision:
        current = timestamp or datetime.now(timezone.utc)
        current = current.astimezone(timezone.utc)
        if current.weekday() >= 5:
            return SessionDecision(False, "CLOSED", "weekend")
        hour = current.hour
        if 7 <= hour < 12:
            return SessionDecision(True, "LONDON", "liquid_session")
        if 12 <= hour < 17:
            return SessionDecision(True, "OVERLAP", "london_new_york_overlap")
        if 17 <= hour < 20:
            return SessionDecision(True, "NEW_YORK", "liquid_session")
        return SessionDecision(False, "OFF_HOURS", "outside_approved_fx_sessions")
