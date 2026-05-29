"""Async CRUD repository for mistake records and patterns.

Provides database operations for the self-learning mistake analysis system,
supporting Requirements 21.1 (mistake recording) and 21.3 (pattern detection).
"""

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import MistakePattern, MistakeRecord


class MistakeRepository:
    """Repository for mistake record and pattern CRUD operations.

    Accepts an AsyncSession and provides async methods for persisting,
    querying, and managing mistake records and detected patterns.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ─── Mistake Records ─────────────────────────────────────────────────

    async def store_record(self, record_data: dict | MistakeRecord) -> MistakeRecord:
        """Persist a new mistake record.

        Args:
            record_data: A dictionary of fields or a MistakeRecord instance.

        Returns:
            The persisted MistakeRecord instance with generated ID.
        """
        if isinstance(record_data, MistakeRecord):
            record = record_data
        else:
            record = MistakeRecord(**record_data)
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def get_records_by_classification(
        self, classification: str, since: datetime
    ) -> list[MistakeRecord]:
        """Query mistake records by classification within a time window.

        Used for pattern detection — checks if a classification has accumulated
        enough occurrences within the rolling window (Req 21.3).

        Args:
            classification: The root-cause classification to filter by.
            since: Start of the time window (inclusive).

        Returns:
            List of matching MistakeRecord instances ordered by creation time.
        """
        stmt = (
            select(MistakeRecord)
            .where(
                MistakeRecord.classification == classification,
                MistakeRecord.created_at >= since,
            )
            .order_by(MistakeRecord.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_records_for_trade(self, trade_id: str | UUID) -> MistakeRecord | None:
        """Get the mistake record associated with a specific trade.

        Args:
            trade_id: The trade UUID (as string or UUID object).

        Returns:
            The MistakeRecord for the trade, or None if not found.
        """
        trade_uuid = uuid.UUID(trade_id) if isinstance(trade_id, str) else trade_id
        stmt = select(MistakeRecord).where(MistakeRecord.trade_id == trade_uuid)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_recent_records(self, limit: int = 50) -> list[MistakeRecord]:
        """Get the most recent mistake records.

        Args:
            limit: Maximum number of records to return (default 50).

        Returns:
            List of MistakeRecord instances ordered by creation time descending.
        """
        stmt = (
            select(MistakeRecord)
            .order_by(MistakeRecord.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_classification(self, classification: str, since: datetime) -> int:
        """Count mistake records by classification within a time window.

        Useful for quick threshold checks without loading full records.

        Args:
            classification: The root-cause classification to count.
            since: Start of the time window (inclusive).

        Returns:
            The number of matching records.
        """
        stmt = (
            select(func.count())
            .select_from(MistakeRecord)
            .where(
                MistakeRecord.classification == classification,
                MistakeRecord.created_at >= since,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # ─── Mistake Patterns ────────────────────────────────────────────────

    async def get_active_patterns(self) -> list[MistakePattern]:
        """Load all active (non-resolved) mistake patterns.

        Used at startup to apply penalties immediately without warm-up (Req 21.10)
        and during signal evaluation for pattern matching.

        Returns:
            List of active MistakePattern instances.
        """
        stmt = (
            select(MistakePattern)
            .where(MistakePattern.active.is_(True))
            .order_by(MistakePattern.last_occurrence.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_pattern(self, pattern_data: dict | MistakePattern) -> MistakePattern:
        """Create a new mistake pattern.

        Called when pattern detection identifies 5+ losses with the same
        classification within a 30-day window (Req 21.3).

        Args:
            pattern_data: A dictionary of fields or a MistakePattern instance.

        Returns:
            The persisted MistakePattern instance with generated ID.
        """
        if isinstance(pattern_data, MistakePattern):
            pattern = pattern_data
        else:
            pattern = MistakePattern(**pattern_data)
        self.session.add(pattern)
        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern

    async def update_pattern_status(
        self, pattern_id: str | UUID, active: bool, **kwargs: object
    ) -> MistakePattern | None:
        """Activate or deactivate a pattern with optional field updates.

        Used for pattern resolution (Req 21.7) and reactivation (Req 21.8).

        Args:
            pattern_id: The pattern UUID (as string or UUID object).
            active: Whether the pattern should be active.
            **kwargs: Additional fields to update (e.g., reactivated,
                confidence_penalty, size_reduction, resolved_at).

        Returns:
            The updated MistakePattern, or None if not found.
        """
        pattern_uuid = uuid.UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id
        stmt = select(MistakePattern).where(MistakePattern.id == pattern_uuid)
        result = await self.session.execute(stmt)
        pattern = result.scalars().first()
        if pattern is None:
            return None

        pattern.active = active
        for key, value in kwargs.items():
            if hasattr(pattern, key):
                setattr(pattern, key, value)

        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern

    async def update_resolution_progress(
        self, pattern_id: str | UUID, progress: int
    ) -> MistakePattern | None:
        """Update the consecutive profit counter for pattern resolution.

        Tracks progress toward the 20 consecutive profitable trades
        needed to resolve a pattern (Req 21.7).

        Args:
            pattern_id: The pattern UUID (as string or UUID object).
            progress: The new resolution progress value (0-20).

        Returns:
            The updated MistakePattern, or None if not found.
        """
        pattern_uuid = uuid.UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id
        stmt = select(MistakePattern).where(MistakePattern.id == pattern_uuid)
        result = await self.session.execute(stmt)
        pattern = result.scalars().first()
        if pattern is None:
            return None

        pattern.resolution_progress = progress
        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern

    async def get_pattern_by_classification(
        self, classification: str
    ) -> MistakePattern | None:
        """Find an active pattern by its classification.

        Args:
            classification: The root-cause classification to look up.

        Returns:
            The active MistakePattern for that classification, or None.
        """
        stmt = select(MistakePattern).where(
            MistakePattern.classification == classification,
            MistakePattern.active.is_(True),
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def update_pattern(self, pattern_id: str | UUID, **kwargs: object) -> MistakePattern | None:
        """Partial update of a mistake pattern by ID.

        Allows updating any combination of fields such as resolution_progress,
        active, reactivated, confidence_penalty, size_reduction, etc.

        Args:
            pattern_id: The pattern UUID (as string or UUID object).
            **kwargs: Fields to update on the pattern.

        Returns:
            The updated MistakePattern, or None if not found.
        """
        pattern_uuid = uuid.UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id
        stmt = select(MistakePattern).where(MistakePattern.id == pattern_uuid)
        result = await self.session.execute(stmt)
        pattern = result.scalars().first()
        if pattern is None:
            return None

        for key, value in kwargs.items():
            if hasattr(pattern, key):
                setattr(pattern, key, value)

        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern

    async def deactivate_pattern(
        self, pattern_id: str | UUID, resolved_at: datetime | None = None
    ) -> MistakePattern | None:
        """Deactivate a pattern by setting active=False and resolved_at.

        Used when a pattern has been resolved (20 consecutive profitable trades
        matching the pattern's conditions — Req 21.7).

        Args:
            pattern_id: The pattern UUID (as string or UUID object).
            resolved_at: The timestamp when the pattern was resolved.
                Defaults to current UTC time if not provided.

        Returns:
            The deactivated MistakePattern, or None if not found.
        """
        pattern_uuid = uuid.UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id
        stmt = select(MistakePattern).where(MistakePattern.id == pattern_uuid)
        result = await self.session.execute(stmt)
        pattern = result.scalars().first()
        if pattern is None:
            return None

        pattern.active = False
        pattern.resolved_at = resolved_at if resolved_at is not None else datetime.now(timezone.utc)

        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern

    async def get_all_patterns(self) -> list[MistakePattern]:
        """Load all mistake patterns (active and resolved).

        Returns:
            List of all MistakePattern instances ordered by creation time descending.
        """
        stmt = (
            select(MistakePattern)
            .order_by(MistakePattern.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def reactivate_pattern(
        self, pattern_id: str | UUID, confidence_penalty: int, size_reduction: float
    ) -> MistakePattern | None:
        """Reactivate a previously resolved pattern with increased penalties.

        When a resolved pattern recurs, it is reactivated with a harsher
        confidence penalty and tighter size reduction (Req 21.8).

        Args:
            pattern_id: The pattern UUID (as string or UUID object).
            confidence_penalty: The new (harsher) confidence penalty value.
            size_reduction: The new (tighter) size reduction multiplier.

        Returns:
            The reactivated MistakePattern, or None if not found.
        """
        pattern_uuid = uuid.UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id
        stmt = select(MistakePattern).where(MistakePattern.id == pattern_uuid)
        result = await self.session.execute(stmt)
        pattern = result.scalars().first()
        if pattern is None:
            return None

        pattern.active = True
        pattern.reactivated = True
        pattern.confidence_penalty = confidence_penalty
        pattern.size_reduction = size_reduction
        pattern.resolution_progress = 0
        pattern.resolved_at = None

        await self.session.flush()
        await self.session.refresh(pattern)
        return pattern
