"""Mistake Database: async storage and retrieval of mistake records and patterns.

Provides a high-level interface for the self-learning mistake analysis system,
wrapping the MistakeRepository with domain-specific operations.

Validates: Requirements 21.1, 21.2
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal
from typing import Any

from src.config.constants import (
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_BASE_SIZE_REDUCTION,
    MISTAKE_PATTERN_THRESHOLD,
    MISTAKE_PATTERN_WINDOW_DAYS,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_SIZE_REDUCTION,
    MISTAKE_RESOLUTION_STREAK,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Enums and Data Classes
# ---------------------------------------------------------------------------


class MistakeClassification(str, Enum):
    """Root-cause classification for losing trades (Req 21.2).

    Six categories covering the primary reasons for trade losses.
    """

    COUNTER_TREND = "counter_trend_entry"
    FALSE_BREAKOUT = "false_breakout"
    VOLATILITY_MISJUDGMENT = "volatility_misjudgment"
    POOR_TIMING = "poor_timing"
    OVEREXPOSURE = "overexposure"
    REGIME_MISCLASSIFICATION = "regime_misclassification"


@dataclass
class MistakeRecord:
    """Structured record of a losing trade with root-cause classification.

    Created within 10 seconds of trade closure (Req 21.1).

    Attributes:
        trade_id: Unique identifier for the trade.
        classification: Root-cause classification.
        entry_conditions: Market conditions at entry.
        regime: Market regime at time of entry.
        strategy: Strategy that generated the trade.
        indicators: Indicator values at entry (dict of name → value).
        confidence_at_entry: Confidence score at entry.
        exit_reason: Reason for trade exit (e.g., stop_loss_hit).
        pnl: Profit/loss amount (negative for losses).
        created_at: When this record was created.
    """

    trade_id: str
    classification: MistakeClassification
    entry_conditions: dict[str, Any]
    regime: str
    strategy: str
    indicators: dict[str, float]
    confidence_at_entry: int
    exit_reason: str
    pnl: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MistakePattern:
    """Detected recurring mistake pattern with penalty configuration.

    Flagged when 5+ losses with same classification occur within 30-day window (Req 21.3).

    Attributes:
        id: Unique pattern identifier.
        classification: The root-cause classification this pattern tracks.
        loss_count: Number of losses that triggered/updated this pattern.
        first_occurrence: Timestamp of the first loss in the pattern.
        last_occurrence: Timestamp of the most recent loss.
        active: Whether the pattern is currently active.
        reactivated: Whether this pattern was previously resolved and reactivated.
        confidence_penalty: Confidence score penalty (20 base, 30 reactivated).
        size_reduction: Position size multiplier (0.70 base, 0.50 reactivated).
        resolution_progress: Consecutive profitable trades toward resolution (0-20).
        indicator_conditions: Averaged indicator values from pattern trades.
    """

    id: str
    classification: MistakeClassification
    loss_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    active: bool
    reactivated: bool
    confidence_penalty: int
    size_reduction: float
    resolution_progress: int
    indicator_conditions: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mistake Database
# ---------------------------------------------------------------------------


class MistakeDatabase:
    """Async storage and retrieval for mistake records and patterns.

    Provides in-memory storage with the same interface as the DB-backed
    MistakeRepository. In production, this delegates to the repository;
    for testing and standalone use, it maintains its own state.

    Validates: Requirements 21.1 (recording), 21.3 (pattern detection support)
    """

    def __init__(self) -> None:
        self._records: list[MistakeRecord] = []
        self._patterns: list[MistakePattern] = []

    @property
    def records(self) -> list[MistakeRecord]:
        """All stored mistake records."""
        return list(self._records)

    @property
    def patterns(self) -> list[MistakePattern]:
        """All stored mistake patterns."""
        return list(self._patterns)

    async def store_record(self, record: MistakeRecord) -> None:
        """Persist a mistake record.

        Called within 10 seconds of losing trade closure (Req 21.1).

        Args:
            record: The structured mistake record to store.
        """
        self._records.append(record)
        logger.info(
            "Stored mistake record: trade_id=%s classification=%s pnl=%.2f",
            record.trade_id,
            record.classification.value,
            record.pnl,
        )

    async def get_records_by_classification(
        self, classification: str, since: datetime
    ) -> list[MistakeRecord]:
        """Query mistake records by classification within a time window.

        Used for pattern detection — checks if a classification has accumulated
        enough occurrences within the rolling window (Req 21.3).

        Args:
            classification: The root-cause classification value to filter by.
            since: Start of the time window (inclusive).

        Returns:
            List of matching MistakeRecord instances ordered by creation time.
        """
        return [
            r
            for r in self._records
            if r.classification.value == classification and r.created_at >= since
        ]

    async def get_active_patterns(self) -> list[MistakePattern]:
        """Load all active (non-resolved) mistake patterns.

        Used at startup to apply penalties immediately without warm-up (Req 21.10)
        and during signal evaluation for pattern matching.

        Returns:
            List of active MistakePattern instances.
        """
        return [p for p in self._patterns if p.active]

    async def update_pattern_status(self, pattern_id: str, active: bool) -> None:
        """Activate or deactivate a pattern.

        Used for pattern resolution (Req 21.7) and reactivation (Req 21.8).

        Args:
            pattern_id: The pattern UUID string.
            active: Whether the pattern should be active.
        """
        for pattern in self._patterns:
            if pattern.id == pattern_id:
                pattern.active = active
                logger.info(
                    "Updated pattern status: id=%s active=%s",
                    pattern_id,
                    active,
                )
                return
        logger.warning("Pattern not found for status update: id=%s", pattern_id)

    async def create_pattern(self, pattern: MistakePattern) -> None:
        """Create a new mistake pattern.

        Called when pattern detection identifies 5+ losses with the same
        classification within a 30-day window (Req 21.3).

        Args:
            pattern: The MistakePattern to persist.
        """
        self._patterns.append(pattern)
        logger.info(
            "Created mistake pattern: id=%s classification=%s loss_count=%d",
            pattern.id,
            pattern.classification.value,
            pattern.loss_count,
        )

    async def update_pattern(self, pattern_id: str, **kwargs: Any) -> MistakePattern | None:
        """Partial update of a mistake pattern by ID.

        Args:
            pattern_id: The pattern UUID string.
            **kwargs: Fields to update on the pattern.

        Returns:
            The updated MistakePattern, or None if not found.
        """
        for pattern in self._patterns:
            if pattern.id == pattern_id:
                for key, value in kwargs.items():
                    if hasattr(pattern, key):
                        setattr(pattern, key, value)
                return pattern
        return None

    async def get_pattern_by_classification(
        self, classification: str
    ) -> MistakePattern | None:
        """Find an active pattern by its classification.

        Args:
            classification: The root-cause classification value to look up.

        Returns:
            The active MistakePattern for that classification, or None.
        """
        for pattern in self._patterns:
            if pattern.classification.value == classification and pattern.active:
                return pattern
        return None

    async def get_all_patterns(self) -> list[MistakePattern]:
        """Load all mistake patterns (active and resolved).

        Returns:
            List of all MistakePattern instances.
        """
        return list(self._patterns)

    async def get_pattern_by_id(self, pattern_id: str) -> MistakePattern | None:
        """Find a pattern by its ID.

        Args:
            pattern_id: The pattern UUID string.

        Returns:
            The MistakePattern, or None if not found.
        """
        for pattern in self._patterns:
            if pattern.id == pattern_id:
                return pattern
        return None


class PersistentMistakeDatabase(MistakeDatabase):
    """Database-backed mistake storage using the same analyzer interface."""

    def __init__(
        self,
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] | None = None,
    ) -> None:
        super().__init__()
        self._session_factory = session_factory

    def _get_session_factory(self) -> Callable[[], AbstractAsyncContextManager[Any]]:
        if self._session_factory is not None:
            return self._session_factory
        from src.db.database import get_session

        return get_session

    async def store_record(self, record: MistakeRecord) -> None:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            await repo.store_record(
                {
                    "trade_id": uuid.UUID(record.trade_id),
                    "classification": record.classification.value,
                    "entry_conditions_json": dict(record.entry_conditions),
                    "regime": record.regime,
                    "strategy": record.strategy,
                    "indicators_json": dict(record.indicators),
                    "confidence_at_entry": record.confidence_at_entry,
                    "exit_reason": record.exit_reason,
                    "pnl": Decimal(str(record.pnl)),
                    "created_at": record.created_at,
                }
            )

    async def get_records_by_classification(
        self, classification: str, since: datetime
    ) -> list[MistakeRecord]:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            records = await repo.get_records_by_classification(classification, since)
            return [self._record_from_orm(record) for record in records]

    async def get_active_patterns(self) -> list[MistakePattern]:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            patterns = await repo.get_active_patterns()
            return [self._pattern_from_orm(pattern) for pattern in patterns]

    async def update_pattern_status(self, pattern_id: str, active: bool) -> None:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            await repo.update_pattern_status(pattern_id, active=active)

    async def create_pattern(self, pattern: MistakePattern) -> None:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            await repo.create_pattern(
                {
                    "id": uuid.UUID(pattern.id),
                    "classification": pattern.classification.value,
                    "loss_count": pattern.loss_count,
                    "first_occurrence": pattern.first_occurrence,
                    "last_occurrence": pattern.last_occurrence,
                    "active": pattern.active,
                    "reactivated": pattern.reactivated,
                    "confidence_penalty": pattern.confidence_penalty,
                    "size_reduction": pattern.size_reduction,
                    "resolution_progress": pattern.resolution_progress,
                    "indicator_conditions_json": dict(pattern.indicator_conditions),
                }
            )

    async def update_pattern(self, pattern_id: str, **kwargs: Any) -> MistakePattern | None:
        from src.db.repositories.mistake_repo import MistakeRepository

        updates = dict(kwargs)
        if "indicator_conditions" in updates:
            updates["indicator_conditions_json"] = updates.pop("indicator_conditions")

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            pattern = await repo.update_pattern(pattern_id, **updates)
            return self._pattern_from_orm(pattern) if pattern is not None else None

    async def get_pattern_by_classification(
        self, classification: str
    ) -> MistakePattern | None:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            pattern = await repo.get_pattern_by_classification(classification)
            return self._pattern_from_orm(pattern) if pattern is not None else None

    async def get_all_patterns(self) -> list[MistakePattern]:
        from src.db.repositories.mistake_repo import MistakeRepository

        async with self._get_session_factory()() as session:
            repo = MistakeRepository(session)
            patterns = await repo.get_all_patterns()
            return [self._pattern_from_orm(pattern) for pattern in patterns]

    async def get_pattern_by_id(self, pattern_id: str) -> MistakePattern | None:
        patterns = await self.get_all_patterns()
        return next((pattern for pattern in patterns if pattern.id == pattern_id), None)

    @staticmethod
    def _classification(value: Any) -> MistakeClassification:
        raw = getattr(value, "value", value)
        return MistakeClassification(str(raw))

    @classmethod
    def _record_from_orm(cls, record: Any) -> MistakeRecord:
        return MistakeRecord(
            trade_id=str(record.trade_id),
            classification=cls._classification(record.classification),
            entry_conditions=dict(record.entry_conditions_json or {}),
            regime=record.regime or "",
            strategy=record.strategy or "",
            indicators={k: float(v) for k, v in (record.indicators_json or {}).items()},
            confidence_at_entry=record.confidence_at_entry,
            exit_reason=record.exit_reason or "",
            pnl=float(record.pnl),
            created_at=record.created_at,
        )

    @classmethod
    def _pattern_from_orm(cls, pattern: Any) -> MistakePattern:
        return MistakePattern(
            id=str(pattern.id),
            classification=cls._classification(pattern.classification),
            loss_count=pattern.loss_count,
            first_occurrence=pattern.first_occurrence,
            last_occurrence=pattern.last_occurrence,
            active=pattern.active,
            reactivated=pattern.reactivated,
            confidence_penalty=pattern.confidence_penalty,
            size_reduction=pattern.size_reduction,
            resolution_progress=pattern.resolution_progress,
            indicator_conditions=dict(pattern.indicator_conditions_json or {}),
        )
