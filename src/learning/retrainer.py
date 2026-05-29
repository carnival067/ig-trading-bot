"""Retrainer for the Continuous Learning Pipeline.

Manages weekly retraining schedule. Triggers retraining when 50+ closed
trades have accumulated since the last retraining.

Validates: Requirements 20.2, 20.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

from src.config.constants import RETRAINING_MIN_TRADES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RetrainingResult:
    """Result of a retraining attempt.

    Attributes:
        triggered: Whether retraining was triggered.
        reason: Reason for the result (triggered or skipped).
        trade_count: Number of trades available for retraining.
        timestamp: When the retraining was attempted.
        success: Whether the retraining completed successfully.
    """

    triggered: bool
    reason: str
    trade_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = False


# ---------------------------------------------------------------------------
# Retrainer
# ---------------------------------------------------------------------------


class Retrainer:
    """Manages weekly model retraining based on accumulated trade data.

    Checks if sufficient trades (50+) have been closed since the last
    retraining and triggers the retraining pipeline on a weekly schedule.

    Args:
        min_trades: Minimum closed trades required before retraining.
        retraining_interval_days: Days between retraining windows.
        retrain_callback: Async callback that performs the actual retraining.
            Receives the ensemble and returns True if retraining succeeded.
    """

    def __init__(
        self,
        min_trades: int = RETRAINING_MIN_TRADES,
        retraining_interval_days: int = 7,
        retrain_callback: Callable[..., Coroutine[Any, Any, bool]] | None = None,
    ) -> None:
        self._min_trades = min_trades
        self._retraining_interval_days = retraining_interval_days
        self._retrain_callback = retrain_callback

        self._last_retraining_at: datetime | None = None
        self._retraining_history: list[RetrainingResult] = []
        self._trade_count_at_last_retraining: int = 0

    @property
    def last_retraining_at(self) -> datetime | None:
        """Timestamp of the last successful retraining."""
        return self._last_retraining_at

    @property
    def retraining_history(self) -> list[RetrainingResult]:
        """History of retraining attempts."""
        return list(self._retraining_history)

    @property
    def min_trades(self) -> int:
        """Minimum trades required for retraining."""
        return self._min_trades

    def is_retraining_due(self, current_time: datetime | None = None) -> bool:
        """Check if the weekly retraining window has been reached.

        Args:
            current_time: Current time for comparison.

        Returns:
            True if retraining interval has elapsed since last retraining.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if self._last_retraining_at is None:
            return True

        elapsed = current_time - self._last_retraining_at
        return elapsed >= timedelta(days=self._retraining_interval_days)

    async def check_and_retrain(
        self,
        ensemble: Any,
        trade_count_since_last: int,
        current_time: datetime | None = None,
    ) -> RetrainingResult:
        """Check if retraining should occur and trigger if conditions are met.

        Retraining is triggered when:
        1. The weekly retraining window has been reached, AND
        2. At least 50 closed trades have accumulated since last retraining.

        Args:
            ensemble: The ML ensemble to retrain.
            trade_count_since_last: Number of closed trades since last retraining.
            current_time: Current time for scheduling checks.

        Returns:
            RetrainingResult indicating whether retraining was triggered.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Check if retraining window has been reached
        if not self.is_retraining_due(current_time):
            result = RetrainingResult(
                triggered=False,
                reason="Retraining interval not yet reached",
                trade_count=trade_count_since_last,
                timestamp=current_time,
                success=False,
            )
            self._retraining_history.append(result)
            logger.debug(
                "Retraining not due: last=%s interval=%d days",
                self._last_retraining_at,
                self._retraining_interval_days,
            )
            return result

        # Check minimum trade count
        if trade_count_since_last < self._min_trades:
            result = RetrainingResult(
                triggered=False,
                reason=(
                    f"Insufficient trades: {trade_count_since_last} "
                    f"(need {self._min_trades})"
                ),
                trade_count=trade_count_since_last,
                timestamp=current_time,
                success=False,
            )
            self._retraining_history.append(result)
            logger.info(
                "Retraining skipped: insufficient data (%d/%d trades)",
                trade_count_since_last,
                self._min_trades,
            )
            return result

        # Trigger retraining
        logger.info(
            "Triggering retraining: %d trades available (min=%d)",
            trade_count_since_last,
            self._min_trades,
        )

        success = False
        if self._retrain_callback:
            try:
                success = await self._retrain_callback(ensemble)
            except Exception as e:
                logger.error("Retraining failed with error: %s", e)
                success = False
        else:
            # No callback configured; mark as successful for testing
            success = True

        if success:
            self._last_retraining_at = current_time
            self._trade_count_at_last_retraining = trade_count_since_last

        result = RetrainingResult(
            triggered=True,
            reason="Retraining triggered successfully" if success else "Retraining failed",
            trade_count=trade_count_since_last,
            timestamp=current_time,
            success=success,
        )
        self._retraining_history.append(result)

        logger.info(
            "Retraining %s: trades=%d timestamp=%s",
            "succeeded" if success else "failed",
            trade_count_since_last,
            current_time.isoformat(),
        )

        return result

    def reset(self) -> None:
        """Reset the retrainer state (for testing)."""
        self._last_retraining_at = None
        self._retraining_history.clear()
        self._trade_count_at_last_retraining = 0
