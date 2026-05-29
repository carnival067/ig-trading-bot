"""Trade Logger for the Continuous Learning Pipeline.

Stores complete trade context on close (indicators, regime, confidence,
ML predictions, outcome) within 5 seconds of trade closure.

Validates: Requirements 20.1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TradeContext:
    """Complete context for a closed trade.

    Attributes:
        trade_id: Unique identifier for the trade.
        indicators: Dict of indicator values at entry.
        regime: Market regime at time of entry.
        confidence: Confidence score at entry.
        ml_predictions: Dict of ML model predictions at entry.
        outcome: Trade outcome ("win" or "loss").
        pnl: Profit/loss amount.
        strategy_name: Strategy that generated the trade.
        instrument: Instrument traded.
        direction: Trade direction ("LONG" or "SHORT").
        entry_price: Entry price.
        exit_price: Exit price.
        entry_time: When the trade was opened.
        exit_time: When the trade was closed.
        logged_at: When this context was stored.
    """

    trade_id: str
    indicators: dict[str, Any]
    regime: str
    confidence: int
    ml_predictions: dict[str, Any]
    outcome: str
    pnl: float
    strategy_name: str = ""
    instrument: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    logged_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Trade Logger
# ---------------------------------------------------------------------------


class TradeLogger:
    """Stores complete trade context within 5 seconds of closure.

    Maintains an in-memory store of trade contexts for the learning pipeline.
    In production, this would persist to a database.

    Args:
        max_log_time_seconds: Maximum time allowed to log a trade context.
    """

    MAX_LOG_TIME_SECONDS = 5

    def __init__(self, max_log_time_seconds: float = 5.0) -> None:
        self._max_log_time_seconds = max_log_time_seconds
        self._trade_contexts: list[TradeContext] = []
        self._log_times: list[float] = []  # Track logging durations

    @property
    def trade_contexts(self) -> list[TradeContext]:
        """All stored trade contexts."""
        return list(self._trade_contexts)

    @property
    def trade_count(self) -> int:
        """Number of stored trade contexts."""
        return len(self._trade_contexts)

    async def log_trade_context(
        self,
        trade_id: str,
        indicators: dict[str, Any],
        regime: str,
        confidence: int,
        ml_predictions: dict[str, Any],
        outcome: str,
        pnl: float,
        strategy_name: str = "",
        instrument: str = "",
        direction: str = "",
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        entry_time: datetime | None = None,
        exit_time: datetime | None = None,
    ) -> TradeContext:
        """Store complete trade context within 5 seconds of closure.

        Args:
            trade_id: Unique trade identifier.
            indicators: Indicator values at entry.
            regime: Market regime at entry.
            confidence: Confidence score at entry.
            ml_predictions: ML model predictions at entry.
            outcome: Trade outcome ("win" or "loss").
            pnl: Profit/loss amount.
            strategy_name: Strategy that generated the trade.
            instrument: Instrument traded.
            direction: Trade direction.
            entry_price: Entry price.
            exit_price: Exit price.
            entry_time: When the trade was opened.
            exit_time: When the trade was closed.

        Returns:
            The stored TradeContext.

        Raises:
            TimeoutError: If logging takes longer than max_log_time_seconds.
        """
        start_time = time.monotonic()

        context = TradeContext(
            trade_id=trade_id,
            indicators=dict(indicators),
            regime=regime,
            confidence=confidence,
            ml_predictions=dict(ml_predictions),
            outcome=outcome,
            pnl=pnl,
            strategy_name=strategy_name,
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            logged_at=datetime.utcnow(),
        )

        self._trade_contexts.append(context)

        elapsed = time.monotonic() - start_time
        self._log_times.append(elapsed)

        if elapsed > self._max_log_time_seconds:
            logger.error(
                "Trade context logging exceeded %ds limit: trade_id=%s elapsed=%.3fs",
                self._max_log_time_seconds,
                trade_id,
                elapsed,
            )
            raise TimeoutError(
                f"Trade context logging took {elapsed:.3f}s "
                f"(limit: {self._max_log_time_seconds}s)"
            )

        logger.info(
            "Logged trade context: trade_id=%s outcome=%s pnl=%.2f elapsed=%.3fs",
            trade_id,
            outcome,
            pnl,
            elapsed,
        )

        return context

    def get_contexts_since(self, since: datetime) -> list[TradeContext]:
        """Get trade contexts logged since a given time.

        Args:
            since: Start time for filtering.

        Returns:
            List of trade contexts logged after `since`.
        """
        return [c for c in self._trade_contexts if c.logged_at >= since]

    def get_contexts_for_strategy(self, strategy_name: str) -> list[TradeContext]:
        """Get trade contexts for a specific strategy.

        Args:
            strategy_name: Strategy identifier.

        Returns:
            List of trade contexts for the strategy.
        """
        return [c for c in self._trade_contexts if c.strategy_name == strategy_name]

    def get_trade_count_since(self, since: datetime) -> int:
        """Get the number of trades logged since a given time.

        Args:
            since: Start time for counting.

        Returns:
            Number of trades logged after `since`.
        """
        return len(self.get_contexts_since(since))

    def clear(self) -> None:
        """Clear all stored trade contexts."""
        self._trade_contexts.clear()
        self._log_times.clear()
