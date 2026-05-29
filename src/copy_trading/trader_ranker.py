"""Trader ranking and discovery for copy trading.

Implements composite risk scoring, eligibility filtering, and weekly re-evaluation
of traders available for copy trading. Supports configurable data sources
(third-party APIs, CSV import, internal tracking) per Cross-Cutting Rule 6.

Validates: Requirements 11.1, 11.2, 11.4, Cross-Cutting Rule 6
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from src.config.constants import (
    COPY_MAX_DRAWDOWN,
    COPY_MIN_TRACK_RECORD_DAYS,
    COPY_MIN_WIN_RATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TraderStats:
    """Statistics for a trader being evaluated for copy trading.

    Attributes:
        trader_id: Unique identifier for the trader.
        win_rate: Win rate as a fraction (0.0 to 1.0).
        max_drawdown: Maximum drawdown as a fraction (0.0 to 1.0).
        sharpe_ratio: Sharpe ratio of the trader's returns.
        consistency: Inverse coefficient of variation of monthly returns (higher = more consistent).
        track_record_days: Number of days the trader has been active.
    """

    trader_id: str
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    consistency: float
    track_record_days: int


@dataclass
class RankedTrader:
    """A trader with their computed risk score.

    Attributes:
        trader: The trader's statistics.
        score: Composite risk score (0-100).
    """

    trader: TraderStats
    score: float


# ---------------------------------------------------------------------------
# Data Source Abstraction (Cross-Cutting Rule 6)
# ---------------------------------------------------------------------------


class TraderDataSource(ABC):
    """Abstract interface for trader data sources.

    Supports configurable data sources per Cross-Cutting Rule 6:
    - Third-party APIs
    - CSV import
    - Internal tracking
    """

    @abstractmethod
    async def fetch_traders(self) -> list[TraderStats]:
        """Fetch trader statistics from the data source.

        Returns:
            List of TraderStats from this source.
        """
        ...


class InternalTraderSource(TraderDataSource):
    """Internal tracking data source for trader statistics."""

    def __init__(self, traders: list[TraderStats] | None = None) -> None:
        self._traders = traders or []

    async def fetch_traders(self) -> list[TraderStats]:
        """Return internally tracked trader statistics."""
        return self._traders

    def update_traders(self, traders: list[TraderStats]) -> None:
        """Update the internal trader list."""
        self._traders = traders


class CSVTraderSource(TraderDataSource):
    """CSV file data source for trader statistics."""

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    async def fetch_traders(self) -> list[TraderStats]:
        """Parse trader statistics from a CSV file.

        Expected columns: trader_id, win_rate, max_drawdown, sharpe_ratio,
        consistency, track_record_days.

        Returns:
            List of TraderStats parsed from the CSV.
        """
        import csv

        traders: list[TraderStats] = []
        try:
            with open(self._file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    traders.append(
                        TraderStats(
                            trader_id=row["trader_id"],
                            win_rate=float(row["win_rate"]),
                            max_drawdown=float(row["max_drawdown"]),
                            sharpe_ratio=float(row["sharpe_ratio"]),
                            consistency=float(row["consistency"]),
                            track_record_days=int(row["track_record_days"]),
                        )
                    )
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error("Failed to load traders from CSV %s: %s", self._file_path, e)
        return traders


class APITraderSource(TraderDataSource):
    """Third-party API data source for trader statistics."""

    def __init__(self, api_url: str, api_key: str | None = None) -> None:
        self._api_url = api_url
        self._api_key = api_key

    async def fetch_traders(self) -> list[TraderStats]:
        """Fetch trader statistics from a third-party API.

        Returns:
            List of TraderStats from the API response.
        """
        # Placeholder for actual API integration
        logger.info("Fetching traders from API: %s", self._api_url)
        return []


# ---------------------------------------------------------------------------
# Trader Ranker
# ---------------------------------------------------------------------------


class TraderRanker:
    """Ranks and filters traders for copy trading eligibility.

    Implements composite risk scoring with equal weighting across four metrics:
    - Win rate (25%)
    - Max drawdown (25%, inverted — lower is better)
    - Sharpe ratio (25%)
    - Consistency (25%)

    Supports weekly re-evaluation with removal of traders below the 50th
    percentile (minimum 10 traders in pool).

    Args:
        data_sources: List of TraderDataSource instances to aggregate traders from.
        min_pool_size: Minimum number of traders to retain in the pool during re-evaluation.
    """

    WEIGHTS = {
        "win_rate": 0.25,
        "max_drawdown": 0.25,
        "sharpe_ratio": 0.25,
        "consistency": 0.25,
    }

    # Normalization bounds for score calculation
    _WIN_RATE_MAX = 1.0
    _DRAWDOWN_MAX = 1.0
    _SHARPE_MAX = 5.0  # Cap Sharpe normalization at 5.0
    _CONSISTENCY_MAX = 10.0  # Cap consistency normalization at 10.0

    def __init__(
        self,
        data_sources: list[TraderDataSource] | None = None,
        min_pool_size: int = 10,
    ) -> None:
        self._data_sources = data_sources or []
        self._min_pool_size = min_pool_size
        self._current_pool: list[RankedTrader] = []

    def calculate_risk_score(self, trader: TraderStats) -> float:
        """Calculate composite risk score for a trader.

        Score is 0-100, weighted equally across four metrics:
        - Win rate: normalized to [0, 100] (higher is better)
        - Max drawdown: normalized to [0, 100] (lower drawdown = higher score)
        - Sharpe ratio: normalized to [0, 100] (higher is better, capped at 5.0)
        - Consistency: normalized to [0, 100] (higher is better, capped at 10.0)

        Args:
            trader: TraderStats to score.

        Returns:
            Composite score in range [0, 100].
        """
        # Normalize each metric to 0-100 scale
        win_rate_score = min(trader.win_rate / self._WIN_RATE_MAX, 1.0) * 100
        drawdown_score = (1.0 - min(trader.max_drawdown / self._DRAWDOWN_MAX, 1.0)) * 100
        sharpe_score = min(max(trader.sharpe_ratio, 0.0) / self._SHARPE_MAX, 1.0) * 100
        consistency_score = min(max(trader.consistency, 0.0) / self._CONSISTENCY_MAX, 1.0) * 100

        # Weighted combination
        score = (
            self.WEIGHTS["win_rate"] * win_rate_score
            + self.WEIGHTS["max_drawdown"] * drawdown_score
            + self.WEIGHTS["sharpe_ratio"] * sharpe_score
            + self.WEIGHTS["consistency"] * consistency_score
        )

        return max(0.0, min(100.0, score))

    def is_eligible(self, trader: TraderStats) -> bool:
        """Check if a trader meets minimum eligibility criteria.

        Criteria:
        - Track record >= 90 days
        - Win rate > 55%
        - Max drawdown < 20%

        Args:
            trader: TraderStats to evaluate.

        Returns:
            True if the trader meets all eligibility criteria.
        """
        if trader.track_record_days < COPY_MIN_TRACK_RECORD_DAYS:
            return False
        if trader.win_rate <= COPY_MIN_WIN_RATE:
            return False
        if trader.max_drawdown >= COPY_MAX_DRAWDOWN:
            return False
        return True

    def rank_traders(self, traders: list[TraderStats]) -> list[RankedTrader]:
        """Rank traders by composite score, filtering for eligibility.

        Only eligible traders are included in the ranking. Results are sorted
        in descending order by score.

        Args:
            traders: List of TraderStats to rank.

        Returns:
            List of RankedTrader sorted by score (highest first), eligible only.
        """
        ranked: list[RankedTrader] = []
        for trader in traders:
            if self.is_eligible(trader):
                score = self.calculate_risk_score(trader)
                ranked.append(RankedTrader(trader=trader, score=score))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    def weekly_reevaluate(self, traders: list[TraderStats]) -> list[RankedTrader]:
        """Perform weekly re-evaluation of the trader pool.

        Re-ranks all traders and removes those below the 50th percentile,
        maintaining a minimum pool size of min_pool_size (default 10).

        Args:
            traders: Full list of TraderStats to re-evaluate.

        Returns:
            Updated pool of RankedTrader after filtering.
        """
        ranked = self.rank_traders(traders)

        if len(ranked) <= self._min_pool_size:
            # Keep all if at or below minimum pool size
            self._current_pool = ranked
            return ranked

        # Calculate 50th percentile score
        scores = [r.score for r in ranked]
        median_idx = len(scores) // 2
        median_score = scores[median_idx]  # Already sorted descending

        # Filter traders above 50th percentile
        above_median = [r for r in ranked if r.score >= median_score]

        # Ensure minimum pool size
        if len(above_median) < self._min_pool_size:
            above_median = ranked[: self._min_pool_size]

        self._current_pool = above_median

        removed_count = len(ranked) - len(above_median)
        if removed_count > 0:
            logger.info(
                "Weekly re-evaluation: removed %d traders below 50th percentile. "
                "Pool size: %d",
                removed_count,
                len(above_median),
            )

        return above_median

    async def fetch_and_rank(self) -> list[RankedTrader]:
        """Fetch traders from all configured data sources and rank them.

        Aggregates traders from all data sources, deduplicates by trader_id
        (keeping the most recent), and returns the ranked list.

        Returns:
            Ranked list of eligible traders from all sources.
        """
        all_traders: dict[str, TraderStats] = {}

        for source in self._data_sources:
            try:
                traders = await source.fetch_traders()
                for trader in traders:
                    all_traders[trader.trader_id] = trader
            except Exception as e:
                logger.error("Failed to fetch from data source: %s", e)

        return self.rank_traders(list(all_traders.values()))

    @property
    def current_pool(self) -> list[RankedTrader]:
        """Return the current trader pool after last re-evaluation."""
        return self._current_pool
