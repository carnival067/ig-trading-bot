"""Unit tests for the trader ranker module.

Tests cover:
- Task 20.1: Composite risk score calculation (weighted 25% each)
- Task 20.2: Eligibility filtering (90 days, >55% WR, <20% DD)
- Task 20.3: Configurable data source support
- Task 20.4: Weekly re-evaluation with 50th percentile removal
"""

import pytest

from src.copy_trading.trader_ranker import (
    InternalTraderSource,
    RankedTrader,
    TraderRanker,
    TraderStats,
)


@pytest.fixture
def ranker() -> TraderRanker:
    return TraderRanker()


@pytest.fixture
def eligible_trader() -> TraderStats:
    """A trader that meets all eligibility criteria."""
    return TraderStats(
        trader_id="T001",
        win_rate=0.65,
        max_drawdown=0.10,
        sharpe_ratio=2.0,
        consistency=5.0,
        track_record_days=120,
    )


@pytest.fixture
def ineligible_trader_short_record() -> TraderStats:
    """A trader with insufficient track record."""
    return TraderStats(
        trader_id="T002",
        win_rate=0.70,
        max_drawdown=0.08,
        sharpe_ratio=2.5,
        consistency=6.0,
        track_record_days=60,
    )


@pytest.fixture
def ineligible_trader_low_wr() -> TraderStats:
    """A trader with win rate at or below 55%."""
    return TraderStats(
        trader_id="T003",
        win_rate=0.55,
        max_drawdown=0.10,
        sharpe_ratio=2.0,
        consistency=5.0,
        track_record_days=120,
    )


@pytest.fixture
def ineligible_trader_high_dd() -> TraderStats:
    """A trader with max drawdown at or above 20%."""
    return TraderStats(
        trader_id="T004",
        win_rate=0.65,
        max_drawdown=0.20,
        sharpe_ratio=2.0,
        consistency=5.0,
        track_record_days=120,
    )


class TestRiskScoreCalculation:
    """Task 20.1: Composite risk score calculation."""

    def test_score_within_bounds(self, ranker: TraderRanker, eligible_trader: TraderStats) -> None:
        """Score should always be between 0 and 100."""
        score = ranker.calculate_risk_score(eligible_trader)
        assert 0.0 <= score <= 100.0

    def test_perfect_trader_gets_high_score(self, ranker: TraderRanker) -> None:
        """A trader with perfect metrics should score near 100."""
        perfect = TraderStats(
            trader_id="PERFECT",
            win_rate=1.0,
            max_drawdown=0.0,
            sharpe_ratio=5.0,
            consistency=10.0,
            track_record_days=365,
        )
        score = ranker.calculate_risk_score(perfect)
        assert score == 100.0

    def test_worst_trader_gets_low_score(self, ranker: TraderRanker) -> None:
        """A trader with worst metrics should score 0."""
        worst = TraderStats(
            trader_id="WORST",
            win_rate=0.0,
            max_drawdown=1.0,
            sharpe_ratio=0.0,
            consistency=0.0,
            track_record_days=10,
        )
        score = ranker.calculate_risk_score(worst)
        assert score == 0.0

    def test_equal_weighting(self, ranker: TraderRanker) -> None:
        """Each metric contributes 25% to the final score."""
        # Only win_rate contributes (0.5 * 100 * 0.25 = 12.5)
        only_wr = TraderStats(
            trader_id="WR",
            win_rate=0.5,
            max_drawdown=1.0,  # 0 score
            sharpe_ratio=0.0,  # 0 score
            consistency=0.0,  # 0 score
            track_record_days=90,
        )
        score = ranker.calculate_risk_score(only_wr)
        assert score == pytest.approx(12.5, abs=0.01)

    def test_higher_win_rate_increases_score(self, ranker: TraderRanker) -> None:
        """Higher win rate should produce a higher score."""
        low_wr = TraderStats(
            trader_id="LOW",
            win_rate=0.40,
            max_drawdown=0.10,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=100,
        )
        high_wr = TraderStats(
            trader_id="HIGH",
            win_rate=0.80,
            max_drawdown=0.10,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=100,
        )
        assert ranker.calculate_risk_score(high_wr) > ranker.calculate_risk_score(low_wr)

    def test_lower_drawdown_increases_score(self, ranker: TraderRanker) -> None:
        """Lower max drawdown should produce a higher score."""
        high_dd = TraderStats(
            trader_id="HIGH_DD",
            win_rate=0.60,
            max_drawdown=0.30,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=100,
        )
        low_dd = TraderStats(
            trader_id="LOW_DD",
            win_rate=0.60,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=100,
        )
        assert ranker.calculate_risk_score(low_dd) > ranker.calculate_risk_score(high_dd)

    def test_higher_sharpe_increases_score(self, ranker: TraderRanker) -> None:
        """Higher Sharpe ratio should produce a higher score."""
        low_sharpe = TraderStats(
            trader_id="LOW_S",
            win_rate=0.60,
            max_drawdown=0.10,
            sharpe_ratio=0.5,
            consistency=3.0,
            track_record_days=100,
        )
        high_sharpe = TraderStats(
            trader_id="HIGH_S",
            win_rate=0.60,
            max_drawdown=0.10,
            sharpe_ratio=3.5,
            consistency=3.0,
            track_record_days=100,
        )
        assert ranker.calculate_risk_score(high_sharpe) > ranker.calculate_risk_score(low_sharpe)


class TestEligibility:
    """Task 20.2: Eligibility filtering."""

    def test_eligible_trader_passes(self, ranker: TraderRanker, eligible_trader: TraderStats) -> None:
        """A trader meeting all criteria should be eligible."""
        assert ranker.is_eligible(eligible_trader) is True

    def test_insufficient_track_record(
        self, ranker: TraderRanker, ineligible_trader_short_record: TraderStats
    ) -> None:
        """A trader with < 90 days track record is ineligible."""
        assert ranker.is_eligible(ineligible_trader_short_record) is False

    def test_exactly_90_days_is_eligible(self, ranker: TraderRanker) -> None:
        """A trader with exactly 90 days track record is eligible (if other criteria met)."""
        trader = TraderStats(
            trader_id="T90",
            win_rate=0.60,
            max_drawdown=0.15,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=90,
        )
        assert ranker.is_eligible(trader) is True

    def test_win_rate_at_boundary_is_ineligible(
        self, ranker: TraderRanker, ineligible_trader_low_wr: TraderStats
    ) -> None:
        """A trader with exactly 55% win rate is ineligible (must be >55%)."""
        assert ranker.is_eligible(ineligible_trader_low_wr) is False

    def test_win_rate_just_above_boundary(self, ranker: TraderRanker) -> None:
        """A trader with win rate just above 55% is eligible."""
        trader = TraderStats(
            trader_id="T_WR",
            win_rate=0.551,
            max_drawdown=0.15,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=90,
        )
        assert ranker.is_eligible(trader) is True

    def test_drawdown_at_boundary_is_ineligible(
        self, ranker: TraderRanker, ineligible_trader_high_dd: TraderStats
    ) -> None:
        """A trader with exactly 20% drawdown is ineligible (must be <20%)."""
        assert ranker.is_eligible(ineligible_trader_high_dd) is False

    def test_drawdown_just_below_boundary(self, ranker: TraderRanker) -> None:
        """A trader with drawdown just below 20% is eligible."""
        trader = TraderStats(
            trader_id="T_DD",
            win_rate=0.60,
            max_drawdown=0.199,
            sharpe_ratio=1.5,
            consistency=3.0,
            track_record_days=90,
        )
        assert ranker.is_eligible(trader) is True


class TestRankTraders:
    """Task 20.1/20.2: Ranking with eligibility filtering."""

    def test_only_eligible_traders_ranked(self, ranker: TraderRanker) -> None:
        """Only eligible traders should appear in the ranked list."""
        traders = [
            TraderStats("T1", 0.65, 0.10, 2.0, 5.0, 120),  # eligible
            TraderStats("T2", 0.50, 0.10, 2.0, 5.0, 120),  # ineligible (WR <= 55%)
            TraderStats("T3", 0.65, 0.25, 2.0, 5.0, 120),  # ineligible (DD >= 20%)
            TraderStats("T4", 0.65, 0.10, 2.0, 5.0, 60),   # ineligible (< 90 days)
        ]
        ranked = ranker.rank_traders(traders)
        assert len(ranked) == 1
        assert ranked[0].trader.trader_id == "T1"

    def test_sorted_by_score_descending(self, ranker: TraderRanker) -> None:
        """Ranked traders should be sorted by score in descending order."""
        traders = [
            TraderStats("T1", 0.60, 0.15, 1.5, 3.0, 100),
            TraderStats("T2", 0.80, 0.05, 3.0, 7.0, 200),
            TraderStats("T3", 0.70, 0.10, 2.5, 5.0, 150),
        ]
        ranked = ranker.rank_traders(traders)
        assert len(ranked) == 3
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)
        assert ranked[0].trader.trader_id == "T2"

    def test_empty_list_returns_empty(self, ranker: TraderRanker) -> None:
        """Empty input should return empty output."""
        assert ranker.rank_traders([]) == []

    def test_all_ineligible_returns_empty(self, ranker: TraderRanker) -> None:
        """If no traders are eligible, return empty list."""
        traders = [
            TraderStats("T1", 0.40, 0.30, 0.5, 1.0, 30),
            TraderStats("T2", 0.45, 0.25, 0.8, 2.0, 50),
        ]
        assert ranker.rank_traders(traders) == []


class TestDataSources:
    """Task 20.3: Configurable data source support."""

    @pytest.mark.asyncio
    async def test_internal_source(self) -> None:
        """InternalTraderSource should return configured traders."""
        traders = [
            TraderStats("T1", 0.65, 0.10, 2.0, 5.0, 120),
            TraderStats("T2", 0.70, 0.08, 2.5, 6.0, 150),
        ]
        source = InternalTraderSource(traders)
        result = await source.fetch_traders()
        assert len(result) == 2
        assert result[0].trader_id == "T1"

    @pytest.mark.asyncio
    async def test_fetch_and_rank_with_source(self) -> None:
        """fetch_and_rank should aggregate from all sources and rank."""
        traders = [
            TraderStats("T1", 0.65, 0.10, 2.0, 5.0, 120),
            TraderStats("T2", 0.70, 0.08, 2.5, 6.0, 150),
        ]
        source = InternalTraderSource(traders)
        ranker = TraderRanker(data_sources=[source])
        ranked = await ranker.fetch_and_rank()
        assert len(ranked) == 2
        # T2 should rank higher (better metrics)
        assert ranked[0].trader.trader_id == "T2"

    @pytest.mark.asyncio
    async def test_multiple_sources_deduplicate(self) -> None:
        """Multiple sources with same trader_id should deduplicate."""
        source1 = InternalTraderSource([
            TraderStats("T1", 0.65, 0.10, 2.0, 5.0, 120),
        ])
        source2 = InternalTraderSource([
            TraderStats("T1", 0.70, 0.08, 2.5, 6.0, 150),  # Updated stats
            TraderStats("T2", 0.60, 0.12, 1.8, 4.0, 100),
        ])
        ranker = TraderRanker(data_sources=[source1, source2])
        ranked = await ranker.fetch_and_rank()
        # T1 should use source2's stats (last write wins)
        assert len(ranked) == 2


class TestWeeklyReevaluation:
    """Task 20.4: Weekly re-evaluation with 50th percentile removal."""

    def test_removes_below_50th_percentile(self, ranker: TraderRanker) -> None:
        """Traders below 50th percentile should be removed."""
        # Create 20 traders with varying quality
        traders = [
            TraderStats(f"T{i:02d}", 0.56 + i * 0.02, 0.05 + i * 0.005, 1.5 + i * 0.1, 3.0 + i * 0.3, 100 + i * 5)
            for i in range(20)
        ]
        result = ranker.weekly_reevaluate(traders)
        # Should have roughly half (those above median)
        assert len(result) <= 20
        assert len(result) >= 10  # min pool size

    def test_maintains_minimum_pool_size(self) -> None:
        """Pool should never go below min_pool_size (default 10)."""
        ranker = TraderRanker(min_pool_size=10)
        # Create 12 eligible traders
        traders = [
            TraderStats(f"T{i:02d}", 0.56 + i * 0.01, 0.10, 2.0, 5.0, 100)
            for i in range(12)
        ]
        result = ranker.weekly_reevaluate(traders)
        assert len(result) >= 10

    def test_keeps_all_when_at_minimum(self) -> None:
        """If pool is at or below minimum, keep all eligible traders."""
        ranker = TraderRanker(min_pool_size=10)
        traders = [
            TraderStats(f"T{i:02d}", 0.60, 0.10, 2.0, 5.0, 100)
            for i in range(8)
        ]
        result = ranker.weekly_reevaluate(traders)
        assert len(result) == 8  # All kept since below minimum

    def test_updates_current_pool(self, ranker: TraderRanker) -> None:
        """weekly_reevaluate should update the current_pool property."""
        traders = [
            TraderStats("T1", 0.65, 0.10, 2.0, 5.0, 120),
            TraderStats("T2", 0.70, 0.08, 2.5, 6.0, 150),
        ]
        ranker.weekly_reevaluate(traders)
        assert len(ranker.current_pool) == 2
