"""Unit tests for the CorrelationMapper weekly update logic.

Tests the update_from_historical_data method and scheduler hook
for weekly correlation mapping updates based on historical price reactions.

Validates: Requirements 23.10
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.news.correlation_mapper import (
    CorrelationMapper,
    MIN_REACTION_PCT,
    REACTION_ADD_THRESHOLD,
    REACTION_REMOVE_THRESHOLD,
)


class TestCorrelationMapperWeeklyUpdate:
    """Tests for CorrelationMapper.update_from_historical_data."""

    def setup_method(self) -> None:
        self.mapper = CorrelationMapper()

    def test_adds_instrument_above_reaction_threshold(self) -> None:
        """Instrument reacting >60% of events is added to mapping."""
        # NEW_INST reacts in 7 out of 10 events (70% > 60% threshold)
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 7 else 0.01  # 7 reactions, 3 non-reactions
            reactions.append({
                "category": "earnings",
                "instrument": "NEW_INST",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        result = self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "NEW_INST" in instruments
        assert "NEW_INST" in result["added"].get("earnings", [])

    def test_does_not_add_at_exactly_60_percent(self) -> None:
        """Instrument reacting exactly 60% is NOT added (threshold is >60%)."""
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 6 else 0.01  # Exactly 60%
            reactions.append({
                "category": "earnings",
                "instrument": "EXACT_60",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "EXACT_60" not in instruments

    def test_does_not_add_below_threshold(self) -> None:
        """Instrument reacting <=60% of events is NOT added."""
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 5 else 0.01  # 50% reaction rate
            reactions.append({
                "category": "earnings",
                "instrument": "BELOW_THRESH",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "BELOW_THRESH" not in instruments

    def test_removes_non_default_instrument_below_30_percent(self) -> None:
        """Non-default instrument reacting <30% is removed from mapping."""
        # First add a non-default instrument
        self.mapper._correlations["earnings"].append("TEMP_INST")
        assert "TEMP_INST" in self.mapper.get_affected_instruments("earnings")

        # TEMP_INST reacts in 2 out of 10 events (20% < 30% threshold)
        reactions = []
        for i in range(10):
            pct = 0.5 if i < 2 else 0.01
            reactions.append({
                "category": "earnings",
                "instrument": "TEMP_INST",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        result = self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "TEMP_INST" not in instruments
        assert "TEMP_INST" in result["removed"].get("earnings", [])

    def test_does_not_remove_at_exactly_30_percent(self) -> None:
        """Instrument reacting exactly 30% is NOT removed (threshold is <30%)."""
        self.mapper._correlations["earnings"].append("EXACT_30")

        reactions = []
        for i in range(10):
            pct = 0.5 if i < 3 else 0.01  # Exactly 30%
            reactions.append({
                "category": "earnings",
                "instrument": "EXACT_30",
                "price_change_pct": pct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "EXACT_30" in instruments

    def test_preserves_default_instruments_even_with_zero_reactions(self) -> None:
        """Default/core instruments are never removed even with 0% reaction rate."""
        # SPX500 is a default for "earnings" - give it 0% reaction rate
        reactions = []
        for _ in range(10):
            reactions.append({
                "category": "earnings",
                "instrument": "SPX500",
                "price_change_pct": 0.01,  # Below MIN_REACTION_PCT
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("earnings")
        assert "SPX500" in instruments  # Still present (core mapping)

    def test_preserves_all_default_instruments_across_categories(self) -> None:
        """All default instruments across all categories are preserved."""
        # Give all default instruments 0% reaction rate
        reactions = []
        for category, instruments in CorrelationMapper.DEFAULT_CORRELATIONS.items():
            for instrument in instruments:
                for _ in range(5):
                    reactions.append({
                        "category": category,
                        "instrument": instrument,
                        "price_change_pct": 0.0,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        self.mapper.update_from_historical_data(reactions)

        for category, default_instruments in CorrelationMapper.DEFAULT_CORRELATIONS.items():
            current = self.mapper.get_affected_instruments(category)
            for inst in default_instruments:
                assert inst in current, (
                    f"Default instrument {inst} was removed from {category}"
                )

    def test_empty_reactions_does_not_change_mappings(self) -> None:
        """Empty reaction list does not change mappings."""
        original = {
            cat: self.mapper.get_affected_instruments(cat)
            for cat in self.mapper.categories
        }
        result = self.mapper.update_from_historical_data([])
        after = {
            cat: self.mapper.get_affected_instruments(cat)
            for cat in self.mapper.categories
        }
        for cat in self.mapper.categories:
            assert set(original[cat]) == set(after[cat])
        assert result["added"] == {}
        assert result["removed"] == {}

    def test_ignores_unknown_category(self) -> None:
        """Reactions with unknown categories are ignored."""
        reactions = [
            {
                "category": "unknown_category",
                "instrument": "SOME_INST",
                "price_change_pct": 5.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        result = self.mapper.update_from_historical_data(reactions)
        assert result["added"] == {}
        assert result["removed"] == {}

    def test_ignores_reactions_with_empty_category(self) -> None:
        """Reactions with empty category are skipped."""
        reactions = [
            {
                "category": "",
                "instrument": "SOME_INST",
                "price_change_pct": 5.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        result = self.mapper.update_from_historical_data(reactions)
        assert result["added"] == {}

    def test_ignores_reactions_with_empty_instrument(self) -> None:
        """Reactions with empty instrument are skipped."""
        reactions = [
            {
                "category": "earnings",
                "instrument": "",
                "price_change_pct": 5.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        result = self.mapper.update_from_historical_data(reactions)
        assert result["added"] == {}

    def test_sets_last_update_timestamp(self) -> None:
        """Update sets the last_weekly_update timestamp."""
        assert self.mapper.last_weekly_update is None
        self.mapper.update_from_historical_data([])
        assert self.mapper.last_weekly_update is not None
        assert isinstance(self.mapper.last_weekly_update, datetime)

    def test_result_contains_iso_timestamp(self) -> None:
        """Update result includes a parseable ISO timestamp."""
        result = self.mapper.update_from_historical_data([])
        assert "timestamp" in result
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(result["timestamp"])
        assert parsed.tzinfo is not None

    def test_multiple_categories_updated_simultaneously(self) -> None:
        """Multiple categories can be updated in a single call."""
        reactions = []
        # Add NEW_FX to monetary_policy (80% reaction rate)
        for i in range(10):
            reactions.append({
                "category": "monetary_policy",
                "instrument": "NEW_FX",
                "price_change_pct": 1.0 if i < 8 else 0.01,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        # Add NEW_OIL to commodity_supply (90% reaction rate)
        for i in range(10):
            reactions.append({
                "category": "commodity_supply",
                "instrument": "NEW_OIL",
                "price_change_pct": 2.0 if i < 9 else 0.01,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        result = self.mapper.update_from_historical_data(reactions)
        assert "NEW_FX" in self.mapper.get_affected_instruments("monetary_policy")
        assert "NEW_OIL" in self.mapper.get_affected_instruments("commodity_supply")
        assert "NEW_FX" in result["added"].get("monetary_policy", [])
        assert "NEW_OIL" in result["added"].get("commodity_supply", [])

    def test_reaction_uses_absolute_price_change(self) -> None:
        """Negative price changes count as reactions (absolute value used)."""
        reactions = []
        for i in range(10):
            # All have significant negative moves
            reactions.append({
                "category": "geopolitical_conflict",
                "instrument": "NEGATIVE_MOVER",
                "price_change_pct": -2.0 if i < 8 else 0.01,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.mapper.update_from_historical_data(reactions)
        instruments = self.mapper.get_affected_instruments("geopolitical_conflict")
        assert "NEGATIVE_MOVER" in instruments


class TestCorrelationMapperSchedulerHook:
    """Tests for the should_run_weekly_update scheduler hook."""

    def setup_method(self) -> None:
        self.mapper = CorrelationMapper()

    def test_should_run_when_never_updated(self) -> None:
        """Returns True when no update has ever been performed."""
        assert self.mapper.should_run_weekly_update() is True

    def test_should_not_run_immediately_after_update(self) -> None:
        """Returns False immediately after an update."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc)
        assert self.mapper.should_run_weekly_update() is False

    def test_should_not_run_after_6_days(self) -> None:
        """Returns False if only 6 days have passed."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc) - timedelta(days=6)
        assert self.mapper.should_run_weekly_update() is False

    def test_should_run_after_7_days(self) -> None:
        """Returns True if 7 days have passed."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc) - timedelta(days=7)
        assert self.mapper.should_run_weekly_update() is True

    def test_should_run_after_8_days(self) -> None:
        """Returns True if more than 7 days have passed."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc) - timedelta(days=8)
        assert self.mapper.should_run_weekly_update() is True

    def test_update_resets_scheduler(self) -> None:
        """Running update_from_historical_data resets the scheduler."""
        self.mapper._last_weekly_update = datetime.now(timezone.utc) - timedelta(days=8)
        assert self.mapper.should_run_weekly_update() is True

        self.mapper.update_from_historical_data([])
        assert self.mapper.should_run_weekly_update() is False


class TestCorrelationMapperDefaults:
    """Tests for default instrument access."""

    def setup_method(self) -> None:
        self.mapper = CorrelationMapper()

    def test_get_default_instruments_returns_core_mapping(self) -> None:
        """get_default_instruments returns the core mapping for known categories."""
        defaults = self.mapper.get_default_instruments("monetary_policy")
        assert "EURUSD" in defaults
        assert "GOLD" in defaults
        assert len(defaults) == 7

    def test_get_default_instruments_unknown_category(self) -> None:
        """get_default_instruments returns empty for unknown category."""
        assert self.mapper.get_default_instruments("unknown") == []

    def test_defaults_are_independent_of_current_mapping(self) -> None:
        """Modifying current mapping does not affect defaults."""
        self.mapper.set_correlations("earnings", ["ONLY_ONE"])
        defaults = self.mapper.get_default_instruments("earnings")
        assert "SPX500" in defaults
        assert len(defaults) == 4
