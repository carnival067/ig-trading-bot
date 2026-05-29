"""News-to-instrument correlation mapping.

Maps news categories to affected financial instruments based on
historical reaction data. Updated weekly with new market data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Threshold for adding an instrument to a category mapping.
# If an instrument reacts to >60% of events in a category, it is added.
REACTION_ADD_THRESHOLD = 0.60

# Threshold for removing an instrument from a category mapping.
# If an instrument reacts to <30% of events in a category, it is removed.
REACTION_REMOVE_THRESHOLD = 0.30

# Minimum absolute price change percentage to count as a "reaction"
MIN_REACTION_PCT = 0.1


class CorrelationMapper:
    """Maps news categories to affected financial instruments.

    Maintains a mapping of news categories to the instruments most
    likely to be affected. Updated weekly based on historical market
    reactions to news events.

    Categories:
    - monetary_policy: Central bank decisions, rate changes
    - geopolitical_conflict: Wars, sanctions, territorial disputes
    - natural_disaster: Earthquakes, hurricanes, supply disruptions
    - earnings: Corporate earnings reports and guidance
    - commodity_supply: Supply chain disruptions, OPEC decisions
    """

    CATEGORIES: list[str] = [
        "monetary_policy",
        "geopolitical_conflict",
        "natural_disaster",
        "earnings",
        "commodity_supply",
    ]

    # Region-based instrument mappings for natural_disaster category
    _REGION_INSTRUMENTS: dict[str, list[str]] = {
        "asia": ["USDJPY", "AUDJPY", "NIKKEI", "HSI", "RICE", "RUBBER"],
        "europe": ["EURUSD", "GBPUSD", "DAX", "FTSE", "WHEAT"],
        "north_america": [
            "USDCAD", "USDMXN", "SPX500", "US30", "OIL", "NATGAS",
        ],
        "south_america": ["USDBRL", "USDARS", "COFFEE", "SUGAR", "SOYBEAN"],
        "middle_east": ["OIL", "BRENT", "GOLD", "USDSAR", "USDJOD"],
        "africa": ["USDZAR", "GOLD", "PLATINUM", "COCOA", "COFFEE"],
        "oceania": ["AUDUSD", "NZDUSD", "IRON_ORE", "COAL", "WOOL"],
    }

    # Default (core) correlation mappings that cannot be removed by weekly updates
    DEFAULT_CORRELATIONS: dict[str, list[str]] = {
        "monetary_policy": [
            "EURUSD", "GBPUSD", "USDJPY", "US10Y", "GOLD", "USDCHF", "US2Y",
        ],
        "geopolitical_conflict": [
            "GOLD", "OIL", "USDCHF", "LMT", "RTX", "NOC", "GD",
        ],
        "natural_disaster": [
            "OIL", "NATGAS", "WHEAT", "CORN", "ALLSTATE", "AIG",
            "TRVL",
        ],
        "earnings": [
            "SPX500", "US30", "NASDAQ", "XLK",
        ],
        "commodity_supply": [
            "OIL", "NATGAS", "GOLD", "SILVER", "WHEAT", "CORN",
            "SOYBEAN", "COFFEE",
        ],
    }

    def __init__(self) -> None:
        # Default correlation mappings per task requirements:
        # monetary_policy -> EUR/USD, GBP/USD, USD/JPY, US10Y, Gold
        # geopolitical_conflict -> Gold, Oil, USD/CHF, defense stocks
        # natural_disaster -> affected region currencies, insurance stocks, commodities
        # earnings -> specific company stocks, sector ETFs
        # commodity_supply -> Oil, Natural Gas, Gold, Silver, agricultural commodities
        self._correlations: dict[str, list[str]] = {
            category: list(instruments)
            for category, instruments in self.DEFAULT_CORRELATIONS.items()
        }

        # Custom overrides applied on top of default mappings
        self._custom_overrides: dict[str, list[str]] = {}

        # Track last weekly update timestamp
        self._last_weekly_update: datetime | None = None

    @property
    def categories(self) -> list[str]:
        """All supported news categories."""
        return list(self.CATEGORIES)

    @property
    def last_weekly_update(self) -> datetime | None:
        """Timestamp of the last weekly correlation update."""
        return self._last_weekly_update

    def get_default_instruments(self, news_category: str) -> list[str]:
        """Get the default (core) instruments for a category.

        These instruments cannot be removed by the weekly update.

        Args:
            news_category: One of the defined CATEGORIES.

        Returns:
            List of default instrument identifiers for this category.
        """
        return list(self.DEFAULT_CORRELATIONS.get(news_category, []))

    def get_affected_instruments(
        self,
        category: str,
        region: str | None = None,
    ) -> list[str]:
        """Get instruments affected by a news category.

        Args:
            category: One of the defined CATEGORIES.
            region: Optional region for location-specific mappings
                (e.g., natural_disaster). Supported regions: asia,
                europe, north_america, south_america, middle_east,
                africa, oceania.

        Returns:
            List of instrument identifiers affected by this category.
            Returns empty list for unknown categories.
        """
        # Check custom overrides first
        if category in self._custom_overrides:
            base = list(self._custom_overrides[category])
        else:
            base = list(self._correlations.get(category, []))

        if not base and category not in self.CATEGORIES:
            return []

        # For natural_disaster with a region, append region-specific instruments
        if category == "natural_disaster" and region:
            region_instruments = self._REGION_INSTRUMENTS.get(
                region.lower(), []
            )
            # Merge region instruments, avoiding duplicates while preserving order
            seen = set(base)
            for inst in region_instruments:
                if inst not in seen:
                    base.append(inst)
                    seen.add(inst)

        return base

    def set_correlations(self, category: str, instruments: list[str]) -> None:
        """Directly set the correlation mapping for a category.

        This updates the base mapping for the given category.

        Args:
            category: News category to update.
            instruments: List of correlated instrument identifiers.
        """
        if category in self.CATEGORIES:
            self._correlations[category] = list(instruments)

    def set_custom_override(
        self, category: str, instruments: list[str]
    ) -> None:
        """Set a custom override mapping for a category.

        Custom overrides take priority over the base mapping when
        get_affected_instruments is called.

        Args:
            category: News category to override.
            instruments: List of instrument identifiers for the override.
        """
        if category in self.CATEGORIES:
            self._custom_overrides[category] = list(instruments)

    def clear_custom_override(self, category: str) -> None:
        """Remove a custom override, reverting to the base mapping.

        Args:
            category: News category to clear the override for.
        """
        self._custom_overrides.pop(category, None)

    def get_all_overrides(self) -> dict[str, list[str]]:
        """Return all active custom overrides.

        Returns:
            Dictionary of category -> instrument list for all overrides.
        """
        return {k: list(v) for k, v in self._custom_overrides.items()}

    def should_run_weekly_update(self) -> bool:
        """Check if the weekly update should be run.

        Returns True if no update has been performed or if 7+ days
        have elapsed since the last update.

        Returns:
            True if the weekly update should be triggered.
        """
        if self._last_weekly_update is None:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_weekly_update
        return elapsed >= timedelta(days=7)

    def update_from_historical_data(
        self, price_reactions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Update correlation mappings based on historical price reaction data.

        Analyzes how instruments reacted to past news events in each category:
        - If an instrument consistently reacts (>60% of events), add it
        - If an instrument stops reacting (<30% of events), remove it
        - Default (core) instruments are never removed

        Args:
            price_reactions: List of dicts with keys:
                - "category": str — news category
                - "instrument": str — instrument identifier
                - "price_change_pct": float — price change percentage
                - "timestamp": str or datetime — when the reaction occurred

        Returns:
            Dict with keys: "added", "removed", "timestamp"
        """
        self._last_weekly_update = datetime.now(timezone.utc)

        result: dict[str, Any] = {
            "added": {},
            "removed": {},
            "timestamp": self._last_weekly_update.isoformat(),
        }

        if not price_reactions:
            return result

        # Group reactions by (category, instrument) and count total events + reactions
        # Structure: {category: {instrument: {"total": int, "reactions": int}}}
        stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"total": 0, "reactions": 0})
        )

        for reaction in price_reactions:
            category = reaction.get("category", "")
            instrument = reaction.get("instrument", "")
            price_change = reaction.get("price_change_pct", 0.0)

            if not category or not instrument:
                continue
            if category not in self.CATEGORIES:
                continue

            stats[category][instrument]["total"] += 1
            if abs(price_change) >= MIN_REACTION_PCT:
                stats[category][instrument]["reactions"] += 1

        # Process each category
        for category, instruments_stats in stats.items():
            added_instruments: list[str] = []
            removed_instruments: list[str] = []
            defaults = set(self.DEFAULT_CORRELATIONS.get(category, []))
            current = set(self._correlations.get(category, []))

            for instrument, counts in instruments_stats.items():
                total = counts["total"]
                reactions = counts["reactions"]

                if total == 0:
                    continue

                reaction_rate = reactions / total

                # Add if reaction rate > 60% and not already in mapping
                if reaction_rate > REACTION_ADD_THRESHOLD and instrument not in current:
                    self._correlations.setdefault(category, []).append(instrument)
                    current.add(instrument)
                    added_instruments.append(instrument)
                    logger.info(
                        "Added instrument to correlation mapping: "
                        "category=%s, instrument=%s, reaction_rate=%.1f%%",
                        category, instrument, reaction_rate * 100,
                    )

                # Remove if reaction rate < 30% and NOT a default instrument
                elif (
                    reaction_rate < REACTION_REMOVE_THRESHOLD
                    and instrument in current
                    and instrument not in defaults
                ):
                    if category in self._correlations:
                        try:
                            self._correlations[category].remove(instrument)
                        except ValueError:
                            pass
                    current.discard(instrument)
                    removed_instruments.append(instrument)
                    logger.info(
                        "Removed instrument from correlation mapping: "
                        "category=%s, instrument=%s, reaction_rate=%.1f%%",
                        category, instrument, reaction_rate * 100,
                    )

            if added_instruments:
                result["added"][category] = added_instruments
            if removed_instruments:
                result["removed"][category] = removed_instruments

        return result

    async def update_weekly(
        self, historical_reactions: list[dict[str, Any]]
    ) -> None:
        """Update correlation mappings based on historical market reactions.

        Analyzes how instruments reacted to past news events in each
        category and updates the correlation strengths accordingly.
        This is designed to be called weekly by the scheduler (task 39.4).

        Args:
            historical_reactions: List of dicts with keys:
                - "category": str — news category
                - "instrument": str — instrument identifier
                - "reaction_magnitude": float — absolute price move
                - "timestamp": datetime — when the reaction occurred
        """
        # Group reactions by category
        category_reactions: dict[str, dict[str, float]] = {
            cat: {} for cat in self.CATEGORIES
        }

        for reaction in historical_reactions:
            category = reaction.get("category", "")
            instrument = reaction.get("instrument", "")
            magnitude = reaction.get("reaction_magnitude", 0.0)

            if category in category_reactions and instrument:
                existing = category_reactions[category].get(instrument, 0.0)
                category_reactions[category][instrument] = (
                    existing + magnitude
                )

        # Update correlations — keep instruments with significant reactions
        for category, instruments in category_reactions.items():
            if not instruments:
                continue
            # Sort by total reaction magnitude and keep top instruments
            sorted_instruments = sorted(
                instruments.items(), key=lambda x: x[1], reverse=True
            )
            # Keep top 10 most reactive instruments
            self._correlations[category] = [
                inst for inst, _ in sorted_instruments[:10]
            ]
