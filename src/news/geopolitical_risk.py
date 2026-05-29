"""Geopolitical risk scoring per region.

Maintains risk scores (0-100) per geographic region, updated every 5 minutes
based on news indicators. High-risk regions (score >= 70) trigger elevated
risk management for correlated instruments.

Risk factors and their contributions:
  - Armed conflict: +30
  - Sanctions: +20
  - Political instability: +15
  - Natural disaster: +25

Scores decay over time if no new negative articles arrive (configurable decay rate).
Publishes score updates to Event Bus when a region crosses the high-risk threshold (70).

Validates: Requirements 23.15
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from src.config.constants import (
    GEOPOLITICAL_RISK_HIGH_THRESHOLD,
    GEOPOLITICAL_RISK_UPDATE_INTERVAL_MINUTES,
)
from src.core.event_bus import Event, RISK_ALERT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_REGIONS: list[str] = [
    "US",
    "Europe",
    "Asia",
    "Middle East",
    "Africa",
    "Latin America",
    "Oceania",
]

HIGH_RISK_THRESHOLD: int = GEOPOLITICAL_RISK_HIGH_THRESHOLD  # 70


class RiskFactor(str, Enum):
    """Risk factor categories that contribute to geopolitical risk scores."""

    ARMED_CONFLICT = "armed_conflict"
    SANCTIONS = "sanctions"
    POLITICAL_INSTABILITY = "political_instability"
    NATURAL_DISASTER = "natural_disaster"


# Points contributed by each risk factor when detected in a news article
RISK_FACTOR_WEIGHTS: dict[RiskFactor, int] = {
    RiskFactor.ARMED_CONFLICT: 30,
    RiskFactor.SANCTIONS: 20,
    RiskFactor.POLITICAL_INSTABILITY: 15,
    RiskFactor.NATURAL_DISASTER: 25,
}

# Default decay rate: points subtracted per update cycle (5 min) when no new articles
DEFAULT_DECAY_RATE: float = 2.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EventPublisher(Protocol):
    """Protocol for event publishing (allows decoupled testing)."""

    async def publish(self, channel: str, event: Event) -> int: ...


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RiskArticle:
    """A news article that contributes to geopolitical risk scoring."""

    article_id: str
    region: str
    risk_factor: RiskFactor
    severity: float = 1.0  # 0.0 to 1.0 multiplier on the factor weight
    received_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class RegionRiskState:
    """Internal state tracking for a single region's risk score."""

    score: float = 0.0
    last_article_at: datetime | None = None
    was_high_risk: bool = False
    factor_contributions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize factor contributions for all risk factors
        if not self.factor_contributions:
            for factor in RiskFactor:
                self.factor_contributions[factor.value] = 0.0


# ---------------------------------------------------------------------------
# GeopoliticalRiskScorer
# ---------------------------------------------------------------------------


class GeopoliticalRiskScorer:
    """Maintains and updates geopolitical risk scores per region.

    Scores range from 0 to 100 per region, updated every 5 minutes
    based on news indicators (armed conflict, sanctions, political
    instability, natural disasters).

    Each risk factor contributes a fixed number of points (scaled by severity):
      - armed_conflict: +30
      - sanctions: +20
      - political_instability: +15
      - natural_disaster: +25

    Scores decay over time if no new negative articles arrive for a region.
    When a region crosses the high-risk threshold (70), an event is published
    to the Event Bus.

    Attributes:
        UPDATE_INTERVAL_MINUTES: How often scores are recalculated (5 min).
    """

    UPDATE_INTERVAL_MINUTES: int = GEOPOLITICAL_RISK_UPDATE_INTERVAL_MINUTES

    def __init__(
        self,
        event_bus: EventPublisher | None = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> None:
        """Initialize the geopolitical risk scorer.

        Args:
            event_bus: Optional event publisher for threshold crossing alerts.
            decay_rate: Points to decay per update cycle when no new articles
                arrive for a region. Defaults to 2.0.
        """
        self._event_bus: EventPublisher | None = event_bus
        self._decay_rate: float = decay_rate
        self._regions: dict[str, RegionRiskState] = {}
        self._last_update: datetime | None = None
        self._update_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._pending_events: list[tuple[str, Event]] = []

        # Initialize all supported regions
        for region in SUPPORTED_REGIONS:
            self._regions[region] = RegionRiskState()

    @property
    def decay_rate(self) -> float:
        """Current decay rate (points per update cycle)."""
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, value: float) -> None:
        """Set the decay rate. Must be non-negative."""
        self._decay_rate = max(0.0, value)

    @property
    def is_running(self) -> bool:
        """Whether the periodic update loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic score update loop (every 5 minutes)."""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._periodic_update_loop())
        logger.info("GeopoliticalRiskScorer started")

    async def stop(self) -> None:
        """Stop the periodic score update loop."""
        self._running = False
        if self._update_task is not None:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        logger.info("GeopoliticalRiskScorer stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_risk_score(self, region: str) -> int:
        """Get the current risk score for a region.

        Args:
            region: Region identifier (e.g., "US", "Europe", "Asia").

        Returns:
            Integer risk score in range [0, 100]. Returns 0 for unknown regions.
        """
        state = self._regions.get(region)
        if state is None:
            return 0
        return int(round(state.score))

    def get_all_scores(self) -> dict[str, int]:
        """Get current risk scores for all tracked regions.

        Returns:
            Dictionary mapping region names to integer scores [0, 100].
        """
        return {
            region: int(round(state.score))
            for region, state in self._regions.items()
        }

    def is_high_risk(self, region: str) -> bool:
        """Check if a region is currently in high-risk state.

        A region is high-risk when its score is >= 70.

        Args:
            region: Region identifier.

        Returns:
            True if the region's score is at or above the high-risk threshold.
        """
        return self.get_risk_score(region) >= HIGH_RISK_THRESHOLD

    def get_high_risk_regions(self) -> list[str]:
        """Get all regions currently in high-risk state (score >= 70).

        Returns:
            List of region identifiers with scores >= threshold.
        """
        return [
            region
            for region in self._regions
            if self.is_high_risk(region)
        ]

    def get_factor_contributions(self, region: str) -> dict[str, float]:
        """Get the breakdown of risk factor contributions for a region.

        Args:
            region: Region identifier.

        Returns:
            Dictionary mapping factor names to their current contribution values.
            Returns empty dict for unknown regions.
        """
        state = self._regions.get(region)
        if state is None:
            return {}
        return dict(state.factor_contributions)

    # ------------------------------------------------------------------
    # Article Processing
    # ------------------------------------------------------------------

    def process_article(self, article: RiskArticle) -> None:
        """Process a news article and update the relevant region's risk score.

        Each article contributes points based on its risk factor type,
        scaled by the article's severity (0.0 to 1.0).

        Args:
            article: A RiskArticle with region, risk_factor, and severity.
        """
        region = article.region
        if region not in self._regions:
            # Auto-register unknown regions
            self._regions[region] = RegionRiskState()

        state = self._regions[region]
        weight = RISK_FACTOR_WEIGHTS.get(article.risk_factor, 0)
        contribution = weight * max(0.0, min(1.0, article.severity))

        # Add contribution to the factor breakdown
        state.factor_contributions[article.risk_factor.value] = min(
            100.0,
            state.factor_contributions.get(article.risk_factor.value, 0.0)
            + contribution,
        )

        # Update the total score (sum of all factor contributions, capped at 100)
        old_score = state.score
        state.score = min(
            100.0,
            sum(state.factor_contributions.values()),
        )
        state.last_article_at = article.received_at

        logger.debug(
            "Processed geopolitical risk article",
            extra={
                "region": region,
                "risk_factor": article.risk_factor.value,
                "contribution": contribution,
                "new_score": state.score,
            },
        )

        # Check threshold crossing
        self._check_threshold_crossing(region, old_score, state.score)

    def process_articles(self, articles: list[RiskArticle]) -> None:
        """Process multiple articles in batch.

        Args:
            articles: List of RiskArticle instances to process.
        """
        for article in articles:
            self.process_article(article)

    # ------------------------------------------------------------------
    # Score Update / Decay
    # ------------------------------------------------------------------

    def apply_decay(self, now: datetime | None = None) -> None:
        """Apply time-based decay to all region scores.

        Regions that haven't received new articles since the last update
        have their scores reduced by the decay rate. Factor contributions
        are decayed proportionally.

        Args:
            now: Current time for decay calculation. Defaults to UTC now.
        """
        now = now or datetime.now(timezone.utc)

        for region, state in self._regions.items():
            if state.score <= 0.0:
                continue

            old_score = state.score

            # Apply decay to each factor contribution proportionally
            if state.score > 0:
                decay_fraction = self._decay_rate / state.score
                for factor in state.factor_contributions:
                    contribution = state.factor_contributions[factor]
                    if contribution > 0:
                        reduction = contribution * decay_fraction
                        state.factor_contributions[factor] = max(
                            0.0, contribution - reduction
                        )

            # Recalculate total score
            state.score = min(
                100.0,
                max(0.0, sum(state.factor_contributions.values())),
            )

            # Check threshold crossing (downward)
            if old_score != state.score:
                self._check_threshold_crossing(region, old_score, state.score)

        self._last_update = now

    def update_scores(self, indicators: dict[str, list[dict[str, Any]]]) -> None:
        """Update risk scores based on incoming indicators (legacy interface).

        Processes news indicators per region and recalculates risk scores.
        Each indicator contributes to the region's overall risk assessment.

        Args:
            indicators: Dict mapping region names to lists of indicator dicts.
                Each indicator dict should have:
                - "type": str (one of: "armed_conflict", "sanctions",
                  "political_instability", "natural_disaster")
                - "severity": float (0.0 to 1.0)
                - "timestamp": datetime (optional)
                - "article_id": str (optional)
        """
        for region, region_indicators in indicators.items():
            for indicator in region_indicators:
                factor_str = indicator.get("type", "")
                try:
                    risk_factor = RiskFactor(factor_str)
                except ValueError:
                    logger.warning(
                        "Unknown risk factor type",
                        extra={"type": factor_str, "region": region},
                    )
                    continue

                article = RiskArticle(
                    article_id=indicator.get(
                        "article_id", f"{region}_{factor_str}_{id(indicator)}"
                    ),
                    region=region,
                    risk_factor=risk_factor,
                    severity=indicator.get("severity", 1.0),
                    received_at=indicator.get(
                        "timestamp", datetime.now(timezone.utc)
                    ),
                )
                self.process_article(article)

    def set_score(self, region: str, score: float) -> None:
        """Directly set a region's risk score.

        Distributes the score evenly across all factor contributions.

        Args:
            region: Region identifier.
            score: Risk score (0-100). Clamped to valid range.
        """
        clamped = max(0.0, min(100.0, score))
        if region not in self._regions:
            self._regions[region] = RegionRiskState()

        state = self._regions[region]
        old_score = state.score
        state.score = clamped

        # Distribute evenly across factors for consistency
        per_factor = clamped / len(RiskFactor)
        for factor in RiskFactor:
            state.factor_contributions[factor.value] = per_factor

        self._check_threshold_crossing(region, old_score, clamped)

    def reset_region(self, region: str) -> None:
        """Reset a region's risk score to zero.

        Args:
            region: Region identifier to reset.
        """
        if region in self._regions:
            state = self._regions[region]
            old_score = state.score
            state.score = 0.0
            state.last_article_at = None
            for factor in state.factor_contributions:
                state.factor_contributions[factor] = 0.0
            self._check_threshold_crossing(region, old_score, 0.0)

    # ------------------------------------------------------------------
    # Event Publishing
    # ------------------------------------------------------------------

    async def dispatch_pending_events(self) -> None:
        """Dispatch any pending threshold crossing events via the Event Bus.

        Should be called after processing articles to ensure async event
        publishing occurs.
        """
        if not self._pending_events or self._event_bus is None:
            return

        events_to_dispatch = list(self._pending_events)
        self._pending_events.clear()

        for channel, event in events_to_dispatch:
            try:
                await self._event_bus.publish(channel, event)
                logger.info(
                    "Geopolitical risk threshold event published",
                    extra={
                        "event_type": event.event_type,
                        "region": event.payload.get("region"),
                        "score": event.payload.get("score"),
                    },
                )
            except Exception as exc:
                logger.error(
                    "Failed to publish geopolitical risk event",
                    extra={"error": str(exc)},
                )

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _check_threshold_crossing(
        self, region: str, old_score: float, new_score: float
    ) -> None:
        """Check if a region has crossed the high-risk threshold.

        Publishes an event when:
        - A region crosses ABOVE the threshold (becomes high-risk)
        - A region crosses BELOW the threshold (recovers from high-risk)

        Args:
            region: Region identifier.
            old_score: Previous score before the change.
            new_score: New score after the change.
        """
        state = self._regions.get(region)
        if state is None:
            return

        was_high = old_score >= HIGH_RISK_THRESHOLD
        is_high = new_score >= HIGH_RISK_THRESHOLD

        if was_high == is_high:
            return  # No threshold crossing

        state.was_high_risk = is_high

        if is_high:
            # Region became high-risk
            event = Event(
                event_type=RISK_ALERT,
                payload={
                    "alert_type": "geopolitical_risk_elevated",
                    "region": region,
                    "score": int(round(new_score)),
                    "threshold": HIGH_RISK_THRESHOLD,
                    "direction": "above",
                    "factor_contributions": (
                        state.factor_contributions.copy()
                        if state
                        else {}
                    ),
                },
            )
            self._pending_events.append((RISK_ALERT, event))
            logger.warning(
                "Region crossed high-risk threshold",
                extra={
                    "region": region,
                    "score": new_score,
                    "threshold": HIGH_RISK_THRESHOLD,
                },
            )
        else:
            # Region recovered from high-risk
            event = Event(
                event_type=RISK_ALERT,
                payload={
                    "alert_type": "geopolitical_risk_recovered",
                    "region": region,
                    "score": int(round(new_score)),
                    "threshold": HIGH_RISK_THRESHOLD,
                    "direction": "below",
                },
            )
            self._pending_events.append((RISK_ALERT, event))
            logger.info(
                "Region recovered from high-risk",
                extra={
                    "region": region,
                    "score": new_score,
                    "threshold": HIGH_RISK_THRESHOLD,
                },
            )

    async def _periodic_update_loop(self) -> None:
        """Background loop that applies decay every UPDATE_INTERVAL_MINUTES."""
        while self._running:
            try:
                await asyncio.sleep(self.UPDATE_INTERVAL_MINUTES * 60)
                if not self._running:
                    break
                self.apply_decay()
                await self.dispatch_pending_events()
                logger.debug("Periodic geopolitical risk decay applied")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Error in geopolitical risk update loop",
                    extra={"error": str(exc)},
                )
                await asyncio.sleep(1.0)
