"""Opposing-sentiment stop tightening handler.

Subscribes to NEWS_HIGH_IMPACT events from the Event Bus and tightens
stop losses on positions that have opposing sentiment:
- For LONG positions: if sentiment < -0.8 (bearish), tighten stop
- For SHORT positions: if sentiment > 0.8 (bullish), tighten stop

The stop is tightened to 0.5 × ATR from the current price, but only
if the new stop is tighter (closer to current price) than the existing stop.

Uses the correlation mapping to determine which instruments are affected
by the news event.

Validates: Requirement 23.14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from src.risk.stop_manager import Direction, Position, StopManager

logger = logging.getLogger(__name__)

# Sentiment magnitude threshold for opposing-sentiment stop tightening
OPPOSING_SENTIMENT_THRESHOLD: float = 0.8


class PositionProvider(Protocol):
    """Protocol for retrieving open positions by instrument."""

    async def get_open_positions_for_instrument(
        self, instrument: str
    ) -> list[dict[str, Any]]:
        """Get open positions for a given instrument.

        Returns:
            List of position dicts with keys:
                - instrument: str
                - direction: str ("LONG" or "SHORT")
                - entry_price: Decimal
                - current_stop: Decimal
                - initial_stop: Decimal
                - atr_at_entry: Decimal
                - position_id: str
        """
        ...


class PriceProvider(Protocol):
    """Protocol for retrieving current market prices."""

    async def get_current_price(self, instrument: str) -> Decimal | None:
        """Get the current market price for an instrument.

        Returns:
            Current price as Decimal, or None if unavailable.
        """
        ...


class ATRProvider(Protocol):
    """Protocol for retrieving current ATR values."""

    async def get_current_atr(self, instrument: str) -> Decimal | None:
        """Get the current ATR value for an instrument.

        Returns:
            Current ATR as Decimal, or None if unavailable.
        """
        ...


class StopUpdateCallback(Protocol):
    """Protocol for applying stop loss updates to positions."""

    async def update_stop_loss(
        self, position_id: str, new_stop: Decimal
    ) -> bool:
        """Update the stop loss for a position.

        Args:
            position_id: Unique identifier of the position.
            new_stop: New stop loss price.

        Returns:
            True if the update was applied successfully.
        """
        ...


@dataclass
class StopTightenResult:
    """Result of a stop tightening operation on a single position.

    Attributes:
        position_id: The position that was evaluated.
        instrument: The instrument of the position.
        direction: Position direction (LONG/SHORT).
        original_stop: The stop before tightening.
        new_stop: The stop after tightening (may be same as original).
        tightened: Whether the stop was actually moved.
        reason: Explanation of the outcome.
    """

    position_id: str
    instrument: str
    direction: str
    original_stop: Decimal
    new_stop: Decimal
    tightened: bool
    reason: str


class SentimentStopHandler:
    """Handles opposing-sentiment stop tightening triggered by NEWS_HIGH_IMPACT events.

    When a HIGH-impact news article has strong opposing sentiment (|sentiment| > 0.8):
    - For LONG positions: if sentiment < -0.8 (bearish), tighten stop to
      current_price - 0.5 × ATR
    - For SHORT positions: if sentiment > 0.8 (bullish), tighten stop to
      current_price + 0.5 × ATR

    The stop is only tightened (moved closer to current price), never loosened.
    Only positions in instruments affected by the news (via correlation mapping)
    are evaluated.

    Args:
        stop_manager: StopManager instance for stop tightening calculations.
        position_provider: Provider for retrieving open positions by instrument.
        price_provider: Provider for retrieving current market prices.
        atr_provider: Provider for retrieving current ATR values.
        stop_update_callback: Callback for applying stop loss updates.
        sentiment_threshold: Minimum |sentiment| to trigger tightening (default 0.8).
    """

    def __init__(
        self,
        stop_manager: StopManager,
        position_provider: PositionProvider,
        price_provider: PriceProvider,
        atr_provider: ATRProvider,
        stop_update_callback: StopUpdateCallback | None = None,
        sentiment_threshold: float = OPPOSING_SENTIMENT_THRESHOLD,
    ) -> None:
        self._stop_manager = stop_manager
        self._position_provider = position_provider
        self._price_provider = price_provider
        self._atr_provider = atr_provider
        self._stop_update_callback = stop_update_callback
        self._sentiment_threshold = sentiment_threshold

    async def handle_high_impact_news(self, event_payload: dict[str, Any]) -> list[StopTightenResult]:
        """Handle a NEWS_HIGH_IMPACT event and tighten stops on opposing positions.

        This is the main entry point, intended to be registered as an event handler
        on the Event Bus for the NEWS_HIGH_IMPACT channel.

        Args:
            event_payload: Payload from the NEWS_HIGH_IMPACT event containing:
                - sentiment_score: float in [-1.0, +1.0]
                - affected_instruments: list[str] of correlated instruments
                - article_id: str (optional, for logging)
                - headline: str (optional, for logging)

        Returns:
            List of StopTightenResult for each position evaluated.
        """
        sentiment_score = event_payload.get("sentiment_score")
        affected_instruments = event_payload.get("affected_instruments", [])
        article_id = event_payload.get("article_id", "unknown")
        headline = event_payload.get("headline", "")

        if sentiment_score is None:
            logger.warning(
                "NEWS_HIGH_IMPACT event missing sentiment_score",
                extra={"article_id": article_id},
            )
            return []

        # Check if sentiment magnitude exceeds threshold
        if abs(sentiment_score) <= self._sentiment_threshold:
            logger.debug(
                "Sentiment magnitude %.2f does not exceed threshold %.2f, skipping",
                abs(sentiment_score),
                self._sentiment_threshold,
                extra={"article_id": article_id},
            )
            return []

        if not affected_instruments:
            logger.debug(
                "No affected instruments for high-impact news, skipping",
                extra={"article_id": article_id},
            )
            return []

        logger.info(
            "Processing opposing-sentiment stop tightening",
            extra={
                "article_id": article_id,
                "sentiment_score": sentiment_score,
                "affected_instruments": affected_instruments,
                "headline": headline[:80] if headline else "",
            },
        )

        results: list[StopTightenResult] = []

        for instrument in affected_instruments:
            instrument_results = await self._process_instrument(
                instrument=instrument,
                sentiment_score=sentiment_score,
                article_id=article_id,
            )
            results.extend(instrument_results)

        # Log summary
        tightened_count = sum(1 for r in results if r.tightened)
        if tightened_count > 0:
            logger.info(
                "Opposing-sentiment stop tightening complete: %d/%d positions tightened",
                tightened_count,
                len(results),
                extra={
                    "article_id": article_id,
                    "tightened_count": tightened_count,
                    "total_evaluated": len(results),
                },
            )

        return results

    async def _process_instrument(
        self,
        instrument: str,
        sentiment_score: float,
        article_id: str,
    ) -> list[StopTightenResult]:
        """Process stop tightening for all positions in a single instrument.

        Args:
            instrument: The instrument identifier.
            sentiment_score: The sentiment score from the news event.
            article_id: Article ID for logging.

        Returns:
            List of StopTightenResult for positions in this instrument.
        """
        results: list[StopTightenResult] = []

        # Get open positions for this instrument
        positions = await self._position_provider.get_open_positions_for_instrument(
            instrument
        )

        if not positions:
            return results

        # Get current price and ATR for the instrument
        current_price = await self._price_provider.get_current_price(instrument)
        if current_price is None:
            logger.warning(
                "Cannot tighten stops: current price unavailable for %s",
                instrument,
                extra={"article_id": article_id},
            )
            return results

        current_atr = await self._atr_provider.get_current_atr(instrument)
        if current_atr is None:
            logger.warning(
                "Cannot tighten stops: ATR unavailable for %s",
                instrument,
                extra={"article_id": article_id},
            )
            return results

        for pos_data in positions:
            result = await self._evaluate_position(
                pos_data=pos_data,
                sentiment_score=sentiment_score,
                current_price=current_price,
                current_atr=current_atr,
                article_id=article_id,
            )
            if result is not None:
                results.append(result)

        return results

    async def _evaluate_position(
        self,
        pos_data: dict[str, Any],
        sentiment_score: float,
        current_price: Decimal,
        current_atr: Decimal,
        article_id: str,
    ) -> StopTightenResult | None:
        """Evaluate whether a single position should have its stop tightened.

        Opposing sentiment logic:
        - LONG position + bearish sentiment (< -threshold): tighten
        - SHORT position + bullish sentiment (> +threshold): tighten
        - Otherwise: no change (aligned or neutral sentiment)

        Args:
            pos_data: Position data dict.
            sentiment_score: The sentiment score from the news event.
            current_price: Current market price for the instrument.
            current_atr: Current ATR value for the instrument.
            article_id: Article ID for logging.

        Returns:
            StopTightenResult if the position was evaluated, None if skipped.
        """
        position_id = pos_data.get("position_id", "unknown")
        instrument = pos_data.get("instrument", "unknown")
        direction_str = pos_data.get("direction", "").upper()
        current_stop = pos_data.get("current_stop")
        entry_price = pos_data.get("entry_price")
        initial_stop = pos_data.get("initial_stop")
        atr_at_entry = pos_data.get("atr_at_entry")

        if current_stop is None or entry_price is None:
            logger.warning(
                "Position %s missing required fields, skipping",
                position_id,
            )
            return None

        # Convert to Decimal if needed
        current_stop = Decimal(str(current_stop))
        entry_price = Decimal(str(entry_price))
        initial_stop = Decimal(str(initial_stop)) if initial_stop is not None else current_stop
        atr_at_entry = Decimal(str(atr_at_entry)) if atr_at_entry is not None else current_atr

        # Determine direction
        if direction_str == "LONG":
            direction = Direction.LONG
        elif direction_str == "SHORT":
            direction = Direction.SHORT
        else:
            logger.warning(
                "Position %s has unknown direction '%s', skipping",
                position_id,
                direction_str,
            )
            return None

        # Check if sentiment is opposing
        is_opposing = self._is_opposing_sentiment(direction, sentiment_score)

        if not is_opposing:
            return StopTightenResult(
                position_id=position_id,
                instrument=instrument,
                direction=direction_str,
                original_stop=current_stop,
                new_stop=current_stop,
                tightened=False,
                reason="Sentiment is not opposing (aligned or below threshold)",
            )

        # Build Position object for StopManager
        position = Position(
            entry_price=entry_price,
            direction=direction,
            initial_stop=initial_stop,
            current_stop=current_stop,
            atr_at_entry=atr_at_entry,
        )

        # Calculate tightened stop using StopManager
        new_stop = self._stop_manager.tighten_stop_on_news(
            position=position,
            current_price=current_price,
            atr=current_atr,
        )

        tightened = new_stop != current_stop

        if tightened:
            logger.info(
                "Tightening stop for position %s: %s → %s (sentiment=%.2f, ATR=%s)",
                position_id,
                current_stop,
                new_stop,
                sentiment_score,
                current_atr,
                extra={
                    "article_id": article_id,
                    "position_id": position_id,
                    "instrument": instrument,
                    "direction": direction_str,
                    "original_stop": str(current_stop),
                    "new_stop": str(new_stop),
                    "sentiment_score": sentiment_score,
                },
            )

            # Apply the stop update if callback is provided
            if self._stop_update_callback is not None:
                await self._stop_update_callback.update_stop_loss(
                    position_id=position_id,
                    new_stop=new_stop,
                )

        return StopTightenResult(
            position_id=position_id,
            instrument=instrument,
            direction=direction_str,
            original_stop=current_stop,
            new_stop=new_stop,
            tightened=tightened,
            reason=(
                f"Stop tightened due to opposing sentiment ({sentiment_score:.2f})"
                if tightened
                else "New stop would not be tighter than current stop"
            ),
        )

    def _is_opposing_sentiment(
        self, direction: Direction, sentiment_score: float
    ) -> bool:
        """Determine if the sentiment opposes the position direction.

        Opposing means:
        - LONG position + bearish sentiment (sentiment < -threshold)
        - SHORT position + bullish sentiment (sentiment > +threshold)

        Args:
            direction: Position direction.
            sentiment_score: Sentiment score in [-1.0, +1.0].

        Returns:
            True if sentiment opposes the position direction with magnitude
            exceeding the threshold.
        """
        if direction == Direction.LONG:
            return sentiment_score < -self._sentiment_threshold
        else:  # SHORT
            return sentiment_score > self._sentiment_threshold
