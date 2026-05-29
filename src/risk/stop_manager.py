"""Dynamic stop loss and take profit management.

Implements ATR-based initial stop loss calculation, configurable take profit
levels at R:R ratios, trailing stop logic with breakeven and step-based
advancement, risk-reward validation, and news/event-based stop adjustments.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 23.4, 23.13, 23.14
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from src.config.constants import (
    ATR_MULTIPLIER_DEFAULT,
    MIN_RISK_REWARD_RATIO,
)

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class StopLossResult:
    """Result of stop loss and take profit calculation for a new position.

    Attributes:
        stop_price: The calculated stop loss price.
        take_profit_levels: List of take profit price levels at configured R:R ratios.
    """

    stop_price: Decimal
    take_profit_levels: list[Decimal] = field(default_factory=list)


@dataclass
class Position:
    """Represents an open trading position for stop management purposes.

    Attributes:
        entry_price: The price at which the position was opened.
        direction: Whether the position is LONG or SHORT.
        initial_stop: The initial stop loss price set at entry.
        current_stop: The current (possibly trailed) stop loss price.
        atr_at_entry: The ATR value at the time of entry.
    """

    entry_price: Decimal
    direction: Direction
    initial_stop: Decimal
    current_stop: Decimal
    atr_at_entry: Decimal


class StopManager:
    """Manages dynamic stop loss and take profit levels for positions.

    Provides ATR-based initial stop calculation, configurable take profit
    levels, trailing stop logic with breakeven and step-based advancement,
    risk-reward validation, and news/event-based stop adjustments.
    """

    def calculate_initial_stop(
        self,
        entry_price: Decimal,
        direction: Direction,
        atr: Decimal,
        atr_multiplier: Decimal = Decimal(str(ATR_MULTIPLIER_DEFAULT)),
    ) -> Decimal:
        """Calculate the initial stop loss using ATR-based distance.

        Places the stop loss at (atr_multiplier * ATR) from the entry price
        on the adverse side.

        Args:
            entry_price: The entry price of the position.
            direction: Trade direction (LONG or SHORT).
            atr: The 14-period Average True Range value.
            atr_multiplier: Multiplier for ATR distance (default 1.5).

        Returns:
            The initial stop loss price.
        """
        stop_distance = atr * atr_multiplier
        if direction == Direction.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profits(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        direction: Direction,
        ratios: list[Decimal] | None = None,
    ) -> list[Decimal]:
        """Calculate take profit levels at configurable risk-to-reward ratios.

        Each ratio represents the reward multiple of the risk (stop distance).
        Maximum of 5 take profit levels allowed.

        Args:
            entry_price: The entry price of the position.
            stop_loss: The stop loss price.
            direction: Trade direction (LONG or SHORT).
            ratios: List of R:R ratios for take profit levels.
                Defaults to [2.0, 3.0]. Maximum 5 levels.

        Returns:
            List of take profit price levels.
        """
        if ratios is None:
            ratios = [Decimal("2.0"), Decimal("3.0")]

        # Enforce maximum of 5 take profit levels
        ratios = ratios[:5]

        risk_distance = abs(entry_price - stop_loss)

        take_profits: list[Decimal] = []
        for ratio in ratios:
            if direction == Direction.LONG:
                tp = entry_price + (risk_distance * ratio)
            else:
                tp = entry_price - (risk_distance * ratio)
            take_profits.append(tp)

        return take_profits

    def update_trailing_stop(
        self,
        position: Position,
        current_price: Decimal,
        atr: Decimal,
    ) -> Decimal | None:
        """Update the trailing stop based on current price movement.

        Trailing stop logic:
        - At 1R profit (price moved by stop distance from entry): move stop
          to breakeven (entry price).
        - Beyond 1R: for each 0.5*ATR move in profit direction beyond the
          last adjustment, advance stop by 0.5*ATR.
        - Stop never moves backward (only advances in profit direction).

        Args:
            position: The current position with entry, direction, and stops.
            current_price: The current market price.
            atr: The current ATR value for step calculation.

        Returns:
            The new stop loss price if it should be updated, or None if no
            change is needed (stop would move backward or conditions not met).
        """
        entry = position.entry_price
        initial_stop = position.initial_stop
        current_stop = position.current_stop
        direction = position.direction

        # Calculate the risk distance (1R)
        risk_distance = abs(entry - initial_stop)

        if risk_distance == Decimal("0"):
            return None

        # Calculate current profit in price terms
        if direction == Direction.LONG:
            profit = current_price - entry
        else:
            profit = entry - current_price

        # If not yet at 1R profit, no trailing
        if profit < risk_distance:
            return None

        # Step size for trailing
        step = atr * Decimal("0.5")

        if step == Decimal("0"):
            return None

        # At 1R: move to breakeven (entry price)
        # Beyond 1R: advance by 0.5*ATR for each 0.5*ATR move beyond 1R
        profit_beyond_1r = profit - risk_distance

        # Number of complete steps beyond 1R
        steps_beyond = int(profit_beyond_1r / step)

        # New stop = entry + steps_beyond * step (for LONG)
        # New stop = entry - steps_beyond * step (for SHORT)
        if direction == Direction.LONG:
            new_stop = entry + (step * steps_beyond)
        else:
            new_stop = entry - (step * steps_beyond)

        # Stop must never move backward
        if direction == Direction.LONG:
            if new_stop <= current_stop:
                return None
            return new_stop
        else:
            if new_stop >= current_stop:
                return None
            return new_stop

    def validate_risk_reward(
        self,
        entry: Decimal,
        stop: Decimal,
        target: Decimal,
        min_rr: Decimal = Decimal(str(MIN_RISK_REWARD_RATIO)),
    ) -> bool:
        """Validate that a trade meets the minimum risk-to-reward ratio.

        Formula: RR = |target - entry| / |entry - stop|

        Args:
            entry: The entry price.
            stop: The stop loss price.
            target: The take profit target price.
            min_rr: Minimum acceptable risk-to-reward ratio (default 1.5).

        Returns:
            True if the trade meets the minimum RR requirement, False otherwise.
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)

        if risk == Decimal("0"):
            return False

        rr_ratio = reward / risk
        return rr_ratio >= min_rr

    def is_sentiment_aligned(
        self,
        position: Position,
        sentiment_score: float,
    ) -> bool:
        """Check if a news sentiment score is aligned with the position direction.

        Aligned sentiment means the news is favorable for the position:
        - For LONG positions: sentiment > 0 (bullish) is aligned.
        - For SHORT positions: sentiment < 0 (bearish) is aligned.

        Args:
            position: The current position with direction.
            sentiment_score: The sentiment score from the news article [-1.0, +1.0].

        Returns:
            True if sentiment is aligned with position direction, False otherwise.
        """
        if position.direction == Direction.LONG and sentiment_score > 0:
            return True
        if position.direction == Direction.SHORT and sentiment_score < 0:
            return True
        return False

    def maintain_position_on_aligned_sentiment(
        self,
        position: Position,
        sentiment_score: float,
        instrument: str = "",
    ) -> bool:
        """Determine if a position should be maintained due to aligned sentiment.

        When a HIGH-impact news article has sentiment that ALIGNS with the
        position direction (bullish for longs, bearish for shorts), no changes
        are made to stops or position. This method checks alignment and logs
        the decision.

        This is the complement of tighten_stop_on_news (opposing sentiment).

        Args:
            position: The current position with direction.
            sentiment_score: The sentiment score from the news article [-1.0, +1.0].
            instrument: Optional instrument identifier for logging.

        Returns:
            True if sentiment is aligned and position should be maintained
            unchanged. False if sentiment is not aligned (caller should
            consider tightening or other adjustments).

        Validates: Requirement 23.13
        """
        if self.is_sentiment_aligned(position, sentiment_score):
            direction_label = position.direction.value
            sentiment_label = "bullish" if sentiment_score > 0 else "bearish"
            logger.info(
                "Position maintained due to aligned sentiment: "
                "direction=%s, sentiment=%s (score=%.3f), instrument=%s. "
                "No stop or position changes applied.",
                direction_label,
                sentiment_label,
                sentiment_score,
                instrument or "unknown",
            )
            return True
        return False

    def tighten_stop_on_news(
        self,
        position: Position,
        current_price: Decimal,
        atr: Decimal,
    ) -> Decimal:
        """Tighten stop loss when opposing high-impact news is detected.

        Moves the stop to 0.5 * ATR from the current price. The new stop
        is only applied if it is tighter (closer to current price) than the
        existing stop. For LONG positions, the stop never moves down.

        Args:
            position: The current position.
            current_price: The current market price.
            atr: The current ATR value.

        Returns:
            The new tightened stop loss price.
        """
        tightened_distance = atr * Decimal("0.5")

        if position.direction == Direction.LONG:
            new_stop = current_price - tightened_distance
            # Stop must never move backward for LONG (only up)
            if new_stop < position.current_stop:
                return position.current_stop
            return new_stop
        else:
            new_stop = current_price + tightened_distance
            # Stop must never move backward for SHORT (only down)
            if new_stop > position.current_stop:
                return position.current_stop
            return new_stop

    def widen_stop_for_event(
        self,
        position: Position,
        atr: Decimal,
        multiplier: Decimal = Decimal("1.0"),
    ) -> Decimal:
        """Widen stop loss for scheduled high-impact economic events.

        Moves the stop further from the current stop by (multiplier * ATR)
        to accommodate expected volatility around events.

        Args:
            position: The current position.
            atr: The current ATR value.
            multiplier: ATR multiplier for widening distance (default 1.0).

        Returns:
            The new widened stop loss price.
        """
        widen_distance = multiplier * atr

        if position.direction == Direction.LONG:
            # For LONG, widen means moving stop further down (away from price)
            return position.current_stop - widen_distance
        else:
            # For SHORT, widen means moving stop further up (away from price)
            return position.current_stop + widen_distance
