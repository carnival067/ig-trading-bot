"""Order lifecycle management for the Institutional AI Trading System.

Implements order creation, submission, fill/reject handling, trailing stops,
partial take profit, and failure handling with retry logic.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from src.core.exceptions import (
    OrderExecutionError,
    OrderTimeoutError,
    OrderValidationError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_SIZE = Decimal("0.01")
"""Default minimum order size if not provided by instrument info."""

ORDER_RETRY_DELAY_SECONDS = 1.0
"""Delay in seconds before retrying a failed order."""

PARTIAL_TP_MIN_PCT = Decimal("0.25")
"""Minimum percentage for partial take profit."""

PARTIAL_TP_MAX_PCT = Decimal("0.75")
"""Maximum percentage for partial take profit."""

PARTIAL_TP_DEFAULT_PCT = Decimal("0.50")
"""Default percentage for partial take profit."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderType(Enum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Order lifecycle states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    FAILED = "failed"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """Represents a trading order through its lifecycle."""

    id: str
    instrument: str
    direction: str  # "BUY" or "SELL"
    order_type: OrderType
    size: Decimal
    price: Decimal | None = None  # None for market orders
    stop_distance: Decimal | None = None
    limit_distance: Decimal | None = None
    trail_distance: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    deal_reference: str | None = None
    deal_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: datetime | None = None
    closed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    entry_price: Decimal | None = None


@dataclass
class TrailingStopConfig:
    """Configuration for an active trailing stop."""

    deal_id: str
    instrument: str
    direction: str  # "BUY" or "SELL"
    trail_distance: Decimal
    current_stop: Decimal
    highest_price: Decimal  # For BUY: highest since entry
    lowest_price: Decimal  # For SELL: lowest since entry


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class IGClientProtocol(Protocol):
    """Protocol defining the IG API client interface used by OrderManager."""

    async def place_order(
        self,
        instrument: str,
        direction: str,
        size: Decimal,
        order_type: str,
        price: Decimal | None = None,
        stop_distance: Decimal | None = None,
        limit_distance: Decimal | None = None,
    ) -> dict[str, Any]: ...

    async def close_position(
        self,
        deal_id: str,
        direction: str,
        size: Decimal,
    ) -> dict[str, Any]: ...

    async def update_position(
        self,
        deal_id: str,
        stop_level: Decimal | None = None,
        limit_level: Decimal | None = None,
    ) -> dict[str, Any]: ...

    async def get_market_info(self, instrument: str) -> dict[str, Any]: ...

    async def get_position(self, deal_id: str) -> dict[str, Any]: ...


class NotificationServiceProtocol(Protocol):
    """Protocol for notification service."""

    async def send_alert(self, message: str, level: str = "info") -> None: ...


class EventBusProtocol(Protocol):
    """Protocol for event bus publishing."""

    async def publish(self, channel: str, event: Any) -> int: ...


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class OrderManager:
    """Manages order lifecycle: create → submit → fill/reject → close.

    Handles Market, Limit, Stop, and Trailing Stop orders with validation,
    partial take profit, and failure handling with retry logic.
    """

    def __init__(
        self,
        ig_client: IGClientProtocol,
        event_bus: EventBusProtocol | None = None,
        notification_service: NotificationServiceProtocol | None = None,
    ) -> None:
        self._ig_client = ig_client
        self._event_bus = event_bus
        self._notification_service = notification_service
        self._orders: dict[str, Order] = {}
        self._active_trailing_stops: dict[str, TrailingStopConfig] = {}

    @property
    def orders(self) -> dict[str, Order]:
        """Access all tracked orders."""
        return self._orders

    @property
    def active_trailing_stops(self) -> dict[str, TrailingStopConfig]:
        """Access active trailing stop configurations."""
        return self._active_trailing_stops

    # ------------------------------------------------------------------
    # Order Placement (Task 12.1, 12.2)
    # ------------------------------------------------------------------

    async def place_market_order(
        self,
        instrument: str,
        direction: str,
        size: Decimal,
        stop_distance: Decimal | None = None,
        limit_distance: Decimal | None = None,
    ) -> Order:
        """Validate and place a market order.

        Args:
            instrument: The instrument epic (e.g., "CS.D.EURUSD.CFD.IP").
            direction: "BUY" or "SELL".
            size: Order size in lots.
            stop_distance: Optional stop distance in points.
            limit_distance: Optional limit distance in points.

        Returns:
            The Order object with updated status.

        Raises:
            OrderValidationError: If validation fails.
            OrderExecutionError: If execution fails after retry.
        """
        order = Order(
            id=self._generate_order_id(),
            instrument=instrument,
            direction=direction,
            order_type=OrderType.MARKET,
            size=size,
            stop_distance=stop_distance,
            limit_distance=limit_distance,
        )
        self._orders[order.id] = order
        self._log_transition(order, OrderStatus.PENDING)

        valid, error_msg = await self._validate_order(instrument, size)
        if not valid:
            order.status = OrderStatus.REJECTED
            order.error_message = error_msg
            self._log_transition(order, OrderStatus.REJECTED)
            await self._notify(
                f"Order rejected for {instrument}: {error_msg}", level="warning"
            )
            raise OrderValidationError(error_msg or "Validation failed", instrument=instrument)

        return await self._submit_order(order)

    async def place_limit_order(
        self,
        instrument: str,
        direction: str,
        size: Decimal,
        price: Decimal,
        stop_distance: Decimal | None = None,
        limit_distance: Decimal | None = None,
    ) -> Order:
        """Validate and place a limit order.

        Args:
            instrument: The instrument epic.
            direction: "BUY" or "SELL".
            size: Order size in lots.
            price: Limit price level.
            stop_distance: Optional stop distance in points.
            limit_distance: Optional limit distance in points.

        Returns:
            The Order object with updated status.

        Raises:
            OrderValidationError: If validation fails.
            OrderExecutionError: If execution fails after retry.
        """
        order = Order(
            id=self._generate_order_id(),
            instrument=instrument,
            direction=direction,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            stop_distance=stop_distance,
            limit_distance=limit_distance,
        )
        self._orders[order.id] = order
        self._log_transition(order, OrderStatus.PENDING)

        valid, error_msg = await self._validate_order(instrument, size)
        if not valid:
            order.status = OrderStatus.REJECTED
            order.error_message = error_msg
            self._log_transition(order, OrderStatus.REJECTED)
            await self._notify(
                f"Order rejected for {instrument}: {error_msg}", level="warning"
            )
            raise OrderValidationError(error_msg or "Validation failed", instrument=instrument)

        return await self._submit_order(order)

    async def place_stop_order(
        self,
        instrument: str,
        direction: str,
        size: Decimal,
        price: Decimal,
        stop_distance: Decimal | None = None,
        limit_distance: Decimal | None = None,
    ) -> Order:
        """Validate and place a stop order.

        Args:
            instrument: The instrument epic.
            direction: "BUY" or "SELL".
            size: Order size in lots.
            price: Stop trigger price.
            stop_distance: Optional stop distance in points.
            limit_distance: Optional limit distance in points.

        Returns:
            The Order object with updated status.

        Raises:
            OrderValidationError: If validation fails.
            OrderExecutionError: If execution fails after retry.
        """
        order = Order(
            id=self._generate_order_id(),
            instrument=instrument,
            direction=direction,
            order_type=OrderType.STOP,
            size=size,
            price=price,
            stop_distance=stop_distance,
            limit_distance=limit_distance,
        )
        self._orders[order.id] = order
        self._log_transition(order, OrderStatus.PENDING)

        valid, error_msg = await self._validate_order(instrument, size)
        if not valid:
            order.status = OrderStatus.REJECTED
            order.error_message = error_msg
            self._log_transition(order, OrderStatus.REJECTED)
            await self._notify(
                f"Order rejected for {instrument}: {error_msg}", level="warning"
            )
            raise OrderValidationError(error_msg or "Validation failed", instrument=instrument)

        return await self._submit_order(order)

    # ------------------------------------------------------------------
    # Trailing Stop (Task 12.3)
    # ------------------------------------------------------------------

    async def setup_trailing_stop(
        self,
        deal_id: str,
        trail_distance: Decimal,
        direction: str,
        instrument: str,
        current_price: Decimal,
        current_stop: Decimal,
    ) -> None:
        """Set up a trailing stop for an existing position.

        Monitors price and adjusts stop by trail_distance on favorable moves.

        Args:
            deal_id: The IG deal ID for the position.
            trail_distance: Distance in points to trail behind price.
            direction: "BUY" or "SELL".
            instrument: The instrument epic.
            current_price: Current market price.
            current_stop: Current stop level.
        """
        if trail_distance <= 0:
            raise OrderValidationError(
                "Trail distance must be positive",
                deal_id=deal_id,
                trail_distance=str(trail_distance),
            )

        config = TrailingStopConfig(
            deal_id=deal_id,
            instrument=instrument,
            direction=direction,
            trail_distance=trail_distance,
            current_stop=current_stop,
            highest_price=current_price if direction == "BUY" else Decimal("0"),
            lowest_price=current_price if direction == "SELL" else Decimal("999999"),
        )
        self._active_trailing_stops[deal_id] = config
        logger.info(
            "Trailing stop configured",
            extra={
                "deal_id": deal_id,
                "trail_distance": str(trail_distance),
                "direction": direction,
                "current_stop": str(current_stop),
            },
        )

    async def _update_trailing_stops(
        self, instrument: str, current_price: Decimal
    ) -> None:
        """Called on each tick to check and adjust trailing stops.

        For BUY positions: if price moves up by trail_distance from the
        highest recorded price, advance the stop upward.
        For SELL positions: if price moves down by trail_distance from the
        lowest recorded price, advance the stop downward.

        The stop NEVER moves backward (against the profitable direction).

        Args:
            instrument: The instrument that received a new tick.
            current_price: The current market price.
        """
        for deal_id, config in list(self._active_trailing_stops.items()):
            if config.instrument != instrument:
                continue

            if config.direction == "BUY":
                # Track highest price
                if current_price > config.highest_price:
                    config.highest_price = current_price

                # Calculate new stop: highest_price - trail_distance
                new_stop = config.highest_price - config.trail_distance

                # Stop only moves up (never backward)
                if new_stop > config.current_stop:
                    old_stop = config.current_stop
                    config.current_stop = new_stop
                    await self._apply_trailing_stop_update(deal_id, new_stop)
                    logger.info(
                        "Trailing stop advanced (BUY)",
                        extra={
                            "deal_id": deal_id,
                            "old_stop": str(old_stop),
                            "new_stop": str(new_stop),
                            "highest_price": str(config.highest_price),
                        },
                    )

            elif config.direction == "SELL":
                # Track lowest price
                if current_price < config.lowest_price:
                    config.lowest_price = current_price

                # Calculate new stop: lowest_price + trail_distance
                new_stop = config.lowest_price + config.trail_distance

                # Stop only moves down (never backward)
                if new_stop < config.current_stop:
                    old_stop = config.current_stop
                    config.current_stop = new_stop
                    await self._apply_trailing_stop_update(deal_id, new_stop)
                    logger.info(
                        "Trailing stop advanced (SELL)",
                        extra={
                            "deal_id": deal_id,
                            "old_stop": str(old_stop),
                            "new_stop": str(new_stop),
                            "lowest_price": str(config.lowest_price),
                        },
                    )

    async def _apply_trailing_stop_update(
        self, deal_id: str, new_stop: Decimal
    ) -> None:
        """Apply a trailing stop update to the IG platform."""
        try:
            await self._ig_client.update_position(deal_id=deal_id, stop_level=new_stop)
        except Exception as e:
            logger.error(
                "Failed to update trailing stop on IG",
                extra={"deal_id": deal_id, "new_stop": str(new_stop), "error": str(e)},
            )

    def remove_trailing_stop(self, deal_id: str) -> None:
        """Remove a trailing stop configuration (e.g., when position is closed)."""
        self._active_trailing_stops.pop(deal_id, None)

    # ------------------------------------------------------------------
    # Partial Take Profit (Task 12.4)
    # ------------------------------------------------------------------

    async def partial_take_profit(
        self,
        deal_id: str,
        close_pct: Decimal = PARTIAL_TP_DEFAULT_PCT,
    ) -> Order:
        """Close a percentage of a position and move remaining stop to breakeven.

        Closes close_pct of the position size and adjusts the stop loss on
        the remaining position to the entry price inclusive of spread.

        Args:
            deal_id: The IG deal ID for the position.
            close_pct: Percentage to close (0.25 to 0.75, default 0.50).

        Returns:
            The Order representing the partial close.

        Raises:
            OrderValidationError: If close_pct is out of range.
            OrderExecutionError: If the partial close fails.
        """
        # Validate close percentage
        if close_pct < PARTIAL_TP_MIN_PCT or close_pct > PARTIAL_TP_MAX_PCT:
            raise OrderValidationError(
                f"Close percentage must be between {PARTIAL_TP_MIN_PCT} and {PARTIAL_TP_MAX_PCT}",
                deal_id=deal_id,
                close_pct=str(close_pct),
            )

        # Get position details from IG
        position_info = await self._ig_client.get_position(deal_id)
        direction = position_info.get("direction", "BUY")
        instrument = position_info.get("instrument", "")
        total_size = Decimal(str(position_info.get("size", "0")))
        entry_price = Decimal(str(position_info.get("entry_price", "0")))
        spread = Decimal(str(position_info.get("spread", "0")))

        # Calculate close size
        close_size = (total_size * close_pct).quantize(Decimal("0.01"))
        if close_size <= 0:
            raise OrderValidationError(
                "Calculated close size is zero",
                deal_id=deal_id,
                total_size=str(total_size),
                close_pct=str(close_pct),
            )

        # Close the partial position
        close_direction = "SELL" if direction == "BUY" else "BUY"
        close_order = await self.close_order(deal_id, close_direction, close_size)

        # Move remaining stop to breakeven (entry price inclusive of spread)
        if direction == "BUY":
            breakeven_stop = entry_price - spread
        else:
            breakeven_stop = entry_price + spread

        try:
            await self._ig_client.update_position(
                deal_id=deal_id, stop_level=breakeven_stop
            )
            logger.info(
                "Moved stop to breakeven after partial TP",
                extra={
                    "deal_id": deal_id,
                    "breakeven_stop": str(breakeven_stop),
                    "entry_price": str(entry_price),
                    "spread": str(spread),
                    "closed_size": str(close_size),
                    "remaining_size": str(total_size - close_size),
                },
            )
        except Exception as e:
            logger.error(
                "Failed to move stop to breakeven",
                extra={"deal_id": deal_id, "error": str(e)},
            )

        # Update trailing stop config if one exists
        if deal_id in self._active_trailing_stops:
            self._active_trailing_stops[deal_id].current_stop = breakeven_stop

        # Mark original order as partially closed
        for order in self._orders.values():
            if order.deal_id == deal_id and order.status == OrderStatus.FILLED:
                order.status = OrderStatus.PARTIALLY_CLOSED
                self._log_transition(order, OrderStatus.PARTIALLY_CLOSED)
                break

        return close_order

    # ------------------------------------------------------------------
    # Close Order (Task 12.1)
    # ------------------------------------------------------------------

    async def close_order(
        self, deal_id: str, direction: str, size: Decimal
    ) -> Order:
        """Close a position (full or partial).

        Args:
            deal_id: The IG deal ID to close.
            direction: Close direction ("SELL" to close a BUY, "BUY" to close a SELL).
            size: Size to close.

        Returns:
            The Order representing the close operation.

        Raises:
            OrderExecutionError: If the close fails after retry.
        """
        order = Order(
            id=self._generate_order_id(),
            instrument="",  # Will be populated from response
            direction=direction,
            order_type=OrderType.MARKET,
            size=size,
            deal_id=deal_id,
        )
        self._orders[order.id] = order
        self._log_transition(order, OrderStatus.PENDING)

        try:
            order.status = OrderStatus.SUBMITTED
            self._log_transition(order, OrderStatus.SUBMITTED)

            response = await self._ig_client.close_position(
                deal_id=deal_id, direction=direction, size=size
            )

            order.status = OrderStatus.CLOSED
            order.deal_reference = response.get("dealReference")
            order.closed_at = datetime.now(timezone.utc)
            self._log_transition(order, OrderStatus.CLOSED)

            # Remove trailing stop if position fully closed
            self.remove_trailing_stop(deal_id)

            await self._publish_event("order.closed", {
                "order_id": order.id,
                "deal_id": deal_id,
                "direction": direction,
                "size": str(size),
            })

            return order

        except Exception as e:
            return await self._handle_failure(order, str(e))

    # ------------------------------------------------------------------
    # Validation (Task 12.2)
    # ------------------------------------------------------------------

    async def _validate_order(
        self, instrument: str, size: Decimal
    ) -> tuple[bool, str | None]:
        """Validate an order before submission.

        Checks:
        1. Instrument is active/valid on IG platform.
        2. Size meets minimum tradeable size.
        3. Sufficient margin is available.

        Args:
            instrument: The instrument epic to validate.
            size: The order size to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        try:
            market_info = await self._ig_client.get_market_info(instrument)
        except Exception as e:
            logger.error(
                "Failed to retrieve market info",
                extra={"instrument": instrument, "error": str(e)},
            )
            return False, f"Failed to retrieve market info: {e}"

        # Check instrument is active
        status = market_info.get("status", "")
        if status == "":
            logger.debug(
                "Market info response structure",
                extra={"instrument": instrument, "market_info_keys": list(market_info.keys())},
            )
        if status != "TRADEABLE":
            return False, f"Instrument {instrument} is not active (status: {status})"

        # Check minimum size
        min_size = Decimal(str(market_info.get("min_size", DEFAULT_MIN_SIZE)))
        if size < min_size:
            return False, (
                f"Order size {size} is below minimum {min_size} for {instrument}"
            )

        # Check margin availability
        margin_available = market_info.get("margin_available")
        margin_required = market_info.get("margin_required")
        if margin_available is not None and margin_required is not None:
            if Decimal(str(margin_available)) < Decimal(str(margin_required)):
                return False, (
                    f"Insufficient margin: available={margin_available}, "
                    f"required={margin_required}"
                )

        return True, None

    # ------------------------------------------------------------------
    # Failure Handling (Task 12.5)
    # ------------------------------------------------------------------

    async def _handle_failure(self, order: Order, error: str) -> Order:
        """Handle order execution failure with retry logic.

        1. Log the failure reason.
        2. Notify the notification service.
        3. Retry once after 1-second delay.
        4. If retry fails, mark order as FAILED.

        Args:
            order: The order that failed.
            error: The error message.

        Returns:
            The order with updated status (FILLED if retry succeeds, FAILED otherwise).
        """
        logger.error(
            "Order execution failed",
            extra={
                "order_id": order.id,
                "instrument": order.instrument,
                "error": error,
                "retry_count": order.retry_count,
            },
        )

        await self._notify(
            f"Order failed for {order.instrument}: {error}", level="error"
        )

        # Retry once if this is the first failure
        if order.retry_count < 1:
            order.retry_count += 1
            logger.info(
                "Retrying order after 1s delay",
                extra={"order_id": order.id, "retry_count": order.retry_count},
            )
            await asyncio.sleep(ORDER_RETRY_DELAY_SECONDS)

            try:
                response = await self._execute_on_ig(order)
                order.status = OrderStatus.FILLED
                order.deal_reference = response.get("dealReference")
                order.deal_id = response.get("dealId")
                order.filled_at = datetime.now(timezone.utc)
                order.entry_price = (
                    Decimal(str(response["entryPrice"]))
                    if "entryPrice" in response
                    else None
                )
                self._log_transition(order, OrderStatus.FILLED)

                await self._publish_event("order.filled", {
                    "order_id": order.id,
                    "deal_id": order.deal_id,
                    "instrument": order.instrument,
                })

                return order

            except Exception as retry_error:
                logger.error(
                    "Order retry also failed",
                    extra={
                        "order_id": order.id,
                        "error": str(retry_error),
                    },
                )
                await self._notify(
                    f"Order retry failed for {order.instrument}: {retry_error}",
                    level="error",
                )

        # Mark as failed
        order.status = OrderStatus.FAILED
        order.error_message = error
        self._log_transition(order, OrderStatus.FAILED)

        await self._publish_event("order.rejected", {
            "order_id": order.id,
            "instrument": order.instrument,
            "error": error,
        })

        raise OrderExecutionError(
            f"Order failed after retry: {error}",
            order_id=order.id,
            instrument=order.instrument,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _submit_order(self, order: Order) -> Order:
        """Submit a validated order to the IG platform."""
        order.status = OrderStatus.SUBMITTED
        self._log_transition(order, OrderStatus.SUBMITTED)

        try:
            response = await self._execute_on_ig(order)

            order.status = OrderStatus.FILLED
            order.deal_reference = response.get("dealReference")
            order.deal_id = response.get("dealId")
            order.filled_at = datetime.now(timezone.utc)
            order.entry_price = (
                Decimal(str(response["entryPrice"]))
                if "entryPrice" in response
                else None
            )
            self._log_transition(order, OrderStatus.FILLED)

            await self._publish_event("order.filled", {
                "order_id": order.id,
                "deal_id": order.deal_id,
                "instrument": order.instrument,
                "direction": order.direction,
                "size": str(order.size),
                "order_type": order.order_type.value,
            })

            return order

        except Exception as e:
            return await self._handle_failure(order, str(e))

    async def _execute_on_ig(self, order: Order) -> dict[str, Any]:
        """Execute an order on the IG platform.

        Maps the internal Order to the IG API call format.
        """
        return await self._ig_client.place_order(
            instrument=order.instrument,
            direction=order.direction,
            size=order.size,
            order_type=order.order_type.value,
            price=order.price,
            stop_distance=order.stop_distance,
            limit_distance=order.limit_distance,
        )

    async def _notify(self, message: str, level: str = "info") -> None:
        """Send a notification if the notification service is available."""
        if self._notification_service is not None:
            try:
                await self._notification_service.send_alert(message, level=level)
            except Exception as e:
                logger.warning(
                    "Failed to send notification",
                    extra={"error": str(e), "message": message},
                )

    async def _publish_event(self, channel: str, payload: dict[str, Any]) -> None:
        """Publish an event to the event bus if available."""
        if self._event_bus is not None:
            try:
                from src.core.event_bus import Event

                event = Event(event_type=channel, payload=payload)
                await self._event_bus.publish(channel, event)
            except Exception as e:
                logger.warning(
                    "Failed to publish event",
                    extra={"channel": channel, "error": str(e)},
                )

    def _log_transition(self, order: Order, new_status: OrderStatus) -> None:
        """Log an order state transition."""
        logger.info(
            "Order state transition",
            extra={
                "order_id": order.id,
                "instrument": order.instrument,
                "direction": order.direction,
                "order_type": order.order_type.value,
                "size": str(order.size),
                "status": new_status.value,
            },
        )

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve an order by its ID.

        Args:
            order_id: The unique order identifier.

        Returns:
            The Order if found, None otherwise.
        """
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[Order]:
        """Retrieve all orders that are currently open (PENDING or SUBMITTED).

        Returns:
            List of orders with PENDING or SUBMITTED status.
        """
        return [
            order
            for order in self._orders.values()
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        ]

    @staticmethod
    def _generate_order_id() -> str:
        """Generate a unique order ID."""
        return f"ORD-{uuid.uuid4().hex[:12]}"
