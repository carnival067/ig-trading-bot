"""Unit tests for the OrderManager module.

Tests cover order lifecycle management, Market/Limit/Stop order types,
trailing stop adjustment, partial take profit, and failure handling.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.order_manager import (
    DEFAULT_MIN_SIZE,
    ORDER_RETRY_DELAY_SECONDS,
    PARTIAL_TP_DEFAULT_PCT,
    PARTIAL_TP_MAX_PCT,
    PARTIAL_TP_MIN_PCT,
    Order,
    OrderManager,
    OrderStatus,
    OrderType,
    TrailingStopConfig,
)
from src.core.exceptions import (
    OrderExecutionError,
    OrderValidationError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ig_client() -> AsyncMock:
    """Create a mock IG client with default successful responses."""
    client = AsyncMock()
    client.place_order.return_value = {
        "dealReference": "REF123",
        "dealId": "DEAL456",
        "entryPrice": "1.1234",
    }
    client.close_position.return_value = {
        "dealReference": "CLOSE_REF",
    }
    client.update_position.return_value = {"status": "SUCCESS"}
    client.get_market_info.return_value = {
        "status": "TRADEABLE",
        "min_size": "0.01",
        "margin_available": "10000",
        "margin_required": "100",
    }
    client.get_position.return_value = {
        "direction": "BUY",
        "instrument": "CS.D.EURUSD.CFD.IP",
        "size": "1.0",
        "entry_price": "1.1234",
        "spread": "0.0002",
    }
    return client


@pytest.fixture
def mock_notification_service() -> AsyncMock:
    """Create a mock notification service."""
    return AsyncMock()


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish.return_value = 1
    return bus


@pytest.fixture
def order_manager(
    mock_ig_client: AsyncMock,
    mock_event_bus: AsyncMock,
    mock_notification_service: AsyncMock,
) -> OrderManager:
    """Create an OrderManager with all mocked dependencies."""
    return OrderManager(
        ig_client=mock_ig_client,
        event_bus=mock_event_bus,
        notification_service=mock_notification_service,
    )


# =============================================================================
# Task 12.1: Order Lifecycle Management
# =============================================================================


class TestOrderLifecycle:
    """Tests for order lifecycle: create → submit → fill/reject → close."""

    @pytest.mark.asyncio
    async def test_market_order_lifecycle_pending_to_filled(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        assert order.status == OrderStatus.FILLED
        assert order.deal_reference == "REF123"
        assert order.deal_id == "DEAL456"
        assert order.filled_at is not None

    @pytest.mark.asyncio
    async def test_order_tracked_in_orders_dict(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        assert order.id in order_manager.orders
        assert order_manager.orders[order.id] is order

    @pytest.mark.asyncio
    async def test_close_order_transitions_to_closed(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.close_order(
            deal_id="DEAL456",
            direction="SELL",
            size=Decimal("1.0"),
        )
        assert order.status == OrderStatus.CLOSED
        assert order.closed_at is not None
        assert order.deal_reference == "CLOSE_REF"

    @pytest.mark.asyncio
    async def test_order_id_is_unique(
        self, order_manager: OrderManager
    ) -> None:
        order1 = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        order2 = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="SELL",
            size=Decimal("0.5"),
        )
        assert order1.id != order2.id


# =============================================================================
# Task 12.2: Market, Limit, Stop Orders with Validation
# =============================================================================


class TestOrderTypes:
    """Tests for Market, Limit, and Stop order types."""

    @pytest.mark.asyncio
    async def test_market_order_no_price(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        assert order.order_type == OrderType.MARKET
        assert order.price is None

    @pytest.mark.asyncio
    async def test_limit_order_has_price(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_limit_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
            price=Decimal("1.1200"),
        )
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("1.1200")

    @pytest.mark.asyncio
    async def test_stop_order_has_price(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_stop_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
            price=Decimal("1.1300"),
        )
        assert order.order_type == OrderType.STOP
        assert order.price == Decimal("1.1300")

    @pytest.mark.asyncio
    async def test_order_with_stop_and_limit_distance(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
            stop_distance=Decimal("20"),
            limit_distance=Decimal("40"),
        )
        assert order.stop_distance == Decimal("20")
        assert order.limit_distance == Decimal("40")


class TestOrderValidation:
    """Tests for order validation (instrument active, min size, margin)."""

    @pytest.mark.asyncio
    async def test_rejects_inactive_instrument(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_market_info.return_value = {
            "status": "CLOSED",
            "min_size": "0.01",
        }
        with pytest.raises(OrderValidationError, match="not active"):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )

    @pytest.mark.asyncio
    async def test_rejects_size_below_minimum(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_market_info.return_value = {
            "status": "TRADEABLE",
            "min_size": "0.5",
            "margin_available": "10000",
            "margin_required": "100",
        }
        with pytest.raises(OrderValidationError, match="below minimum"):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("0.1"),
            )

    @pytest.mark.asyncio
    async def test_rejects_insufficient_margin(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_market_info.return_value = {
            "status": "TRADEABLE",
            "min_size": "0.01",
            "margin_available": "50",
            "margin_required": "100",
        }
        with pytest.raises(OrderValidationError, match="Insufficient margin"):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )

    @pytest.mark.asyncio
    async def test_rejects_when_market_info_fails(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_market_info.side_effect = Exception("API error")
        with pytest.raises(OrderValidationError, match="Failed to retrieve"):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )

    @pytest.mark.asyncio
    async def test_rejected_order_status(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_market_info.return_value = {
            "status": "CLOSED",
            "min_size": "0.01",
        }
        with pytest.raises(OrderValidationError):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )
        # The order should be tracked with REJECTED status
        orders = list(order_manager.orders.values())
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_notification_sent_on_rejection(
        self, order_manager: OrderManager,
        mock_ig_client: AsyncMock,
        mock_notification_service: AsyncMock,
    ) -> None:
        mock_ig_client.get_market_info.return_value = {
            "status": "CLOSED",
            "min_size": "0.01",
        }
        with pytest.raises(OrderValidationError):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )
        mock_notification_service.send_alert.assert_called_once()


# =============================================================================
# Task 12.3: Trailing Stop
# =============================================================================


class TestTrailingStop:
    """Tests for trailing stop price monitoring and adjustment."""

    @pytest.mark.asyncio
    async def test_setup_trailing_stop(
        self, order_manager: OrderManager
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL1",
            trail_distance=Decimal("10"),
            direction="BUY",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("1.1234"),
            current_stop=Decimal("1.1134"),
        )
        assert "DEAL1" in order_manager.active_trailing_stops
        config = order_manager.active_trailing_stops["DEAL1"]
        assert config.trail_distance == Decimal("10")
        assert config.direction == "BUY"

    @pytest.mark.asyncio
    async def test_trailing_stop_rejects_zero_distance(
        self, order_manager: OrderManager
    ) -> None:
        with pytest.raises(OrderValidationError, match="positive"):
            await order_manager.setup_trailing_stop(
                deal_id="DEAL1",
                trail_distance=Decimal("0"),
                direction="BUY",
                instrument="CS.D.EURUSD.CFD.IP",
                current_price=Decimal("1.1234"),
                current_stop=Decimal("1.1134"),
            )

    @pytest.mark.asyncio
    async def test_trailing_stop_advances_on_favorable_move_buy(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL1",
            trail_distance=Decimal("10"),
            direction="BUY",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("90"),
        )
        # Price moves up to 115 → new stop = 115 - 10 = 105 > 90
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("115")
        )
        config = order_manager.active_trailing_stops["DEAL1"]
        assert config.current_stop == Decimal("105")
        mock_ig_client.update_position.assert_called_with(
            deal_id="DEAL1", stop_level=Decimal("105")
        )

    @pytest.mark.asyncio
    async def test_trailing_stop_never_moves_backward_buy(
        self, order_manager: OrderManager
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL1",
            trail_distance=Decimal("10"),
            direction="BUY",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("90"),
        )
        # Price goes up to 115 → stop = 105
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("115")
        )
        # Price drops to 108 → stop should NOT move backward
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("108")
        )
        config = order_manager.active_trailing_stops["DEAL1"]
        assert config.current_stop == Decimal("105")

    @pytest.mark.asyncio
    async def test_trailing_stop_advances_on_favorable_move_sell(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL2",
            trail_distance=Decimal("10"),
            direction="SELL",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("110"),
        )
        # Price moves down to 85 → new stop = 85 + 10 = 95 < 110
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("85")
        )
        config = order_manager.active_trailing_stops["DEAL2"]
        assert config.current_stop == Decimal("95")

    @pytest.mark.asyncio
    async def test_trailing_stop_never_moves_backward_sell(
        self, order_manager: OrderManager
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL2",
            trail_distance=Decimal("10"),
            direction="SELL",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("110"),
        )
        # Price drops to 85 → stop = 95
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("85")
        )
        # Price bounces to 92 → stop should NOT move up
        await order_manager._update_trailing_stops(
            "CS.D.EURUSD.CFD.IP", Decimal("92")
        )
        config = order_manager.active_trailing_stops["DEAL2"]
        assert config.current_stop == Decimal("95")

    @pytest.mark.asyncio
    async def test_trailing_stop_ignores_other_instruments(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL1",
            trail_distance=Decimal("10"),
            direction="BUY",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("90"),
        )
        # Update for a different instrument should not affect EURUSD stop
        await order_manager._update_trailing_stops(
            "CS.D.GBPUSD.CFD.IP", Decimal("200")
        )
        config = order_manager.active_trailing_stops["DEAL1"]
        assert config.current_stop == Decimal("90")

    @pytest.mark.asyncio
    async def test_remove_trailing_stop(
        self, order_manager: OrderManager
    ) -> None:
        await order_manager.setup_trailing_stop(
            deal_id="DEAL1",
            trail_distance=Decimal("10"),
            direction="BUY",
            instrument="CS.D.EURUSD.CFD.IP",
            current_price=Decimal("100"),
            current_stop=Decimal("90"),
        )
        order_manager.remove_trailing_stop("DEAL1")
        assert "DEAL1" not in order_manager.active_trailing_stops


# =============================================================================
# Task 12.4: Partial Take Profit
# =============================================================================


class TestPartialTakeProfit:
    """Tests for partial take profit with breakeven stop adjustment."""

    @pytest.mark.asyncio
    async def test_partial_tp_closes_configured_percentage(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        order = await order_manager.partial_take_profit(
            deal_id="DEAL456",
            close_pct=Decimal("0.50"),
        )
        # Should close 50% of size 1.0 = 0.5
        mock_ig_client.close_position.assert_called_once_with(
            deal_id="DEAL456",
            direction="SELL",  # Opposite of BUY position
            size=Decimal("0.50"),
        )
        assert order.status == OrderStatus.CLOSED

    @pytest.mark.asyncio
    async def test_partial_tp_moves_stop_to_breakeven_buy(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # Position: BUY at 1.1234, spread 0.0002
        # Breakeven stop = entry - spread = 1.1234 - 0.0002 = 1.1232
        await order_manager.partial_take_profit(
            deal_id="DEAL456",
            close_pct=Decimal("0.50"),
        )
        mock_ig_client.update_position.assert_called_with(
            deal_id="DEAL456",
            stop_level=Decimal("1.1232"),
        )

    @pytest.mark.asyncio
    async def test_partial_tp_moves_stop_to_breakeven_sell(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        mock_ig_client.get_position.return_value = {
            "direction": "SELL",
            "instrument": "CS.D.EURUSD.CFD.IP",
            "size": "2.0",
            "entry_price": "1.1234",
            "spread": "0.0002",
        }
        await order_manager.partial_take_profit(
            deal_id="DEAL456",
            close_pct=Decimal("0.50"),
        )
        # Breakeven for SELL = entry + spread = 1.1234 + 0.0002 = 1.1236
        mock_ig_client.update_position.assert_called_with(
            deal_id="DEAL456",
            stop_level=Decimal("1.1236"),
        )

    @pytest.mark.asyncio
    async def test_partial_tp_rejects_below_min_pct(
        self, order_manager: OrderManager
    ) -> None:
        with pytest.raises(OrderValidationError, match="between"):
            await order_manager.partial_take_profit(
                deal_id="DEAL456",
                close_pct=Decimal("0.10"),
            )

    @pytest.mark.asyncio
    async def test_partial_tp_rejects_above_max_pct(
        self, order_manager: OrderManager
    ) -> None:
        with pytest.raises(OrderValidationError, match="between"):
            await order_manager.partial_take_profit(
                deal_id="DEAL456",
                close_pct=Decimal("0.80"),
            )

    @pytest.mark.asyncio
    async def test_partial_tp_marks_original_order_as_partially_closed(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # First place an order that gets filled
        filled_order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.deal_id == "DEAL456"

        # Now do partial TP on that deal
        await order_manager.partial_take_profit(
            deal_id="DEAL456",
            close_pct=Decimal("0.50"),
        )
        assert filled_order.status == OrderStatus.PARTIALLY_CLOSED

    @pytest.mark.asyncio
    async def test_partial_tp_uses_default_pct(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        await order_manager.partial_take_profit(deal_id="DEAL456")
        # Default is 50% of 1.0 = 0.5
        mock_ig_client.close_position.assert_called_once_with(
            deal_id="DEAL456",
            direction="SELL",
            size=Decimal("0.50"),
        )


# =============================================================================
# Task 12.5: Order Failure Handling
# =============================================================================


class TestFailureHandling:
    """Tests for order failure handling with retry logic."""

    @pytest.mark.asyncio
    async def test_retries_once_on_failure_then_succeeds(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # First call fails, second succeeds
        mock_ig_client.place_order.side_effect = [
            Exception("Network error"),
            {
                "dealReference": "RETRY_REF",
                "dealId": "RETRY_DEAL",
                "entryPrice": "1.1234",
            },
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            order = await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )
        assert order.status == OrderStatus.FILLED
        assert order.deal_id == "RETRY_DEAL"
        assert order.retry_count == 1
        mock_sleep.assert_called_once_with(ORDER_RETRY_DELAY_SECONDS)

    @pytest.mark.asyncio
    async def test_marks_failed_after_retry_fails(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # Both calls fail
        mock_ig_client.place_order.side_effect = Exception("Persistent error")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OrderExecutionError, match="failed after retry"):
                await order_manager.place_market_order(
                    instrument="CS.D.EURUSD.CFD.IP",
                    direction="BUY",
                    size=Decimal("1.0"),
                )
        # Check the order is marked as FAILED
        orders = list(order_manager.orders.values())
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FAILED
        assert orders[0].error_message is not None

    @pytest.mark.asyncio
    async def test_notifies_on_failure(
        self, order_manager: OrderManager,
        mock_ig_client: AsyncMock,
        mock_notification_service: AsyncMock,
    ) -> None:
        mock_ig_client.place_order.side_effect = Exception("API down")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OrderExecutionError):
                await order_manager.place_market_order(
                    instrument="CS.D.EURUSD.CFD.IP",
                    direction="BUY",
                    size=Decimal("1.0"),
                )
        # Should notify at least twice: initial failure + retry failure
        assert mock_notification_service.send_alert.call_count >= 2

    @pytest.mark.asyncio
    async def test_publishes_event_on_fill(
        self, order_manager: OrderManager, mock_event_bus: AsyncMock
    ) -> None:
        await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        mock_event_bus.publish.assert_called()
        # Check the channel was order.filled
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "order.filled"

    @pytest.mark.asyncio
    async def test_publishes_event_on_rejection_after_failure(
        self, order_manager: OrderManager,
        mock_ig_client: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        mock_ig_client.place_order.side_effect = Exception("Error")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OrderExecutionError):
                await order_manager.place_market_order(
                    instrument="CS.D.EURUSD.CFD.IP",
                    direction="BUY",
                    size=Decimal("1.0"),
                )
        # Should have published order.rejected event
        calls = mock_event_bus.publish.call_args_list
        channels = [c[0][0] for c in calls]
        assert "order.rejected" in channels

    @pytest.mark.asyncio
    async def test_close_order_failure_retries(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # close_position fails first, then place_order succeeds on retry
        mock_ig_client.close_position.side_effect = Exception("Close failed")
        mock_ig_client.place_order.return_value = {
            "dealReference": "RETRY_CLOSE",
            "dealId": "RETRY_DEAL",
            "entryPrice": "1.1234",
        }
        with patch("asyncio.sleep", new_callable=AsyncMock):
            order = await order_manager.close_order(
                deal_id="DEAL456",
                direction="SELL",
                size=Decimal("1.0"),
            )
        # Retry uses _execute_on_ig which calls place_order
        assert order.status == OrderStatus.FILLED


# =============================================================================
# Helper Methods: get_order and get_open_orders
# =============================================================================


class TestHelperMethods:
    """Tests for get_order and get_open_orders helper methods."""

    @pytest.mark.asyncio
    async def test_get_order_returns_existing_order(
        self, order_manager: OrderManager
    ) -> None:
        order = await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        retrieved = order_manager.get_order(order.id)
        assert retrieved is order

    def test_get_order_returns_none_for_unknown_id(
        self, order_manager: OrderManager
    ) -> None:
        assert order_manager.get_order("NONEXISTENT") is None

    @pytest.mark.asyncio
    async def test_get_open_orders_returns_empty_when_all_filled(
        self, order_manager: OrderManager
    ) -> None:
        await order_manager.place_market_order(
            instrument="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=Decimal("1.0"),
        )
        # All orders are FILLED, so open orders should be empty
        assert order_manager.get_open_orders() == []

    @pytest.mark.asyncio
    async def test_get_open_orders_includes_pending_and_submitted(
        self, order_manager: OrderManager, mock_ig_client: AsyncMock
    ) -> None:
        # Create an order that stays in REJECTED state (not open)
        mock_ig_client.get_market_info.return_value = {
            "status": "CLOSED",
            "min_size": "0.01",
        }
        with pytest.raises(OrderValidationError):
            await order_manager.place_market_order(
                instrument="CS.D.EURUSD.CFD.IP",
                direction="BUY",
                size=Decimal("1.0"),
            )
        # REJECTED orders are not "open"
        assert order_manager.get_open_orders() == []
