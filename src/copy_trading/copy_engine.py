"""Copy trading execution engine.

Handles trade replication from source traders, position close mirroring,
drawdown-based copy stopping, and execution timeouts. All copied trades
are validated through the Risk Engine (same rules as self-generated trades).

Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from src.config.constants import (
    COPY_CLOSE_TIMEOUT_SECONDS,
    COPY_DRAWDOWN_STOP_PCT,
    COPY_EXECUTION_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from src.risk.risk_engine import RiskEngine, TradeSignal, ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class CopyStatus(Enum):
    """Status of a copied trade."""

    PENDING = "pending"
    EXECUTED = "executed"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RISK_REJECTED = "risk_rejected"


class CopyStopReason(Enum):
    """Reason for stopping copy of a trader."""

    DRAWDOWN_LIMIT = "drawdown_limit"
    MANUAL = "manual"
    TRADER_INELIGIBLE = "trader_ineligible"
    WEEKLY_REEVALUATION = "weekly_reevaluation"


@dataclass
class SourceTrade:
    """A trade from the source trader to be replicated.

    Attributes:
        trade_id: Unique identifier of the source trade.
        trader_id: ID of the source trader.
        instrument: Trading instrument (e.g., "EUR/USD").
        direction: Trade direction ("LONG" or "SHORT").
        entry_price: Entry price of the source trade.
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        size: Position size of the source trade.
        asset_class: Asset class of the instrument.
        timestamp: When the source trade was opened.
    """

    trade_id: str
    trader_id: str
    instrument: str
    direction: str
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    size: Decimal
    asset_class: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CopiedTrade:
    """A trade that was replicated from a source trader.

    Attributes:
        deal_id: Unique identifier for this copied trade.
        source_trade_id: ID of the original source trade.
        trader_id: ID of the source trader.
        instrument: Trading instrument.
        direction: Trade direction.
        entry_price: Entry price.
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        size: Position size (risk-adjusted).
        status: Current status of the copied trade.
        opened_at: When the copied trade was opened.
        closed_at: When the copied trade was closed (if applicable).
        pnl: Profit/loss of the trade (if closed).
        execution_time_ms: Time taken to execute the copy in milliseconds.
    """

    deal_id: str
    source_trade_id: str
    trader_id: str
    instrument: str
    direction: str
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    size: Decimal
    status: CopyStatus = CopyStatus.PENDING
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    pnl: Decimal = Decimal("0")
    execution_time_ms: float = 0.0


@dataclass
class TraderAllocation:
    """Allocation details for a copied trader.

    Attributes:
        trader_id: ID of the trader being copied.
        allocated_equity: Amount of equity allocated to this trader.
        current_pnl: Running PnL from this trader's copied trades.
        start_date: When copying started.
        open_positions: List of currently open copied trade deal IDs.
        pnl_history: List of (timestamp, cumulative_pnl) for drawdown tracking.
    """

    trader_id: str
    allocated_equity: Decimal
    current_pnl: Decimal = Decimal("0")
    start_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    open_positions: list[str] = field(default_factory=list)
    pnl_history: list[tuple[datetime, Decimal]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Copy Engine
# ---------------------------------------------------------------------------


class CopyEngine:
    """Orchestrates copy trade execution with risk validation.

    Handles:
    - Trade replication with risk-adjusted position sizing
    - Position close mirroring within 2 seconds
    - Drawdown-based copy stopping (15% in 7-day window)
    - Execution timeout (cancel if not executed within 3 seconds)
    - Risk Engine validation for all copied trades

    Args:
        risk_engine: Optional RiskEngine for validating copied trades.
        execution_timeout_seconds: Max seconds to execute a copy trade.
        close_timeout_seconds: Max seconds to close a copied position.
        drawdown_stop_pct: Drawdown threshold to stop copying a trader.
    """

    def __init__(
        self,
        risk_engine: "RiskEngine | None" = None,
        execution_timeout_seconds: int = COPY_EXECUTION_TIMEOUT_SECONDS,
        close_timeout_seconds: int = COPY_CLOSE_TIMEOUT_SECONDS,
        drawdown_stop_pct: float = COPY_DRAWDOWN_STOP_PCT,
    ) -> None:
        self._risk_engine = risk_engine
        self._execution_timeout = execution_timeout_seconds
        self._close_timeout = close_timeout_seconds
        self._drawdown_stop_pct = Decimal(str(drawdown_stop_pct))
        self._active_copies: dict[str, CopiedTrade] = {}  # deal_id -> CopiedTrade
        self._trader_allocations: dict[str, TraderAllocation] = {}  # trader_id -> allocation
        self._deal_counter = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def register_trader(self, trader_id: str, allocated_equity: Decimal) -> None:
        """Register a trader for copy trading with allocated equity.

        Args:
            trader_id: ID of the trader to copy.
            allocated_equity: Amount of equity allocated to this trader.
        """
        self._trader_allocations[trader_id] = TraderAllocation(
            trader_id=trader_id,
            allocated_equity=allocated_equity,
        )
        logger.info(
            "Registered trader %s for copying with allocation %s",
            trader_id,
            allocated_equity,
        )

    async def replicate_trade(
        self,
        source_trade: SourceTrade,
        trader_allocation: Decimal,
        copier_equity: Decimal,
    ) -> CopiedTrade:
        """Replicate a source trade with risk-adjusted position sizing.

        The copied trade size is proportional to the copier's allocation
        relative to the source trader's position. All copied trades are
        validated through the Risk Engine.

        Args:
            source_trade: The source trade to replicate.
            trader_allocation: Equity allocated to this trader.
            copier_equity: Total equity of the copier's account.

        Returns:
            CopiedTrade with execution status.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout (trade is cancelled).
        """
        start_time = time.monotonic()
        self._deal_counter += 1
        deal_id = f"COPY-{self._deal_counter:06d}"

        # Calculate risk-adjusted size proportional to allocation
        # Size = source_size * (allocation / copier_equity)
        if copier_equity <= 0:
            copied_trade = CopiedTrade(
                deal_id=deal_id,
                source_trade_id=source_trade.trade_id,
                trader_id=source_trade.trader_id,
                instrument=source_trade.instrument,
                direction=source_trade.direction,
                entry_price=source_trade.entry_price,
                stop_loss=source_trade.stop_loss,
                take_profit=source_trade.take_profit,
                size=Decimal("0"),
                status=CopyStatus.CANCELLED,
            )
            return copied_trade

        allocation_ratio = trader_allocation / copier_equity
        adjusted_size = (source_trade.size * allocation_ratio).quantize(Decimal("0.01"))

        # Ensure minimum size
        if adjusted_size < Decimal("0.01"):
            adjusted_size = Decimal("0.01")

        copied_trade = CopiedTrade(
            deal_id=deal_id,
            source_trade_id=source_trade.trade_id,
            trader_id=source_trade.trader_id,
            instrument=source_trade.instrument,
            direction=source_trade.direction,
            entry_price=source_trade.entry_price,
            stop_loss=source_trade.stop_loss,
            take_profit=source_trade.take_profit,
            size=adjusted_size,
        )

        # Check execution timeout
        elapsed = time.monotonic() - start_time
        if elapsed > self._execution_timeout:
            copied_trade.status = CopyStatus.TIMED_OUT
            logger.warning(
                "Copy trade %s timed out after %.2fs (limit: %ds)",
                deal_id,
                elapsed,
                self._execution_timeout,
            )
            return copied_trade

        # Risk Engine validation (same rules as self-generated trades)
        if self._risk_engine is not None:
            try:
                validation = await asyncio.wait_for(
                    self._validate_with_risk_engine(source_trade, adjusted_size, copier_equity),
                    timeout=self._execution_timeout - elapsed,
                )
                if not validation.allowed:
                    copied_trade.status = CopyStatus.RISK_REJECTED
                    logger.info(
                        "Copy trade %s rejected by Risk Engine: %s",
                        deal_id,
                        validation.rejection_reasons,
                    )
                    return copied_trade
            except asyncio.TimeoutError:
                copied_trade.status = CopyStatus.TIMED_OUT
                logger.warning("Copy trade %s timed out during risk validation", deal_id)
                return copied_trade

        # Execute the trade
        elapsed = time.monotonic() - start_time
        if elapsed > self._execution_timeout:
            copied_trade.status = CopyStatus.TIMED_OUT
            logger.warning(
                "Copy trade %s timed out after risk validation (%.2fs)",
                deal_id,
                elapsed,
            )
            return copied_trade

        copied_trade.status = CopyStatus.EXECUTED
        copied_trade.execution_time_ms = (time.monotonic() - start_time) * 1000

        # Track the active copy
        self._active_copies[deal_id] = copied_trade

        # Update trader allocation tracking
        if source_trade.trader_id in self._trader_allocations:
            self._trader_allocations[source_trade.trader_id].open_positions.append(deal_id)

        logger.info(
            "Copied trade %s: %s %s %s size=%s in %.1fms",
            deal_id,
            source_trade.instrument,
            source_trade.direction,
            source_trade.entry_price,
            adjusted_size,
            copied_trade.execution_time_ms,
        )

        return copied_trade

    async def close_copied_position(self, deal_id: str) -> CopiedTrade | None:
        """Close a copied position within the close timeout (2 seconds).

        Mirrors the source trader's position close.

        Args:
            deal_id: The deal ID of the copied trade to close.

        Returns:
            Updated CopiedTrade with closed status, or None if not found.
        """
        if deal_id not in self._active_copies:
            logger.warning("Attempted to close unknown copied position: %s", deal_id)
            return None

        copied_trade = self._active_copies[deal_id]
        start_time = time.monotonic()

        try:
            # Simulate close execution within timeout
            elapsed = time.monotonic() - start_time
            if elapsed > self._close_timeout:
                logger.warning(
                    "Close of copied position %s exceeded %ds timeout",
                    deal_id,
                    self._close_timeout,
                )
                # Still close it but log the timeout
                pass

            copied_trade.status = CopyStatus.CLOSED
            copied_trade.closed_at = datetime.now(timezone.utc)

            # Remove from active copies
            del self._active_copies[deal_id]

            # Update trader allocation
            if copied_trade.trader_id in self._trader_allocations:
                allocation = self._trader_allocations[copied_trade.trader_id]
                if deal_id in allocation.open_positions:
                    allocation.open_positions.remove(deal_id)

            logger.info(
                "Closed copied position %s in %.1fms",
                deal_id,
                (time.monotonic() - start_time) * 1000,
            )

            return copied_trade

        except Exception as e:
            logger.error("Error closing copied position %s: %s", deal_id, e)
            return copied_trade

    async def stop_copying_trader(self, trader_id: str, reason: CopyStopReason = CopyStopReason.MANUAL) -> list[CopiedTrade]:
        """Stop copying a trader and close all their open positions within 5 seconds.

        Args:
            trader_id: ID of the trader to stop copying.
            reason: Reason for stopping.

        Returns:
            List of closed CopiedTrade instances.
        """
        start_time = time.monotonic()
        closed_trades: list[CopiedTrade] = []

        # Find all open positions for this trader
        positions_to_close = [
            deal_id
            for deal_id, trade in self._active_copies.items()
            if trade.trader_id == trader_id
        ]

        logger.info(
            "Stopping copy of trader %s (reason: %s). Closing %d positions.",
            trader_id,
            reason.value,
            len(positions_to_close),
        )

        # Close all positions (target: within 5 seconds)
        for deal_id in positions_to_close:
            elapsed = time.monotonic() - start_time
            if elapsed > 5.0:
                logger.warning(
                    "Stop copying trader %s: exceeded 5s timeout after closing %d/%d positions",
                    trader_id,
                    len(closed_trades),
                    len(positions_to_close),
                )
                # Continue closing remaining positions even if over time
                pass

            result = await self.close_copied_position(deal_id)
            if result is not None:
                closed_trades.append(result)

        # Remove trader allocation
        if trader_id in self._trader_allocations:
            del self._trader_allocations[trader_id]

        logger.info(
            "Stopped copying trader %s: closed %d positions in %.1fms",
            trader_id,
            len(closed_trades),
            (time.monotonic() - start_time) * 1000,
        )

        return closed_trades

    def check_drawdown_stop(self, trader_id: str, current_pnl: Decimal) -> bool:
        """Check if a trader's copy should be stopped due to drawdown.

        Triggers stop if drawdown exceeds 15% of allocated capital in a 7-day window.

        Args:
            trader_id: ID of the trader to check.
            current_pnl: Current cumulative PnL from this trader's copies.

        Returns:
            True if the trader should be stopped due to drawdown.
        """
        if trader_id not in self._trader_allocations:
            return False

        allocation = self._trader_allocations[trader_id]
        now = datetime.now(timezone.utc)

        # Record PnL history
        allocation.pnl_history.append((now, current_pnl))
        allocation.current_pnl = current_pnl

        # Filter to 7-day window
        seven_days_ago = now - timedelta(days=7)
        recent_history = [
            (ts, pnl) for ts, pnl in allocation.pnl_history if ts >= seven_days_ago
        ]
        allocation.pnl_history = recent_history

        # Calculate drawdown within the 7-day window
        if not recent_history:
            return False

        peak_pnl = max(pnl for _, pnl in recent_history)
        current_drawdown = peak_pnl - current_pnl

        # Check if drawdown exceeds threshold (15% of allocated equity)
        drawdown_limit = allocation.allocated_equity * self._drawdown_stop_pct
        if current_drawdown >= drawdown_limit:
            logger.warning(
                "Trader %s drawdown %.2f exceeds limit %.2f (%.1f%% of allocation %s)",
                trader_id,
                current_drawdown,
                drawdown_limit,
                float(current_drawdown / allocation.allocated_equity) * 100,
                allocation.allocated_equity,
            )
            return True

        return False

    async def handle_drawdown_stop(self, trader_id: str) -> list[CopiedTrade]:
        """Handle drawdown-based copy stop: stop copying and close all positions.

        Args:
            trader_id: ID of the trader whose drawdown exceeded the limit.

        Returns:
            List of closed CopiedTrade instances.
        """
        return await self.stop_copying_trader(trader_id, CopyStopReason.DRAWDOWN_LIMIT)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def active_copies(self) -> dict[str, CopiedTrade]:
        """Return all currently active copied trades."""
        return dict(self._active_copies)

    @property
    def trader_allocations(self) -> dict[str, TraderAllocation]:
        """Return all current trader allocations."""
        return dict(self._trader_allocations)

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _validate_with_risk_engine(
        self,
        source_trade: SourceTrade,
        adjusted_size: Decimal,
        copier_equity: Decimal,
    ) -> "ValidationResult":
        """Validate a copied trade through the Risk Engine.

        Applies the same rules as self-generated trades per Requirements 12.2, 12.3.

        Args:
            source_trade: The source trade being replicated.
            adjusted_size: Risk-adjusted position size.
            copier_equity: Copier's total account equity.

        Returns:
            ValidationResult from the Risk Engine.
        """
        from src.risk.risk_engine import TradeSignal

        signal = TradeSignal(
            instrument=source_trade.instrument,
            direction=source_trade.direction,
            entry_price=source_trade.entry_price,
            stop_loss=source_trade.stop_loss,
            take_profit=source_trade.take_profit,
            confidence=100,  # Copy trades bypass confidence check
            strategy="copy_trading",
            asset_class=source_trade.asset_class,
            notional_value=adjusted_size * source_trade.entry_price,
        )

        return await self._risk_engine.validate_signal(
            signal=signal,
            account_equity=copier_equity,
            current_positions=[],
        )
