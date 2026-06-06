"""Async CRUD operations for trades and positions."""

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Position, PositionStatus, Trade, TradeStatus


class TradeRepository:
    """Repository for Trade and Position CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ─── Trade Operations ───────────────────────────────────────────────

    async def create_trade(self, trade_data: dict | Trade) -> Trade:
        """Insert a new trade record.

        Accepts either a dict of trade attributes or a Trade model instance.
        """
        if isinstance(trade_data, Trade):
            trade = trade_data
        else:
            trade = Trade(**trade_data)
        self._session.add(trade)
        await self._session.flush()
        await self._session.refresh(trade)
        return trade

    async def get_trade(self, trade_id: str | uuid.UUID) -> Trade | None:
        """Get a trade by ID."""
        if isinstance(trade_id, str):
            trade_id = uuid.UUID(trade_id)
        stmt = select(Trade).where(Trade.id == trade_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_open_trades(self, instrument: str | None = None) -> list[Trade]:
        """Get all open trades, optionally filtered by instrument."""
        stmt = select(Trade).where(Trade.status == TradeStatus.OPEN)
        if instrument is not None:
            stmt = stmt.where(Trade.instrument == instrument)
        stmt = stmt.order_by(Trade.opened_at.desc())
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_trade_by_ig_deal_id(self, ig_deal_id: str) -> Trade | None:
        """Get a trade by its broker deal ID."""
        stmt = select(Trade).where(Trade.ig_deal_id == ig_deal_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_trades_by_strategy(
        self, strategy: str, since: datetime | None = None, limit: int = 100
    ) -> list[Trade]:
        """Get trades filtered by strategy name, optionally since a given datetime."""
        stmt = select(Trade).where(Trade.strategy == strategy)
        if since is not None:
            stmt = stmt.where(Trade.opened_at >= since)
        stmt = stmt.order_by(Trade.opened_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_trades_by_instrument(
        self, instrument: str, limit: int = 100
    ) -> list[Trade]:
        """Get trades filtered by instrument."""
        stmt = (
            select(Trade)
            .where(Trade.instrument == instrument)
            .order_by(Trade.opened_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_trades(self, limit: int = 20) -> list[Trade]:
        """Get most recent trades ordered by opened_at descending."""
        stmt = (
            select(Trade)
            .order_by(Trade.opened_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_trades_since(self, since: datetime) -> list[Trade]:
        """Get all trades opened since a given datetime."""
        stmt = (
            select(Trade)
            .where(Trade.opened_at >= since)
            .order_by(Trade.opened_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_closed_trades_since(self, since: datetime) -> list[Trade]:
        """Get closed trades after a given timestamp."""
        stmt = (
            select(Trade)
            .where(
                Trade.status == TradeStatus.CLOSED,
                Trade.closed_at >= since,
            )
            .order_by(Trade.closed_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_daily_pnl(self, target_date: date) -> Decimal:
        """Sum of PnL for all trades closed on a given day. Returns Decimal('0') if none."""
        start_of_day = datetime(
            target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
        )
        end_of_day = datetime(
            target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999,
            tzinfo=timezone.utc,
        )
        stmt = select(func.coalesce(func.sum(Trade.pnl), Decimal("0"))).where(
            Trade.closed_at >= start_of_day,
            Trade.closed_at <= end_of_day,
            Trade.pnl.is_not(None),
        )
        result = await self._session.execute(stmt)
        value = result.scalar_one()
        return Decimal(str(value)) if value is not None else Decimal("0")

    async def get_trades_in_date_range(
        self, start: datetime, end: datetime
    ) -> list[Trade]:
        """Get trades opened within a date range."""
        stmt = (
            select(Trade)
            .where(Trade.opened_at >= start, Trade.opened_at <= end)
            .order_by(Trade.opened_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_trades_in_range(
        self, start: datetime, end: datetime
    ) -> list[Trade]:
        """Alias for get_trades_in_date_range."""
        return await self.get_trades_in_date_range(start, end)

    async def update_trade(
        self, trade_id: str | uuid.UUID, updates: dict | None = None, **kwargs
    ) -> Trade | None:
        """Partially update a trade by ID. Accepts a dict or keyword arguments."""
        trade = await self.get_trade(trade_id)
        if trade is None:
            return None
        # Merge dict and kwargs, kwargs take precedence
        merged = {**(updates or {}), **kwargs}
        for key, value in merged.items():
            if hasattr(trade, key):
                setattr(trade, key, value)
        await self._session.flush()
        await self._session.refresh(trade)
        return trade

    async def close_trade(
        self,
        trade_id: str | uuid.UUID,
        exit_price: Decimal,
        pnl: Decimal,
        closed_at: datetime | None = None,
    ) -> Trade | None:
        """Close a trade with exit price and PnL."""
        trade = await self.get_trade(trade_id)
        if trade is None:
            return None
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.closed_at = closed_at if closed_at is not None else datetime.now(timezone.utc)
        trade.status = TradeStatus.CLOSED
        await self._session.flush()
        await self._session.refresh(trade)
        return trade

    # ─── Position Operations ────────────────────────────────────────────

    async def create_position(self, position_data: dict | Position) -> Position:
        """Insert a new position record.

        Accepts either a dict of position attributes or a Position model instance.
        """
        if isinstance(position_data, Position):
            position = position_data
        else:
            position = Position(**position_data)
        self._session.add(position)
        await self._session.flush()
        await self._session.refresh(position)
        return position

    async def get_position(self, position_id: str | uuid.UUID) -> Position | None:
        """Get a position by ID."""
        if isinstance(position_id, str):
            position_id = uuid.UUID(position_id)
        stmt = select(Position).where(Position.id == position_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_open_positions(self) -> list[Position]:
        """Get all positions with OPEN status."""
        stmt = (
            select(Position)
            .where(Position.status == PositionStatus.OPEN)
            .order_by(Position.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_position_by_ig_deal_id(self, ig_deal_id: str) -> Position | None:
        """Get a position by its broker deal ID."""
        stmt = select(Position).where(Position.ig_deal_id == ig_deal_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_positions_by_instrument(self, instrument: str) -> list[Position]:
        """Get positions filtered by instrument."""
        stmt = (
            select(Position)
            .where(Position.instrument == instrument)
            .order_by(Position.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_positions_by_strategy(self, strategy: str) -> list[Position]:
        """Get positions filtered by strategy via their associated trade."""
        stmt = (
            select(Position)
            .join(Trade, Position.trade_id == Trade.id)
            .where(Trade.strategy == strategy)
            .order_by(Position.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_position(
        self, position_id: str | uuid.UUID, updates: dict | None = None, **kwargs
    ) -> Position | None:
        """Partially update a position by ID. Accepts a dict or keyword arguments."""
        if isinstance(position_id, str):
            position_id = uuid.UUID(position_id)
        stmt = select(Position).where(Position.id == position_id)
        result = await self._session.execute(stmt)
        position = result.scalar_one_or_none()
        if position is None:
            return None
        # Merge dict and kwargs, kwargs take precedence
        merged = {**(updates or {}), **kwargs}
        for key, value in merged.items():
            if hasattr(position, key):
                setattr(position, key, value)
        await self._session.flush()
        await self._session.refresh(position)
        return position

    async def update_position_stop(
        self, position_id: str | uuid.UUID, new_stop: Decimal
    ) -> Position | None:
        """Update the stop loss for a position."""
        if isinstance(position_id, str):
            position_id = uuid.UUID(position_id)
        stmt = select(Position).where(Position.id == position_id)
        result = await self._session.execute(stmt)
        position = result.scalar_one_or_none()
        if position is None:
            return None
        position.stop_loss = new_stop
        await self._session.flush()
        await self._session.refresh(position)
        return position

    async def close_position(
        self, position_id: str | uuid.UUID, status: str = "closed"
    ) -> Position | None:
        """Close a position by setting its status.

        Args:
            position_id: The position UUID.
            status: Target status string (default "closed"). Mapped to PositionStatus enum.
        """
        if isinstance(position_id, str):
            position_id = uuid.UUID(position_id)
        stmt = select(Position).where(Position.id == position_id)
        result = await self._session.execute(stmt)
        position = result.scalar_one_or_none()
        if position is None:
            return None
        # Map string status to enum, default to CLOSED
        status_map = {
            "closed": PositionStatus.CLOSED,
            "pending": PositionStatus.PENDING,
            "open": PositionStatus.OPEN,
        }
        position.status = status_map.get(status.lower(), PositionStatus.CLOSED)
        await self._session.flush()
        await self._session.refresh(position)
        return position

    async def get_hft_positions(self) -> list[Position]:
        """Get all open positions flagged as HFT."""
        stmt = (
            select(Position)
            .where(
                Position.is_hft == True,  # noqa: E712
                Position.status == PositionStatus.OPEN,
            )
            .order_by(Position.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ─── Analytical Queries ─────────────────────────────────────────────

    async def get_losing_trades(self, since: datetime) -> list[Trade]:
        """Get trades with pnl < 0 since a given datetime."""
        stmt = (
            select(Trade)
            .where(
                Trade.pnl < Decimal("0"),
                Trade.closed_at >= since,
                Trade.pnl.is_not(None),
            )
            .order_by(Trade.closed_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_trades_today(self, strategy: str) -> int:
        """Count trades opened today for a given strategy."""
        now = datetime.now(timezone.utc)
        start_of_day = datetime(
            now.year, now.month, now.day, tzinfo=timezone.utc
        )
        stmt = select(func.count()).select_from(Trade).where(
            Trade.strategy == strategy,
            Trade.opened_at >= start_of_day,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()
