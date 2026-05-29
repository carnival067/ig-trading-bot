"""Unit tests for TradeRepository."""

import os
import uuid

# Set required env vars before any src imports trigger settings validation
os.environ.setdefault("IG_API_KEY", "test_key")
os.environ.setdefault("IG_USERNAME", "test_user")
os.environ.setdefault("IG_PASSWORD", "test_pass")

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.db.database import Base
from src.db.models import Position, Trade, TradeDirection, TradeStatus, PositionStatus
from src.db.repositories.trade_repo import TradeRepository


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
async def async_engine():
    """Create an in-memory SQLite async engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Enable foreign key support for SQLite
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def session(async_engine):
    """Create an async session for testing."""
    session_factory = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest.fixture
def repo(session):
    """Create a TradeRepository instance."""
    return TradeRepository(session)


def _trade_data(**overrides) -> dict:
    """Helper to create valid trade data with defaults."""
    data = {
        "instrument": "EUR/USD",
        "direction": TradeDirection.LONG,
        "size": Decimal("1.0"),
        "entry_price": Decimal("1.1050"),
        "strategy": "trend_following",
        "opened_at": _now(),
        "confidence_score": 75,
        "regime": "trending",
        "status": TradeStatus.OPEN,
    }
    data.update(overrides)
    return data


def _position_data(trade_id: uuid.UUID, **overrides) -> dict:
    """Helper to create valid position data with defaults."""
    data = {
        "trade_id": trade_id,
        "instrument": "EUR/USD",
        "direction": TradeDirection.LONG,
        "size": Decimal("1.0"),
        "entry_price": Decimal("1.1050"),
        "status": PositionStatus.OPEN,
    }
    data.update(overrides)
    return data


# ─── Trade Tests ────────────────────────────────────────────────────────


class TestCreateTrade:
    async def test_creates_trade_with_required_fields(self, repo):
        trade = await repo.create_trade(_trade_data())
        assert trade.id is not None
        assert trade.instrument == "EUR/USD"
        assert trade.direction == TradeDirection.LONG
        assert trade.size == Decimal("1.0")
        assert trade.entry_price == Decimal("1.1050")
        assert trade.strategy == "trend_following"
        assert trade.opened_at is not None

    async def test_creates_trade_with_all_fields(self, repo):
        now = _now()
        trade = await repo.create_trade(
            _trade_data(
                instrument="GBP/USD",
                direction=TradeDirection.SHORT,
                size=Decimal("0.5"),
                entry_price=Decimal("1.2700"),
                exit_price=Decimal("1.2650"),
                pnl=Decimal("25.00"),
                strategy="mean_reversion",
                opened_at=now,
                closed_at=now + timedelta(hours=2),
                confidence_score=85,
                regime="ranging",
                status=TradeStatus.CLOSED,
            )
        )
        assert trade.exit_price == Decimal("1.2650")
        assert trade.pnl == Decimal("25.00")
        assert trade.confidence_score == 85
        assert trade.regime == "ranging"
        assert trade.status == TradeStatus.CLOSED


class TestGetTrade:
    async def test_returns_trade_by_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        found = await repo.get_trade(trade.id)
        assert found is not None
        assert found.id == trade.id
        assert found.instrument == "EUR/USD"

    async def test_returns_trade_by_string_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        found = await repo.get_trade(str(trade.id))
        assert found is not None
        assert found.id == trade.id

    async def test_returns_none_for_nonexistent_id(self, repo):
        result = await repo.get_trade(uuid.uuid4())
        assert result is None


class TestGetTradesByStrategy:
    async def test_filters_by_strategy(self, repo):
        await repo.create_trade(_trade_data(strategy="trend_following"))
        await repo.create_trade(_trade_data(strategy="mean_reversion"))
        trades = await repo.get_trades_by_strategy("trend_following")
        assert len(trades) == 1
        assert trades[0].strategy == "trend_following"

    async def test_respects_limit(self, repo):
        for i in range(5):
            await repo.create_trade(
                _trade_data(instrument=f"PAIR/{i}", strategy="scalping")
            )
        trades = await repo.get_trades_by_strategy("scalping", limit=3)
        assert len(trades) == 3

    async def test_filters_by_since(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(strategy="trend_following", opened_at=now - timedelta(days=5))
        )
        await repo.create_trade(
            _trade_data(strategy="trend_following", opened_at=now - timedelta(days=1))
        )
        trades = await repo.get_trades_by_strategy(
            "trend_following", since=now - timedelta(days=2)
        )
        assert len(trades) == 1


class TestGetTradesByInstrument:
    async def test_filters_by_instrument(self, repo):
        await repo.create_trade(_trade_data(instrument="EUR/USD"))
        await repo.create_trade(_trade_data(instrument="GBP/USD"))
        trades = await repo.get_trades_by_instrument("EUR/USD")
        assert len(trades) == 1
        assert trades[0].instrument == "EUR/USD"


class TestGetOpenTrades:
    async def test_returns_only_open_trades(self, repo):
        now = _now()
        await repo.create_trade(_trade_data(instrument="EUR/USD"))  # open (default status)
        await repo.create_trade(
            _trade_data(instrument="GBP/USD", closed_at=now, status=TradeStatus.CLOSED)
        )  # closed
        trades = await repo.get_open_trades()
        assert len(trades) == 1
        assert trades[0].instrument == "EUR/USD"

    async def test_filters_by_instrument(self, repo):
        await repo.create_trade(_trade_data(instrument="EUR/USD"))
        await repo.create_trade(_trade_data(instrument="GBP/USD"))
        trades = await repo.get_open_trades(instrument="EUR/USD")
        assert len(trades) == 1
        assert trades[0].instrument == "EUR/USD"

    async def test_returns_empty_when_all_closed(self, repo):
        now = _now()
        await repo.create_trade(_trade_data(closed_at=now, status=TradeStatus.CLOSED))
        trades = await repo.get_open_trades()
        assert len(trades) == 0


class TestGetRecentTrades:
    async def test_returns_trades_ordered_by_opened_at_desc(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(
                instrument="EUR/USD",
                opened_at=now - timedelta(hours=2),
            )
        )
        await repo.create_trade(
            _trade_data(
                instrument="GBP/USD",
                opened_at=now - timedelta(hours=1),
            )
        )
        await repo.create_trade(
            _trade_data(
                instrument="USD/JPY",
                opened_at=now,
            )
        )
        trades = await repo.get_recent_trades(limit=10)
        assert len(trades) == 3
        assert trades[0].instrument == "USD/JPY"  # Most recent
        assert trades[1].instrument == "GBP/USD"
        assert trades[2].instrument == "EUR/USD"

    async def test_respects_limit(self, repo):
        now = _now()
        for i in range(5):
            await repo.create_trade(
                _trade_data(instrument=f"PAIR/{i}", opened_at=now - timedelta(hours=i))
            )
        trades = await repo.get_recent_trades(limit=3)
        assert len(trades) == 3


class TestGetClosedTradesSince:
    async def test_returns_closed_trades_after_timestamp(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(
                instrument="EUR/USD",
                closed_at=now - timedelta(hours=1),
                status=TradeStatus.CLOSED,
            )
        )
        await repo.create_trade(
            _trade_data(
                instrument="GBP/USD",
                closed_at=now - timedelta(days=2),
                status=TradeStatus.CLOSED,
            )
        )
        trades = await repo.get_closed_trades_since(since=now - timedelta(days=1))
        assert len(trades) == 1
        assert trades[0].instrument == "EUR/USD"

    async def test_excludes_open_trades(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(instrument="EUR/USD", status=TradeStatus.OPEN)
        )
        await repo.create_trade(
            _trade_data(
                instrument="GBP/USD",
                closed_at=now - timedelta(hours=1),
                status=TradeStatus.CLOSED,
            )
        )
        trades = await repo.get_closed_trades_since(since=now - timedelta(days=1))
        assert len(trades) == 1
        assert trades[0].instrument == "GBP/USD"

    async def test_returns_empty_when_no_closed_trades(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(instrument="EUR/USD", status=TradeStatus.OPEN)
        )
        trades = await repo.get_closed_trades_since(since=now - timedelta(days=1))
        assert len(trades) == 0


class TestGetDailyPnl:
    async def test_sums_pnl_for_given_day(self, repo):
        from datetime import date as date_type

        now = _now()
        today = now.date()
        await repo.create_trade(
            _trade_data(
                pnl=Decimal("50.00"),
                closed_at=now - timedelta(hours=1),
            )
        )
        await repo.create_trade(
            _trade_data(
                pnl=Decimal("-20.00"),
                closed_at=now - timedelta(hours=2),
            )
        )
        result = await repo.get_daily_pnl(today)
        assert result == Decimal("30.00")

    async def test_returns_zero_when_no_trades(self, repo):
        from datetime import date as date_type

        yesterday = (_now() - timedelta(days=1)).date()
        result = await repo.get_daily_pnl(yesterday)
        assert result == Decimal("0")

    async def test_excludes_trades_from_other_days(self, repo):
        from datetime import date as date_type

        now = _now()
        today = now.date()
        # Trade closed yesterday
        await repo.create_trade(
            _trade_data(
                pnl=Decimal("100.00"),
                closed_at=now - timedelta(days=1),
            )
        )
        # Trade closed today
        await repo.create_trade(
            _trade_data(
                pnl=Decimal("25.00"),
                closed_at=now,
            )
        )
        result = await repo.get_daily_pnl(today)
        assert result == Decimal("25.00")


class TestGetTradesInDateRange:
    async def test_returns_trades_within_range(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(instrument="EUR/USD", opened_at=now - timedelta(days=5))
        )
        await repo.create_trade(
            _trade_data(instrument="GBP/USD", opened_at=now - timedelta(days=2))
        )
        await repo.create_trade(
            _trade_data(instrument="USD/JPY", opened_at=now + timedelta(days=1))
        )
        trades = await repo.get_trades_in_date_range(
            start=now - timedelta(days=6),
            end=now - timedelta(days=1),
        )
        assert len(trades) == 2


class TestUpdateTrade:
    async def test_updates_specified_fields(self, repo):
        trade = await repo.create_trade(_trade_data())
        updated = await repo.update_trade(
            trade.id, {"confidence_score": 92, "regime": "trending"}
        )
        assert updated is not None
        assert updated.confidence_score == 92
        assert updated.regime == "trending"

    async def test_updates_via_kwargs(self, repo):
        trade = await repo.create_trade(_trade_data())
        updated = await repo.update_trade(
            trade.id, confidence_score=88, regime="volatile"
        )
        assert updated is not None
        assert updated.confidence_score == 88
        assert updated.regime == "volatile"

    async def test_returns_none_for_nonexistent_trade(self, repo):
        result = await repo.update_trade(uuid.uuid4(), {"regime": "volatile"})
        assert result is None

    async def test_ignores_unknown_fields(self, repo):
        trade = await repo.create_trade(_trade_data())
        updated = await repo.update_trade(
            trade.id, {"nonexistent_field": "value", "regime": "crisis"}
        )
        assert updated is not None
        assert updated.regime == "crisis"


class TestCloseTrade:
    async def test_closes_trade_with_exit_price_and_pnl(self, repo):
        trade = await repo.create_trade(_trade_data())
        closed = await repo.close_trade(
            trade.id,
            exit_price=Decimal("1.1100"),
            pnl=Decimal("50.00"),
        )
        assert closed is not None
        assert closed.exit_price == Decimal("1.1100")
        assert closed.pnl == Decimal("50.00")
        assert closed.closed_at is not None

    async def test_returns_none_for_nonexistent_trade(self, repo):
        result = await repo.close_trade(
            uuid.uuid4(), Decimal("1.0"), Decimal("0.0")
        )
        assert result is None


# ─── Position Tests ─────────────────────────────────────────────────────


class TestCreatePosition:
    async def test_creates_position_with_required_fields(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(
            _position_data(
                trade.id,
                stop_loss=Decimal("1.1000"),
                take_profit=Decimal("1.1150"),
            )
        )
        assert position.id is not None
        assert position.instrument == "EUR/USD"
        assert position.status == PositionStatus.OPEN
        assert position.trade_id == trade.id


class TestGetOpenPositions:
    async def test_returns_only_open_positions(self, repo):
        trade = await repo.create_trade(_trade_data())
        await repo.create_position(
            _position_data(trade.id, instrument="EUR/USD", status=PositionStatus.OPEN)
        )
        await repo.create_position(
            _position_data(trade.id, instrument="GBP/USD", status=PositionStatus.CLOSED)
        )
        positions = await repo.get_open_positions()
        assert len(positions) == 1
        assert positions[0].instrument == "EUR/USD"


class TestGetPositionsByInstrument:
    async def test_filters_by_instrument(self, repo):
        trade = await repo.create_trade(_trade_data())
        await repo.create_position(
            _position_data(trade.id, instrument="EUR/USD")
        )
        await repo.create_position(
            _position_data(trade.id, instrument="GBP/USD")
        )
        positions = await repo.get_positions_by_instrument("EUR/USD")
        assert len(positions) == 1
        assert positions[0].instrument == "EUR/USD"

    async def test_returns_empty_list_for_no_matches(self, repo):
        trade = await repo.create_trade(_trade_data())
        await repo.create_position(
            _position_data(trade.id, instrument="EUR/USD")
        )
        positions = await repo.get_positions_by_instrument("USD/JPY")
        assert len(positions) == 0


class TestGetPositionsByStrategy:
    async def test_filters_by_strategy(self, repo):
        trade1 = await repo.create_trade(_trade_data(strategy="trend_following"))
        trade2 = await repo.create_trade(_trade_data(strategy="scalping"))
        await repo.create_position(_position_data(trade1.id, instrument="EUR/USD"))
        await repo.create_position(_position_data(trade2.id, instrument="GBP/USD"))
        positions = await repo.get_positions_by_strategy("trend_following")
        assert len(positions) == 1
        assert positions[0].instrument == "EUR/USD"


class TestUpdatePosition:
    async def test_updates_specified_fields(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        updated = await repo.update_position(
            position.id,
            {"stop_loss": Decimal("1.1020"), "take_profit": Decimal("1.1150")},
        )
        assert updated is not None
        assert updated.stop_loss == Decimal("1.1020")
        assert updated.take_profit == Decimal("1.1150")

    async def test_updates_via_kwargs(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        updated = await repo.update_position(
            position.id,
            stop_loss=Decimal("1.0990"),
            trailing_stop=Decimal("1.0980"),
        )
        assert updated is not None
        assert updated.stop_loss == Decimal("1.0990")
        assert updated.trailing_stop == Decimal("1.0980")

    async def test_returns_none_for_nonexistent_position(self, repo):
        result = await repo.update_position(uuid.uuid4(), {"status": PositionStatus.CLOSED})
        assert result is None


class TestUpdatePositionStop:
    async def test_updates_stop_loss(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(
            _position_data(trade.id, stop_loss=Decimal("1.1000"))
        )
        updated = await repo.update_position_stop(position.id, Decimal("1.1020"))
        assert updated is not None
        assert updated.stop_loss == Decimal("1.1020")

    async def test_returns_none_for_nonexistent_position(self, repo):
        result = await repo.update_position_stop(uuid.uuid4(), Decimal("1.1020"))
        assert result is None

    async def test_accepts_string_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(
            _position_data(trade.id, stop_loss=Decimal("1.1000"))
        )
        updated = await repo.update_position_stop(str(position.id), Decimal("1.1030"))
        assert updated is not None
        assert updated.stop_loss == Decimal("1.1030")


class TestClosePosition:
    async def test_sets_status_to_closed(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        closed = await repo.close_position(position.id)
        assert closed is not None
        assert closed.status == PositionStatus.CLOSED

    async def test_returns_none_for_nonexistent_position(self, repo):
        result = await repo.close_position(uuid.uuid4())
        assert result is None


# ─── Analytical Query Tests ─────────────────────────────────────────────


class TestGetLosingTrades:
    async def test_returns_trades_with_negative_pnl(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(pnl=Decimal("-10.00"), closed_at=now - timedelta(hours=1))
        )
        await repo.create_trade(
            _trade_data(pnl=Decimal("50.00"), closed_at=now - timedelta(hours=2))
        )
        await repo.create_trade(
            _trade_data(pnl=Decimal("-5.00"), closed_at=now - timedelta(hours=3))
        )
        trades = await repo.get_losing_trades(since=now - timedelta(days=1))
        assert len(trades) == 2
        assert all(t.pnl < Decimal("0") for t in trades)

    async def test_respects_since_filter(self, repo):
        now = _now()
        # Old losing trade (before since)
        await repo.create_trade(
            _trade_data(pnl=Decimal("-10.00"), closed_at=now - timedelta(days=5))
        )
        # Recent losing trade (after since)
        await repo.create_trade(
            _trade_data(pnl=Decimal("-20.00"), closed_at=now - timedelta(hours=1))
        )
        trades = await repo.get_losing_trades(since=now - timedelta(days=1))
        assert len(trades) == 1
        assert trades[0].pnl == Decimal("-20.00")

    async def test_returns_empty_when_no_losing_trades(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(pnl=Decimal("100.00"), closed_at=now - timedelta(hours=1))
        )
        trades = await repo.get_losing_trades(since=now - timedelta(days=1))
        assert len(trades) == 0

    async def test_excludes_trades_with_null_pnl(self, repo):
        now = _now()
        # Trade with no pnl (still open)
        await repo.create_trade(_trade_data(pnl=None, closed_at=now))
        trades = await repo.get_losing_trades(since=now - timedelta(days=1))
        assert len(trades) == 0


class TestCountTradesToday:
    async def test_counts_trades_for_strategy_today(self, repo):
        now = _now()
        await repo.create_trade(_trade_data(strategy="scalping", opened_at=now))
        await repo.create_trade(
            _trade_data(strategy="scalping", opened_at=now - timedelta(hours=1))
        )
        await repo.create_trade(
            _trade_data(strategy="trend_following", opened_at=now)
        )
        count = await repo.count_trades_today("scalping")
        assert count == 2

    async def test_excludes_trades_from_previous_days(self, repo):
        now = _now()
        await repo.create_trade(_trade_data(strategy="scalping", opened_at=now))
        await repo.create_trade(
            _trade_data(
                strategy="scalping", opened_at=now - timedelta(days=1)
            )
        )
        count = await repo.count_trades_today("scalping")
        assert count == 1

    async def test_returns_zero_when_no_trades(self, repo):
        count = await repo.count_trades_today("nonexistent_strategy")
        assert count == 0

# ─── New Method Tests ───────────────────────────────────────────────────


class TestGetTradesSince:
    async def test_returns_trades_opened_since_datetime(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(instrument="EUR/USD", opened_at=now - timedelta(hours=1))
        )
        await repo.create_trade(
            _trade_data(instrument="GBP/USD", opened_at=now - timedelta(days=2))
        )
        trades = await repo.get_trades_since(since=now - timedelta(days=1))
        assert len(trades) == 1
        assert trades[0].instrument == "EUR/USD"

    async def test_returns_empty_when_no_trades_since(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(opened_at=now - timedelta(days=5))
        )
        trades = await repo.get_trades_since(since=now - timedelta(hours=1))
        assert len(trades) == 0

    async def test_returns_all_trades_when_since_is_old(self, repo):
        now = _now()
        await repo.create_trade(
            _trade_data(instrument="EUR/USD", opened_at=now - timedelta(hours=2))
        )
        await repo.create_trade(
            _trade_data(instrument="GBP/USD", opened_at=now - timedelta(hours=1))
        )
        trades = await repo.get_trades_since(since=now - timedelta(days=10))
        assert len(trades) == 2


class TestGetPosition:
    async def test_returns_position_by_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        found = await repo.get_position(position.id)
        assert found is not None
        assert found.id == position.id
        assert found.instrument == "EUR/USD"

    async def test_returns_position_by_string_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        found = await repo.get_position(str(position.id))
        assert found is not None
        assert found.id == position.id

    async def test_returns_none_for_nonexistent_id(self, repo):
        result = await repo.get_position(uuid.uuid4())
        assert result is None


class TestCloseTradeWithClosedAt:
    async def test_uses_provided_closed_at(self, repo):
        trade = await repo.create_trade(_trade_data())
        custom_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        closed = await repo.close_trade(
            trade.id,
            exit_price=Decimal("1.1100"),
            pnl=Decimal("50.00"),
            closed_at=custom_time,
        )
        assert closed is not None
        # SQLite strips timezone info, so compare naive datetime values
        assert closed.closed_at.replace(tzinfo=None) == custom_time.replace(tzinfo=None)
        assert closed.status == TradeStatus.CLOSED

    async def test_defaults_to_now_when_closed_at_not_provided(self, repo):
        trade = await repo.create_trade(_trade_data())
        closed = await repo.close_trade(
            trade.id,
            exit_price=Decimal("1.1100"),
            pnl=Decimal("50.00"),
        )
        assert closed is not None
        assert closed.closed_at is not None
        assert closed.status == TradeStatus.CLOSED


class TestClosePositionWithStatus:
    async def test_defaults_to_closed_status(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        closed = await repo.close_position(position.id)
        assert closed is not None
        assert closed.status == PositionStatus.CLOSED

    async def test_accepts_explicit_closed_status(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        closed = await repo.close_position(position.id, status="closed")
        assert closed is not None
        assert closed.status == PositionStatus.CLOSED

    async def test_accepts_string_position_id(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = await repo.create_position(_position_data(trade.id))
        closed = await repo.close_position(str(position.id), status="closed")
        assert closed is not None
        assert closed.status == PositionStatus.CLOSED


class TestCreateTradeWithModel:
    async def test_accepts_trade_model_instance(self, repo):
        trade = Trade(
            instrument="USD/JPY",
            direction=TradeDirection.SHORT,
            size=Decimal("2.0"),
            entry_price=Decimal("150.50"),
            strategy="momentum",
            opened_at=_now(),
            confidence_score=80,
            regime="volatile",
            status=TradeStatus.OPEN,
        )
        created = await repo.create_trade(trade)
        assert created.id is not None
        assert created.instrument == "USD/JPY"
        assert created.direction == TradeDirection.SHORT


class TestCreatePositionWithModel:
    async def test_accepts_position_model_instance(self, repo):
        trade = await repo.create_trade(_trade_data())
        position = Position(
            trade_id=trade.id,
            instrument="EUR/USD",
            direction=TradeDirection.LONG,
            size=Decimal("1.0"),
            entry_price=Decimal("1.1050"),
            status=PositionStatus.OPEN,
        )
        created = await repo.create_position(position)
        assert created.id is not None
        assert created.instrument == "EUR/USD"
        assert created.trade_id == trade.id


class TestGetHftPositions:
    async def test_returns_only_open_hft_positions(self, repo):
        trade = await repo.create_trade(_trade_data())
        await repo.create_position(
            _position_data(trade.id, instrument="EUR/USD", is_hft=True, status=PositionStatus.OPEN)
        )
        await repo.create_position(
            _position_data(trade.id, instrument="GBP/USD", is_hft=False, status=PositionStatus.OPEN)
        )
        await repo.create_position(
            _position_data(trade.id, instrument="USD/JPY", is_hft=True, status=PositionStatus.CLOSED)
        )
        positions = await repo.get_hft_positions()
        assert len(positions) == 1
        assert positions[0].instrument == "EUR/USD"
        assert positions[0].is_hft is True

    async def test_returns_empty_when_no_hft_positions(self, repo):
        trade = await repo.create_trade(_trade_data())
        await repo.create_position(
            _position_data(trade.id, instrument="EUR/USD", is_hft=False)
        )
        positions = await repo.get_hft_positions()
        assert len(positions) == 0
