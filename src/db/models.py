"""SQLAlchemy ORM models for the Institutional AI Trading System.

Uses SQLAlchemy 2.0 Mapped columns style with UUID primary keys,
proper foreign key relationships, JSON columns, and indexed fields.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Uuid as SAUuid
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db.database import Base


# --- Enums ---


class TradeDirection(str, enum.Enum):
    """Trade direction enum."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, enum.Enum):
    """Trade lifecycle status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class PositionStatus(str, enum.Enum):
    """Position lifecycle status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


class CopiedTraderStatus(str, enum.Enum):
    """Copied trader tracking status."""

    ACTIVE = "ACTIVE"
    STOPPED = "STOPPED"
    REMOVED = "REMOVED"


class MistakeClassification(str, enum.Enum):
    """Root-cause classification for losing trades."""

    COUNTER_TREND_ENTRY = "counter_trend_entry"
    FALSE_BREAKOUT = "false_breakout"
    VOLATILITY_MISJUDGMENT = "volatility_misjudgment"
    POOR_TIMING = "poor_timing"
    OVEREXPOSURE = "overexposure"
    REGIME_MISCLASSIFICATION = "regime_misclassification"


class ImpactLevel(str, enum.Enum):
    """News/event impact level classification."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# --- Models ---


class Trade(Base):
    """Completed trade records with full execution details."""

    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    instrument: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(
        SAEnum(TradeDirection, name="trade_direction", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    size: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence_score: Mapped[int] = mapped_column(Integer, nullable=False)
    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    is_hft: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_copied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(TradeStatus, name="trade_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TradeStatus.OPEN,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    positions: Mapped[list["Position"]] = relationship(back_populates="trade")
    context: Mapped["TradeContext | None"] = relationship(back_populates="trade", uselist=False)
    mistake_record: Mapped["MistakeRecord | None"] = relationship(
        back_populates="trade", uselist=False
    )

    __table_args__ = (
        Index("ix_trades_instrument_status", "instrument", "status"),
        Index("ix_trades_opened_at", "opened_at"),
        Index("ix_trades_strategy_opened_at", "strategy", "opened_at"),
    )


class Position(Base):
    """Active and historical position records."""

    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID] = mapped_column(
        SAUuid, ForeignKey("trades.id"), nullable=False, index=True
    )
    instrument: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(
        SAEnum(TradeDirection, name="trade_direction", values_callable=lambda x: [e.value for e in x], create_type=False),
        nullable=False,
    )
    size: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    status: Mapped[str] = mapped_column(
        SAEnum(PositionStatus, name="position_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=PositionStatus.OPEN,
    )
    trailing_stop: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    is_hft: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    trade: Mapped["Trade"] = relationship(back_populates="positions")

    __table_args__ = (
        Index("ix_positions_status", "status"),
        Index("ix_positions_instrument_status", "instrument", "status"),
    )


class AccountSnapshot(Base):
    """Periodic account state snapshots for equity tracking."""

    __tablename__ = "account_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    margin_used: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StrategyPerformance(Base):
    """Daily strategy performance metrics for monitoring and auto-disable."""

    __tablename__ = "strategy_performance"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    profit_factor: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_strategy_performance_name_date", "strategy_name", "date"),
    )


class AuditLog(Base):
    """Audit trail for all admin and system actions."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    user: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TradeContext(Base):
    """Full market context captured at trade execution for learning."""

    __tablename__ = "trade_context"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID] = mapped_column(
        SAUuid, ForeignKey("trades.id"), nullable=False, unique=True, index=True
    )
    indicators_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ml_predictions_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    trade: Mapped["Trade"] = relationship(back_populates="context")


class MLModelState(Base):
    """ML model versioning and performance tracking."""

    __tablename__ = "ml_model_state"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    weights_path: Mapped[str] = mapped_column(Text, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CopiedTrader(Base):
    """Tracked traders for copy trading with allocation and status."""

    __tablename__ = "copied_traders"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    trader_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    allocation_pct: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(CopiedTraderStatus, name="copied_trader_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=CopiedTraderStatus.ACTIVE,
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MistakeRecord(Base):
    """Individual losing trade records with root-cause classification."""

    __tablename__ = "mistake_records"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID] = mapped_column(
        SAUuid, ForeignKey("trades.id"), nullable=False, unique=True, index=True
    )
    classification: Mapped[str] = mapped_column(
        SAEnum(
            MistakeClassification,
            name="mistake_classification",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        index=True,
    )
    entry_conditions_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(50), nullable=True)
    indicators_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence_at_entry: Mapped[int] = mapped_column(Integer, nullable=False)
    exit_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    trade: Mapped["Trade"] = relationship(back_populates="mistake_record")

    __table_args__ = (
        Index("ix_mistake_records_classification_created", "classification", "created_at"),
    )


class MistakePattern(Base):
    """Detected recurring mistake patterns with penalty configuration."""

    __tablename__ = "mistake_patterns"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    classification: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    loss_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_occurrence: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_occurrence: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reactivated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence_penalty: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    size_reduction: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    resolution_progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_mistake_patterns_active", "active"),
        Index("ix_mistake_patterns_classification_active", "classification", "active"),
    )


class NewsArticle(Base):
    """Ingested news articles with sentiment analysis results."""

    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_tier: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    impact_level: Mapped[str] = mapped_column(
        SAEnum(ImpactLevel, name="impact_level", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
    )
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    correlated_instruments_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_news_articles_received_impact", "received_at", "impact_level"),
    )


class CrisisAlert(Base):
    """Detected crisis events with resolution tracking."""

    __tablename__ = "crisis_alerts"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    region: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    trigger_articles_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sentiment_avg: Mapped[float] = mapped_column(Float, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    escalated_to_kill_switch: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_crisis_alerts_region_active", "region", "active"),
    )


class EconomicEvent(Base):
    """Scheduled economic events from the economic calendar."""

    __tablename__ = "economic_events"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    impact_level: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    correlated_instruments_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    actual_value: Mapped[str | None] = mapped_column(String(100), nullable=True)
    forecast_value: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_economic_events_scheduled_impact", "scheduled_at", "impact_level"),
    )


class GeopoliticalRiskScore(Base):
    """Regional geopolitical risk scores for exposure management."""

    __tablename__ = "geopolitical_risk_scores"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    region: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    indicators_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_geopolitical_risk_scores_region_updated", "region", "updated_at"),
    )


class HFTMetrics(Base):
    """High-frequency trading performance metrics snapshots."""

    __tablename__ = "hft_metrics"

    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    orders_per_second: Mapped[float] = mapped_column(Float, nullable=False)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    net_pnl_1min: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    net_pnl_5min: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    net_pnl_daily: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    circuit_breaker_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    total_exposure_pct: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
