"""Initial schema - all tables.

Revision ID: 001
Revises: None
Create Date: 2024-01-01 00:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables for the Institutional AI Trading System."""

    # --- Create enum types ---
    trade_direction_enum = sa.Enum("LONG", "SHORT", name="trade_direction")
    trade_direction_enum.create(op.get_bind(), checkfirst=True)

    trade_status_enum = sa.Enum("OPEN", "CLOSED", "CANCELLED", name="trade_status")
    trade_status_enum.create(op.get_bind(), checkfirst=True)

    position_status_enum = sa.Enum("OPEN", "PARTIALLY_CLOSED", "CLOSED", name="position_status")
    position_status_enum.create(op.get_bind(), checkfirst=True)

    copied_trader_status_enum = sa.Enum("ACTIVE", "STOPPED", "REMOVED", name="copied_trader_status")
    copied_trader_status_enum.create(op.get_bind(), checkfirst=True)

    mistake_classification_enum = sa.Enum(
        "counter_trend_entry",
        "false_breakout",
        "volatility_misjudgment",
        "poor_timing",
        "overexposure",
        "regime_misclassification",
        name="mistake_classification",
    )
    mistake_classification_enum.create(op.get_bind(), checkfirst=True)

    impact_level_enum = sa.Enum("HIGH", "MEDIUM", "LOW", name="impact_level")
    impact_level_enum.create(op.get_bind(), checkfirst=True)

    # --- trades ---
    op.create_table(
        "trades",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("instrument", sa.String(50), nullable=False),
        sa.Column(
            "direction",
            trade_direction_enum,
            nullable=False,
        ),
        sa.Column("size", sa.Numeric(18, 8), nullable=False),
        sa.Column("entry_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("exit_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("pnl", sa.Numeric(18, 8), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence_score", sa.Integer, nullable=False),
        sa.Column("regime", sa.String(20), nullable=False),
        sa.Column("is_hft", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_copied", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "status",
            trade_status_enum,
            nullable=False,
            server_default="OPEN",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_trades_instrument", "trades", ["instrument"])
    op.create_index("ix_trades_strategy", "trades", ["strategy"])
    op.create_index("ix_trades_opened_at", "trades", ["opened_at"])
    op.create_index("ix_trades_strategy_opened_at", "trades", ["strategy", "opened_at"])
    op.create_index("ix_trades_instrument_status", "trades", ["instrument", "status"])

    # --- positions ---
    op.create_table(
        "positions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("trade_id", UUID, sa.ForeignKey("trades.id"), nullable=False),
        sa.Column("instrument", sa.String(50), nullable=False),
        sa.Column(
            "direction",
            trade_direction_enum,
            nullable=False,
        ),
        sa.Column("size", sa.Numeric(18, 8), nullable=False),
        sa.Column("entry_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("stop_loss", sa.Numeric(18, 8), nullable=True),
        sa.Column("take_profit", sa.Numeric(18, 8), nullable=True),
        sa.Column(
            "status",
            position_status_enum,
            nullable=False,
            server_default="OPEN",
        ),
        sa.Column("trailing_stop_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_hft", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_positions_trade_id", "positions", ["trade_id"])
    op.create_index("ix_positions_instrument", "positions", ["instrument"])
    op.create_index("ix_positions_status", "positions", ["status"])
    op.create_index("ix_positions_instrument_status", "positions", ["instrument", "status"])

    # --- account_snapshots ---
    op.create_table(
        "account_snapshots",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Numeric(18, 2), nullable=False),
        sa.Column("balance", sa.Numeric(18, 2), nullable=False),
        sa.Column("margin_used", sa.Numeric(18, 2), nullable=False),
        sa.Column("drawdown_pct", sa.Float, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_account_snapshots_timestamp", "account_snapshots", ["timestamp"])

    # --- strategy_performance ---
    op.create_table(
        "strategy_performance",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sharpe_ratio", sa.Float, nullable=False),
        sa.Column("win_rate", sa.Float, nullable=False),
        sa.Column("profit_factor", sa.Float, nullable=False),
        sa.Column("trade_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_strategy_performance_strategy_name", "strategy_performance", ["strategy_name"]
    )
    op.create_index(
        "ix_strategy_performance_name_date", "strategy_performance", ["strategy_name", "date"]
    )

    # --- audit_log ---
    op.create_table(
        "audit_log",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user", sa.String(100), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("details", JSON, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])

    # --- trade_context ---
    op.create_table(
        "trade_context",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("trade_id", UUID, sa.ForeignKey("trades.id"), nullable=False, unique=True),
        sa.Column("indicators_json", JSON, nullable=True),
        sa.Column("regime", sa.String(20), nullable=True),
        sa.Column("confidence", sa.Integer, nullable=True),
        sa.Column("ml_predictions_json", JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_trade_context_trade_id", "trade_context", ["trade_id"])

    # --- ml_model_state ---
    op.create_table(
        "ml_model_state",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("accuracy", sa.Float, nullable=False),
        sa.Column("weights_path", sa.Text, nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_ml_model_state_model_name", "ml_model_state", ["model_name"])

    # --- copied_traders ---
    op.create_table(
        "copied_traders",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("trader_id", sa.String(100), nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("risk_score", sa.Float, nullable=False),
        sa.Column("allocation_pct", sa.Float, nullable=False),
        sa.Column(
            "status",
            copied_trader_status_enum,
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_copied_traders_trader_id", "copied_traders", ["trader_id"])

    # --- mistake_records ---
    op.create_table(
        "mistake_records",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("trade_id", UUID, sa.ForeignKey("trades.id"), nullable=False, unique=True),
        sa.Column(
            "classification",
            mistake_classification_enum,
            nullable=False,
        ),
        sa.Column("entry_conditions_json", JSON, nullable=True),
        sa.Column("regime", sa.String(20), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("indicators_json", JSON, nullable=True),
        sa.Column("confidence_at_entry", sa.Integer, nullable=False),
        sa.Column("exit_reason", sa.String(100), nullable=True),
        sa.Column("pnl", sa.Numeric(18, 8), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_mistake_records_trade_id", "mistake_records", ["trade_id"])
    op.create_index("ix_mistake_records_classification", "mistake_records", ["classification"])
    op.create_index(
        "ix_mistake_records_classification_created",
        "mistake_records",
        ["classification", "created_at"],
    )

    # --- mistake_patterns ---
    op.create_table(
        "mistake_patterns",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("classification", sa.String(50), nullable=False),
        sa.Column("loss_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("first_occurrence", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_occurrence", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("reactivated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("confidence_penalty", sa.Integer, nullable=False, server_default="-20"),
        sa.Column("size_reduction", sa.Float, nullable=False, server_default="0.7"),
        sa.Column("resolution_progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_mistake_patterns_classification", "mistake_patterns", ["classification"])
    op.create_index("ix_mistake_patterns_active", "mistake_patterns", ["active"])
    op.create_index(
        "ix_mistake_patterns_classification_active",
        "mistake_patterns",
        ["classification", "active"],
    )

    # --- news_articles ---
    op.create_table(
        "news_articles",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_tier", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("headline", sa.Text, nullable=False),
        sa.Column("body_hash", sa.String(64), nullable=False),
        sa.Column("sentiment_score", sa.Float, nullable=False),
        sa.Column(
            "impact_level",
            impact_level_enum,
            nullable=False,
        ),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("correlated_instruments_json", JSON, nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_news_articles_source", "news_articles", ["source"])
    op.create_index("ix_news_articles_impact_level", "news_articles", ["impact_level"])
    op.create_index("ix_news_articles_received_at", "news_articles", ["received_at"])
    op.create_index(
        "ix_news_articles_received_impact", "news_articles", ["received_at", "impact_level"]
    )

    # --- crisis_alerts ---
    op.create_table(
        "crisis_alerts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("region", sa.String(50), nullable=False),
        sa.Column("trigger_articles_json", JSON, nullable=True),
        sa.Column("sentiment_avg", sa.Float, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "escalated_to_kill_switch", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_crisis_alerts_region", "crisis_alerts", ["region"])
    op.create_index("ix_crisis_alerts_region_active", "crisis_alerts", ["region", "active"])

    # --- economic_events ---
    op.create_table(
        "economic_events",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("event_name", sa.String(200), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("impact_level", sa.String(10), nullable=False),
        sa.Column("correlated_instruments_json", JSON, nullable=True),
        sa.Column("actual_value", sa.String(100), nullable=True),
        sa.Column("forecast_value", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_economic_events_event_type", "economic_events", ["event_type"])
    op.create_index("ix_economic_events_scheduled_at", "economic_events", ["scheduled_at"])
    op.create_index("ix_economic_events_impact_level", "economic_events", ["impact_level"])
    op.create_index(
        "ix_economic_events_scheduled_impact",
        "economic_events",
        ["scheduled_at", "impact_level"],
    )

    # --- geopolitical_risk_scores ---
    op.create_table(
        "geopolitical_risk_scores",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("region", sa.String(50), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("indicators_json", JSON, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_geopolitical_risk_scores_region", "geopolitical_risk_scores", ["region"])
    op.create_index(
        "ix_geopolitical_risk_scores_region_updated",
        "geopolitical_risk_scores",
        ["region", "updated_at"],
    )

    # --- hft_metrics ---
    op.create_table(
        "hft_metrics",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("orders_per_second", sa.Float, nullable=False),
        sa.Column("avg_latency_ms", sa.Float, nullable=False),
        sa.Column("net_pnl_1min", sa.Numeric(18, 8), nullable=False),
        sa.Column("net_pnl_5min", sa.Numeric(18, 8), nullable=False),
        sa.Column("net_pnl_daily", sa.Numeric(18, 8), nullable=False),
        sa.Column("circuit_breaker_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("total_exposure_pct", sa.Float, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_hft_metrics_timestamp", "hft_metrics", ["timestamp"])


def downgrade() -> None:
    """Drop all tables in reverse order of creation."""
    op.drop_table("hft_metrics")
    op.drop_table("geopolitical_risk_scores")
    op.drop_table("economic_events")
    op.drop_table("crisis_alerts")
    op.drop_table("news_articles")
    op.drop_table("mistake_patterns")
    op.drop_table("mistake_records")
    op.drop_table("copied_traders")
    op.drop_table("ml_model_state")
    op.drop_table("trade_context")
    op.drop_table("audit_log")
    op.drop_table("strategy_performance")
    op.drop_table("account_snapshots")
    op.drop_table("positions")
    op.drop_table("trades")

    # Drop enum types
    sa.Enum(name="impact_level").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="mistake_classification").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="copied_trader_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="position_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="trade_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="trade_direction").drop(op.get_bind(), checkfirst=True)
