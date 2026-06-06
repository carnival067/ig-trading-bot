"""Add IG deal identifiers to trades and positions.

Revision ID: 002
Revises: 001
Create Date: 2026-06-06 00:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("ig_deal_id", sa.String(length=100), nullable=True))
    op.add_column("trades", sa.Column("ig_deal_reference", sa.String(length=100), nullable=True))
    op.add_column("positions", sa.Column("ig_deal_id", sa.String(length=100), nullable=True))

    op.create_index("ix_trades_ig_deal_id", "trades", ["ig_deal_id"])
    op.create_index("ix_trades_ig_deal_reference", "trades", ["ig_deal_reference"])
    op.create_index("ix_positions_ig_deal_id", "positions", ["ig_deal_id"])


def downgrade() -> None:
    op.drop_index("ix_positions_ig_deal_id", table_name="positions")
    op.drop_index("ix_trades_ig_deal_reference", table_name="trades")
    op.drop_index("ix_trades_ig_deal_id", table_name="trades")

    op.drop_column("positions", "ig_deal_id")
    op.drop_column("trades", "ig_deal_reference")
    op.drop_column("trades", "ig_deal_id")
