"""Add persisted mistake pattern indicator conditions.

Revision ID: 003
Revises: 002
Create Date: 2026-06-06 00:30:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "mistake_patterns",
        sa.Column("indicator_conditions_json", JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("mistake_patterns", "indicator_conditions_json")
