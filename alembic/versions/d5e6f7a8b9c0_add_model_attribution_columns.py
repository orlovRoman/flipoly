"""add model attribution columns to trade_history

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-07-30 10:30:00.000000+00:00
"""

from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "c4d5e6f7a8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trade_history",
        sa.Column("model_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("confirm_model_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("confirm_model_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("model_attribution_source", sa.String(length=16), nullable=True),
    )
    op.create_index(
        "idx_trade_history_exact_model",
        "trade_history",
        ["model_key", "model_version", "mode", "position_status", "closed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_trade_history_exact_model", table_name="trade_history")
    op.drop_column("trade_history", "model_attribution_source")
    op.drop_column("trade_history", "confirm_model_version")
    op.drop_column("trade_history", "confirm_model_key")
    op.drop_column("trade_history", "model_key")
