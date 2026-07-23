"""add_config_presets_and_trade_snapshot

Revision ID: d470f6f21b90
Revises: c369e5f12a01
Create Date: 2026-07-23 18:40:00.000000+00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd470f6f21b90'
down_revision: Union[str, None] = 'c369e5f12a01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        "config_presets",
        sa.Column("id",              sa.Integer(),     primary_key=True, autoincrement=True),
        sa.Column("name",            sa.String(128),   nullable=False),
        sa.Column("description",     sa.String(512),   nullable=True),
        sa.Column("preset_type",     sa.String(32),    nullable=False, server_default="manual"),
        sa.Column("snapshot",        sa.Text(),        nullable=False),
        sa.Column("capital_at_save", sa.Float(),       nullable=True),
        sa.Column("pnl_at_save",     sa.Float(),       nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by",      sa.String(64),    nullable=False, server_default="user"),
        sa.Column("is_active",       sa.Boolean(),     nullable=False, server_default="true"),
    )
    op.create_index("idx_config_presets_created_at", "config_presets", ["created_at"])
    op.create_index("idx_config_presets_type",       "config_presets", ["preset_type"])

    op.add_column("trade_history",
        sa.Column("config_snapshot", sa.Text(), nullable=True)
    )

def downgrade() -> None:
    op.drop_column("trade_history", "config_snapshot")
    op.drop_table("config_presets")
