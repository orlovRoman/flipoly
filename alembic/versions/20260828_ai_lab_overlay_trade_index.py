"""Index PAPER trade history for overlay runtime metrics."""

import sqlalchemy as sa

from alembic import op

revision = "20260828_ai_lab_overlay_trade_index"
down_revision = "20260828_merge_ai_lab_live_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {item["name"] for item in inspector.get_indexes("trade_history")}
    if "idx_trade_history_mode_asset_created" not in indexes:
        op.create_index(
            "idx_trade_history_mode_asset_created",
            "trade_history",
            ["mode", "asset", "created_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {item["name"] for item in inspector.get_indexes("trade_history")}
    if "idx_trade_history_mode_asset_created" in indexes:
        op.drop_index(
            "idx_trade_history_mode_asset_created", table_name="trade_history"
        )
