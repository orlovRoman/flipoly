"""Persist AI Lab overlay IDs on every PAPER trade."""

from alembic import op
import sqlalchemy as sa

revision = "20260828_ai_lab_overlay_trade_trace"
down_revision = "20260827_ai_llm_catalog_split"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "trade_history" not in tables:
        return
    columns = {item["name"] for item in inspector.get_columns("trade_history")}
    if "ai_lab_overlay_ids" not in columns:
        op.add_column(
            "trade_history",
            sa.Column("ai_lab_overlay_ids", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "trade_history" not in set(inspector.get_table_names()):
        return
    columns = {item["name"] for item in inspector.get_columns("trade_history")}
    if "ai_lab_overlay_ids" in columns:
        op.drop_column("trade_history", "ai_lab_overlay_ids")
