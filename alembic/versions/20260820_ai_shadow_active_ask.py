"""Store the active model's same-snapshot ask for exact shadow PnL."""

from alembic import op
import sqlalchemy as sa


revision = "20260820_ai_shadow_active_ask"
down_revision = "20260819_ai_artifact_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_shadow_observations")}
    if "active_ask" not in columns:
        op.add_column(
            "ai_shadow_observations",
            sa.Column("active_ask", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_shadow_observations")}
    if "active_ask" in columns:
        op.drop_column("ai_shadow_observations", "active_ask")
