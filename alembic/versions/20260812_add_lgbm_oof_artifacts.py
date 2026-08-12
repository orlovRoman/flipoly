"""store reproducible LightGBM OOF backtest artifacts"""
from alembic import op
import sqlalchemy as sa

revision = "20260812_lgbm_oof"
down_revision = "merge_all_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_registry_oof_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "model_registry_id",
            sa.Integer(),
            sa.ForeignKey("model_registry.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("artifact_blob", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_model_registry_oof_model",
        "model_registry_oof_artifacts",
        ["model_registry_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_model_registry_oof_model", table_name="model_registry_oof_artifacts")
    op.drop_table("model_registry_oof_artifacts")
