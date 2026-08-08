"""add model_type to model_registry

Revision ID: a1b2c3d4e5f7
Revises: merge_all_heads_merge_multiple_heads
Create Date: 2026-08-08 12:35:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f7'
down_revision = 'merge_all_heads_merge_multiple_heads'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Добавить колонку с дефолтом
    op.add_column(
        "model_registry",
        sa.Column("model_type", sa.String(20), nullable=False, server_default="logreg"),
    )
    # 2. Заполнить исторические записи
    op.execute("""
        UPDATE model_registry
        SET model_type = CASE WHEN asset LIKE '%USDT%' THEN 'lgbm' ELSE 'logreg' END
    """)
    # 3. Добавить CHECK constraint
    op.create_check_constraint(
        "ck_model_registry_model_type",
        "model_registry",
        "model_type IN ('logreg', 'lgbm')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_model_registry_model_type", "model_registry")
    op.drop_column("model_registry", "model_type")
