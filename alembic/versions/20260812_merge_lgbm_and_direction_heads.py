"""merge lgbm configs and market direction signals heads"""
from alembic import op
import sqlalchemy as sa

revision = "20260812_merge_heads"
down_revision = ("20260812_lgbm_configs", "c3d4e5f6a7b9")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
