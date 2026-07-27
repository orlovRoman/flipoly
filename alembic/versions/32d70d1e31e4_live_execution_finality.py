"""live_execution_finality

Revision ID: 32d70d1e31e4
Revises: bd4b4a245cd0
Create Date: 2026-07-27 00:43:19.452915+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '32d70d1e31e4'
down_revision: Union[str, None] = 'bd4b4a245cd0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


from sqlalchemy.dialects import postgresql

def upgrade() -> None:
    op.add_column("execution_attempts", sa.Column("provider_trade_ids", sa.JSON().with_variant(postgresql.JSONB, "postgresql"), nullable=False, server_default="[]"))
    op.add_column("execution_attempts", sa.Column("transaction_hashes", sa.JSON().with_variant(postgresql.JSONB, "postgresql"), nullable=False, server_default="[]"))
    op.add_column("execution_attempts", sa.Column("settlement_state", sa.String(length=32), nullable=False, server_default="PENDING"))

def downgrade() -> None:
    op.drop_column("execution_attempts", "settlement_state")
    op.drop_column("execution_attempts", "transaction_hashes")
    op.drop_column("execution_attempts", "provider_trade_ids")
