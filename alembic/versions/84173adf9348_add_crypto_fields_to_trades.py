"""add_crypto_fields_to_trades

Revision ID: 84173adf9348
Revises: f7b5ac2856cb
Create Date: 2026-07-04 16:23:50.416172+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84173adf9348'
down_revision: Union[str, None] = 'f7b5ac2856cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trade_history", sa.Column("p_up", sa.Float(), nullable=True))
    op.add_column("trade_history", sa.Column("strike", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("trade_history", "p_up")
    op.drop_column("trade_history", "strike")

