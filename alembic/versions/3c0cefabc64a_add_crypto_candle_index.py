"""add_crypto_candle_index

Revision ID: 3c0cefabc64a
Revises: 84173adf9348
Create Date: 2026-07-04 16:47:23.149113+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c0cefabc64a'
down_revision: Union[str, None] = '84173adf9348'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_crypto_candle_symbol_interval_time",
        "crypto_candle",
        ["symbol", "interval", sa.text("open_time DESC")],
        postgresql_using="btree",
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_crypto_candle_symbol_interval_time", table_name="crypto_candle")

