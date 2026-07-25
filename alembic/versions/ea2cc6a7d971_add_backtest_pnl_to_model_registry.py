"""add_backtest_pnl_to_model_registry

Revision ID: ea2cc6a7d971
Revises: d470f6f21b90
Create Date: 2026-07-25 06:38:42.645385+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ea2cc6a7d971'
down_revision: Union[str, None] = 'd470f6f21b90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("backtest_pnl", sa.Float(), nullable=True))
    op.add_column("model_registry", sa.Column("backtest_trades", sa.Integer(), nullable=True))
    op.add_column("model_registry", sa.Column("backtest_wr", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_registry", "backtest_wr")
    op.drop_column("model_registry", "backtest_trades")
    op.drop_column("model_registry", "backtest_pnl")
