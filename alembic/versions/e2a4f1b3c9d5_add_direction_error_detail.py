"""add_direction_error_detail

Revision ID: e2a4f1b3c9d5
Revises: d9f3a1c2e8b4
Create Date: 2026-08-03

Добавляет direction_error_detail в decision_funnel_log:
  - хранит текст исключения при INFERENCE_FAILED
  - хранит недостающий ключ режима при REGIME_UNAVAILABLE
  - хранит причину риска при risk_vetoed

P0-фикс: теперь UI видит конкретную причину сбоя Direction Model.
"""
from alembic import op
import sqlalchemy as sa

revision = 'e2a4f1b3c9d5'
down_revision = 'd9f3a1c2e8b4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'decision_funnel_log',
        sa.Column('direction_error_detail', sa.String(512), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('decision_funnel_log', 'direction_error_detail')
