"""Migrate outsider settings: replace 4 keys with OUTSIDER_MAX_PRICE

Revision ID: f1a2b3c4d5e6
Revises: e1cf7d68b1b6
Create Date: 2026-07-02 16:36:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e1cf7d68b1b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Удаляем 4 старых параметра аутсайдера
    op.execute("""
        DELETE FROM runtime_settings
        WHERE key IN (
            'OUTSIDER_NO_MIN_PRICE',
            'OUTSIDER_NO_MAX_PRICE',
            'OUTSIDER_YES_MIN_PRICE',
            'OUTSIDER_YES_MAX_PRICE'
        )
    """)

    # Добавляем единый новый параметр (если уже есть — не трогаем)
    op.execute("""
        INSERT INTO runtime_settings (key, value, updated_by, updated_at)
        VALUES ('OUTSIDER_MAX_PRICE', '0.45', 'migration', NOW())
        ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    # Откатываем: удаляем новый ключ, восстанавливаем 4 старых
    op.execute("DELETE FROM runtime_settings WHERE key = 'OUTSIDER_MAX_PRICE'")

    op.execute("""
        INSERT INTO runtime_settings (key, value, updated_by, updated_at)
        VALUES
            ('OUTSIDER_NO_MIN_PRICE',  '0.10', 'migration', NOW()),
            ('OUTSIDER_NO_MAX_PRICE',  '0.45', 'migration', NOW()),
            ('OUTSIDER_YES_MIN_PRICE', '0.05', 'migration', NOW()),
            ('OUTSIDER_YES_MAX_PRICE', '0.45', 'migration', NOW())
        ON CONFLICT (key) DO NOTHING
    """)
