"""Migrate favorite settings: replace YES/NO min/max prices with FAVORITE_MIN/MAX_PRICE

Revision ID: f1a2b3c4d5e7
Revises: f1a2b3c4d5e6
Create Date: 2026-07-03 12:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e7'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Сначала вставляем новые, затем удаляем старые
    op.execute("""
        INSERT INTO runtime_settings (key, value, updated_by, updated_at)
        VALUES 
            ('FAVORITE_MIN_PRICE', '0.55', 'migration', NOW()),
            ('FAVORITE_MAX_PRICE', '0.95', 'migration', NOW())
        ON CONFLICT (key) DO NOTHING;
    """)
    op.execute("""
        DELETE FROM runtime_settings
        WHERE key IN ('YES_MIN_PRICE', 'YES_MAX_PRICE', 'NO_MIN_PRICE', 'NO_MAX_PRICE');
    """)


def downgrade() -> None:
    # Откатываем: восстанавливаем старые, удаляем новые
    op.execute("""
        INSERT INTO runtime_settings (key, value, updated_by, updated_at)
        VALUES
            ('YES_MIN_PRICE', '0.55', 'migration', NOW()),
            ('YES_MAX_PRICE', '0.95', 'migration', NOW()),
            ('NO_MIN_PRICE',  '0.55', 'migration', NOW()),
            ('NO_MAX_PRICE',  '0.95', 'migration', NOW())
        ON CONFLICT (key) DO NOTHING;
    """)
    op.execute("""
        DELETE FROM runtime_settings
        WHERE key IN ('FAVORITE_MIN_PRICE', 'FAVORITE_MAX_PRICE');
    """)
