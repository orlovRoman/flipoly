"""add final_outcome check constraint

Revision ID: eb4cc6a7d972
Revises: ea2cc6a7d971
Create Date: 2026-07-26 13:42:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'eb4cc6a7d972'
down_revision = 'ea2cc6a7d971'
branch_labels = None
depends_on = None

from sqlalchemy.sql import text

def upgrade() -> None:
    # 1. Alter flip_vs_final to be nullable
    op.alter_column('market_snapshots', 'flip_vs_final',
               existing_type=sa.BOOLEAN(),
               nullable=True)

    # 2. Normalize existing data
    conn = op.get_bind()
    
    # YES / UP -> YES
    conn.execute(text("UPDATE market_snapshots SET final_outcome = 'YES' WHERE final_outcome IN ('YES', 'UP')"))
    # NO / DOWN -> NO
    conn.execute(text("UPDATE market_snapshots SET final_outcome = 'NO' WHERE final_outcome IN ('NO', 'DOWN')"))
    # INVALID -> INVALID
    conn.execute(text("UPDATE market_snapshots SET final_outcome = 'INVALID' WHERE final_outcome = 'INVALID'"))
    
    # Unknowns -> PENDING
    conn.execute(text("UPDATE market_snapshots SET final_outcome = 'PENDING' WHERE final_outcome NOT IN ('YES', 'NO', 'INVALID', 'PENDING')"))

    # PENDING / INVALID -> flip_vs_final = NULL
    conn.execute(text("UPDATE market_snapshots SET flip_vs_final = NULL WHERE final_outcome IN ('PENDING', 'INVALID')"))

    # 3. Add constraint
    op.create_check_constraint(
        'ck_market_snapshot_outcome',
        'market_snapshots',
        "final_outcome IN ('PENDING', 'YES', 'NO', 'INVALID')"
    )

def downgrade() -> None:
    op.drop_constraint('ck_market_snapshot_outcome', 'market_snapshots', type_='check')
    
    op.alter_column('market_snapshots', 'flip_vs_final',
               existing_type=sa.BOOLEAN(),
               nullable=False)
