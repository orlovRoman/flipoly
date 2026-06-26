"""Add token_ids to live_markets

Revision ID: 003
Revises: 002
Create Date: 2026-06-25 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add columns with a default value to prevent breaking existing rows, then alter to drop default
    op.add_column('live_markets', sa.Column('yes_token_id', sa.String(length=128), server_default='N/A', nullable=False))
    op.add_column('live_markets', sa.Column('no_token_id', sa.String(length=128), server_default='N/A', nullable=False))
    
    # Remove the server default now that rows are populated
    op.alter_column('live_markets', 'yes_token_id', server_default=None, existing_type=sa.String(length=128), existing_nullable=False)
    op.alter_column('live_markets', 'no_token_id', server_default=None, existing_type=sa.String(length=128), existing_nullable=False)

def downgrade() -> None:
    pass
