import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '12845e0151a9'
down_revision = 'f47295cae843'
branch_labels = None
depends_on = None

def upgrade() -> None:
    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.add_column(sa.Column('execution_mode', sa.String(length=16), nullable=True))
        batch_op.alter_column('trading_mode',
               existing_type=sa.VARCHAR(length=32),
               type_=sa.String(length=16),
               nullable=True)

def downgrade() -> None:
    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.alter_column('trading_mode',
               existing_type=sa.String(length=16),
               type_=sa.VARCHAR(length=32),
               nullable=False)
        batch_op.drop_column('execution_mode')
