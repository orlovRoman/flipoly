"""Reconcile persisted tables with the AI Lab ORM union.

Revision ID: 20260814_schema_compat
Revises: 20260814_rt_settings_desc
"""

from alembic import op
import sqlalchemy as sa

from polyflip.db.models import Base


revision = "20260814_schema_compat"
down_revision = "20260814_rt_settings_desc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    added_live_market_id = False

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue
        existing_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        for column in table.columns:
            if column.name in existing_columns:
                continue
            # Existing rows make a new NOT NULL column unsafe. Runtime code
            # already treats these compatibility fields as optional.
            op.add_column(
                table_name,
                sa.Column(column.name, column.type, nullable=True),
            )
            if table_name == "live_markets" and column.name == "id":
                added_live_market_id = True

    if added_live_market_id:
        op.execute("CREATE SEQUENCE IF NOT EXISTS live_markets_id_seq")
        op.execute(
            "ALTER TABLE live_markets "
            "ALTER COLUMN id SET DEFAULT nextval('live_markets_id_seq')"
        )
        op.execute(
            "SELECT setval("
            "'live_markets_id_seq', "
            "COALESCE((SELECT MAX(id) FROM live_markets), 0) + 1, false)"
        )
        op.execute(
            "UPDATE live_markets "
            "SET id = nextval('live_markets_id_seq') WHERE id IS NULL"
        )
        op.execute(
            "ALTER TABLE live_markets ALTER COLUMN id SET NOT NULL"
        )
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_live_markets_id_compat ON live_markets (id)"
        )


def downgrade() -> None:
    # Compatibility columns are intentionally retained on downgrade. They
    # may contain live telemetry and removing them would be destructive.
    pass
