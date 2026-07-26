"""fix exposure_reservations and backfill

Revision ID: d1e2f3g4h5i6
Revises: c1d2e3f4g5h6
Create Date: 2026-07-26 22:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd1e2f3g4h5i6'
down_revision = 'c1d2e3f4g5h6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Снять старый CHECK.
    op.drop_constraint(
        "ck_execution_request_trade_reference",
        "execution_requests",
        type_="check",
    )

    conn = op.get_bind()
    
    # 2. Backfill OPEN in execution_requests
    conn.execute(sa.text("""
        UPDATE execution_requests er
        SET trade_history_id = split_part(er.idempotency_key, ':', 2)::integer
        WHERE er.trade_history_id IS NULL
          AND er.intent = 'OPEN'
          AND er.idempotency_key ~ '^OPEN:[0-9]+$'
          AND EXISTS (
              SELECT 1 FROM trade_history th
              WHERE th.id = split_part(er.idempotency_key, ':', 2)::integer
          )
    """))

    # 3. Check for invalid execution requests
    invalid = conn.scalar(sa.text("""
        SELECT count(*)
        FROM execution_requests
        WHERE trade_history_id IS NULL OR intent NOT IN ('OPEN', 'CLOSE')
    """))

    if invalid:
        raise RuntimeError(
            f"ExecutionRequest backfill incomplete: {invalid} invalid rows"
        )

    # 4. Make trade_history_id NOT NULL and add CHECK
    op.alter_column(
        "execution_requests",
        "trade_history_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.create_check_constraint(
        "ck_execution_request_intent",
        "execution_requests",
        "intent IN ('OPEN', 'CLOSE')",
    )

    # 5. Restore nullable fields for exposure_reservations
    op.add_column(
        "exposure_reservations",
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "exposure_reservations",
        sa.Column("trade_history_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "exposure_reservations",
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("exposure_reservations", "released_at")
    op.drop_column("exposure_reservations", "trade_history_id")
    op.drop_column("exposure_reservations", "request_id")

    op.drop_constraint("ck_execution_request_intent", "execution_requests", type_="check")
    op.alter_column(
        "execution_requests",
        "trade_history_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.create_check_constraint(
        "ck_execution_request_trade_reference",
        "execution_requests",
        "(intent = 'OPEN' AND trade_history_id IS NULL) OR (intent = 'CLOSE' AND trade_history_id IS NOT NULL)",
    )
