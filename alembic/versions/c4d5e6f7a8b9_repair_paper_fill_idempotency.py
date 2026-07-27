"""repair PAPER fill idempotency and recover stuck requests

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-07-27 13:00:00.000000+00:00

The ORM declared uq_execution_provider_trade, but the historical Alembic
migrations only added the two columns. PostgreSQL therefore rejected
ON CONFLICT (gateway, provider_trade_id), leaving PAPER requests in SUBMITTING.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: Union[str, None] = "b3c4d5e6f7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_RECOVERY_REASON = "Recovered after missing execution fill idempotency constraint"


def _has_unique_constraint(inspector: sa.Inspector, table: str, name: str) -> bool:
    return any(
        item.get("name") == name for item in inspector.get_unique_constraints(table)
    )


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name
    now = datetime.now(timezone.utc)

    # Keep the oldest copy of an already duplicated provider fill. This is
    # required before adding the unique constraint on an existing database.
    if dialect == "postgresql":
        conn.execute(sa.text("""
                WITH ranked AS (
                    SELECT
                        ctid,
                        row_number() OVER (
                            PARTITION BY gateway, provider_trade_id
                            ORDER BY timestamp ASC, id ASC
                        ) AS duplicate_no
                    FROM execution_fills
                    WHERE gateway IS NOT NULL
                      AND provider_trade_id IS NOT NULL
                )
                DELETE FROM execution_fills AS fill
                USING ranked
                WHERE fill.ctid = ranked.ctid
                  AND ranked.duplicate_no > 1
                """))

    inspector = sa.inspect(conn)
    if not _has_unique_constraint(
        inspector, "execution_fills", "uq_execution_provider_trade"
    ):
        op.create_unique_constraint(
            "uq_execution_provider_trade",
            "execution_fills",
            ["gateway", "provider_trade_id"],
        )

    # Attempts that could not persist a fill never reached an external venue
    # in PAPER mode. Mark the broken attempt, then retry only recent requests.
    conn.execute(
        sa.text("""
            UPDATE execution_attempts AS attempt
            SET status = 'FAILED',
                finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
                error_msg = :reason
            FROM execution_requests AS request
            WHERE attempt.request_id = request.id
              AND request.requested_mode = 'PAPER'
              AND request.state IN (
                  'SUBMITTING', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
              )
              AND attempt.provider_order_id IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_fills AS fill
                  WHERE fill.attempt_id = attempt.id
              )
            """),
        {"reason": _RECOVERY_REASON},
    )

    conn.execute(
        sa.text("""
            UPDATE execution_requests AS request
            SET state = 'READY',
                claimed_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                expires_at = :retry_expires_at,
                updated_at = :now,
                error_reason = :reason
            WHERE request.requested_mode = 'PAPER'
              AND request.state IN (
                  'SUBMITTING', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
              )
              AND request.created_at >= :recent_cutoff
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_attempts AS attempt
                  WHERE attempt.request_id = request.id
                    AND attempt.provider_order_id IS NOT NULL
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_attempts AS attempt
                  JOIN execution_fills AS fill
                    ON fill.attempt_id = attempt.id
                  WHERE attempt.request_id = request.id
              )
            """),
        {
            "reason": _RECOVERY_REASON,
            "now": now,
            "recent_cutoff": now - timedelta(minutes=15),
            "retry_expires_at": now + timedelta(minutes=10),
        },
    )

    # Old virtual requests must not be filled against an expired market.
    # Finalize them explicitly so the dashboard no longer shows PENDING.
    conn.execute(
        sa.text("""
            UPDATE execution_requests AS request
            SET state = 'REJECTED',
                updated_at = CURRENT_TIMESTAMP,
                error_reason = :reason
            WHERE request.requested_mode = 'PAPER'
              AND request.state IN (
                  'SUBMITTING', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_attempts AS attempt
                  WHERE attempt.request_id = request.id
                    AND attempt.provider_order_id IS NOT NULL
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_attempts AS attempt
                  JOIN execution_fills AS fill
                    ON fill.attempt_id = attempt.id
                  WHERE attempt.request_id = request.id
              )
            """),
        {"reason": f"{_RECOVERY_REASON}; request expired"},
    )

    conn.execute(
        sa.text("""
            UPDATE trade_history AS trade
            SET status = 'FAILED',
                position_status = 'ENTRY_FAILED',
                error_msg = request.error_reason
            FROM execution_requests AS request
            WHERE request.trade_history_id = trade.id
              AND request.requested_mode = 'PAPER'
              AND request.intent = 'OPEN'
              AND request.state = 'REJECTED'
              AND request.error_reason LIKE :reason_prefix
            """),
        {"reason_prefix": f"{_RECOVERY_REASON}%"},
    )

    conn.execute(
        sa.text("""
            UPDATE exposure_reservations AS reservation
            SET released_at = COALESCE(released_at, CURRENT_TIMESTAMP)
            FROM execution_requests AS request
            WHERE reservation.request_id = request.id
              AND request.state = 'REJECTED'
              AND request.error_reason LIKE :reason_prefix
            """),
        {"reason_prefix": f"{_RECOVERY_REASON}%"},
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_execution_provider_trade",
        "execution_fills",
        type_="unique",
    )
