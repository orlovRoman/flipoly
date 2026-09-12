"""Make the PAPER fee coefficient agree with the Polymarket price curve.

Legacy deployments used ``PAPER_FEE_RATE=0.002`` while the
``POLYMARKET_PRICE_DEPENDENT`` model expects the curve coefficient ``0.07``.
Only rows that still contain the legacy default are migrated; an operator's
other explicit value is left untouched.
"""

from alembic import op
import sqlalchemy as sa


revision = "20260913_canonical_paper_fee"
down_revision = "20260911_rtds_observations"
branch_labels = None
depends_on = None

_MARKER = "migration:20260913_canonical_paper_fee"


def upgrade() -> None:
    bind = op.get_bind()
    row = bind.execute(
        sa.text(
            "SELECT value FROM runtime_settings "
            "WHERE key = 'PAPER_FEE_RATE'"
        )
    ).first()
    if row is None:
        bind.execute(
            sa.text(
                "INSERT INTO runtime_settings "
                "(key, value, updated_at, updated_by) "
                "VALUES ('PAPER_FEE_RATE', '0.07', CURRENT_TIMESTAMP, :marker)"
            ),
            {"marker": _MARKER},
        )
        return

    # Do not overwrite a non-legacy operator choice.
    if str(row[0]).strip() in {"0.002", "0.0020", "0.00200"}:
        bind.execute(
            sa.text(
                "UPDATE runtime_settings SET value = '0.07', "
                "updated_at = CURRENT_TIMESTAMP, updated_by = :marker "
                "WHERE key = 'PAPER_FEE_RATE' AND value IN "
                "('0.002', '0.0020', '0.00200')"
            ),
            {"marker": _MARKER},
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE runtime_settings SET value = '0.002', "
            "updated_at = CURRENT_TIMESTAMP, updated_by = 'migration:downgrade' "
            "WHERE key = 'PAPER_FEE_RATE' AND value = '0.07' "
            "AND updated_by = :marker"
        ),
        {"marker": _MARKER},
    )

