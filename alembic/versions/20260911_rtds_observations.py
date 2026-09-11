"""RTDS streaming storage: keeper observations, raw journal, conflicts, sessions,
daily volume aggregates, and journal archive registry.

Revision ID: 20260911_rtds_observations
Revises: 20260910_ct_decision_reservations

Parent status (review item 1): the local chain resolves to a SINGLE head ending
here (verified via ScriptDirectory, not the space-broken CLI). No merge needed
locally. Before prod deploy, confirm the server's alembic_version matches this
line (coverage_probe.py reports it read-only); do not auto-upgrade.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260911_rtds_observations"
down_revision = "20260910_ct_decision_reservations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rtds_stream_sessions",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "topics",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column(
            "symbols",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column(
            "reconnect_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "disconnect_windows",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column("last_error", sa.String(length=256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_table(
        "rtds_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("topic", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("asset", sa.String(length=16), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column(
            "window_s", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("raw_value_text", sa.String(length=128), nullable=False),
        sa.Column("raw_e18", sa.Numeric(38, 0), nullable=False),
        sa.Column("value_source", sa.String(length=16), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "extra_data",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source", "symbol", "window_s", "observed_at", name="uix_rtds_obs_keeper"
        ),
    )
    op.create_index(
        "idx_rtds_obs_lookup", "rtds_observations", ["source", "symbol", "observed_at"]
    )
    op.create_index(
        "idx_rtds_obs_received", "rtds_observations", ["symbol", "received_at"]
    )
    op.create_index("idx_rtds_obs_session", "rtds_observations", ["session_id"])
    op.create_table(
        "rtds_raw_journal",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("topic", sa.String(length=64), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_rtds_journal_session_time",
        "rtds_raw_journal",
        ["session_id", "received_at"],
    )
    op.create_index(
        "idx_rtds_journal_topic", "rtds_raw_journal", ["topic", "received_at"]
    )
    op.create_table(
        "rtds_conflicts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column(
            "window_s", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("keeper_raw_e18", sa.Numeric(38, 0), nullable=False),
        sa.Column("keeper_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rejected_raw_e18", sa.Numeric(38, 0), nullable=False),
        sa.Column("rejected_value_text", sa.String(length=128), nullable=False),
        sa.Column("rejected_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source",
            "symbol",
            "window_s",
            "observed_at",
            "rejected_raw_e18",
            "rejected_received_at",
            name="uix_rtds_conflict_once",
        ),
    )
    op.create_index(
        "idx_rtds_conflicts_key",
        "rtds_conflicts",
        ["source", "symbol", "window_s", "observed_at"],
    )
    op.create_table(
        "rtds_daily_volume",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("topic", sa.String(length=64), nullable=False),
        sa.Column(
            "messages", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "bytes", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "events", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("day", "topic", name="uix_rtds_daily_volume"),
    )
    op.create_table(
        "rtds_journal_archives",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("part", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=False),
        sa.Column(
            "max_id", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "archived_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("day", "part", name="uix_rtds_journal_archive_day_part"),
    )


def downgrade() -> None:
    op.drop_table("rtds_journal_archives")
    op.drop_table("rtds_daily_volume")
    op.drop_index("idx_rtds_conflicts_key", table_name="rtds_conflicts")
    op.drop_table("rtds_conflicts")
    op.drop_index("idx_rtds_journal_topic", table_name="rtds_raw_journal")
    op.drop_index("idx_rtds_journal_session_time", table_name="rtds_raw_journal")
    op.drop_table("rtds_raw_journal")
    op.drop_index("idx_rtds_obs_session", table_name="rtds_observations")
    op.drop_index("idx_rtds_obs_received", table_name="rtds_observations")
    op.drop_index("idx_rtds_obs_lookup", table_name="rtds_observations")
    op.drop_table("rtds_observations")
    op.drop_table("rtds_stream_sessions")
