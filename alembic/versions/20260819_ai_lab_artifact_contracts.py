"""Complete AI Lab artifact/result contracts without discarding legacy rows.

Revision ID: 20260819_ai_lab_artifact_contracts
Revises: 20260819_ai_lab_research_mode
"""

from alembic import op
import sqlalchemy as sa


revision = "20260819_ai_lab_artifact_contracts"
down_revision = "20260819_ai_lab_research_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The first artifact migration only stored a URI/hash.  These fields are
    # nullable for historical rows, while every new AI Lab TRAIN artifact is
    # written with all links and exact bytes populated.
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def add_if_missing(table: str, column: sa.Column) -> None:
        existing = {item["name"] for item in inspector.get_columns(table)}
        if column.name not in existing:
            op.add_column(table, column)

    add_if_missing("ai_model_artifacts", sa.Column("config_id", sa.Integer(), nullable=True))
    add_if_missing("ai_model_artifacts", sa.Column("run_id", sa.Integer(), nullable=True))
    add_if_missing("ai_model_artifacts", sa.Column("step_id", sa.Integer(), nullable=True))
    add_if_missing("ai_model_artifacts", sa.Column("artifact_bytes", sa.LargeBinary(), nullable=True))
    add_if_missing("ai_model_artifacts", sa.Column("artifact_hash", sa.String(length=64), nullable=True))
    op.execute(
        "UPDATE ai_model_artifacts SET artifact_hash = sha256 "
        "WHERE artifact_hash IS NULL"
    )
    existing_fks = {item.get("name") for item in inspector.get_foreign_keys("ai_model_artifacts")}
    for name, referred, column, ondelete in (
        ("fk_ai_artifacts_config", "ai_experiment_configs", "config_id", "CASCADE"),
        ("fk_ai_artifacts_run", "ai_optimization_runs", "run_id", "SET NULL"),
        ("fk_ai_artifacts_step", "ai_run_steps", "step_id", "SET NULL"),
    ):
        if name not in existing_fks:
            op.create_foreign_key(name, "ai_model_artifacts", referred, [column], ["id"], ondelete=ondelete)

    # Preserve the old experiment_configs reference before moving the active
    # config_id contract to AIExperimentConfig.  Existing result rows remain
    # queryable through legacy_config_id and are never deleted.
    add_if_missing("experiment_results", sa.Column("legacy_config_id", sa.Integer(), nullable=True))
    add_if_missing("experiment_results", sa.Column("step_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE experiment_results SET legacy_config_id = config_id "
        "WHERE legacy_config_id IS NULL"
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE experiment_results DROP CONSTRAINT IF EXISTS "
            "experiment_results_config_id_fkey"
        )
        op.alter_column("experiment_results", "config_id", nullable=True)
        op.execute("UPDATE experiment_results SET config_id = NULL")
        op.create_foreign_key(
            "fk_experiment_results_ai_config",
            "experiment_results",
            "ai_experiment_configs",
            ["config_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        op.create_foreign_key(
            "fk_experiment_results_legacy_config",
            "experiment_results",
            "experiment_configs",
            ["legacy_config_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_foreign_key(
            "fk_experiment_results_step",
            "experiment_results",
            "ai_run_steps",
            ["step_id"],
            ["id"],
            ondelete="SET NULL",
        )
    else:
        # SQLite development databases do not enforce the production FK names.
        # The ORM and PostgreSQL migration above carry the authoritative split;
        # retaining the old SQLite constraint avoids an unsafe table rewrite.
        pass

    op.create_table(
        "ai_shadow_observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("market_id", sa.String(length=128), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active_model_key", sa.String(length=64)),
        sa.Column("candidate_model_key", sa.String(length=64), nullable=False),
        sa.Column("active_action", sa.String(length=32)),
        sa.Column("candidate_action", sa.String(length=32)),
        sa.Column("active_probability", sa.Float()),
        sa.Column("candidate_probability", sa.Float()),
        sa.Column("candidate_ask", sa.Float()),
        sa.Column("active_net_edge", sa.Float()),
        sa.Column("candidate_net_edge", sa.Float()),
        sa.Column("market_outcome", sa.String(length=16)),
        sa.Column("active_pnl", sa.Float()),
        sa.Column("candidate_pnl", sa.Float()),
        sa.Column("lr_direction_vote", sa.String(length=16)),
        sa.Column("lgbm_direction_vote", sa.String(length=16)),
        sa.Column("consensus_type", sa.String(length=32)),
        sa.Column("shadow_logreg_action", sa.String(length=32)),
        sa.Column("actual_combined_action", sa.String(length=32)),
        sa.Column("shadow_logreg_net_edge", sa.Float()),
        sa.Column("actual_net_edge", sa.Float()),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="PENDING"),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("idempotency_key", sa.String(length=256), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["assignment_id"], ["ai_shadow_assignments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["ai_optimization_runs.id"], ondelete="SET NULL"),
        sa.CheckConstraint("status IN ('PENDING', 'RESOLVED', 'ABSTAINED', 'INVALID')", name="ck_ai_shadow_observation_status"),
    )
    op.create_index("idx_ai_shadow_obs_assignment_market", "ai_shadow_observations", ["assignment_id", "market_id"])
    op.create_index("idx_ai_shadow_obs_status", "ai_shadow_observations", ["status", "created_at"])
    op.create_table(
        "ai_experiment_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("step_id", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="QUEUED"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(length=256), nullable=False, unique=True),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("traceback", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["ai_optimization_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["step_id"], ["ai_run_steps.id"], ondelete="CASCADE"),
        sa.CheckConstraint("status IN ('QUEUED', 'RUNNING', 'RETRY_WAIT', 'SUCCEEDED', 'FAILED', 'STALE', 'CANCELLED')", name="ck_ai_experiment_jobs_status"),
    )
    op.create_index("idx_ai_jobs_status_heartbeat", "ai_experiment_jobs", ["status", "heartbeat_at"])
    op.create_index("idx_ai_jobs_run_step", "ai_experiment_jobs", ["run_id", "step_id"])


def downgrade() -> None:
    op.drop_index("idx_ai_jobs_run_step", table_name="ai_experiment_jobs")
    op.drop_index("idx_ai_jobs_status_heartbeat", table_name="ai_experiment_jobs")
    op.drop_table("ai_experiment_jobs")
    op.drop_index("idx_ai_shadow_obs_status", table_name="ai_shadow_observations")
    op.drop_index("idx_ai_shadow_obs_assignment_market", table_name="ai_shadow_observations")
    op.drop_table("ai_shadow_observations")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for name in (
            "fk_experiment_results_step",
            "fk_experiment_results_legacy_config",
            "fk_experiment_results_ai_config",
        ):
            op.drop_constraint(name, "experiment_results", type_="foreignkey")
        op.create_foreign_key(
            "experiment_results_config_id_fkey",
            "experiment_results",
            "experiment_configs",
            ["config_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    for name in ("fk_ai_artifacts_step", "fk_ai_artifacts_run", "fk_ai_artifacts_config"):
        op.drop_constraint(name, "ai_model_artifacts", type_="foreignkey")
    op.drop_column("experiment_results", "step_id")
    op.drop_column("experiment_results", "legacy_config_id")
    op.drop_column("ai_model_artifacts", "artifact_hash")
    op.drop_column("ai_model_artifacts", "artifact_bytes")
    op.drop_column("ai_model_artifacts", "step_id")
    op.drop_column("ai_model_artifacts", "run_id")
    op.drop_column("ai_model_artifacts", "config_id")
