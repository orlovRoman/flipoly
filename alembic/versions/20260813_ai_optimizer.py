"""add autonomous AI optimizer laboratory foundation

Revision ID: 20260813_ai_optimizer
Revises: 20260813_lgbm_training_jobs
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_ai_optimizer"
down_revision = "20260813_lgbm_training_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_optimization_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("autonomy_level", sa.String(length=24), nullable=False, server_default="EXPERIMENT"),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="DRAFT"),
        sa.Column("agent_type", sa.String(length=32), nullable=False, server_default="CODEX"),
        sa.Column("agent_thread_id", sa.String(length=128), nullable=True),
        sa.Column("budget_experiments", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("budget_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(length=128), nullable=False, server_default="system"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'LIVE_PROPOSE')",
            name="ck_ai_runs_autonomy_level",
        ),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'PLANNING', 'RUNNING', 'EVALUATING', 'SHADOW', "
            "'PENDING_APPROVAL', 'ACTIVE', 'INSUFFICIENT_DATA', 'FAILED', "
            "'REJECTED', 'CANCELLED', 'ROLLED_BACK')",
            name="ck_ai_runs_status",
        ),
    )
    op.create_index(
        "idx_ai_runs_status_created",
        "ai_optimization_runs",
        ["status", "created_at"],
    )

    op.create_table(
        "ai_run_steps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=True),
        sa.Column("input_payload", sa.JSON(), nullable=True),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("run_id", "step_index", name="uix_ai_run_step_index"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED')",
            name="ck_ai_run_steps_status",
        ),
    )
    op.create_index(
        "idx_ai_run_steps_run_status",
        "ai_run_steps",
        ["run_id", "status"],
    )

    op.create_table(
        "experiment_configs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("asset", sa.String(length=32), nullable=True),
        sa.Column("regime", sa.String(length=32), nullable=True),
        sa.Column("model_family", sa.String(length=32), nullable=False),
        sa.Column("feature_set", sa.String(length=32), nullable=False),
        sa.Column("feature_pipeline_version", sa.String(length=64), nullable=False),
        sa.Column("model_params", sa.JSON(), nullable=False),
        sa.Column("strategy_params", sa.JSON(), nullable=False),
        sa.Column("backtest_params", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("experiment_configs.id"),
            nullable=True,
        ),
        sa.Column("created_by", sa.String(length=128), nullable=False, server_default="system"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_ai_experiment_configs_scope",
        "experiment_configs",
        ["asset", "regime", "model_family"],
    )
    op.create_index(
        "idx_ai_experiment_configs_created_at",
        "experiment_configs",
        ["created_at"],
    )

    op.create_table(
        "ai_model_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "model_registry_id",
            sa.Integer(),
            sa.ForeignKey("model_registry.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("artifact_uri", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("feature_pipeline_version", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("loadability_status", sa.String(length=16), nullable=False, server_default="UNVERIFIED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "loadability_status IN ('UNVERIFIED', 'VALID', 'INVALID')",
            name="ck_ai_artifacts_loadability_status",
        ),
    )
    op.create_index(
        "idx_ai_artifacts_registry",
        "ai_model_artifacts",
        ["model_registry_id"],
    )

    op.create_table(
        "experiment_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "config_id",
            sa.Integer(),
            sa.ForeignKey("experiment_configs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "artifact_id",
            sa.Integer(),
            sa.ForeignKey("ai_model_artifacts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("evaluation_kind", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="PENDING"),
        sa.Column("code_sha", sa.String(length=64), nullable=True),
        sa.Column("dataset_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("train_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("oot_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("oot_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("slices", sa.JSON(), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "evaluation_kind IN ('TRAIN', 'OOT', 'POLYMARKET_OOT', 'SHADOW', 'LIVE')",
            name="ck_ai_results_evaluation_kind",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INSUFFICIENT_DATA')",
            name="ck_ai_results_status",
        ),
    )
    op.create_index(
        "idx_ai_results_run_status",
        "experiment_results",
        ["run_id", "status"],
    )
    op.create_index(
        "idx_ai_results_config_kind",
        "experiment_results",
        ["config_id", "evaluation_kind"],
    )

    op.create_table(
        "deployment_revisions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("revision_key", sa.String(length=64), nullable=False, unique=True),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("deployment_revisions.id"),
            nullable=True,
        ),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="DRAFT"),
        sa.Column("created_by", sa.String(length=128), nullable=False, server_default="system"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'REJECTED', 'ROLLED_BACK')",
            name="ck_deployment_revisions_status",
        ),
    )
    op.create_index(
        "idx_deployment_revisions_status_created",
        "deployment_revisions",
        ["status", "created_at"],
    )

    op.create_table(
        "deployment_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "revision_id",
            sa.Integer(),
            sa.ForeignKey("deployment_revisions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("previous_hash", sa.String(length=64), nullable=True),
        sa.Column("event_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('CREATED', 'SHADOW_ASSIGNED', 'APPROVED', 'ACTIVATED', 'REJECTED', 'ROLLED_BACK')",
            name="ck_deployment_events_type",
        ),
    )
    op.create_index(
        "idx_deployment_events_revision_created",
        "deployment_events",
        ["revision_id", "created_at"],
    )

    op.create_table(
        "ai_shadow_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "candidate_artifact_id",
            sa.Integer(),
            sa.ForeignKey("ai_model_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "baseline_artifact_id",
            sa.Integer(),
            sa.ForeignKey("ai_model_artifacts.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'STOPPED', 'FAILED')",
            name="ck_ai_shadow_status",
        ),
    )
    op.create_index(
        "idx_ai_shadow_scope_status",
        "ai_shadow_assignments",
        ["asset", "regime", "status"],
    )

    op.create_table(
        "ai_permissions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("profile_name", sa.String(length=64), nullable=False, unique=True),
        sa.Column("allowed_actions", sa.JSON(), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("limits", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_by", sa.String(length=128), nullable=False, server_default="system"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "ai_approval_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("requested_action", sa.String(length=32), nullable=False),
        sa.Column("diff", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(length=128), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED')",
            name="ck_ai_approval_status",
        ),
    )
    op.create_index(
        "idx_ai_approval_status_requested",
        "ai_approval_requests",
        ["status", "requested_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_ai_approval_status_requested", table_name="ai_approval_requests")
    op.drop_table("ai_approval_requests")
    op.drop_table("ai_permissions")
    op.drop_index("idx_ai_shadow_scope_status", table_name="ai_shadow_assignments")
    op.drop_table("ai_shadow_assignments")
    op.drop_index("idx_deployment_events_revision_created", table_name="deployment_events")
    op.drop_table("deployment_events")
    op.drop_index("idx_deployment_revisions_status_created", table_name="deployment_revisions")
    op.drop_table("deployment_revisions")
    op.drop_index("idx_ai_results_config_kind", table_name="experiment_results")
    op.drop_index("idx_ai_results_run_status", table_name="experiment_results")
    op.drop_table("experiment_results")
    op.drop_index("idx_ai_artifacts_registry", table_name="ai_model_artifacts")
    op.drop_table("ai_model_artifacts")
    op.drop_index("idx_ai_experiment_configs_created_at", table_name="experiment_configs")
    op.drop_index("idx_ai_experiment_configs_scope", table_name="experiment_configs")
    op.drop_table("experiment_configs")
    op.drop_index("idx_ai_run_steps_run_status", table_name="ai_run_steps")
    op.drop_table("ai_run_steps")
    op.drop_index("idx_ai_runs_status_created", table_name="ai_optimization_runs")
    op.drop_table("ai_optimization_runs")
