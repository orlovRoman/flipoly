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
    op.add_column(
        "ai_model_artifacts",
        sa.Column("config_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ai_model_artifacts",
        sa.Column("run_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ai_model_artifacts",
        sa.Column("step_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ai_model_artifacts",
        sa.Column("artifact_bytes", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        "ai_model_artifacts",
        sa.Column("artifact_hash", sa.String(length=64), nullable=True),
    )
    op.execute(
        "UPDATE ai_model_artifacts SET artifact_hash = sha256 "
        "WHERE artifact_hash IS NULL"
    )
    op.create_foreign_key(
        "fk_ai_artifacts_config",
        "ai_model_artifacts",
        "ai_experiment_configs",
        ["config_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_ai_artifacts_run",
        "ai_model_artifacts",
        "ai_optimization_runs",
        ["run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_ai_artifacts_step",
        "ai_model_artifacts",
        "ai_run_steps",
        ["step_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Preserve the old experiment_configs reference before moving the active
    # config_id contract to AIExperimentConfig.  Existing result rows remain
    # queryable through legacy_config_id and are never deleted.
    op.add_column(
        "experiment_results",
        sa.Column("legacy_config_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "experiment_results",
        sa.Column("step_id", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE experiment_results SET legacy_config_id = config_id "
        "WHERE legacy_config_id IS NULL"
    )
    op.execute("UPDATE experiment_results SET config_id = NULL")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE experiment_results DROP CONSTRAINT IF EXISTS "
            "experiment_results_config_id_fkey"
        )
        op.alter_column("experiment_results", "config_id", nullable=True)
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


def downgrade() -> None:
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
