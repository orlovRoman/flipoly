"""Allow Alembic revision identifiers longer than 32 characters."""

import sqlalchemy as sa

from alembic import op

revision = "20260824_widen_alembic_version"
down_revision = "20260824_ai_run_llm_snapshot"
branch_labels = None
depends_on = None


def _alter_version_num(length: int) -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("alembic_version"):
        return
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("alembic_version") as batch_op:
            batch_op.alter_column(
                "version_num",
                existing_type=sa.String(length=32),
                type_=sa.String(length=length),
                existing_nullable=False,
            )
        return
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=32),
        type_=sa.String(length=length),
        existing_nullable=False,
    )


def upgrade() -> None:
    _alter_version_num(255)


def downgrade() -> None:
    _alter_version_num(32)
