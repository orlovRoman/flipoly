"""add_model_activation_audit

Revision ID: a1b2c3d4e5f6
Revises: 783e6d92bc81
Create Date: 2026-08-03

Добавляет в model_registry поля Quality Gate и Activation Audit:
    quality_gate_passed, quality_gate_reasons,
    activation_source, activated_at, activated_by, activation_reason

А также частичный уникальный индекс, гарантирующий,
что у каждого asset одновременно активна не более одной версии.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = '783e6d92bc81'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Quality Gate columns
    op.add_column('model_registry',
        sa.Column('quality_gate_passed', sa.Boolean(), nullable=True))
    op.add_column('model_registry',
        sa.Column('quality_gate_reasons', sa.JSON(), nullable=True))

    # Activation Audit columns
    op.add_column('model_registry',
        sa.Column('activation_source', sa.String(16), nullable=True))
    op.add_column('model_registry',
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('model_registry',
        sa.Column('activated_by', sa.String(128), nullable=True))
    op.add_column('model_registry',
        sa.Column('activation_reason', sa.Text(), nullable=True))

    # Частичный уникальный индекс: одна активная версия на asset
    # Используем raw SQL т.к. SQLAlchemy не поддерживает partial index в op.create_index
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_one_active_version
        ON model_registry (asset)
        WHERE is_active IS TRUE
    """)

    # Backfill: все существующие активные модели помечаем как AUTO
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'AUTO',
            quality_gate_passed = TRUE
        WHERE is_active = TRUE AND activation_source IS NULL
    """)

    # Неактивные — ставим quality_gate_passed = FALSE (консервативно, мы не знаем точно)
    # Оставляем NULL, т.к. реально не знаем причину деактивации
    # NULL = "данные до внедрения аудита"


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_model_registry_one_active_version")
    op.drop_column('model_registry', 'activation_reason')
    op.drop_column('model_registry', 'activated_by')
    op.drop_column('model_registry', 'activated_at')
    op.drop_column('model_registry', 'activation_source')
    op.drop_column('model_registry', 'quality_gate_reasons')
    op.drop_column('model_registry', 'quality_gate_passed')
