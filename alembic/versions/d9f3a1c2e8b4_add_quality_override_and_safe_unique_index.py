"""add_quality_override_and_safe_unique_index

Revision ID: d9f3a1c2e8b4
Revises: c8f1e9a2b7d3
Create Date: 2026-08-03

Добавляет колонку quality_override в model_registry.

Переименовывает устаревшие значения activation_source:
    AUTO   -> TRAINER  (активировал trainer)
    MANUAL -> DASHBOARD (активировал пользователь из дашборда)

Безопасно создаёт уникальный индекс uq_model_registry_one_active_version:
    Перед созданием проверяет и нормализует ситуации, когда у одного asset
    несколько записей с is_active=TRUE (оставляем только новейшую по id).
"""
from alembic import op
import sqlalchemy as sa

revision = 'd9f3a1c2e8b4'
down_revision = 'c8f1e9a2b7d3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Добавляем quality_override (баг #4)
    op.add_column('model_registry',
        sa.Column('quality_override', sa.Boolean(), nullable=True, server_default='false'))

    # 2. Переименовываем старые значения activation_source (баг #4)
    #    AUTO   -> TRAINER  (автоматическая активация тренером)
    #    MANUAL -> DASHBOARD (ручная через дашборд с override)
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'TRAINER'
        WHERE activation_source = 'AUTO'
    """)
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'DASHBOARD',
            quality_override = TRUE
        WHERE activation_source = 'MANUAL'
    """)

    # 3. Нормализуем дублирующиеся активные версии (баг #5)
    #    Если у одного asset > 1 активной версии — оставляем только с max(id),
    #    остальные деактивируем. Это предотвращает ошибку CREATE UNIQUE INDEX.
    op.execute("""
        UPDATE model_registry m
        SET is_active = FALSE
        WHERE is_active = TRUE
          AND id NOT IN (
              SELECT MAX(id)
              FROM model_registry
              WHERE is_active = TRUE
              GROUP BY asset
          )
    """)

    # 4. Теперь можно безопасно создать уникальный индекс
    #    IF NOT EXISTS — идемпотентно, если уже применён через предыдущую миграцию
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_one_active_version
        ON model_registry (asset)
        WHERE is_active IS TRUE
    """)

    # 5. Backfill: активные записи без activation_source → TRAINER (legacy)
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'TRAINER',
            quality_gate_passed = TRUE
        WHERE is_active = TRUE AND activation_source IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_model_registry_one_active_version")

    # Откатываем переименование
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'AUTO'
        WHERE activation_source = 'TRAINER'
    """)
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'MANUAL'
        WHERE activation_source = 'DASHBOARD'
    """)

    op.drop_column('model_registry', 'quality_override')
