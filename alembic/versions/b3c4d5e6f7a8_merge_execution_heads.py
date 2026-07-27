"""merge_execution_heads_and_repair_close_index

Merge-миграция: объединяет две независимые ветки от 32d70d1e31e4:
  - 41c8d1f7c5e2 (fix_state_indexes)
  - a2b3c4d5e6f7 (fix_close_unique_index)

В upgrade():
  1. Проверяет отсутствие активных CLOSE-дубликатов по trade_history_id.
  2. Безопасно пересоздаёт uq_active_close_request:
     - Дропает существующий индекс (любого вида).
     - Создаёт новый: UNIQUE по (trade_history_id) с WHERE-предикатом.

Это корректно работает во всех 4 сценариях:
  1. Новая БД             → применяется: 41c8 → a2b3 → b3c4
  2. БД только с 41c8     → применяется: a2b3 → b3c4
  3. БД только с a2b3     → применяется: b3c4
  4. БД с обеими строками → применяется: b3c4

После alembic upgrade head: ровно одна строка b3c4d5e6f7a8 в alembic_version.

Revision ID: b3c4d5e6f7a8
Revises: 41c8d1f7c5e2, a2b3c4d5e6f7
Create Date: 2026-07-27 12:00:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: Union[tuple[str, str], None] = ("41c8d1f7c5e2", "a2b3c4d5e6f7")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Предикат активных CLOSE-заявок — используется для partial index
_ACTIVE_CLOSE_SQL = (
    "intent = 'CLOSE' AND state IN ("
    "'AWAITING_APPROVAL', 'READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED',"
    " 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'"
    ")"
)


def upgrade() -> None:
    conn = op.get_bind()

    # --- 1. Защита: активные CLOSE-дубликаты по trade_history_id ---
    duplicates = conn.execute(sa.text(f"""
            SELECT trade_history_id, count(*) AS cnt
            FROM execution_requests
            WHERE {_ACTIVE_CLOSE_SQL}
            GROUP BY trade_history_id
            HAVING count(*) > 1
            """)).fetchall()

    if duplicates:
        rows = [(r[0], r[1]) for r in duplicates]
        raise RuntimeError(
            f"Migration aborted: {len(rows)} trade(s) have multiple active CLOSE requests. "
            f"Resolve manually before migrating. Duplicates: {rows}"
        )

    # --- 2. Безопасный дроп существующего CLOSE-индекса ---
    dialect = conn.dialect.name
    if dialect == "postgresql":
        conn.execute(sa.text("DROP INDEX IF EXISTS uq_active_close_request"))
    else:
        # SQLite: пробуем через op.drop_index, игнорируем если нет
        try:
            op.drop_index("uq_active_close_request", table_name="execution_requests")
        except Exception:
            pass

    # --- 3. Создаём правильный CLOSE-индекс: UNIQUE по trade_history_id ---
    op.create_index(
        "uq_active_close_request",
        "execution_requests",
        ["trade_history_id"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_CLOSE_SQL),
        sqlite_where=sa.text(_ACTIVE_CLOSE_SQL),
    )


def downgrade() -> None:
    # Merge-миграция не откатывается — откат меняет структуру головы
    # и создаёт снова два head, что опасно в production.
    raise NotImplementedError(
        "Downgrade of merge migration b3c4d5e6f7a8 is not supported. "
        "Restore from backup if needed."
    )
