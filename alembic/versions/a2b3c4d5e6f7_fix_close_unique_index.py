"""fix_close_unique_index_and_position_version

Исправляет уникальный индекс для CLOSE-заявок:
- Старый индекс: (requested_mode, trade_history_id) — допускал два CLOSE
  для одной сделки в разных режимах (PAPER vs LIVE).
- Новый индекс: (trade_history_id) — одна сделка = один активный CLOSE.

Также добавляет колонку position_version в trade_history,
если она ещё не существует (без неё outbox не может атомарно инкрементировать
версию позиции при создании CLOSE-запроса).

Revision ID: a2b3c4d5e6f7
Revises: 32d70d1e31e4
Create Date: 2026-07-27 05:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "a2b3c4d5e6f7"
down_revision: Union[str, None] = "41c8d1f7c5e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Предикат для активных CLOSE-заявок
_ACTIVE_CLOSE_SQL = """
    intent = 'CLOSE' AND state IN (
        'AWAITING_APPROVAL', 'READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED',
        'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
    )
"""


def upgrade() -> None:
    conn = op.get_bind()

    # --- Защита: проверяем дублирующиеся активные CLOSE-заявки ---
    duplicate_check = conn.execute(
        text(
            """
            SELECT trade_history_id, count(*) AS cnt
            FROM execution_requests
            WHERE intent = 'CLOSE'
              AND state IN (
                'AWAITING_APPROVAL', 'READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED',
                'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
              )
            GROUP BY trade_history_id
            HAVING count(*) > 1
            """
        )
    ).fetchall()

    if duplicate_check:
        rows = [(r[0], r[1]) for r in duplicate_check]
        raise RuntimeError(
            f"Cannot apply migration: found {len(rows)} trade(s) with multiple active "
            f"CLOSE requests. Resolve manually before migrating. "
            f"Duplicates: {rows}"
        )

    # --- Удаляем старый CLOSE-индекс (если существует) ---
    # Попытка безопасного drop: ошибку игнорируем, если индекс не найден
    try:
        op.drop_index(
            "uq_active_close_request", table_name="execution_requests"
        )
    except Exception:
        pass  # индекс мог не существовать

    # --- Создаём новый CLOSE-индекс по (trade_history_id) ---
    op.create_index(
        "uq_active_close_request",
        "execution_requests",
        ["trade_history_id"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_CLOSE_SQL),
        sqlite_where=sa.text(_ACTIVE_CLOSE_SQL),
    )

    # --- Добавляем position_version в trade_history, если нет ---
    inspector = sa.inspect(conn)
    existing_cols = {c["name"] for c in inspector.get_columns("trade_history")}
    if "position_version" not in existing_cols:
        op.add_column(
            "trade_history",
            sa.Column(
                "position_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    # Удаляем новый индекс
    op.drop_index("uq_active_close_request", table_name="execution_requests")

    # Восстанавливаем старый CLOSE-индекс по (requested_mode, trade_history_id)
    op.create_index(
        "uq_active_close_request",
        "execution_requests",
        ["requested_mode", "trade_history_id"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_CLOSE_SQL),
        sqlite_where=sa.text(_ACTIVE_CLOSE_SQL),
    )
    # position_version не удаляем — downgrade схемы опасен при наличии данных
