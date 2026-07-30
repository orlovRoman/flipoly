"""add live_mirror_candidates table and source_paper linkage columns

Revision ID: e7f8a9b0c1d2
Revises: d5e6f7a8b9c0
Create Date: 2026-07-30 12:00:00.000000+00:00

Этап 4 плана введения LIVE-торговли.

Что делает:
1. Создаёт таблицу live_mirror_candidates — буфер между PAPER-сигналом
   и реальным LIVE-ордером. mirror-воркер пишет только сюда; release-gate
   потом атомарно превращает кандидата в LIVE TradeHistory + ExecutionRequest.

2. Добавляет nullable FK source_paper_trade_id в trade_history
   (NULL для PAPER-строк; заполняется только у LIVE-строк, зеркалированных из PAPER).

3. Добавляет nullable FK source_paper_request_id в execution_requests
   (та же логика).

4. Создаёт частичные уникальные индексы:
   - uq_live_trade_source_paper: одна LIVE-сделка на каждую PAPER-сделку
   - uq_live_request_source_paper: один LIVE-OPEN на каждую PAPER-заявку

Все поля nullable — существующие строки не изменяются.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Таблица live_mirror_candidates ────────────────────────────────────
    op.create_table(
        "live_mirror_candidates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "source_paper_request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("execution_requests.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_paper_trade_id",
            sa.Integer(),
            sa.ForeignKey("trade_history.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("target_mode", sa.String(16), nullable=False, server_default="SHADOW"),
        sa.Column("state", sa.String(32), nullable=False, server_default="NEW"),
        sa.Column("signal_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("signal_hash", sa.String(64), nullable=False),
        sa.Column(
            "released_trade_id",
            sa.Integer(),
            sa.ForeignKey("trade_history.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "released_request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("execution_requests.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        # Ограничения
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_paper_request_id",
            "target_mode",
            name="uq_live_mirror_source_mode",
        ),
        sa.CheckConstraint(
            "target_mode IN ('SHADOW', 'LIVE')",
            name="ck_live_mirror_target_mode",
        ),
        sa.CheckConstraint(
            "state IN ('NEW', 'ELIGIBLE', 'REJECTED', 'RELEASED')",
            name="ck_live_mirror_state",
        ),
    )
    op.create_index(
        "ix_live_mirror_candidates_state",
        "live_mirror_candidates",
        ["state"],
    )
    op.create_index(
        "ix_live_mirror_candidates_created_at",
        "live_mirror_candidates",
        ["created_at"],
    )
    op.create_index(
        "ix_live_mirror_candidates_source_paper_request_id",
        "live_mirror_candidates",
        ["source_paper_request_id"],
    )

    # ── 2. source_paper_trade_id → trade_history ─────────────────────────────
    op.add_column(
        "trade_history",
        sa.Column(
            "source_paper_trade_id",
            sa.Integer(),
            sa.ForeignKey("trade_history.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    # Частичный уникальный индекс: одна LIVE-сделка на каждую PAPER-сделку
    op.create_index(
        "uq_live_trade_source_paper",
        "trade_history",
        ["source_paper_trade_id"],
        unique=True,
        postgresql_where=sa.text(
            "mode = 'LIVE' AND source_paper_trade_id IS NOT NULL"
        ),
    )

    # ── 3. source_paper_request_id → execution_requests ──────────────────────
    op.add_column(
        "execution_requests",
        sa.Column(
            "source_paper_request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("execution_requests.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    # Частичный уникальный индекс: один LIVE-OPEN на каждую PAPER-заявку
    op.create_index(
        "uq_live_request_source_paper",
        "execution_requests",
        ["source_paper_request_id"],
        unique=True,
        postgresql_where=sa.text(
            "requested_mode = 'LIVE' "
            "AND intent = 'OPEN' "
            "AND source_paper_request_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    # Удаляем в обратном порядке

    # 3. execution_requests
    op.drop_index("uq_live_request_source_paper", table_name="execution_requests")
    op.drop_column("execution_requests", "source_paper_request_id")

    # 2. trade_history
    op.drop_index("uq_live_trade_source_paper", table_name="trade_history")
    op.drop_column("trade_history", "source_paper_trade_id")

    # 1. live_mirror_candidates (дочерние индексы удаляются каскадом)
    op.drop_index("ix_live_mirror_candidates_source_paper_request_id", table_name="live_mirror_candidates")
    op.drop_index("ix_live_mirror_candidates_created_at", table_name="live_mirror_candidates")
    op.drop_index("ix_live_mirror_candidates_state", table_name="live_mirror_candidates")
    op.drop_table("live_mirror_candidates")
