"""add_index_trade_history_model_version

Revision ID: b258f4d83d08
Revises: 42a7b3c4d5e9
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b258f4d83d08"
down_revision: Union[str, None] = "42a7b3c4d5e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    columns = {column["name"] for column in inspector.get_columns("trade_history")}

    # В некоторых старых production-схемах колонки уже существовали,
    # но в чистой цепочке Alembic они отсутствовали.
    if "lgbm_metadata" not in columns:
        op.add_column(
            "trade_history",
            sa.Column(
                "lgbm_metadata",
                sa.String(),
                nullable=True,
            ),
        )

    if "edge" not in columns:
        op.add_column(
            "trade_history",
            sa.Column(
                "edge",
                sa.Float(),
                nullable=True,
            ),
        )

    indexes = {index["name"] for index in inspector.get_indexes("trade_history")}

    if "idx_trade_history_model_version" not in indexes:
        op.create_index(
            "idx_trade_history_model_version",
            "trade_history",
            [
                "asset",
                "model_version",
                "status",
                "created_at",
            ],
            unique=False,
        )

    # take_profit_sell_size удалять нельзя:
    # колонка присутствует в ORM и используется последующими миграциями.


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    indexes = {index["name"] for index in inspector.get_indexes("trade_history")}

    if "idx_trade_history_model_version" in indexes:
        op.drop_index(
            "idx_trade_history_model_version",
            table_name="trade_history",
        )

    # lgbm_metadata не удаляем: в части исторических production-баз
    # она существовала до появления этой ревизии.
