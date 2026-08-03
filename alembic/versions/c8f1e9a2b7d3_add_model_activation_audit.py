"""add_model_activation_audit

Revision ID: a1b2c3d4e5f6
Revises: 783e6d92bc81
Create Date: 2026-08-03

Р”РѕР±Р°РІР»СЏРµС‚ РІ model_registry РїРѕР»СЏ Quality Gate Рё Activation Audit:
    quality_gate_passed, quality_gate_reasons,
    activation_source, activated_at, activated_by, activation_reason

Рђ С‚Р°РєР¶Рµ С‡Р°СЃС‚РёС‡РЅС‹Р№ СѓРЅРёРєР°Р»СЊРЅС‹Р№ РёРЅРґРµРєСЃ, РіР°СЂР°РЅС‚РёСЂСѓСЋС‰РёР№,
С‡С‚Рѕ Сѓ РєР°Р¶РґРѕРіРѕ asset РѕРґРЅРѕРІСЂРµРјРµРЅРЅРѕ Р°РєС‚РёРІРЅР° РЅРµ Р±РѕР»РµРµ РѕРґРЅРѕР№ РІРµСЂСЃРёРё.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'c8f1e9a2b7d3'
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


def downgrade() -> None:
    op.drop_column('model_registry', 'activation_reason')
    op.drop_column('model_registry', 'activated_by')
    op.drop_column('model_registry', 'activated_at')
    op.drop_column('model_registry', 'activation_source')
    op.drop_column('model_registry', 'quality_gate_reasons')
    op.drop_column('model_registry', 'quality_gate_passed')
