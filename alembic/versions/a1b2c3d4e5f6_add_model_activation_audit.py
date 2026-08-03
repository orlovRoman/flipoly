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

    # Р§Р°СЃС‚РёС‡РЅС‹Р№ СѓРЅРёРєР°Р»СЊРЅС‹Р№ РёРЅРґРµРєСЃ: РѕРґРЅР° Р°РєС‚РёРІРЅР°СЏ РІРµСЂСЃРёСЏ РЅР° asset
    # РСЃРїРѕР»СЊР·СѓРµРј raw SQL С‚.Рє. SQLAlchemy РЅРµ РїРѕРґРґРµСЂР¶РёРІР°РµС‚ partial index РІ op.create_index
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_one_active_version
        ON model_registry (asset)
        WHERE is_active IS TRUE
    """)

    # Backfill: РІСЃРµ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РёРµ Р°РєС‚РёРІРЅС‹Рµ РјРѕРґРµР»Рё РїРѕРјРµС‡Р°РµРј РєР°Рє AUTO
    op.execute("""
        UPDATE model_registry
        SET activation_source = 'AUTO',
            quality_gate_passed = TRUE
        WHERE is_active = TRUE AND activation_source IS NULL
    """)

    # РќРµР°РєС‚РёРІРЅС‹Рµ вЂ” СЃС‚Р°РІРёРј quality_gate_passed = FALSE (РєРѕРЅСЃРµСЂРІР°С‚РёРІРЅРѕ, РјС‹ РЅРµ Р·РЅР°РµРј С‚РѕС‡РЅРѕ)
    # РћСЃС‚Р°РІР»СЏРµРј NULL, С‚.Рє. СЂРµР°Р»СЊРЅРѕ РЅРµ Р·РЅР°РµРј РїСЂРёС‡РёРЅСѓ РґРµР°РєС‚РёРІР°С†РёРё
    # NULL = "РґР°РЅРЅС‹Рµ РґРѕ РІРЅРµРґСЂРµРЅРёСЏ Р°СѓРґРёС‚Р°"


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_model_registry_one_active_version")
    op.drop_column('model_registry', 'activation_reason')
    op.drop_column('model_registry', 'activated_by')
    op.drop_column('model_registry', 'activated_at')
    op.drop_column('model_registry', 'activation_source')
    op.drop_column('model_registry', 'quality_gate_reasons')
    op.drop_column('model_registry', 'quality_gate_passed')
