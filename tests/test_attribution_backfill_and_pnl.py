import pytest
import json
from datetime import datetime, timezone, timedelta
from polyflip.db.models import TradeHistory, ModelRegistry, DecisionFunnelLog
from polyflip.api.dashboard import get_model_pnl
from polyflip.api.crypto_dashboard import crypto_model_pnl
from polyflip.api.analytics import delete_model, get_active_models_summary
from scripts.backfill_model_keys import run_backfill

@pytest.mark.asyncio
async def test_backfill_dry_run_and_apply_keyset_pagination(db_session):
    """Тест честного backfill: dry-run не вносит изменений, apply реконструирует с валидацией по ModelRegistry."""
    m_reg = ModelRegistry(asset="BTCUSDT_low_vol", version=2, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    db_session.add(m_reg)

    t1 = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=2,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=10.0, status="SUCCESS",
        lgbm_metadata=json.dumps({"ml_phase_model": "BTCUSDT_low_vol", "lgbm_model_key": "BTCUSDT_low_vol", "lgbm_version": 2}),
        created_at=datetime.now(timezone.utc)
    )
    t_invalid = TradeHistory(
        market_id="m2", asset="BTC", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=99,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=5.0, status="SUCCESS",
        lgbm_metadata=json.dumps({"ml_phase_model": "UNKNOWN_MODEL_KEY"}),
        created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([t1, t_invalid])
    await db_session.commit()

    # 1. Проверяем dry-run (изменений не совершено)
    await run_backfill(apply_changes=False, mode="PAPER", session_override=db_session)
    await db_session.refresh(t1)
    assert t1.model_key is None

    # 2. Проверяем apply
    await run_backfill(apply_changes=True, mode="PAPER", session_override=db_session)
    await db_session.refresh(t1)
    await db_session.refresh(t_invalid)

    assert t1.model_key == "BTCUSDT_low_vol"
    assert t1.confirm_model_key == "BTCUSDT_low_vol"
    assert t1.confirm_model_version == 2
    assert t1.model_attribution_source == "RECONSTRUCTED"

    # Невалидный ключ отклоняется реестром моделей и становится AMBIGUOUS
    assert t_invalid.model_key is None
    assert t_invalid.model_attribution_source == "AMBIGUOUS"

@pytest.mark.asyncio
async def test_funnel_ambiguous_matching_not_assigned(db_session):
    """Сделки с несколькими кандидатными funnel-логами становятся AMBIGUOUS без выдуманного model_key."""
    m_reg = ModelRegistry(asset="DOGE_leaning", version=1, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    db_session.add(m_reg)

    now = datetime.now(timezone.utc)
    t = TradeHistory(
        market_id="m_ambig", asset="DOGE", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=1,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=2.5, status="SUCCESS",
        created_at=now
    )
    f1 = DecisionFunnelLog(market_id="m_ambig", asset="DOGE", trading_mode="ML", final_action="BUY_YES", used_model="DOGE_leaning", created_at=now)
    f2 = DecisionFunnelLog(market_id="m_ambig", asset="DOGE", trading_mode="ML", final_action="BUY_YES", used_model="DOGE_decided", created_at=now)
    db_session.add_all([t, f1, f2])
    await db_session.commit()

    await run_backfill(apply_changes=True, mode="PAPER", session_override=db_session)
    await db_session.refresh(t)

    assert t.model_key is None
    assert t.model_attribution_source == "AMBIGUOUS"

@pytest.mark.asyncio
async def test_crypto_model_pnl_isolation(db_session):
    """crypto_model_pnl эндпоинт группирует строго по model_key и разграничивает подтверждающие модели."""
    m1 = ModelRegistry(asset="BTCUSDT_low_vol", version=1, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    m2 = ModelRegistry(asset="BTCUSDT_high_vol", version=1, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    db_session.add_all([m1, m2])

    t1 = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=1, model_key="BTCUSDT_low_vol",
        confirm_model_key="BTCUSDT_high_vol", confirm_model_version=1,
        model_attribution_source="EXACT", mode="PAPER", position_status="CLOSED",
        realized_pnl_usdc=8.0, status="SUCCESS", created_at=datetime.now(timezone.utc)
    )
    db_session.add(t1)
    await db_session.commit()

    res = await crypto_model_pnl(requested_mode="PAPER", db=db_session)
    assert res["BTCUSDT_low_vol_v1"]["pnl"] == 8.0
    assert res["BTCUSDT_low_vol_v1"]["total_trades"] == 1
    assert res["BTCUSDT_high_vol_v1"]["pnl"] == 0.0
    assert res["BTCUSDT_high_vol_v1"]["confirmed_pnl"] == 8.0
    assert res["BTCUSDT_high_vol_v1"]["confirmed_trades"] == 1

@pytest.mark.asyncio
async def test_delete_archive_model(db_session):
    """API удаляет архивную модель из ModelRegistry."""
    m_active = ModelRegistry(asset="ETH", version=1, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    m_archive = ModelRegistry(asset="ETH", version=2, model_blob=b"mock", accuracy=0.65, is_active=False, trained_at=datetime.now(timezone.utc))
    db_session.add_all([m_active, m_archive])
    await db_session.commit()

    del_res = await delete_model(asset="ETH", version=2, db=db_session)
    assert del_res["status"] == "success"

    active_summary = await get_active_models_summary(requested_mode="PAPER", db=db_session)
    assets_versions = [(m["asset_full"], m["version"]) for m in active_summary["data"]]
    assert ("ETH", 1) in assets_versions
    assert ("ETH", 2) not in assets_versions
