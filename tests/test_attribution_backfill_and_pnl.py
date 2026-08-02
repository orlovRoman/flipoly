import pytest
import json
from datetime import datetime, timezone, timedelta
from polyflip.db.models import TradeHistory, ModelRegistry, DecisionFunnelLog
from polyflip.api.dashboard import get_model_pnl
from polyflip.api.crypto_dashboard import crypto_models_analytics
from polyflip.api.analytics import delete_model, get_active_models_summary
import scripts.backfill_model_keys as backfill_module
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
    stats_dry = await run_backfill(apply_changes=False, mode="PAPER", session_override=db_session)
    await db_session.refresh(t1)
    assert t1.model_key is None
    assert stats_dry["PROCESSED"] == 2

    # 2. Проверяем apply
    stats_apply = await run_backfill(apply_changes=True, mode="PAPER", session_override=db_session)
    await db_session.refresh(t1)
    await db_session.refresh(t_invalid)

    assert stats_apply["PROCESSED"] == 2
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
async def test_backfill_multiple_batches_and_idempotency(db_session, monkeypatch):
    """Тест работы с несколькими чанками (BATCH_SIZE=2) и проверки повторной идемпотентности."""
    monkeypatch.setattr(backfill_module, "BATCH_SIZE", 2)

    m_reg = ModelRegistry(asset="ETHUSDT_low_vol", version=1, model_blob=b"mock", accuracy=0.65, is_active=True, trained_at=datetime.now(timezone.utc))
    db_session.add(m_reg)

    now = datetime.now(timezone.utc)
    t1 = TradeHistory(
        market_id="b1", asset="ETH", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=1,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=1.0, status="SUCCESS",
        lgbm_metadata=json.dumps({"ml_phase_model": "ETHUSDT_low_vol"}), created_at=now
    )
    t2 = TradeHistory(
        market_id="b2", asset="ETH", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=1,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=2.0, status="SUCCESS",
        lgbm_metadata=json.dumps({"ml_phase_model": "ETHUSDT_low_vol"}), created_at=now
    )
    t3 = TradeHistory(
        market_id="b3", asset="ETH", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test", model_version=1,
        mode="PAPER", position_status="CLOSED", realized_pnl_usdc=3.0, status="SUCCESS",
        lgbm_metadata=json.dumps({"ml_phase_model": "ETHUSDT_low_vol"}), created_at=now
    )
    db_session.add_all([t1, t2, t3])
    await db_session.commit()

    # Запуск 1: Сделки обрабатываются в 2 пакетах
    stats1 = await run_backfill(apply_changes=True, mode="PAPER", session_override=db_session)
    assert stats1["PROCESSED"] == 3
    assert stats1["RECONSTRUCTED"] == 3

    await db_session.refresh(t1)
    await db_session.refresh(t2)
    await db_session.refresh(t3)
    assert t1.model_key == "ETHUSDT_low_vol"
    assert t2.model_key == "ETHUSDT_low_vol"
    assert t3.model_key == "ETHUSDT_low_vol"

    # Запуск 2: Идемпотентность — сделки уже получили model_attribution_source и не выбираются повторно
    stats2 = await run_backfill(apply_changes=True, mode="PAPER", session_override=db_session)
    assert stats2["PROCESSED"] == 0

@pytest.mark.asyncio
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
