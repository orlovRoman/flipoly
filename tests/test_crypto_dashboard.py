# tests/test_crypto_dashboard.py

def test_numpy_import_no_name_error():
    """Гарантирует, что np доступен на уровне модуля."""
    import polyflip.api.crypto_dashboard as m
    assert hasattr(m, 'np')


def test_backtest_endpoint_no_global_mutation():
    """После фикса — run_backtest принимает параметр напрямую, не мутируя глобальный constants."""
    from polyflip.crypto.backtester import run_backtest
    import inspect
    params = inspect.signature(run_backtest).parameters
    assert "min_edge" in params, "run_backtest должен принимать min_edge явно"
    assert "commission" in params, "run_backtest должен принимать commission явно"


def test_router_paths_no_duplicate_prefix():
    """Маршруты не дублируют /crypto в пути при prefix=/crypto."""
    from polyflip.api.crypto_dashboard import router
    paths = [r.path for r in router.routes]
    for p in paths:
        assert not p.startswith("/api/crypto"), (
            f"Путь {p!r} дублирует prefix. Используй /api/status и т.д."
        )

import pytest
from httpx import AsyncClient




@pytest.mark.asyncio
async def test_crypto_models_analytics_veto_logic(db_session):
    from polyflip.api.crypto_dashboard import crypto_models_analytics
    from polyflip.db.models import ModelRegistry, DecisionFunnelLog
    from fastapi import HTTPException
    from datetime import datetime, timezone
    
    # 1. HTTP 422 for invalid requested_mode
    with pytest.raises(HTTPException) as exc_info:
        await crypto_models_analytics(requested_mode="INVALID", db=db_session)
    assert exc_info.value.status_code == 422, "Ожидался 422 для невалидного режима"
    
    # Populate ModelRegistry
    now = datetime.now(timezone.utc)
    m = ModelRegistry(
        asset="BTCUSDT_low_vol",
        version=1,
        model_blob=b"fake",
        is_active=True,
        accuracy=0.55,
        baseline=0.5,
        trained_at=now
    )
    db_session.add(m)
    
    # Populate VETO logs (3 logs: before, inside, after)
    # Inside (matches date range 2026-08-02)
    l_inside = DecisionFunnelLog(
        market_id="m1", asset="BTCUSDT_low_vol",
        execution_mode="PAPER", confirm_passed=False,
        confirm_model_key="BTCUSDT_low_vol", confirm_model_version=1,
        final_action="SKIP",
        created_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    )
    # Before range (2026-08-01)
    l_before = DecisionFunnelLog(
        market_id="m2", asset="BTCUSDT_low_vol",
        execution_mode="PAPER", confirm_passed=False,
        confirm_model_key="BTCUSDT_low_vol", confirm_model_version=1,
        final_action="SKIP",
        created_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    # After range (2026-08-03)
    l_after = DecisionFunnelLog(
        market_id="m3", asset="BTCUSDT_low_vol",
        execution_mode="PAPER", confirm_passed=False,
        confirm_model_key="BTCUSDT_low_vol", confirm_model_version=1,
        final_action="SKIP",
        created_at=datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
    )
    db_session.add_all([l_inside, l_before, l_after])
    await db_session.commit()

    # 2. Range is exactly 2026-08-02 -> only 1 veto inside
    data = await crypto_models_analytics(requested_mode="PAPER", date_from="2026-08-02", date_to="2026-08-02", db=db_session)
    model_key = "BTCUSDT_low_vol_v1"
    assert model_key in data
    assert data[model_key]["veto_count"] == 1, "VETO должно фильтроваться выбранным диапазоном (1 запись внутри)"
    
    # 3. Requesting LIVE should still read PAPER VETO
    data_live = await crypto_models_analytics(requested_mode="LIVE", date_from="2026-08-02", date_to="2026-08-02", db=db_session)
    assert data_live[model_key]["veto_count"] == 1, "LIVE режим должен читать VETO из PAPER"

@pytest.mark.asyncio
async def test_lightgbm_direction_pnl_only(db_session):
    """LightGBM со сделками только в direction/confirm роли показывает PnL."""
    from polyflip.api.crypto_dashboard import crypto_models_analytics
    from polyflip.db.models import ModelRegistry, TradeHistory
    
    m = ModelRegistry(asset="LGBM_ONLY_VOL", version=1, model_blob=b"fake", is_active=True, accuracy=0.55, baseline=0.5)
    db_session.add(m)
    
    t = TradeHistory(
        mode="PAPER", position_status="CLOSED", pnl=10.0,
        direction_model_key="LGBM_ONLY_VOL", direction_model_version=1,
        confirm_model_key=None, confirm_model_version=None,
        model_key=None, model_version=None
    )
    db_session.add(t)
    await db_session.commit()
    
    data = await crypto_models_analytics(requested_mode="PAPER", db=db_session)
    assert "LGBM_ONLY_VOL_v1" in data
    assert data["LGBM_ONLY_VOL_v1"]["direction_pnl"] == 10.0
    assert data["LGBM_ONLY_VOL_v1"]["direction_trades"] == 1

@pytest.mark.asyncio
async def test_lightgbm_no_duplicate_legacy_and_direction(db_session):
    """Одна сделка не задваивается при legacy и direction атрибуции."""
    from polyflip.api.crypto_dashboard import crypto_models_analytics
    from polyflip.db.models import ModelRegistry, TradeHistory
    
    m = ModelRegistry(asset="LGBM_DEDUP_VOL", version=1, model_blob=b"fake", is_active=True, accuracy=0.55, baseline=0.5)
    db_session.add(m)
    
    # Сделка имеет и direction_model_key и confirm_model_key (legacy fallback)
    t = TradeHistory(
        mode="PAPER", position_status="CLOSED", pnl=-5.0,
        direction_model_key="LGBM_DEDUP_VOL", direction_model_version=1,
        confirm_model_key="LGBM_DEDUP_VOL", confirm_model_version=1
    )
    db_session.add(t)
    await db_session.commit()
    
    data = await crypto_models_analytics(requested_mode="PAPER", db=db_session)
    assert "LGBM_DEDUP_VOL_v1" in data
    # Сделка должна посчитаться один раз!
    assert data["LGBM_DEDUP_VOL_v1"]["direction_trades"] == 1
    assert data["LGBM_DEDUP_VOL_v1"]["direction_pnl"] == -5.0


# ────────────────────────────────────────────────────────────
# Quality Gate Activation Tests
# ────────────────────────────────────────────────────────────

@pytest.fixture
def _bad_model_factory():
    """Фабрика — создаёт ModelRegistry с quality_gate_passed=False."""
    from datetime import datetime, timezone
    from polyflip.db.models import ModelRegistry
    def _make(asset="BTCUSDT_high_vol", version=1):
        return ModelRegistry(
            asset=asset,
            version=version,
            model_blob=b"fake",
            is_active=False,
            accuracy=0.49,
            baseline=0.50,
            ece=0.21,
            trained_at=datetime.now(timezone.utc),
            quality_gate_passed=False,
            quality_gate_reasons={"reasons": ["Negative lift: -0.01"], "auc": 0.49, "ece": 0.21},
        )
    return _make


@pytest.mark.asyncio
async def test_bad_model_without_force_returns_409(db_session, _bad_model_factory):
    """Плохая модель без force=True возвращает HTTP 409."""
    from fastapi import HTTPException
    from polyflip.api.crypto_dashboard import activate_crypto_model, ActivateModelRequest

    m = _bad_model_factory()
    db_session.add(m)
    await db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        await activate_crypto_model(
            asset="BTCUSDT_high_vol",
            version=1,
            payload=ActivateModelRequest(force=False),
            db=db_session,
        )
    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["code"] == "QUALITY_GATE_OVERRIDE_REQUIRED"
    assert "metrics" in detail


@pytest.mark.asyncio
async def test_bad_model_with_force_becomes_active(db_session, _bad_model_factory):
    """Плохая модель с force=True становится активной с source=MANUAL."""
    from polyflip.api.crypto_dashboard import activate_crypto_model, ActivateModelRequest
    from polyflip.db.models import ModelRegistry
    from sqlalchemy import select

    m = _bad_model_factory()
    db_session.add(m)
    await db_session.commit()

    result = await activate_crypto_model(
        asset="BTCUSDT_high_vol",
        version=1,
        payload=ActivateModelRequest(force=True, reason="PAPER тест"),
        db=db_session,
    )

    assert result["status"] == "success"
    assert result["activation_source"] == "MANUAL"
    assert result["quality_gate_passed"] is False
    assert "warning" in result

    row = (await db_session.execute(
        select(ModelRegistry).where(
            ModelRegistry.asset == "BTCUSDT_high_vol",
            ModelRegistry.version == 1,
        )
    )).scalar_one()
    assert row.is_active is True
    assert row.activation_source == "MANUAL"
    assert row.activated_by == "dashboard"
    assert row.activation_reason == "PAPER тест"


@pytest.mark.asyncio
async def test_previous_version_deactivated(db_session):
    """При активации v2 предыдущая v1 деактивируется."""
    from polyflip.api.crypto_dashboard import activate_crypto_model, ActivateModelRequest
    from polyflip.db.models import ModelRegistry
    from sqlalchemy import select
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    v1 = ModelRegistry(
        asset="ETHUSDT_low_vol", version=1, model_blob=b"v1",
        is_active=True, accuracy=0.55, baseline=0.50, trained_at=now,
        quality_gate_passed=True, activation_source="AUTO",
    )
    v2 = ModelRegistry(
        asset="ETHUSDT_low_vol", version=2, model_blob=b"v2",
        is_active=False, accuracy=0.57, baseline=0.50, trained_at=now,
        quality_gate_passed=True,
    )
    db_session.add_all([v1, v2])
    await db_session.commit()

    result = await activate_crypto_model(
        asset="ETHUSDT_low_vol",
        version=2,
        payload=ActivateModelRequest(force=False),
        db=db_session,
    )
    assert result["active_version"] == 2
    assert result["previous_version"] == 1

    rows = (await db_session.execute(
        select(ModelRegistry).where(ModelRegistry.asset == "ETHUSDT_low_vol")
    )).scalars().all()
    active = [r for r in rows if r.is_active]
    assert len(active) == 1
    assert active[0].version == 2


@pytest.mark.asyncio
async def test_predictor_invalidated_after_activation(db_session, monkeypatch):
    """CryptoPredictor.invalidate_all вызывается после активации."""
    from polyflip.api.crypto_dashboard import activate_crypto_model, ActivateModelRequest
    from polyflip.db.models import ModelRegistry
    from datetime import datetime, timezone

    invalidated = []
    monkeypatch.setattr(
        "polyflip.api.crypto_dashboard.CryptoPredictor.invalidate_all",
        lambda symbol: invalidated.append(symbol),
    )

    now = datetime.now(timezone.utc)
    m = ModelRegistry(
        asset="SOLUSDT_mid_vol", version=5, model_blob=b"sol",
        is_active=False, accuracy=0.58, baseline=0.50, trained_at=now,
        quality_gate_passed=True,
    )
    db_session.add(m)
    await db_session.commit()

    await activate_crypto_model(
        asset="SOLUSDT_mid_vol",
        version=5,
        payload=ActivateModelRequest(force=False),
        db=db_session,
    )
    assert "SOLUSDT" in invalidated

