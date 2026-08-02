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
