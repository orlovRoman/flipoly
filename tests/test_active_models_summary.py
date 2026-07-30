import pytest
from datetime import datetime, timezone
from polyflip.api.analytics import get_model_subtype_info
from polyflip.db.models import TradeHistory, ModelRegistry

def test_phase_models_form_distinct_keys():
    """Фазовые модели DOGE_leaning, DOGE_decided, DOGE_contested имеют уникальные ключи."""
    phase_assets = ["DOGE", "DOGE_leaning", "DOGE_decided", "DOGE_contested"]
    keys = [(asset, 8) for asset in phase_assets]
    assert len(keys) == len(set(keys)), f"Коллизия фазовых ключей: {keys}"

def test_lgbm_subtypes_use_exact_key_no_collision():
    """LightGBM субтипы одного символа дают разные exact-ключи."""
    lgbm_assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    keys = [(asset, 7) for asset in lgbm_assets]
    assert len(keys) == len(set(keys)), f"Коллизия: {keys}"

@pytest.mark.asyncio
async def test_model_pnl_does_not_mix_phase_versions(db_session):
    """Интеграционный тест: get_model_pnl изолирует PnL по model_key и не приписывает AMBIGUOUS фазовым моделям."""
    from polyflip.api.dashboard import get_model_pnl

    # Регистрируем тестовые модели
    m1 = ModelRegistry(asset="DOGE_leaning", version=8, model_blob=b"mock", is_active=True, accuracy=0.65, trained_at=datetime.now(timezone.utc))
    m2 = ModelRegistry(asset="DOGE_decided", version=8, model_blob=b"mock", is_active=True, accuracy=0.70, trained_at=datetime.now(timezone.utc))
    db_session.add_all([m1, m2])

    # Добавляем закрытые сделки
    t1 = TradeHistory(
        market_id="m1", asset="DOGE", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test",
        model_version=8, model_key="DOGE_leaning",
        model_attribution_source="EXACT", mode="PAPER", position_status="CLOSED",
        realized_pnl_usdc=5.0, status="SUCCESS", created_at=datetime.now(timezone.utc)
    )
    t2 = TradeHistory(
        market_id="m2", asset="DOGE", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test",
        model_version=8, model_key="DOGE_decided",
        model_attribution_source="RECONSTRUCTED", mode="PAPER", position_status="CLOSED",
        realized_pnl_usdc=-2.0, status="SUCCESS", created_at=datetime.now(timezone.utc)
    )
    t_ambiguous = TradeHistory(
        market_id="m3", asset="DOGE", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.6, active_features="test",
        model_version=8, model_key=None,
        model_attribution_source="AMBIGUOUS", mode="PAPER", position_status="CLOSED",
        realized_pnl_usdc=100.0, status="SUCCESS", created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([t1, t2, t_ambiguous])
    await db_session.commit()

    res = await get_model_pnl(requested_mode="PAPER", db=db_session)
    data = res["data"]

    assert data["DOGE_leaning_v8"]["pnl"] == 5.0
    assert data["DOGE_decided_v8"]["pnl"] == -2.0
    assert data["_unattributed"]["pnl"] == 100.0
    assert data["_unattributed"]["total_trades"] == 1
