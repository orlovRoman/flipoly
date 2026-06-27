import pytest
from unittest.mock import patch, PropertyMock
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.trading.utils import compute_kelly_multiplier

def test_compute_kelly_multiplier_strong_signal():
    f, mult = compute_kelly_multiplier(p_win=0.70, buy_price=0.30)
    assert f > 0.0
    assert 1.0 < mult <= 2.0

def test_compute_kelly_multiplier_weak_signal():
    f, mult = compute_kelly_multiplier(p_win=0.40, buy_price=0.50)
    assert f == 0.0
    assert mult == 0.5  # штраф за нулевой edge

def test_compute_kelly_multiplier_boundary_price():
    f, mult = compute_kelly_multiplier(p_win=0.80, buy_price=0.0)
    assert f == 0.0 and mult == 0.5  # деление на 0 → безопасный fallback

def test_compute_kelly_multiplier_max_fraction():
    f, mult = compute_kelly_multiplier(p_win=0.99, buy_price=0.01)
    assert f == 0.10  # зафиксировано max_fraction
    assert mult == 2.0

def test_capital_not_referenced():
    import inspect
    import polyflip.trading.engine as engine_module
    
    source = inspect.getsource(engine_module.trade_worker_cycle)
    lines = [l for l in source.split('\n') if 'capital' in l and '#' not in l]
    assert len(lines) == 0, f"Найдены строки с 'capital' в trade_worker_cycle: {lines}"

def test_kelly_multiplier_range():
    # kelly_f = 0.0  → multiplier = 0.5 (штраф за слабый сигнал)
    # kelly_f = 0.05 → multiplier = 1.25
    # kelly_f = 0.10 → multiplier = 2.0
    for kelly_f, expected in [(0.0, 0.5), (0.05, 1.25), (0.10, 2.0)]:
        mult = 0.5 + (kelly_f / 0.10) * 1.5
        assert abs(mult - expected) < 0.01

@pytest.mark.asyncio
async def test_kelly_stats_exclude_zero_fraction(db_session):
    from polyflip.api.trading_dashboard import get_trading_stats
    from polyflip.config import Settings

    # Настраиваем INITIAL_CAPITAL
    db_session.add(RuntimeSettings(key="INITIAL_CAPITAL", value="1000", updated_at=datetime.now(timezone.utc), updated_by="test"))
    
    now = datetime.now(timezone.utc)
    # Создаём 1 SUCCESS с kelly_fraction=0.08, kelly_multiplier=1.7
    t1 = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=17.0, executed_price=0.5,
        predicted_flip_prob=0.8, active_features="", status="SUCCESS", pnl=10.0,
        kelly_fraction=0.08, kelly_multiplier=1.7, created_at=now
    )
    # Создаём 1 SUCCESS с kelly_fraction=0.0, kelly_multiplier=0.5 (легитимная сделка с нулевым edge)
    t2 = TradeHistory(
        market_id="m2", asset="BTC", outcome_bought="YES", amount_usdc=5.0, executed_price=0.5,
        predicted_flip_prob=0.5, active_features="", status="SUCCESS", pnl=2.0,
        kelly_fraction=0.0, kelly_multiplier=0.5, created_at=now
    )
    # Создаём 5 SKIPPED с kelly_fraction=None, kelly_multiplier=None
    skipped_trades = [
        TradeHistory(
            market_id=f"m_skip_{i}", asset="BTC", outcome_bought="NONE", amount_usdc=0.0, executed_price=0.0,
            predicted_flip_prob=0.5, active_features="", status="SKIPPED", pnl=None,
            kelly_fraction=None, kelly_multiplier=None, created_at=now
        )
        for i in range(5)
    ]

    db_session.add_all([t1, t2])
    db_session.add_all(skipped_trades)
    await db_session.commit()

    with patch.object(Settings, "asset_list", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = ["BTC"]
        stats = await get_trading_stats(db_session)
        
    k = stats["kelly_stats"]
    assert abs(k["avg_f"] - 0.04) < 0.001, f"Ожидали avg_f=0.04, получили {k['avg_f']}"
    assert abs(k["avg_mult"] - 1.1) < 0.01, f"Ожидали avg_mult=1.1, получили {k['avg_mult']}"
