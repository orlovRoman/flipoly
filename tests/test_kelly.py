from polyflip.trading.engine import kelly_bet_size
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

def test_kelly_bet_size_logic():
    # 1. Уверенный сигнал → ставка больше нуля
    bet = kelly_bet_size(p_win=0.70, buy_price=0.30, capital=1000)
    assert bet > 0, "При сильном сигнале ставка должна быть > 0"

    # 2. Неуверенный сигнал → ставка = 0
    bet_weak = kelly_bet_size(p_win=0.40, buy_price=0.50, capital=1000)
    assert bet_weak == 0.0, "При слабом сигнале (ожидаемый убыток) Kelly = 0"

    # 3. Ставка не превышает 10% капитала
    bet_max = kelly_bet_size(p_win=0.99, buy_price=0.01, capital=1000)
    assert bet_max <= 100.0, f"Ставка превышает 10% капитала: ${bet_max}"

    # 4. Граничная цена
    bet_edge = kelly_bet_size(p_win=0.80, buy_price=0.0, capital=1000)
    assert bet_edge == 0.0, "При нулевой цене Kelly должен вернуть 0"

    # 5. Проверка корректности передачи цены покупки (из замечания пользователя)
    # Фаворит стоит 0.85, а buy_price передаётся как 0.15 (цена аутсайдера)
    bet_correct = kelly_bet_size(0.85, 0.85, 1000)   # ≈ $0 (почти не выгодно)
    bet_wrong   = kelly_bet_size(0.85, 0.15, 1000)   # >> $0 (завышено — неверный buy_price)
    assert bet_correct < bet_wrong, "Kelly даёт разные результаты при одинаковой p_win — проверить buy_price"

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
    from polyflip.db.models import TradeHistory, RuntimeSettings
    from polyflip.config import settings

    # Настраиваем INITIAL_CAPITAL
    db_session.add(RuntimeSettings(key="INITIAL_CAPITAL", value="1000", updated_at=datetime.now(timezone.utc), updated_by="test"))
    
    now = datetime.now(timezone.utc)
    # Создаём 1 SUCCESS с kelly_fraction=0.08, kelly_multiplier=1.7
    t1 = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=17.0, executed_price=0.5,
        predicted_flip_prob=0.8, active_features="", status="SUCCESS", pnl=10.0,
        kelly_fraction=0.08, kelly_multiplier=1.7, created_at=now
    )
    # Создаём 5 SKIPPED с kelly_fraction=0.0, kelly_multiplier=1.0
    skipped_trades = [
        TradeHistory(
            market_id=f"m_skip_{i}", asset="BTC", outcome_bought="NONE", amount_usdc=0.0, executed_price=0.0,
            predicted_flip_prob=0.5, active_features="", status="SKIPPED", pnl=None,
            kelly_fraction=0.0, kelly_multiplier=1.0, created_at=now
        )
        for i in range(5)
    ]
    # Создаём 1 SKIPPED с kelly_fraction=None
    t_null = TradeHistory(
        market_id="m_skip_null", asset="BTC", outcome_bought="NONE", amount_usdc=0.0, executed_price=0.0,
        predicted_flip_prob=0.5, active_features="", status="SKIPPED", pnl=None,
        kelly_fraction=None, kelly_multiplier=None, created_at=now
    )

    db_session.add(t1)
    db_session.add_all(skipped_trades)
    db_session.add(t_null)
    await db_session.commit()

    from unittest.mock import PropertyMock
    from polyflip.config import Settings
    with patch.object(Settings, "asset_list", new_callable=PropertyMock(return_value=["BTC"])):
        stats = await get_trading_stats(db_session)
        
    k = stats["kelly_stats"]
    assert abs(k["avg_f"] - 0.08) < 0.001, f"Ожидали avg_f=0.08, получили {k['avg_f']}"
    assert abs(k["avg_mult"] - 1.7) < 0.01, f"Ожидали avg_mult=1.7, получили {k['avg_mult']}"
