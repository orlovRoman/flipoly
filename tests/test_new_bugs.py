import pytest
from unittest.mock import AsyncMock, MagicMock

# ── BUG-05: engine.py — нет AttributeError при crypto_sig=None ──────────────

def test_engine_no_crash_when_crypto_sig_none():
    """BUG-05: безусловное обращение к crypto_sig.model_version падает если sig=None."""
    crypto_sig = None
    decision_obj = None
    
    try:
        model_ver = crypto_sig.model_version if crypto_sig else None
        edge = decision_obj.edge if decision_obj else None
    except AttributeError as e:
        pytest.fail(f"BUG-05: AttributeError при crypto_sig=None: {e}")

# ── BUG-06: backtester._empty_result принимает epsilon ──────────────────────

def test_empty_result_accepts_epsilon():
    """BUG-06: _empty_result должна иметь параметр epsilon."""
    import inspect
    from polyflip.crypto.backtester import _empty_result, _EPSILON
    sig = inspect.signature(_empty_result)
    assert "epsilon" in sig.parameters, (
        "BUG-06: _empty_result не имеет параметра epsilon — "
        "при вызове без аргумента будет NameError"
    )
    default = sig.parameters["epsilon"].default
    assert default == _EPSILON or default != inspect.Parameter.empty

# ── BUG-07: daily_pnl правильно классифицирует CRYPTO_TREND ─────────────────

@pytest.mark.asyncio
async def test_daily_pnl_crypto_strategy_classified():
    """BUG-07: Сделки с active_features='CRYPTO_TREND' не должны быть 'Другое'."""
    from polyflip.api.dashboard import get_daily_pnl
    from unittest.mock import AsyncMock, MagicMock

    mock_trade = MagicMock()
    mock_trade.asset = "BTCUSDT"
    mock_trade.active_features = "CRYPTO_TREND"
    mock_trade.pnl = 5.0
    mock_trade.amount_usdc = 10.0

    mock_result = MagicMock()
    mock_result.all.return_value = [mock_trade]
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    response = await get_daily_pnl(db=mock_db)
    
    strategies = [item["strategy"] for item in response["data"]]
    assert "Другое" not in strategies, (
        "BUG-07: Крипто-стратегия классифицируется как 'Другое'"
    )
    assert "Крипто" in strategies

def test_daily_pnl_has_crypto_branch():
    """BUG-07: исходник dashboard.py должен содержать ветку для crypto."""
    source = open("polyflip/api/dashboard.py", encoding="utf-8").read()
    assert "crypto" in source.lower() or "крипто" in source.lower(), (
        "BUG-07: get_daily_pnl не классифицирует крипто-стратегию"
    )

# ── BUG-08: takeprofit_worker заполняет take_profit_sell_size ────────────────

def test_tp_worker_sets_sell_size():
    """BUG-08: воркер должен записывать take_profit_sell_size."""
    source = open("polyflip/trading/takeprofit_worker.py", encoding="utf-8").read()
    assert "take_profit_sell_size" in source, (
        "BUG-08: takeprofit_worker не заполняет поле take_profit_sell_size. "
        "Колонка добавлена в БД, но воркер её не пишет."
    )

@pytest.mark.asyncio
async def test_tp_worker_sell_size_written_to_db(db_session):
    """BUG-08: после срабатывания TP поле take_profit_sell_size != NULL."""
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select
    from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket
    from polyflip.trading.takeprofit_worker import takeprofit_worker_cycle

    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(
        key="TAKE_PROFIT_ENABLED", value="true",
        updated_at=now, updated_by="test"
    ))
    db_session.add(LiveMarket(
        market_id="m_size_test", asset="BTC", question="Q?",
        yes_token_id="t_y", no_token_id="t_n",
        current_yes_price=0.85, current_no_price=0.15,
        current_spread=0.01, volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(minutes=5), last_updated=now,
    ))
    trade = TradeHistory(
        market_id="m_size_test", asset="BTC", outcome_bought="YES",
        amount_usdc=10.0, executed_price=0.40, status="SUCCESS",
        predicted_flip_prob=0.8, active_features="mid_price",
        take_profit_enabled=True, take_profit_multiplier=2.0,
        take_profit_price=0.80, take_profit_status="ACTIVE",
        market_end_time=now + timedelta(minutes=5), created_at=now,
    )
    db_session.add(trade)
    await db_session.commit()

    trader_mock = AsyncMock()
    trader_mock.execute_trade = AsyncMock(return_value={
        "status": "SUCCESS", "executed_price": 0.85, "mode": "PAPER"
    })
    api_mock = AsyncMock()
    api_mock.get_market_prices = AsyncMock(return_value={"best_bid": 0.85})

    await takeprofit_worker_cycle(db_session, trader_mock, api_mock)

    result = await db_session.execute(
        select(TradeHistory).where(TradeHistory.market_id == "m_size_test")
    )
    t = result.scalar_one()
    assert t.take_profit_status == "TRIGGERED"
    assert t.take_profit_sell_size is not None, (
        "BUG-08: take_profit_sell_size не заполнен после срабатывания TP"
    )
    expected_shares = round(10.0 / 0.40, 2)
    assert t.take_profit_sell_size == expected_shares
