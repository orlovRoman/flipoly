import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from polyflip.trading.engine import trade_worker_cycle
from polyflip.db.models import TradeHistory

def make_market(
    market_id="test-market-1",
    asset="BTC",
    yes_price=0.70,
    end_offset_sec=200,  # закрывается через 200 сек
):
    """Хелпер — создаёт mock LiveMarket."""
    m = MagicMock()
    m.market_id = market_id
    m.asset = asset
    m.current_yes_price = yes_price
    m.current_spread = 0.02
    m.price_velocity = 0.0
    m.volume_5min = 100.0
    m.yes_token_id = "yes-token-123"
    m.no_token_id = "no-token-456"
    m.end_time_est = datetime.now(timezone.utc) + timedelta(seconds=end_offset_sec)
    return m


def make_settings_db(
    trading_mode="favorite",
    entry_sec=180,
    bet_size=1.0,
    trade_assets="BTC,ETH",
):
    """Хелпер — возвращает список RuntimeSettings моков."""
    pairs = {
        "TRADING_ENABLED": "true",
        "TRADE_MIN_TIME_LEFT_SEC": "10",
        "TRADE_MAX_TIME_LEFT_SEC": "360",
        "TRADE_BET_SIZE_USDC": str(bet_size),
        "TRADE_NO_FLIP_THRESHOLD": "0.70",
        "DEAD_ZONE_WIDTH": "0.15",
        "DAILY_LOSS_LIMIT_USDC": "-100.0",
        "ACTIVE_FEATURES": "time_left_min,mid_price,spread",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "TRADE_ASSETS": trade_assets,
        "TRADE_CAPITAL_USDC": "100",
        "TRADING_MODE": trading_mode,
        "FAVORITE_MODE_ENTRY_SEC": str(entry_sec),
        "FAVORITE_MIN_EDGE": "-0.05",
    }
    result = []
    for k, v in pairs.items():
        s = MagicMock()
        s.key = k
        s.value = v
        result.append(s)
    return result


@pytest.mark.asyncio
async def test_pure_favorite_buys_yes_when_yes_is_favorite():
    """YES фаворит (цена 0.70) → бот покупает YES, predicted_flip_prob = None."""
    market = make_market(yes_price=0.70, end_offset_sec=200)  # 200 сек = в окне [180, 240]
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0  # нет потерь сегодня
    
    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = []  # нет дублей
    
    db_session.execute.side_effect = [
        settings_scalars,    # 1. settings_keys
        empty_scalars,       # 2. per-asset thresholds
        markets_scalars,     # 3. live markets
        daily_pnl_scalar,    # 4. daily PnL (т.к. есть рынки)
        trade_check_scalars, # 5. trade history check (внутри цикла)
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "mode": "PAPER"})
    
    api_client = AsyncMock()
    api_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.70, "best_bid": 0.68})
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_called_once()
    call_kwargs = trader.execute_trade.call_args.kwargs
    assert call_kwargs["token_id"] == "yes-token-123"
    assert call_kwargs["side"] == "BUY"
    
    # Проверяем запись в историю
    added = db_session.add.call_args_list
    trade_record = next(
        (a.args[0] for a in added if isinstance(a.args[0], TradeHistory)), None
    )
    assert trade_record is not None
    assert trade_record.outcome_bought == "YES"
    assert trade_record.predicted_flip_prob == 0.0  # ML не использовался
    assert trade_record.active_features == "PURE_FAVORITE"
    assert trade_record.model_version is None


@pytest.mark.asyncio
async def test_pure_favorite_buys_no_when_no_is_favorite():
    """NO фаворит (YES цена 0.30) → покупаем NO токен."""
    market = make_market(yes_price=0.30, end_offset_sec=200)
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = []
    
    db_session.execute.side_effect = [
        settings_scalars,    # 1. settings_keys
        empty_scalars,       # 2. per-asset thresholds
        markets_scalars,     # 3. live markets
        daily_pnl_scalar,    # 4. daily PnL
        trade_check_scalars, # 5. trade history check
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "mode": "PAPER"})
    
    api_client = AsyncMock()
    api_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.70, "best_bid": 0.68})
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_called_once()
    call_kwargs = trader.execute_trade.call_args.kwargs
    assert call_kwargs["token_id"] == "no-token-456"
    assert call_kwargs["side"] == "BUY"
    
    added = db_session.add.call_args_list
    trade_record = next(
        (a.args[0] for a in added if isinstance(a.args[0], TradeHistory)), None
    )
    assert trade_record is not None
    assert trade_record.outcome_bought == "NO"
    assert trade_record.predicted_flip_prob == 0.0


@pytest.mark.asyncio
async def test_pure_favorite_skips_when_outside_time_window():
    """Рынок закрывается через 500 сек — вне окна [180, 240] → пропускаем."""
    market = make_market(yes_price=0.70, end_offset_sec=500)
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    db_session.execute.side_effect = [
        settings_scalars,   # 1. settings_keys
        empty_scalars,      # 2. per-asset thresholds
        markets_scalars,    # 3. live markets
        daily_pnl_scalar,   # 4. daily PnL (поскольку рынок вернулся в моке, но время не пройдет)
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    api_client = AsyncMock()
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_pure_favorite_skips_duplicate_trade():
    """Уже есть SUCCESS-сделка на этом рынке → пропускаем."""
    market = make_market(yes_price=0.70, end_offset_sec=200)
    
    existing_trade = MagicMock()
    existing_trade.status = "SUCCESS"
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = [existing_trade]
    
    db_session.execute.side_effect = [
        settings_scalars,
        empty_scalars,
        markets_scalars,
        daily_pnl_scalar,
        trade_check_scalars
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    api_client = AsyncMock()
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_pure_favorite_skips_when_price_exactly_05():
    """Цена YES == 0.5 → нет явного фаворита → SKIPPED."""
    market = make_market(yes_price=0.5, end_offset_sec=200)
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = []
    
    db_session.execute.side_effect = [
        settings_scalars,
        empty_scalars,
        markets_scalars,
        daily_pnl_scalar,
        trade_check_scalars
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    api_client = AsyncMock()
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_not_called()
    
    added = db_session.add.call_args_list
    trade_record = next(
        (a.args[0] for a in added if isinstance(a.args[0], TradeHistory)), None
    )
    assert trade_record is not None
    assert trade_record.status == "SKIPPED"
    assert "no clear favorite" in trade_record.error_msg


@pytest.mark.asyncio
async def test_ml_mode_unchanged_when_trading_mode_is_ml():
    """При TRADING_MODE=ml ветка Pure Favorite не выполняется, и код идет по ML пути (где пытается загрузить модели)."""
    market = make_market(yes_price=0.70, end_offset_sec=200)
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    # Устанавливаем режим 'ml'
    settings_scalars.scalars.return_value.all.return_value = make_settings_db(trading_mode="ml")
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    # Поскольку мы в ML режиме, код будет грузить активные модели
    # Нам нужно, чтобы execute вернул пустой список моделей, чтобы код пропустил сделку с "No active model"
    active_models_scalars = MagicMock()
    active_models_scalars.scalars.return_value.all.return_value = []

    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = []
    
    db_session.execute.side_effect = [
        settings_scalars,      # 1. settings_keys
        empty_scalars,         # 2. per-asset thresholds
        markets_scalars,       # 3. live markets
        daily_pnl_scalar,      # 4. daily PnL
        active_models_scalars, # 5. active models
        trade_check_scalars,   # 6. trade history check (внутри цикла)
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    api_client = AsyncMock()
    api_client.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.70, "current_spread": 0.02})
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    # execute_trade не должен быть вызван, т.к. нет моделей
    trader.execute_trade.assert_not_called()
    
    # Проверим, что была добавлена запись о пропуске из-за отсутствия моделей
    added = db_session.add.call_args_list
    trade_record = next(
        (a.args[0] for a in added if isinstance(a.args[0], TradeHistory)), None
    )
    assert trade_record is not None
    assert trade_record.status == "SKIPPED"
    assert "No active model" in trade_record.error_msg


@pytest.mark.asyncio
async def test_pure_favorite_skips_no_when_yes_becomes_favorite():
    """NO фаворит в БД (YES цена 0.30), но по API NO подешевел до 0.30 (YES подорожал до 0.70) -> пропускаем."""
    market = make_market(yes_price=0.30, end_offset_sec=200)
    
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    
    settings_scalars = MagicMock()
    settings_scalars.scalars.return_value.all.return_value = make_settings_db()
    
    empty_scalars = MagicMock()
    empty_scalars.scalars.return_value.all.return_value = []
    
    markets_scalars = MagicMock()
    markets_scalars.scalars.return_value.all.return_value = [market]

    daily_pnl_scalar = MagicMock()
    daily_pnl_scalar.scalar.return_value = 0.0
    
    trade_check_scalars = MagicMock()
    trade_check_scalars.scalars.return_value.all.return_value = []
    
    db_session.execute.side_effect = [
        settings_scalars,
        empty_scalars,
        markets_scalars,
        daily_pnl_scalar,
        trade_check_scalars
    ]
    db_session.scalar = AsyncMock(return_value=0.0)
    
    trader = AsyncMock()
    api_client = AsyncMock()
    # Мокаем цену NO-токена равной 0.30 (т.е. NO теперь аутсайдер, YES = 0.70)
    api_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.30, "best_bid": 0.28})
    
    await trade_worker_cycle(db_session, trader, api_client)
    
    trader.execute_trade.assert_not_called()
    
    added = db_session.add.call_args_list
    trade_record = next(
        (a.args[0] for a in added if isinstance(a.args[0], TradeHistory)), None
    )
    assert trade_record is not None
    assert trade_record.status == "SKIPPED"
    assert "Price drift" in trade_record.error_msg or "out of bounds" in trade_record.error_msg
