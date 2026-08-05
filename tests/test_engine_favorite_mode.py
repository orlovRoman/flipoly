import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from polyflip.trading.engine import trade_worker_cycle
from polyflip.db.models import TradeHistory

def make_market(market_id='test-market-1', asset='BTC', yes_price=0.7, end_offset_sec=200):
    """Хелпер — создаёт mock LiveMarket."""
    m = MagicMock()
    m.market_id = market_id
    m.asset = asset
    m.current_yes_price = yes_price
    m.current_spread = 0.02
    m.price_velocity = 0.0
    m.volume_5min = 100.0
    m.yes_token_id = 'yes-token-123'
    m.no_token_id = 'no-token-456'
    m.end_time_est = datetime.now(timezone.utc) + timedelta(seconds=end_offset_sec)
    return m

def make_settings_db(trading_mode='favorite', entry_sec=180, bet_size=1.0, trade_assets='BTC,ETH'):
    """Хелпер — возвращает список RuntimeSettings моков."""
    pairs = {'TRADING_ENABLED': 'true', 'FAVOR_MIN_TIME_LEFT_SEC': '10', 'FAVOR_MAX_TIME_LEFT_SEC': '360', 'TRADE_BET_SIZE_USDC': str(bet_size), 'TRADE_NO_FLIP_THRESHOLD': '0.70', 'DEAD_ZONE_WIDTH': '0.15', 'DAILY_LOSS_LIMIT_USDC': '-100.0', 'ACTIVE_FEATURES': 'time_left_min,mid_price,spread', 'TRADE_MIN_PRICE': '0.05', 'TRADE_MAX_PRICE': '0.95', 'TRADE_ASSETS': trade_assets, 'TRADE_CAPITAL_USDC': '100', 'TRADING_MODE': trading_mode, 'FAVORITE_MODE_ENTRY_SEC': str(entry_sec), 'FAVORITE_THRESHOLD': '0.70', 'FAVORITE_MIN_EDGE': '-0.05', 'OUTS_MIN_EDGE': '-0.05'}
    result = []
    for k, v in pairs.items():
        s = MagicMock()
        s.key = k
        s.value = v
        result.append(s)
    return result

@pytest.mark.asyncio
async def test_pure_favorite_skips_when_outside_time_window():
    """Рынок закрывается через 500 сек — вне окна [180, 240] → пропускаем."""
    market = make_market(yes_price=0.7, end_offset_sec=500)
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
    db_session.execute.side_effect = [settings_scalars, empty_scalars, markets_scalars, daily_pnl_scalar]
    db_session.scalar = AsyncMock(return_value=0.0)
    trader = AsyncMock()
    api_client = AsyncMock()
    await trade_worker_cycle(db_session, api_client)
    trader.execute_trade.assert_not_called()

@pytest.mark.asyncio
async def test_pure_favorite_skips_duplicate_trade():
    """Уже есть SUCCESS-сделка на этом рынке → пропускаем."""
    market = make_market(yes_price=0.7, end_offset_sec=200)
    existing_trade = MagicMock()
    existing_trade.status = 'SUCCESS'
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
    db_session.execute.side_effect = [settings_scalars, empty_scalars, markets_scalars, daily_pnl_scalar, trade_check_scalars]
    db_session.scalar = AsyncMock(return_value=0.0)
    trader = AsyncMock()
    api_client = AsyncMock()
    await trade_worker_cycle(db_session, api_client)
    trader.execute_trade.assert_not_called()