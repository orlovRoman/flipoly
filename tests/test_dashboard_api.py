import pytest
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory
from polyflip.api.dashboard import get_trade_logs

@pytest.mark.asyncio
async def test_trade_logs_pagination(db_session):
    now = datetime.now(timezone.utc)
    for i in range(60):
        t = TradeHistory(market_id=f'm_{i}', asset='BTC', outcome_bought='YES', amount_usdc=10.0, remaining_shares=20.0, executed_price=0.5, predicted_flip_prob=0.8, active_features='test', status='SUCCESS', created_at=now)
        db_session.add(t)
    await db_session.commit()
    res1 = await get_trade_logs(db_session, page=1, page_size=25)
    assert len(res1['items']) == 25
    assert res1['total'] == 60
    assert res1['pages'] == 3
    assert res1['page'] == 1
    assert res1['page_size'] == 25
    assert 'edge' in res1['items'][0]
    res3 = await get_trade_logs(db_session, page=3, page_size=25)
    assert len(res3['items']) == 10
    assert res3['total'] == 60
    assert res3['pages'] == 3
    res99 = await get_trade_logs(db_session, page=99, page_size=25)
    assert len(res99['items']) == 0
    assert res99['total'] == 60

@pytest.mark.asyncio
async def test_get_daily_pnl(db_session):
    from polyflip.api.dashboard import get_daily_pnl
    now = datetime.now(timezone.utc)
    t = TradeHistory(market_id='m_pnl_1', asset='BTC', outcome_bought='YES', amount_usdc=10.0, remaining_shares=20.0, executed_price=0.6, predicted_flip_prob=0.8, active_features='other', status='SUCCESS', position_status='CLOSED', pnl=1.5, created_at=now)
    db_session.add(t)
    await db_session.commit()
    res = await get_daily_pnl(db=db_session)
    assert res['status'] == 'success'
    assert len(res['data']) == 1
    assert res['data'][0]['strategy'] == 'Фаворит'