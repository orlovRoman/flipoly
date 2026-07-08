import pytest
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory

def test_trade_history_has_stoploss_columns():
    cols = {c.key for c in TradeHistory.__table__.columns}
    assert "stop_loss_pct"        in cols
    assert "stop_loss_price"      in cols
    assert "stop_loss_status"     in cols
    assert "stop_loss_hit_at"     in cols
    assert "stop_loss_sell_price" in cols

@pytest.mark.asyncio
async def test_stop_loss_status_default(db_session):
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.3,
        active_features="test",
        created_at=datetime.now(timezone.utc),
        status="SUCCESS"
    )
    db_session.add(trade)
    await db_session.commit()
    
    assert trade.stop_loss_status == "ACTIVE"
