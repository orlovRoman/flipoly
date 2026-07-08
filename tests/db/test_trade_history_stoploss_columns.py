def test_trade_history_has_stoploss_columns():
    from polyflip.db.models import TradeHistory
    cols = {c.key for c in TradeHistory.__table__.columns}
    assert "stop_loss_pct"        in cols
    assert "stop_loss_price"      in cols
    assert "stop_loss_status"     in cols
    assert "stop_loss_hit_at"     in cols
    assert "stop_loss_sell_price" in cols

def test_stop_loss_status_default():
    trade = TradeHistory()
    assert trade.stop_loss_status == "ACTIVE"
