import os

filepath = "tests/test_crypto_backtest_api.py"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

append_str = """

async def test_crypto_backtest_passes_lgbm_params_to_run_backtest(
    client, db_session, monkeypatch
):
    \"\"\"Проверяем что API передаёт lgbm_params из БД в run_backtest.\"\"\"
    from polyflip.db.models import RuntimeSettings
    # Кладём кастомный параметр в БД
    db_session.add(RuntimeSettings(key="CRYPTO_LGBM_N_ESTIMATORS", value="7", updated_by="test"))
    await db_session.commit()

    captured = {}
    def fake_run_backtest(df, symbol, min_edge, commission, **kwargs):
        captured.update(kwargs)
        class FakeResult:
            symbol = "BTCUSDT"
            n_candles_total = 1000
            n_candles_test = 500
            n_trades = 10
            win_rate = 0.5
            total_return_net = 0.05
            sharpe_ratio = 1.0
            max_drawdown = 0.01
            edge_rate = 0.1
            epsilon = 0.001
            train_auc = 0.55
            pnl_curve = []
            def is_profitable(self): return True
            def summary(self): return "ok"
        return FakeResult()

    monkeypatch.setattr("polyflip.api.crypto_dashboard.run_backtest", fake_run_backtest)
    
    # Мокаем get_recent_candles чтобы не требовать реальных данных
    async def fake_get_recent_candles(*args, **kwargs):
        class FakeCandle:
            def __init__(self):
                self.open_time = 0
                self.close = 1.0
                self.volume = 1.0
                self.is_closed = True
                self.symbol = "BTCUSDT"
                self.interval = "15m"
        return [FakeCandle() for _ in range(601)]
    monkeypatch.setattr("polyflip.api.crypto_dashboard.get_recent_candles", fake_get_recent_candles)

    resp = await client.get("/api/crypto/backtest?symbol=BTCUSDT&interval=15m")
    assert resp.status_code == 200
    assert captured.get("lgbm_params", {}).get("n_estimators") == 7
"""

with open(filepath, "a", encoding="utf-8") as f:
    f.write(append_str)
