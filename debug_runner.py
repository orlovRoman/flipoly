from polyflip.backtesting.runner import BacktestRunner
from polyflip.backtesting.market_replay import MarketReplay, SnapshotRow
from datetime import datetime, timezone

class FakeModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]] * len(X)

def test():
    config = {
        "MIN_TIME_LEFT_MIN": 1.0,
        "MAX_TIME_LEFT_MIN": 60.0,
        "STRATEGY_MODE": "ML",
        "NO_FLIP_THRESHOLD": 0.35,
        "FAVORITE_THRESHOLD": 0.65,
        "YES_MIN_PRICE": 0.55,
        "YES_MAX_PRICE": 0.95,
        "NO_MIN_PRICE": 0.55,
        "NO_MAX_PRICE": 0.95,
        "AUTO_DEAD_ZONE_WIDTH": 0.10,
        "MIN_EDGE": -0.05,
        "MAX_EDGE": 0.50,
        "TRADE_BET_SIZE_USDC": 5.0,
        "MAX_BET_SIZE_USDC": 50.0,
        "BET_SIZING_MODE": "scaled",
        "SLIPPAGE_PCT": 0.005,
        "ENTRY_STRATEGY": "first",
    }
    
    import pickle
    model_blob = pickle.dumps(FakeModel())
    features = "time_left_min,mid_price"
    
    runner = BacktestRunner(config, model_blob, features)
    
    # 1. Снимок, где mid_price = 0.8 (должен пройти FAVORITE_THRESHOLD = 0.65)
    snaps = [
        SnapshotRow(
            id=1, market_id="m1", asset="BTC", recorded_at=datetime.utcnow(),
            mid_price=0.8, price_velocity=0.01, time_left_min=10.0,
            final_outcome="YES", flip_vs_final=False, volume_5min=1000.0,
            spread=0.02, hour_of_day=12
        ),
        SnapshotRow(
            id=2, market_id="m1", asset="BTC", recorded_at=datetime.utcnow(),
            mid_price=0.8, price_velocity=0.01, time_left_min=9.0,
            final_outcome="YES", flip_vs_final=False, volume_5min=1000.0,
            spread=0.02, hour_of_day=12
        ),
        SnapshotRow(
            id=3, market_id="m1", asset="BTC", recorded_at=datetime.utcnow(),
            mid_price=0.8, price_velocity=0.01, time_left_min=8.0,
            final_outcome="YES", flip_vs_final=False, volume_5min=1000.0,
            spread=0.02, hour_of_day=12
        )
    ]
    
    replay = MarketReplay(snaps)
    trades = runner.run_all({"m1": replay})
    print(f"Trades count: {len(trades)}")
    for t in trades:
        print(t)

if __name__ == "__main__":
    test()
