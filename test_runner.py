import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///Z:/polymarket-bot/vault/database.sqlite"
import asyncio
from polyflip.db.connection import async_session
from polyflip.api.backtest_api import _execute_backtest_logic
from polyflip.api.backtest_schemas import BacktestConfig
from polyflip.api.backtest_api import Job

async def run():
    config = BacktestConfig(
        assets=["BTC", "ETH"],
        min_snapshots_per_market=3,
        strategy_mode="ML"
    )
    job = Job("test_123")
    async with async_session() as db:
        res = await _execute_backtest_logic(db, config, "test_123", job)
        print(f"Loaded: {res['total_loaded']}")
        print(f"Tradeable: {res['tradeable']}")
        print(f"Skipped: {res['skipped']}")
        print(f"Trades: {res['total_trades']}")

asyncio.run(run())
