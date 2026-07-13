import asyncio
from sqlalchemy import select, desc, func
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, LiveMarket, RuntimeSettings
from datetime import datetime, timezone

async def check():
    print("Checking system activity...\n")
    async with async_session() as session:
        # 1. Settings
        stmt = select(RuntimeSettings)
        res = await session.execute(stmt)
        settings = {s.key: s.value for s in res.scalars().all()}
        print(f"TRADING_ENABLED: {settings.get('TRADING_ENABLED')}")
        print(f"TRADE_ON_FAVORITE: {settings.get('TRADE_ON_FAVORITE')}")
        print(f"TRADE_ON_FLIP: {settings.get('TRADE_ON_FLIP')}")
        print(f"TRADE_FLIP_THRESHOLD: {settings.get('TRADE_FLIP_THRESHOLD')}")
        print(f"FLIP_THRESHOLD: {settings.get('FLIP_THRESHOLD')}")
        
        # 2. Live Markets
        stmt = select(func.count(LiveMarket.market_id))
        res = await session.execute(stmt)
        live_count = res.scalar()
        print(f"\nCurrently tracked Live Markets: {live_count}")
        
        # 3. Latest trades
        stmt = select(TradeHistory).order_by(desc(TradeHistory.created_at)).limit(5)
        res = await session.execute(stmt)
        latest_trades = res.scalars().all()
        
        print("\nLast 5 TradeHistory entries (any status):")
        if not latest_trades:
            print("No trades found.")
        for t in latest_trades:
            p_flip_str = f"{t.predicted_flip_prob:.3f}" if t.predicted_flip_prob is not None else "N/A"
            print(f"[{t.created_at}] {t.asset} | {t.outcome_bought} | {t.status} | p_flip={p_flip_str} | {t.error_msg}")

if __name__ == "__main__":
    asyncio.run(check())
