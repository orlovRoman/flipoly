import asyncio
from datetime import datetime, timezone, timedelta
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select, func, cast, Date

async def test():
    async with async_session() as s:
        # PostgreSQL timezone ('Asia/Ho_Chi_Minh') -> UTC+7 (Нячанг)
        local_created_at = func.timezone('Asia/Ho_Chi_Minh', TradeHistory.created_at)
        local_date = cast(local_created_at, Date)
        
        q = select(
            TradeHistory.created_at,
            local_created_at.label("nha_trang_dt"),
            local_date.label("nha_trang_date")
        ).order_by(TradeHistory.created_at.desc()).limit(3)
        
        rows = (await s.execute(q)).all()
        for r in rows:
            print(f"UTC: {r.created_at} | Nha Trang (UTC+7): {r.nha_trang_dt} | Date: {r.nha_trang_date}")

if __name__ == "__main__":
    asyncio.run(test())
